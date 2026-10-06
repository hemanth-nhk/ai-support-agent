"""
Safe Execution Engine for Compiled DuckDB Queries.
Runs against ephemeral in-memory connections, scans read-only source files,
enforces 512MB memory boundaries and hardware-level thread cancellation timers.
"""
from datetime import datetime, date
import json
from pathlib import Path
import threading
import time
from typing import Any, Optional
import duckdb
import pandas as pd

from app.analytics.schemas import AnalysisPlan, AnalysisResult
from app.analytics.compiler import CompiledQuery, compile_plan
from app.analytics.dataset_service import get_dataset, get_dataset_file_path

DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2MB JSON threshold


class ExecutionTimeoutError(Exception):
    """Raised when a query exceeds the allocated execution time limit."""
    pass


class ExecutionError(Exception):
    """Raised when query execution fails in the DuckDB kernel."""
    pass


def _sanitize_record_value(val: Any) -> Any:
    """Converts numpy/pandas/duckdb types to standard JSON-serializable Python types."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if hasattr(val, "item"):
        # numpy scalar to python scalar
        return val.item()
    return val


def execute_compiled_query(
    query: CompiledQuery,
    dataset_id: str,
    session_id: str,
    plan: AnalysisPlan,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> AnalysisResult:
    """
    Executes a CompiledQuery in an ephemeral, resource-bounded DuckDB instance.
    Interception timer enforces true query cancellation on long-running operations.
    """
    start_time = time.perf_counter()
    file_path = get_dataset_file_path(dataset_id, session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    conn = duckdb.connect(database=":memory:")
    interrupted = threading.Event()

    def _interrupt_conn():
        interrupted.set()
        try:
            conn.interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout_seconds, _interrupt_conn)

    try:
        # Enforce memory and concurrency bounds
        conn.execute("SET memory_limit = '512MB';")
        conn.execute("SET threads = 2;")

        # Register read-only source table
        fmt = metadata.format.lower()
        if fmt == "csv":
            rel = conn.read_csv(str(file_path))
            conn.register("source_table", rel)
        elif fmt == "json":
            rel = conn.read_json(str(file_path))
            conn.register("source_table", rel)
        elif fmt == "xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
            conn.register("source_table", df)
        else:
            raise ExecutionError(f"Unsupported dataset format '{fmt}'.")

        # Start cancellation timer
        timer.start()

        try:
            cursor = conn.execute(query.sql, query.parameters)
            df_result = cursor.fetchdf()
        except (duckdb.InterruptException, duckdb.FatalException) as e:
            if interrupted.is_set():
                raise ExecutionTimeoutError(
                    f"Query execution timed out after {timeout_seconds} seconds."
                )
            raise ExecutionError(f"DuckDB engine error: {e}")
        except Exception as e:
            if interrupted.is_set():
                raise ExecutionTimeoutError(
                    f"Query execution timed out after {timeout_seconds} seconds."
                )
            raise ExecutionError(f"Query execution failed: {e}")

    finally:
        timer.cancel()
        try:
            conn.close()
        except Exception:
            pass

    execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Process and sanitize output records
    col_names = [str(c) for c in df_result.columns]
    raw_records = df_result.to_dict(orient="records")

    rows: list[dict[str, Any]] = []
    current_bytes = 0
    truncated = False

    for record in raw_records:
        cleaned_record = {k: _sanitize_record_value(v) for k, v in record.items()}
        record_bytes = len(json.dumps(cleaned_record, default=str).encode("utf-8"))

        if current_bytes + record_bytes > MAX_RESPONSE_BYTES:
            truncated = True
            break

        rows.append(cleaned_record)
        current_bytes += record_bytes

    return AnalysisResult(
        plan=plan,
        compiled_sql=query.sql,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        columns=col_names,
        execution_time_ms=execution_time_ms,
    )


def execute_analysis_plan(
    plan: AnalysisPlan,
    session_id: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> AnalysisResult:
    """
    End-to-end execution pipeline:
    1. Validates session ownership and fetches metadata.
    2. Compiles plan into parameterized SQL.
    3. Executes query in DuckDB within resource boundaries.
    """
    metadata = get_dataset(plan.dataset_id, session_id=session_id)
    compiled = compile_plan(plan, metadata)
    return execute_compiled_query(
        query=compiled,
        dataset_id=plan.dataset_id,
        session_id=session_id,
        plan=plan,
        timeout_seconds=timeout_seconds,
    )
