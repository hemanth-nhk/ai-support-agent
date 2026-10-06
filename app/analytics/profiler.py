"""
Deterministic fast dataset profiler using Pandas and DuckDB.
Extracts shape, data types, missing value metrics, and duplicate rows.
No LLM or stochastic operations.
"""
import io
import json
from typing import Union
import pandas as pd

from app.analytics.schemas import ColumnProfile, DatasetProfile


class DatasetProfileError(Exception):
    """Raised when profiling fails due to malformed or unparseable tabular data."""
    pass


def load_dataframe_from_bytes(content_bytes: bytes, file_format: str) -> pd.DataFrame:
    """
    Safely loads raw bytes into a pandas DataFrame based on file_format ('csv', 'xlsx', 'json').
    Raises DatasetProfileError on parsing errors or empty content.
    """
    fmt = file_format.lower().lstrip(".")

    if fmt == "csv":
        try:
            # Try UTF-8 first
            df = pd.read_csv(io.BytesIO(content_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(content_bytes), encoding="latin-1")
            except Exception as e:
                raise DatasetProfileError(f"Failed to decode and parse CSV file: {e}")
        except pd.errors.EmptyDataError:
            raise DatasetProfileError("CSV file is empty or contains no readable columns.")
        except Exception as e:
            raise DatasetProfileError(f"Failed to parse CSV file: {e}")

    elif fmt == "xlsx":
        try:
            df = pd.read_excel(io.BytesIO(content_bytes), engine="openpyxl")
        except Exception as e:
            raise DatasetProfileError(f"Failed to parse Excel (XLSX) file: {e}")

    elif fmt == "json":
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("latin-1")
            except Exception as e:
                raise DatasetProfileError(f"Failed to decode JSON file: {e}")

        try:
            raw_data = json.loads(text)
        except json.JSONDecodeError as e:
            raise DatasetProfileError(f"Invalid JSON syntax: {e}")

        if isinstance(raw_data, list):
            if not raw_data:
                raise DatasetProfileError("JSON array is empty.")
            # List of dicts or records
            try:
                df = pd.DataFrame(raw_data)
            except Exception as e:
                raise DatasetProfileError(f"Failed to construct table from JSON array: {e}")
        elif isinstance(raw_data, dict):
            if not raw_data:
                raise DatasetProfileError("JSON object is empty.")
            # Check if dict wraps records under common keys ('data', 'records', 'items', 'rows')
            record_key = None
            for key in ("data", "records", "items", "rows"):
                if key in raw_data and isinstance(raw_data[key], list) and len(raw_data[key]) > 0:
                    record_key = key
                    break
            if record_key is not None:
                try:
                    df = pd.DataFrame(raw_data[record_key])
                except Exception as e:
                    raise DatasetProfileError(f"Failed to construct table from JSON key '{record_key}': {e}")
            else:
                try:
                    # Column-oriented or single record
                    df = pd.DataFrame(raw_data)
                except Exception:
                    try:
                        # Fallback for single record dictionary
                        df = pd.DataFrame([raw_data])
                    except Exception as e:
                        raise DatasetProfileError(f"Failed to convert JSON dictionary to tabular data: {e}")
        else:
            raise DatasetProfileError("JSON must contain an array or object representing tabular records.")

    else:
        raise DatasetProfileError(f"Unsupported dataset format '{fmt}'.")

    if df.empty:
        raise DatasetProfileError("Dataset contains no data rows.")
    if len(df.columns) == 0:
        raise DatasetProfileError("Dataset contains no columns.")

    # Automatically infer datetime columns where applicable
    for col in df.columns:
        if any(k in str(col).lower() for k in ("date", "time", "_at", "timestamp")):
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                pass

    return df


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """
    Computes deterministic statistical and schema profile from a pandas DataFrame.
    """
    row_count = int(len(df))
    col_count = int(len(df.columns))

    columns: list[ColumnProfile] = []
    for col in df.columns:
        col_name = str(col)
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / row_count, 4) if row_count > 0 else 0.0
        
        # Calculate unique count safely (handling unhashable types if any exist)
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            # For columns containing unhashable objects (e.g. lists or dicts)
            unique_count = int(series.astype(str).nunique(dropna=True))

        columns.append(
            ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=unique_count,
            )
        )

    # Missing cells across all columns
    missing_cells = int(df.isna().sum().sum())

    # Duplicate rows count
    try:
        duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0
    except TypeError:
        # Fallback if unhashable columns exist in JSON
        duplicate_rows = int(df.astype(str).duplicated().sum()) if row_count > 0 else 0

    return DatasetProfile(
        row_count=row_count,
        column_count=col_count,
        columns=columns,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
    )


def profile_dataset_bytes(content_bytes: bytes, file_format: str) -> DatasetProfile:
    """
    End-to-end helper: parses raw bytes and generates a structured DatasetProfile.
    """
    df = load_dataframe_from_bytes(content_bytes, file_format)
    return profile_dataframe(df)
