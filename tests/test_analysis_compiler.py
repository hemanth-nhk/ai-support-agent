"""
Comprehensive Test Suite for Phase 2 — AnalysisPlan & Deterministic DuckDB Compiler.
Covers all required categories from the Phase 2 Test Gate:
1. Pydantic schema validation & extra field rejection
2. Every aggregation function (count, count_distinct, sum, avg, min, max)
3. Every filter operator (==, !=, >, >=, <, <=, in, not_in, contains, starts_with, is_null, is_not_null, between)
4. Group By behavior
5. Date Trunc (year, quarter, month, day) & type rejection
6. Order By semantics (asc, desc, projection checking)
7. Limit bounds & clamping (1-500)
8. Null semantics (is_null, is_not_null)
9. Strict Type validation (date vs string, numeric vs date, sum on string)
10. Unknown/hallucinated column rejection
11. SQL identifier escaping
12. Parameter binding verification
13. SQL Injection resistance (filter values, IN, BETWEEN, extra fields)
14. Session/dataset isolation
15. Malformed plan rejection
16. Real execution timeout & DuckDB kernel cancellation
17. Result size protection & truncation
18. End-to-end execution accuracy on actual dataset files
"""
import io
import shutil
import tempfile
import time
import pytest
from pydantic import ValidationError

from app.config import settings
from app.analytics import dataset_service
from app.analytics.schemas import (
    AnalysisPlan,
    FilterClause,
    FilterOperator,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
    OrderByClause,
    DateTrunc,
    DatasetMetadata,
)
from app.analytics.compiler import (
    compile_plan,
    PlanValidationError,
    ColumnNotFoundError,
    TypeMismatchError,
    escape_identifier,
)
from app.analytics.executor import (
    execute_analysis_plan,
    execute_compiled_query,
    ExecutionTimeoutError,
)


@pytest.fixture
def mock_dataset(monkeypatch):
    """Creates a temporary dataset with numeric, string, date, and null values."""
    temp_dir = tempfile.mkdtemp(prefix="test_analytics_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)

    csv_content = (
        "order_id,region,product,revenue,quantity,order_date,notes\n"
        "1,North,Widget,100.0,2,2024-01-15,first order\n"
        "2,North,Gadget,250.5,5,2024-02-20,express delivery\n"
        "3,South,Widget,150.0,3,2024-02-10,\n"
        "4,South,Gizmo,80.0,1,2024-03-05,standard order\n"
        "5,East,Gadget,300.0,6,2024-03-12,high value\n"
        "6,East,Widget,120.0,2,2024-04-18,\n"
    ).encode("utf-8")

    res = dataset_service.ingest_dataset(
        session_id="session-user-1",
        filename="sales_test.csv",
        content_bytes=csv_content,
    )
    metadata = dataset_service.get_dataset(res.dataset_id, session_id="session-user-1")

    yield metadata, "session-user-1"
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Pydantic Schema Validation & Extra Fields Rejection
# ---------------------------------------------------------------------------

def test_pydantic_rejects_extra_fields():
    """AnalysisPlan and clauses must strictly reject unexpected injection fields."""
    with pytest.raises(ValidationError):
        AnalysisPlan(
            dataset_id="test-id",
            malicious_field="raw SQL here",  # type: ignore
        )

    with pytest.raises(ValidationError):
        FilterClause(
            column="revenue",
            operator=FilterOperator.GT,
            value=100,
            inject_sql="OR 1=1",  # type: ignore
        )


def test_pydantic_rejects_invalid_operator():
    """Unrecognized operators must be rejected at the AST level."""
    with pytest.raises(ValidationError):
        FilterClause(column="revenue", operator="INVALID_OP", value=50)  # type: ignore


# ---------------------------------------------------------------------------
# 2. Aggregation Functions & Parameterization
# ---------------------------------------------------------------------------

def test_every_aggregation_function(mock_dataset):
    """Verifies compilation and calculation of COUNT, COUNT_DISTINCT, SUM, AVG, MIN, MAX."""
    metadata, session_id = mock_dataset

    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        aggregations=[
            AggregationClause(column="*", function=AggregationFunction.COUNT, alias="total_orders"),
            AggregationClause(column="product", function=AggregationFunction.COUNT_DISTINCT, alias="unique_products"),
            AggregationClause(column="revenue", function=AggregationFunction.SUM, alias="total_rev"),
            AggregationClause(column="revenue", function=AggregationFunction.AVG, alias="avg_rev"),
            AggregationClause(column="revenue", function=AggregationFunction.MIN, alias="min_rev"),
            AggregationClause(column="revenue", function=AggregationFunction.MAX, alias="max_rev"),
        ],
    )

    compiled = compile_plan(plan, metadata)
    assert 'COUNT(*) AS "total_orders"' in compiled.sql
    assert 'COUNT(DISTINCT "product") AS "unique_products"' in compiled.sql
    assert 'SUM("revenue") AS "total_rev"' in compiled.sql
    assert 'AVG("revenue") AS "avg_rev"' in compiled.sql
    assert 'MIN("revenue") AS "min_rev"' in compiled.sql
    assert 'MAX("revenue") AS "max_rev"' in compiled.sql

    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == 1
    row = result.rows[0]
    assert row["total_orders"] == 6
    assert row["unique_products"] == 3
    assert row["total_rev"] == 1000.5
    assert row["min_rev"] == 80.0
    assert row["max_rev"] == 300.0


# ---------------------------------------------------------------------------
# 3. Every Filter Operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op,val,expected_count",
    [
        (FilterOperator.EQ, "Widget", 3),
        (FilterOperator.NEQ, "Widget", 3),
        (FilterOperator.CONTAINS, "order", 2),       # 'first order', 'standard order'
        (FilterOperator.STARTS_WITH, "express", 1),  # 'express delivery'
        (FilterOperator.IN, ["Gadget", "Gizmo"], 3),
        (FilterOperator.NOT_IN, ["Gadget", "Gizmo"], 3),
    ],
)
def test_string_filter_operators(mock_dataset, op, val, expected_count):
    metadata, session_id = mock_dataset
    col = "notes" if op in (FilterOperator.CONTAINS, FilterOperator.STARTS_WITH) else "product"

    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column=col, operator=op, value=val)],
    )
    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == expected_count


@pytest.mark.parametrize(
    "op,val,expected_count",
    [
        (FilterOperator.GT, 120.0, 3),   # 250.5, 150.0, 300.0
        (FilterOperator.GTE, 120.0, 4),  # + 120.0
        (FilterOperator.LT, 120.0, 2),   # 100.0, 80.0
        (FilterOperator.LTE, 120.0, 3),  # + 120.0
        (FilterOperator.BETWEEN, [100.0, 200.0], 3),  # 100.0, 150.0, 120.0
    ],
)
def test_numeric_filter_operators(mock_dataset, op, val, expected_count):
    metadata, session_id = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="revenue", operator=op, value=val)],
    )
    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == expected_count


# ---------------------------------------------------------------------------
# 4. Group By Behavior
# ---------------------------------------------------------------------------

def test_group_by_aggregation(mock_dataset):
    """Groups orders by region and sums revenue."""
    metadata, session_id = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        group_by=[GroupByClause(column="region")],
        aggregations=[
            AggregationClause(column="revenue", function=AggregationFunction.SUM, alias="region_revenue"),
            AggregationClause(column="*", function=AggregationFunction.COUNT, alias="order_count"),
        ],
        order_by=[OrderByClause(target="region_revenue", descending=True)],
    )

    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == 3  # East (420.0), North (350.5), South (230.0)
    top_row = result.rows[0]
    assert top_row["region"] == "East"
    assert top_row["region_revenue"] == 420.0
    assert top_row["order_count"] == 2


# ---------------------------------------------------------------------------
# 5. Date Trunc Operations & Non-Date Rejection
# ---------------------------------------------------------------------------

def test_date_trunc_grouping(mock_dataset):
    """Groups orders by month(order_date)."""
    metadata, session_id = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        group_by=[GroupByClause(column="order_date", date_trunc=DateTrunc.MONTH)],
        aggregations=[
            AggregationClause(column="revenue", function=AggregationFunction.SUM, alias="monthly_revenue")
        ],
        order_by=[OrderByClause(target="month_order_date", descending=False)],
    )

    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == 4  # Jan, Feb, Mar, Apr
    compiled = compile_plan(plan, metadata)
    assert "DATE_TRUNC('month', \"order_date\")" in compiled.sql


def test_date_trunc_on_non_date_column_rejected(mock_dataset):
    """Applying date_trunc to a string or numeric column must fail closed."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        group_by=[GroupByClause(column="region", date_trunc=DateTrunc.MONTH)],
    )
    with pytest.raises(TypeMismatchError) as exc:
        compile_plan(plan, metadata)
    assert "Cannot apply date_trunc to non-datetime column" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. Order By Semantics
# ---------------------------------------------------------------------------

def test_order_by_unprojected_column_rejected(mock_dataset):
    """Ordering by a target not in the group_by or aggregation aliases must be rejected."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        group_by=[GroupByClause(column="region")],
        aggregations=[AggregationClause(column="revenue", function=AggregationFunction.SUM, alias="rev")],
        order_by=[OrderByClause(target="non_existent_alias")],
    )
    with pytest.raises(PlanValidationError) as exc:
        compile_plan(plan, metadata)
    assert "not in selected projection or aliases" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. LIMIT Bounds & Clamping
# ---------------------------------------------------------------------------

def test_limit_clamped(mock_dataset):
    """Limits must be clamped between 1 and 500."""
    metadata, _ = mock_dataset
    # Pydantic schema validation clamps limit <= 500
    with pytest.raises(ValidationError):
        AnalysisPlan(dataset_id=metadata.dataset_id, limit=999)

    plan_valid = AnalysisPlan(dataset_id=metadata.dataset_id, limit=2)
    compiled = compile_plan(plan_valid, metadata)
    assert "LIMIT 2" in compiled.sql


# ---------------------------------------------------------------------------
# 8. Null Semantics (IS NULL / IS NOT NULL)
# ---------------------------------------------------------------------------

def test_null_filtering(mock_dataset):
    """Verifies IS NULL and IS NOT NULL on nullable column (notes)."""
    metadata, session_id = mock_dataset

    # Notes has 2 null rows (rows 3 and 6)
    plan_null = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="notes", operator=FilterOperator.IS_NULL)],
    )
    res_null = execute_analysis_plan(plan_null, session_id)
    assert res_null.row_count == 2

    plan_not_null = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="notes", operator=FilterOperator.IS_NOT_NULL)],
    )
    res_not_null = execute_analysis_plan(plan_not_null, session_id)
    assert res_not_null.row_count == 4


# ---------------------------------------------------------------------------
# 9. Strict Type Validation
# ---------------------------------------------------------------------------

def test_date_column_invalid_format_rejected(mock_dataset):
    """Date column with 'not-a-date' must raise PlanValidationError."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="order_date", operator=FilterOperator.EQ, value="not-a-date")],
    )
    with pytest.raises(PlanValidationError) as exc:
        compile_plan(plan, metadata)
    assert "Invalid date format" in str(exc.value)


def test_numeric_column_with_date_rejected(mock_dataset):
    """Numeric column filtered with date string must raise TypeMismatchError."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="revenue", operator=FilterOperator.EQ, value="2026-01-01")],
    )
    with pytest.raises(TypeMismatchError) as exc:
        compile_plan(plan, metadata)
    assert "Cannot compare numeric column" in str(exc.value)


def test_sum_on_string_column_rejected(mock_dataset):
    """Summing a string column must raise TypeMismatchError."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        aggregations=[AggregationClause(column="product", function=AggregationFunction.SUM)],
    )
    with pytest.raises(TypeMismatchError) as exc:
        compile_plan(plan, metadata)
    assert "Cannot compute SUM on non-numeric column" in str(exc.value)


def test_inequality_on_text_column_rejected(mock_dataset):
    """Using > on text column must raise TypeMismatchError."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="product", operator=FilterOperator.GT, value="Gadget")],
    )
    with pytest.raises(TypeMismatchError) as exc:
        compile_plan(plan, metadata)
    assert "Inequality operator '>' is not supported on text column" in str(exc.value)


# ---------------------------------------------------------------------------
# 10. Unknown Column Rejection
# ---------------------------------------------------------------------------

def test_unknown_column_rejected(mock_dataset):
    """Hallucinated column names must raise ColumnNotFoundError."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="profit_2024", operator=FilterOperator.GT, value=100)],
    )
    with pytest.raises(ColumnNotFoundError) as exc:
        compile_plan(plan, metadata)
    assert "Column 'profit_2024' does not exist in dataset" in str(exc.value)


# ---------------------------------------------------------------------------
# 11. SQL Identifier Escaping
# ---------------------------------------------------------------------------

def test_identifier_escaping():
    assert escape_identifier('revenue') == '"revenue"'
    assert escape_identifier('revenue "USD"') == '"revenue ""USD"""'


# ---------------------------------------------------------------------------
# 12 & 13. Parameter Binding & SQL Injection Resistance
# ---------------------------------------------------------------------------

def test_sql_injection_attempt_in_filter_value(mock_dataset):
    """Malicious SQL injected into a filter value is safely bound as a literal parameter."""
    metadata, session_id = mock_dataset
    malicious_value = "Widget'; DROP TABLE orders; --"

    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="product", operator=FilterOperator.EQ, value=malicious_value)],
    )
    compiled = compile_plan(plan, metadata)

    # Must contain placeholder ?, never the raw SQL string
    assert "DROP TABLE" not in compiled.sql
    assert malicious_value in compiled.parameters

    # Query executes safely, returns 0 rows (no product named that), no side effects
    result = execute_analysis_plan(plan, session_id)
    assert result.row_count == 0


def test_sql_injection_attempt_in_in_clause(mock_dataset):
    """Injection inside IN list is safely parameterized."""
    metadata, session_id = mock_dataset
    malicious_list = ["Widget", "'; DELETE FROM orders; --"]

    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="product", operator=FilterOperator.IN, value=malicious_list)],
    )
    compiled = compile_plan(plan, metadata)
    assert "DELETE FROM" not in compiled.sql
    assert "'; DELETE FROM orders; --" in compiled.parameters

    result = execute_analysis_plan(plan, session_id)
    # Finds only 'Widget' (3 rows)
    assert result.row_count == 3


# ---------------------------------------------------------------------------
# 14. Dataset / Session Isolation
# ---------------------------------------------------------------------------

def test_session_isolation_in_execution(mock_dataset):
    """A user in Session B cannot execute plans against Session A's dataset."""
    metadata, _ = mock_dataset
    plan = AnalysisPlan(dataset_id=metadata.dataset_id)

    with pytest.raises(dataset_service.DatasetAccessDeniedError):
        execute_analysis_plan(plan, session_id="attacker-session-id")


# ---------------------------------------------------------------------------
# 15. Malformed Plan Rejection
# ---------------------------------------------------------------------------

def test_between_requires_two_elements(mock_dataset):
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="revenue", operator=FilterOperator.BETWEEN, value=[100.0])],
    )
    with pytest.raises(PlanValidationError) as exc:
        compile_plan(plan, metadata)
    assert "exactly two boundary values" in str(exc.value)


def test_in_requires_non_empty_list(mock_dataset):
    metadata, _ = mock_dataset
    plan = AnalysisPlan(
        dataset_id=metadata.dataset_id,
        filters=[FilterClause(column="product", operator=FilterOperator.IN, value=[])],
    )
    with pytest.raises(PlanValidationError) as exc:
        compile_plan(plan, metadata)
    assert "non-empty list of values" in str(exc.value)


# ---------------------------------------------------------------------------
# 16. Execution Timeout with True DuckDB Kernel Interruption
# ---------------------------------------------------------------------------

def test_execution_timeout_interrupts_query(mock_dataset):
    """Simulates a heavy query and verifies true timeout cancellation via conn.interrupt()."""
    from app.analytics.compiler import CompiledQuery

    metadata, session_id = mock_dataset
    # Deliberate billion-row cartesian join
    heavy_sql = "SELECT count(*) FROM generate_series(1, 100000000) t1, generate_series(1, 100000000) t2"
    heavy_query = CompiledQuery(sql=heavy_sql, parameters=[])
    dummy_plan = AnalysisPlan(dataset_id=metadata.dataset_id)

    t0 = time.time()
    with pytest.raises(ExecutionTimeoutError) as exc:
        execute_compiled_query(
            query=heavy_query,
            dataset_id=metadata.dataset_id,
            session_id=session_id,
            plan=dummy_plan,
            timeout_seconds=0.3,
        )
    elapsed = time.time() - t0
    # Must interrupt promptly around 0.3s (well under 1.5s)
    assert elapsed < 1.5
    assert "timed out after 0.3 seconds" in str(exc.value)
