"""
Unit and execution tests for Phase 6 Deterministic DuckDB Compiler extensions:
6A: Derived metrics (ratios, percentages, safe division by zero)
6B: Statistical functions (MEDIAN, STDDEV, PERCENTILE_CONT)
6C: Bivariate Pearson Correlation (CORR, constant variance safety)
6D: Period Comparison (LAG window CTE, growth %, delta, single/missing period)
"""
import duckdb
import pytest

from app.analytics.compiler import (
    compile_plan,
    PlanValidationError,
    ColumnNotFoundError,
    TypeMismatchError,
)
from app.analytics.schemas import (
    AnalysisPlan,
    AggregationClause,
    AggregationFunction,
    DerivedMetric,
    DerivedMetricOperation,
    CorrelationClause,
    PeriodComparison,
    PeriodComparisonMode,
    DateTrunc,
    GroupByClause,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
)


@pytest.fixture
def mock_metadata():
    """Metadata with diverse column types: numeric, string, datetime."""
    cols = [
        ColumnProfile(name="order_id", dtype="int64", missing_count=0, missing_percentage=0.0, unique_count=100),
        ColumnProfile(name="category", dtype="str", missing_count=0, missing_percentage=0.0, unique_count=5),
        ColumnProfile(name="revenue", dtype="float64", missing_count=0, missing_percentage=0.0, unique_count=80),
        ColumnProfile(name="profit", dtype="float64", missing_count=0, missing_percentage=0.0, unique_count=75),
        ColumnProfile(name="order_date", dtype="datetime64[us]", missing_count=0, missing_percentage=0.0, unique_count=30),
    ]
    prof = DatasetProfile(
        row_count=100,
        column_count=5,
        columns=cols,
        missing_cells=0,
        duplicate_rows=0,
    )
    return DatasetMetadata(
        dataset_id="test-dataset-v2",
        session_id="test-session-v2",
        filename="sales_v2.csv",
        format="csv",
        size_bytes=4096,
        row_count=100,
        column_count=5,
        created_at="2026-01-01T00:00:00Z",
        profile=prof,
    )


# ---------------------------------------------------------------------------
# 6A: Derived Metrics (Ratios & Percentages)
# ---------------------------------------------------------------------------

def test_derived_metric_ratio_compiles_and_executes(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        group_by=[GroupByClause(column="category")],
        derived_metrics=[
            DerivedMetric(
                operation=DerivedMetricOperation.RATIO,
                numerator="profit",
                denominator="revenue",
                multiplier=1.0,
                alias="profit_ratio",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'SUM("profit")' in compiled.sql
    assert 'NULLIF(SUM("revenue"), 0)' in compiled.sql
    assert 'AS "profit_ratio"' in compiled.sql

    # Execute against in-memory DuckDB table
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE source_table (category VARCHAR, profit DOUBLE, revenue DOUBLE);
        INSERT INTO source_table VALUES
          ('Electronics', 200.0, 1000.0),
          ('Books', 50.0, 100.0),
          ('ZeroRevenue', 10.0, 0.0);
    """)
    res = con.execute(compiled.sql).fetchall()
    row_map = {r[0]: r[1] for r in res}
    assert row_map["Electronics"] == pytest.approx(0.2)
    assert row_map["Books"] == pytest.approx(0.5)
    # Zero revenue MUST result in NULL (None), never crash or 0
    assert row_map["ZeroRevenue"] is None
    con.close()


def test_derived_metric_percentage_multiplier(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        derived_metrics=[
            DerivedMetric(
                operation=DerivedMetricOperation.PERCENTAGE,
                numerator="profit",
                denominator="revenue",
                alias="profit_margin_pct",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert '100.0' in compiled.sql

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE source_table (profit DOUBLE, revenue DOUBLE); INSERT INTO source_table VALUES (25.0, 100.0);")
    res = con.execute(compiled.sql).fetchall()
    assert res[0][0] == pytest.approx(25.0)
    con.close()


def test_derived_metric_rejects_non_numeric(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        derived_metrics=[
            DerivedMetric(
                numerator="category",  # string!
                denominator="revenue",
                alias="invalid_ratio",
            )
        ],
    )
    with pytest.raises(TypeMismatchError, match="must be numeric"):
        compile_plan(plan, mock_metadata)


# ---------------------------------------------------------------------------
# 6B: Statistical Aggregations (MEDIAN, STDDEV, PERCENTILE_CONT)
# ---------------------------------------------------------------------------

def test_statistical_aggregations_compile_and_execute(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        aggregations=[
            AggregationClause(column="profit", function=AggregationFunction.MEDIAN, alias="med_profit"),
            AggregationClause(column="profit", function=AggregationFunction.STDDEV, alias="std_profit"),
            AggregationClause(
                column="profit",
                function=AggregationFunction.PERCENTILE_CONT,
                quantile=0.75,
                alias="p75_profit",
            ),
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'MEDIAN("profit") AS "med_profit"' in compiled.sql
    assert 'STDDEV_SAMP("profit") AS "std_profit"' in compiled.sql
    assert 'QUANTILE_CONT("profit", 0.75) AS "p75_profit"' in compiled.sql

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE source_table (profit DOUBLE);
        INSERT INTO source_table VALUES (10.0), (20.0), (30.0), (40.0), (50.0);
    """)
    res = con.execute(compiled.sql).fetchone()
    # median of [10, 20, 30, 40, 50] is 30.0
    assert res[0] == pytest.approx(30.0)
    # stddev of [10, 20, 30, 40, 50] is 15.811388...
    assert res[1] == pytest.approx(15.811388, rel=1e-3)
    # 75th percentile is 40.0
    assert res[2] == pytest.approx(40.0)
    con.close()


def test_percentile_cont_requires_valid_quantile():
    # Missing quantile must fail Pydantic validation
    with pytest.raises(ValueError, match="requires a 'quantile' parameter"):
        AggregationClause(column="profit", function=AggregationFunction.PERCENTILE_CONT)

    # Quantile outside [0.0, 1.0] must fail
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        AggregationClause(column="profit", function=AggregationFunction.PERCENTILE_CONT, quantile=1.5)

    # Non-percentile function with quantile must fail
    with pytest.raises(ValueError, match="does not accept a 'quantile' parameter"):
        AggregationClause(column="profit", function=AggregationFunction.SUM, quantile=0.5)


def test_statistical_aggregations_reject_string(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        aggregations=[
            AggregationClause(column="category", function=AggregationFunction.MEDIAN, alias="med_cat"),
        ],
    )
    with pytest.raises(TypeMismatchError, match="non-numeric"):
        compile_plan(plan, mock_metadata)


# ---------------------------------------------------------------------------
# 6C: Bivariate Pearson Correlation (CORR)
# ---------------------------------------------------------------------------

def test_correlation_compiles_and_executes(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        correlations=[
            CorrelationClause(column_a="profit", column_b="revenue", alias="r_profit_revenue"),
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'CORR("profit", "revenue")' in compiled.sql
    assert 'AS "r_profit_revenue"' in compiled.sql

    con = duckdb.connect(":memory:")
    # Perfect positive correlation (profit = 0.5 * revenue)
    con.execute("""
        CREATE TABLE source_table (profit DOUBLE, revenue DOUBLE);
        INSERT INTO source_table VALUES
          (10.0, 20.0),
          (20.0, 40.0),
          (30.0, 60.0),
          (40.0, 80.0);
    """)
    res = con.execute(compiled.sql).fetchone()
    assert res[0] == pytest.approx(1.0)

    # Test constant variance case: profit is always 10.0 -> correlation is undefined -> returns None (NULL)
    con.execute("DELETE FROM source_table;")
    con.execute("""
        INSERT INTO source_table VALUES
          (10.0, 20.0),
          (10.0, 40.0),
          (10.0, 60.0);
    """)
    res_const = con.execute(compiled.sql).fetchone()
    assert res_const[0] is None
    con.close()


def test_correlation_rejects_non_numeric(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        correlations=[
            CorrelationClause(column_a="category", column_b="revenue"),
        ],
    )
    with pytest.raises(TypeMismatchError, match="requires numeric"):
        compile_plan(plan, mock_metadata)


# ---------------------------------------------------------------------------
# 6D: Period Comparison (LAG CTE, growth_pct & delta)
# ---------------------------------------------------------------------------

def test_period_comparison_growth_pct_executes(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        period_comparison=PeriodComparison(
            date_column="order_date",
            metric="revenue",
            period=DateTrunc.MONTH,
            mode=PeriodComparisonMode.GROWTH_PCT,
            alias="mom_growth",
        ),
    )
    compiled = compile_plan(plan, mock_metadata)
    assert "WITH period_summary AS" in compiled.sql
    assert "LAG(current_val) OVER (ORDER BY period)" in compiled.sql
    assert 'AS "mom_growth"' in compiled.sql

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE source_table (order_date TIMESTAMP, revenue DOUBLE);
        INSERT INTO source_table VALUES
          ('2024-01-10 10:00:00', 10000.0),
          ('2024-02-15 11:00:00', 12000.0),
          ('2024-03-20 12:00:00', 15000.0);
    """)
    rows = con.execute(compiled.sql).fetchall()
    assert len(rows) == 3

    # Jan 2024: first period -> growth is NULL
    assert rows[0][1] == 10000.0
    assert rows[0][2] is None  # prev_val
    assert rows[0][3] is None  # growth_pct

    # Feb 2024: (12000 - 10000) / 10000 * 100 = 20%
    assert rows[1][1] == 12000.0
    assert rows[1][2] == 10000.0
    assert rows[1][3] == pytest.approx(20.0)

    # Mar 2024: (15000 - 12000) / 12000 * 100 = 25%
    assert rows[2][1] == 15000.0
    assert rows[2][2] == 12000.0
    assert rows[2][3] == pytest.approx(25.0)
    con.close()


def test_period_comparison_single_period_handled_safely(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        period_comparison=PeriodComparison(
            date_column="order_date",
            metric="revenue",
            period=DateTrunc.MONTH,
            mode=PeriodComparisonMode.GROWTH_PCT,
        ),
    )
    compiled = compile_plan(plan, mock_metadata)

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE source_table (order_date TIMESTAMP, revenue DOUBLE);
        INSERT INTO source_table VALUES ('2024-01-10', 500.0);
    """)
    rows = con.execute(compiled.sql).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == 500.0
    assert rows[0][2] is None
    assert rows[0][3] is None  # Graceful None for single period
    con.close()


def test_period_comparison_delta_mode(mock_metadata):
    plan = AnalysisPlan(
        dataset_id="test-dataset-v2",
        period_comparison=PeriodComparison(
            date_column="order_date",
            metric="revenue",
            period=DateTrunc.MONTH,
            mode=PeriodComparisonMode.DELTA,
            alias="revenue_diff",
        ),
    )
    compiled = compile_plan(plan, mock_metadata)

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE source_table (order_date TIMESTAMP, revenue DOUBLE);
        INSERT INTO source_table VALUES
          ('2024-01-01', 100.0),
          ('2024-02-01', 150.0);
    """)
    rows = con.execute(compiled.sql).fetchall()
    assert rows[0][3] is None
    assert rows[1][3] == pytest.approx(50.0)
    con.close()
