"""
Comprehensive unit and execution tests for Phase 7 Deterministic DuckDB Compiler extensions:
7A: Rolling window compilation (AVG, SUM, MIN, MAX, custom preceding, partitioning)
7B: Cumulative metrics compilation (SUM, AVG, COUNT, unbounded preceding, partitioning)
7C: Outlier compilation (Z-score stats CTE, zero-variance safety, single-observation safety, partitioned outliers)
7D: Quantile segmentation (NTILE, ascending/descending, partitioning)
7E: Fail-closed boundary, type, and schema validation
"""
import duckdb
import pandas as pd
import pytest
from pydantic import ValidationError

from app.analytics.compiler import (
    compile_plan,
    PlanValidationError,
    ColumnNotFoundError,
    TypeMismatchError,
)
from app.analytics.schemas import (
    AnalysisPlan,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    RollingWindowSpec,
    WindowFunction,
    CumulativeSpec,
    CumulativeFunction,
    OutlierSpec,
    QuantileSpec,
    OrderByClause,
)


@pytest.fixture
def mock_metadata():
    """Metadata with diverse column types: numeric, string, datetime."""
    cols = [
        ColumnProfile(name="order_id", dtype="int64", missing_count=0, missing_percentage=0.0, unique_count=100),
        ColumnProfile(name="region", dtype="str", missing_count=0, missing_percentage=0.0, unique_count=4),
        ColumnProfile(name="revenue", dtype="float64", missing_count=0, missing_percentage=0.0, unique_count=80),
        ColumnProfile(name="order_date", dtype="datetime64[us]", missing_count=0, missing_percentage=0.0, unique_count=30),
    ]
    prof = DatasetProfile(
        row_count=100,
        column_count=4,
        columns=cols,
        missing_cells=0,
        duplicate_rows=0,
    )
    return DatasetMetadata(
        dataset_id="test-dataset-v3",
        session_id="test-session-v3",
        filename="sales_v3.csv",
        format="csv",
        row_count=100,
        column_count=4,
        created_at="2026-01-01T00:00:00Z",
        profile=prof,
    )


# ---------------------------------------------------------------------------
# 7A: Rolling Windows Compilation & Execution
# ---------------------------------------------------------------------------

def test_rolling_window_sql_generation(mock_metadata):
    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        rolling_windows=[
            RollingWindowSpec(
                function=WindowFunction.AVG,
                column="revenue",
                order_by="order_date",
                preceding=6,
                alias="rolling_7d_avg",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'AVG("revenue") OVER (ORDER BY "order_date" ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS "rolling_7d_avg"' in compiled.sql


def test_rolling_window_with_partitioning_execution(mock_metadata):
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "region": ["North", "North", "North", "South", "South"],
        "revenue": [10.0, 20.0, 30.0, 100.0, 200.0],
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"],
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        rolling_windows=[
            RollingWindowSpec(
                function=WindowFunction.SUM,
                column="revenue",
                order_by="order_date",
                preceding=1,
                partition_by=["region"],
                alias="rolling_2p_sum",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    res = conn.execute(compiled.sql).fetchdf()

    assert "rolling_2p_sum" in res.columns
    # North: row 1 (10), row 2 (10+20=30), row 3 (20+30=50)
    north_rows = res[res["region"] == "North"].sort_values("order_date")
    assert north_rows["rolling_2p_sum"].tolist() == [10.0, 30.0, 50.0]


# ---------------------------------------------------------------------------
# 7B: Cumulative Metrics Compilation & Execution
# ---------------------------------------------------------------------------

def test_cumulative_metrics_sql_and_execution(mock_metadata):
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "region": ["East", "East", "West", "West"],
        "revenue": [100.0, 150.0, 200.0, 250.0],
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        cumulative_metrics=[
            CumulativeSpec(
                function=CumulativeFunction.SUM,
                column="revenue",
                order_by="order_date",
                alias="running_rev",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'SUM("revenue") OVER (ORDER BY "order_date" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS "running_rev"' in compiled.sql

    res = conn.execute(compiled.sql).fetchdf()
    assert res["running_rev"].iloc[-1] == 700.0  # Total accumulated sum


# ---------------------------------------------------------------------------
# 7C: Quantile Segmentation (NTILE)
# ---------------------------------------------------------------------------

def test_quantile_segmentation_ntile_execution(mock_metadata):
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": list(range(1, 9)),
        "region": ["All"] * 8,
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "order_date": ["2026-01-01"] * 8,
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        quantiles=[
            QuantileSpec(
                column="revenue",
                num_quantiles=4,
                order_descending=False,
                alias="spend_quartile",
            )
        ],
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'NTILE(4) OVER (ORDER BY "revenue" ASC) AS "spend_quartile"' in compiled.sql

    res = conn.execute(compiled.sql).fetchdf().sort_values("revenue")
    assert res["spend_quartile"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


# ---------------------------------------------------------------------------
# 7D: Outlier Compilation & Mathematical Safety
# ---------------------------------------------------------------------------

def test_outlier_compilation_with_anomalies(mock_metadata):
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "region": ["Central"] * 5,
        "revenue": [10.0, 10.5, 9.8, 10.2, 100.0],  # 100 is anomaly
        "order_date": ["2026-01-01"] * 5,
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        outlier_analysis=OutlierSpec(
            column="revenue",
            threshold=1.5,
            filter_outliers_only=True,
        ),
    )
    compiled = compile_plan(plan, mock_metadata)
    assert "WITH stats_cte AS" in compiled.sql
    assert "STDDEV_SAMP" in compiled.sql

    res = conn.execute(compiled.sql).fetchdf()
    assert len(res) == 1
    assert res.iloc[0]["order_id"] == 5
    assert res.iloc[0]["is_outlier"] == True
    assert res.iloc[0]["z_score_revenue"] > 1.5


def test_outlier_zero_variance_fail_closed(mock_metadata):
    """When standard deviation is zero (all values identical), z_score must be NULL and is_outlier must be False."""
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "region": ["West"] * 3,
        "revenue": [50.0, 50.0, 50.0],
        "order_date": ["2026-01-01"] * 3,
    })
    conn.register("source_table", df)

    # When filter_outliers_only is False, verify NULL z-score and False is_outlier
    plan_unfiltered = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        outlier_analysis=OutlierSpec(
            column="revenue",
            threshold=2.0,
            filter_outliers_only=False,
        ),
    )
    compiled_unfiltered = compile_plan(plan_unfiltered, mock_metadata)
    res_unfiltered = conn.execute(compiled_unfiltered.sql).fetchdf()

    assert len(res_unfiltered) == 3
    assert res_unfiltered["stddev_revenue"].iloc[0] == 0.0
    assert res_unfiltered["z_score_revenue"].isna().all()
    assert (res_unfiltered["is_outlier"] == False).all()

    # When filter_outliers_only is True, verify zero rows returned (fail-closed, no false alarms)
    plan_filtered = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        outlier_analysis=OutlierSpec(
            column="revenue",
            threshold=2.0,
            filter_outliers_only=True,
        ),
    )
    compiled_filtered = compile_plan(plan_filtered, mock_metadata)
    res_filtered = conn.execute(compiled_filtered.sql).fetchdf()
    assert len(res_filtered) == 0


def test_outlier_single_observation_safety(mock_metadata):
    """When N=1, sample STDDEV is NULL. Must evaluate z_score to NULL and is_outlier to False."""
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1],
        "region": ["West"],
        "revenue": [42.0],
        "order_date": ["2026-01-01"],
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        outlier_analysis=OutlierSpec(
            column="revenue",
            threshold=3.0,
            filter_outliers_only=False,
        ),
    )
    compiled = compile_plan(plan, mock_metadata)
    res = conn.execute(compiled.sql).fetchdf()

    assert len(res) == 1
    assert res["stddev_revenue"].isna().iloc[0]
    assert res["z_score_revenue"].isna().iloc[0]
    assert res["is_outlier"].iloc[0] == False


def test_outlier_partitioned_execution(mock_metadata):
    """Outliers computed within category partitions."""
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "region": ["A", "A", "B", "B"],
        "revenue": [10.0, 100.0, 500.0, 510.0],  # 100 is outlier in A, but B is around 500
        "order_date": ["2026-01-01"] * 4,
    })
    conn.register("source_table", df)

    plan = AnalysisPlan(
        dataset_id=mock_metadata.dataset_id,
        outlier_analysis=OutlierSpec(
            column="revenue",
            partition_by=["region"],
            threshold=1.0,
            filter_outliers_only=False,
        ),
    )
    compiled = compile_plan(plan, mock_metadata)
    assert 'PARTITION BY "region"' in compiled.sql

    res = conn.execute(compiled.sql).fetchdf()
    assert len(res) == 4
    # Mean for A is 55.0, mean for B is 505.0
    a_mean = res[res["region"] == "A"]["mean_revenue"].iloc[0]
    b_mean = res[res["region"] == "B"]["mean_revenue"].iloc[0]
    assert a_mean == 55.0
    assert b_mean == 505.0


# ---------------------------------------------------------------------------
# 7E: Fail-Closed Validation, Type Checks & Bounds
# ---------------------------------------------------------------------------

def test_missing_column_rejections(mock_metadata):
    # Missing target column in rolling window
    with pytest.raises(ColumnNotFoundError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                rolling_windows=[RollingWindowSpec(function=WindowFunction.AVG, column="missing", order_by="order_date", preceding=1)],
            ),
            mock_metadata,
        )

    # Missing order_by in cumulative
    with pytest.raises(ColumnNotFoundError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                cumulative_metrics=[CumulativeSpec(function=CumulativeFunction.SUM, column="revenue", order_by="missing_date")],
            ),
            mock_metadata,
        )

    # Missing column in outlier
    with pytest.raises(ColumnNotFoundError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                outlier_analysis=OutlierSpec(column="ghost_metric"),
            ),
            mock_metadata,
        )


def test_type_mismatch_rejections(mock_metadata):
    # Rolling window on text column
    with pytest.raises(TypeMismatchError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                rolling_windows=[RollingWindowSpec(function=WindowFunction.AVG, column="region", order_by="order_date", preceding=1)],
            ),
            mock_metadata,
        )

    # Outlier analysis on text column
    with pytest.raises(TypeMismatchError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                outlier_analysis=OutlierSpec(column="region"),
            ),
            mock_metadata,
        )

    # Quantile on text column
    with pytest.raises(TypeMismatchError):
        compile_plan(
            AnalysisPlan(
                dataset_id=mock_metadata.dataset_id,
                quantiles=[QuantileSpec(column="region", num_quantiles=4)],
            ),
            mock_metadata,
        )


def test_pydantic_schema_boundary_rejections():
    # Preceding out of bounds (< 1)
    with pytest.raises(ValidationError):
        RollingWindowSpec(function=WindowFunction.AVG, column="x", order_by="d", preceding=0)

    # Preceding out of bounds (> 365)
    with pytest.raises(ValidationError):
        RollingWindowSpec(function=WindowFunction.AVG, column="x", order_by="d", preceding=400)

    # Threshold out of bounds (< 1.0)
    with pytest.raises(ValidationError):
        OutlierSpec(column="x", threshold=0.5)

    # Quantiles out of bounds (< 2)
    with pytest.raises(ValidationError):
        QuantileSpec(column="x", num_quantiles=1)

    # Cumulative SUM missing target column
    with pytest.raises(ValidationError):
        CumulativeSpec(function=CumulativeFunction.SUM, column=None, order_by="d")

    # Extra fields strictly forbidden
    with pytest.raises(ValidationError):
        OutlierSpec(column="x", threshold=3.0, injected_field="malicious")
