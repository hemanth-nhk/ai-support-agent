"""
Phase 9 — Dedicated Cross-Tab / Pivot Analysis & Heatmap Visualization Test Suite.
Tests:
- 9A: PivotSpec & PivotResult Pydantic schema validation, bounds, forbidden extras
- 9B: Deterministic DuckDB pivot compiler, identifier escaping, safe literal filters, SQL injection defense
- 9C: Matrix executor, normalization, null-handling, empty results, row/column/grand totals
- 9D: Heatmap visualization, PNG artifact verification, dimension clamping, compatibility validation
- 9E: Multi-turn conversational memory, TOGGLE_HEATMAP, CHANGE_PIVOT_COLUMNS, rollback, session isolation, grounded findings
"""
import io
import shutil
import tempfile
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.api.rest import _sessions
from app.analytics.schemas import (
    AnalysisPlan,
    AnalysisResult,
    AggregationFunction,
    FilterClause,
    FilterOperator,
    PivotSpec,
    PivotResult,
    PivotDelta,
    PlanMutation,
    PlanMutationType,
)
from app.analytics.compiler import (
    compile_plan,
    PlanValidationError,
    ColumnNotFoundError,
    TypeMismatchError,
)
from app.analytics.executor import execute_analysis_plan, execute_compiled_query
from app.analytics.visualizer import (
    validate_chart_compatibility,
    render_chart,
    ChartRenderError,
    PNG_MAGIC_BYTES,
)
from app.analytics.mutator import apply_plan_mutation, MutationError
from app.analytics.planner import format_deterministic_result
from app.analytics.dataset_service import ingest_dataset, get_dataset


SAMPLE_PIVOT_CSV = """region,category,sales,quantity,order_date
North,Electronics,1200.50,5,2026-01-10
North,Furniture,450.00,2,2026-01-11
South,Electronics,800.00,3,2026-01-12
South,Furniture,600.00,4,2026-01-13
East,Electronics,950.00,4,2026-01-14
East,Furniture,300.00,1,2026-01-15
West,Electronics,1100.00,6,2026-01-16
West,Office Supplies,250.00,10,2026-01-17
"""


@pytest.fixture(autouse=True)
def isolated_pivot_env(monkeypatch):
    """Isolates dataset and charts directories and resets session memory between tests."""
    temp_datasets = tempfile.mkdtemp(prefix="test_pivot_ds_")
    temp_charts = tempfile.mkdtemp(prefix="test_pivot_ch_")
    monkeypatch.setattr(settings, "datasets_dir", temp_datasets)
    monkeypatch.setattr(settings, "charts_dir", temp_charts)
    _sessions.clear()
    yield temp_datasets
    _sessions.clear()
    shutil.rmtree(temp_datasets, ignore_errors=True)
    shutil.rmtree(temp_charts, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


def _upload_dataset(session_id: str, csv_content: str = SAMPLE_PIVOT_CSV) -> str:
    res = ingest_dataset(
        session_id=session_id,
        filename="sales_data.csv",
        content_bytes=csv_content.encode("utf-8"),
    )
    return res.dataset_id


# ===========================================================================
# 9A: AST & Schema Validation
# ===========================================================================

def test_pivot_spec_valid():
    spec = PivotSpec(
        row_dimension="region",
        column_dimension="category",
        metric="sales",
        aggregation=AggregationFunction.SUM,
        max_rows=25,
        max_columns=10,
    )
    assert spec.row_dimension == "region"
    assert spec.column_dimension == "category"
    assert spec.metric == "sales"
    assert spec.aggregation == AggregationFunction.SUM
    assert spec.max_rows == 25
    assert spec.max_columns == 10
    assert spec.include_totals is True


def test_pivot_spec_rejects_identical_dimensions():
    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="region",
            column_dimension="region",
            metric="sales",
        )

    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="Category",
            column_dimension="category",
            metric="sales",
        )


def test_pivot_spec_rejects_out_of_bounds_cardinality():
    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            max_rows=0,  # ge=1
        )
    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            max_rows=51,  # le=50
        )
    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            max_columns=35,  # le=30
        )


def test_pivot_spec_forbids_extra_fields():
    with pytest.raises(ValidationError):
        PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            arbitrary_field="malicious",  # type: ignore[call-arg]
        )


# ===========================================================================
# 9B: Deterministic DuckDB Pivot Compiler
# ===========================================================================

def test_compile_pivot_valid():
    session_id = "sess_comp_1"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
            max_rows=20,
            max_columns=15,
        ),
    )
    compiled = compile_plan(plan, metadata)
    assert 'PIVOT (' in compiled.sql
    assert 'ON "category"' in compiled.sql
    assert 'USING SUM("sales")' in compiled.sql
    assert 'GROUP BY "region"' in compiled.sql
    assert 'ORDER BY "region" ASC' in compiled.sql
    assert 'LIMIT 20' in compiled.sql
    assert compiled.parameters == []


def test_compile_pivot_missing_column():
    session_id = "sess_comp_2"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="nonexistent_dim",
            column_dimension="category",
            metric="sales",
        ),
    )
    with pytest.raises(ColumnNotFoundError):
        compile_plan(plan, metadata)


def test_compile_pivot_non_numeric_metric_for_sum():
    session_id = "sess_comp_3"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="category",  # string column
            aggregation=AggregationFunction.SUM,
        ),
    )
    with pytest.raises(TypeMismatchError):
        compile_plan(plan, metadata)


def test_compile_pivot_count_aggregation_accepts_string_metric():
    session_id = "sess_comp_4"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="category",
            aggregation=AggregationFunction.COUNT,
        ),
    )
    compiled = compile_plan(plan, metadata)
    assert 'USING COUNT("category")' in compiled.sql


def test_compile_pivot_with_safe_filters_and_injection_defense():
    session_id = "sess_comp_5"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    malicious_input = "Electronics'; DROP TABLE source_table; --"
    plan = AnalysisPlan(
        dataset_id=dataset_id,
        filters=[
            FilterClause(column="category", operator=FilterOperator.EQ, value=malicious_input),
            FilterClause(column="sales", operator=FilterOperator.GTE, value=500.0),
        ],
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
        ),
    )
    compiled = compile_plan(plan, metadata)
    # Quotes are properly escaped to prevent breakout
    assert "''" in compiled.sql
    assert "DROP TABLE" in compiled.sql  # inside the escaped literal string
    assert "WHERE" in compiled.sql


# ===========================================================================
# 9C: Matrix Result & Executor
# ===========================================================================

def test_execute_pivot_matrix():
    session_id = "sess_exec_1"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
            include_totals=True,
        ),
    )
    result = execute_analysis_plan(plan, session_id=session_id)
    assert result.pivot_table is not None

    pt = result.pivot_table
    assert pt.row_dimension == "region"
    assert pt.column_dimension == "category"
    assert pt.metric == "sales"
    assert pt.aggregation == "sum"

    assert sorted(pt.rows) == ["East", "North", "South", "West"]
    assert "Electronics" in pt.columns
    assert "Furniture" in pt.columns
    assert "Office Supplies" in pt.columns

    # Check 2D values shape
    assert len(pt.values) == len(pt.rows)
    for r in pt.values:
        assert len(r) == len(pt.columns)

    # Check totals
    assert pt.row_totals is not None
    assert len(pt.row_totals) == len(pt.rows)
    assert pt.column_totals is not None
    assert len(pt.column_totals) == len(pt.columns)
    assert pt.grand_total is not None
    assert pt.grand_total > 0


def test_execute_pivot_null_cells():
    session_id = "sess_exec_2"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
        ),
    )
    result = execute_analysis_plan(plan, session_id=session_id)
    pt = result.pivot_table
    assert pt is not None

    # North has no 'Office Supplies'
    north_idx = pt.rows.index("North")
    office_idx = pt.columns.index("Office Supplies")
    assert pt.values[north_idx][office_idx] is None


def test_execute_pivot_empty_results():
    session_id = "sess_exec_3"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        filters=[
            FilterClause(column="region", operator=FilterOperator.EQ, value="NonExistentRegion")
        ],
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
            include_totals=True,
        ),
    )
    result = execute_analysis_plan(plan, session_id=session_id)
    pt = result.pivot_table
    assert pt is not None
    assert pt.rows == []
    assert pt.values == []


# ===========================================================================
# 9D: Bounded Heatmap Visualization
# ===========================================================================

def test_validate_heatmap_compatibility():
    session_id = "sess_vis_1"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
        ),
    )
    result = execute_analysis_plan(plan, session_id=session_id)

    is_compat, eff_type, err = validate_chart_compatibility(result, "heatmap", is_explicit=True)
    assert is_compat is True
    assert eff_type == "heatmap"
    assert err is None


def test_render_heatmap_artifact():
    session_id = "sess_vis_2"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
        ),
        requires_chart=True,
        chart_type="heatmap",
    )
    result = execute_analysis_plan(plan, session_id=session_id)
    artifact = render_chart(result, "heatmap", session_id=session_id)

    assert artifact.chart_type == "heatmap"
    assert artifact.filename.endswith(".png")

    # Verify PNG bytes from stored chart file
    from app.analytics.visualizer import resolve_chart_file
    target_file = resolve_chart_file(session_id, artifact.chart_id)
    assert target_file.is_file()
    with open(target_file, "rb") as f:
        data = f.read()
    assert data.startswith(PNG_MAGIC_BYTES)
    assert len(data) > 1000


def test_heatmap_incompatible_without_pivot():
    plan = AnalysisPlan(
        dataset_id="mock_id",
        aggregations=[],
    )
    mock_result = AnalysisResult(
        plan=plan,
        compiled_sql="SELECT 1",
        rows=[{"col1": "A", "col2": 10}],
        row_count=1,
        columns=["col1", "col2"],
        execution_time_ms=1.0,
        pivot_table=None,
    )
    is_compat, eff_type, err = validate_chart_compatibility(mock_result, "heatmap", is_explicit=True)
    assert is_compat is False
    assert eff_type is None
    assert "requires a valid 2D cross-tab pivot result" in err


# ===========================================================================
# 9E: Mutations & Conversational Integration
# ===========================================================================

def test_toggle_heatmap_mutation():
    session_id = "sess_mut_1"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    initial_plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
        ),
        requires_chart=False,
    )

    mutation = PlanMutation(
        mutation_type=PlanMutationType.TOGGLE_HEATMAP,
        chart_type="heatmap",
        requires_chart=True,
    )
    mutated = apply_plan_mutation(initial_plan, mutation, metadata)
    assert mutated.requires_chart is True
    assert mutated.chart_type == "heatmap"


def test_change_pivot_columns_mutation():
    session_id = "sess_mut_2"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    initial_plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
        ),
    )

    mutation = PlanMutation(
        mutation_type=PlanMutationType.CHANGE_PIVOT_COLUMNS,
        pivot_delta=PivotDelta(
            row_dimension="category",
            column_dimension="region",
            metric="quantity",
        ),
    )
    mutated = apply_plan_mutation(initial_plan, mutation, metadata)
    assert mutated.pivot is not None
    assert mutated.pivot.row_dimension == "category"
    assert mutated.pivot.column_dimension == "region"
    assert mutated.pivot.metric == "quantity"


def test_mutation_rollback_on_invalid_column():
    session_id = "sess_mut_3"
    dataset_id = _upload_dataset(session_id)
    metadata = get_dataset(dataset_id, session_id=session_id)

    initial_plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
        ),
    )

    mutation = PlanMutation(
        mutation_type=PlanMutationType.CHANGE_PIVOT_COLUMNS,
        pivot_delta=PivotDelta(
            row_dimension="nonexistent_col",
            column_dimension="category",
            metric="sales",
        ),
    )
    with pytest.raises(MutationError):
        apply_plan_mutation(initial_plan, mutation, metadata)

    # Initial plan unchanged
    assert initial_plan.pivot.row_dimension == "region"


def test_grounded_matrix_findings():
    session_id = "sess_find_1"
    dataset_id = _upload_dataset(session_id)

    plan = AnalysisPlan(
        dataset_id=dataset_id,
        pivot=PivotSpec(
            row_dimension="region",
            column_dimension="category",
            metric="sales",
            aggregation=AggregationFunction.SUM,
            include_totals=True,
        ),
    )
    result = execute_analysis_plan(plan, session_id=session_id)
    findings = format_deterministic_result(result, question="cross tab sales by region and category")

    assert "Cross-tabulation of sum sales" in findings
    assert "region values" in findings
    assert "category values" in findings
    assert "Leading cell observed in" in findings
    assert "grand total" in findings


# ===========================================================================
# Multi-Turn Conversational API Sequence
# ===========================================================================

def test_multiturn_pivot_filter_metric_heatmap_flow(client):
    session_id = "sess_api_flow_1"
    res = client.post(
        "/datasets/upload",
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_PIVOT_CSV.encode("utf-8")), "text/csv")},
        data={"session_id": session_id},
    )
    assert res.status_code in (200, 201)
    dataset_id = res.json()["dataset_id"]

    # Turn 1: Initial Cross-Tab Pivot
    t1 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Show cross-tab of sales by region and category",
        },
    )
    assert t1.status_code == 200
    data1 = t1.json()
    assert data1["status"] == "ok"
    assert data1["pivot"] is not None
    assert data1["pivot"]["row_dimension"] == "region"
    assert data1["pivot"]["column_dimension"] == "category"

    # Turn 2: Filter to North and South
    t2 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Filter to region == 'North'",
        },
    )
    assert t2.status_code == 200
    data2 = t2.json()
    assert data2["status"] == "ok"
    assert data2["pivot"] is not None
    assert data2["pivot"]["rows"] == ["North"]

    # Turn 3: Switch Metric to quantity
    t3 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Now show quantity instead",
        },
    )
    assert t3.status_code == 200
    data3 = t3.json()
    assert data3["status"] == "ok"
    assert data3["pivot"] is not None
    assert data3["pivot"]["metric"] == "quantity"

    # Turn 4: Toggle Heatmap Visualization
    t4 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Make it a heatmap",
        },
    )
    assert t4.status_code == 200
    data4 = t4.json()
    assert data4["status"] == "ok"
    assert data4["chart"] is not None
    assert data4["chart"]["chart_type"] == "heatmap"


def test_session_isolation_and_dataset_switch(client):
    sess_a = "sess_iso_a"
    sess_b = "sess_iso_b"

    res_a = client.post(
        "/datasets/upload",
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_PIVOT_CSV.encode("utf-8")), "text/csv")},
        data={"session_id": sess_a},
    )
    ds_a = res_a.json()["dataset_id"]

    # Session A starts pivot
    r1 = client.post(
        "/chat",
        json={"session_id": sess_a, "dataset_id": ds_a, "message": "cross-tab sales by region and category"},
    )
    assert r1.status_code == 200
    assert r1.json()["pivot"] is not None

    # Session B has separate state
    r2 = client.post(
        "/chat",
        json={"session_id": sess_b, "message": "Hello there"},
    )
    assert r2.status_code == 200
    assert r2.json().get("pivot") is None

    # Switching dataset in Session A resets plan
    other_csv = "id,value\n1,10\n2,20\n"
    res_a2 = client.post(
        "/datasets/upload",
        files={"file": ("other.csv", io.BytesIO(other_csv.encode("utf-8")), "text/csv")},
        data={"session_id": sess_a},
    )
    ds_a2 = res_a2.json()["dataset_id"]

    r3 = client.post(
        "/chat",
        json={"session_id": sess_a, "dataset_id": ds_a2, "message": "show total value"},
    )
    assert r3.status_code == 200
    assert r3.json().get("pivot") is None
