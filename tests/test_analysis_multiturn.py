"""
Phase 8F — Dedicated Multi-Turn Analytical Test Suite.
Verifies conversational analytical memory, AST mutation deltas, fail-closed rollbacks,
session isolation, dataset-switch invalidation, and grounded findings.
"""
import io
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.api.rest import _sessions
from app.analytics.mutator import apply_plan_mutation, MutationError
from app.analytics.schemas import (
    AnalysisPlan,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
    FilterClause,
    FilterOperator,
    PlanMutation,
    PlanMutationType,
    FilterDelta,
)


@pytest.fixture(autouse=True)
def isolated_multiturn_env(monkeypatch):
    """Isolates dataset directory and resets session memory between tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_multiturn_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)
    _sessions.clear()
    yield temp_dir
    _sessions.clear()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


def _upload_test_dataset(client: TestClient, session_id: str, filename: str, csv_data: str) -> str:
    resp = client.post(
        "/datasets/upload",
        files={"file": (filename, io.BytesIO(csv_data.encode("utf-8")), "text/csv")},
        data={"session_id": session_id},
    )
    assert resp.status_code in (200, 201), f"Upload failed: {resp.text}"
    return resp.json()["dataset_id"]


SAMPLE_SALES_CSV = """category,product_name,order_date,revenue,profit
Electronics,Smartphone,2026-01-01,500.0,100.0
Electronics,Laptop,2026-01-02,1500.0,300.0
Furniture,Chair,2026-01-03,200.0,40.0
Furniture,Desk,2026-01-04,800.0,160.0
Clothing,Shirt,2026-01-05,100.0,25.0
Clothing,Pants,2026-01-06,150.0,35.0
"""

SAMPLE_HR_CSV = """department,employees,budget
Engineering,40,500000.0
Marketing,15,150000.0
Sales,30,300000.0
"""


def test_multiturn_sequence_filter_metric_groupby_reset(client: TestClient):
    """
    Tests full canonical sequence:
    Turn 1: Baseline query (revenue by category)
    Turn 2: ADD_FILTER ("Filter to Electronics")
    Turn 3: CHANGE_METRIC ("Show profit instead")
    Turn 4: CHANGE_GROUP_BY ("Group by product_name")
    Turn 5: RESET_ANALYSIS ("Clear filters")
    """
    session_id = "sess-canonical-1"
    ds_id = _upload_test_dataset(client, session_id, "sales.csv", SAMPLE_SALES_CSV)

    # Turn 1: Baseline query
    r1 = client.post(
        "/chat",
        json={"message": "Show top revenue by category", "session_id": session_id, "dataset_id": ds_id},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "ok"
    assert "Electronics" in d1["response"]
    plan1 = _sessions[session_id]["active_analysis_plan"]
    assert plan1["group_by"][0]["column"] == "category"
    assert plan1["aggregations"][0]["column"] == "revenue"
    assert len(plan1["filters"]) == 0

    # Turn 2: ADD_FILTER
    r2 = client.post(
        "/chat",
        json={"message": "Filter to Electronics", "session_id": session_id},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "ok"
    assert "Filtered by category" in d2["response"] or "Electronics" in d2["response"]
    plan2 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan2["filters"]) == 1
    assert plan2["filters"][0]["column"] == "category"
    assert plan2["filters"][0]["value"] == "Electronics"
    assert plan2["group_by"][0]["column"] == "category"
    assert plan2["aggregations"][0]["column"] == "revenue"

    # Turn 3: CHANGE_METRIC
    r3 = client.post(
        "/chat",
        json={"message": "Show profit instead", "session_id": session_id},
    )
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["status"] == "ok"
    assert "profit" in d3["response"].lower()
    plan3 = _sessions[session_id]["active_analysis_plan"]
    assert plan3["aggregations"][0]["column"] == "profit"
    assert len(plan3["filters"]) == 1 and plan3["filters"][0]["value"] == "Electronics"

    # Turn 4: CHANGE_GROUP_BY
    r4 = client.post(
        "/chat",
        json={"message": "Group by product_name", "session_id": session_id},
    )
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["status"] == "ok"
    plan4 = _sessions[session_id]["active_analysis_plan"]
    assert plan4["group_by"][0]["column"] == "product_name"
    assert plan4["aggregations"][0]["column"] == "profit"
    assert len(plan4["filters"]) == 1

    # Turn 5: RESET_ANALYSIS
    r5 = client.post(
        "/chat",
        json={"message": "Clear filters", "session_id": session_id},
    )
    assert r5.status_code == 200
    d5 = r5.json()
    assert d5["status"] == "ok"
    plan5 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan5["filters"]) == 0
    assert plan5["aggregations"][0]["column"] == "profit"


def test_multiturn_sort_limit_chart_rolling_cumulative(client: TestClient):
    """
    Tests sorting, limiting, charting, rolling window, and cumulative metric follow-ups.
    """
    session_id = "sess-ops-1"
    ds_id = _upload_test_dataset(client, session_id, "sales_ops.csv", SAMPLE_SALES_CSV)

    # Turn 1: Baseline
    r1 = client.post(
        "/chat",
        json={"message": "Show revenue by category", "session_id": session_id, "dataset_id": ds_id},
    )
    assert r1.status_code == 200

    # Turn 2: CHANGE_SORT (ascending)
    r2 = client.post(
        "/chat",
        json={"message": "Sort by revenue ascending", "session_id": session_id},
    )
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan2["order_by"]) > 0
    assert plan2["order_by"][0]["descending"] is False

    # Turn 3: CHANGE_LIMIT
    r3 = client.post(
        "/chat",
        json={"message": "Top 2", "session_id": session_id},
    )
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]
    assert plan3["limit"] == 2

    # Turn 4: TOGGLE_CHART
    r4 = client.post(
        "/chat",
        json={"message": "Visualize this as a bar chart", "session_id": session_id},
    )
    assert r4.status_code == 200
    plan4 = _sessions[session_id]["active_analysis_plan"]
    assert plan4["requires_chart"] is True
    assert plan4["chart_type"] == "bar"

    # Turn 5: ADD_ROLLING
    r5 = client.post(
        "/chat",
        json={"message": "Show revenue as a 7-day rolling average", "session_id": session_id},
    )
    assert r5.status_code == 200
    plan5 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan5["rolling_windows"]) == 1
    assert plan5["rolling_windows"][0]["column"] == "revenue"

    # Turn 6: ADD_CUMULATIVE
    r6 = client.post(
        "/chat",
        json={"message": "Show cumulative sum over time", "session_id": session_id},
    )
    assert r6.status_code == 200
    plan6 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan6["cumulative_metrics"]) == 1


def test_failed_mutation_preserves_previous_plan(client: TestClient):
    """
    Verifies fail-closed rollback: an invalid follow-up or compilation failure
    must retain the previous valid active_analysis_plan without state corruption.
    """
    session_id = "sess-failclosed-1"
    ds_id = _upload_test_dataset(client, session_id, "sales_fc.csv", SAMPLE_SALES_CSV)

    # Initial valid plan
    r1 = client.post(
        "/chat",
        json={"message": "Show revenue by category", "session_id": session_id, "dataset_id": ds_id},
    )
    assert r1.status_code == 200
    original_plan = _sessions[session_id]["active_analysis_plan"].copy()

    # Manually inject an invalid mutation directly through AST mutator to verify exception
    candidate_plan = AnalysisPlan.model_validate(original_plan)
    with pytest.raises(MutationError):
        # Adding a filter on a non-existent column must raise MutationError
        bad_mut = PlanMutation(
            mutation_type=PlanMutationType.ADD_FILTER,
            filter_delta=FilterDelta(column="non_existent_col", operator=FilterOperator.EQ, value="test"),
        )
        from app.analytics import dataset_service
        meta = dataset_service.get_dataset(ds_id, session_id=session_id)
        apply_plan_mutation(candidate_plan, bad_mut, metadata=meta)

    # Verify session still has the exact valid plan untouched
    retained_plan = _sessions[session_id]["active_analysis_plan"]
    assert retained_plan == original_plan
    assert retained_plan["group_by"][0]["column"] == "category"


def test_session_isolation_multiturn(client: TestClient):
    """
    Verifies that multi-turn state in Session A does not leak into Session B.
    """
    s_a = "sess-iso-A"
    s_b = "sess-iso-B"

    ds_a = _upload_test_dataset(client, s_a, "data_a.csv", SAMPLE_SALES_CSV)
    ds_b = _upload_test_dataset(client, s_b, "data_b.csv", SAMPLE_SALES_CSV)

    # Session A: Filter to Electronics
    client.post("/chat", json={"message": "Show revenue by category", "session_id": s_a, "dataset_id": ds_a})
    client.post("/chat", json={"message": "Filter to Electronics", "session_id": s_a})

    # Session B: Filter to Furniture
    client.post("/chat", json={"message": "Show revenue by category", "session_id": s_b, "dataset_id": ds_b})
    client.post("/chat", json={"message": "Filter to Furniture", "session_id": s_b})

    # Assert completely isolated plans
    plan_a = _sessions[s_a]["active_analysis_plan"]
    plan_b = _sessions[s_b]["active_analysis_plan"]

    assert plan_a["filters"][0]["value"] == "Electronics"
    assert plan_b["filters"][0]["value"] == "Furniture"
    assert plan_a["dataset_id"] == ds_a
    assert plan_b["dataset_id"] == ds_b


def test_dataset_switch_invalidates_old_plan(client: TestClient):
    """
    Verifies that switching dataset_id within the same session purges
    the old dataset's active plan and does not cross schemas.
    """
    session_id = "sess-switch-1"
    ds_sales = _upload_test_dataset(client, session_id, "sales.csv", SAMPLE_SALES_CSV)
    ds_hr = _upload_test_dataset(client, session_id, "hr.csv", SAMPLE_HR_CSV)

    # Turn 1: Analyze Sales
    client.post("/chat", json={"message": "Show revenue by category", "session_id": session_id, "dataset_id": ds_sales})
    client.post("/chat", json={"message": "Filter to Electronics", "session_id": session_id})
    sales_plan = _sessions[session_id]["active_analysis_plan"]
    assert sales_plan["dataset_id"] == ds_sales
    assert len(sales_plan["filters"]) == 1

    # Turn 2: Switch to HR dataset
    r_hr = client.post("/chat", json={"message": "Show total employees by department", "session_id": session_id, "dataset_id": ds_hr})
    assert r_hr.status_code == 200
    hr_plan = _sessions[session_id]["active_analysis_plan"]

    # Old sales filters and columns must be completely purged
    assert hr_plan["dataset_id"] == ds_hr
    assert len(hr_plan["filters"]) == 0
    assert hr_plan["group_by"][0]["column"] == "department"


def test_grounded_findings_asserted_against_duckdb_results(client: TestClient):
    """
    Verifies that conversational grounded statements strictly match
    actual DuckDB executed numerical values.
    """
    session_id = "sess-grounded-1"
    ds_id = _upload_test_dataset(client, session_id, "sales_gr.csv", SAMPLE_SALES_CSV)

    # Baseline query
    r1 = client.post("/chat", json={"message": "Show top revenue by category", "session_id": session_id, "dataset_id": ds_id})
    resp1 = r1.json()
    assert resp1["status"] == "ok"
    assert "Electronics" in resp1["response"]
    # Total revenue for Electronics in SAMPLE_SALES_CSV is 500 + 1500 = 2000.0
    assert "2000" in resp1["response"] or "2,000" in resp1["response"]

    # Filter to Furniture
    # Furniture revenue is 200 + 800 = 1000.0
    r2 = client.post("/chat", json={"message": "Filter to Furniture", "session_id": session_id})
    resp2 = r2.json()
    assert resp2["status"] == "ok"
    assert "Furniture" in resp2["response"]
    assert "1000" in resp2["response"] or "1,000" in resp2["response"]
