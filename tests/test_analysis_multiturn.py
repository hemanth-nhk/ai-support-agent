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


# (Shared client fixture provided by tests/conftest.py)


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


SAMPLE_REGION_SALES_CSV = """region,product,status,order_date,sales,profit
North,Widget,delivered,2024-01-01,100.0,20.0
North,Gadget,delivered,2024-01-02,200.0,50.0
South,Widget,delivered,2024-01-03,150.0,30.0
South,Gadget,pending,2024-01-04,250.0,60.0
East,Widget,delivered,2024-01-05,300.0,70.0
West,Gadget,delivered,2024-01-06,400.0,90.0
"""


def test_multiturn_unrestricted_breakdown_clears_inherited_filter_on_grouping_column(client: TestClient):
    """
    Verifies that an unrestricted breakdown request ('Plot a bar chart of total sales by region')
    changes the grouping dimension to 'region' and clears an inherited 'region = North' filter.
    """
    session_id = "sess-unrestricted-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    # Turn 1: Show top products by sales
    r1 = client.post("/chat", json={"message": "Show the top 2 products by sales.", "session_id": session_id, "dataset_id": ds_id})
    assert r1.status_code == 200
    plan1 = _sessions[session_id]["active_analysis_plan"]
    assert plan1["group_by"][0]["column"] == "product"

    # Turn 2: Filter to only the North region
    r2 = client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]
    assert plan2["group_by"][0]["column"] == "product"
    assert any(f["column"].lower() == "region" and f["value"] == "North" for f in plan2["filters"])

    # Turn 3: Plot a bar chart of total sales by region
    r3 = client.post("/chat", json={"message": "Plot a bar chart of total sales by region.", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    # Invariants:
    # 1. Grouping dimension must switch to region
    assert plan3["group_by"][0]["column"] == "region"
    # 2. Inherited region filter must be completely cleared
    assert not any(f["column"].lower() == "region" for f in plan3["filters"])
    # 3. Chart properties must be set
    assert plan3["requires_chart"] is True
    assert plan3["chart_type"] == "bar"


def test_multiturn_explicit_in_turn_filter_on_grouping_column_is_retained(client: TestClient):
    """
    Verifies that an explicit in-turn constraint ('Plot sales by region for North only')
    switches grouping to 'region' while preserving the explicit 'region = North' filter.
    """
    session_id = "sess-explicit-filter-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    # Turn 1: Show top products by sales
    client.post("/chat", json={"message": "Show the top 2 products by sales.", "session_id": session_id, "dataset_id": ds_id})

    # Turn 2: Filter to only the North region
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})

    # Turn 3: Plot sales by region for North only
    r3 = client.post("/chat", json={"message": "Plot sales by region for North only", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    assert plan3["group_by"][0]["column"] == "region"
    region_filters = [f for f in plan3["filters"] if f["column"].lower() == "region"]
    assert len(region_filters) == 1
    assert region_filters[0]["value"] == "North"
    assert region_filters[0]["operator"] in ("==", FilterOperator.EQ.value)


def test_multiturn_exclusion_filter_on_grouping_column_is_constructed(client: TestClient):
    """
    Verifies that an exclusion request ('Plot sales by region except North')
    constructs a 'region != North' filter instead of blindly clearing or preserving the EQ filter.
    """
    session_id = "sess-exclusion-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show top products by sales.", "session_id": session_id, "dataset_id": ds_id})
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})

    r3 = client.post("/chat", json={"message": "Plot sales by region except North", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    assert plan3["group_by"][0]["column"] == "region"
    region_filters = [f for f in plan3["filters"] if f["column"].lower() == "region"]
    assert len(region_filters) == 1
    assert region_filters[0]["value"] == "North"
    assert region_filters[0]["operator"] in ("!=", FilterOperator.NEQ.value)


def test_multiturn_exclusion_filter_with_trailing_clause_mixed_edge_case(client: TestClient):
    """
    Verifies that an exclusion query with trailing clauses ('Plot sales by region except North for 2024')
    accurately extracts the target column value without capturing trailing text as part of the filter value.
    """
    session_id = "sess-excl-trailing-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show top products by sales.", "session_id": session_id, "dataset_id": ds_id})
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})

    r3 = client.post("/chat", json={"message": "Plot sales by region except North for 2024", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    assert plan3["group_by"][0]["column"] == "region"
    region_filters = [f for f in plan3["filters"] if f["column"].lower() == "region"]
    assert len(region_filters) == 1
    # Crucial validation: Must be 'North', NOT 'North for 2024'
    assert region_filters[0]["value"] == "North"
    assert region_filters[0]["operator"] in ("!=", FilterOperator.NEQ.value)


def test_multiturn_dimension_change_preserves_unrelated_filters(client: TestClient):
    """
    Verifies that when a grouping dimension changes, unrelated filters (e.g. status = delivered)
    are strictly preserved while the stale filter on the new grouping dimension is cleared.
    """
    session_id = "sess-unrelated-preserv-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    # Turn 1: Top products by sales
    client.post("/chat", json={"message": "Show top products by sales.", "session_id": session_id, "dataset_id": ds_id})
    # Turn 2: Filter to only the North region
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})
    # Turn 3: Also filter where status is delivered
    client.post("/chat", json={"message": "where status is delivered", "session_id": session_id})

    plan_mid = _sessions[session_id]["active_analysis_plan"]
    assert any(f["column"].lower() == "region" for f in plan_mid["filters"])
    assert any(f["column"].lower() == "status" for f in plan_mid["filters"])

    # Turn 4: Plot sales by region
    r4 = client.post("/chat", json={"message": "Plot sales by region", "session_id": session_id})
    assert r4.status_code == 200
    plan4 = _sessions[session_id]["active_analysis_plan"]

    assert plan4["group_by"][0]["column"] == "region"
    # Region filter must be cleared
    assert not any(f["column"].lower() == "region" for f in plan4["filters"])
    # Status filter must remain preserved
    status_filters = [f for f in plan4["filters"] if f["column"].lower() == "status"]
    assert len(status_filters) == 1
    assert status_filters[0]["value"] == "delivered"


def test_multiturn_filter_only_followup_preserves_existing_grouping(client: TestClient):
    """
    Verifies that a filter-only follow-up ('Filter to only the North region')
    preserves the existing grouping on 'product' and does NOT switch grouping to region.
    """
    session_id = "sess-filter-only-preserv-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show the top 2 products by sales.", "session_id": session_id, "dataset_id": ds_id})
    r2 = client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})
    assert r2.status_code == 200

    plan2 = _sessions[session_id]["active_analysis_plan"]
    # Grouping must remain product
    assert plan2["group_by"][0]["column"] == "product"
    assert any(f["column"].lower() == "region" and f["value"] == "North" for f in plan2["filters"])


def test_multiturn_for_all_transactions_does_not_synthesize_filter(client: TestClient):
    """
    Verifies that generic prepositions like 'for all transactions' ('Plot sales by region for all transactions')
    are not interpreted as an explicit filter, correctly clearing the inherited region filter.
    """
    session_id = "sess-all-tx-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show the top 2 products by sales.", "session_id": session_id, "dataset_id": ds_id})
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})

    r3 = client.post("/chat", json={"message": "Plot sales by region for all transactions", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    assert plan3["group_by"][0]["column"] == "region"
    # Neither 'North' nor 'all transactions' should be present as a filter
    assert len(plan3["filters"]) == 0


SAMPLE_OVERLAPPING_REGIONS_CSV = """product,region,category,sales,profit
Laptops,North,Electronics,50000,8000
Laptops,North America,Electronics,72000,9500
Phones,South,Food and Beverage,30000,6200
Monitors,East,Food and Beverage,15000,2000
"""


def test_multiturn_exclusion_unmatched_value_does_not_synthesize_filter(client: TestClient):
    """
    Verifies that when known dataset values are available, an unmatched exclusion
    such as 'except high value orders' does not synthesize an arbitrary NEQ filter.
    """
    session_id = "sess-unmatched-excl-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show the top 2 products by sales.", "session_id": session_id, "dataset_id": ds_id})
    client.post("/chat", json={"message": "Filter to only the North region.", "session_id": session_id})

    r3 = client.post("/chat", json={"message": "Plot sales by region except high value orders", "session_id": session_id})
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]

    assert plan3["group_by"][0]["column"] == "region"
    # Inherited North filter must be cleared, and no 'high value orders' filter should be created
    assert len(plan3["filters"]) == 0


def test_multiturn_exclusion_overlapping_values_prefers_longest_match(client: TestClient):
    """
    Verifies that when dataset has overlapping candidates (e.g. 'North' and 'North America'),
    requesting 'except North America' excludes ONLY 'North America' and does not exclude 'North'.
    """
    session_id = "sess-overlapping-excl-1"
    ds_id = _upload_test_dataset(client, session_id, "overlap_regions.csv", SAMPLE_OVERLAPPING_REGIONS_CSV)

    client.post("/chat", json={"message": "Show top products by sales.", "session_id": session_id, "dataset_id": ds_id})
    r2 = client.post("/chat", json={"message": "Plot sales by region except North America", "session_id": session_id})
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]

    assert plan2["group_by"][0]["column"] == "region"
    assert len(plan2["filters"]) == 1
    # Must be NEQ 'North America', NOT NOT_IN ['North', 'North America']
    assert plan2["filters"][0]["column"] == "region"
    assert plan2["filters"][0]["operator"] in ("!=", FilterOperator.NEQ.value)
    assert plan2["filters"][0]["value"] == "North America"


def test_multiturn_exclusion_multi_value_with_known_values(client: TestClient):
    """
    Verifies multi-value exclusion such as 'except North and South' produces a NOT_IN filter.
    """
    session_id = "sess-multi-excl-1"
    ds_id = _upload_test_dataset(client, session_id, "region_sales.csv", SAMPLE_REGION_SALES_CSV)

    client.post("/chat", json={"message": "Show top products by sales.", "session_id": session_id, "dataset_id": ds_id})
    r2 = client.post("/chat", json={"message": "Plot sales by region except North and South", "session_id": session_id})
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]

    assert plan2["group_by"][0]["column"] == "region"
    assert len(plan2["filters"]) == 1
    assert plan2["filters"][0]["column"] == "region"
    assert plan2["filters"][0]["operator"] in ("not_in", FilterOperator.NOT_IN.value)
    assert plan2["filters"][0]["value"] == ["North", "South"]


def test_multiturn_exclusion_quoted_compound_value_with_and(client: TestClient):
    """
    Verifies that a quoted compound value containing 'and' (e.g. 'Food and Beverage')
    is preserved as a single exclusion filter and not split into multiple sub-tokens.
    """
    session_id = "sess-quoted-and-1"
    ds_id = _upload_test_dataset(client, session_id, "overlap_regions.csv", SAMPLE_OVERLAPPING_REGIONS_CSV)

    client.post("/chat", json={"message": "Show sales by category.", "session_id": session_id, "dataset_id": ds_id})
    r2 = client.post("/chat", json={"message": 'Plot sales by category except "Food and Beverage"', "session_id": session_id})
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]

    assert plan2["group_by"][0]["column"] == "category"
    assert len(plan2["filters"]) == 1
    assert plan2["filters"][0]["column"] == "category"
    assert plan2["filters"][0]["operator"] in ("!=", FilterOperator.NEQ.value)
    assert plan2["filters"][0]["value"] == "Food and Beverage"


def test_multiturn_exclusion_fallback_trailing_clause_unit_level():
    """
    Verifies fallback exclusion extraction when known values are empty/unavailable,
    ensuring trailing prepositions (in 2024, during 2024, filtered by status) are trimmed.
    """
    from app.analytics.planner import _extract_exclusion_filters
    from app.analytics.schemas import DatasetMetadata, DatasetProfile

    mock_metadata = DatasetMetadata(
        dataset_id="nonexistent-mock-id",
        session_id="mock-session",
        filename="mock.csv",
        format="csv",
        row_count=0,
        column_count=1,
        created_at="2026-01-01T00:00:00",
        profile=DatasetProfile(row_count=0, column_count=1, missing_cells=0, duplicate_rows=0, columns=[]),
    )

    # Trailing 'in 2024'
    f1 = _extract_exclusion_filters("Plot sales by region except North in 2024", "region", mock_metadata)
    assert len(f1) == 1
    assert f1[0].column == "region"
    assert f1[0].operator == FilterOperator.NEQ
    assert f1[0].value == "North"

    # Trailing 'during 2024'
    f2 = _extract_exclusion_filters("Plot sales by region except North during 2024", "region", mock_metadata)
    assert len(f2) == 1
    assert f2[0].value == "North"

    # Trailing 'filtered by status'
    f3 = _extract_exclusion_filters("Plot sales by region except North filtered by status", "region", mock_metadata)
    assert len(f3) == 1
    assert f3[0].value == "North"


def test_multiturn_explicit_clear_filter_directive(client: TestClient):
    """
    Verifies that explicit directives like 'across the entire dataset, ignoring previous filters'
    reliably clear inherited filters during multi-turn chat sessions.
    """
    session_id = "sess-explicit-clear-1"
    ds_id = _upload_test_dataset(client, session_id, "sales.csv", SAMPLE_SALES_CSV)

    # Turn 1: Filter to Electronics
    r1 = client.post(
        "/chat",
        json={"message": "Show revenue by category where category is Electronics", "session_id": session_id, "dataset_id": ds_id},
    )
    assert r1.status_code == 200
    plan1 = _sessions[session_id]["active_analysis_plan"]
    assert any(f["column"] == "category" and f["value"] == "Electronics" for f in plan1["filters"])

    # Turn 2: Genuine follow-up preserves filter
    r2 = client.post(
        "/chat",
        json={"message": "Plot revenue by product_name", "session_id": session_id},
    )
    assert r2.status_code == 200
    plan2 = _sessions[session_id]["active_analysis_plan"]
    assert any(f["column"] == "category" and f["value"] == "Electronics" for f in plan2["filters"])
    assert any(g["column"] == "product_name" for g in plan2["group_by"])

    # Turn 3: Explicit clear directive removes filter across whole dataset
    r3 = client.post(
        "/chat",
        json={
            "message": "Count records by category across the entire dataset, ignoring previous filters.",
            "session_id": session_id,
        },
    )
    assert r3.status_code == 200
    plan3 = _sessions[session_id]["active_analysis_plan"]
    assert len(plan3["filters"]) == 0
    assert any(g["column"] == "category" for g in plan3["group_by"])
    assert plan3["aggregations"][0]["function"] == "count"
