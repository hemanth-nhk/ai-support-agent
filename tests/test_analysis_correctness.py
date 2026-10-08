"""
Focused regression tests for UI-4 Backend Analytics Correctness:
1. Top-N and superlative natural language planning (top 5, top 10, bottom 5, invalid N, etc.)
2. Deterministic finding ranking agreement with result tables (never calling lowest 'leading')
3. Floating-point presentation normalization while preserving numeric float types
4. Result/Finding consistency invariant
5. End-to-end chat integration for "Show me the top 5 products by sales"
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analytics.schemas import (
    AnalysisPlan,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
    OrderByClause,
    DatasetMetadata,
)
from app.analytics.planner import (
    generate_analysis_plan,
    format_deterministic_result,
    verify_finding_result_consistency,
)
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query, _sanitize_record_value
from app.analytics import dataset_service


@pytest.fixture
def sales_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    csv_data = (
        "Product,City,Sales,Profit\n"
        "Widget A,Jaipur,52357.08,5000.0\n"
        "Widget B,Bengaluru,55468.29,6000.0\n"
        "Widget C,Delhi,64116.26,7500.0\n"
        "Widget D,Mumbai,72469.45,9000.0\n"
        "Widget E,Pune,3908.8700000000003,450.0\n"
        "Widget F,Chennai,15000.0,2000.0\n"
        "Widget G,Kolkata,28000.0,3200.0\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset(
        session_id="correctness-session",
        filename="sales_data.csv",
        content_bytes=csv_data,
    )
    return meta, "correctness-session"


# ===========================================================================
# 1. TOP-N PLANNING TESTS
# ===========================================================================

def test_top_5_products_by_sales_plan(sales_dataset):
    meta, _ = sales_dataset
    query = "Show me the top 5 products by sales"
    plan = generate_analysis_plan(query, meta)

    assert plan.group_by[0].column == "Product"
    assert len(plan.aggregations) == 1
    assert plan.aggregations[0].column == "Sales"
    assert plan.aggregations[0].function == AggregationFunction.SUM
    assert len(plan.order_by) == 1
    assert plan.order_by[0].target in ("Sales", "sum_Sales")
    assert plan.order_by[0].descending is True
    assert plan.limit == 5


def test_top_10_products_by_sales_plan(sales_dataset):
    meta, _ = sales_dataset
    query = "top 10 products by sales"
    plan = generate_analysis_plan(query, meta)

    assert plan.group_by[0].column == "Product"
    assert plan.order_by[0].descending is True
    assert plan.limit == 10


def test_bottom_5_products_by_sales_plan(sales_dataset):
    meta, _ = sales_dataset
    query = "bottom 5 products by sales"
    plan = generate_analysis_plan(query, meta)

    assert plan.group_by[0].column == "Product"
    assert plan.order_by[0].descending is False
    assert plan.limit == 5


def test_superlatives_without_explicit_n(sales_dataset):
    meta, _ = sales_dataset

    # highest selling products
    plan_high = generate_analysis_plan("highest selling products", meta)
    assert plan_high.group_by[0].column == "Product"
    assert plan_high.order_by[0].descending is True

    # lowest selling products
    plan_low = generate_analysis_plan("lowest selling products", meta)
    assert plan_low.group_by[0].column == "Product"
    assert plan_low.order_by[0].descending is False


def test_invalid_n_fails_closed(sales_dataset):
    meta, _ = sales_dataset
    # Limit 0 or 1000 violates AnalysisPlan limit bounds [1, 500]
    with pytest.raises(Exception):
        generate_analysis_plan("top 0 products by sales", meta)

    with pytest.raises(Exception):
        generate_analysis_plan("top 1000 products by sales", meta)


def test_normal_groupby_without_top_n(sales_dataset):
    meta, _ = sales_dataset
    query = "What is the total sales by city?"
    plan = generate_analysis_plan(query, meta)

    assert plan.group_by[0].column == "City"
    assert plan.aggregations[0].function == AggregationFunction.SUM
    # Normal query should not force a limit of 5
    assert plan.limit >= 10


# ===========================================================================
# 2. DETERMINISTIC FINDING RANKING TESTS
# ===========================================================================

def test_finding_ranking_agrees_with_table_ascending(sales_dataset):
    """
    Query: 'What is the total sales by city?'
    Result table ascending:
      Jaipur       52357.08
      Bengaluru    55468.29
      Delhi        64116.26
      Mumbai       72469.45
    Finding must say Mumbai is highest, Delhi is followed by, and NEVER Jaipur is leading.
    """
    meta, session_id = sales_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_sales")],
        order_by=[OrderByClause(target="sum_sales", descending=False)],
        limit=10,
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    # Result rows are ordered ASC
    assert result.rows[0]["City"] == "Pune"  # smallest sales
    # Mumbai is highest
    finding = format_deterministic_result(result, "What is the total sales by city?")

    assert "Mumbai has the highest total sales" in finding
    assert "72,469.45" in finding
    assert "Pune has the highest" not in finding
    assert "Jaipur has the highest" not in finding
    assert "leading City is Jaipur" not in finding
    assert "leading City is Pune" not in finding


def test_finding_ranking_agrees_with_table_descending(sales_dataset):
    meta, session_id = sales_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_sales")],
        order_by=[OrderByClause(target="sum_sales", descending=True)],
        limit=10,
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    finding = format_deterministic_result(result, "What is the total sales by city?")
    assert "Mumbai has the highest total sales" in finding
    assert "72,469.45" in finding
    assert "Delhi" in finding


def test_finding_ranking_agrees_with_bottom_query(sales_dataset):
    meta, session_id = sales_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_sales")],
        order_by=[OrderByClause(target="sum_sales", descending=False)],
        limit=5,
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    finding = format_deterministic_result(result, "bottom 5 cities by sales")
    assert "Pune has the lowest total sales" in finding
    assert "3,908.87" in finding


# ===========================================================================
# 3. NUMERIC NORMALIZATION TESTS
# ===========================================================================

def test_floating_point_sanitization_removes_noise():
    # 3908.8700000000003 -> 3908.87
    v1 = _sanitize_record_value(3908.8700000000003)
    assert v1 == 3908.87
    assert isinstance(v1, float)

    # 5655.93 -> 5655.93
    v2 = _sanitize_record_value(5655.93)
    assert v2 == 5655.93
    assert isinstance(v2, float)

    # 72469.44999999998 -> 72469.45
    v3 = _sanitize_record_value(72469.44999999998)
    assert v3 == 72469.45
    assert isinstance(v3, float)


def test_query_results_have_clean_floats(sales_dataset):
    meta, session_id = sales_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_sales")],
        limit=10,
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    pune_row = next(r for r in result.rows if r["City"] == "Pune")
    # Float precision noise in original CSV (3908.8700000000003) must be cleaned
    assert pune_row["sum_sales"] == 3908.87
    assert isinstance(pune_row["sum_sales"], float)


# ===========================================================================
# 4. RESULT/FINDING CONSISTENCY INVARIANT
# ===========================================================================

def test_verify_finding_result_consistency_invariant(sales_dataset):
    meta, session_id = sales_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_sales")],
        order_by=[OrderByClause(target="sum_sales", descending=False)],
        limit=10,
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    valid_finding = format_deterministic_result(result, "What is the total sales by city?")
    assert verify_finding_result_consistency(valid_finding, result) is True

    # Intentionally contradictory finding claiming Jaipur is highest
    contradictory_finding = "Across 7 cities, Jaipur has the highest total sales at 52,357.08."
    assert verify_finding_result_consistency(contradictory_finding, result) is False


# ===========================================================================
# 5. END-TO-END CHAT INTEGRATION TEST ("Show me the top 5 products by sales")
# ===========================================================================

def test_chat_e2e_top_5_products_by_sales(sales_dataset):
    meta, session_id = sales_dataset
    client = TestClient(app)

    payload = {
        "message": "Show me the top 5 products by sales.",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert data["tool_used"] == "execute_analysis_plan"
    assert data["table"] is not None
    assert len(data["table"]["rows"]) == 5

    # Top product is Widget D (72,469.45)
    response_text = data["response"]
    assert "Widget D" in response_text
    assert "highest" in response_text
    assert "72,469.45" in response_text
