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
    FilterClause,
    FilterOperator,
    DatasetMetadata,
)
from app.analytics.planner import (
    generate_analysis_plan,
    format_deterministic_result,
    verify_finding_result_consistency,
)
from app.analytics.compiler import compile_plan, PlanValidationError, ColumnNotFoundError
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

def test_chat_e2e_top_5_products_by_sales(sales_dataset, client):
    meta, session_id = sales_dataset

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


# ===========================================================================
# 6. ANA-10, ANA-11, ANA-12 REGRESSION TESTS
# ===========================================================================

def test_ana_10_comparison_filter_not_parsed_as_limit(sales_dataset):
    """
    ANA-10: 'at least 3000' must be parsed as a numeric comparison filter (Profit >= 3000.0),
    NEVER as a row limit (which would trip the limit <= 500 constraint).
    """
    meta, session_id = sales_dataset
    query = "List products where profit is at least 3000"
    plan = generate_analysis_plan(query, meta)

    assert plan.limit == 50
    assert len(plan.filters) == 1
    f = plan.filters[0]
    assert f.column == "Profit"
    assert f.operator in (FilterOperator.GTE, ">=")
    assert f.value == 3000.0

    # Compiles and executes cleanly
    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    # Matching products with profit >= 3000 in sales_dataset:
    # Widget A (5000), Widget B (6000), Widget C (7500), Widget D (9000), Widget G (3200) -> 5 products
    matched_prods = [r["Product"] for r in res.rows]
    assert len(matched_prods) == 5
    assert "Widget A" in matched_prods
    assert "Widget E" not in matched_prods  # Profit 450


def test_ana_11_unsupported_metric_rejected(sales_dataset):
    """
    ANA-11: Queries requesting non-existent metrics (e.g. customer satisfaction rating)
    must fail closed via ColumnNotFoundError rather than substituting unrelated columns.
    """
    meta, _ = sales_dataset
    query = "Show customer satisfaction rating by product"
    with pytest.raises(ColumnNotFoundError) as exc_info:
        generate_analysis_plan(query, meta)
    assert "satisfaction" in str(exc_info.value).lower() or "rating" in str(exc_info.value).lower()


def test_ana_12_unsupported_prediction_rejected(sales_dataset):
    """
    ANA-12: Queries requesting speculative future forecasting or stock price prediction
    must fail closed via PlanValidationError rather than generating hallucinated SQL queries.
    """
    meta, _ = sales_dataset
    query = "Predict our company stock price in 2035"
    with pytest.raises(PlanValidationError) as exc_info:
        generate_analysis_plan(query, meta)
    assert "predict" in str(exc_info.value).lower() or "speculative" in str(exc_info.value).lower()


def test_ana_12_existing_forecast_data_allowed(monkeypatch, tmp_path):
    """
    ANA-12: Datasets that legitimately include a forecast column allow querying existing
    forecast data without raising speculative prediction errors.
    """
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    csv_data = (
        "Product,Region,Sales,Forecast\n"
        "Widget A,North,100.0,120.0\n"
        "Widget B,South,200.0,210.0\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset(
        session_id="forecast-session",
        filename="forecast_data.csv",
        content_bytes=csv_data,
    )
    query = "Show total forecast by Region"
    plan = generate_analysis_plan(query, meta)
    assert plan is not None
    # Verify column Forecast is targeted in aggregations
    agg_cols = [m.column.lower() for m in plan.aggregations]
    assert "forecast" in agg_cols


def test_ana_12_prediction_rejected_even_if_forecast_or_inventory_stock_present(monkeypatch, tmp_path):
    """
    ANA-12: Even if a dataset contains inventory stock or pre-existing forecast columns,
    unsupported requests to generate future predictions or stock price predictions must be rejected.
    """
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    csv_data = (
        "Product,inventory_stock,forecast_sales\n"
        "Widget A,500,600.0\n"
        "Widget B,250,300.0\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset(
        session_id="stock-forecast-session",
        filename="stock_forecast.csv",
        content_bytes=csv_data,
    )
    # 1. Predicting company stock price in 2035 is rejected
    with pytest.raises(PlanValidationError):
        generate_analysis_plan("Predict our company stock price in 2035", meta)

    # 2. Speculative prediction generation is rejected
    with pytest.raises(PlanValidationError):
        generate_analysis_plan("Predict sales for next year", meta)


def test_ana_neighboring_least_limit_and_breakdown_preserved(sales_dataset):
    """
    Verify neighboring behavior is preserved:
    1. Genuine 'least N' ranking without 'at' still parses as limit.
    2. Ordinary categorical breakdown without explicit metric still works.
    """
    meta, _ = sales_dataset
    # 1. Genuine least N
    plan_least = generate_analysis_plan("Show least 3 products by sales", meta)
    assert plan_least.limit == 3
    assert plan_least.order_by[0].descending is False

    # 2. Ordinary breakdown
    plan_breakdown = generate_analysis_plan("Breakdown by City", meta)
    assert plan_breakdown.group_by[0].column == "City"


def test_ana_10_benchmark_store_fixture_reconciliation(monkeypatch, tmp_path):
    """
    Reconcile fixture discrepancy:
    - sales_dataset has 7 widgets with 5 having Profit >= 3000 (Widgets A, B, C, D, G).
    - store_sales.csv (benchmark dataset) has 15 items with exactly 4 having Profit >= 3000
      (Cheese, Milk, Coffee, Chocolate).
    Verifies that the benchmark dataset produces precisely 4 products.
    """
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    from evaluation.eval_35_cases import STORE_CSV_TEXT
    meta = dataset_service.ingest_dataset(
        session_id="benchmark-reconciliation-session",
        filename="store_sales.csv",
        content_bytes=STORE_CSV_TEXT.encode("utf-8"),
    )
    query = "List products where profit is at least 3000"
    plan = generate_analysis_plan(query, meta)

    assert plan.limit == 50
    assert len(plan.filters) == 1
    assert plan.filters[0].column == "Profit"
    assert plan.filters[0].value == 3000.0

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, "benchmark-reconciliation-session", plan)
    prods = [r["Product"] for r in res.rows]
    assert len(prods) == 4
    assert sorted(prods) == sorted(["Cheese", "Milk", "Coffee", "Chocolate"])


def test_ana_neighboring_existing_forecast_column_not_blocked(monkeypatch, tmp_path):
    """
    Edge case: If dataset actually has a column named 'forecast_sales',
    the predictive guard must NOT block valid descriptive queries against it.
    """
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    csv_data = (
        "Product,Forecast_Sales,Actual_Sales\n"
        "Widget A,60000,52000\n"
        "Widget B,40000,45000\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset(
        session_id="forecast-col-session",
        filename="forecast_data.csv",
        content_bytes=csv_data,
    )
    # Query referencing the existing forecast column must succeed
    plan = generate_analysis_plan("Show forecast_sales by product", meta)
    assert plan.group_by[0].column == "Product"
    assert plan.aggregations[0].column == "Forecast_Sales"


def test_ana_neighboring_existing_rating_column_not_blocked(monkeypatch, tmp_path):
    """
    Edge case: If dataset actually has a column named 'customer_rating',
    the unsupported metric guard must NOT block valid descriptive queries against it.
    """
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    csv_data = (
        "Product,Customer_Rating,Sales\n"
        "Widget A,4.8,52000\n"
        "Widget B,4.2,45000\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset(
        session_id="rating-col-session",
        filename="ratings_data.csv",
        content_bytes=csv_data,
    )
    plan = generate_analysis_plan("Show customer_rating by product", meta)
    assert plan.group_by[0].column == "Product"
    assert plan.aggregations[0].column == "Customer_Rating"


def test_ana_neighboring_at_most_filter(sales_dataset):
    """
    Edge case: 'at most' comparison filter must extract <= (LTE) operator.
    """
    meta, session_id = sales_dataset
    query = "List products where profit is at most 2000"
    plan = generate_analysis_plan(query, meta)

    assert len(plan.filters) == 1
    assert plan.filters[0].column == "Profit"
    assert plan.filters[0].operator in (FilterOperator.LTE, "<=")
    assert plan.filters[0].value == 2000.0

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    # In sales_dataset: Widget E (450), Widget F (2000) -> 2 products
    prods = [r["Product"] for r in res.rows]
    assert len(prods) == 2
    assert "Widget E" in prods
    assert "Widget F" in prods


def test_ana_neighboring_ranking_and_filter_combined(sales_dataset):
    """
    Edge case: Query combining Top-N ranking and comparison filter
    e.g. 'Top 2 products where profit is at least 3000'
    must correctly set limit=2, sort descending, and retain Profit >= 3000.
    """
    meta, session_id = sales_dataset
    query = "Top 2 products where profit is at least 3000"
    plan = generate_analysis_plan(query, meta)

    assert plan.limit == 2
    assert plan.order_by[0].descending is True
    assert len(plan.filters) == 1
    assert plan.filters[0].column == "Profit"
    assert plan.filters[0].value == 3000.0

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    # Highest profits in sales_dataset with Profit >= 3000:
    # Widget D (9000), Widget C (7500)
    assert len(res.rows) == 2
    assert res.rows[0]["Product"] == "Widget D"
    assert res.rows[1]["Product"] == "Widget C"


def test_ana_07_average_profit_by_region_fallback(monkeypatch, tmp_path):
    """
    ANA-07: When LLM outputs malformed JSON for 'What is the average profit by region?',
    the heuristic fallback must correctly select AVG(Profit) instead of defaulting to SUM.
    """
    from unittest.mock import patch, MagicMock
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    from evaluation.eval_35_cases import STORE_CSV_TEXT
    meta = dataset_service.ingest_dataset(
        session_id="ana07-avg-fallback-session",
        filename="store_sales.csv",
        content_bytes=STORE_CSV_TEXT.encode("utf-8"),
    )
    mock_bad_llm = MagicMock(invoke=MagicMock(side_effect=ValueError("Expecting ':' delimiter: line 4 column 48")))
    with patch("app.analytics.planner.get_llm", return_value=mock_bad_llm):
        plan = generate_analysis_plan("What is the average profit by region?", meta)

    assert plan.group_by[0].column == "Region"
    assert plan.aggregations[0].column == "Profit"
    assert plan.aggregations[0].function in (AggregationFunction.AVG, "avg")


def test_ana_07_mean_profit_by_region_fallback(monkeypatch, tmp_path):
    """
    ANA-07: Equivalent 'mean' wording must also trigger AVG selection in the heuristic fallback.
    """
    from unittest.mock import patch, MagicMock
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    from evaluation.eval_35_cases import STORE_CSV_TEXT
    meta = dataset_service.ingest_dataset(
        session_id="ana07-mean-fallback-session",
        filename="store_sales.csv",
        content_bytes=STORE_CSV_TEXT.encode("utf-8"),
    )
    mock_bad_llm = MagicMock(invoke=MagicMock(side_effect=ValueError("Invalid non-json output from model")))
    with patch("app.analytics.planner.get_llm", return_value=mock_bad_llm):
        plan = generate_analysis_plan("What is the mean profit by region?", meta)

    assert plan.group_by[0].column == "Region"
    assert plan.aggregations[0].column == "Profit"
    assert plan.aggregations[0].function in (AggregationFunction.AVG, "avg")


def test_ana_07_normal_sum_query_retains_sum_fallback(monkeypatch, tmp_path):
    """
    ANA-07: Standard sum/total queries must still select SUM in the heuristic fallback.
    """
    from unittest.mock import patch, MagicMock
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    from evaluation.eval_35_cases import STORE_CSV_TEXT
    meta = dataset_service.ingest_dataset(
        session_id="ana07-sum-fallback-session",
        filename="store_sales.csv",
        content_bytes=STORE_CSV_TEXT.encode("utf-8"),
    )
    mock_bad_llm = MagicMock(invoke=MagicMock(side_effect=ValueError("Malformed JSON output")))
    # 1. Explicit total query
    with patch("app.analytics.planner.get_llm", return_value=mock_bad_llm):
        plan_total = generate_analysis_plan("What is the total profit by region?", meta)
    assert plan_total.group_by[0].column == "Region"
    assert plan_total.aggregations[0].column == "Profit"
    assert plan_total.aggregations[0].function in (AggregationFunction.SUM, "sum")

    # 2. General metric breakdown without average/mean
    with patch("app.analytics.planner.get_llm", return_value=mock_bad_llm):
        plan_general = generate_analysis_plan("What is profit by region?", meta)
    assert plan_general.group_by[0].column == "Region"
    assert plan_general.aggregations[0].column == "Profit"
    assert plan_general.aggregations[0].function in (AggregationFunction.SUM, "sum")


def test_ana_07_malformed_llm_json_triggers_fallback_and_executes_cleanly(monkeypatch, tmp_path):
    """
    ANA-07 End-to-End: Malformed LLM JSON for 'What is the average profit by region?'
    triggers the fallback, compiles, and computes the exact expected regional averages.
    """
    from unittest.mock import patch, MagicMock
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    from evaluation.eval_35_cases import STORE_CSV_TEXT
    meta = dataset_service.ingest_dataset(
        session_id="ana07-e2e-fallback-session",
        filename="store_sales.csv",
        content_bytes=STORE_CSV_TEXT.encode("utf-8"),
    )
    mock_bad_llm = MagicMock(invoke=MagicMock(side_effect=ValueError("Expecting ':' delimiter: line 4 column 48 (char 120)")))
    with patch("app.analytics.planner.get_llm", return_value=mock_bad_llm):
        plan = generate_analysis_plan("What is the average profit by region?", meta)

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, "ana07-e2e-fallback-session", plan)

    assert len(res.rows) == 5
    # Region averages computed on STORE_CSV_TEXT:
    reg_avgs = {}
    for r in res.rows:
        reg = r["Region"]
        val = r.get("Average Profit") if "Average Profit" in r else r.get("avg_Profit")
        reg_avgs[reg] = round(float(val), 2)

    assert reg_avgs["Central"] == 1625.0
    assert reg_avgs["East"] == 3625.0
    assert reg_avgs["North"] == 3133.33
    assert reg_avgs["South"] == 2366.67
    assert reg_avgs["West"] == 1300.0


# ===========================================================================
# 11. DATASET INTELLIGENCE QA REGRESSION TESTS (SUPERMARKET DATASET)
# ===========================================================================

@pytest.fixture
def supermarket_dataset(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setattr("app.config.settings.datasets_dir", str(tmp_path))
    excel_path = Path("data/datasets/6d035b56-25d6-483b-9fb5-812160f04acb/e5279024-488a-4f7f-b981-28dc1f6e5ae7/original.xlsx")
    if excel_path.exists():
        content_bytes = excel_path.read_bytes()
        meta = dataset_service.ingest_dataset(
            session_id="supermarket-qa-session",
            filename="supermarket_sales.xlsx",
            content_bytes=content_bytes,
        )
    else:
        csv_data = (
            "Invoice ID,Branch,City,Customer Type,Gender,Product,Category,Unit Price,Quantity,Date,Payment,Sales,Rating\n"
            "INV-001,A,Jaipur,Member,Female,Health and beauty,Cosmetics,74.69,7,2019-01-05,Ewallet,52357.08,9.1\n"
            "INV-002,C,Bengaluru,Normal,Female,Electronic accessories,Electronics,15.28,5,2019-03-08,Cash,55468.29,6.6\n"
            "INV-003,A,Delhi,Normal,Male,Home and lifestyle,Home,46.33,1,2019-03-03,Credit card,64116.26,7.4\n"
            "INV-004,B,Mumbai,Normal,Male,Sports and travel,Sports,86.31,7,2019-02-08,Ewallet,72469.45,4.8\n"
        ).encode("utf-8")
        meta = dataset_service.ingest_dataset(
            session_id="supermarket-qa-session",
            filename="supermarket_sales.csv",
            content_bytes=csv_data,
        )
    return meta, "supermarket-qa-session"


def test_regression_filtered_query_followed_by_unrelated_total_by_category(supermarket_dataset):
    """
    Verifies that a filtered query (e.g. Sales > 20000) does not contaminate an
    unrelated standalone query (e.g. Total sales by category), while intentional
    follow-ups (e.g. Now break down by product) retain previous filters.
    """
    meta, session_id = supermarket_dataset
    prev_plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="City")],
        aggregations=[AggregationClause(column="Sales", function=AggregationFunction.SUM, alias="sum_Sales")],
        filters=[FilterClause(column="Sales", operator=FilterOperator.GT, value=20000)],
        limit=5,
    )

    # 1. Standalone query: must start clean with no inherited filters
    standalone_query = "What is the total Sales by Category?"
    plan_new = generate_analysis_plan(standalone_query, meta, previous_plan=prev_plan)
    assert plan_new.filters == [], f"Expected clean filters, got {plan_new.filters}"
    assert any(gb.column == "Category" for gb in plan_new.group_by)
    assert plan_new.aggregations[0].column == "Sales"
    assert plan_new.aggregations[0].function in (AggregationFunction.SUM, "sum")

    # 2. Intentional follow-up query: must retain filter from previous turn
    followup_query = "Now break down by product"
    plan_followup = generate_analysis_plan(followup_query, meta, previous_plan=prev_plan)
    assert len(plan_followup.filters) == 1
    assert plan_followup.filters[0].column == "Sales"
    assert plan_followup.filters[0].operator == FilterOperator.GT
    assert plan_followup.filters[0].value == 20000
    assert any(gb.column == "Product" for gb in plan_followup.group_by)


def test_regression_top_5_return_only_no_false_filter(supermarket_dataset):
    """
    Verifies that 'return only the top 5' does not create a false filter
    like Product == 'return' and orders descending.
    """
    meta, _ = supermarket_dataset
    query = "Return only the top 5 products by sales"
    plan = generate_analysis_plan(query, meta)

    # Must not contain any false filter on 'return'
    for f in plan.filters:
        assert str(f.value).lower() != "return"
        assert f.column != "Product" or str(f.value).lower() != "return"

    assert plan.limit == 5
    assert len(plan.order_by) > 0
    assert plan.order_by[0].descending is True


def test_regression_highest_to_lowest_compound_sort(supermarket_dataset):
    """
    Verifies that compound sort directions like 'highest to lowest' evaluate to
    descending=True and 'lowest to highest' evaluate to descending=False.
    """
    meta, _ = supermarket_dataset
    plan_desc = generate_analysis_plan("Show sales by city from highest to lowest", meta)
    assert len(plan_desc.order_by) > 0
    assert plan_desc.order_by[0].descending is True

    plan_asc = generate_analysis_plan("Show sales by city from lowest to highest", meta)
    assert len(plan_asc.order_by) > 0
    assert plan_asc.order_by[0].descending is False


def test_regression_multi_dimension_grouping_city_and_category(supermarket_dataset):
    """
    Verifies that multi-column grouping 'City and Category' preserves both dimensions.
    """
    meta, _ = supermarket_dataset
    query = "What is the total sales broken down by City and Category?"
    plan = generate_analysis_plan(query, meta)

    gb_cols = [gb.column for gb in plan.group_by]
    assert "City" in gb_cols, f"Expected City in {gb_cols}"
    assert "Category" in gb_cols, f"Expected Category in {gb_cols}"


def test_regression_explicit_aggregation_with_negation(supermarket_dataset):
    """
    Verifies that 'sum of Sales... Do not use average' enforces SUM and rejects AVG.
    """
    meta, _ = supermarket_dataset
    query = "Create a bar chart showing the sum of Sales for each City. Do not use average Sales."
    plan = generate_analysis_plan(query, meta)

    assert plan.requires_chart is True
    assert len(plan.aggregations) > 0
    assert plan.aggregations[0].function in (AggregationFunction.SUM, "sum")
    assert plan.aggregations[0].function not in (AggregationFunction.AVG, "avg")


def test_regression_record_count_by_city(supermarket_dataset):
    """
    Verifies that 'Count the number of records for each City' produces COUNT(*)
    rather than a sum of an arbitrary numeric metric.
    """
    meta, session_id = supermarket_dataset
    query = "Count the number of records for each City."
    plan = generate_analysis_plan(query, meta)

    assert any(gb.column == "City" for gb in plan.group_by)
    assert len(plan.aggregations) > 0
    assert plan.aggregations[0].function in (AggregationFunction.COUNT, "count")
    assert plan.aggregations[0].column is None

    compiled = compile_plan(plan, meta)
    assert "COUNT(*)" in compiled.sql.upper()

    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    assert len(res.rows) > 0
    for r in res.rows:
        count_val = r.get("count_records") or r.get("count_rows") or r.get("Count")
        assert count_val is not None
        assert int(count_val) > 0


def test_regression_raw_row_query_without_aggregation(supermarket_dataset):
    """
    Verifies that a request for individual sales rows without aggregation
    preserves raw rows without forcing group_by or sum.
    """
    meta, session_id = supermarket_dataset
    query = "List individual sales rows without aggregation"
    plan = generate_analysis_plan(query, meta)

    assert plan.group_by == []
    assert plan.aggregations == []

    compiled = compile_plan(plan, meta)
    assert "GROUP BY" not in compiled.sql.upper()

    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    assert len(res.rows) > 0
    assert "Sales" in res.columns
    for r in res.rows:
        assert r["Sales"] is not None


def test_regression_min_and_max_together(supermarket_dataset):
    """
    Verifies that requesting MIN and MAX together preserves both aggregations.
    """
    meta, session_id = supermarket_dataset
    query = "Show minimum and maximum sales by city"
    plan = generate_analysis_plan(query, meta)

    agg_fns = {a.function for a in plan.aggregations}
    assert AggregationFunction.MIN in agg_fns or "min" in agg_fns
    assert AggregationFunction.MAX in agg_fns or "max" in agg_fns

    compiled = compile_plan(plan, meta)
    assert "MIN(" in compiled.sql.upper()
    assert "MAX(" in compiled.sql.upper()

    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    assert len(res.rows) > 0
    for r in res.rows:
        min_val = r.get("min_sales") if "min_sales" in r else r.get("min_Sales")
        max_val = r.get("max_sales") if "max_sales" in r else r.get("max_Sales")
        assert min_val is not None
        assert max_val is not None
        assert float(min_val) <= float(max_val)


def test_regression_chart_sum_sales_by_city_supermarket(supermarket_dataset, tmp_path, monkeypatch):
    """
    Verifies end-to-end chart generation for 'sum of Sales for each City' against
    the supermarket dataset:
    1. Chart artifact is generated with valid PNG bytes
    2. Metric is SUM (not AVG)
    3. Plotted values agree with independently calculated city totals:
       Bengaluru: 55468.29, Delhi: 64116.26, Jaipur: 52357.08, Mumbai: 72469.45
    """
    from pathlib import Path
    from app.analytics.visualizer import render_chart, validate_chart_compatibility
    meta, session_id = supermarket_dataset

    query = "Create a bar chart showing the sum of Sales for each City. Do not use average Sales."
    plan = generate_analysis_plan(query, meta)

    assert plan.requires_chart is True
    assert plan.aggregations[0].function in (AggregationFunction.SUM, "sum")

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    city_totals = {}
    for r in result.rows:
        city = r["City"]
        val = r.get("sum_Sales") or r.get("Sales") or r.get("Total Sales")
        city_totals[city] = round(float(val), 2)

    assert city_totals.get("Bengaluru") == 55468.29
    assert city_totals.get("Delhi") == 64116.26
    assert city_totals.get("Jaipur") == 52357.08
    assert city_totals.get("Mumbai") == 72469.45

    from app.analytics.visualizer import render_chart, validate_chart_compatibility, resolve_chart_file
    is_compat, eff_type, err_reason = validate_chart_compatibility(result, plan.chart_type, is_explicit=True)
    assert is_compat is True
    assert eff_type in ("bar", "bar_chart")

    artifact = render_chart(result, eff_type, session_id)
    assert artifact.chart_type == eff_type
    assert "sum_Sales" in artifact.title or "Sales" in artifact.title
    assert artifact.url.startswith("/charts/")
    assert len(result.rows) == 4

    file_path = resolve_chart_file(session_id, artifact.chart_id)
    assert file_path.exists()
    assert file_path.stat().st_size > 0

    with open(file_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_regression_universal_total_across_all_cities(supermarket_dataset):
    """
    Verifies that 'What is the total Sales across all cities?' produces a scalar
    SUM(Sales) with no GROUP BY City and no inherited limit (both standalone and follow-up).
    """
    meta, session_id = supermarket_dataset

    # 1. Standalone query
    q = "What is the total Sales across all cities?"
    plan_solo = generate_analysis_plan(q, meta)
    assert plan_solo.group_by == [], f"Expected no group_by, got {plan_solo.group_by}"
    assert plan_solo.limit == 500
    assert len(plan_solo.aggregations) > 0
    assert plan_solo.aggregations[0].column == "Sales"
    assert plan_solo.aggregations[0].function in (AggregationFunction.SUM, "sum")

    compiled_solo = compile_plan(plan_solo, meta)
    assert "GROUP BY" not in compiled_solo.sql.upper()
    res_solo = execute_compiled_query(compiled_solo, meta.dataset_id, session_id, plan_solo)
    assert len(res_solo.rows) == 1
    total_val = res_solo.rows[0].get("total_Sales") or res_solo.rows[0].get("sum_Sales") or res_solo.rows[0].get("Total Sales") or list(res_solo.rows[0].values())[0]
    assert round(float(total_val), 2) == 244411.08

    # 2. Multi-turn follow-up after City grouping
    p1 = generate_analysis_plan("Show sales by city", meta)
    assert any(gb.column == "City" for gb in p1.group_by)
    plan_followup = generate_analysis_plan(q, meta, previous_plan=p1)
    assert plan_followup.group_by == [], f"Expected cleared group_by, got {plan_followup.group_by}"
    assert plan_followup.limit == 500

    compiled_followup = compile_plan(plan_followup, meta)
    assert "GROUP BY" not in compiled_followup.sql.upper()
    res_followup = execute_compiled_query(compiled_followup, meta.dataset_id, session_id, plan_followup)
    assert len(res_followup.rows) == 1
    total_f_val = res_followup.rows[0].get("total_Sales") or res_followup.rows[0].get("sum_Sales") or res_followup.rows[0].get("Total Sales") or list(res_followup.rows[0].values())[0]
    assert round(float(total_f_val), 2) == 244411.08


def test_regression_dataset_record_count(supermarket_dataset):
    """
    Verifies that 'How many records are in the dataset?' returns a single scalar
    COUNT(*) row (500 records) for both standalone and follow-up, while preserving
    explicit breakdown requests like 'count records by Branch'.
    """
    meta, session_id = supermarket_dataset

    # 1. Standalone
    q = "How many records are in the dataset?"
    plan_solo = generate_analysis_plan(q, meta)
    assert plan_solo.group_by == [], f"Expected empty group_by, got {plan_solo.group_by}"
    assert plan_solo.aggregations[0].function in (AggregationFunction.COUNT, "count")

    compiled_solo = compile_plan(plan_solo, meta)
    assert "GROUP BY" not in compiled_solo.sql.upper()
    res_solo = execute_compiled_query(compiled_solo, meta.dataset_id, session_id, plan_solo)
    assert len(res_solo.rows) == 1
    cnt = res_solo.rows[0].get("count_records") or res_solo.rows[0].get("count")
    assert cnt == 500

    # 2. Follow-up after Branch grouping
    p_branch = generate_analysis_plan("Show sales by Branch", meta)
    assert any(gb.column == "Branch" for gb in p_branch.group_by)
    plan_followup = generate_analysis_plan(q, meta, previous_plan=p_branch)
    assert plan_followup.group_by == [], f"Expected cleared group_by, got {plan_followup.group_by}"

    compiled_followup = compile_plan(plan_followup, meta)
    assert "GROUP BY" not in compiled_followup.sql.upper()
    res_followup = execute_compiled_query(compiled_followup, meta.dataset_id, session_id, plan_followup)
    assert len(res_followup.rows) == 1
    cnt_f = res_followup.rows[0].get("count_records") or res_followup.rows[0].get("count")
    assert cnt_f == 500

    # 3. Explicit breakdown request must preserve grouping
    q_breakdown = "count records by Branch"
    plan_breakdown = generate_analysis_plan(q_breakdown, meta)
    assert any(gb.column == "Branch" for gb in plan_breakdown.group_by)
    compiled_bd = compile_plan(plan_breakdown, meta)
    assert "GROUP BY" in compiled_bd.sql.upper()
    res_bd = execute_compiled_query(compiled_bd, meta.dataset_id, session_id, plan_breakdown)
    assert len(res_bd.rows) == 4


def test_regression_bengaluru_filter_column_reconciliation(supermarket_dataset):
    """
    Verifies that 'Show Sales for Bengaluru only' and 'Show Sales for Bengaluru'
    consistently resolve Bengaluru to the City column and return matching rows,
    while ambiguous or unmatched values do not guess.
    """
    from app.analytics.planner import _reconcile_filter_column, _DATASET_COLUMN_VALUES_CACHE, _get_known_column_values

    meta, session_id = supermarket_dataset

    # 1. "Show Sales for Bengaluru only" -> City='Bengaluru'
    p_only = generate_analysis_plan("Show Sales for Bengaluru only", meta)
    assert len(p_only.filters) == 1
    assert p_only.filters[0].column == "City"
    assert p_only.filters[0].value == "Bengaluru"
    compiled_only = compile_plan(p_only, meta)
    res_only = execute_compiled_query(compiled_only, meta.dataset_id, session_id, p_only)
    assert len(res_only.rows) > 0

    # 2. "Show Sales for Bengaluru" -> City='Bengaluru'
    p_no_only = generate_analysis_plan("Show Sales for Bengaluru", meta)
    assert len(p_no_only.filters) == 1
    assert p_no_only.filters[0].column == "City"
    assert p_no_only.filters[0].value == "Bengaluru"
    compiled_no_only = compile_plan(p_no_only, meta)
    res_no_only = execute_compiled_query(compiled_no_only, meta.dataset_id, session_id, p_no_only)
    assert len(res_no_only.rows) > 0

    # 3. Unmatched filter value preserves original column without guessing
    unmatched_col = _reconcile_filter_column("Branch", "NonExistentLocationXYZ", meta)
    assert unmatched_col == "Branch"

    # 4. Ambiguous value matching multiple columns preserves original column without guessing
    _get_known_column_values("City", meta)
    cache_cat = _DATASET_COLUMN_VALUES_CACHE.get((meta.dataset_id, "Category"), set())
    cache_pay = _DATASET_COLUMN_VALUES_CACHE.get((meta.dataset_id, "Payment"), set())
    cache_cat.add("AmbiguousToken123")
    cache_pay.add("AmbiguousToken123")
    ambiguous_col = _reconcile_filter_column("Branch", "AmbiguousToken123", meta)
    assert ambiguous_col == "Branch"
    cache_cat.discard("AmbiguousToken123")
    cache_pay.discard("AmbiguousToken123")


def test_regression_preservation_of_previously_passing_behaviors(supermarket_dataset):
    """
    Verifies that the 4 previously passing manual QA behaviors remain fully intact:
    1. Top-5 products ranked by total Sales
    2. City sales breakdown
    3. Category grouping as a follow-up that preserves the Bengaluru filter
    4. City bar-chart request
    """
    meta, session_id = supermarket_dataset

    # 1. Top-5 products ranked by total Sales
    p_top5 = generate_analysis_plan("top 5 products by sales", meta)
    assert any(gb.column == "Product" for gb in p_top5.group_by)
    assert p_top5.limit == 5
    assert len(p_top5.order_by) > 0
    assert p_top5.order_by[0].descending is True

    # 2. City sales breakdown
    p_city = generate_analysis_plan("Show sales by city", meta)
    assert any(gb.column == "City" for gb in p_city.group_by)
    assert p_city.aggregations[0].column == "Sales"

    # 3. Category grouping as a follow-up that preserves the Bengaluru filter
    p_turn1 = generate_analysis_plan("Show sales for Bengaluru", meta)
    assert any(f.column == "City" and f.value == "Bengaluru" for f in p_turn1.filters)
    p_turn2 = generate_analysis_plan("Plot sales by category", meta, previous_plan=p_turn1)
    assert any(gb.column == "Category" for gb in p_turn2.group_by)
    assert any(f.column == "City" and f.value == "Bengaluru" for f in p_turn2.filters), f"Expected preserved City=Bengaluru, got {p_turn2.filters}"

    # 4. City bar-chart request
    p_chart = generate_analysis_plan("Show sales by city as a bar chart", meta)
    assert any(gb.column == "City" for gb in p_chart.group_by)
    assert p_chart.requires_chart is True
    assert p_chart.chart_type == "bar"


def test_regression_explicit_clear_previous_filters_multiturn(supermarket_dataset):
    """
    Verifies that explicit instructions to clear or ignore previous filters
    (e.g., 'across the entire dataset', 'ignore previous filters', 'without filters',
    'clear all filters') reliably remove inherited filters while preserving the dataset,
    whereas genuine follow-up queries (e.g. 'Plot sales by category') retain filters.
    """
    meta, session_id = supermarket_dataset

    # Turn 1: Filter to Bengaluru
    p_turn1 = generate_analysis_plan("Show sales for Bengaluru", meta)
    assert any(f.column == "City" and f.value == "Bengaluru" for f in p_turn1.filters)

    # Turn 2: Genuine follow-up must retain Bengaluru filter
    p_turn2 = generate_analysis_plan("Plot sales by category", meta, previous_plan=p_turn1)
    assert any(gb.column == "Category" for gb in p_turn2.group_by)
    assert any(f.column == "City" and f.value == "Bengaluru" for f in p_turn2.filters), (
        f"Expected preserved City=Bengaluru, got {p_turn2.filters}"
    )

    # Turn 3: Explicit clear instruction must remove inherited filters and evaluate across whole dataset
    q_clear = "Count records by Branch across the entire dataset, ignoring previous filters."
    p_turn3 = generate_analysis_plan(q_clear, meta, previous_plan=p_turn2)
    assert p_turn3.filters == [], f"Expected filters to be cleared, got {p_turn3.filters}"
    assert any(gb.column == "Branch" for gb in p_turn3.group_by)
    assert p_turn3.requires_chart is False

    # Execute and verify branch distribution across entire 500-record dataset
    compiled = compile_plan(p_turn3, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, meta.session_id, plan=p_turn3)
    branch_counts = {r["Branch"]: r["count_records"] for r in result.rows}
    assert branch_counts == {"A": 105, "B": 133, "C": 143, "D": 119}
    assert sum(branch_counts.values()) == 500

    # Test other explicit clear variations
    for variation in [
        "Count records by Branch across the entire dataset",
        "Count records by Branch, ignore previous filters",
        "Count records by Branch without filters",
        "Clear all filters and count records by Branch",
    ]:
        p_var = generate_analysis_plan(variation, meta, previous_plan=p_turn2)
        assert p_var.filters == [], f"Expected empty filters for '{variation}', got {p_var.filters}"
        assert any(gb.column == "Branch" for gb in p_var.group_by)
