"""
Comprehensive regression suite for Top-N Dataset Analytics Backend Correctness:
1. Top 5 products by sales planning and SQL compilation
2. Correct DESC ordering (highest to lowest, strictly decreasing)
3. Correct Top-5 selection across the full dataset
4. Grounded finding narrative agrees 100% with table rows
5. No unwanted 'top_products' column or dimension count hallucination
6. Bottom-N behavior (ASC ordering, lowest first, finding reports lowest)
7. Ordinary grouped queries without explicit sorting (preserves order_by=[])
8. Explicit ASC / DESC sorting directives
9. Phase 8 multi-turn analytical memory & mutations remain intact
"""
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.api.session_security import sign_session_id
from app.api.rest import _sessions
from app.analytics.schemas import (
    AnalysisPlan,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
    OrderByClause,
    FilterClause,
    FilterOperator,
)
from app.analytics.planner import (
    generate_analysis_plan,
    format_deterministic_result,
    verify_finding_result_consistency,
    _normalize_and_align_plan,
)
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query
from app.analytics import dataset_service


SAMPLE_STORE_CSV = (
    "Product,Region,Total Sales,Profit\n"
    "Apples,North,17057.41,2100.0\n"
    "Bananas,South,6205.76,900.0\n"
    "Biscuits,East,3908.87,450.0\n"
    "Bread,West,6516.10,800.0\n"
    "Cheese,North,27906.30,4200.0\n"
    "Milk,South,35120.50,5100.0\n"
    "Coffee,East,41890.00,6800.0\n"
    "Tea,West,12450.25,1800.0\n"
    "Chocolate,North,19300.00,3100.0\n"
    "Yogurt,South,8750.60,1100.0\n"
).encode("utf-8")


@pytest.fixture
def store_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))
    meta = dataset_service.ingest_dataset(
        session_id="top-n-session",
        filename="store_sales.csv",
        content_bytes=SAMPLE_STORE_CSV,
    )
    return meta, "top-n-session"


# ===========================================================================
# 1. TOP 5 PRODUCTS BY SALES PLANNING & SQL COMPILATION
# ===========================================================================

def test_top_5_products_by_sales_plan(store_dataset):
    meta, _ = store_dataset
    plan = generate_analysis_plan("Show me the top 5 products by sales", meta)

    assert plan.group_by[0].column == "Product"
    assert len(plan.aggregations) == 1
    assert plan.aggregations[0].column == "Total Sales"
    assert plan.aggregations[0].function == AggregationFunction.SUM
    assert plan.limit == 5
    assert len(plan.order_by) == 1
    assert plan.order_by[0].descending is True
    assert plan.order_by[0].target in ("sum_Total Sales", "Total Sales")

    compiled = compile_plan(plan, meta)
    sql = compiled.sql.upper()
    assert 'GROUP BY "PRODUCT"' in sql
    assert 'SUM("TOTAL SALES")' in sql
    assert 'ORDER BY "SUM_TOTAL SALES" DESC' in sql or 'ORDER BY "TOTAL SALES" DESC' in sql
    assert "LIMIT 5" in sql


# ===========================================================================
# 2. CORRECT DESC ORDERING & 3. CORRECT TOP-5 SELECTION
# ===========================================================================

def test_correct_desc_ordering_and_top_5_selection(store_dataset):
    meta, session_id = store_dataset
    plan = generate_analysis_plan("Show me the top 5 products by sales", meta)
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    assert result.row_count == 5
    rows = result.rows

    # Verify strictly decreasing order
    for i in range(len(rows) - 1):
        assert rows[i]["sum_Total Sales"] >= rows[i + 1]["sum_Total Sales"]

    # In our dataset, the true top 5 products are:
    # 1. Coffee (41,890.00)
    # 2. Milk (35,120.50)
    # 3. Cheese (27,906.30)
    # 4. Chocolate (19,300.00)
    # 5. Apples (17,057.41)
    assert rows[0]["Product"] == "Coffee"
    assert rows[0]["sum_Total Sales"] == 41890.00
    assert rows[1]["Product"] == "Milk"
    assert rows[1]["sum_Total Sales"] == 35120.50
    assert rows[2]["Product"] == "Cheese"
    assert rows[2]["sum_Total Sales"] == 27906.30
    assert rows[3]["Product"] == "Chocolate"
    assert rows[3]["sum_Total Sales"] == 19300.00
    assert rows[4]["Product"] == "Apples"
    assert rows[4]["sum_Total Sales"] == 17057.41


# ===========================================================================
# 4. FINDING AGREES WITH TABLE
# ===========================================================================

def test_finding_agrees_with_table(store_dataset):
    meta, session_id = store_dataset
    plan = generate_analysis_plan("Show me the top 5 products by sales", meta)
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    finding = format_deterministic_result(result, "Show me the top 5 products by sales")
    # Finding must declare Coffee as highest, followed by Milk
    assert "Coffee has the highest Total Sales" in finding or "coffee has the highest total sales" in finding.lower()
    assert "41,890.00" in finding
    assert "Milk" in finding
    assert "35,120.50" in finding
    assert verify_finding_result_consistency(finding, result) is True


# ===========================================================================
# 5. NO UNWANTED top_products FIELD
# ===========================================================================

def test_no_unwanted_top_products_column(store_dataset):
    meta, session_id = store_dataset
    # Test normalization strips any hallucinated dimension aggregation or alias
    mock_hallucinated_plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="Product")],
        aggregations=[
            AggregationClause(column="Product", function=AggregationFunction.COUNT, alias="top_products"),
            AggregationClause(column="Total Sales", function=AggregationFunction.SUM, alias="sum_Total Sales"),
        ],
        order_by=[OrderByClause(target="top_products", descending=False)],
        limit=5,
    )
    cleaned_plan = _normalize_and_align_plan(mock_hallucinated_plan, "Show me the top 5 products by sales", meta)

    # top_products must be purged
    aliases = [a.alias for a in cleaned_plan.aggregations]
    assert "top_products" not in aliases
    assert len(cleaned_plan.aggregations) == 1
    assert cleaned_plan.aggregations[0].column == "Total Sales"
    assert cleaned_plan.order_by[0].descending is True

    compiled = compile_plan(cleaned_plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, cleaned_plan)
    assert "top_products" not in result.columns
    assert result.columns == ["Product", "sum_Total Sales"]


# ===========================================================================
# 6. BOTTOM-N BEHAVIOR
# ===========================================================================

def test_bottom_n_behavior(store_dataset):
    meta, session_id = store_dataset
    plan = generate_analysis_plan("Show me the bottom 3 products by sales", meta)

    assert plan.group_by[0].column == "Product"
    assert plan.limit == 3
    assert plan.order_by[0].descending is False

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    assert result.row_count == 3
    # Lowest 3 in dataset: Biscuits (3,908.87), Bananas (6,205.76), Bread (6,516.10)
    assert result.rows[0]["Product"] == "Biscuits"
    assert result.rows[0]["sum_Total Sales"] == 3908.87
    assert result.rows[1]["Product"] == "Bananas"
    assert result.rows[1]["sum_Total Sales"] == 6205.76
    assert result.rows[2]["Product"] == "Bread"
    assert result.rows[2]["sum_Total Sales"] == 6516.10

    # Verify rows strictly increasing
    assert result.rows[0]["sum_Total Sales"] <= result.rows[1]["sum_Total Sales"] <= result.rows[2]["sum_Total Sales"]

    finding = format_deterministic_result(result, "Show me the bottom 3 products by sales")
    assert "Biscuits has the lowest" in finding
    assert "3,908.87" in finding


# ===========================================================================
# 7. ORDINARY GROUPED QUERIES WITHOUT EXPLICIT SORTING
# ===========================================================================

def test_ordinary_grouped_query_no_explicit_sort(store_dataset):
    meta, _ = store_dataset
    plan = generate_analysis_plan("What is total sales by Region?", meta)

    assert plan.group_by[0].column == "Region"
    assert len(plan.aggregations) == 1
    assert plan.aggregations[0].column == "Total Sales"
    # Unordered grouped query should not force order_by
    assert plan.order_by == []

    compiled = compile_plan(plan, meta)
    assert "ORDER BY" not in compiled.sql.upper()


# ===========================================================================
# 8. EXPLICIT ASC / DESC SORTING
# ===========================================================================

def test_explicit_asc_desc(store_dataset):
    meta, _ = store_dataset

    # Explicit DESC
    plan_desc = generate_analysis_plan("Total sales by Region descending", meta)
    assert len(plan_desc.order_by) == 1
    assert plan_desc.order_by[0].descending is True

    # Explicit ASC
    plan_asc = generate_analysis_plan("Total sales by Region ascending", meta)
    assert len(plan_asc.order_by) == 1
    assert plan_asc.order_by[0].descending is False


# ===========================================================================
# 9. PHASE 8 MUTATIONS REMAIN INTACT
# ===========================================================================

def test_phase_8_multiturn_top_n_mutation(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Group by Region
    plan1 = generate_analysis_plan("What is total sales by Region?", meta)
    assert plan1.group_by[0].column == "Region"

    # Turn 2: Top 5 products by sales (switches dimension, adds top 5 ranking)
    plan2 = generate_analysis_plan(
        question="Show me the top 5 products by sales",
        metadata=meta,
        previous_plan=plan1,
    )
    assert plan2.group_by[0].column == "Product"
    assert plan2.limit == 5
    assert len(plan2.order_by) == 1
    assert plan2.order_by[0].descending is True

    compiled2 = compile_plan(plan2, meta)
    res2 = execute_compiled_query(compiled2, meta.dataset_id, session_id, plan2)
    assert res2.row_count == 5
    assert res2.rows[0]["Product"] == "Coffee"
    assert res2.rows[4]["Product"] == "Apples"


def test_chat_e2e_top_5_products_correctness(store_dataset):
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    payload = {
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert data["table"] is not None
    assert len(data["table"]["rows"]) == 5

    # Check that columns do not include top_products
    assert "top_products" not in data["table"]["columns"]

    # Table rows must be strictly descending: highest first
    table_rows = data["table"]["rows"]
    sales_idx = 1
    for i in range(len(table_rows) - 1):
        assert float(table_rows[i][sales_idx]) >= float(table_rows[i + 1][sales_idx])

    # Top product is Coffee at 41,890.00
    assert table_rows[0][0] == "Coffee"
    assert "Coffee" in data["response"]
    assert "41,890.00" in data["response"]


# ===========================================================================
# 10. PHASE 8 CHART MUTATION PRESERVES TOP-N SEMANTICS
# ===========================================================================

def test_phase_8_chart_mutation_preserves_top_n_semantics(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Show me the top 5 products by sales
    plan1 = generate_analysis_plan("Show me the top 5 products by sales", meta)
    assert plan1.group_by[0].column == "Product"
    assert plan1.limit == 5
    assert len(plan1.order_by) == 1
    assert plan1.order_by[0].descending is True
    assert plan1.aggregations[0].column == "Total Sales"
    assert plan1.aggregations[0].function == AggregationFunction.SUM
    assert plan1.requires_chart is False

    # Turn 2: Plot this as a bar chart
    plan2 = generate_analysis_plan(
        question="Plot this as a bar chart",
        metadata=meta,
        previous_plan=plan1,
    )

    # Invariants on Turn 2:
    assert plan2.requires_chart is True
    assert plan2.chart_type == "bar"
    # Existing metric strictly preserved (SUM on Total Sales)
    assert plan2.aggregations[0].column == "Total Sales"
    assert plan2.aggregations[0].function == AggregationFunction.SUM
    # Existing grouping strictly preserved
    assert len(plan2.group_by) == 1
    assert plan2.group_by[0].column == "Product"
    # Existing Top-N limit strictly preserved
    assert plan2.limit == 5
    # Existing DESC strictly preserved
    assert len(plan2.order_by) == 1
    assert plan2.order_by[0].descending is True
    assert plan2.order_by[0].target == plan1.order_by[0].target

    # Verify DuckDB execution gives identical correct Top 5 DESC rows
    compiled2 = compile_plan(plan2, meta)
    res2 = execute_compiled_query(compiled2, meta.dataset_id, session_id, plan2)
    assert res2.row_count == 5
    assert res2.rows[0]["Product"] == "Coffee"
    assert res2.rows[1]["Product"] == "Milk"
    assert res2.rows[4]["Product"] == "Apples"
    for i in range(len(res2.rows) - 1):
        assert res2.rows[i]["sum_Total Sales"] >= res2.rows[i + 1]["sum_Total Sales"]

    # Equivalent follow-ups must also preserve full AST
    for followup in ("Show this as a chart", "Visualize this", "Can you plot this as a bar chart"):
        plan_alt = generate_analysis_plan(
            question=followup,
            metadata=meta,
            previous_plan=plan1,
        )
        assert plan_alt.requires_chart is True
        assert plan_alt.chart_type == "bar"
        assert plan_alt.aggregations[0].column == "Total Sales"
        assert plan_alt.group_by[0].column == "Product"
        assert plan_alt.limit == 5
        assert plan_alt.order_by[0].descending is True


def test_chart_mutation_does_not_switch_to_quantity_when_first_numeric(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))
    # CSV where Quantity appears BEFORE Sales in column order
    csv_bytes = (
        "Product,Quantity,Sales\n"
        "Coffee,10,41890.0\n"
        "Milk,20,35120.5\n"
        "Cheese,15,27906.3\n"
        "Apples,50,17057.4\n"
        "Tea,30,12450.2\n"
        "Bread,40,6516.1\n"
    ).encode("utf-8")

    meta = dataset_service.ingest_dataset(
        session_id="qty-sales-session",
        filename="qty_sales.csv",
        content_bytes=csv_bytes,
    )

    # Verify Quantity is the first numeric column in profile
    num_cols = [c.name for c in meta.profile.columns if c.dtype in ("int64", "float64")]
    assert num_cols[0] == "Quantity"
    assert num_cols[1] == "Sales"

    # Turn 1: User explicitly asks top 5 products by sales
    plan1 = generate_analysis_plan("Show me the top 5 products by sales", meta)
    assert plan1.aggregations[0].column == "Sales"
    assert plan1.group_by[0].column == "Product"
    assert plan1.limit == 5
    assert plan1.order_by[0].descending is True

    # Turn 2: Plot this as a bar chart
    plan2 = generate_analysis_plan(
        question="Plot this as a bar chart",
        metadata=meta,
        previous_plan=plan1,
    )

    # Invariants: MUST NOT switch to Quantity!
    assert plan2.aggregations[0].column == "Sales"
    assert plan2.aggregations[0].column != "Quantity"
    assert "sales" in plan2.aggregations[0].alias.lower()
    assert plan2.group_by[0].column == "Product"
    assert plan2.limit == 5
    assert plan2.order_by[0].descending is True
    assert plan2.requires_chart is True
    assert plan2.chart_type == "bar"


def test_clean_finding_no_zero_and_clean_metric_label(store_dataset):
    meta, session_id = store_dataset
    plan = generate_analysis_plan("Show me the top 5 products by sales", meta)
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    # 1. Clean metric label without dummy filters
    finding = format_deterministic_result(result, "Show me the top 5 products by sales")
    assert "Coffee has the highest Total Sales" in finding
    assert "sum_" not in finding
    assert "total Total Sales" not in finding
    assert "'0'" not in finding
    assert "in '0'" not in finding

    # 2. Add a numeric threshold filter with value 0 or '0'
    plan_with_zero_filter = plan.model_copy(deep=True)
    plan_with_zero_filter.filters = [
        FilterClause(column="Total Sales", operator=FilterOperator.GT, value=0)
    ]
    result.plan = plan_with_zero_filter

    finding_with_filter = format_deterministic_result(result, "Show me the top 5 products by sales")
    # Must NOT have "in '0'" or "highest total Sales in '0'"
    assert "'0'" not in finding_with_filter
    assert "in '0'" not in finding_with_filter
    assert "in 0" not in finding_with_filter
    assert "Coffee has the highest Total Sales at 41,890.00" in finding_with_filter


def test_chat_e2e_two_turn_top_5_then_plot_chart(store_dataset):
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    # Turn 1: Show me the top 5 products by sales
    res1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "ok"
    assert len(data1["table"]["rows"]) == 5
    assert data1["table"]["rows"][0][0] == "Coffee"

    # Turn 2: Plot this as a bar chart
    res2 = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "ok"
    # Table must remain 5 rows, DESC, Coffee top
    assert data2["table"] is not None
    assert len(data2["table"]["rows"]) == 5
    assert data2["table"]["rows"][0][0] == "Coffee"
    # Chart must be generated
    assert data2.get("chart") is not None
    chart_t = data2["chart"].get("chart_type") or data2["chart"].get("type")
    assert chart_t in ("bar", "bar_chart")
    # Response finding must not contain '0'
    assert "'0'" not in data2["response"]
    assert "in '0'" not in data2["response"]


def test_phase_8_table_mutation_preserves_plan_and_rows(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Show me the top 5 products by sales
    plan1 = generate_analysis_plan("Show me the top 5 products by sales", meta)
    assert plan1.group_by[0].column == "Product"
    assert plan1.aggregations[0].column == "Total Sales"
    assert plan1.aggregations[0].function == AggregationFunction.SUM
    assert plan1.limit == 5
    assert plan1.order_by[0].descending is True
    assert plan1.requires_chart is False

    # Test A: Turn 1 -> Turn 2 (Chart mutation)
    plan_chart = generate_analysis_plan("Plot this as a bar chart", meta, previous_plan=plan1)
    assert plan_chart.requires_chart is True
    assert plan_chart.chart_type == "bar"
    assert plan_chart.group_by[0].column == "Product"
    assert plan_chart.aggregations[0].column == "Total Sales"
    assert plan_chart.aggregations[0].function == AggregationFunction.SUM
    assert plan_chart.limit == 5
    assert plan_chart.order_by[0].descending is True

    # Test B: Turn 1 -> Turn 2 directly to table
    plan_table_direct = generate_analysis_plan("Show this as a table", meta, previous_plan=plan1)
    assert plan_table_direct.requires_chart is False
    assert plan_table_direct.chart_type is None
    assert plan_table_direct.group_by[0].column == "Product"
    assert plan_table_direct.aggregations[0].column == "Total Sales"
    assert plan_table_direct.aggregations[0].function == AggregationFunction.SUM
    assert plan_table_direct.limit == 5
    assert plan_table_direct.order_by[0].descending is True

    # Test C: Full three-turn sequence: Turn 1 -> Plot bar chart -> Show this as a table
    plan3 = generate_analysis_plan("Show this as a table", meta, previous_plan=plan_chart)
    assert plan3.requires_chart is False
    assert plan3.chart_type is None
    assert plan3.group_by[0].column == "Product"
    assert plan3.aggregations[0].column == "Total Sales"
    assert plan3.aggregations[0].function == AggregationFunction.SUM
    assert plan3.limit == 5
    assert plan3.order_by[0].descending is True

    # Verify execution produces the exact same 5 rows in DESC order
    compiled3 = compile_plan(plan3, meta)
    res3 = execute_compiled_query(compiled3, meta.dataset_id, session_id, plan3)
    assert res3.row_count == 5
    assert res3.rows[0]["Product"] == "Coffee"
    assert res3.rows[1]["Product"] == "Milk"
    assert res3.rows[2]["Product"] == "Cheese"
    assert res3.rows[3]["Product"] == "Chocolate"
    assert res3.rows[4]["Product"] == "Apples"

    # Variations of table follow-ups must all preserve full analytical AST
    for followup in ("Display this as a table", "Show the results as a table", "Switch back to a table", "Show as a table"):
        plan_var = generate_analysis_plan(followup, meta, previous_plan=plan_chart)
        assert plan_var.requires_chart is False
        assert plan_var.chart_type is None
        assert plan_var.group_by[0].column == "Product"
        assert plan_var.aggregations[0].column == "Total Sales"
        assert plan_var.limit == 5
        assert plan_var.order_by[0].descending is True


def test_generic_table_mutation_different_dataset(monkeypatch, tmp_path):
    # Test D: Genericity - non-sales, non-product, custom dataset
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))
    custom_csv = (
        "Department,Score\n"
        "Engineering,98.5\n"
        "Research,94.2\n"
        "Design,91.0\n"
        "Marketing,85.0\n"
        "Support,82.1\n"
    ).encode("utf-8")
    meta = dataset_service.ingest_dataset("dept-session", "dept_scores.csv", custom_csv)

    # Initial plan
    p1 = generate_analysis_plan("Show top 3 departments by score", meta)
    assert p1.group_by[0].column == "Department"
    assert p1.limit == 3
    assert p1.aggregations[0].column == "Score"

    # Turn 2: chart
    p2 = generate_analysis_plan("Visualize this as a bar chart", meta, previous_plan=p1)
    assert p2.requires_chart is True
    assert p2.group_by[0].column == "Department"

    # Turn 3: table
    p3 = generate_analysis_plan("Switch back to a table", meta, previous_plan=p2)
    assert p3.requires_chart is False
    assert p3.chart_type is None
    assert p3.group_by[0].column == "Department"
    assert p3.limit == 3
    assert p3.aggregations[0].column == "Score"
    assert p3.aggregations[0].function == AggregationFunction.SUM

    compiled = compile_plan(p3, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, "dept-session", p3)
    assert res.row_count == 3
    assert res.rows[0]["Department"] == "Engineering"
    assert res.rows[1]["Department"] == "Research"
    assert res.rows[2]["Department"] == "Design"


def test_chat_e2e_three_turn_top_5_then_chart_then_table(store_dataset):
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    # Turn 1: Show me the top 5 products by sales
    res1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "ok"
    assert len(data1["table"]["rows"]) == 5
    assert data1["table"]["rows"][0][0] == "Coffee"

    # Turn 2: Plot this as a bar chart
    res2 = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "ok"
    assert data2.get("chart") is not None
    assert len(data2["table"]["rows"]) == 5

    # Turn 3: Show this as a table
    res3 = client.post("/chat", json={
        "message": "Show this as a table",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["status"] == "ok"
    # Must preserve Top 5 table presentation and NOT fallback to count_rows
    assert data3.get("chart") is None
    assert data3["table"] is not None
    assert data3["table"]["row_count"] == 5
    assert len(data3["table"]["rows"]) == 5
    assert data3["table"]["rows"][0][0] == "Coffee"
    assert data3["table"]["rows"][1][0] == "Milk"
    assert data3["table"]["rows"][2][0] == "Cheese"
    assert data3["table"]["rows"][3][0] == "Chocolate"
    assert data3["table"]["rows"][4][0] == "Apples"
    # Ensure active analysis plan retains Top 5
    plan_dict = data3.get("active_analysis_plan") or _sessions.get(session_id, {}).get("active_analysis_plan")
    assert plan_dict is not None
    assert plan_dict.get("requires_chart") is False
    assert plan_dict.get("limit") == 5
    assert plan_dict["group_by"][0]["column"] == "Product"
    assert plan_dict["aggregations"][0]["column"] == "Total Sales"


def test_repeatability_same_query_multiple_runs(store_dataset):
    meta, session_id = store_dataset
    runs_plans = []
    runs_rows = []

    for _ in range(5):
        plan = generate_analysis_plan("Show me the top 5 products by sales", meta)
        compiled = compile_plan(plan, meta)
        res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
        runs_plans.append(plan)
        runs_rows.append(res.rows)

    # All 5 runs must produce identical plan and identical rows
    for i in range(1, 5):
        assert runs_plans[i].group_by[0].column == runs_plans[0].group_by[0].column == "Product"
        assert runs_plans[i].aggregations[0].column == runs_plans[0].aggregations[0].column == "Total Sales"
        assert runs_plans[i].limit == runs_plans[0].limit == 5
        assert runs_plans[i].order_by[0].descending == runs_plans[0].order_by[0].descending is True
        assert runs_rows[i] == runs_rows[0]


def test_format_clean_metric_label_units():
    """Unit tests for presentation layer clean metric labels."""
    from app.analytics.chart_style import format_clean_metric_label

    assert format_clean_metric_label("sum_Sales") == "Total Sales"
    assert format_clean_metric_label("sum_Total Sales") == "Total Sales"
    assert format_clean_metric_label("sum_Quantity") == "Total Quantity"
    assert format_clean_metric_label("avg_Price") == "Average Price"
    assert format_clean_metric_label("avg_Unit_Price") == "Average Unit Price"
    assert format_clean_metric_label("min_Score") == "Minimum Score"
    assert format_clean_metric_label("max_Revenue") == "Maximum Revenue"
    assert format_clean_metric_label("count_Orders") == "Count of Orders"
    assert format_clean_metric_label("count_distinct_Customer") == "Distinct Customer"
    assert format_clean_metric_label("Product") == "Product"
    assert format_clean_metric_label("Region") == "Region"


def test_metric_label_presentation_layer(store_dataset):
    """
    Validates Dataset Intelligence metric-label presentation invariants:
    1. Table columns display clean user-facing labels ('Total Sales', not 'sum_Sales' or 'sum_Total Sales').
    2. Chart artifact title displays 'Total Sales by Product' (not 'sum_Sales by Product').
    3. Finding narrative references 'Total Sales'.
    4. Numeric values, row order, row count, and internal engine execution remain strictly preserved.
    """
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    # Turn 1: Top 5 products by sales
    res = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

    # Presentation Table Column & Canonical Identifier Verification
    assert data["table"] is not None
    assert data["table"]["display_columns"] == ["Product", "Total Sales"]
    assert "sum_Total Sales" in data["table"]["columns"] or "Total Sales" in data["table"]["columns"]
    assert len(data["table"]["rows"]) == 5
    assert data["table"]["rows"][0] == ["Coffee", 41890.0]
    assert data["table"]["rows"][1] == ["Milk", 35120.5]
    assert data["table"]["rows"][2] == ["Cheese", 27906.3]
    assert data["table"]["rows"][3] == ["Chocolate", 19300.0]
    assert data["table"]["rows"][4] == ["Apples", 17057.41]

    # Narrative Verification
    assert "sum_Sales" not in data["response"]
    assert "sum_Total Sales" not in data["response"]
    assert "Total Sales" in data["response"]

    # Turn 2: Bar chart
    res2 = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "ok"
    assert data2["chart"] is not None
    assert data2["chart"]["title"] == "Total Sales by Product"
    assert data2["table"]["display_columns"] == ["Product", "Total Sales"]

    # Turn 3: Table
    res3 = client.post("/chat", json={
        "message": "Show this as a table",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["table"]["display_columns"] == ["Product", "Total Sales"]
    assert len(data3["table"]["rows"]) == 5
    assert data3["table"]["rows"][0] == ["Coffee", 41890.0]


def test_identifier_preservation_and_display_label_separation(store_dataset):
    """
    Focused regression test ensuring:
    1. Canonical/internal column identifiers (e.g. sum_Total Sales) are preserved in table.columns
       so programmatic contracts and downstream consumers never break.
    2. Friendly presentation labels (e.g. Total Sales) are exposed via table.display_columns
       and applied at the presentation boundary.
    """
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    res = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

    # Canonical columns vs display columns
    cols = data["table"]["columns"]
    display_cols = data["table"]["display_columns"]

    assert cols == ["Product", "sum_Total Sales"]
    assert display_cols == ["Product", "Total Sales"]
    # Check that display_columns does not change data rows
    assert len(data["table"]["rows"]) == 5


def test_mandatory_sorting_regression_scenario(store_dataset):
    """
    Mandatory sorting regression scenario:
    1. 'Show me the top 5 products by sales.'
    2. 'Sort descending.'
    3. 'Sort ascending.'
    4. 'Turn these results into a bar chart.'

    Verifies:
    - Initial request returns 5 highest products in DESC order.
    - DESC follow-up is handled as a valid no-op.
    - ASC follow-up produces ASC rows, order_by descending=False, and matching narrative.
    - Bar chart follow-up preserves ASC order and generates the visualization.
    - Active analysis plan, result rows, and metadata agree.
    """
    meta, session_id = store_dataset
    client = TestClient(app)
    client.cookies.set("app_session_token", sign_session_id(session_id))

    # 1. 'Show me the top 5 products by sales.'
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "ok"
    assert len(d1["table"]["rows"]) == 5
    # Verify strictly decreasing order
    rows1 = d1["table"]["rows"]
    for i in range(len(rows1) - 1):
        assert rows1[i][1] >= rows1[i + 1][1]
    assert rows1[0][0] == "Coffee"
    plan1 = d1.get("active_analysis_plan") or _sessions.get(session_id, {}).get("active_analysis_plan")
    assert plan1 is not None
    assert plan1["order_by"][0]["descending"] is True

    # 2. 'Sort descending.'
    r2 = client.post("/chat", json={
        "message": "Sort descending",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "ok"
    assert len(d2["table"]["rows"]) == 5
    rows2 = d2["table"]["rows"]
    for i in range(len(rows2) - 1):
        assert rows2[i][1] >= rows2[i + 1][1]
    assert rows2[0][0] == "Coffee"
    plan2 = d2.get("active_analysis_plan") or _sessions.get(session_id, {}).get("active_analysis_plan")
    assert plan2 is not None
    assert plan2["order_by"][0]["descending"] is True

    # 3. 'Sort ascending.'
    r3 = client.post("/chat", json={
        "message": "Sort ascending",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["status"] == "ok"
    rows3 = d3["table"]["rows"]
    assert len(rows3) == 5
    # Verify strictly increasing order
    for i in range(len(rows3) - 1):
        assert rows3[i][1] <= rows3[i + 1][1]
    plan3 = d3.get("active_analysis_plan") or _sessions.get(session_id, {}).get("active_analysis_plan")
    assert plan3 is not None
    assert plan3["order_by"][0]["descending"] is False
    # Ensure narrative indicates ascending and does not contradict
    assert "ascending" in d3["response"].lower() or "lowest" in d3["response"].lower()

    # 4. 'Turn these results into a bar chart.'
    r4 = client.post("/chat", json={
        "message": "Turn these results into a bar chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    })
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["status"] == "ok"
    assert d4.get("chart") is not None
    assert d4["chart"]["chart_type"] == "bar"
    # Order remains ascending
    rows4 = d4["table"]["rows"]
    assert len(rows4) == 5
    for i in range(len(rows4) - 1):
        assert rows4[i][1] <= rows4[i + 1][1]
    plan4 = d4.get("active_analysis_plan") or _sessions.get(session_id, {}).get("active_analysis_plan")
    assert plan4 is not None
    assert plan4["order_by"][0]["descending"] is False
    assert plan4["requires_chart"] is True


# ===========================================================================
# 16. REGRESSION TESTS: FILTER FOLLOW-UP PRESERVES GROUPING (BUG 1)
# ===========================================================================

def test_filter_followup_preserves_product_grouping(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Show top 5 products by sales
    p1 = generate_analysis_plan("Show me the top 5 products by Total Sales", meta)
    assert p1.group_by[0].column == "Product"
    assert p1.limit == 5
    assert p1.order_by[0].descending is True

    # Turn 2: Filter to Region North
    p2 = generate_analysis_plan("Filter to Region North", meta, previous_plan=p1)

    # Invariant: Product grouping MUST be preserved, not overwritten by Region!
    assert len(p2.group_by) == 1
    assert p2.group_by[0].column == "Product"
    assert p2.group_by[0].column != "Region"
    # Invariant: Region filter applied
    assert len(p2.filters) == 1
    assert p2.filters[0].column == "Region"
    assert p2.filters[0].operator == FilterOperator.EQ
    assert p2.filters[0].value == "North"
    # Invariant: Limit and order preserved
    assert p2.limit == 5
    assert p2.order_by[0].descending is True

    # Execution check: Returns exactly 3 North products (Cheese, Chocolate, Apples)
    c2 = compile_plan(p2, meta)
    r2 = execute_compiled_query(c2, meta.dataset_id, session_id, p2)
    assert r2.row_count == 3
    prods = [r["Product"] for r in r2.rows]
    assert prods == ["Cheese", "Chocolate", "Apples"]
    assert r2.rows[0]["sum_Total Sales"] == 27906.30

    narr = format_deterministic_result(r2, "Filter to Region North", previous_plan=p1)
    assert "Filtered by Region = 'North'" in narr
    assert "Cheese" in narr


def test_filter_followup_multi_turn_change_and_clear(store_dataset):
    meta, session_id = store_dataset

    p1 = generate_analysis_plan("Show me the top 5 products by Total Sales", meta)
    p2 = generate_analysis_plan("Filter to Region North", meta, previous_plan=p1)
    assert p2.group_by[0].column == "Product"
    assert p2.filters[0].value == "North"

    # Turn 3: Change Region to South
    p3 = generate_analysis_plan("Change Region to South", meta, previous_plan=p2)
    assert p3.group_by[0].column == "Product"
    assert len(p3.filters) == 1
    assert p3.filters[0].column == "Region"
    assert p3.filters[0].value == "South"

    c3 = compile_plan(p3, meta)
    r3 = execute_compiled_query(c3, meta.dataset_id, session_id, p3)
    assert r3.row_count == 3
    prods_south = [r["Product"] for r in r3.rows]
    assert prods_south == ["Milk", "Yogurt", "Bananas"]

    # Turn 4: Clear filters
    p4 = generate_analysis_plan("Clear filters", meta, previous_plan=p3)
    assert p4.group_by[0].column == "Product"
    assert len(p4.filters) == 0


# ===========================================================================
# 17. REGRESSION TESTS: 'AT LEAST' COMPARISON FILTER & BOUNDARY (BUG 2)
# ===========================================================================

ORDERS_BOUNDARY_CSV = (
    "Product,Region,Orders,Total Sales\n"
    "AboveTarget,North,150,20000.0\n"
    "ExactTarget,South,100,10000.0\n"
    "BelowTarget,East,99,5000.0\n"
    "FarBelowTarget,West,40,2000.0\n"
    "HighTarget,North,300,50000.0\n"
).encode("utf-8")


@pytest.fixture
def orders_boundary_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))
    meta = dataset_service.ingest_dataset(
        session_id="boundary-session",
        filename="orders_boundary.csv",
        content_bytes=ORDERS_BOUNDARY_CSV,
    )
    return meta, "boundary-session"


def test_at_least_filter_retained_as_gte(orders_boundary_dataset):
    meta, session_id = orders_boundary_dataset

    q = "Show products with at least 100 orders, by sales"
    plan = generate_analysis_plan(q, meta)

    # Invariants on plan:
    assert plan.group_by[0].column == "Product"
    # Filter on Orders must be present with >= operator
    assert len(plan.filters) == 1
    assert plan.filters[0].column == "Orders"
    assert plan.filters[0].operator == FilterOperator.GTE
    assert plan.filters[0].value in (100, 100.0)

    # Compile and execute query
    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    # Boundary verification:
    # 1. Row with > 100 orders (AboveTarget: 150, HighTarget: 300) must be included
    # 2. Row with EXACTLY 100 orders (ExactTarget: 100) MUST be included
    # 3. Row with 99 orders (BelowTarget: 99) MUST be excluded
    # 4. Row with 40 orders (FarBelowTarget: 40) MUST be excluded
    returned_products = [r["Product"] for r in res.rows]
    assert "HighTarget" in returned_products
    assert "AboveTarget" in returned_products
    assert "ExactTarget" in returned_products, "Row with exactly 100 orders must satisfy 'at least 100' boundary condition!"
    assert "BelowTarget" not in returned_products, "Row with 99 orders must NOT satisfy 'at least 100'!"
    assert "FarBelowTarget" not in returned_products
    assert res.row_count == 3


def test_legitimate_least_ranking_query_not_broken(store_dataset):
    meta, session_id = store_dataset

    # Legitimate least query: "Show products with the least sales"
    q = "Show products with the least sales"
    plan = generate_analysis_plan(q, meta)

    # Invariants on plan:
    assert plan.group_by[0].column == "Product"
    assert len(plan.order_by) == 1
    assert plan.order_by[0].descending is False, "Legitimate 'least' query must sort ascending (lowest first)!"
    assert len(plan.filters) == 0

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)

    # Lowest sales first: Biscuits (3,908.87) then Bananas (6,205.76)
    assert res.rows[0]["Product"] == "Biscuits"
    assert res.rows[0]["sum_Total Sales"] == 3908.87
    assert res.rows[1]["Product"] == "Bananas"

    narr = format_deterministic_result(res, q)
    assert "Biscuits has the lowest" in narr
    assert "3,908.87" in narr


def test_legitimate_least_n_ranking_query(store_dataset):
    meta, session_id = store_dataset

    # Legitimate least query with limit: "Show least 3 products by sales"
    q = "Show least 3 products by sales"
    plan = generate_analysis_plan(q, meta)

    assert plan.group_by[0].column == "Product"
    assert plan.limit == 3
    assert plan.order_by[0].descending is False
    assert len(plan.filters) == 0

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    assert res.row_count == 3
    assert [r["Product"] for r in res.rows] == ["Biscuits", "Bananas", "Bread"]


# ===========================================================================
# 18. OPTION B REGRESSION TESTS: SEPARATE COHORT SELECTION FROM DISPLAY SORT
# ===========================================================================

def test_option_b_top_5_sort_ascending_preserves_cohort_membership_and_reverses_order(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Show top 5 products by Total Sales
    q1 = "Show me the top 5 products by Total Sales"
    p1 = generate_analysis_plan(q1, meta)
    assert len(p1.selection_order_by) == 1
    assert p1.selection_order_by[0].descending is True
    assert p1.order_by[0].descending is True
    assert p1.limit == 5

    c1 = compile_plan(p1, meta)
    r1 = execute_compiled_query(c1, meta.dataset_id, session_id, p1)
    # The true top 5 products in dataset in DESC order:
    expected_top5 = ["Coffee", "Milk", "Cheese", "Chocolate", "Apples"]
    assert [r["Product"] for r in r1.rows] == expected_top5

    # Turn 2: Sort ascending
    q2 = "Sort ascending"
    p2 = generate_analysis_plan(q2, meta, previous_plan=p1)

    # Invariants on plan schema:
    # 1. selection_order_by MUST remain DESC (preserves Top 5 cohort)
    assert len(p2.selection_order_by) == 1
    assert p2.selection_order_by[0].descending is True
    assert p2.selection_order_by[0].target in ("sum_Total Sales", "Total Sales")
    # 2. order_by MUST be ASC (presentation ordering)
    assert len(p2.order_by) == 1
    assert p2.order_by[0].descending is False
    assert p2.limit == 5

    # Compilation invariant: CTE must be used to enforce selection before display ordering
    c2 = compile_plan(p2, meta)
    assert "WITH selected_cohort AS" in c2.sql
    assert "LIMIT 5" in c2.sql

    r2 = execute_compiled_query(c2, meta.dataset_id, session_id, p2)
    assert r2.row_count == 5

    # Row identity invariant: Exact same Top 5 products
    assert set(r["Product"] for r in r2.rows) == set(expected_top5)
    # Display order invariant: Strictly ascending
    expected_asc_top5 = ["Apples", "Chocolate", "Cheese", "Milk", "Coffee"]
    assert [r["Product"] for r in r2.rows] == expected_asc_top5
    sales_values = [r["sum_Total Sales"] for r in r2.rows]
    for i in range(len(sales_values) - 1):
        assert sales_values[i] <= sales_values[i + 1]


def test_option_b_bottom_5_sort_descending_preserves_cohort_membership_and_reverses_order(store_dataset):
    meta, session_id = store_dataset

    # Turn 1: Show bottom 5 products by Total Sales
    qb1 = "Show me the bottom 5 products by Total Sales"
    pb1 = generate_analysis_plan(qb1, meta)
    assert len(pb1.selection_order_by) == 1
    assert pb1.selection_order_by[0].descending is False
    assert pb1.order_by[0].descending is False
    assert pb1.limit == 5

    cb1 = compile_plan(pb1, meta)
    rb1 = execute_compiled_query(cb1, meta.dataset_id, session_id, pb1)
    expected_bottom5 = ["Biscuits", "Bananas", "Bread", "Yogurt", "Tea"]
    assert [r["Product"] for r in rb1.rows] == expected_bottom5

    # Turn 2: Sort descending
    qb2 = "Sort descending"
    pb2 = generate_analysis_plan(qb2, meta, previous_plan=pb1)

    # Invariants on plan schema:
    # 1. selection_order_by MUST remain ASC (preserves Bottom 5 cohort)
    assert len(pb2.selection_order_by) == 1
    assert pb2.selection_order_by[0].descending is False
    # 2. order_by MUST be DESC (presentation ordering)
    assert len(pb2.order_by) == 1
    assert pb2.order_by[0].descending is True
    assert pb2.limit == 5

    cb2 = compile_plan(pb2, meta)
    assert "WITH selected_cohort AS" in cb2.sql

    rb2 = execute_compiled_query(cb2, meta.dataset_id, session_id, pb2)
    assert rb2.row_count == 5

    # Row identity invariant: Exact same Bottom 5 products
    assert set(r["Product"] for r in rb2.rows) == set(expected_bottom5)
    # Display order invariant: Strictly descending
    expected_desc_bottom5 = ["Tea", "Yogurt", "Bread", "Bananas", "Biscuits"]
    assert [r["Product"] for r in rb2.rows] == expected_desc_bottom5
    sales_values = [r["sum_Total Sales"] for r in rb2.rows]
    for i in range(len(sales_values) - 1):
        assert sales_values[i] >= sales_values[i + 1]


def test_option_b_chart_table_toggle_preserves_cohort_and_order(store_dataset):
    meta, session_id = store_dataset

    p1 = generate_analysis_plan("Show me the top 5 products by Total Sales", meta)
    p2 = generate_analysis_plan("Sort ascending", meta, previous_plan=p1)

    # Turn 3: Plot this as a bar chart
    pc = generate_analysis_plan("Plot this as a bar chart", meta, previous_plan=p2)
    assert pc.requires_chart is True
    assert [o.model_dump() for o in pc.selection_order_by] == [o.model_dump() for o in p2.selection_order_by]
    assert [o.model_dump() for o in pc.order_by] == [o.model_dump() for o in p2.order_by]

    rc = execute_compiled_query(compile_plan(pc, meta), meta.dataset_id, session_id, pc)
    assert [r["Product"] for r in rc.rows] == ["Apples", "Chocolate", "Cheese", "Milk", "Coffee"]

    # Turn 4: Show table
    pt = generate_analysis_plan("Show table", meta, previous_plan=pc)
    assert pt.requires_chart is False
    assert [o.model_dump() for o in pt.selection_order_by] == [o.model_dump() for o in p2.selection_order_by]
    assert [o.model_dump() for o in pt.order_by] == [o.model_dump() for o in p2.order_by]
    rt = execute_compiled_query(compile_plan(pt, meta), meta.dataset_id, session_id, pt)
    assert [r["Product"] for r in rt.rows] == ["Apples", "Chocolate", "Cheese", "Milk", "Coffee"]


def test_option_b_metric_swap_reranks_cohort_by_new_metric(store_dataset):
    meta, session_id = store_dataset

    p1 = generate_analysis_plan("Show me the top 5 products by Total Sales", meta)

    # Turn 2: Use profit instead
    pm = generate_analysis_plan("Use profit instead", meta, previous_plan=p1)
    cm = compile_plan(pm, meta)
    rm = execute_compiled_query(cm, meta.dataset_id, session_id, pm)

    assert pm.aggregations[0].column == "Profit"
    assert rm.row_count == 5
    # In our dataset, Profit top 5:
    # Coffee (6800.0), Milk (5100.0), Cheese (4200.0), Chocolate (3100.0), Apples (2100.0)
    expected_profit_top5 = ["Coffee", "Milk", "Cheese", "Chocolate", "Apples"]
    assert [r["Product"] for r in rm.rows] == expected_profit_top5


def test_option_b_legacy_plan_without_selection_order_by(store_dataset):
    meta, session_id = store_dataset

    # Simulates loading a legacy serialized plan from SQLite that lacks selection_order_by
    legacy_plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="Product")],
        aggregations=[AggregationClause(column="Total Sales", function=AggregationFunction.SUM, alias="sum_Total Sales")],
        order_by=[OrderByClause(target="sum_Total Sales", descending=True)],
        limit=5,
    )
    assert legacy_plan.selection_order_by == []

    c_legacy = compile_plan(legacy_plan, meta)
    # Must emit standard single-level query without CTE
    assert "WITH selected_cohort AS" not in c_legacy.sql
    r_legacy = execute_compiled_query(c_legacy, meta.dataset_id, session_id, legacy_plan)
    assert [r["Product"] for r in r_legacy.rows] == ["Coffee", "Milk", "Cheese", "Chocolate", "Apples"]


def test_option_b_generic_limit_without_ranking_intent(store_dataset):
    meta, _ = store_dataset

    # Generic limit: e.g. "What is total sales by Region? Limit 3"
    q_generic = "What is total sales by Region? Limit 3"
    p_generic = generate_analysis_plan(q_generic, meta)

    # Must NOT infer selection_order_by without ranking intent
    assert p_generic.selection_order_by == []
    assert p_generic.limit == 3

    c_generic = compile_plan(p_generic, meta)
    assert "WITH selected_cohort AS" not in c_generic.sql
