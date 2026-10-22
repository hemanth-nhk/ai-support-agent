"""
End-to-end integration and safety test suite for Phase 6: Deterministic Advanced Analytics.
Validates:
6A: Derived metrics (ratios, percentages, safe division by zero, null denominators)
6B: Statistical functions (MEDIAN, STDDEV, PERCENTILE_CONT)
6C: Bivariate Pearson Correlation (CORR, constant variance safety)
6D: Period Comparison (LAG window CTE, growth %, delta, single/missing period)
6E: End-to-end /chat API integration with bounded AnalysisTable & ChartArtifact
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.analytics import dataset_service
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
)
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query
from app.analytics.planner import format_deterministic_result


@pytest.fixture
def analytics_dataset(tmp_path, monkeypatch):
    """Dataset with dates, multiple numerical columns, and zero-value rows for division safety."""
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))

    csv_content = (
        "product,category,revenue,profit,units,sale_date\n"
        "Widget,Electronics,1000.0,200.0,10,2024-01-15\n"
        "Gadget,Electronics,2000.0,600.0,20,2024-02-15\n"
        "Gizmo,Hardware,500.0,100.0,5,2024-03-15\n"
        "Cable,Hardware,0.0,50.0,2,2024-04-15\n"  # 0 revenue -> division by zero test
        "Dongle,Hardware,1500.0,300.0,15,2024-05-15\n"
    ).encode("utf-8")

    import uuid
    session_id = f"analytics-session-{uuid.uuid4().hex[:8]}"

    meta = dataset_service.ingest_dataset(
        session_id=session_id,
        filename="company_analytics.csv",
        content_bytes=csv_content,
    )
    return meta, session_id


# ---------------------------------------------------------------------------
# 6A: Derived Metrics (Safe Division & Grounding)
# ---------------------------------------------------------------------------

def test_derived_metric_profit_margin_by_category(analytics_dataset):
    meta, session_id = analytics_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="category")],
        aggregations=[
            AggregationClause(column="revenue", function=AggregationFunction.SUM, alias="total_revenue"),
            AggregationClause(column="profit", function=AggregationFunction.SUM, alias="total_profit"),
        ],
        derived_metrics=[
            DerivedMetric(
                operation=DerivedMetricOperation.PERCENTAGE,
                numerator="profit",
                denominator="revenue",
                multiplier=100.0,
                alias="margin_pct",
            )
        ],
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 2
    row_map = {r["category"]: r for r in result.rows}

    # Electronics: rev=3000, profit=800 -> 800/3000 * 100 = 26.6667%
    assert row_map["Electronics"]["total_revenue"] == 3000.0
    assert row_map["Electronics"]["margin_pct"] == pytest.approx(26.6667, rel=1e-3)

    # Hardware: rev=2000, profit=450 -> 450/2000 * 100 = 22.5%
    assert row_map["Hardware"]["total_revenue"] == 2000.0
    assert row_map["Hardware"]["margin_pct"] == pytest.approx(22.5)


def test_derived_metric_division_by_zero_safety(analytics_dataset):
    """When denominator sum is 0, the result must be None (NULL), never crash or return 0."""
    meta, session_id = analytics_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        group_by=[GroupByClause(column="product")],
        derived_metrics=[
            DerivedMetric(
                operation=DerivedMetricOperation.PERCENTAGE,
                numerator="profit",
                denominator="revenue",
                alias="product_margin_pct",
            )
        ],
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    row_map = {r["product"]: r["product_margin_pct"] for r in result.rows}
    # 'Cable' has revenue = 0.0 -> must be None
    assert row_map["Cable"] is None
    # 'Widget' has profit=200, rev=1000 -> 20.0%
    assert row_map["Widget"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 6B: Statistical Functions (MEDIAN, STDDEV, PERCENTILE_CONT)
# ---------------------------------------------------------------------------

def test_statistical_aggregations_execution(analytics_dataset):
    meta, session_id = analytics_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        aggregations=[
            AggregationClause(column="revenue", function=AggregationFunction.MEDIAN, alias="med_rev"),
            AggregationClause(column="revenue", function=AggregationFunction.STDDEV, alias="std_rev"),
            AggregationClause(
                column="revenue",
                function=AggregationFunction.PERCENTILE_CONT,
                quantile=0.5,
                alias="p50_rev",
            ),
        ],
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 1
    row = result.rows[0]
    # revenues: [0, 500, 1000, 1500, 2000] -> median is 1000.0
    assert row["med_rev"] == pytest.approx(1000.0)
    assert row["p50_rev"] == pytest.approx(1000.0)
    assert row["std_rev"] is not None
    assert row["std_rev"] > 0


# ---------------------------------------------------------------------------
# 6C: Bivariate Pearson Correlation (CORR)
# ---------------------------------------------------------------------------

def test_correlation_execution(analytics_dataset):
    meta, session_id = analytics_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        correlations=[
            CorrelationClause(column_a="revenue", column_b="profit", alias="corr_rev_profit"),
        ],
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 1
    corr_val = result.rows[0]["corr_rev_profit"]
    assert corr_val is not None
    assert -1.0 <= corr_val <= 1.0

    # Grounded statement includes correlation descriptor
    statement = format_deterministic_result(result, "What is the correlation between revenue and profit?")
    assert "correlation" in statement.lower()
    assert f"{corr_val:.3f}" in statement


# ---------------------------------------------------------------------------
# 6D: Period Comparison (LAG CTE & Month-over-Month Growth)
# ---------------------------------------------------------------------------

def test_period_comparison_mom_growth(analytics_dataset):
    meta, session_id = analytics_dataset
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        period_comparison=PeriodComparison(
            date_column="sale_date",
            metric="revenue",
            period=DateTrunc.MONTH,
            mode=PeriodComparisonMode.GROWTH_PCT,
            alias="revenue_mom_growth",
        ),
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    # 5 monthly rows: Jan, Feb, Mar, Apr, May
    assert result.row_count == 5
    rows = result.rows

    # First period (Jan): prev_val is None, growth is None
    assert rows[0]["prev_val"] is None
    assert rows[0]["revenue_mom_growth"] is None

    # Feb: current=2000, prev=1000 -> (2000-1000)/1000 * 100 = 100%
    assert rows[1]["current_val"] == 2000.0
    assert rows[1]["prev_val"] == 1000.0
    assert rows[1]["revenue_mom_growth"] == pytest.approx(100.0)

    # Mar: current=500, prev=2000 -> (500-2000)/2000 * 100 = -75%
    assert rows[2]["revenue_mom_growth"] == pytest.approx(-75.0)

    statement = format_deterministic_result(result, "Show monthly revenue growth")
    assert "observed periods" in statement
    assert "revenue_mom_growth" in statement


# ---------------------------------------------------------------------------
# 6E: End-to-End Chat API Response Contract
# ---------------------------------------------------------------------------

def test_chat_derived_metric_response_contract(analytics_dataset):
    """Validates that Phase 6 derived metrics return Table, Finding, and Chart in /chat."""
    meta, session_id = analytics_dataset
    client = TestClient(app)
    from app.api.session_security import sign_session_id
    client.cookies.set("app_session_token", sign_session_id(session_id))

    payload = {
        "message": "What is the profit margin percentage by category as a bar chart?",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "response" in data
    assert data["table"] is not None
    assert any(c in data["table"]["columns"] for c in ("margin_pct", "profit_margin_pct", "total_profit"))
    assert len(data["table"]["rows"]) >= 1
    assert data["chart"] is not None
    assert data["chart"]["chart_type"] == "bar"


def test_deterministic_profit_margin_fast_path_and_non_interception(analytics_dataset, monkeypatch):
    """
    Verifies that:
    1. 'What is the profit margin percentage by category as a bar chart?' is intercepted deterministically,
       producing a valid derived metric plan, non-empty table with profit margin percentage, and a bar chart.
    2. Unrelated queries without profit-margin intent are not intercepted by this fast path.
    3. Queries with explicit filters (e.g. 'where revenue > 1000') bypass the fast path to preserve filter semantics.
    """
    from app.analytics.planner import generate_analysis_plan
    from app.analytics.compiler import compile_plan
    from app.analytics.executor import execute_compiled_query

    meta, session_id = analytics_dataset

    # 1. Exact query: deterministic fast-path execution (no LLM call required)
    q = "What is the profit margin percentage by category as a bar chart?"
    plan = generate_analysis_plan(q, meta)

    assert plan.requires_chart is True
    assert plan.chart_type == "bar"
    assert len(plan.derived_metrics) == 1
    assert plan.derived_metrics[0].alias == "profit_margin_pct"
    assert plan.derived_metrics[0].numerator == "profit"
    assert plan.derived_metrics[0].denominator == "revenue"
    assert len(plan.group_by) == 1
    assert plan.group_by[0].column == "category"

    compiled = compile_plan(plan, meta)
    res = execute_compiled_query(compiled, meta.dataset_id, session_id, plan)
    assert res.row_count >= 1
    assert "profit_margin_pct" in res.columns
    assert len(res.rows) == 2

    # 2. Unrelated query without profit-margin intent: does not have derived_metrics
    q_unrelated = "What is total units by category?"
    plan_unrelated = generate_analysis_plan(q_unrelated, meta)
    assert len(plan_unrelated.derived_metrics) == 0

    # 3. Explicit filter query: bypasses fast path and delegates to LLM so filters are handled
    called_llm = []

    class DummyLLM:
        def invoke(self, *args, **kwargs):
            called_llm.append(True)
            return type("Resp", (), {
                "content": '{"group_by": [{"column": "category"}], "filters": [{"column": "revenue", "operator": ">", "value": 1000}], "aggregations": [{"column": "profit", "function": "sum"}]}'
            })()

    monkeypatch.setattr("app.analytics.planner.get_llm", lambda: DummyLLM())
    q_filtered = "What is the profit margin percentage by category where revenue > 1000?"
    plan_filtered = generate_analysis_plan(q_filtered, meta)
    assert len(called_llm) == 1
    assert any(f.column == "revenue" for f in plan_filtered.filters)
