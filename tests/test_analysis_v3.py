"""
End-to-end integration and grounded findings test suite for Phase 7:
Windowed Analytics, Cumulative Metrics, Outliers, and Quantiles.
Validates:
7A: Planner -> Compiler -> Executor integration for rolling windows, cumulative totals, outliers, quantiles
7B: Grounded natural-language finding generation (zero hallucination, exact values from DuckDB)
7C: Zero-variance and single-observation grounded explanations
7D: End-to-end /chat API integration with bounded AnalysisTable & ChartArtifact
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.analytics import dataset_service
from app.analytics.schemas import (
    AnalysisPlan,
    RollingWindowSpec,
    WindowFunction,
    CumulativeSpec,
    CumulativeFunction,
    OutlierSpec,
    QuantileSpec,
)
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query
from app.analytics.planner import generate_analysis_plan, format_deterministic_result


@pytest.fixture
def phase7_dataset(tmp_path, monkeypatch):
    """Dataset with chronological dates, categories, variance, and clear outliers."""
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))

    csv_content = (
        "transaction_id,department,amount,tx_date\n"
        "TX1,Sales,100.0,2026-01-01\n"
        "TX2,Sales,120.0,2026-01-02\n"
        "TX3,Sales,110.0,2026-01-03\n"
        "TX4,Sales,105.0,2026-01-04\n"
        "TX5,Sales,1000.0,2026-01-05\n"  # Clear outlier in Sales
        "TX6,Marketing,50.0,2026-01-01\n"
        "TX7,Marketing,50.0,2026-01-02\n"
        "TX8,Marketing,50.0,2026-01-03\n"  # Zero variance in Marketing (all 50.0)
    ).encode("utf-8")

    meta = dataset_service.ingest_dataset(
        session_id="phase7-session",
        filename="company_tx.csv",
        content_bytes=csv_content,
    )
    return meta, "phase7-session"


# ---------------------------------------------------------------------------
# Planner -> Compiler -> Grounded Result Tests
# ---------------------------------------------------------------------------

def test_planner_and_grounded_result_rolling_window(phase7_dataset):
    meta, session_id = phase7_dataset
    query = "What is the 3-day rolling average amount over time?"
    plan = generate_analysis_plan(query, meta)

    assert len(plan.rolling_windows) == 1
    assert plan.rolling_windows[0].function == WindowFunction.AVG
    assert plan.rolling_windows[0].column == "amount"
    assert plan.rolling_windows[0].preceding == 2

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 8
    assert "rolling_avg_amount" in result.columns

    statement = format_deterministic_result(result, query)
    assert "rolling avg of 'amount'" in statement
    assert "peaked at" in statement


def test_planner_and_grounded_result_cumulative_metric(phase7_dataset):
    meta, session_id = phase7_dataset
    query = "Show cumulative amount over time"
    plan = generate_analysis_plan(query, meta)

    assert len(plan.cumulative_metrics) == 1
    assert plan.cumulative_metrics[0].function == CumulativeFunction.SUM
    assert plan.cumulative_metrics[0].column == "amount"

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 8
    assert "cumulative_sum_amount" in result.columns

    statement = format_deterministic_result(result, query)
    assert "cumulative sum of amount reached" in statement


def test_planner_and_grounded_result_outlier_detection(phase7_dataset):
    meta, session_id = phase7_dataset
    query = "Find transactions with amount > 2.0 standard deviations from the mean"
    plan = generate_analysis_plan(query, meta)

    assert plan.outlier_analysis is not None
    assert plan.outlier_analysis.column == "amount"
    assert plan.outlier_analysis.threshold == 2.0
    assert plan.outlier_analysis.filter_outliers_only is True

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 1
    assert result.rows[0]["transaction_id"] == "TX5"
    assert result.rows[0]["amount"] == 1000.0
    assert result.rows[0]["is_outlier"] == True

    statement = format_deterministic_result(result, query)
    assert "Identified 1 statistical outlier in 'amount'" in statement
    assert "exceeding 2.0 standard deviations" in statement
    assert "1000" in statement


def test_planner_and_grounded_result_quantile_segmentation(phase7_dataset):
    meta, session_id = phase7_dataset
    query = "Segment transactions into 4 quartiles based on amount"
    plan = generate_analysis_plan(query, meta)

    assert len(plan.quantiles) == 1
    assert plan.quantiles[0].column == "amount"
    assert plan.quantiles[0].num_quantiles == 4

    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 8
    assert "quantile_4_amount" in result.columns

    statement = format_deterministic_result(result, query)
    assert "Segmented 8 records into 4 quartiles based on 'amount'" in statement


def test_grounded_result_zero_variance_explanation(phase7_dataset):
    """When analyzing outliers in a zero-variance partition or subset, grounded finding explains it clearly."""
    meta, session_id = phase7_dataset

    # Plan with filter department == 'Marketing' (all 50.0) and filter_outliers_only = False
    from app.analytics.schemas import FilterClause, FilterOperator
    plan = AnalysisPlan(
        dataset_id=meta.dataset_id,
        filters=[FilterClause(column="department", operator=FilterOperator.EQ, value="Marketing")],
        outlier_analysis=OutlierSpec(column="amount", filter_outliers_only=False),
    )
    compiled = compile_plan(plan, meta)
    result = execute_compiled_query(compiled, dataset_id=meta.dataset_id, session_id=session_id, plan=plan)

    assert result.row_count == 3
    statement = format_deterministic_result(result, "Check outliers in Marketing")
    assert "zero variance or insufficient observations" in statement


# ---------------------------------------------------------------------------
# End-to-End Chat API Response Contract Tests
# ---------------------------------------------------------------------------

def test_chat_rolling_window_response_contract(phase7_dataset, client):
    meta, session_id = phase7_dataset

    payload = {
        "message": "Plot 3-day rolling average amount as a line chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "response" in data
    assert "rolling" in data["response"].lower()
    assert data["table"] is not None
    assert "rolling_avg_amount" in data["table"]["columns"]
    assert len(data["table"]["rows"]) >= 1
    assert data["chart"] is not None
    assert data["chart"]["chart_type"] == "line"


def test_chat_outlier_response_contract(phase7_dataset, client):
    meta, session_id = phase7_dataset

    payload = {
        "message": "Show transactions with amount > 2.0 standard deviations from the mean",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "response" in data
    assert "outlier" in data["response"].lower()
    assert data["table"] is not None
    assert "is_outlier" in data["table"]["columns"]
    assert len(data["table"]["rows"]) == 1
    assert data["table"]["rows"][0][0] == "TX5"
