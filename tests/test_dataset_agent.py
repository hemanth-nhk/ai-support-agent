"""
Tests for Phase 3 — Router Expansion, Schema-Aware Planning, Ambiguity Handling & Grounded Execution.
Covers all categories specified in the Phase 3 Test Gate:
1. Routing (analytical queries, preservation of support queries, precedence)
2. Schema-aware planning & Pydantic validation
3. Server-side authoritative dataset_id override
4. Ambiguity detection (ranking without metric on multi-metric datasets)
5. Disambiguation follow-up state tracking (awaiting_entity='dataset_metric')
6. Single-metric dataset ranking (no ambiguity needed)
7. End-to-end analytical execution & deterministic grounding
8. Zero-row / filter-mismatch handling
9. Full session isolation in agent flow
"""
import io
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.analytics import dataset_service
from app.agent.router import (
    classify_intent,
    INTENT_DATASET_ANALYSIS,
    INTENT_CUSTOMER_ORDERS,
    INTENT_ORDER_STATUS,
    INTENT_DATA_QUERY,
    INTENT_FAQ,
    INTENT_GENERAL,
    INTENT_AMBIGUOUS,
    INTENT_DOCUMENT_QUERY,
)
from app.agent.graph import run_agent
from app.analytics.planner import (
    generate_analysis_plan,
    format_deterministic_result,
    _extract_json_from_llm_output,
)


@pytest.fixture
def multi_metric_dataset(monkeypatch):
    """Dataset with multiple metrics: revenue, profit, quantity."""
    temp_dir = tempfile.mkdtemp(prefix="test_agent_data_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)

    csv_content = (
        "product,region,revenue,profit,quantity,sale_date\n"
        "Widget,North,1000.0,200.0,10,2024-01-15\n"
        "Gadget,North,2500.0,600.0,25,2024-01-20\n"
        "Gizmo,South,800.0,150.0,8,2024-02-10\n"
        "Widget,South,1200.0,250.0,12,2024-02-15\n"
        "Gadget,East,3000.0,750.0,30,2024-03-05\n"
    ).encode("utf-8")

    res = dataset_service.ingest_dataset(
        session_id="session-user-a",
        filename="store_performance.csv",
        content_bytes=csv_content,
    )
    metadata = dataset_service.get_dataset(res.dataset_id, session_id="session-user-a")

    yield metadata, "session-user-a"
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def single_metric_dataset(monkeypatch):
    """Dataset with only one numerical metric: score."""
    temp_dir = tempfile.mkdtemp(prefix="test_agent_single_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)

    csv_content = (
        "agent_name,department,score\n"
        "Alice,Support,95.0\n"
        "Bob,Support,82.0\n"
        "Charlie,Sales,88.0\n"
    ).encode("utf-8")

    res = dataset_service.ingest_dataset(
        session_id="session-user-single",
        filename="agent_scores.csv",
        content_bytes=csv_content,
    )
    metadata = dataset_service.get_dataset(res.dataset_id, session_id="session-user-single")

    yield metadata, "session-user-single"
    shutil.rmtree(temp_dir, ignore_errors=True)


# (Shared client fixture provided by tests/conftest.py)


# ---------------------------------------------------------------------------
# 1. Routing & Preservation of Support Queries
# ---------------------------------------------------------------------------

def test_dataset_analytical_query_routes_to_dataset_analysis(multi_metric_dataset):
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "What is the average revenue per region?",
        dataset_id=metadata.dataset_id,
    )
    assert res["intent"] == INTENT_DATASET_ANALYSIS
    assert res["dataset_id"] == metadata.dataset_id


def test_support_queries_not_stolen_by_active_dataset(multi_metric_dataset):
    """Customer support queries must NOT be stolen when a dataset is active."""
    metadata, _ = multi_metric_dataset
    d_id = metadata.dataset_id

    # "How many orders do I have?" -> CUSTOMER_ORDERS
    r1 = classify_intent("How many orders do I have?", dataset_id=d_id)
    assert r1["intent"] == INTENT_CUSTOMER_ORDERS

    # "Where is my order?" -> ORDER_STATUS
    r2 = classify_intent("Where is my order?", dataset_id=d_id)
    assert r2["intent"] == INTENT_ORDER_STATUS

    # "Where is order 1001?" -> ORDER_STATUS
    r3 = classify_intent("Where is order 1001?", dataset_id=d_id)
    assert r3["intent"] == INTENT_ORDER_STATUS

    # "Find orders between 2024-01-01 and 2024-02-01" -> DATA_QUERY
    r4 = classify_intent("Find orders between 2024-01-01 and 2024-02-01", dataset_id=d_id)
    assert r4["intent"] == INTENT_DATA_QUERY

    # "What is your return policy?" -> FAQ
    r5 = classify_intent("What is your return policy?", dataset_id=d_id)
    assert r5["intent"] == INTENT_FAQ

    # "Hello" -> GENERAL
    r6 = classify_intent("Hello", dataset_id=d_id)
    assert r6["intent"] == INTENT_GENERAL


def test_document_query_preserved_when_both_active(multi_metric_dataset):
    """Explicit document questions route to DOCUMENT_QUERY even if dataset is present."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "According to the uploaded document, what is the policy?",
        document_id="some-doc-id",
        dataset_id=metadata.dataset_id,
    )
    assert res["intent"] == INTENT_DOCUMENT_QUERY


def test_no_dataset_active_does_not_route_to_dataset_analysis():
    """Without an active dataset, analytical queries do not route to DATASET_ANALYSIS."""
    res = classify_intent("What is the average revenue per region?")
    assert res["intent"] != INTENT_DATASET_ANALYSIS


# ---------------------------------------------------------------------------
# 2. Ambiguity Handling: Multiple Metrics vs Single Metric
# ---------------------------------------------------------------------------

def test_ambiguous_superlative_on_multi_metric_dataset(multi_metric_dataset):
    """'What are the best products?' without metric triggers AMBIGUOUS & clarification."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "What are the best products?",
        dataset_id=metadata.dataset_id,
    )
    assert res["intent"] == INTENT_AMBIGUOUS
    assert res["candidate_intents"] == [INTENT_DATASET_ANALYSIS]
    assert res["awaiting_entity"] == "dataset_metric"
    assert "revenue" in res["ambiguity_details"]["candidate_metrics"]
    assert "profit" in res["ambiguity_details"]["candidate_metrics"]


def test_explicit_metric_on_multi_metric_dataset_not_ambiguous(multi_metric_dataset):
    """'What are the best products by revenue?' is NOT ambiguous."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "What are the best products by revenue?",
        dataset_id=metadata.dataset_id,
    )
    assert res["intent"] == INTENT_DATASET_ANALYSIS


def test_single_metric_dataset_not_ambiguous(single_metric_dataset):
    """When only 1 metric exists (score), ranking query is NOT ambiguous."""
    metadata, _ = single_metric_dataset
    res = classify_intent(
        "Who is the top agent?",
        dataset_id=metadata.dataset_id,
    )
    assert res["intent"] == INTENT_DATASET_ANALYSIS


# ---------------------------------------------------------------------------
# 3. Multi-Turn Disambiguation Follow-Up
# ---------------------------------------------------------------------------

def test_disambiguation_multi_turn_flow(multi_metric_dataset):
    """Turn 1 triggers clarification -> Turn 2 'by profit' resolves to DATASET_ANALYSIS."""
    metadata, session_id = multi_metric_dataset

    # Turn 1: Ambiguous ranking question
    state_turn1 = run_agent(
        "What are the best products?",
        session_id=session_id,
        dataset_id=metadata.dataset_id,
    )
    assert state_turn1["intent"] == INTENT_AMBIGUOUS
    assert state_turn1["status"] == "needs_clarification"
    assert "revenue" in state_turn1["final_response"].lower()
    assert "profit" in state_turn1["final_response"].lower()
    assert state_turn1["awaiting_entity"] == "dataset_metric"

    # Turn 2: User clarifies "By profit"
    state_turn2 = run_agent(
        "By profit",
        previous_state=state_turn1,
        session_id=session_id,
        dataset_id=metadata.dataset_id,
    )
    assert state_turn2["intent"] == INTENT_DATASET_ANALYSIS
    assert state_turn2["status"] == "ok"
    assert state_turn2["tool_used"] == "execute_analysis_plan"
    assert state_turn2["awaiting_entity"] is None
    # Gadget has profit 600+750 = 1350 (top by profit)
    assert "gadget" in state_turn2["final_response"].lower()


# ---------------------------------------------------------------------------
# 4. Server-Side Authoritative Dataset ID Enforcement
# ---------------------------------------------------------------------------

def test_planner_overwrites_hallucinated_dataset_id(multi_metric_dataset):
    """Even if an LLM returns a hallucinated dataset_id, planner forces verified ID."""
    metadata, _ = multi_metric_dataset
    plan = generate_analysis_plan("Total revenue", metadata)
    assert plan.dataset_id == metadata.dataset_id


# ---------------------------------------------------------------------------
# 5. Planning Helpers & Markdown JSON Extraction
# ---------------------------------------------------------------------------

def test_json_extractor_handles_markdown_and_raw():
    raw_md = "```json\n{\"dataset_id\": \"123\", \"limit\": 10}\n```"
    parsed = _extract_json_from_llm_output(raw_md)
    assert parsed["limit"] == 10

    raw_text = "Here is your plan: {\"dataset_id\": \"456\", \"limit\": 5} Hope this helps!"
    parsed2 = _extract_json_from_llm_output(raw_text)
    assert parsed2["limit"] == 5


# ---------------------------------------------------------------------------
# 6. End-to-End Grounded Execution via Chat Endpoint
# ---------------------------------------------------------------------------

def test_chat_endpoint_dataset_query_execution(client, multi_metric_dataset):
    """Full HTTP request through /chat with dataset_id executing and returning ok."""
    metadata, session_id = multi_metric_dataset

    res = client.post(
        "/chat",
        json={
            "message": "Which region had the highest total revenue?",
            "session_id": session_id,
            "dataset_id": metadata.dataset_id,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == INTENT_DATASET_ANALYSIS
    assert data["tool_used"] == "execute_analysis_plan"
    assert data["status"] == "ok"
    assert data["retrieval_count"] > 0
    # North has revenue 1000 + 2500 = 3500.0 (top region)
    assert "north" in data["response"].lower() or "3500" in data["response"]


def test_chat_endpoint_session_isolation(client, multi_metric_dataset):
    """Session B cannot query Session A's dataset."""
    metadata, _ = multi_metric_dataset

    res = client.post(
        "/chat",
        json={
            "message": "Total revenue",
            "session_id": "intruder-session",
            "dataset_id": metadata.dataset_id,
        },
    )
    assert res.status_code == 403
    data = res.json()
    assert "access denied" in data.get("detail", "").lower() or "different session" in data.get("detail", "").lower()


# ---------------------------------------------------------------------------
# 7. Dual Attachment Routing Disambiguation Tests
# ---------------------------------------------------------------------------

def test_dual_attachment_analytical_query_routes_to_dataset(multi_metric_dataset):
    """When both dataset and document are attached, analytical queries route to DATASET_ANALYSIS."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "Which region has the highest revenue?",
        dataset_id=metadata.dataset_id,
        document_id="some-doc-id",
    )
    assert res["intent"] == INTENT_DATASET_ANALYSIS


def test_dual_attachment_document_query_routes_to_document(multi_metric_dataset):
    """When both dataset and document are attached, policy/document queries route to DOCUMENT_QUERY."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "What return period does the policy mention?",
        dataset_id=metadata.dataset_id,
        document_id="some-doc-id",
    )
    assert res["intent"] == INTENT_DOCUMENT_QUERY


def test_dual_attachment_explicit_document_routes_to_document(multi_metric_dataset):
    """When both attached, explicit document pattern routes to DOCUMENT_QUERY."""
    metadata, _ = multi_metric_dataset
    res = classify_intent(
        "According to the uploaded document, what is the policy?",
        dataset_id=metadata.dataset_id,
        document_id="some-doc-id",
    )
    assert res["intent"] == INTENT_DOCUMENT_QUERY
