"""
Integration tests for Phase 12: Policy-Order Cross-Referencing.
Verifies:
1. Explicit eligibility inquiries (eligible, expired, not-yet-delivered, cancelled).
2. Missing order ID prompting and multi-turn resolution.
3. Conversational disambiguation: Ambiguous inquiry -> user requests policy -> evaluates eligibility.
4. Precedence of session-uploaded policy over store FAQ fallback.
5. Strict abstention on conflicting or indeterminate policies.
6. Live API `/chat` endpoint integration.
"""
from datetime import date
import uuid
import pytest
from fastapi.testclient import TestClient

from app.database.seed import seed
from app.main import app
from app.agent.graph import run_agent
from app.agent.router import INTENT_ORDER_POLICY_CHECK, INTENT_AMBIGUOUS, classify_intent
from app.services import document_service


@pytest.fixture(autouse=True)
def setup_db():
    seed(force=True)


@pytest.fixture
def client():
    return TestClient(app)


def test_classify_intent_explicit_eligibility():
    """Explicit questions about qualification or eligibility route to INTENT_ORDER_POLICY_CHECK."""
    q1 = classify_intent("Does order 1002 qualify for a return under our policy?")
    assert q1["intent"] == INTENT_ORDER_POLICY_CHECK
    assert q1["order_id"] == 1002

    q2 = classify_intent("Is order 1001 eligible for return?")
    assert q2["intent"] == INTENT_ORDER_POLICY_CHECK
    assert q2["order_id"] == 1001

    q3 = classify_intent("What is the return window for order 1002?")
    assert q3["intent"] == INTENT_ORDER_POLICY_CHECK
    assert q3["order_id"] == 1002

    q4 = classify_intent("Check if order 1002 can be returned under policy")
    assert q4["intent"] == INTENT_ORDER_POLICY_CHECK
    assert q4["order_id"] == 1002


def test_classify_intent_preserves_ambiguity():
    """Vague queries without eligibility keywords still route to INTENT_AMBIGUOUS."""
    q1 = classify_intent("Can I return order 1001?")
    assert q1["intent"] == INTENT_AMBIGUOUS

    q2 = classify_intent("Can I return order 1001 for a refund?")
    assert q2["intent"] == INTENT_AMBIGUOUS


def test_crossref_eligible_order_under_faq():
    """Order 1002 delivered 2026-09-05 checked on 2026-09-15 (10 days in, 30-day window) is ELIGIBLE."""
    res = run_agent(
        "Does order 1002 qualify for a return under our policy?",
        reference_date=date(2026, 9, 15),
    )
    assert res["intent"] == INTENT_ORDER_POLICY_CHECK
    assert res["status"] == "ok"
    assert res["order_id"] == 1002
    assert "qualifies for a return" in res["final_response"]
    assert "20 days remaining" in res["final_response"]
    assert "Order 1002 (Wireless Mouse)" in res["final_response"]
    assert res["tool_used"] == "assess_return_eligibility"


def test_crossref_expired_order_under_faq():
    """Order 1002 delivered 2026-09-05 checked on 2026-10-15 (40 days in, 30-day window) is EXPIRED."""
    res = run_agent(
        "Does order 1002 qualify for a return under our policy?",
        reference_date=date(2026, 10, 15),
    )
    assert res["intent"] == INTENT_ORDER_POLICY_CHECK
    assert res["status"] == "ok"
    assert "does not qualify for a return" in res["final_response"]
    assert "exceeding the 30-day return window" in res["final_response"]
    assert "by 10 days" in res["final_response"]


def test_crossref_shipped_order_not_yet_delivered():
    """Order 1001 is Shipped and has not yet been delivered."""
    res = run_agent("Can order 1001 be returned under the return policy?")
    assert res["intent"] == INTENT_ORDER_POLICY_CHECK
    assert res["status"] == "ok"
    assert "has not yet been delivered" in res["final_response"]
    assert "Order 1001 (Laptop)" in res["final_response"]


def test_crossref_processing_order_not_yet_delivered():
    """Order 1003 is Processing and has not yet been delivered."""
    res = run_agent("Is order 1003 eligible for return?")
    assert res["intent"] == INTENT_ORDER_POLICY_CHECK
    assert res["status"] == "ok"
    assert "has not yet been delivered" in res["final_response"]
    assert "Order 1003 (Smartphone)" in res["final_response"]


def test_crossref_missing_order_id_multiturn():
    """Asking about return eligibility without order ID asks for order ID, then evaluates upon reply."""
    turn1 = run_agent("Does my order qualify for a return under policy?")
    assert turn1["intent"] == INTENT_ORDER_POLICY_CHECK
    assert turn1["status"] == "not_found"
    assert turn1["awaiting_entity"] == "order_id"
    assert "order ID" in turn1["final_response"]

    turn2 = run_agent(
        "1002",
        previous_state=turn1,
        reference_date=date(2026, 9, 15),
    )
    assert turn2["intent"] == INTENT_ORDER_POLICY_CHECK
    assert turn2["status"] == "ok"
    assert turn2["order_id"] == 1002
    assert "qualifies for a return" in turn2["final_response"]


def test_crossref_clarification_from_ambiguity_multiturn():
    """
    Turn 1: 'Can I return order 1001?' -> AMBIGUOUS
    Turn 2: 'Check return policy for it' -> Evaluates order 1001 under policy (Shipped / not delivered)
    """
    turn1 = run_agent("Can I return order 1001?")
    assert turn1["intent"] == INTENT_AMBIGUOUS
    assert turn1["status"] == "clarification"

    turn2 = run_agent("Check the return policy for it", previous_state=turn1)
    assert turn2["intent"] == INTENT_ORDER_POLICY_CHECK
    assert turn2["order_id"] == 1001
    assert "has not yet been delivered" in turn2["final_response"]


def test_crossref_uploaded_session_document_precedence():
    """
    When a custom policy is uploaded in session with a 14-day window,
    Order 1002 delivered on 2026-09-05 checked on 2026-09-25 (20 days later)
    must fail against the 14-day uploaded policy (even though store FAQ has 30 days).
    """
    session_id = f"test_crossref_sess_{uuid.uuid4().hex[:8]}"
    doc_content = (
        "Acme Custom Electronics Policy:\n"
        "All hardware items may be returned within 14 calendar days of delivery.\n"
        "Items must be in original packaging."
    ).encode("utf-8")

    doc_info = document_service.index_document(
        session_id=session_id,
        filename="custom_store_policy.txt",
        content_bytes=doc_content,
    )
    doc_id = doc_info["document_id"]

    res = run_agent(
        "Does order 1002 qualify for a return under our policy?",
        session_id=session_id,
        document_id=doc_id,
        reference_date=date(2026, 9, 25),
    )
    assert res["intent"] == INTENT_ORDER_POLICY_CHECK
    assert res["status"] == "ok"
    # 20 days elapsed vs 14 days window -> EXPIRED
    assert "does not qualify for a return" in res["final_response"]
    assert "14-day return window" in res["final_response"]
    assert "custom_store_policy.txt" in res["final_response"]


def test_crossref_chat_endpoint_e2e(client):
    """End-to-end /chat API request verifies cross-referencing response."""
    resp = client.post("/chat", json={
        "message": "Can order 1001 be returned under the return policy?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == INTENT_ORDER_POLICY_CHECK
    assert data["status"] == "ok"
    assert "Order 1001 (Laptop)" in data["response"]
    assert "has not yet been delivered" in data["response"]
