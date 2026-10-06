"""
Exercises POST /chat through the real FastAPI app (real SQLite, real
router, real guardrails). Order/customer/FAQ paths are fully deterministic
now, so no mocking is needed for them; only general_node calls the LLM.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.database.seed import seed

client = TestClient(app)


def setup_module():
    seed(force=True)


def test_chat_order_status_end_to_end():
    r = client.post("/chat", json={"message": "Where is order 1001?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"response", "intent", "tool_used", "retrieval_count", "status"}
    assert body["status"] == "ok"
    assert body["intent"] == "ORDER_STATUS"
    assert body["tool_used"] == "get_order_status"
    assert "1001" in body["response"]


def test_chat_missing_order_no_llm_needed():
    r = client.post("/chat", json={"message": "Where is order 9999?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_found"
    assert body["retrieval_count"] == 0


def test_chat_out_of_scope():
    r = client.post("/chat", json={"message": "write me a python script"})
    assert r.status_code == 200
    assert r.json()["status"] == "out_of_scope"


def test_chat_empty_message_rejected():
    r = client.post("/chat", json={"message": ""})
    assert r.status_code in (400, 422)  # 422 from Pydantic min_length, 400 from guardrail


def test_chat_response_never_leaks_extra_fields():
    with patch("app.agent.nodes.generate_text", return_value="Sure, happy to help!"):
        r = client.post("/chat", json={"message": "Hello"})
    body = r.json()
    assert len(body) == 5  # exactly the 5 contracted fields, nothing more


def test_guardrail_rejects_inconsistent_state_and_fails_closed():
    from app.agent.guardrails import build_validated_envelope

    # ORDER_STATUS with tool_used="none" and status="ok" must be rejected
    inconsistent_state_1 = {
        "final_response": "Order 1001 is shipped.",
        "intent": "ORDER_STATUS",
        "tool_used": "none",
        "retrieval_count": 0,
        "status": "ok",
    }
    env1 = build_validated_envelope(inconsistent_state_1)
    assert env1["status"] == "error"
    assert env1["intent"] == "ERROR"

    # FAQ intent with get_order_status tool must be rejected
    inconsistent_state_2 = {
        "final_response": "Some return policy.",
        "intent": "FAQ",
        "tool_used": "get_order_status",
        "retrieval_count": 1,
        "status": "ok",
    }
    env2 = build_validated_envelope(inconsistent_state_2)
    assert env2["status"] == "error"
    assert env2["intent"] == "ERROR"


def test_chat_multi_turn_session_flow():
    session_id = "test_multi_turn_101"

    # Turn 1: user asks where their order is without specifying ID
    r1 = client.post("/chat", json={"message": "Where is my order?", "session_id": session_id})
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["status"] == "not_found"
    assert "order id" in b1["response"].lower()

    # Turn 2: user responds with only the number "1001"
    r2 = client.post("/chat", json={"message": "1001", "session_id": session_id})
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["status"] == "ok"
    assert b2["intent"] == "ORDER_STATUS"
    assert b2["tool_used"] == "get_order_status"
    assert "1001" in b2["response"]
    assert "Laptop" in b2["response"]


