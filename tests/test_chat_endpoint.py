"""
Exercises POST /chat through the real FastAPI app (real SQLite, real
router, real guardrails). Order/customer/FAQ paths are fully deterministic
now, so no mocking is needed for them; only general_node calls the LLM.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.database.seed import seed

from app.config import settings

client = TestClient(app)
TEST_API_KEY = "test-chat-key"


def setup_module():
    seed(force=True)
    settings.support_api_key = TEST_API_KEY


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


def test_chat_document_query_with_attachment_flow(monkeypatch):
    import shutil
    import tempfile
    from chromadb import EmbeddingFunction, Documents, Embeddings
    from app.config import settings
    from app.services import document_service

    class StubEmb(EmbeddingFunction):
        def __init__(self):
            pass
        def __call__(self, input: Documents) -> Embeddings:
            vocab = ["policy", "refund", "deadline", "december", "contract", "terms"]
            return [[1.0 if w in t.lower() else 0.0 for w in vocab] for t in input]
        def name(self):
            return "stub-chat-doc"

    stub_fn = StubEmb()
    temp_dir = tempfile.mkdtemp(prefix="test_chat_docs_")
    monkeypatch.setattr(settings, "chroma_persist_dir", temp_dir)
    monkeypatch.setattr(document_service, "get_embedding_function", lambda: stub_fn)

    session_a = "session_attach_user_a"
    session_b = "session_attach_user_b"

    # 1. Upload document in Session A
    txt_bytes = b"Contract terms state that the deadline is December 25."
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("contract.txt", txt_bytes, "text/plain")},
        data={"session_id": session_a},
    )
    assert upload_res.status_code == 201
    doc_info = upload_res.json()
    doc_id = doc_info["document_id"]
    assert doc_id is not None
    assert doc_info["filename"] == "contract.txt"

    # 2. Query in Session A passing document_id in ChatRequest
    with patch("app.agent.nodes.generate_text", return_value="The deadline is December 25 according to contract.txt."):
        chat_res_a = client.post(
            "/chat",
            json={
                "message": "What is the deadline in the uploaded file?",
                "session_id": session_a,
                "document_id": doc_id,
            },
        )
        assert chat_res_a.status_code == 200
        body_a = chat_res_a.json()
        assert body_a["intent"] == "DOCUMENT_QUERY"
        assert body_a["status"] == "ok"
        assert body_a["tool_used"] == "retrieve_document_chunks"
        assert "December 25" in body_a["response"]

    # 3. Query from Session B (attempting to use Session A's document) fails closed
    chat_res_b = client.post(
        "/chat",
        json={
            "message": "What is the deadline in the uploaded file?",
            "session_id": session_b,
            "document_id": doc_id,
        },
    )
    assert chat_res_b.status_code == 200
    body_b = chat_res_b.json()
    assert body_b["intent"] == "DOCUMENT_QUERY"
    assert body_b["status"] == "not_found"
    assert "no document is available in this chat" in body_b["response"].lower()

    # 4. Delete document via API
    del_res = client.delete(f"/documents/{doc_id}?session_id={session_a}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # 5. Subsequent query in Session A fails closed
    chat_res_del = client.post(
        "/chat",
        json={
            "message": "What is the deadline in the uploaded file?",
            "session_id": session_a,
            "document_id": doc_id,
        },
    )
    assert chat_res_del.status_code == 200
    body_del = chat_res_del.json()
    assert body_del["status"] == "not_found"

    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Customer Orders Auth Enforcement Tests (Fix 1)
# ============================================================================

def test_chat_customer_orders_unauthenticated_two_turn_attack_rejected():
    """Turn 1 prompts for missing customer_id; Turn 2 supplying customer_id without API key is rejected with 401."""
    session_id = "test_attack_2turn"

    # Turn 1: user asks for orders without customer ID and without API key
    r1 = client.post("/chat", json={"message": "Show my orders", "session_id": session_id})
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["status"] == "not_found"
    assert "customer id" in b1["response"].lower()

    # Turn 2: user supplies customer ID "1" without API key -> MUST be rejected with 401
    r2 = client.post("/chat", json={"message": "1", "session_id": session_id})
    assert r2.status_code == 401
    assert r2.json()["detail"] == "Invalid or missing API key."

    # Verify session was not contaminated with customer 1's orders
    with patch("app.agent.nodes.generate_text", return_value="Hello! How can I assist you today?"):
        r3 = client.post("/chat", json={"message": "Hello", "session_id": session_id})
        assert r3.status_code == 200
        assert "1001" not in r3.json()["response"]


def test_chat_customer_orders_invalid_key_rejected():
    """Requests with an invalid API key are rejected with 401, both single-turn and multi-turn."""
    # Single-turn attack with invalid key
    r_single = client.post(
        "/chat",
        json={"message": "What orders does customer 1 have?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r_single.status_code == 401
    assert r_single.json()["detail"] == "Invalid or missing API key."

    # Multi-turn with invalid key on turn 2
    session_id = "test_invalid_key_2turn"
    r1 = client.post(
        "/chat",
        json={"message": "What are my orders?", "session_id": session_id},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "not_found"

    r2 = client.post(
        "/chat",
        json={"message": "1", "session_id": session_id},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r2.status_code == 401
    assert r2.json()["detail"] == "Invalid or missing API key."


def test_chat_customer_orders_valid_key_authorized():
    """Requests with a valid API key successfully return customer orders in single-turn and multi-turn."""
    # Single-turn with valid key
    r_single = client.post(
        "/chat",
        json={"message": "What orders does customer 1 have?"},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r_single.status_code == 200
    b_single = r_single.json()
    assert b_single["status"] == "ok"
    assert b_single["intent"] == "CUSTOMER_ORDERS"
    assert b_single["tool_used"] == "get_customer_orders"
    assert b_single["retrieval_count"] == 2
    assert "1001" in b_single["response"]
    assert "1002" in b_single["response"]

    # Multi-turn with valid key
    session_id = "test_valid_key_2turn"
    r1 = client.post(
        "/chat",
        json={"message": "Show my orders", "session_id": session_id},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "not_found"

    r2 = client.post(
        "/chat",
        json={"message": "1", "session_id": session_id},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["status"] == "ok"
    assert b2["intent"] == "CUSTOMER_ORDERS"
    assert b2["tool_used"] == "get_customer_orders"
    assert "1001" in b2["response"]


def test_chat_direct_orders_and_unprotected_intents_remain_functional():
    """Direct order endpoints require API key, while public chat intents (ORDER_STATUS, FAQ) remain functional without API key."""
    # Direct order endpoint requires API key
    assert client.get("/orders/1001").status_code == 401
    r_order = client.get("/orders/1001", headers={"X-API-Key": TEST_API_KEY})
    assert r_order.status_code == 200
    assert r_order.json()["id"] == 1001

    # Chat ORDER_STATUS is public
    r_chat_order = client.post("/chat", json={"message": "Where is order 1001?"})
    assert r_chat_order.status_code == 200
    assert r_chat_order.json()["status"] == "ok"
    assert r_chat_order.json()["intent"] == "ORDER_STATUS"

    # Chat FAQ is public
    r_faq = client.post("/chat", json={"message": "What is your return policy?"})
    assert r_faq.status_code == 200
    assert r_faq.json()["intent"] == "FAQ"

    # Direct customer orders still requires API key
    r_cust_unauth = client.get("/customers/1/orders")
    assert r_cust_unauth.status_code == 401
    r_cust_auth = client.get("/customers/1/orders", headers={"X-API-Key": TEST_API_KEY})
    assert r_cust_auth.status_code == 200
    assert len(r_cust_auth.json()) == 2


def test_chat_customer_orders_pre_execution_guard_proves_tool_never_called_unauthorized():
    """Mock/spy proves get_customer_orders() is NOT called for unauthorized requests (single-turn and 2-turn)."""
    with patch("app.agent.nodes.get_customer_orders") as mock_tool:
        # 1. Single-turn unauthorized request -> tool never called
        r_single = client.post("/chat", json={"message": "What orders does customer 1 have?"})
        assert r_single.status_code == 401
        assert r_single.json()["detail"] == "Invalid or missing API key."
        mock_tool.assert_not_called()

        # 2. Two-turn flow: turn 1 asks for customer ID, turn 2 supplies it without API key -> tool never called
        session_id = "test_spy_2turn"
        r1 = client.post("/chat", json={"message": "Show my orders", "session_id": session_id})
        assert r1.status_code == 200
        assert r1.json()["status"] == "not_found"
        assert "customer id" in r1.json()["response"].lower()
        mock_tool.assert_not_called()

        r2 = client.post("/chat", json={"message": "1", "session_id": session_id})
        assert r2.status_code == 401
        assert r2.json()["detail"] == "Invalid or missing API key."
        mock_tool.assert_not_called()

        # 3. Two-turn flow with invalid key -> tool never called
        session_bad = "test_spy_badkey"
        r1_bad = client.post(
            "/chat",
            json={"message": "Show my orders", "session_id": session_bad},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r1_bad.status_code == 200
        mock_tool.assert_not_called()

        r2_bad = client.post(
            "/chat",
            json={"message": "1", "session_id": session_bad},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r2_bad.status_code == 401
        assert r2_bad.json()["detail"] == "Invalid or missing API key."
        mock_tool.assert_not_called()

        # 4. Authorized request DOES call the tool
        mock_tool.return_value = {
            "found": True,
            "orders": [{"id": 1001, "product": "Laptop", "status": "Shipped"}],
        }
        r_auth = client.post(
            "/chat",
            json={"message": "What orders does customer 1 have?"},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert r_auth.status_code == 200
        assert r_auth.json()["status"] == "ok"
        assert mock_tool.call_count == 1
