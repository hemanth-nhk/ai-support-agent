"""
Agent tests. Order/customer/FAQ paths are now fully deterministic (no LLM
involvement at all, per the fabrication-guarantee fix) so they're tested
directly against real output. general_node is the only path that calls
the LLM, so it's tested with generate_text mocked.
"""
from unittest.mock import patch
from app.database.seed import seed
from app.agent.graph import run_agent
from app.agent.router import classify_intent


def setup_module():
    seed(force=True)


def test_order_status_intent_classified_correctly():
    result = classify_intent("Where is order 1001?")
    assert result["intent"] == "ORDER_STATUS"


def test_faq_intent_classified_correctly():
    result = classify_intent("What is your return policy?")
    assert result["intent"] == "FAQ"


def test_out_of_scope_classified_correctly():
    result = classify_intent("Tell me a joke.")
    assert result["intent"] == "OUT_OF_SCOPE"


def test_out_of_scope_never_calls_llm():
    with patch("app.agent.nodes.generate_text") as mock_llm:
        result = run_agent("write me some python code")
        assert result["status"] == "out_of_scope"
        mock_llm.assert_not_called()


def test_missing_order_never_calls_llm():
    with patch("app.agent.nodes.generate_text") as mock_llm:
        result = run_agent("where is order 9999?")
        assert result["status"] == "not_found"
        assert result["retrieval_count"] == 0
        mock_llm.assert_not_called()


def test_found_order_never_calls_llm_and_response_matches_data_exactly():
    """
    Order responses are built with a deterministic template, not the LLM
    -- this is the fabrication-guarantee fix. Confirms the LLM is never
    invoked AND that the response text traces exactly to the DB fields.
    """
    with patch("app.agent.nodes.generate_text") as mock_llm:
        result = run_agent("Where is order 1001?")
        assert result["status"] == "ok"
        assert result["tool_used"] == "get_order_status"
        assert result["retrieval_count"] == 1
        mock_llm.assert_not_called()
        assert "1001" in result["final_response"]
        assert "Laptop" in result["final_response"]
        assert "Shipped" in result["final_response"]


def test_found_customer_orders_never_calls_llm():
    with patch("app.agent.nodes.generate_text") as mock_llm:
        result = run_agent("What orders does customer 1 have?")
        assert result["status"] == "ok"
        assert result["tool_used"] == "get_customer_orders"
        assert result["retrieval_count"] == 2
        mock_llm.assert_not_called()
        assert "1001" in result["final_response"]
        assert "1002" in result["final_response"]


def test_general_greeting_routes_to_general_node_and_uses_llm():
    with patch("app.agent.nodes.generate_text") as mock_llm:
        mock_llm.return_value = "Hi there! How can I help?"
        result = run_agent("Hello")
        assert result["intent"] == "GENERAL"
        assert result["status"] == "ok"
        mock_llm.assert_called_once()


def test_follow_up_order_id_resolution():
    # Turn 1: user asks for order without ID
    t1 = run_agent("Where is my order?")
    assert t1["intent"] == "ORDER_STATUS"
    assert t1["order_id"] is None
    assert t1["awaiting_entity"] == "order_id"
    assert t1["status"] == "not_found"

    # Turn 2: user replies with bare numeric ID "1001"
    t2 = run_agent("1001", previous_state=t1)
    assert t2["intent"] == "ORDER_STATUS"
    assert t2["order_id"] == 1001
    assert t2["status"] == "ok"
    assert t2["tool_used"] == "get_order_status"
    assert "1001" in t2["final_response"]
    assert "Laptop" in t2["final_response"]


def test_ambiguity_returns_clarification_prompt():
    # Query mentions both returning and order 1001
    result = run_agent("Can I return order 1001?")
    assert result["intent"] == "AMBIGUOUS"
    assert result["status"] == "clarification"
    assert result["tool_used"] == "none"
    assert "1001" in result["final_response"]
    assert "return" in result["final_response"].lower()


def test_ambiguity_order_and_refund_query():
    # Query mentions order and refund
    result = run_agent("My order hasn't arrived and I want a refund")
    assert result["intent"] == "AMBIGUOUS"
    assert result["status"] == "clarification"
    assert result["tool_used"] == "none"

