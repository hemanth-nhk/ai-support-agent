"""
Tests for Structured Private Data Retrieval v2 (DATA_QUERY).
Covers:
- Filter validation (valid inputs, extra fields forbidden, invalid dates/times, inverted ranges, cross-midnight error)
- Node behavior (malformed JSON -> needs_clarification, no DB search)
- Customer resolution (ambiguous -> candidate names, resolved -> exact customer_id filtering)
- Deterministic search (matches -> exact DB fields without LLM phrasing, 0 matches -> not_found)
- SQL injection safety
- Truncation message on total_matches > 20
- Auth requirement via X-API-Key
"""
import json
from unittest.mock import patch
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.database.connection import get_db
from app.database.seed import seed
from app.agent.filters import OrderQueryFilter
from app.agent.router import classify_intent, INTENT_DATA_QUERY
from app.agent.graph import run_agent

TEST_API_KEY = "test-support-key"
client = TestClient(app)


def setup_module():
    seed(force=True)
    settings.support_api_key = TEST_API_KEY


def teardown_module():
    seed(force=True)


# --- 1. Filter Validation Tests ---

def test_valid_filter_construction():
    f = OrderQueryFilter(
        customer_name="Asha Rao",
        product="Laptop",
        status="Shipped",
        date_from="2026-09-01",
        date_to="2026-09-15",
        time_from="10:00",
        time_to="18:00",
        amount_min=1000.0,
        amount_max=70000.0,
    )
    assert f.customer_name == "Asha Rao"
    assert f.status == "Shipped"
    assert not f.is_empty()


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        OrderQueryFilter(unknown_field="unsupported")


def test_bad_date_rejected():
    with pytest.raises(ValidationError):
        OrderQueryFilter(date_from="2026/09/10")

    with pytest.raises(ValidationError):
        OrderQueryFilter(date_to="2026-02-31")


def test_date_from_after_date_to_rejected():
    with pytest.raises(ValidationError) as exc_info:
        OrderQueryFilter(date_from="2026-09-20", date_to="2026-09-10")
    assert "date_from cannot be after date_to" in str(exc_info.value)


def test_amount_min_greater_than_amount_max_rejected():
    with pytest.raises(ValidationError) as exc_info:
        OrderQueryFilter(amount_min=5000.0, amount_max=1000.0)
    assert "amount_min cannot be greater than amount_max" in str(exc_info.value)


def test_cross_midnight_time_range_rejected():
    with pytest.raises(ValidationError) as exc_info:
        OrderQueryFilter(time_from="23:00", time_to="02:00")
    assert "Cross-midnight time ranges aren't supported yet — please split into two separate queries." in str(exc_info.value)


# --- 2. Router DATA_QUERY Detection ---

def test_data_query_intent_routing():
    queries = [
        "find orders for Laptop",
        "search orders shipped today",
        "all orders from 2026-09-01",
        "filter orders by status Shipped",
        "list orders over 5000",
        "orders between 2026-09-01 and 2026-09-15",
    ]
    for q in queries:
        res = classify_intent(q)
        assert res["intent"] == INTENT_DATA_QUERY, f"Failed for query: {q}"


# --- 3. Node & Execution Tests ---

def test_malformed_llm_json_needs_clarification_no_db_search():
    with patch("app.agent.nodes.generate_text", return_value="Invalid json text"):
        with patch("app.services.order_service.search_orders") as mock_search:
            result = run_agent("find orders for Laptop", api_key=TEST_API_KEY)
            assert result["intent"] == INTENT_DATA_QUERY
            assert result["status"] == "needs_clarification"
            assert result["tool_used"] == "none"
            mock_search.assert_not_called()


def test_ambiguous_customer_name_returns_candidates():
    # Insert another customer with same first name "Asha"
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO customers (id, name, email) VALUES (99, 'Asha Sharma', 'asha.sharma@example.com')")

    try:
        extraction = json.dumps({"customer_name": "Asha"})
        with patch("app.agent.nodes.generate_text", return_value=extraction):
            with patch("app.services.order_service.search_orders") as mock_search:
                result = run_agent("find orders for Asha", api_key=TEST_API_KEY)
                assert result["status"] == "needs_clarification"
                assert "Asha Rao" in result["final_response"]
                assert "Asha Sharma" in result["final_response"]
                mock_search.assert_not_called()
    finally:
        with get_db() as conn:
            conn.execute("DELETE FROM customers WHERE id = 99")


def test_resolved_customer_exact_customer_id_filtering():
    extraction = json.dumps({"customer_name": "Asha Rao"})
    with patch("app.agent.nodes.generate_text", return_value=extraction):
        result = run_agent("find orders for Asha Rao", api_key=TEST_API_KEY)
        assert result["status"] == "ok"
        assert result["tool_used"] == "search_orders"
        assert "Laptop" in result["final_response"]
        assert "Wireless Mouse" in result["final_response"]


def test_valid_search_zero_matches_returns_not_found():
    extraction = json.dumps({"product": "NonExistentGadgetXYZ"})
    with patch("app.agent.nodes.generate_text", return_value=extraction):
        result = run_agent("find orders for NonExistentGadgetXYZ", api_key=TEST_API_KEY)
        assert result["status"] == "not_found"
        assert result["tool_used"] == "search_orders"
        assert result["retrieval_count"] == 0
        assert "No matching orders found" in result["final_response"]


def test_valid_search_with_matches_exact_db_values_no_llm_phrasing():
    extraction = json.dumps({"product": "Laptop"})
    # Mock generate_text so we can verify it was called only once for extraction, never for final answer
    with patch("app.agent.nodes.generate_text", return_value=extraction) as mock_llm:
        result = run_agent("search orders for Laptop", api_key=TEST_API_KEY)
        assert result["status"] == "ok"
        assert result["tool_used"] == "search_orders"
        assert result["retrieval_count"] == 1
        # Exactly 1 LLM call for extraction
        assert mock_llm.call_count == 1
        # Contains exact database values
        assert "1001" in result["final_response"]
        assert "Laptop" in result["final_response"]
        assert "Shipped" in result["final_response"]
        assert "2026-09-10" in result["final_response"]
        assert "62,500.00" in result["final_response"]


def test_sql_injection_safe():
    injection_str = "' OR 1=1 --"
    extraction = json.dumps({"product": injection_str})
    with patch("app.agent.nodes.generate_text", return_value=extraction):
        result = run_agent(f"find orders for {injection_str}", api_key=TEST_API_KEY)
        assert result["status"] == "not_found"
        assert result["tool_used"] == "search_orders"


def test_total_matches_reported_when_over_20():
    # Seed 25 orders with status "Processing"
    extra_orders = [
        (2000 + i, 1, f"Bulk Item {i}", "Processing", "2026-09-25", "11:00", 100.0 * i, "2026-09-10")
        for i in range(25)
    ]
    with get_db() as conn:
        conn.executemany(
            """INSERT INTO orders
               (id, customer_id, product, status, estimated_delivery, order_time, amount, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            extra_orders,
        )

    try:
        extraction = json.dumps({"status": "Processing"})
        with patch("app.agent.nodes.generate_text", return_value=extraction):
            result = run_agent("find orders with status Processing", api_key=TEST_API_KEY)
            assert result["status"] == "ok"
            assert result["retrieval_count"] == 20
            # Total matches is at least 26 (1 original + 25 added)
            assert "Showing 20 of" in result["final_response"]
            assert "narrow your search (date range, status, or product) to see more." in result["final_response"]
    finally:
        with get_db() as conn:
            conn.execute("DELETE FROM orders WHERE id >= 2000")


def test_data_query_chat_endpoint_auth():
    extraction = json.dumps({"product": "Laptop"})
    with patch("app.agent.nodes.generate_text", return_value=extraction):
        # 1. Missing API key -> 401
        r_unauth = client.post("/chat", json={"message": "find orders for Laptop"})
        assert r_unauth.status_code == 401

        # 2. Wrong API key -> 401
        r_wrong = client.post("/chat", json={"message": "find orders for Laptop"}, headers={"X-API-Key": "wrong-key"})
        assert r_wrong.status_code == 401

        # 3. Valid API key -> 200
        r_auth = client.post("/chat", json={"message": "find orders for Laptop"}, headers={"X-API-Key": TEST_API_KEY})
        assert r_auth.status_code == 200
        body = r_auth.json()
        assert body["status"] == "ok"
        assert body["intent"] == INTENT_DATA_QUERY
        assert body["tool_used"] == "search_orders"
        assert "Laptop" in body["response"]
