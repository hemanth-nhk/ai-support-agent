from fastapi.testclient import TestClient
from app.main import app
from app.database.seed import seed
from app.config import settings

client = TestClient(app)

TEST_API_KEY = "test-key-for-pytest"


def setup_module():
    seed(force=True)
    settings.support_api_key = TEST_API_KEY


def _gql(query: str, headers: dict | None = None):
    return client.post("/graphql", json={"query": query}, headers=headers or {})


def test_valid_order_query():
    r = _gql(
        "{ order(id: 1001) { id product status } }",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r.status_code == 200
    data = r.json()["data"]["order"]
    assert data["id"] == 1001
    assert data["product"] == "Laptop"


def test_order_requires_api_key():
    r = _gql("{ order(id: 1001) { id product status } }")  # no header
    assert r.status_code == 200
    body = r.json()
    assert "errors" in body
    assert "Invalid or missing API key" in body["errors"][0]["message"]


def test_invalid_order_returns_null():
    r = _gql(
        "{ order(id: 9999) { id product status } }",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r.status_code == 200
    assert r.json()["data"]["order"] is None


def test_requested_fields_only():
    r = _gql(
        "{ order(id: 1001) { status } }",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r.status_code == 200
    data = r.json()["data"]["order"]
    assert list(data.keys()) == ["status"]
    assert data["status"] == "Shipped"


def test_customer_orders_requires_api_key():
    r = _gql("{ customerOrders(customerId: 1) { id product } }")  # no header
    assert r.status_code == 200  # GraphQL errors surface as 200 with an "errors" array
    body = r.json()
    assert "errors" in body


def test_customer_orders_query_with_valid_key():
    r = _gql(
        "{ customerOrders(customerId: 1) { id product } }",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert r.status_code == 200
    data = r.json()["data"]["customerOrders"]
    assert len(data) == 2
