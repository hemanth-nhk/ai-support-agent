from fastapi.testclient import TestClient
from app.main import app
from app.database.seed import seed
from app.config import settings

client = TestClient(app)

TEST_API_KEY = "test-key-for-pytest"


def setup_module():
    seed(force=True)  # deterministic demo data before tests run
    settings.support_api_key = TEST_API_KEY  # so protected endpoint tests can authenticate


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_get_existing_order_with_valid_key():
    r = client.get("/orders/1001", headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1001
    assert body["product"] == "Laptop"


def test_get_existing_order_requires_api_key():
    r = client.get("/orders/1001")  # no header
    assert r.status_code == 401


def test_get_existing_order_rejects_wrong_key():
    r = client.get("/orders/1001", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_get_missing_order_returns_404_with_valid_key():
    r = client.get("/orders/9999", headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 404


def test_get_customer_orders_requires_api_key():
    r = client.get("/customers/1/orders")  # no header
    assert r.status_code == 401


def test_get_customer_orders_rejects_wrong_key():
    r = client.get("/customers/1/orders", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_get_customer_orders_with_valid_key():
    r = client.get("/customers/1/orders", headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert {o["id"] for o in body} == {1001, 1002}


def test_get_missing_customer_returns_404_with_valid_key():
    r = client.get("/customers/999/orders", headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 404
