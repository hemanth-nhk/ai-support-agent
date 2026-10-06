from app.database.seed import seed
from app.tools.order_tool import get_order_status, get_customer_orders


def setup_module():
    seed(force=True)


def test_get_order_status_found():
    result = get_order_status(1001)
    assert result["found"] is True
    assert result["order"]["product"] == "Laptop"


def test_get_order_status_not_found():
    result = get_order_status(9999)
    assert result["found"] is False
    assert result["order"] is None


def test_get_customer_orders_found():
    result = get_customer_orders(1)
    assert result["found"] is True
    assert len(result["orders"]) == 2


def test_get_customer_orders_not_found():
    result = get_customer_orders(999)
    assert result["found"] is False
    assert result["orders"] == []
