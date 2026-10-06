"""
Order tool: the only thing LangGraph nodes call for order data. This wraps
the service layer -- it never talks to the repository or SQLite directly,
keeping the Tool -> Service -> Repository -> DB layering intact.
"""
from app.services import order_service


def get_order_status(order_id: int) -> dict:
    """
    Returns {"found": True, "order": {...}} or {"found": False} on a
    missing order. Never fabricates a status -- a miss is reported as a
    miss, full stop.
    """
    try:
        order = order_service.get_order(order_id)
        return {"found": True, "order": order}
    except order_service.OrderNotFoundError:
        return {"found": False, "order": None}


def get_customer_orders(customer_id: int) -> dict:
    try:
        orders = order_service.get_customer_orders(customer_id)
        return {"found": True, "orders": orders}
    except order_service.CustomerNotFoundError:
        return {"found": False, "orders": []}
