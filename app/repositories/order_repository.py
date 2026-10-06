"""
Repository layer: raw data access only. No business logic here — that
belongs in the service layer. This keeps the LangGraph tools, REST, and
GraphQL layers all calling the exact same code path down to SQLite.
"""
from typing import Optional
from app.database.connection import get_db


def find_order_by_id(order_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, customer_id, product, status, estimated_delivery, order_time, amount, created_at "
            "FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None


def find_orders_by_customer_id(customer_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, customer_id, product, status, estimated_delivery, order_time, amount, created_at "
            "FROM orders WHERE customer_id = ? ORDER BY id",
            (customer_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def search_orders(
    customer_id: Optional[int] = None,
    product: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    amount_min: Optional[float] = None,
    amount_max: Optional[float] = None,
) -> tuple[list[dict], int]:
    """
    Deterministic parameterized search.
    - Never filter customers by customer_name here.
    - If customer_id is present: o.customer_id = ?
    - Bound parameters only.
    - Hard-coded LIMIT 20.
    - Returns (orders, total_matches).
    """
    conditions = []
    params = []

    if customer_id is not None:
        conditions.append("o.customer_id = ?")
        params.append(customer_id)

    if product is not None:
        conditions.append("LOWER(o.product) LIKE LOWER(?)")
        params.append(f"%{product.strip()}%")

    if status is not None:
        conditions.append("LOWER(o.status) = LOWER(?)")
        params.append(status.strip())

    if date_from is not None:
        conditions.append("o.created_at >= ?")
        params.append(date_from.strip())

    if date_to is not None:
        conditions.append("o.created_at <= ?")
        params.append(date_to.strip())

    if time_from is not None:
        conditions.append("o.order_time >= ?")
        params.append(time_from.strip())

    if time_to is not None:
        conditions.append("o.order_time <= ?")
        params.append(time_to.strip())

    if amount_min is not None:
        conditions.append("o.amount >= ?")
        params.append(amount_min)

    if amount_max is not None:
        conditions.append("o.amount <= ?")
        params.append(amount_max)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as conn:
        count_query = f"""
            SELECT COUNT(*)
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            {where_clause}
        """
        total_matches = conn.execute(count_query, params).fetchone()[0]

        select_query = f"""
            SELECT o.id, o.customer_id, c.name as customer_name, o.product, o.status,
                   o.estimated_delivery, o.order_time, o.amount, o.created_at
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            {where_clause}
            ORDER BY o.id
            LIMIT 20
        """
        rows = conn.execute(select_query, params).fetchall()
        orders = [dict(row) for row in rows]
        return orders, total_matches
