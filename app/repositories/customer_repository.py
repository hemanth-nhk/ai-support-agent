from typing import Optional
from app.database.connection import get_db


def find_customer_by_id(customer_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, name, email FROM customers WHERE id = ?",
            (customer_id,),
        ).fetchone()
        return dict(row) if row else None


def find_customers_by_name(customer_name: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, email FROM customers WHERE LOWER(name) LIKE LOWER(?) ORDER BY id",
            (f"%{customer_name.strip()}%",),
        ).fetchall()
        return [dict(row) for row in rows]

