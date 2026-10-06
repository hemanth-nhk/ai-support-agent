"""
Deterministic demo seed data — fixed IDs, fixed values, no randomness.
This lets tests assert exact expected results every run.

DEMO DATA ONLY. Replace with real integration before any non-local use.
"""
from app.database.connection import get_db
from app.database.models import init_schema

CUSTOMERS = [
    (1, "Asha Rao", "asha.rao@example.com"),
    (2, "Liam Chen", "liam.chen@example.com"),
    (3, "Priya Nair", "priya.nair@example.com"),
]

ORDERS = [
    # id, customer_id, product, status, estimated_delivery, order_time, amount, created_at
    (1001, 1, "Laptop", "Shipped", "2026-09-15", "14:30", 62500.0, "2026-09-10"),
    (1002, 1, "Wireless Mouse", "Delivered", "2026-09-05", "10:15", 1500.0, "2026-09-01"),
    (1003, 2, "Smartphone", "Processing", "2026-09-20", "18:45", 45000.0, "2026-09-12"),
    (1004, 3, "Mechanical Keyboard", "Out for delivery", "2026-09-12", "09:00", 4500.0, "2026-09-08"),
]


def seed(force: bool = False) -> None:
    """
    Idempotent by default: if data already exists, this is a no-op unless
    force=True is passed explicitly.
    """
    with get_db() as conn:
        init_schema(conn)
        existing = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing > 0 and not force:
            # Check if existing orders need column values updated
            sample_order = conn.execute("SELECT order_time, amount FROM orders LIMIT 1").fetchone()
            if sample_order and sample_order[0] is not None:
                print(f"Database already has {existing} customers -- skipping seed. "
                      f"Pass force=True (or --force) to reset to demo data.")
                return

        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM customers")
        conn.executemany(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)", CUSTOMERS
        )
        conn.executemany(
            """INSERT INTO orders
               (id, customer_id, product, status, estimated_delivery, order_time, amount, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ORDERS,
        )
    print(f"Seeded {len(CUSTOMERS)} customers and {len(ORDERS)} orders (deterministic demo data).")


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    seed(force=force_flag)
