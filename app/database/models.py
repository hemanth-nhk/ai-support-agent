"""
Schema definitions. Kept as raw SQL (not an ORM) deliberately — this is a
small demo project and an ORM would add a dependency for no real benefit.
"""

CREATE_CUSTOMERS_TABLE = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);
"""

CREATE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product TEXT NOT NULL,
    status TEXT NOT NULL,
    estimated_delivery TEXT,
    order_time TEXT,
    amount REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);
"""


def init_schema(conn) -> None:
    conn.execute(CREATE_CUSTOMERS_TABLE)
    conn.execute(CREATE_ORDERS_TABLE)

    # Ensure backward compatibility with existing SQLite files
    cursor = conn.execute("PRAGMA table_info(orders)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "order_time" not in existing_cols:
        conn.execute("ALTER TABLE orders ADD COLUMN order_time TEXT")
    if "amount" not in existing_cols:
        conn.execute("ALTER TABLE orders ADD COLUMN amount REAL")
