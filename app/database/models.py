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

CREATE_UPLOADED_DATASETS_TABLE = """
CREATE TABLE IF NOT EXISTS uploaded_datasets (
    dataset_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    format TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    workspace_mode TEXT NOT NULL DEFAULT 'dataset',
    dataset_id TEXT,
    active_analysis_plan TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CREATE_CONVERSATION_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    sender TEXT NOT NULL,
    user_message TEXT,
    assistant_response TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);
"""


def init_schema(conn) -> None:
    conn.execute(CREATE_CUSTOMERS_TABLE)
    conn.execute(CREATE_ORDERS_TABLE)
    conn.execute(CREATE_UPLOADED_DATASETS_TABLE)
    conn.execute(CREATE_CONVERSATIONS_TABLE)
    conn.execute(CREATE_CONVERSATION_MESSAGES_TABLE)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_mode_updated ON conversations (workspace_mode, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_messages_seq ON conversation_messages (conversation_id, sequence_number ASC)")

    # Ensure backward compatibility with existing SQLite files
    cursor = conn.execute("PRAGMA table_info(orders)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "order_time" not in existing_cols:
        conn.execute("ALTER TABLE orders ADD COLUMN order_time TEXT")
    if "amount" not in existing_cols:
        conn.execute("ALTER TABLE orders ADD COLUMN amount REAL")

    conv_cursor = conn.execute("PRAGMA table_info(conversations)")
    conv_cols = {row[1] for row in conv_cursor.fetchall()}
    if "session_id" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
