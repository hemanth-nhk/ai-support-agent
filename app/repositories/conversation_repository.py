"""
Repository for conversations and conversation_messages tables in SQLite.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
import sqlite3

_UNSET = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    title: str,
    workspace_mode: str = "dataset",
    dataset_id: Optional[str] = None,
    active_analysis_plan: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    now = created_at or _now_iso()
    up_now = updated_at or now
    sess_id = session_id or conversation_id

    conn.execute(
        """
        INSERT INTO conversations (id, title, workspace_mode, dataset_id, active_analysis_plan, session_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (conversation_id, title, workspace_mode, dataset_id, active_analysis_plan, sess_id, now, up_now),
    )

    return {
        "id": conversation_id,
        "title": title,
        "workspace_mode": workspace_mode,
        "dataset_id": dataset_id,
        "active_analysis_plan": active_analysis_plan,
        "session_id": sess_id,
        "created_at": now,
        "updated_at": up_now,
    }


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> Optional[dict]:
    cursor = conn.execute(
        "SELECT id, title, workspace_mode, dataset_id, active_analysis_plan, session_id, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return dict(row)


def list_conversations(
    conn: sqlite3.Connection,
    workspace_mode: Optional[str] = "dataset",
    session_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    clauses = []
    params = []
    if workspace_mode:
        clauses.append("c.workspace_mode = ?")
        params.append(workspace_mode)
    if session_id:
        clauses.append("(c.session_id = ? OR c.id = ?)")
        params.extend([session_id, session_id])

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    query = f"""
        SELECT c.id, c.title, c.workspace_mode, c.dataset_id, c.active_analysis_plan, c.session_id, c.created_at, c.updated_at,
               COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN conversation_messages m ON c.id = m.conversation_id
        {where_sql}
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT ?
    """
    cursor = conn.execute(query, tuple(params))
    return [dict(row) for row in cursor.fetchall()]


def update_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    title: Optional[str] = None,
    dataset_id: Any = _UNSET,
    active_analysis_plan: Any = _UNSET,
    session_id: Any = _UNSET,
    updated_at: Optional[str] = None,
) -> Optional[dict]:
    conv = get_conversation(conn, conversation_id)
    if not conv:
        return None

    new_title = title if title is not None else conv["title"]
    new_dataset_id = dataset_id if dataset_id is not _UNSET else conv["dataset_id"]
    new_plan = active_analysis_plan if active_analysis_plan is not _UNSET else conv["active_analysis_plan"]
    new_session_id = session_id if session_id is not _UNSET else conv.get("session_id")
    now = updated_at or _now_iso()

    conn.execute(
        """
        UPDATE conversations
        SET title = ?, dataset_id = ?, active_analysis_plan = ?, session_id = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_title, new_dataset_id, new_plan, new_session_id, now, conversation_id),
    )

    conv["title"] = new_title
    conv["dataset_id"] = new_dataset_id
    conv["active_analysis_plan"] = new_plan
    conv["session_id"] = new_session_id
    conv["updated_at"] = now
    return conv


def delete_conversation(conn: sqlite3.Connection, conversation_id: str) -> bool:
    conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
    cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0


def add_message(
    conn: sqlite3.Connection,
    conversation_id: str,
    sender: str,
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
    metadata: Optional[dict] = None,
    message_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict:
    mid = message_id or str(uuid.uuid4())
    now = created_at or _now_iso()

    # Get max sequence number
    cursor = conn.execute(
        "SELECT COALESCE(MAX(sequence_number), 0) FROM conversation_messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    max_seq = cursor.fetchone()[0]
    next_seq = max_seq + 1

    def _json_default(o):
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "dict"):
            return o.dict()
        return str(o)

    meta_json = json.dumps(metadata, default=_json_default) if metadata is not None else None

    conn.execute(
        """
        INSERT INTO conversation_messages (id, conversation_id, sequence_number, sender, user_message, assistant_response, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (mid, conversation_id, next_seq, sender, user_message, assistant_response, meta_json, now),
    )

    # Touch conversation updated_at
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )

    return {
        "id": mid,
        "conversation_id": conversation_id,
        "sequence_number": next_seq,
        "sender": sender,
        "user_message": user_message,
        "assistant_response": assistant_response,
        "metadata": metadata,
        "created_at": now,
    }


def get_messages(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    cursor = conn.execute(
        """
        SELECT id, conversation_id, sequence_number, sender, user_message, assistant_response, metadata, created_at
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY sequence_number ASC
        """,
        (conversation_id,),
    )
    messages = []
    for row in cursor.fetchall():
        item = dict(row)
        if item.get("metadata"):
            try:
                item["metadata"] = json.loads(item["metadata"])
            except Exception:
                item["metadata"] = None
        else:
            item["metadata"] = None
        messages.append(item)
    return messages


def delete_messages(conn: sqlite3.Connection, conversation_id: str) -> int:
    cursor = conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
    return cursor.rowcount
