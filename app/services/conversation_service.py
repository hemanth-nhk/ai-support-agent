"""
Service layer for analytical and support conversation history backed by SQLite.
"""
import json
import logging
import uuid
from typing import Optional, Any

from app.database.connection import get_db
from app.repositories import conversation_repository as repo

logger = logging.getLogger("app.services.conversation_service")

from app.repositories.conversation_repository import _UNSET


def _serialize_plan(plan: Any) -> Optional[str]:
    if plan is None:
        return None
    if isinstance(plan, str):
        return plan
    if hasattr(plan, "model_dump"):
        return json.dumps(plan.model_dump())
    if hasattr(plan, "dict"):
        return json.dumps(plan.dict())
    if isinstance(plan, (dict, list)):
        return json.dumps(plan)
    try:
        return json.dumps(plan)
    except Exception:
        return None


def create_conversation(
    conversation_id: Optional[str] = None,
    title: str = "New Analysis",
    workspace_mode: str = "dataset",
    dataset_id: Optional[str] = None,
    active_analysis_plan: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> dict:
    cid = conversation_id or str(uuid.uuid4())
    plan_json = _serialize_plan(active_analysis_plan)

    with get_db() as conn:
        existing = repo.get_conversation(conn, cid)
        if existing:
            return _deserialize_plan(existing)
        conv = repo.create_conversation(
            conn,
            conversation_id=cid,
            title=title,
            workspace_mode=workspace_mode,
            dataset_id=dataset_id,
            active_analysis_plan=plan_json,
            session_id=session_id,
        )
        return _deserialize_plan(conv)


def get_conversation(conversation_id: str) -> Optional[dict]:
    with get_db() as conn:
        conv = repo.get_conversation(conn, conversation_id)
        if not conv:
            return None
        return _deserialize_plan(conv)


def get_conversation_with_messages(conversation_id: str) -> Optional[dict]:
    with get_db() as conn:
        conv = repo.get_conversation(conn, conversation_id)
        if not conv:
            return None
        conv_dict = _deserialize_plan(conv)
        conv_dict["messages"] = repo.get_messages(conn, conversation_id)
        return conv_dict


def list_conversations(
    workspace_mode: Optional[str] = "dataset",
    session_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    with get_db() as conn:
        convs = repo.list_conversations(conn, workspace_mode=workspace_mode, session_id=session_id, limit=limit)
        return [_deserialize_plan(c) for c in convs]


def update_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    dataset_id: Any = _UNSET,
    active_analysis_plan: Any = _UNSET,
    session_id: Any = _UNSET,
) -> Optional[dict]:
    plan_val = _UNSET
    if active_analysis_plan is not _UNSET:
        plan_val = _serialize_plan(active_analysis_plan)

    with get_db() as conn:
        conv = repo.update_conversation(
            conn,
            conversation_id=conversation_id,
            title=title,
            dataset_id=dataset_id,
            active_analysis_plan=plan_val,
            session_id=session_id,
        )
        if not conv:
            return None
        return _deserialize_plan(conv)


def delete_conversation(conversation_id: str) -> bool:
    with get_db() as conn:
        return repo.delete_conversation(conn, conversation_id)


def record_turn(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    metadata: Optional[dict] = None,
    workspace_mode: str = "dataset",
    dataset_id: Optional[str] = None,
    active_analysis_plan: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Persists a complete turn (user query and assistant response) into SQLite.
    Guarantees atomic update of conversation metadata, active plan, and sequence.
    """
    plan_json = _serialize_plan(active_analysis_plan)

    with get_db() as conn:
        conv = repo.get_conversation(conn, conversation_id)
        if not conv:
            # Auto-create conversation with a clean title derived from question
            derived_title = _derive_title(user_message, workspace_mode)
            conv = repo.create_conversation(
                conn,
                conversation_id=conversation_id,
                title=derived_title,
                workspace_mode=workspace_mode,
                dataset_id=dataset_id,
                active_analysis_plan=plan_json,
                session_id=session_id,
            )
        else:
            # If title is generic default, auto-generate from first user message
            current_title = conv.get("title", "")
            new_title = current_title
            if current_title in ("New Analysis", "New Conversation", ""):
                new_title = _derive_title(user_message, workspace_mode)

            repo.update_conversation(
                conn,
                conversation_id=conversation_id,
                title=new_title,
                dataset_id=dataset_id if dataset_id is not None else conv.get("dataset_id"),
                active_analysis_plan=plan_json,
            )

        # 1. Add user message
        user_msg = repo.add_message(
            conn,
            conversation_id=conversation_id,
            sender="user",
            user_message=user_message,
        )

        # 2. Add assistant message
        asst_msg = repo.add_message(
            conn,
            conversation_id=conversation_id,
            sender="assistant",
            assistant_response=assistant_response,
            metadata=metadata,
        )

        return {
            "conversation_id": conversation_id,
            "user_message": user_msg,
            "assistant_message": asst_msg,
        }


def clear_conversation_messages(conversation_id: str) -> bool:
    with get_db() as conn:
        repo.delete_messages(conn, conversation_id)
        # Clear analytical plan upon clear chat
        repo.update_conversation(
            conn,
            conversation_id=conversation_id,
            active_analysis_plan=None,
        )
        return True


def _deserialize_plan(conv: dict) -> dict:
    conv_copy = dict(conv)
    raw_plan = conv_copy.get("active_analysis_plan")
    if raw_plan and isinstance(raw_plan, str):
        try:
            conv_copy["active_analysis_plan"] = json.loads(raw_plan)
        except Exception:
            conv_copy["active_analysis_plan"] = None
    return conv_copy


def _derive_title(user_message: str, workspace_mode: str) -> str:
    cleaned = user_message.strip()
    # Strip common leading verbs for analytical queries
    for prefix in ("show me the ", "show me ", "plot this as ", "plot ", "find ", "what is the ", "what is "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    if not cleaned:
        return "New Analysis" if workspace_mode == "dataset" else "New Conversation"
    title = cleaned[0].upper() + cleaned[1:]
    if len(title) > 36:
        title = title[:33].rstrip() + "..."
    return title
