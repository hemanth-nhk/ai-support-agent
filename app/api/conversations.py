import uuid
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from app.services import conversation_service
from app.api.session_security import (
    get_verified_session,
    require_verified_session,
    set_session_cookie,
    verify_csrf_protection,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional client-provided conversation UUID")
    title: Optional[str] = Field(default="New Analysis", description="Conversation display title")
    workspace_mode: str = Field(default="dataset", description="Workspace mode: 'dataset' or 'support'")
    dataset_id: Optional[str] = Field(default=None, description="Bound dataset ID")
    session_id: Optional[str] = Field(default=None, description="Owning session ID for dataset authorization")


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Updated conversation display title")
    dataset_id: Optional[str] = Field(default=None, description="Updated bound dataset ID")
    session_id: Optional[str] = Field(default=None, description="Ignored: conversation ownership is immutable")


@router.get("", response_model=list[dict])
def list_conversations(
    request: Request,
    workspace_mode: Optional[str] = Query(default=None, description="Filter by workspace mode (dataset, support)"),
    session_id: Optional[str] = Query(default=None, description="Optional session ID filter"),
    limit: int = Query(default=50, ge=1, le=100),
):
    verified_session = get_verified_session(request)

    if session_id and session_id.strip():
        req_sess = session_id.strip()
        if not verified_session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: a valid signed session token or cookie is required.",
            )
        if verified_session != req_sess:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: session token does not match requested session_id.",
            )
        target_session = req_sess
    else:
        target_session = verified_session

    if not target_session:
        return []

    return conversation_service.list_conversations(
        workspace_mode=workspace_mode,
        session_id=target_session,
        limit=limit,
    )


@router.post("", response_model=dict, status_code=201)
def create_conversation(req: CreateConversationRequest, request: Request, response: Response):
    verify_csrf_protection(request)

    verified_session = get_verified_session(request)

    if req.session_id and req.session_id.strip():
        req_sess = req.session_id.strip()
        if verified_session and verified_session != req_sess:
            raise HTTPException(
                status_code=403,
                detail="Access denied: session token does not match requested session_id.",
            )

    if req.dataset_id is not None:
        clean_ds_id = req.dataset_id.strip()
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?",
                (clean_ds_id,),
            )
            row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset with ID '{clean_ds_id}' not found.",
            )

        stored_owner = (row["session_id"] or "").strip()
        if not verified_session or verified_session != stored_owner:
            raise HTTPException(
                status_code=403,
                detail="Access denied: dataset belongs to a different session.",
            )
        conv_owner = verified_session
    else:
        if verified_session:
            conv_owner = verified_session
        else:
            # Defined session creation path: a missing token may establish a new session,
            # but it must NOT select an existing identity without a token.
            if req.session_id and req.session_id.strip():
                candidate = req.session_id.strip()
                from app.database.connection import get_db
                with get_db() as conn:
                    cursor1 = conn.execute("SELECT 1 FROM uploaded_datasets WHERE session_id = ? LIMIT 1", (candidate,))
                    cursor2 = conn.execute("SELECT 1 FROM conversations WHERE session_id = ? LIMIT 1", (candidate,))
                    if cursor1.fetchone() or cursor2.fetchone():
                        raise HTTPException(
                            status_code=403,
                            detail="Cannot select an existing session identity without a valid session token.",
                        )
                conv_owner = candidate
            elif req.id and req.id.strip():
                candidate = req.id.strip()
                from app.database.connection import get_db
                with get_db() as conn:
                    cursor1 = conn.execute("SELECT 1 FROM uploaded_datasets WHERE session_id = ? LIMIT 1", (candidate,))
                    cursor2 = conn.execute("SELECT 1 FROM conversations WHERE session_id = ? LIMIT 1", (candidate,))
                    if cursor1.fetchone() or cursor2.fetchone():
                        raise HTTPException(
                            status_code=403,
                            detail="Cannot select an existing session identity without a valid session token.",
                        )
                conv_owner = candidate
            else:
                conv_owner = str(uuid.uuid4())

    set_session_cookie(response, conv_owner, request=request)

    return conversation_service.create_conversation(
        conversation_id=req.id,
        title=req.title or "New Analysis",
        workspace_mode=req.workspace_mode,
        dataset_id=req.dataset_id,
        session_id=conv_owner,
    )


@router.get("/{conversation_id}", response_model=dict)
def get_conversation(conversation_id: str, request: Request):
    conv = conversation_service.get_conversation_with_messages(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    conv_owner = (conv.get("session_id") or conv.get("id") or "").strip()
    verified_session = get_verified_session(request)

    if conv.get("session_id"):
        if not verified_session or verified_session != conv_owner:
            raise HTTPException(
                status_code=403,
                detail="Access denied: you do not own this conversation.",
            )
    else:
        # Legacy NULL ownership: do not silently grant access
        is_authorized = False
        if verified_session:
            if verified_session == (conv.get("id") or "").strip():
                is_authorized = True
            elif conv.get("dataset_id"):
                from app.database.connection import get_db
                with get_db() as conn:
                    cursor = conn.execute("SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?", (conv.get("dataset_id"),))
                    row = cursor.fetchone()
                    if row and (row["session_id"] or "").strip() == verified_session:
                        is_authorized = True
        if not is_authorized:
            raise HTTPException(
                status_code=403,
                detail="Access denied: legacy conversation requires matching session ownership.",
            )

    return conv


@router.patch("/{conversation_id}", response_model=dict)
def update_conversation(conversation_id: str, req: UpdateConversationRequest, request: Request):
    verify_csrf_protection(request)
    verified_session = require_verified_session(request)

    existing = conversation_service.get_conversation(conversation_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    existing_owner = (existing.get("session_id") or existing.get("id") or "").strip()
    if verified_session != existing_owner:
        raise HTTPException(
            status_code=403,
            detail="Access denied: you do not own this conversation.",
        )

    if req.session_id and req.session_id.strip():
        req_sess = req.session_id.strip()
        if verified_session != req_sess:
            raise HTTPException(
                status_code=403,
                detail="Access denied: session token does not match requested session_id.",
            )

    if "dataset_id" in req.model_fields_set and req.dataset_id is not None:
        clean_ds_id = req.dataset_id.strip()
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?",
                (clean_ds_id,),
            )
            row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset with ID '{clean_ds_id}' not found.",
            )

        stored_owner = (row["session_id"] or "").strip()
        if verified_session != stored_owner:
            raise HTTPException(
                status_code=403,
                detail="Access denied: dataset belongs to a different session.",
            )

    kwargs = {}
    if req.title is not None:
        kwargs["title"] = req.title

    dataset_changed = False
    if "dataset_id" in req.model_fields_set:
        kwargs["dataset_id"] = req.dataset_id
        if req.dataset_id != existing.get("dataset_id") or req.dataset_id is None:
            dataset_changed = True
            kwargs["active_analysis_plan"] = None

    # Ownership is strictly immutable through PATCH: session_id is never modified
    updated = conversation_service.update_conversation(conversation_id, **kwargs)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    from app.api.rest import _sessions
    if conversation_id in _sessions and "dataset_id" in req.model_fields_set:
        _sessions[conversation_id]["dataset_id"] = req.dataset_id
        if dataset_changed:
            _sessions[conversation_id]["active_analysis_plan"] = None
            _sessions[conversation_id]["analysis_plan"] = None

    return updated


@router.delete("/{conversation_id}", response_model=dict)
def delete_conversation(conversation_id: str, request: Request):
    verify_csrf_protection(request)
    verified_session = require_verified_session(request)

    existing = conversation_service.get_conversation(conversation_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    existing_owner = (existing.get("session_id") or existing.get("id") or "").strip()
    if verified_session != existing_owner:
        raise HTTPException(
            status_code=403,
            detail="Access denied: you do not own this conversation.",
        )

    deleted = conversation_service.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    # Also clean in-memory cache if present
    from app.api.rest import _sessions
    _sessions.pop(conversation_id, None)

    return {"id": conversation_id, "deleted": True}


@router.post("/{conversation_id}/clear", response_model=dict)
def clear_conversation(conversation_id: str, request: Request):
    verify_csrf_protection(request)
    verified_session = require_verified_session(request)

    existing = conversation_service.get_conversation(conversation_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    existing_owner = (existing.get("session_id") or existing.get("id") or "").strip()
    if verified_session != existing_owner:
        raise HTTPException(
            status_code=403,
            detail="Access denied: you do not own this conversation.",
        )

    conversation_service.clear_conversation_messages(conversation_id)

    from app.api.rest import _sessions
    if conversation_id in _sessions:
        _sessions[conversation_id]["active_analysis_plan"] = None
        _sessions[conversation_id]["analysis_plan"] = None

    return {"id": conversation_id, "cleared": True}
