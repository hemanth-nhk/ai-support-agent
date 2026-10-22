from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request, Response, status

from app.api.schemas import OrderOut, ChatRequest, ChatResponse
from app.api.auth import require_api_key, check_api_key
from app.services import order_service
from app.agent.graph import run_agent, _UNSET
from app.agent.router import INTENT_DATA_QUERY, INTENT_CUSTOMER_ORDERS
from app.agent.guardrails import validate_input_message, build_validated_envelope, ERROR_ENVELOPE
from app.config import settings
import logging

logger = logging.getLogger("app.chat")

router = APIRouter()


@router.get("/")
def root():
    return {"name": settings.app_name, "status": "running"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get(
    "/orders/{order_id}",
    response_model=OrderOut,
    dependencies=[Depends(require_api_key)],
)
def get_order(order_id: int):
    try:
        return order_service.get_order(order_id)
    except order_service.OrderNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/customers/{customer_id}/orders",
    response_model=list[OrderOut],
    dependencies=[Depends(require_api_key)],
)
def get_customer_orders(customer_id: int):
    # Demo-level protection only -- see app/api/auth.py docstring.
    try:
        return order_service.get_customer_orders(customer_id)
    except order_service.CustomerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


_sessions: dict[str, dict] = {}


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(
    request: ChatRequest,
    req: Request,
    response: Response,
    x_api_key: Optional[str] = Header(default=None),
):
    logger.info("Chat request received (length=%d, session_id=%s)", len(request.message), request.session_id)

    is_valid, reason = validate_input_message(request.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    from app.services import conversation_service
    from app.api.session_security import get_verified_session, set_session_cookie

    verified_session = get_verified_session(req)
    if verified_session:
        set_session_cookie(response, verified_session, request=req)

    saved_conv = None
    if request.session_id:
        saved_conv = conversation_service.get_conversation(request.session_id)
        if saved_conv:
            is_dataset_conv = bool(saved_conv.get("dataset_id") or saved_conv.get("workspace_mode") == "dataset")
            if is_dataset_conv:
                conv_owner = (saved_conv.get("session_id") or saved_conv.get("id") or "").strip()
                if saved_conv.get("session_id"):
                    if not verified_session or verified_session != conv_owner:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied: conversation belongs to a different session.",
                        )
                else:
                    # Legacy NULL ownership record: do not silently grant access
                    is_authorized = False
                    if verified_session:
                        if verified_session == (saved_conv.get("id") or "").strip():
                            is_authorized = True
                        elif saved_conv.get("dataset_id"):
                            from app.database.connection import get_db
                            with get_db() as c_conn:
                                c_cursor = c_conn.execute("SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?", (saved_conv.get("dataset_id"),))
                                ds_r = c_cursor.fetchone()
                                if ds_r and (ds_r["session_id"] or "").strip() == verified_session:
                                    is_authorized = True
                    if not is_authorized:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied: legacy conversation requires matching session ownership.",
                        )

    # Check dataset authorization before executing analytical queries or agent tools
    target_dataset_id = None
    if "dataset_id" in request.model_fields_set and request.dataset_id is not None:
        target_dataset_id = request.dataset_id.strip()
    elif request.dataset_id is not None:
        target_dataset_id = request.dataset_id.strip()
    elif saved_conv and saved_conv.get("dataset_id"):
        target_dataset_id = saved_conv.get("dataset_id").strip()

    if target_dataset_id:
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.execute("SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?", (target_dataset_id,))
            ds_row = cursor.fetchone()
        if ds_row:
            ds_owner = (ds_row["session_id"] or "").strip()
            if ds_owner:
                if not verified_session or verified_session != ds_owner:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied: dataset belongs to a different session.",
                    )

    previous_state = _sessions.get(request.session_id, {}) if request.session_id else None
    if request.session_id and not previous_state:
        # Recover analytical state from SQLite if server restarted or session not in memory
        if saved_conv:
            previous_state = {
                "session_id": request.session_id,
                "dataset_id": saved_conv.get("dataset_id"),
                "workspace_mode": saved_conv.get("workspace_mode"),
                "active_analysis_plan": saved_conv.get("active_analysis_plan"),
                "analysis_plan": saved_conv.get("active_analysis_plan"),
            }
            _sessions[request.session_id] = previous_state


    req_dataset_id = _UNSET
    if "dataset_id" in request.model_fields_set:
        req_dataset_id = request.dataset_id
        if request.dataset_id is None and previous_state:
            previous_state["dataset_id"] = None
            previous_state["active_analysis_plan"] = None
            previous_state["analysis_plan"] = None
    elif request.dataset_id is not None:
        req_dataset_id = request.dataset_id

    if previous_state and request.dataset_id and request.dataset_id != previous_state.get("dataset_id"):
        previous_state["active_analysis_plan"] = None
        previous_state["analysis_plan"] = None
        previous_state["dataset_id"] = request.dataset_id

    ws_mode = request.workspace_mode
    if not ws_mode:
        if request.dataset_id:
            ws_mode = "dataset"
        elif previous_state and previous_state.get("workspace_mode"):
            ws_mode = previous_state.get("workspace_mode")

    try:
        agent_state = run_agent(
            request.message,
            previous_state=previous_state,
            api_key=x_api_key,
            session_id=request.session_id,
            document_id=request.document_id if request.document_id is not None else _UNSET,
            dataset_id=req_dataset_id,
            workspace_mode=ws_mode,
        )

        # Enforce API key requirement for protected intents (DATA_QUERY, CUSTOMER_ORDERS)
        if (
            agent_state.get("intent") in (INTENT_DATA_QUERY, INTENT_CUSTOMER_ORDERS)
            and agent_state.get("awaiting_entity") != "customer_id"
            and not check_api_key(x_api_key)
        ):
            require_api_key(x_api_key)

        status_val = agent_state.get("status", "ok")
        if request.session_id:
            if status_val != "error":
                _sessions[request.session_id] = agent_state

                # Persist turn to SQLite
                meta = {
                    "intent": agent_state.get("intent"),
                    "tool_used": agent_state.get("tool_used"),
                    "status": status_val,
                    "retrieval_count": agent_state.get("retrieval_count", 0),
                }
                tbl = agent_state.get("analysis_table") or agent_state.get("table")
                if tbl:
                    meta["table"] = tbl
                chrt = agent_state.get("chart_artifact") or agent_state.get("chart")
                if chrt:
                    meta["chart"] = chrt
                plan = agent_state.get("active_analysis_plan") or agent_state.get("analysis_plan")
                if plan:
                    if hasattr(plan, "model_dump"):
                        meta["active_analysis_plan"] = plan.model_dump()
                    elif hasattr(plan, "dict"):
                        meta["active_analysis_plan"] = plan.dict()
                    else:
                        meta["active_analysis_plan"] = plan
                record_ws_mode = ws_mode or ("dataset" if (request.dataset_id or agent_state.get("dataset_id") or agent_state.get("intent") == "DATASET_ANALYSIS") else "support")
                bound_ds = request.dataset_id or agent_state.get("dataset_id")

                conversation_service.record_turn(
                    conversation_id=request.session_id,
                    user_message=request.message,
                    assistant_response=agent_state.get("final_response", ""),
                    metadata=meta,
                    workspace_mode=record_ws_mode,
                    dataset_id=bound_ds,
                    active_analysis_plan=meta.get("active_analysis_plan") or agent_state.get("active_analysis_plan"),
                    session_id=verified_session or request.session_id,
                )

        logger.info(
            "Agent route selected: intent=%s tool=%s status=%s",
            agent_state.get("intent"), agent_state.get("tool_used"), agent_state.get("status"),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Agent execution failed")
        return ERROR_ENVELOPE

    # Enforce API key requirement for protected intents on the chat endpoint (defense-in-depth)
    if (
        agent_state.get("intent") in (INTENT_DATA_QUERY, INTENT_CUSTOMER_ORDERS)
        and agent_state.get("awaiting_entity") != "customer_id"
        and not check_api_key(x_api_key)
    ):
        require_api_key(x_api_key)

    envelope = build_validated_envelope(agent_state)
    return envelope


@router.get("/charts/{session_id}/{chart_id}")
def get_chart_artifact(session_id: str, chart_id: str, req: Request):
    from fastapi.responses import FileResponse
    from app.analytics.visualizer import resolve_chart_file, ChartSecurityError
    from app.api.session_security import get_verified_session

    clean_session_id = (session_id or "").strip()
    verified_session = get_verified_session(req)

    if not verified_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: a valid signed session token or cookie is required.",
        )

    is_authorized = (verified_session == clean_session_id)
    if not is_authorized:
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT session_id, dataset_id FROM conversations WHERE id = ?",
                (clean_session_id,),
            )
            conv_row = cursor.fetchone()
            if conv_row:
                conv_owner = (conv_row["session_id"] or "").strip()
                if conv_owner and conv_owner == verified_session:
                    is_authorized = True
                elif not conv_owner and conv_row["dataset_id"]:
                    ds_cursor = conn.execute(
                        "SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?",
                        (conv_row["dataset_id"],),
                    )
                    ds_row = ds_cursor.fetchone()
                    if ds_row and (ds_row["session_id"] or "").strip() == verified_session:
                        is_authorized = True
            else:
                # Also check if clean_session_id is a dataset owned by verified_session
                ds_cursor = conn.execute(
                    "SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?",
                    (clean_session_id,),
                )
                ds_row = ds_cursor.fetchone()
                if ds_row and (ds_row["session_id"] or "").strip() == verified_session:
                    is_authorized = True

    if not is_authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: chart artifact belongs to a different session.",
        )

    try:
        chart_path = resolve_chart_file(session_id=clean_session_id, chart_id=chart_id)
        return FileResponse(chart_path, media_type="image/png")
    except (ChartSecurityError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid chart request or security violation.")
    except FileNotFoundError:
        if clean_session_id != verified_session:
            try:
                chart_path = resolve_chart_file(session_id=verified_session, chart_id=chart_id)
                return FileResponse(chart_path, media_type="image/png")
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Chart not found.")
