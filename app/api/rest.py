from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header

from app.api.schemas import OrderOut, ChatRequest, ChatResponse
from app.api.auth import require_api_key, check_api_key
from app.services import order_service
from app.agent.graph import run_agent
from app.agent.router import INTENT_DATA_QUERY
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


@router.get("/orders/{order_id}", response_model=OrderOut)
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
def chat(request: ChatRequest, x_api_key: Optional[str] = Header(default=None)):
    logger.info("Chat request received (length=%d, session_id=%s)", len(request.message), request.session_id)

    is_valid, reason = validate_input_message(request.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    previous_state = _sessions.get(request.session_id, {}) if request.session_id else None
    if previous_state and request.dataset_id and request.dataset_id != previous_state.get("dataset_id"):
        previous_state["active_analysis_plan"] = None
        previous_state["analysis_plan"] = None

    try:
        agent_state = run_agent(
            request.message,
            previous_state=previous_state,
            api_key=x_api_key,
            session_id=request.session_id,
            document_id=request.document_id,
            dataset_id=request.dataset_id,
        )
        if request.session_id:
            _sessions[request.session_id] = agent_state

        logger.info(
            "Agent route selected: intent=%s tool=%s status=%s",
            agent_state.get("intent"), agent_state.get("tool_used"), agent_state.get("status"),
        )
    except Exception:
        logger.exception("Agent execution failed")
        return ERROR_ENVELOPE

    # Enforce API key requirement for DATA_QUERY on the chat endpoint
    if agent_state.get("intent") == INTENT_DATA_QUERY and not check_api_key(x_api_key):
        require_api_key(x_api_key)

    envelope = build_validated_envelope(agent_state)
    return envelope


@router.get("/charts/{session_id}/{chart_id}")
def get_chart_artifact(session_id: str, chart_id: str):
    from fastapi.responses import FileResponse
    from app.analytics.visualizer import resolve_chart_file, ChartSecurityError

    try:
        chart_path = resolve_chart_file(session_id=session_id, chart_id=chart_id)
        return FileResponse(chart_path, media_type="image/png")
    except (ChartSecurityError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid chart request or security violation.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chart not found.")
