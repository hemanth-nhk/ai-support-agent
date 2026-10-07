"""
Guardrails: validates and constructs the final response envelope. This is
the single choke point every chat response passes through before reaching
the API layer.

Design rule: the LLM never produces this envelope. The application always
builds it from deterministic fields (intent, tool_used, retrieval_count,
status) plus the LLM's plain-text phrasing (response). If, for any reason,
a malformed or unexpected structured payload were ever introduced upstream,
validation failure here means we fail closed with status="error" -- never
a guessed answer.
"""
from typing import Optional
from pydantic import BaseModel, ValidationError, field_validator, model_validator

VALID_STATUSES = {"ok", "not_found", "out_of_scope", "clarification", "needs_clarification", "error"}
VALID_TOOLS = {
    "none",
    "get_order_status",
    "get_customer_orders",
    "retrieve_faq",
    "search_orders",
    "retrieve_document_chunks",
    "execute_analysis_plan",
}

MAX_MESSAGE_LENGTH = 500


class ResponseEnvelope(BaseModel):
    response: str
    intent: str
    tool_used: str
    retrieval_count: int
    status: str
    chart: Optional[dict] = None
    table: Optional[dict] = None
    pivot: Optional[dict] = None

    @field_validator("status")
    @classmethod
    def status_must_be_known(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"Unknown status: {v}")
        return v

    @field_validator("tool_used")
    @classmethod
    def tool_must_be_known(cls, v: str) -> str:
        if v not in VALID_TOOLS:
            raise ValueError(f"Unknown tool_used: {v}")
        return v

    @field_validator("retrieval_count")
    @classmethod
    def retrieval_count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retrieval_count cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_semantic_consistency(self) -> "ResponseEnvelope":
        # Cross-field consistency rules:
        # 1. Successful status must align with appropriate tool
        if self.status == "ok":
            if self.intent == "ORDER_STATUS" and self.tool_used != "get_order_status":
                raise ValueError("ORDER_STATUS with status 'ok' must use 'get_order_status'")
            if self.intent == "CUSTOMER_ORDERS" and self.tool_used != "get_customer_orders":
                raise ValueError("CUSTOMER_ORDERS with status 'ok' must use 'get_customer_orders'")
            if self.intent == "FAQ" and self.tool_used != "retrieve_faq":
                raise ValueError("FAQ with status 'ok' must use 'retrieve_faq'")
            if self.intent == "DATA_QUERY" and self.tool_used != "search_orders":
                raise ValueError("DATA_QUERY with status 'ok' must use 'search_orders'")
            if self.intent == "DOCUMENT_QUERY" and self.tool_used != "retrieve_document_chunks":
                raise ValueError("DOCUMENT_QUERY with status 'ok' must use 'retrieve_document_chunks'")
            if self.intent == "DATASET_ANALYSIS" and self.tool_used != "execute_analysis_plan":
                raise ValueError("DATASET_ANALYSIS with status 'ok' must use 'execute_analysis_plan'")
            if self.intent == "GENERAL" and self.tool_used != "none":
                raise ValueError("GENERAL intent must have tool_used 'none'")

        # 2. Tool alignment
        if self.tool_used == "get_order_status" and self.intent not in ("ORDER_STATUS", "AMBIGUOUS"):
            raise ValueError("get_order_status tool can only be used with ORDER_STATUS or AMBIGUOUS")
        if self.tool_used == "get_customer_orders" and self.intent not in ("CUSTOMER_ORDERS", "AMBIGUOUS"):
            raise ValueError("get_customer_orders tool can only be used with CUSTOMER_ORDERS or AMBIGUOUS")
        if self.tool_used == "retrieve_faq" and self.intent not in ("FAQ", "AMBIGUOUS"):
            raise ValueError("retrieve_faq tool can only be used with FAQ or AMBIGUOUS")
        if self.tool_used == "search_orders" and self.intent != "DATA_QUERY":
            raise ValueError("search_orders tool can only be used with DATA_QUERY")
        if self.tool_used == "retrieve_document_chunks" and self.intent != "DOCUMENT_QUERY":
            raise ValueError("retrieve_document_chunks tool can only be used with DOCUMENT_QUERY")
        if self.tool_used == "execute_analysis_plan" and self.intent != "DATASET_ANALYSIS":
            raise ValueError("execute_analysis_plan tool can only be used with DATASET_ANALYSIS")

        # 3. Out-of-scope must not use tools
        if self.status == "out_of_scope" and self.tool_used != "none":
            raise ValueError("out_of_scope status cannot use tools")

        return self


ERROR_ENVELOPE = {
    "response": "Something went wrong processing your request. Please try again.",
    "intent": "ERROR",
    "tool_used": "none",
    "retrieval_count": 0,
    "status": "error",
    "chart": None,
    "table": None,
}


def validate_input_message(message: str) -> tuple[bool, str]:
    """Input validation guardrail. Returns (is_valid, error_reason)."""
    if message is None or not message.strip():
        return False, "Message cannot be empty."
    if len(message) > MAX_MESSAGE_LENGTH:
        return False, f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."
    return True, ""


def build_validated_envelope(agent_state: dict) -> dict:
    """
    Takes the raw agent state dict and returns a validated response
    envelope as a plain dict. Fails closed (returns ERROR_ENVELOPE) on
    any validation failure -- never propagates a guessed or malformed
    response to the caller.
    """
    try:
        envelope = ResponseEnvelope(
            response=agent_state.get("final_response", ""),
            intent=agent_state.get("intent", "UNKNOWN"),
            tool_used=agent_state.get("tool_used", "none"),
            retrieval_count=agent_state.get("retrieval_count", 0),
            status=agent_state.get("status", "error"),
            chart=agent_state.get("chart_artifact"),
            table=agent_state.get("analysis_table"),
            pivot=agent_state.get("pivot_table"),
        )
        return envelope.model_dump()
    except ValidationError:
        return dict(ERROR_ENVELOPE)
