from typing import Optional
from pydantic import BaseModel, Field


class OrderOut(BaseModel):
    id: int
    customer_id: int
    product: str
    status: str
    estimated_delivery: Optional[str] = None
    order_time: Optional[str] = None
    amount: Optional[float] = None
    created_at: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
    The final response contract enforced everywhere in this system.
    No field beyond these is ever returned — no chain-of-thought,
    no raw model output, no internal reasoning.
    """
    response: str
    intent: str
    tool_used: str
    retrieval_count: int
    status: str  # "ok" | "not_found" | "out_of_scope" | "clarification" | "needs_clarification" | "error"
