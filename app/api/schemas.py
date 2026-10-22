from typing import Optional
from pydantic import BaseModel, Field

from app.analytics.schemas import ChartArtifact, AnalysisTable, PivotResult


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
    document_id: Optional[str] = None
    dataset_id: Optional[str] = None
    workspace_mode: Optional[str] = Field(default=None, description="Active workspace mode: 'support' or 'dataset'")


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
    chart: Optional[ChartArtifact] = None
    table: Optional[AnalysisTable] = None
    pivot: Optional[PivotResult] = None
    active_analysis_plan: Optional[dict] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    session_id: Optional[str] = None
    session_token: Optional[str] = None


class DocumentDeleteResponse(BaseModel):
    status: str
    document_id: str
    deleted_chunks: int

