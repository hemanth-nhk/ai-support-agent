"""
Agent state. Intent is set by deterministic, application-controlled
classification (see router.py) -- never by asking the LLM to decide.
Includes multi-turn session tracking and entity resolution metadata.
"""
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    session_id: Optional[str]
    user_message: str
    conversation_history: list[dict]
    previous_intent: Optional[str]
    awaiting_entity: Optional[str]   # e.g., "order_id" | "customer_id"
    intent: str                    # set deterministically, never by the LLM
    confidence: float
    order_id: Optional[int]
    customer_id: Optional[int]
    document_id: Optional[str]
    dataset_id: Optional[str]
    pending_analysis: Optional[dict]
    resolved_metric: Optional[str]
    analysis_plan: Optional[dict]
    analysis_result: Optional[dict]
    ambiguity_details: Optional[dict]
    candidate_intents: list[str]
    retrieved_documents: list[str]
    retrieved_chunks: list[dict]
    document_sources: list[str]
    similarity_threshold: Optional[float]
    retrieval_count: int
    tool_result: Optional[dict]
    tool_used: str
    chart_artifact: Optional[dict]
    analysis_table: Optional[dict]
    final_response: str
    status: str                    # "ok" | "not_found" | "out_of_scope" | "clarification" | "needs_clarification" | "error"
    api_key: Optional[str]

