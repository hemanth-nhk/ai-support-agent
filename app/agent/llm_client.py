"""
LLM client — Ollama, local only. No cloud provider, no API key.

CRITICAL DESIGN RULE (per project guardrails):
The LLM is used for natural-language phrasing ONLY. It never produces,
controls, or is trusted for: intent, tool_used, retrieval_count, or status.
Those fields are always constructed and validated by application code.
The LLM's raw text output only ever fills the `response` string, and only
when the application has already decided (deterministically) that the LLM
should be invoked for this turn.
"""
from langchain_ollama import ChatOllama
from app.config import settings

_llm = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
    return _llm


def generate_text(system_prompt: str, user_message: str) -> str:
    """
    Returns plain text only. Callers must not attempt to parse this as
    JSON or extract structured fields from it — see module docstring.
    """
    llm = get_llm()
    messages = [
        ("system", system_prompt),
        ("human", user_message),
    ]
    result = llm.invoke(messages)
    return result.content.strip()
