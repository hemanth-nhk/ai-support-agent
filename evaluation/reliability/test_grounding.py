"""
Benchmark Suite: Groundedness & Source Attribution (Category D3).
Evaluates that the agent attributes retrieved facts to specific document sources
and maintains strict citation tracking.
"""
import time
import tempfile
import shutil
from unittest.mock import patch
import pytest

from app.agent.nodes import document_query_node
from app.services.document_service import index_document
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_gro_")
    try:
        emb_fn = get_embedding_function()
        session_id = "bench_session_grounding"

        index_document(session_id, "refund_guide.txt", b"Refunds are issued within 5 business days to original card.", persist_dir=temp_dir, embedding_fn=emb_fn)
        index_document(session_id, "shipping_guide.txt", b"Express shipping reaches within 2 business days nationwide.", persist_dir=temp_dir, embedding_fn=emb_fn)

        def is_ollama_available() -> bool:
            import urllib.request
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    return resp.status == 200
            except Exception:
                return False

        # Case GRO_01: Source attribution tracking for single document
        t0 = time.perf_counter()
        patch_llm = None
        if not is_ollama_available():
            patch_llm = patch("app.agent.nodes.generate_text", return_value="Refund timeline is 5 days.")
            patch_llm.start()

        try:
            with patch("app.config.settings.chroma_persist_dir", temp_dir):
                state = {
                    "session_id": session_id,
                    "user_message": "What is the refund timeline?",
                }
                res = document_query_node(state)
        finally:
            if patch_llm:
                patch_llm.stop()
        sources = res.get("document_sources", [])
        passed = ("refund_guide.txt" in sources)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="GRO_01",
            category="GROUNDING",
            file_type="TXT",
            scenario="Single document citation source attribution",
            expected_behavior="document_sources includes 'refund_guide.txt'",
            actual_behavior=f"Sources returned: {sources}",
            passed=passed,
            failure_class=None if passed else "GROUNDING",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case GRO_02: Multi-document citation source attribution
        t0 = time.perf_counter()
        patch_llm = None
        if not is_ollama_available():
            patch_llm = patch("app.agent.nodes.generate_text", return_value="Refund timeline is 5 days, shipping is 2 days.")
            patch_llm.start()

        try:
            with patch("app.config.settings.chroma_persist_dir", temp_dir):
                state = {
                    "session_id": session_id,
                    "user_message": "Explain refund timeline and express shipping delivery options",
                }
                res = document_query_node(state)
        finally:
            if patch_llm:
                patch_llm.stop()
        sources = res.get("document_sources", [])
        has_both = ("refund_guide.txt" in sources and "shipping_guide.txt" in sources)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="GRO_02",
            category="GROUNDING",
            file_type="TXT",
            scenario="Multi-document citation source attribution",
            expected_behavior="document_sources includes both 'refund_guide.txt' and 'shipping_guide.txt'",
            actual_behavior=f"Sources returned: {sources}",
            passed=has_both,
            failure_class=None if has_both else "GROUNDING",
            severity=None if has_both else "P2",
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_grounding_suite():
    cases = run_cases()
    assert all(c.passed for c in cases)
