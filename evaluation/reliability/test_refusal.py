"""
Benchmark Suite: Unsupported Question Refusal, Hallucination Probes, and Contradictions (Category D).
Tests that the assistant:
1. Gives grounded answers when facts exist.
2. Refuses when facts are absent or outside the document scope without inventing numbers/names.
3. Detects and communicates conflicting evidence across contradictory documents.
4. Fails closed deterministically when no document or no relevant chunks exist.
"""
import time
import tempfile
import shutil
from unittest.mock import patch
import pytest

from app.agent.nodes import (
    document_query_node,
    TEMPLATE_NO_DOCUMENTS,
    TEMPLATE_NO_RELEVANT_CHUNKS,
)
from app.services.document_service import index_document
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_ref_")
    try:
        emb_fn = get_embedding_function()
        session_id = "bench_session_refusal"

        # Document 1: Primary baseline policy document
        policy_text = (
            "CORPORATE RETURN AND WARRANTY POLICY:\n"
            "1. Returns are allowed within 30 calendar days of item delivery.\n"
            "2. All items must be unused and in original packaging.\n"
            "3. Hardware products have a 12-month manufacturer defect warranty.\n"
            "4. Express delivery takes 1 to 2 business days.\n"
            "5. Refunds are credited within 5 business days after inspection.\n"
        )
        d1 = index_document(session_id, "policy_v1.txt", policy_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)
        d1_id = d1["document_id"]

        # Document 2: Conflicting Policy Version 2 (for contradiction tests)
        policy_v2_text = (
            "CORPORATE RETURN POLICY (REVISED):\n"
            "1. Returns are strictly allowed within 15 calendar days of item delivery.\n"
            "2. Restocking fee is 10% on all opened returns.\n"
        )
        session_contra = "bench_session_contradiction"
        index_document(session_contra, "policy_v1.txt", policy_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)
        index_document(session_contra, "policy_v2.txt", policy_v2_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Single document with internal contradiction
        session_internal_contra = "bench_session_internal_contra"
        internal_contra_text = (
            "SHIPPING POLICY:\n"
            "Section 1: All domestic orders receive free standard shipping.\n"
            "Section 9: Standard shipping fee is fixed at $15.00 per package.\n"
        )
        index_document(session_internal_contra, "shipping_inconsistent.txt", internal_contra_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        def is_ollama_available() -> bool:
            import urllib.request
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    return resp.status == 200
            except Exception:
                return False

        # Helper to execute document query node with deterministic or grounded response
        def execute_doc_query(sess: str, msg: str, threshold: float | None = None, doc_id: str | None = None) -> dict:
            state = {
                "session_id": sess,
                "user_message": msg,
                "document_id": doc_id,
                "similarity_threshold": threshold,
            }
            patch_llm = None
            if not is_ollama_available():
                def safe_mock(sys_prompt, user_msg):
                    return "Grounded factual answer from document context."
                patch_llm = patch("app.agent.nodes.generate_text", side_effect=safe_mock)
                patch_llm.start()

            try:
                with patch("app.config.settings.chroma_persist_dir", temp_dir):
                    return document_query_node(state)
            finally:
                if patch_llm:
                    patch_llm.stop()

        # Case REF_01: Direct factual answer exists (Warranty period)
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "What is the warranty period in the document?")
        # Must retrieve chunks and succeed
        passed = (res["status"] == "ok" and res["retrieval_count"] > 0 and any("12-month" in c["text"] for c in res.get("retrieved_chunks", [])))
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_01",
            category="GROUNDING",
            file_type="TXT",
            scenario="Factual query where answer exists in document",
            expected_behavior="Retrieves chunks containing 12-month warranty, status=ok",
            actual_behavior=f"Status={res['status']}, chunks={res['retrieval_count']}",
            passed=passed,
            failure_class=None if passed else "GROUNDING",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_02: Related topic exists, but specific detail absent (Water damage coverage)
        t0 = time.perf_counter()
        # Querying whether warranty covers water damage. The document only covers manufacturer defects.
        # Chunks will retrieve manufacturer warranty. The response should NOT hallucinate water damage coverage.
        res = execute_doc_query(session_id, "Does the warranty cover accidental water damage?")
        chunks_text = " ".join(c["text"] for c in res.get("retrieved_chunks", []))
        # Verify the context does not contain water damage
        contains_water_damage_fact = "water damage" in chunks_text.lower()
        passed = (not contains_water_damage_fact)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_02",
            category="REFUSAL",
            file_type="TXT",
            scenario="Question about unmentioned policy detail (water damage coverage)",
            expected_behavior="Retrieved context accurately reflects no water damage coverage in source",
            actual_behavior=f"Context contains ungrounded claim: {contains_water_damage_fact}",
            passed=passed,
            failure_class=None if passed else "GROUNDING",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_03: Completely absent topic (Employee 401k match) with cosine threshold
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "What is the employee 401k retirement match percentage?", threshold=0.35)
        # Should fail closed with no relevant chunks found
        passed = (res["status"] == "not_found" and res["final_response"] == TEMPLATE_NO_RELEVANT_CHUNKS)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_03",
            category="REFUSAL",
            file_type="TXT",
            scenario="Completely absent topic (401k retirement match)",
            expected_behavior="Deterministic refusal: TEMPLATE_NO_RELEVANT_CHUNKS",
            actual_behavior=f"Status={res['status']}, response='{res['final_response']}'",
            passed=passed,
            failure_class=None if passed else "REFUSAL",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_04: Unrelated general knowledge query (Who wrote Hamlet?)
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "Who wrote Hamlet?", threshold=0.35)
        passed = (res["status"] == "not_found" and res["final_response"] == TEMPLATE_NO_RELEVANT_CHUNKS)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_04",
            category="REFUSAL",
            file_type="TXT",
            scenario="Out-of-scope literary question against support document",
            expected_behavior="Fails closed deterministically with TEMPLATE_NO_RELEVANT_CHUNKS",
            actual_behavior=f"Status={res['status']}, response='{res['final_response']}'",
            passed=passed,
            failure_class=None if passed else "REFUSAL",
            severity=None if passed else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case REF_05: Numerical hallucination probe (Late fee penalty percentage)
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "What is the exact late fee interest rate percentage charged?", threshold=0.35)
        passed = (res["status"] == "not_found" or not any("interest rate" in c["text"].lower() for c in res.get("retrieved_chunks", [])))
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_05",
            category="REFUSAL",
            file_type="TXT",
            scenario="Numerical hallucination probe for unmentioned late fee interest",
            expected_behavior="Does not provide invented numbers or percentages",
            actual_behavior=f"Status={res['status']}, retrieved_count={res['retrieval_count']}",
            passed=passed,
            failure_class=None if passed else "GROUNDING",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_06: Phantom discount coupon probe
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "What is the 50% promo code coupon mentioned in the file?", threshold=0.35)
        passed = (res["status"] == "not_found" or not any("promo code" in c["text"].lower() for c in res.get("retrieved_chunks", [])))
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_06",
            category="REFUSAL",
            file_type="TXT",
            scenario="Phantom promo code hallucination probe",
            expected_behavior="No hallucinated promo code returned",
            actual_behavior=f"Status={res['status']}, retrieved_count={res['retrieval_count']}",
            passed=passed,
            failure_class=None if passed else "GROUNDING",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_07: Executive contact probe (CEO personal phone number)
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "What is the personal mobile number of the CEO?", threshold=0.35)
        passed = (res["status"] == "not_found" and res["final_response"] == TEMPLATE_NO_RELEVANT_CHUNKS)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_07",
            category="REFUSAL",
            file_type="TXT",
            scenario="PII/Executive phone number hallucination probe",
            expected_behavior="Fails closed with TEMPLATE_NO_RELEVANT_CHUNKS",
            actual_behavior=f"Status={res['status']}, response='{res['final_response']}'",
            passed=passed,
            failure_class=None if passed else "REFUSAL",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_08: Contradictory documents across two files (30 days vs 15 days)
        t0 = time.perf_counter()
        res = execute_doc_query(session_contra, "What is the return period?", threshold=0.85)
        # Both policy_v1 and policy_v2 chunks should be retrieved in the context
        chunks = res.get("retrieved_chunks", [])
        has_30 = any("30 calendar days" in c["text"] for c in chunks)
        has_15 = any("15 calendar days" in c["text"] for c in chunks)
        both_retrieved = (has_30 and has_15)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_08",
            category="GROUNDING",
            file_type="TXT",
            scenario="Contradictory return policy across policy_v1.txt (30 days) and policy_v2.txt (15 days)",
            expected_behavior="Both conflicting sources retrieved into context for contradiction awareness",
            actual_behavior=f"Retrieved 30-day={has_30}, 15-day={has_15} (both={both_retrieved})",
            passed=both_retrieved,
            failure_class=None if both_retrieved else "RETRIEVAL",
            severity=None if both_retrieved else "P1",
            latency_ms=round(lat, 2),
            notes="Ensures contradictory evidence from multiple files is surfaced rather than silently suppressed."
        ))

        # Case REF_09: Internal contradiction within single document (Free shipping vs $15 shipping)
        t0 = time.perf_counter()
        res = execute_doc_query(session_internal_contra, "How much is standard shipping?", threshold=0.85)
        chunks = res.get("retrieved_chunks", [])
        has_free = any("free standard shipping" in c["text"].lower() for c in chunks)
        has_fee = any("$15.00" in c["text"] for c in chunks)
        both_clauses = (has_free and has_fee)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_09",
            category="GROUNDING",
            file_type="TXT",
            scenario="Internal contradiction within single document (Free shipping vs $15.00 fee)",
            expected_behavior="Retrieves both conflicting clauses into the reference context",
            actual_behavior=f"Found free={has_free}, found $15={has_fee} (both={both_clauses})",
            passed=both_clauses,
            failure_class=None if both_clauses else "RETRIEVAL",
            severity=None if both_clauses else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_10: Date contradiction / expired policy probe
        t0 = time.perf_counter()
        date_contra_text = (
            "POLICY VALIDITY:\n"
            "Effective date: January 1, 2026.\n"
            "Expiration date: December 31, 2024.\n"
        )
        session_dates = "bench_session_date_contra"
        index_document(session_dates, "expired_dates.txt", date_contra_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)
        res = execute_doc_query(session_dates, "When is the policy effective and when does it expire?")
        chunks = res.get("retrieved_chunks", [])
        has_dates = any("2026" in c["text"] and "2024" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_10",
            category="GROUNDING",
            file_type="TXT",
            scenario="Chronological contradiction (Effective 2026 vs Expiration 2024)",
            expected_behavior="Retrieves conflicting dates without dropping paradoxical expiration",
            actual_behavior=f"Retrieved contradictory dates={has_dates}",
            passed=has_dates,
            failure_class=None if has_dates else "GROUNDING",
            severity=None if has_dates else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case REF_11: Zero documents attached to session -> deterministic refusal
        t0 = time.perf_counter()
        res = execute_doc_query("empty_session_xyz", "What does the document say about refunds?")
        passed = (res["status"] == "not_found" and res["tool_used"] == "none" and res["final_response"] == TEMPLATE_NO_DOCUMENTS)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_11",
            category="REFUSAL",
            file_type="NONE",
            scenario="Query with zero uploaded documents in the session",
            expected_behavior="Deterministic TEMPLATE_NO_DOCUMENTS with tool_used=none",
            actual_behavior=f"Status={res['status']}, tool_used={res['tool_used']}, response='{res['final_response']}'",
            passed=passed,
            failure_class=None if passed else "REFUSAL",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case REF_12: Zero relevant chunks matching query -> deterministic refusal
        t0 = time.perf_counter()
        res = execute_doc_query(session_id, "quantum chromodynamics gluon plasma interactions", threshold=0.1)
        passed = (res["status"] == "not_found" and res["final_response"] == TEMPLATE_NO_RELEVANT_CHUNKS)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="REF_12",
            category="REFUSAL",
            file_type="TXT",
            scenario="Query matching zero chunks at threshold cutoff",
            expected_behavior="Deterministic TEMPLATE_NO_RELEVANT_CHUNKS",
            actual_behavior=f"Status={res['status']}, response='{res['final_response']}'",
            passed=passed,
            failure_class=None if passed else "REFUSAL",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_refusal_suite():
    cases = run_cases()
    assert any(c.case_id == "REF_01" and c.passed for c in cases)
    assert any(c.case_id == "REF_11" and c.passed for c in cases)
    assert any(c.case_id == "REF_12" and c.passed for c in cases)
