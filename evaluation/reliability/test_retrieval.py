"""
Benchmark Suite: Retrieval Precision & Recall (Category C).
Tests Precision@K, Recall@K, MRR, needle positioning (beginning, middle, end),
semantic distractors, document length variation, and cosine threshold cutoffs.
"""
import time
import tempfile
import shutil
import pytest

from app.services.document_service import index_document, retrieve_document_chunks
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult
from evaluation.reliability.fixtures_generator import generate_needle_document


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_ret_")
    try:
        emb_fn = get_embedding_function()
        session_id = "bench_session_retrieval"

        # Setup Document 1: Multi-section policy document (Medium length)
        doc1_text = (
            "POLICY CHAPTER 1: GENERAL OVERVIEW\n"
            "This company sells electronic hardware and smart peripherals.\n\n"
            "POLICY CHAPTER 2: RETURN WINDOW\n"
            "Customers may return eligible items within 30 calendar days of delivery for a full refund.\n\n"
            "POLICY CHAPTER 3: EXPRESS SHIPPING\n"
            "Express deliveries arrive in 1 to 2 business days. Standard deliveries take 3 to 5 business days.\n\n"
            "POLICY CHAPTER 4: HARDWARE WARRANTY\n"
            "All hardware carries a 12-month manufacturer defect warranty.\n\n"
            "POLICY CHAPTER 5: REFUND PROCESSING\n"
            "Once received, inspection takes 48 hours and refunds take 5 business days to post to the original payment method.\n"
        )
        d1_meta = index_document(session_id, "policy_full.txt", doc1_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)
        d1_id = d1_meta["document_id"]

        # Setup Document 2: Needle at Start
        needle_start_text = (
            "SECRET TOKEN: DELTA-9988-START.\n"
            + ("General operational background notes for corporate team members. " * 30)
        )
        index_document(session_id, "needle_start.txt", needle_start_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Setup Document 3: Needle in Middle
        needle_mid_text = (
            ("General operational background notes for corporate team members. " * 20)
            + "\nSECRET TOKEN: DELTA-5544-MIDDLE.\n"
            + ("Additional filler notes on procedures and standard operating manuals. " * 20)
        )
        index_document(session_id, "needle_mid.txt", needle_mid_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Setup Document 4: Needle at End
        needle_end_text = (
            ("General operational background notes for corporate team members. " * 30)
            + "\nSECRET TOKEN: DELTA-1122-END."
        )
        index_document(session_id, "needle_end.txt", needle_end_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Setup Document 5: Semantically Similar Distractors
        distractor_text = (
            "Tier A basic accounts have a return period of 7 days.\n"
            "Tier B professional accounts have a return period of 14 days.\n"
            "Tier C enterprise accounts have a return period of 60 days.\n"
            "Standard consumer accounts have a return period of 30 days.\n"
        )
        index_document(session_id, "tiers.txt", distractor_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Setup Document 6: Short single-chunk document
        index_document(session_id, "short_note.txt", b"Direct hotline for escalations is 1-800-SUPPORT.", persist_dir=temp_dir, embedding_fn=emb_fn)

        # Case RET_01: Needle at Beginning
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the secret token at start?", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found = any("DELTA-9988-START" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_01",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle retrieval located at beginning of document",
            expected_behavior="Target needle DELTA-9988-START retrieved in top-3 chunks",
            actual_behavior=f"Retrieved {len(chunks)} chunks, needle found={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_02: Needle in Middle
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the secret token in middle?", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found = any("DELTA-5544-MIDDLE" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_02",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle retrieval located in middle of document",
            expected_behavior="Target needle DELTA-5544-MIDDLE retrieved in top-3 chunks",
            actual_behavior=f"Retrieved {len(chunks)} chunks, needle found={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_03: Needle at End
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the secret token at end?", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found = any("DELTA-1122-END" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_03",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle retrieval located at end of document",
            expected_behavior="Target needle DELTA-1122-END retrieved in top-3 chunks",
            actual_behavior=f"Retrieved {len(chunks)} chunks, needle found={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_04: Semantic Distractors (Enterprise Tier vs Standard)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the return period for Tier C enterprise accounts?", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=1)
        found = (len(chunks) > 0 and "60 days" in chunks[0]["text"])
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_04",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Query with semantically similar distractor alternatives (Tier A/B/C)",
            expected_behavior="Top-1 chunk isolates 60 days enterprise tier",
            actual_behavior=f"Top chunk: '{chunks[0]['text'][:80]}...' match={found}" if chunks else "No chunk returned",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_05: Short single-chunk document
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "hotline phone number for escalations", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=2)
        found = any("1-800-SUPPORT" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_05",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Retrieval against short 1-chunk document",
            expected_behavior="Accurately retrieves short chunk with hotline number",
            actual_behavior=f"Found hotline={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_06: Medium document precision (Warranty Chapter)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the warranty period for hardware?", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=1)
        found = (len(chunks) > 0 and "12-month" in chunks[0]["text"])
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_06",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Medium document targeted query (Warranty)",
            expected_behavior="Top-1 chunk contains 12-month hardware warranty",
            actual_behavior=f"Top-1 match={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_07: Medium document precision (Express shipping)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "How fast is express delivery?", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=1)
        found = (len(chunks) > 0 and "1 to 2 business days" in chunks[0]["text"])
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_07",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Medium document targeted query (Express delivery)",
            expected_behavior="Top-1 chunk contains 1 to 2 business days",
            actual_behavior=f"Top-1 match={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_08: Specific Document ID Filtering
        t0 = time.perf_counter()
        # Query matching needle_start.txt, but explicitly filter to document_id=d1_id
        chunks = retrieve_document_chunks(session_id, "DELTA-9988-START", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        # Should NOT return needle_start because d1_id does not contain it
        found_needle = any("DELTA-9988-START" in c["text"] for c in chunks)
        passed = (not found_needle)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_08",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Document ID scoping prevents retrieval from other documents in same session",
            expected_behavior="Targeted document filtering excludes chunks from other session docs",
            actual_behavior=f"Needle leaked={found_needle}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case RET_09: Completely Irrelevant Query with threshold cutoff
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "What is the orbital velocity of Jupiter?", persist_dir=temp_dir, embedding_fn=emb_fn, threshold=0.35)
        passed = (len(chunks) == 0)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_09",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Unrelated query rejected by distance threshold",
            expected_behavior="0 chunks returned under strict cosine threshold",
            actual_behavior=f"Returned {len(chunks)} chunks",
            passed=passed,
            failure_class=None if passed else "RETRIEVAL",
            severity=None if passed else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_10: Semantic Paraphrase (No exact keyword match)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "Within how many weeks can a buyer return purchased gear?", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=2)
        found = any("30 calendar days" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_10",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Paraphrased semantic query without exact keyword overlap",
            expected_behavior="Retrieves 30 calendar days chunk via dense embedding",
            actual_behavior=f"Retrieved return window={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_11: Acronym query (SLA -> business days timeline)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "refund turnaround SLA", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=2)
        found = any("5 business days" in c["text"] for c in chunks)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_11",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Acronym/terminology query (refund turnaround SLA)",
            expected_behavior="Retrieves 5 business days refund processing chunk",
            actual_behavior=f"Found refund timeline={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_12: Multi-topic query (Warranty + Shipping)
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "shipping timelines and hardware warranty coverage", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        has_shipping = any("express" in c["text"].lower() or "standard" in c["text"].lower() for c in chunks)
        has_warranty = any("12-month" in c["text"] for c in chunks)
        passed = (has_shipping and has_warranty)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_12",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Composite query spanning multiple document sections",
            expected_behavior="Top-3 chunks cover both shipping and warranty sections",
            actual_behavior=f"Covered shipping={has_shipping}, warranty={has_warranty}",
            passed=passed,
            failure_class=None if passed else "RETRIEVAL",
            severity=None if passed else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_13: Negative polarity distinction
        t0 = time.perf_counter()
        neg_text = "We strictly NEVER accept returns or issue refunds under any circumstances whatsoever."
        index_document(session_id, "strict_no_returns.txt", neg_text.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)
        chunks = retrieve_document_chunks(session_id, "NEVER accept returns", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=1)
        found = (len(chunks) > 0 and "never accept returns" in chunks[0]["text"].lower())
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_13",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Negative polarity statement retrieval ('NEVER accept returns')",
            expected_behavior="Top chunk ranks the exact negative clause first",
            actual_behavior=f"Top chunk matched={found}",
            passed=found,
            failure_class=None if found else "RETRIEVAL",
            severity=None if found else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_14: Empty string query handling
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(session_id, "   ", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        passed = (len(chunks) == 0)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_14",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Empty/whitespace query string",
            expected_behavior="Fails closed with empty list, no vector query execution",
            actual_behavior=f"Returned {len(chunks)} chunks",
            passed=passed,
            failure_class=None if passed else "RETRIEVAL",
            severity=None if passed else "P2",
            latency_ms=round(lat, 2)
        ))

        # Case RET_15: MRR (Mean Reciprocal Rank) evaluation for Top-1 retrieval
        t0 = time.perf_counter()
        # Query explicitly for Chapter 4 Warranty: expect it at rank 1
        chunks = retrieve_document_chunks(session_id, "POLICY CHAPTER 4: HARDWARE WARRANTY", document_id=d1_id, persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        at_rank_1 = (len(chunks) > 0 and "CHAPTER 4" in chunks[0]["text"])
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="RET_15",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Exact heading Top-1 ranking (Reciprocal Rank = 1.0)",
            expected_behavior="Target chunk appears at rank index 0",
            actual_behavior=f"Rank 1 match={at_rank_1}",
            passed=at_rank_1,
            failure_class=None if at_rank_1 else "RETRIEVAL",
            severity=None if at_rank_1 else "P2",
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_retrieval_suite():
    cases = run_cases()
    assert any(c.case_id == "RET_01" and c.passed for c in cases)
    assert any(c.case_id == "RET_06" and c.passed for c in cases)
    assert any(c.case_id == "RET_08" and c.passed for c in cases)
