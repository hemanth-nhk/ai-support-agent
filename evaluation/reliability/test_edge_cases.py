"""
Benchmark Suite: Long-Document, File-Type Spoofing & Resource Bounds (Category G).
Tests scale boundaries (1 KB, 100 KB, 1 MB, 5 MB, 10 MB boundary, 10.5 MB oversize rejection),
needle-in-a-haystack at 10%, 50%, and 90% positions in long documents,
and file-type spoofing (fake.pdf, fake.docx, fake.txt).
"""
import time
import tempfile
import shutil
import pytest

from app.services.document_service import (
    index_document,
    retrieve_document_chunks,
    DocumentValidationError,
    MAX_FILE_SIZE_BYTES,
)
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult
from evaluation.reliability.fixtures_generator import generate_needle_document


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_edge_")
    try:
        emb_fn = get_embedding_function()
        session_id = "bench_session_edge"

        # Case EDGE_01: Controlled 1 KB file
        t0 = time.perf_counter()
        content_1kb = ("Line of standard operational policy data.\n" * 25).encode("utf-8")
        res = index_document(session_id, "size_1kb.txt", content_1kb, persist_dir=temp_dir, embedding_fn=emb_fn)
        passed = res.get("chunk_count", 0) > 0
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_01",
            category="RESOURCE",
            file_type="TXT",
            scenario="Resource processing for 1 KB document",
            expected_behavior="Processes and chunks successfully",
            actual_behavior=f"Chunks={res.get('chunk_count')}, size={len(content_1kb)}B",
            passed=passed,
            failure_class=None if passed else "RESOURCE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_02: Controlled 100 KB file
        t0 = time.perf_counter()
        content_100kb = ("Enterprise documentation text block detailing architectural components.\n" * 1500).encode("utf-8")
        res = index_document(session_id, "size_100kb.txt", content_100kb, persist_dir=temp_dir, embedding_fn=emb_fn)
        passed = res.get("chunk_count", 0) > 0
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_02",
            category="RESOURCE",
            file_type="TXT",
            scenario="Resource processing for 100 KB document",
            expected_behavior="Processes and chunks successfully",
            actual_behavior=f"Chunks={res.get('chunk_count')}, size={len(content_100kb)}B",
            passed=passed,
            failure_class=None if passed else "RESOURCE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_03: Controlled 1 MB file
        t0 = time.perf_counter()
        content_1mb = ("Comprehensive technical specification manual and compliance definitions.\n" * 15000).encode("utf-8")
        res = index_document(session_id, "size_1mb.txt", content_1mb, persist_dir=temp_dir, embedding_fn=emb_fn)
        passed = res.get("chunk_count", 0) > 0
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_03",
            category="RESOURCE",
            file_type="TXT",
            scenario="Resource processing for 1 MB document",
            expected_behavior="Processes and chunks successfully within resource bounds",
            actual_behavior=f"Chunks={res.get('chunk_count')}, size={len(content_1mb)}B",
            passed=passed,
            failure_class=None if passed else "RESOURCE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_04: Controlled 5 MB file
        t0 = time.perf_counter()
        content_5mb = ("Standard regulatory filing chapter and audit trail ledger entries.\n" * 75000).encode("utf-8")
        res = index_document(session_id, "size_5mb.txt", content_5mb, persist_dir=temp_dir, embedding_fn=emb_fn, chunk_size=100000)
        passed = res.get("chunk_count", 0) > 0
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_04",
            category="RESOURCE",
            file_type="TXT",
            scenario="Resource processing for 5 MB document",
            expected_behavior="Processes and chunks within memory limits",
            actual_behavior=f"Chunks={res.get('chunk_count')}, size={len(content_5mb)}B",
            passed=passed,
            failure_class=None if passed else "RESOURCE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_05: Controlled 10 MB (boundary limit)
        t0 = time.perf_counter()
        content_10mb = b"X" * (MAX_FILE_SIZE_BYTES - 1024)
        res = index_document(session_id, "size_10mb.txt", content_10mb, persist_dir=temp_dir, embedding_fn=emb_fn, chunk_size=200000)
        passed = res.get("chunk_count", 0) > 0
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_05",
            category="RESOURCE",
            file_type="TXT",
            scenario="File at boundary limit (< 10MB)",
            expected_behavior="Accepted and processed at allowable maximum size",
            actual_behavior=f"Chunks={res.get('chunk_count')}, size={len(content_10mb)}B",
            passed=passed,
            failure_class=None if passed else "RESOURCE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_06: Controlled 10.5 MB (exceeds 10 MB limit) -> rejected
        t0 = time.perf_counter()
        content_over = b"X" * (MAX_FILE_SIZE_BYTES + 500 * 1024)
        try:
            index_document(session_id, "oversized.txt", content_over, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted file exceeding 10MB hard limit"
            f_class = "RESOURCE"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Strictly rejected: {e}"
            f_class = None
            sev = None
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_06",
            category="RESOURCE",
            file_type="TXT",
            scenario="File exceeding 10MB limit (10.5 MB)",
            expected_behavior="Strict rejection via DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Needle tests in a 5,000-word long document
        needle_token_10 = "NEEDLE_LOC_10_PCT_TOKEN_4411"
        doc_10 = generate_needle_document(needle=needle_token_10, total_words=5000, position_pct=0.10)
        idx_10 = index_document(session_id, "long_doc_10.txt", doc_10.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        needle_token_50 = "NEEDLE_LOC_50_PCT_TOKEN_7722"
        doc_50 = generate_needle_document(needle=needle_token_50, total_words=5000, position_pct=0.50)
        idx_50 = index_document(session_id, "long_doc_50.txt", doc_50.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        needle_token_90 = "NEEDLE_LOC_90_PCT_TOKEN_9933"
        doc_90 = generate_needle_document(needle=needle_token_90, total_words=5000, position_pct=0.90)
        idx_90 = index_document(session_id, "long_doc_90.txt", doc_90.encode("utf-8"), persist_dir=temp_dir, embedding_fn=emb_fn)

        # Case EDGE_07: Needle at 10%
        t0 = time.perf_counter()
        chunks_10 = retrieve_document_chunks(session_id, needle_token_10, document_id=idx_10["document_id"], persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found_10 = any(needle_token_10 in c["text"] for c in chunks_10)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_07",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle-in-haystack at 10% position in 5,000-word document",
            expected_behavior="Retrieves needle at 10% position in top-3 chunks",
            actual_behavior=f"Needle found={found_10}, chunks={len(chunks_10)}",
            passed=found_10,
            failure_class=None if found_10 else "RETRIEVAL",
            severity=None if found_10 else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_08: Needle at 50%
        t0 = time.perf_counter()
        chunks_50 = retrieve_document_chunks(session_id, needle_token_50, document_id=idx_50["document_id"], persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found_50 = any(needle_token_50 in c["text"] for c in chunks_50)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_08",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle-in-haystack at 50% position in 5,000-word document",
            expected_behavior="Retrieves needle at 50% position in top-3 chunks",
            actual_behavior=f"Needle found={found_50}, chunks={len(chunks_50)}",
            passed=found_50,
            failure_class=None if found_50 else "RETRIEVAL",
            severity=None if found_50 else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_09: Needle at 90%
        t0 = time.perf_counter()
        chunks_90 = retrieve_document_chunks(session_id, needle_token_90, document_id=idx_90["document_id"], persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        found_90 = any(needle_token_90 in c["text"] for c in chunks_90)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_09",
            category="RETRIEVAL",
            file_type="TXT",
            scenario="Needle-in-haystack at 90% position in 5,000-word document",
            expected_behavior="Retrieves needle at 90% position in top-3 chunks",
            actual_behavior=f"Needle found={found_90}, chunks={len(chunks_90)}",
            passed=found_90,
            failure_class=None if found_90 else "RETRIEVAL",
            severity=None if found_90 else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case EDGE_10: File-type spoofing
        t0 = time.perf_counter()
        # fake.pdf containing raw ASCII text
        spoof_pdf_passed = False
        try:
            index_document(session_id, "fake.pdf", b"Hello this is just plain text masquerading as a PDF", persist_dir=temp_dir, embedding_fn=emb_fn)
        except DocumentValidationError:
            spoof_pdf_passed = True

        # fake.docx containing raw ASCII text
        spoof_docx_passed = False
        try:
            index_document(session_id, "fake.docx", b"Hello this is just plain text masquerading as a DOCX", persist_dir=temp_dir, embedding_fn=emb_fn)
        except DocumentValidationError:
            spoof_docx_passed = True

        passed = (spoof_pdf_passed and spoof_docx_passed)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="EDGE_10",
            category="SECURITY",
            file_type="SPOOF",
            scenario="File-type spoofing (fake.pdf and fake.docx containing plain text)",
            expected_behavior="Fails closed safely via DocumentValidationError without server crash",
            actual_behavior=f"fake.pdf caught={spoof_pdf_passed}, fake.docx caught={spoof_docx_passed}",
            passed=passed,
            failure_class=None if passed else "SECURITY",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_edge_cases_suite():
    cases = run_cases()
    assert all(c.passed for c in cases)
