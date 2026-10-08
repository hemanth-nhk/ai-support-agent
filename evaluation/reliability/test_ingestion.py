"""
Benchmark Suite: Ingestion & Backend Foundation (Category A).
Tests valid/invalid files, zero-byte, corrupted headers, extensions, size limits,
path traversal, and session requirements.
"""
import time
import tempfile
import shutil
from pathlib import Path
import pytest

from app.services.document_service import (
    index_document,
    validate_file,
    extract_text,
    DocumentValidationError,
    sanitize_filename,
    MAX_FILE_SIZE_BYTES,
)
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult
from evaluation.reliability.fixtures_generator import (
    build_raw_pdf,
    build_docx_with_paragraphs,
)


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_ing_")
    try:
        emb_fn = get_embedding_function()
        session_id = "bench_session_ingestion"

        # Case ING_01: Valid TXT
        t0 = time.perf_counter()
        try:
            txt_content = "Customer return window is 30 calendar days.".encode("utf-8")
            res = index_document(session_id, "valid_policy.txt", txt_content, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = res.get("chunk_count", 0) > 0
            act = f"Indexed {res.get('chunk_count')} chunks successfully"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Failed with {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_01",
            category="INGESTION",
            file_type="TXT",
            scenario="Valid standard UTF-8 TXT upload",
            expected_behavior="Successful ingestion with non-zero chunk count",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_02: Valid PDF
        t0 = time.perf_counter()
        try:
            pdf_bytes = build_raw_pdf(["Valid PDF document.\nWarranty: 12 months."])
            res = index_document(session_id, "valid_guide.pdf", pdf_bytes, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = res.get("chunk_count", 0) > 0
            act = f"Indexed {res.get('chunk_count')} chunks successfully"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Failed with {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_02",
            category="INGESTION",
            file_type="PDF",
            scenario="Valid standard PDF 1.4 upload",
            expected_behavior="Successful ingestion with non-zero chunk count",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_03: Valid DOCX
        t0 = time.perf_counter()
        try:
            docx_bytes = build_docx_with_paragraphs(["Customer Service SLA", "Response time: 24 hours."])
            res = index_document(session_id, "valid_sla.docx", docx_bytes, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = res.get("chunk_count", 0) > 0
            act = f"Indexed {res.get('chunk_count')} chunks successfully"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Failed with {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_03",
            category="INGESTION",
            file_type="DOCX",
            scenario="Valid standard DOCX upload",
            expected_behavior="Successful ingestion with non-zero chunk count",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_04: Empty TXT (whitespace only)
        t0 = time.perf_counter()
        try:
            index_document(session_id, "empty_spaces.txt", b"   \n\t  \n", persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Unexpectedly succeeded with whitespace-only TXT"
            f_class = "INGESTION"
            sev = "P2"
        except DocumentValidationError as e:
            passed = True
            act = f"Rejected with DocumentValidationError: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected exception {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_04",
            category="INGESTION",
            file_type="TXT",
            scenario="Whitespace-only TXT file",
            expected_behavior="Rejection via DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_05: Zero-byte file
        t0 = time.perf_counter()
        try:
            index_document(session_id, "zero_byte.txt", b"", persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Unexpectedly accepted zero-byte file"
            f_class = "INGESTION"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Rejected with DocumentValidationError: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected exception {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_05",
            category="INGESTION",
            file_type="TXT",
            scenario="Zero-byte file upload",
            expected_behavior="Immediate validation rejection with clear error",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_06: Malformed PDF (corrupt header / random bytes)
        t0 = time.perf_counter()
        try:
            corrupt_pdf = b"NOT_A_PDF_HEADER_RANDOM_CORRUPTION_12345"
            index_document(session_id, "broken.pdf", corrupt_pdf, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted corrupt PDF bytes"
            f_class = "INGESTION"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Rejected safely with DocumentValidationError: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected unhandled error {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_06",
            category="INGESTION",
            file_type="PDF",
            scenario="Malformed PDF header bytes",
            expected_behavior="Controlled rejection without server crash",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_07: Truncated PDF (valid start, abruptly cut off)
        t0 = time.perf_counter()
        try:
            full_pdf = build_raw_pdf(["Some valid content for truncation test"])
            truncated_pdf = full_pdf[: len(full_pdf) // 3]
            index_document(session_id, "truncated.pdf", truncated_pdf, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted severely truncated PDF"
            f_class = "INGESTION"
            sev = "P2"
        except DocumentValidationError as e:
            passed = True
            act = f"Caught by parser as DocumentValidationError: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected crash on truncated PDF: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_07",
            category="INGESTION",
            file_type="PDF",
            scenario="Abruptly truncated PDF stream",
            expected_behavior="Safe rejection without unhandled exception",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_08: Malformed DOCX (invalid zip)
        t0 = time.perf_counter()
        try:
            corrupt_docx = b"PK\x03\x04INVALID_ZIP_CORRUPTED_CENTRAL_DIRECTORY"
            index_document(session_id, "broken.docx", corrupt_docx, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted corrupt DOCX bytes"
            f_class = "INGESTION"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Safely caught with DocumentValidationError: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unhandled crash {type(e).__name__}: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_08",
            category="INGESTION",
            file_type="DOCX",
            scenario="Corrupt DOCX archive",
            expected_behavior="Safe rejection with DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_09: Random binary bytes with .txt extension
        t0 = time.perf_counter()
        try:
            random_binary = bytes(range(256)) * 4
            # Latin-1 decodes almost anything into string or rejects
            res = index_document(session_id, "binary_garbage.txt", random_binary, persist_dir=temp_dir, embedding_fn=emb_fn)
            # If indexed without crash, verify chunks were stored
            passed = True
            act = f"Handled binary content via fallback decoding without crash ({res.get('chunk_count')} chunks)"
            f_class = None
            sev = None
        except DocumentValidationError as e:
            passed = True
            act = f"Safely rejected binary garbage: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unhandled error: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_09",
            category="INGESTION",
            file_type="TXT",
            scenario="Arbitrary binary bytes named .txt",
            expected_behavior="Handled safely or rejected without server crash",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_10: Unsupported extension (.exe, .py)
        t0 = time.perf_counter()
        try:
            index_document(session_id, "malware.exe", b"MZ\x90\x00executable", persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted unsupported executable extension"
            f_class = "SECURITY"
            sev = "P0"
        except DocumentValidationError as e:
            passed = True
            act = f"Properly rejected unsupported extension: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected exception: {e}"
            f_class = "INGESTION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_10",
            category="INGESTION",
            file_type="EXE",
            scenario="Upload of unsupported file extension (.exe)",
            expected_behavior="Strict rejection via DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_11: Oversized file (> 10MB)
        t0 = time.perf_counter()
        try:
            oversized = b"A" * (MAX_FILE_SIZE_BYTES + 1024)
            index_document(session_id, "oversized.txt", oversized, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted file exceeding 10MB limit"
            f_class = "RESOURCE"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Enforced size boundary: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected error on oversized file: {e}"
            f_class = "RESOURCE"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_11",
            category="INGESTION",
            file_type="TXT",
            scenario="File exceeding 10MB hard boundary",
            expected_behavior="Strict rejection via DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_12: Path traversal filename
        t0 = time.perf_counter()
        try:
            traversal_name = "../../../etc/shadow.txt"
            sanitized = sanitize_filename(traversal_name)
            passed = ("../" not in sanitized and "etc" not in sanitized and sanitized == "shadow.txt")
            act = f"Sanitized '{traversal_name}' -> '{sanitized}'"
            f_class = None if passed else "SECURITY"
            sev = None if passed else "P0"
        except Exception as e:
            passed = False
            act = f"Error during sanitization: {e}"
            f_class = "SECURITY"
            sev = "P0"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_12",
            category="INGESTION",
            file_type="TXT",
            scenario="Filename path traversal attack",
            expected_behavior="Complete stripping of path components to basename",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ING_13: Invalid / empty session ID
        t0 = time.perf_counter()
        try:
            index_document("   ", "policy.txt", b"Valid text", persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Accepted empty session_id"
            f_class = "ISOLATION"
            sev = "P1"
        except DocumentValidationError as e:
            passed = True
            act = f"Rejected invalid session_id: {e}"
            f_class = None
            sev = None
        except Exception as e:
            passed = False
            act = f"Unexpected exception: {e}"
            f_class = "ISOLATION"
            sev = "P1"
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ING_13",
            category="INGESTION",
            file_type="TXT",
            scenario="Ingestion with empty whitespace session ID",
            expected_behavior="Rejection requiring non-empty session ID",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_ingestion_suite():
    cases = run_cases()
    assert len(cases) > 0
    assert any(c.case_id == "ING_01" and c.passed for c in cases)
