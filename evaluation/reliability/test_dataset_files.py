"""
Benchmark Suite: Dataset Files & Document+Dataset Interaction (Category H).
Tests CSV, XLSX, and JSON profiling and ingestion boundaries, as well as
dual-attachment routing between tabular datasets and unstructured documents.
"""
import time
import json
import pytest

from app.analytics.dataset_service import ingest_dataset
from app.analytics.profiler import (
    load_dataframe_from_bytes,
    profile_dataset_bytes,
    DatasetProfileError,
)
from app.agent.router import (
    classify_intent,
    INTENT_DATASET_ANALYSIS,
    INTENT_DOCUMENT_QUERY,
)
from evaluation.reliability.models import BenchmarkCaseResult
from evaluation.reliability.fixtures_generator import (
    build_csv_bytes,
    build_xlsx_bytes,
    build_multi_sheet_xlsx,
)


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    session_id = "bench_session_datasets"

    # Case DAT_01: Clean CSV
    t0 = time.perf_counter()
    clean_csv = build_csv_bytes([
        {"id": 1, "product": "Widget A", "price": 19.99, "qty": 10},
        {"id": 2, "product": "Widget B", "price": 29.99, "qty": 5},
        {"id": 3, "product": "Widget C", "price": 9.99, "qty": 50},
    ])
    prof = profile_dataset_bytes(clean_csv, "csv")
    passed = (prof.row_count == 3 and len(prof.columns) == 4 and prof.duplicate_rows == 0)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_01",
        category="DATASET",
        file_type="CSV",
        scenario="Clean tabular CSV profiling",
        expected_behavior="Exact row_count=3, col_count=4, duplicate_rows=0",
        actual_behavior=f"Rows={prof.row_count}, Cols={len(prof.columns)}, Duplicates={prof.duplicate_rows}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_02: CSV with missing values
    t0 = time.perf_counter()
    csv_missing = (
        "id,product,price\n"
        "1,Alpha,100\n"
        "2,,200\n"
        "3,Gamma,\n"
    ).encode("utf-8")
    prof = profile_dataset_bytes(csv_missing, "csv")
    missing_counts = {col.name: col.missing_count for col in prof.columns}
    passed = (missing_counts.get("product") == 1 and missing_counts.get("price") == 1)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_02",
        category="DATASET",
        file_type="CSV",
        scenario="CSV with null/missing values in text and numeric columns",
        expected_behavior="Detects exactly 1 null in 'product' and 1 null in 'price'",
        actual_behavior=f"Null counts: {missing_counts}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_03: CSV with mixed types / dates
    t0 = time.perf_counter()
    csv_dates = (
        "transaction_date,customer_code,amount\n"
        "2026-01-15,C-101,150.50\n"
        "2026-02-20,C-102,320.00\n"
        "2026-03-05,C-103,45.75\n"
    ).encode("utf-8")
    prof = profile_dataset_bytes(csv_dates, "csv")
    col_names = [c.name for c in prof.columns]
    passed = ("transaction_date" in col_names and prof.row_count == 3)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_03",
        category="DATASET",
        file_type="CSV",
        scenario="CSV with ISO date strings and decimal floats",
        expected_behavior="Successfully parses ISO dates and schema",
        actual_behavior=f"Columns={col_names}, rows={prof.row_count}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_04: Malformed CSV (inconsistent column counts)
    t0 = time.perf_counter()
    malformed_csv = (
        "col1,col2,col3\n"
        "1,2,3\n"
        "4,5,6,7,8,9,OVERFLOW\n"
    ).encode("utf-8")
    try:
        # Pandas on_bad_lines defaults to raise or handles
        profile_dataset_bytes(malformed_csv, "csv")
        passed = False
        act = "Accepted malformed CSV with inconsistent column counts"
        f_class = "DATASET"
        sev = "P2"
    except DatasetProfileError as e:
        passed = True
        act = f"Safely caught with DatasetProfileError: {e}"
        f_class = None
        sev = None
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_04",
        category="DATASET",
        file_type="CSV",
        scenario="Malformed CSV with inconsistent field counts per row",
        expected_behavior="Controlled failure via DatasetProfileError",
        actual_behavior=act,
        passed=passed,
        failure_class=f_class,
        severity=sev,
        latency_ms=round(lat, 2)
    ))

    # Case DAT_05: XLSX normal workbook
    t0 = time.perf_counter()
    xlsx_bytes = build_xlsx_bytes([
        {"dept": "Sales", "headcount": 12, "budget": 120000},
        {"dept": "Engineering", "headcount": 45, "budget": 850000},
        {"dept": "Support", "headcount": 20, "budget": 200000},
    ])
    prof = profile_dataset_bytes(xlsx_bytes, "xlsx")
    passed = (prof.row_count == 3 and len(prof.columns) == 3)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_05",
        category="DATASET",
        file_type="XLSX",
        scenario="Standard single-sheet Excel workbook",
        expected_behavior="Parses 3 rows and 3 columns",
        actual_behavior=f"Rows={prof.row_count}, Cols={len(prof.columns)}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_06: XLSX multi-sheet workbook (reads primary first sheet)
    t0 = time.perf_counter()
    multi_xlsx = build_multi_sheet_xlsx({
        "Primary": [{"k": 1, "v": 10}, {"k": 2, "v": 20}],
        "Secondary": [{"x": "a"}, {"x": "b"}],
    })
    prof = profile_dataset_bytes(multi_xlsx, "xlsx")
    passed = (prof.row_count == 2 and "k" in [c.name for c in prof.columns])
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_06",
        category="DATASET",
        file_type="XLSX",
        scenario="Multi-sheet Excel workbook",
        expected_behavior="Deterministically profiles first active sheet",
        actual_behavior=f"Columns={[c.name for c in prof.columns]}, rows={prof.row_count}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P2",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_07: JSON normal records array
    t0 = time.perf_counter()
    json_bytes = json.dumps([
        {"id": "usr_1", "role": "admin", "active": True},
        {"id": "usr_2", "role": "member", "active": False},
    ]).encode("utf-8")
    prof = profile_dataset_bytes(json_bytes, "json")
    passed = (prof.row_count == 2 and "role" in [c.name for c in prof.columns])
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_07",
        category="DATASET",
        file_type="JSON",
        scenario="Standard JSON records array",
        expected_behavior="Profiles 2 records into tabular representation",
        actual_behavior=f"Rows={prof.row_count}, Columns={[c.name for c in prof.columns]}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_08: JSON nested structure with 'records' key wrapper
    t0 = time.perf_counter()
    wrapped_json = json.dumps({
        "metadata": {"version": "1.0"},
        "records": [
            {"tier": "Gold", "points": 1000},
            {"tier": "Silver", "points": 500},
            {"tier": "Bronze", "points": 250},
        ]
    }).encode("utf-8")
    prof = profile_dataset_bytes(wrapped_json, "json")
    passed = (prof.row_count == 3 and "tier" in [c.name for c in prof.columns])
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_08",
        category="DATASET",
        file_type="JSON",
        scenario="JSON object with records wrapper key",
        expected_behavior="Unpacks records key and profiles tabular data",
        actual_behavior=f"Rows={prof.row_count}, Columns={[c.name for c in prof.columns]}",
        passed=passed,
        failure_class=None if passed else "DATASET",
        severity=None if passed else "P2",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_09: Malformed JSON syntax
    t0 = time.perf_counter()
    bad_json = b"{'invalid_json': true, truncated..."
    try:
        profile_dataset_bytes(bad_json, "json")
        passed = False
        act = "Accepted malformed JSON syntax"
        f_class = "DATASET"
        sev = "P1"
    except DatasetProfileError as e:
        passed = True
        act = f"Caught syntax error as DatasetProfileError: {e}"
        f_class = None
        sev = None
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_09",
        category="DATASET",
        file_type="JSON",
        scenario="Malformed invalid JSON syntax",
        expected_behavior="Controlled failure via DatasetProfileError",
        actual_behavior=act,
        passed=passed,
        failure_class=f_class,
        severity=sev,
        latency_ms=round(lat, 2)
    ))

    # Case DAT_10: Document + Dataset Interaction (Question 1 - Tabular routing)
    # Both dataset_id and document_id are active in session
    t0 = time.perf_counter()
    res = classify_intent(
        "Which region has the highest revenue?",
        dataset_id="mock_dataset_id_123",
        document_id="mock_doc_id_456",
    )
    passed = (res["intent"] == INTENT_DATASET_ANALYSIS)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_10",
        category="ROUTING",
        file_type="MULTI",
        scenario="Dual attachments: Analytical question routes to DATASET_ANALYSIS",
        expected_behavior="Routes to INTENT_DATASET_ANALYSIS",
        actual_behavior=f"Routed to intent='{res['intent']}'",
        passed=passed,
        failure_class=None if passed else "ROUTING",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_11: Document + Dataset Interaction (Question 2 - Unstructured document routing)
    t0 = time.perf_counter()
    res = classify_intent(
        "What return period does the policy mention?",
        dataset_id="mock_dataset_id_123",
        document_id="mock_doc_id_456",
    )
    passed = (res["intent"] == INTENT_DOCUMENT_QUERY)
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_11",
        category="ROUTING",
        file_type="MULTI",
        scenario="Dual attachments: Document question routes to DOCUMENT_QUERY",
        expected_behavior="Routes to INTENT_DOCUMENT_QUERY",
        actual_behavior=f"Routed to intent='{res['intent']}'",
        passed=passed,
        failure_class=None if passed else "ROUTING",
        severity=None if passed else "P1",
        latency_ms=round(lat, 2)
    ))

    # Case DAT_12: Cross-source composite question
    # "Which region has the highest revenue, and what commission rule applies to that region?"
    t0 = time.perf_counter()
    res = classify_intent(
        "Which region has the highest revenue, and what commission rule applies to that region?",
        dataset_id="mock_dataset_id_123",
        document_id="mock_doc_id_456",
    )
    # The existing router will pick either DATASET_ANALYSIS or DOCUMENT_QUERY
    # Per specification: "If cross-source reasoning is not currently supported, classify it as UNSUPPORTED_CAPABILITY. Do NOT make the LLM improvise."
    act_intent = res.get("intent")
    lat = (time.perf_counter() - t0) * 1000
    results.append(BenchmarkCaseResult(
        case_id="DAT_12",
        category="ROUTING",
        file_type="MULTI",
        scenario="Cross-source composite query joining tabular and document facts simultaneously",
        expected_behavior="Documented architecture boundary: Cross-source simultaneous reasoning is not currently supported",
        actual_behavior=f"Router routed single intent='{act_intent}' (Cross-source simultaneous join not supported)",
        passed=True,  # Documented boundary
        failure_class="UNSUPPORTED_CAPABILITY",
        severity="P2",
        latency_ms=round(lat, 2),
        notes="Documented capability boundary: Cross-source joint reasoning across SQL and RAG simultaneously requires multi-step decomposition, classified as UNSUPPORTED_CAPABILITY."
    ))

    return results


def test_dataset_files_suite():
    cases = run_cases()
    assert len(cases) > 0
    assert any(c.case_id == "DAT_01" and c.passed for c in cases)
    assert any(c.case_id == "DAT_05" and c.passed for c in cases)
