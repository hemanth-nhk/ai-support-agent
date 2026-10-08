"""
Benchmark Suite: Session & Security Isolation (Category F).
HARD SECURITY GATE: Cross-session leakage must be 0.0%.
Tests document isolation across sessions, cross-session deletion resistance,
stale document references, cross-session dataset access, and dataset-switch state purging.
"""
import time
import tempfile
import shutil
import pytest

from app.services.document_service import (
    index_document,
    retrieve_document_chunks,
    delete_document,
    DocumentNotFoundError,
    DocumentValidationError,
)
from app.analytics.dataset_service import (
    ingest_dataset,
    get_dataset,
    DatasetAccessDeniedError,
    DatasetNotFoundError,
)
from app.analytics.schemas import (
    AnalysisPlan,
    GroupByClause,
    AggregationClause,
    AggregationFunction,
)
from app.agent.graph import run_agent
from app.rag.embeddings import get_embedding_function
from evaluation.reliability.models import BenchmarkCaseResult
from evaluation.reliability.fixtures_generator import build_csv_bytes


def run_cases() -> list[BenchmarkCaseResult]:
    results: list[BenchmarkCaseResult] = []
    temp_dir = tempfile.mkdtemp(prefix="bench_iso_")
    try:
        emb_fn = get_embedding_function()

        sess_a = "session_alpha_101"
        sess_b = "session_beta_202"

        # Index Document A in Session A
        doc_a = index_document(sess_a, "alpha_secret.txt", b"CONFIDENTIAL_ALPHA: Project Apollo code is 9988.", persist_dir=temp_dir, embedding_fn=emb_fn)
        doc_a_id = doc_a["document_id"]

        # Index Document B in Session B
        doc_b = index_document(sess_b, "beta_secret.txt", b"CONFIDENTIAL_BETA: Project Artemis code is 1122.", persist_dir=temp_dir, embedding_fn=emb_fn)
        doc_b_id = doc_b["document_id"]

        # Case ISO_01: Session B queries Session A's document
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(sess_b, "Project Apollo code", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        leaked = any("CONFIDENTIAL_ALPHA" in c["text"] for c in chunks)
        all_belong_to_b = all(c.get("metadata", {}).get("session_id") == sess_b for c in chunks)
        passed = (not leaked and all_belong_to_b)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_01",
            category="ISOLATION",
            file_type="TXT",
            scenario="Session B queries document uploaded in Session A",
            expected_behavior="Zero Session A chunks returned; 0% cross-session leakage",
            actual_behavior=f"Chunks returned: {len(chunks)}, leaked={leaked}, all_belong_to_sess_b={all_belong_to_b}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P0",
            latency_ms=round(lat, 2),
            notes="Hard security gate: Cross-session document leakage must be 0%."
        ))

        # Case ISO_02: Session A queries Session B's document
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(sess_a, "Project Artemis code", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        leaked = any("CONFIDENTIAL_BETA" in c["text"] for c in chunks)
        all_belong_to_a = all(c.get("metadata", {}).get("session_id") == sess_a for c in chunks)
        passed = (not leaked and all_belong_to_a)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_02",
            category="ISOLATION",
            file_type="TXT",
            scenario="Session A queries document uploaded in Session B",
            expected_behavior="Zero Session B chunks returned; 0% cross-session leakage",
            actual_behavior=f"Chunks returned: {len(chunks)}, leaked={leaked}, all_belong_to_sess_a={all_belong_to_a}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P0",
            latency_ms=round(lat, 2)
        ))

        # Case ISO_03: Document deletion removes all vectors and fails closed
        t0 = time.perf_counter()
        del_res = delete_document(doc_a_id, session_id=sess_a, persist_dir=temp_dir, embedding_fn=emb_fn)
        post_chunks = retrieve_document_chunks(sess_a, "Project Apollo", persist_dir=temp_dir, embedding_fn=emb_fn, top_k=3)
        passed = (del_res.get("status") == "deleted" and len(post_chunks) == 0)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_03",
            category="ISOLATION",
            file_type="TXT",
            scenario="Session A deletes document and attempts subsequent retrieval",
            expected_behavior="All vectors deleted; post-deletion retrieval returns 0 chunks",
            actual_behavior=f"Deleted chunks={del_res.get('deleted_chunks')}, post_retrieval={len(post_chunks)}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P0",
            latency_ms=round(lat, 2)
        ))

        # Case ISO_04: Session B attempts to delete Session A's document
        t0 = time.perf_counter()
        # Re-index doc A for test
        doc_a2 = index_document(sess_a, "alpha2.txt", b"Alpha content 2", persist_dir=temp_dir, embedding_fn=emb_fn)
        doc_a2_id = doc_a2["document_id"]
        try:
            delete_document(doc_a2_id, session_id=sess_b, persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Session B successfully deleted Session A's document!"
            f_class = "ISOLATION"
            sev = "P0"
        except DocumentNotFoundError:
            passed = True
            act = "Access denied: DocumentNotFoundError raised on session mismatch"
            f_class = None
            sev = None
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_04",
            category="ISOLATION",
            file_type="TXT",
            scenario="Cross-session document deletion attempt (Session B tries to delete Session A doc)",
            expected_behavior="Strict rejection (DocumentNotFoundError 404 / access denied)",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ISO_05: Query with non-existent / stale document ID
        t0 = time.perf_counter()
        chunks = retrieve_document_chunks(sess_a, "Alpha content", document_id="00000000-0000-0000-0000-000000000000", persist_dir=temp_dir, embedding_fn=emb_fn)
        passed = (len(chunks) == 0)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_05",
            category="ISOLATION",
            file_type="TXT",
            scenario="Query with non-existent/stale document UUID",
            expected_behavior="Returns empty list, fails closed safely",
            actual_behavior=f"Chunks returned: {len(chunks)}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case ISO_06: Attachment switching in same session
        t0 = time.perf_counter()
        doc_a3 = index_document(sess_a, "alpha3.txt", b"Alpha 3: Unique identifier KILO-99", persist_dir=temp_dir, embedding_fn=emb_fn)
        doc_a3_id = doc_a3["document_id"]
        # Query scoped to doc_a2_id should not see KILO-99
        c_a2 = retrieve_document_chunks(sess_a, "KILO-99", document_id=doc_a2_id, persist_dir=temp_dir, embedding_fn=emb_fn)
        c_a3 = retrieve_document_chunks(sess_a, "KILO-99", document_id=doc_a3_id, persist_dir=temp_dir, embedding_fn=emb_fn)
        c_a2_has_kilo = any("KILO-99" in c["text"] for c in c_a2)
        c_a3_has_kilo = any("KILO-99" in c["text"] for c in c_a3)
        passed = (not c_a2_has_kilo and c_a3_has_kilo)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_06",
            category="ISOLATION",
            file_type="TXT",
            scenario="Attachment switching between multiple documents in the same session",
            expected_behavior="Scoping accurately isolates target document within session",
            actual_behavior=f"c_a2_has_kilo={c_a2_has_kilo}, c_a3_has_kilo={c_a3_has_kilo}",
            passed=passed,
            failure_class=None if passed else "ISOLATION",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case ISO_07: Cross-session dataset query isolation
        t0 = time.perf_counter()
        csv_bytes = build_csv_bytes([{"region": "North", "sales": 100}, {"region": "South", "sales": 200}])
        ds_a = ingest_dataset(sess_a, "sales.csv", csv_bytes)
        ds_a_id = ds_a.dataset_id
        # Session B attempts to read Session A's dataset metadata
        try:
            get_dataset(ds_a_id, session_id=sess_b)
            passed = False
            act = "Session B accessed Session A dataset metadata!"
            f_class = "ISOLATION"
            sev = "P0"
        except (DatasetAccessDeniedError, DatasetNotFoundError):
            passed = True
            act = "Access denied: DatasetAccessDeniedError/NotFound raised"
            f_class = None
            sev = None
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_07",
            category="ISOLATION",
            file_type="CSV",
            scenario="Cross-session dataset access attempt (Session B queries Session A dataset)",
            expected_behavior="Strict denial (DatasetAccessDeniedError)",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2),
            notes="Dataset session boundary enforcement."
        ))

        # Case ISO_08: Dataset switching invalidates active_analysis_plan
        t0 = time.perf_counter()
        csv_bytes_2 = build_csv_bytes([{"category": "Electronics", "units": 50}])
        ds_a2 = ingest_dataset(sess_a, "inventory.csv", csv_bytes_2)
        ds_a2_id = ds_a2.dataset_id

        # Initial state with dataset 1 active and active_analysis_plan set
        dummy_plan = AnalysisPlan(
            dataset_id=ds_a_id,
            group_by=[GroupByClause(column="region")],
            aggregations=[AggregationClause(column="sales", function=AggregationFunction.SUM, alias="total_sales")],
        )
        prev_state = {
            "session_id": sess_a,
            "dataset_id": ds_a_id,
            "active_analysis_plan": dummy_plan.model_dump(),
        }
        # Run agent switching to dataset 2
        new_state = run_agent(
            user_message="Show inventory",
            previous_state=prev_state,
            session_id=sess_a,
            dataset_id=ds_a2_id,
        )
        passed = (new_state.get("active_analysis_plan") is None and new_state.get("dataset_id") == ds_a2_id)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_08",
            category="STATE",
            file_type="CSV",
            scenario="Switching active dataset in session purges previous active_analysis_plan",
            expected_behavior="Old plan purged (active_analysis_plan=None); cannot cross datasets",
            actual_behavior=f"active_plan={new_state.get('active_analysis_plan')}, dataset_id={new_state.get('dataset_id')}",
            passed=passed,
            failure_class=None if passed else "STATE",
            severity=None if passed else "P1",
            latency_ms=round(lat, 2)
        ))

        # Case ISO_09: Missing / empty session ID rejected on retrieval
        t0 = time.perf_counter()
        try:
            retrieve_document_chunks("   ", "query", persist_dir=temp_dir, embedding_fn=emb_fn)
            passed = False
            act = "Allowed retrieval with whitespace session_id"
            f_class = "ISOLATION"
            sev = "P1"
        except DocumentValidationError:
            passed = True
            act = "Rejected with DocumentValidationError"
            f_class = None
            sev = None
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_09",
            category="ISOLATION",
            file_type="TXT",
            scenario="Document retrieval with empty whitespace session ID",
            expected_behavior="Strict rejection via DocumentValidationError",
            actual_behavior=act,
            passed=passed,
            failure_class=f_class,
            severity=sev,
            latency_ms=round(lat, 2)
        ))

        # Case ISO_10: Session ID with path traversal characters
        t0 = time.perf_counter()
        traversal_sess = "../../admin_session"
        chunks = retrieve_document_chunks(traversal_sess, "query", persist_dir=temp_dir, embedding_fn=emb_fn)
        # Chroma where-filter uses string equality on session_id; must not match regular session docs
        passed = (len(chunks) == 0)
        lat = (time.perf_counter() - t0) * 1000
        results.append(BenchmarkCaseResult(
            case_id="ISO_10",
            category="ISOLATION",
            file_type="TXT",
            scenario="Session ID containing path traversal characters ('../../admin_session')",
            expected_behavior="Isolated query; cannot access other session records",
            actual_behavior=f"Chunks returned: {len(chunks)}",
            passed=passed,
            failure_class=None if passed else "SECURITY",
            severity=None if passed else "P0",
            latency_ms=round(lat, 2)
        ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_isolation_suite():
    cases = run_cases()
    assert all(c.passed for c in cases)
