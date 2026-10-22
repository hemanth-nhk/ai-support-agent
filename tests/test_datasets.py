"""
Tests for Phase 1 — Dataset Ingestion, Storage & Fast Profiling.
Verifies all 18 test gate criteria:
1. CSV upload & profiling
2. XLSX upload & profiling
3. JSON upload & profiling
4. Unsupported extension rejection
5. Malformed CSV rejection
6. Malformed XLSX rejection
7. Malformed JSON rejection
8. Empty file rejection
9. Oversized file rejection
10. Duplicate column handling
11. Missing value profiling accuracy
12. Duplicate row profiling accuracy
13. Datatype detection
14. Session isolation in listing
15. Dataset deletion & cleanup
16. Unknown dataset 404 handling
17. Cross-session access 403 handling
18. Filename / path traversal sanitization
"""
import io
import json
from pathlib import Path
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient
import openpyxl

from app.main import app
from app.config import settings
from app.analytics import dataset_service
from app.analytics.dataset_service import (
    DatasetValidationError,
    DatasetNotFoundError,
    DatasetAccessDeniedError,
    validate_file,
    sanitize_filename,
)
from app.analytics.profiler import profile_dataset_bytes, DatasetProfileError


@pytest.fixture(autouse=True)
def isolated_datasets_env(monkeypatch):
    """Isolates dataset storage and sqlite db for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_datasets_store_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)
    monkeypatch.setattr(settings, "database_path", str(Path(temp_dir) / "test_support.db"))
    monkeypatch.setattr(settings, "session_secret_key", "d5a5c7e7b925bcbb2e8325c76339192e2a62a8d9ca85d4abcff1a2bf376c7d17")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Ingestion & Profiling Tests
# ---------------------------------------------------------------------------

def test_1_csv_upload_and_profile(client):
    """Test 1: Standard CSV upload and deterministic profiling."""
    csv_content = (
        "id,product,price,in_stock\n"
        "1,Widget,19.99,True\n"
        "2,Gadget,49.50,False\n"
        "3,Gizmo,9.99,True\n"
    ).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("inventory.csv", csv_content, "text/csv")},
    )

    assert res.status_code == 201
    data = res.json()
    assert "dataset_id" in data
    assert data["filename"] == "inventory.csv"
    assert data["format"] == "csv"
    assert data["row_count"] == 3
    assert data["column_count"] == 4

    profile = data["profile"]
    col_names = [c["name"] for c in profile["columns"]]
    assert col_names == ["id", "product", "price", "in_stock"]
    assert profile["missing_cells"] == 0
    assert profile["duplicate_rows"] == 0


def test_2_xlsx_upload_and_profile(client):
    """Test 2: Standard XLSX upload and deterministic profiling."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["order_id", "revenue", "region"])
    ws.append([101, 1500.50, "North"])
    ws.append([102, 2300.00, "South"])
    ws.append([103, 850.75, "East"])

    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("sales_q1.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    assert res.status_code == 201
    data = res.json()
    assert data["format"] == "xlsx"
    assert data["row_count"] == 3
    assert data["column_count"] == 3
    assert [c["name"] for c in data["profile"]["columns"]] == ["order_id", "revenue", "region"]


def test_3_json_upload_and_profile(client):
    """Test 3: Standard JSON upload (array of records) and profiling."""
    records = [
        {"customer_id": "C01", "name": "Alice", "score": 92.5},
        {"customer_id": "C02", "name": "Bob", "score": 88.0},
        {"customer_id": "C03", "name": "Charlie", "score": 79.0},
        {"customer_id": "C04", "name": "David", "score": 95.0},
    ]
    json_bytes = json.dumps(records).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-beta"},
        files={"file": ("customers.json", json_bytes, "application/json")},
    )

    assert res.status_code == 201
    data = res.json()
    assert data["format"] == "json"
    assert data["row_count"] == 4
    assert data["column_count"] == 3
    assert data["profile"]["duplicate_rows"] == 0


def test_4_unsupported_extension_rejected(client):
    """Test 4: Unsupported extension rejected with 400 Bad Request."""
    dummy_bytes = b"header1,header2\n1,2"
    for ext in ("data.parquet", "document.pdf", "notes.txt", "script.py", "archive.zip"):
        res = client.post(
            "/datasets/upload",
            data={"session_id": "sess-alpha"},
            files={"file": (ext, dummy_bytes, "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]


def test_5_malformed_csv_rejected(client):
    """Test 5: Malformed/binary corrupted CSV rejected."""
    # Binary random garbage that is not valid CSV rows
    corrupt_bytes = b"\x00\x01\x02\x03\x04\xff\xfe\xfd\x00\x00" * 20
    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("corrupt.csv", corrupt_bytes, "text/csv")},
    )
    # Pandas parser either fails on NUL byte or raises parser error
    assert res.status_code == 400


def test_6_malformed_xlsx_rejected(client):
    """Test 6: Corrupted/non-zip XLSX rejected with 400."""
    corrupt_bytes = b"This is plainly not a zip or xlsx file."
    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("broken.xlsx", corrupt_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert res.status_code == 400
    assert "Excel" in res.json()["detail"] or "parse" in res.json()["detail"].lower()


def test_7_malformed_json_rejected(client):
    """Test 7: Malformed syntax or non-tabular JSON rejected."""
    bad_syntax = b'{"name": "Alice", "broken": '
    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("broken.json", bad_syntax, "application/json")},
    )
    assert res.status_code == 400
    assert "JSON" in res.json()["detail"]

    # Primitive non-tabular JSON
    primitive_json = b"12345"
    res2 = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("number.json", primitive_json, "application/json")},
    )
    assert res2.status_code == 400


def test_8_empty_file_rejected(client):
    """Test 8: 0-byte file rejected."""
    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_9_oversized_file_rejected(monkeypatch):
    """Test 9: File exceeding configured max file size rejected."""
    # Set limit to 1MB for unit test verification
    monkeypatch.setattr(settings, "max_dataset_file_size_mb", 1)
    huge_bytes = b"x" * (2 * 1024 * 1024)

    with pytest.raises(DatasetValidationError) as exc:
        validate_file("huge.csv", huge_bytes)
    assert "exceeds maximum allowed size" in str(exc.value)


def test_10_duplicate_column_handling(client):
    """Test 10: Duplicate column headers in input are handled cleanly."""
    csv_content = (
        "item,metric,metric\n"
        "A,10,20\n"
        "B,30,40\n"
    ).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("duplicates.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["column_count"] == 3
    col_names = [c["name"] for c in data["profile"]["columns"]]
    assert col_names[0] == "item"
    assert "metric" in col_names[1]
    assert "metric" in col_names[2]


def test_11_missing_value_profiling(client):
    """Test 11: Missing cells and percentages are profiled accurately."""
    # 4 rows: age has 2 missing (50%), salary has 1 missing (25%) -> total 3 missing cells
    csv_content = (
        "name,age,salary\n"
        "Alice,25,50000\n"
        "Bob,,60000\n"
        "Charlie,35,\n"
        "David,,\n"
    ).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("staff.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    profile = res.json()["profile"]

    assert profile["missing_cells"] == 4  # Bob:age, Charlie:salary, David:age, David:salary
    cols = {c["name"]: c for c in profile["columns"]}
    assert cols["age"]["missing_count"] == 2
    assert cols["age"]["missing_percentage"] == 0.5
    assert cols["salary"]["missing_count"] == 2
    assert cols["salary"]["missing_percentage"] == 0.5


def test_12_duplicate_row_profiling(client):
    """Test 12: Duplicate rows are accurately counted."""
    csv_content = (
        "sku,category\n"
        "S1,Electronics\n"
        "S2,Apparel\n"
        "S1,Electronics\n"  # duplicate of row 1
        "S3,Home\n"
        "S1,Electronics\n"  # duplicate of row 1
    ).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("skus.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    profile = res.json()["profile"]
    assert profile["row_count"] == 5
    assert profile["duplicate_rows"] == 2


def test_13_datatype_detection(client):
    """Test 13: Accurate data type inference for int, float, string, boolean."""
    csv_content = (
        "count,rating,name,active\n"
        "10,4.8,Alpha,True\n"
        "20,3.5,Beta,False\n"
    ).encode("utf-8")

    res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-alpha"},
        files={"file": ("types.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    cols = {c["name"]: c["dtype"] for c in res.json()["profile"]["columns"]}

    assert "int" in cols["count"]
    assert "float" in cols["rating"]
    assert any(t in cols["name"] for t in ("object", "string", "str"))
    assert "bool" in cols["active"]


def test_14_session_isolation(client):
    """Test 14: Datasets are strictly isolated between sessions."""
    csv_a = b"col1,col2\n1,2"
    csv_b = b"valA,valB\n10,20"

    res_a = client.post(
        "/datasets/upload",
        data={"session_id": "session-111"},
        files={"file": ("dataset_a.csv", csv_a, "text/csv")},
    )
    res_b = client.post(
        "/datasets/upload",
        data={"session_id": "session-222"},
        files={"file": ("dataset_b.csv", csv_b, "text/csv")},
    )

    token_a = res_a.json().get("session_token")
    token_b = res_b.json().get("session_token")

    # Session 111 list should only contain dataset_a
    list_a = client.get("/datasets", params={"session_id": "session-111"}, headers={"X-Session-Token": token_a}).json()
    assert len(list_a) == 1
    assert list_a[0]["filename"] == "dataset_a.csv"

    # Session 222 list should only contain dataset_b
    list_b = client.get("/datasets", params={"session_id": "session-222"}, headers={"X-Session-Token": token_b}).json()
    assert len(list_b) == 1
    assert list_b[0]["filename"] == "dataset_b.csv"


def test_15_dataset_deletion(client):
    """Test 15: Dataset deletion cleans SQLite and filesystem."""
    csv_content = b"x,y\n1,2"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "sess-delete"},
        files={"file": ("to_delete.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["dataset_id"]

    del_res = client.delete(f"/datasets/{dataset_id}?session_id=sess-delete")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Re-fetching must return 404
    get_res = client.get(f"/datasets/{dataset_id}?session_id=sess-delete")
    assert get_res.status_code == 404


def test_16_unknown_dataset_404(client):
    """Test 16: Requesting or deleting unknown UUID returns 404."""
    non_existent = "00000000-0000-0000-0000-000000000000"
    get_res = client.get(f"/datasets/{non_existent}?session_id=any-session")
    assert get_res.status_code == 404

    del_res = client.delete(f"/datasets/{non_existent}?session_id=any-session")
    assert del_res.status_code == 404


def test_17_cross_session_access_forbidden(client):
    """Test 17: Querying or deleting another session's dataset returns 403 Forbidden."""
    csv_content = b"a,b\n1,2"
    res = client.post(
        "/datasets/upload",
        data={"session_id": "owner-session"},
        files={"file": ("private.csv", csv_content, "text/csv")},
    )
    dataset_id = res.json()["dataset_id"]

    # Other session tries to read
    peek_res = client.get(f"/datasets/{dataset_id}?session_id=attacker-session")
    assert peek_res.status_code == 403

    # Other session tries to delete
    hack_res = client.delete(f"/datasets/{dataset_id}?session_id=attacker-session")
    assert hack_res.status_code == 403

    # Victim's dataset must remain intact
    intact_res = client.get(f"/datasets/{dataset_id}?session_id=owner-session")
    assert intact_res.status_code == 200


def test_18_filename_path_traversal_sanitized(client):
    """Test 18: Filename traversal attacks are sanitized and contained."""
    malicious_name = "../../../../etc/passwd.csv"
    csv_content = b"col1,col2\n1,2"

    res = client.post(
        "/datasets/upload",
        data={"session_id": "traversal-sess"},
        files={"file": (malicious_name, csv_content, "text/csv")},
    )
    assert res.status_code == 201
    data = res.json()
    assert "/" not in data["filename"]
    assert ".." not in data["filename"]
    assert data["filename"] == "passwd.csv"


# ---------------------------------------------------------------------------
# Upload Contract & Edge Case Regression Tests (Fix HTTP 422)
# ---------------------------------------------------------------------------

def test_upload_dataset_without_session_id_succeeds(client):
    """Uploading without session_id generates a fallback session_id and succeeds."""
    csv_content = b"colA,colB\n10,20\n30,40"
    res = client.post(
        "/datasets/upload",
        files={"file": ("no_session.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    data = res.json()
    assert "dataset_id" in data
    assert data["row_count"] == 2
    assert data["column_count"] == 2


def test_upload_dataset_empty_session_id_succeeds(client):
    """Uploading with empty session_id ('') generates a fallback session_id and succeeds."""
    csv_content = b"colA,colB\n10,20\n30,40"
    res = client.post(
        "/datasets/upload",
        data={"session_id": ""},
        files={"file": ("empty_session.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 201
    data = res.json()
    assert "dataset_id" in data


def test_upload_dataset_missing_file_returns_422(client):
    """Missing file multipart parameter returns FastAPI 422 Unprocessable Entity."""
    res = client.post(
        "/datasets/upload",
        data={"session_id": "some-session"},
    )
    assert res.status_code == 422
    err_body = res.json()
    assert "detail" in err_body
    assert any(err["loc"] == ["body", "file"] for err in err_body["detail"])


def test_upload_dataset_empty_file_returns_400(client):
    """Empty 0-byte file returns 400 Bad Request."""
    res = client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_upload_dataset_unsupported_type_returns_400(client):
    """Unsupported file extension returns 400 Bad Request."""
    res = client.post(
        "/datasets/upload",
        files={"file": ("script.py", b"print('hello')", "text/x-python")},
    )
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_upload_and_run_analysis_query(client):
    """Uploaded dataset can be bound to conversation and analyzed via /chat."""
    csv_content = (
        "region,sales\n"
        "North,1000\n"
        "South,2000\n"
        "East,1500\n"
        "West,3000\n"
    ).encode("utf-8")

    # 1. Upload dataset
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "regional-session"},
        files={"file": ("regional_sales.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # 2. Create conversation
    conv_res = client.post(
        "/conversations",
        json={
            "session_id": "regional-session",
            "title": "Regional Analysis",
            "workspace_mode": "dataset",
            "dataset_id": dataset_id,
        },
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    # 3. Run analysis query
    chat_res = client.post(
        "/chat",
        json={
            "message": "Which region has the highest sales?",
            "session_id": conv_id,
            "dataset_id": dataset_id,
        },
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data.get("intent") == "DATASET_ANALYSIS" or "West" in chat_data.get("response", "")


# ---------------------------------------------------------------------------
# Dataset Deletion Ownership & Isolation Regression Tests
# ---------------------------------------------------------------------------

def test_delete_dataset_owner_succeeds(client):
    """Owner session can delete their own dataset."""
    csv_content = b"x,y\n10,20\n30,40"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "owner-session-ok"},
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    del_res = client.delete(f"/datasets/{dataset_id}?session_id=owner-session-ok")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Confirm it is gone
    get_res = client.get(f"/datasets/{dataset_id}?session_id=owner-session-ok")
    assert get_res.status_code == 404


def test_delete_dataset_cross_session_forbidden(client):
    """Another session cannot delete the dataset, and victim's dataset remains intact."""
    csv_content = b"x,y\n1,2"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session"},
        files={"file": ("victim.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    del_res = client.delete(f"/datasets/{dataset_id}?session_id=attacker-session")
    assert del_res.status_code == 403
    assert "Access denied" in del_res.json()["detail"]

    # Victim dataset remains intact
    get_res = client.get(f"/datasets/{dataset_id}?session_id=victim-session")
    assert get_res.status_code == 200


def test_delete_dataset_missing_session_id_rejected(client):
    """Omitting session_id is rejected with 422 without deleting anything."""
    csv_content = b"x,y\n1,2"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session-missing"},
        files={"file": ("victim_missing.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # Delete without session_id query param
    del_res = client.delete(f"/datasets/{dataset_id}")
    assert del_res.status_code == 422

    # Dataset still exists
    get_res = client.get(f"/datasets/{dataset_id}?session_id=victim-session-missing")
    assert get_res.status_code == 200


def test_delete_dataset_empty_session_id_rejected(client):
    """Empty session_id is rejected with 422 without deleting anything."""
    csv_content = b"x,y\n1,2"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session-empty"},
        files={"file": ("victim_empty.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # Delete with empty session_id query param
    del_res = client.delete(f"/datasets/{dataset_id}?session_id=")
    assert del_res.status_code == 422

    # Dataset still exists
    get_res = client.get(f"/datasets/{dataset_id}?session_id=victim-session-empty")
    assert get_res.status_code == 200


def test_delete_dataset_whitespace_session_id_rejected(client):
    """Whitespace-only session_id is rejected with 400 without deleting anything."""
    csv_content = b"x,y\n1,2"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session-space"},
        files={"file": ("victim_space.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # Delete with whitespace session_id query param
    del_res = client.delete(f"/datasets/{dataset_id}?session_id=%20%20%20")
    assert del_res.status_code == 400
    assert "valid, non-empty session_id is required" in del_res.json()["detail"]

    # Dataset still exists
    get_res = client.get(f"/datasets/{dataset_id}?session_id=victim-session-space")
    assert get_res.status_code == 200


def test_delete_dataset_nonexistent_returns_404(client):
    """Nonexistent dataset with valid session ID returns 404."""
    del_res = client.delete("/datasets/nonexistent-dataset-id?session_id=any-session")
    assert del_res.status_code == 404


def test_delete_dataset_service_layer_enforcement():
    """Service layer directly rejects empty or whitespace IDs and enforces ownership."""
    # Empty or whitespace dataset_id
    with pytest.raises(DatasetValidationError):
        dataset_service.delete_dataset("", "some-session")
    with pytest.raises(DatasetValidationError):
        dataset_service.delete_dataset("   ", "some-session")

    # Empty or whitespace session_id
    with pytest.raises(DatasetValidationError):
        dataset_service.delete_dataset("some-dataset", "")
    with pytest.raises(DatasetValidationError):
        dataset_service.delete_dataset("some-dataset", "   ")

    # Nonexistent dataset
    with pytest.raises(DatasetNotFoundError):
        dataset_service.delete_dataset("unknown-id-12345", "some-session")


def test_frontend_dataset_deletion_sends_session_id():
    """Verifies static/app.js sends session_id query parameter and derives it correctly."""
    app_js_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
    content = app_js_path.read_text(encoding="utf-8")

    # executeDeleteDataset must include ?session_id=
    assert "fetch(`/datasets/${encodeURIComponent(datasetId)}${queryParam}`" in content
    assert "const queryParam = resolvedSessionId ? `?session_id=${encodeURIComponent(resolvedSessionId)}` : '';" in content

    # promptDeleteDataset must accept and store sessionId
    assert "function promptDeleteDataset(datasetId, datasetName, sessionId)" in content
    assert "pendingDeleteDatasetSessionId = sessionId" in content

    # renderSwitchDatasetList must pass ds.session_id
    assert "promptDeleteDataset(ds.dataset_id, ds.filename, ds.session_id);" in content

    # confirmDelete must pass dsSessionId to executeDeleteDataset
    assert "await executeDeleteDataset(dsId, dsSessionId);" in content


# ---------------------------------------------------------------------------
# Cross-Session Dataset Isolation Bypass Regression Tests
# ---------------------------------------------------------------------------

def test_cross_session_dataset_association_rejected_on_create(client):
    """Attacker cannot bind victim's dataset during conversation creation (returns 403). Nonexistent returns 404."""
    csv_content = b"user,score\nAlice,100\nBob,200"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session-create"},
        files={"file": ("victim_data.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    victim_ds_id = upload_res.json()["dataset_id"]

    # Attacker with explicit attacker-session tries to bind victim dataset
    bad_create = client.post(
        "/conversations",
        json={
            "title": "Attacker Session Conv",
            "workspace_mode": "dataset",
            "dataset_id": victim_ds_id,
            "session_id": "attacker-session",
        },
    )
    assert bad_create.status_code == 403
    assert "Access denied" in bad_create.json()["detail"]

    # Attacker without session_id (or newly generated ID) tries to bind victim dataset
    attacker_client = TestClient(app)
    bad_create_no_sess = attacker_client.post(
        "/conversations",
        json={
            "title": "Attacker Conv No Session",
            "workspace_mode": "dataset",
            "dataset_id": victim_ds_id,
        },
    )
    assert bad_create_no_sess.status_code == 403
    assert "Access denied" in bad_create_no_sess.json()["detail"]

    # Nonexistent dataset returns 404
    not_found_create = client.post(
        "/conversations",
        json={
            "title": "Nonexistent Conv",
            "workspace_mode": "dataset",
            "dataset_id": "nonexistent-ds-9999",
            "session_id": "victim-session-create",
        },
    )
    assert not_found_create.status_code == 404


def test_cross_session_dataset_association_rejected_on_patch(client):
    """Attacker cannot bind victim's dataset via PATCH (returns 403). Nonexistent returns 404. Rejected attempts leave state untouched."""
    from app.api.rest import _sessions
    from app.services import conversation_service

    csv_content = b"item,price\nWidget,10\nGadget,20"
    upload_res = client.post(
        "/datasets/upload",
        data={"session_id": "victim-session-patch"},
        files={"file": ("victim_prices.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 201
    victim_ds_id = upload_res.json()["dataset_id"]

    # Create attacker's own conversation without dataset
    conv_res = client.post(
        "/conversations",
        json={
            "title": "Attacker Safe Conv",
            "workspace_mode": "dataset",
        },
    )
    assert conv_res.status_code == 201
    attacker_cid = conv_res.json()["id"]

    # Setup initial analytical plan in DB and memory
    conversation_service.update_conversation(attacker_cid, active_analysis_plan={"metric": "safe_metric"})
    _sessions[attacker_cid] = {
        "dataset_id": None,
        "active_analysis_plan": {"metric": "safe_metric"},
        "analysis_plan": {"metric": "safe_metric"},
    }

    # Attacker attempts PATCH to associate victim's dataset
    patch_res = client.patch(
        f"/conversations/{attacker_cid}",
        json={
            "dataset_id": victim_ds_id,
            "session_id": "attacker-session",
        },
    )
    assert patch_res.status_code == 403
    assert "Access denied" in patch_res.json()["detail"]

    # Verify conversation metadata, plans, and in-memory state remain UNTOUCHED
    conv_after = conversation_service.get_conversation(attacker_cid)
    assert conv_after["dataset_id"] is None
    assert conv_after["active_analysis_plan"] == {"metric": "safe_metric"}
    assert _sessions[attacker_cid]["active_analysis_plan"] == {"metric": "safe_metric"}

    # Attacker attempts PATCH with nonexistent dataset -> 404
    patch_404 = client.patch(
        f"/conversations/{attacker_cid}",
        json={"dataset_id": "nonexistent-ds-404"},
    )
    assert patch_404.status_code == 404
    assert _sessions[attacker_cid]["active_analysis_plan"] == {"metric": "safe_metric"}


def test_legitimate_dataset_association_and_switching(client):
    """Legitimate owner can bind and switch datasets between conversations."""
    from app.services import conversation_service

    csv_1 = b"col1,col2\n1,2"
    csv_2 = b"col1,col2\n3,4"

    up1 = client.post(
        "/datasets/upload",
        data={"session_id": "owner-switch-session"},
        files={"file": ("ds1.csv", csv_1, "text/csv")},
    )
    assert up1.status_code == 201
    ds1_id = up1.json()["dataset_id"]

    up2 = client.post(
        "/datasets/upload",
        data={"session_id": "owner-switch-session"},
        files={"file": ("ds2.csv", csv_2, "text/csv")},
    )
    assert up2.status_code == 201
    ds2_id = up2.json()["dataset_id"]

    # Create conversation bound to ds1
    conv = client.post(
        "/conversations",
        json={
            "title": "Owner Analysis",
            "workspace_mode": "dataset",
            "dataset_id": ds1_id,
            "session_id": "owner-switch-session",
        },
    ).json()
    cid = conv["id"]
    assert conv["dataset_id"] == ds1_id

    # Switch to ds2 via PATCH
    patch_res = client.patch(
        f"/conversations/{cid}",
        json={
            "dataset_id": ds2_id,
            "session_id": "owner-switch-session",
        },
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["dataset_id"] == ds2_id

    # Switch to None (detach)
    patch_none = client.patch(
        f"/conversations/{cid}",
        json={"dataset_id": None},
    )
    assert patch_none.status_code == 200
    assert patch_none.json()["dataset_id"] is None


def test_cross_session_defense_in_depth_get_dataset_isolated(client):
    """Direct and conversation-based cross-session access to dataset remains forbidden."""
    from app.analytics.dataset_service import get_dataset, DatasetAccessDeniedError
    import pytest

    csv_content = b"secret,value\nkey,12345"
    up = client.post(
        "/datasets/upload",
        data={"session_id": "victim-secret-session"},
        files={"file": ("secret.csv", csv_content, "text/csv")},
    )
    ds_id = up.json()["dataset_id"]

    # 1. Direct cross-session read returns 403
    direct_read = client.get(f"/datasets/{ds_id}?session_id=attacker-session")
    assert direct_read.status_code == 403

    # 2. Service layer rejects unrelated session
    with pytest.raises(DatasetAccessDeniedError):
        get_dataset(ds_id, session_id="attacker-session")

    # 3. Legitimate owner can read
    owner_read = client.get(f"/datasets/{ds_id}?session_id=victim-secret-session")
    assert owner_read.status_code == 200
    assert owner_read.json()["dataset_id"] == ds_id


def test_upload_dataset_early_size_limit_rejection(client, monkeypatch):
    """Verifies that oversized dataset uploads are rejected with HTTP 413 before full buffering (SEC-01)."""
    from app.config import settings
    # Set to a tiny limit (0.0001 MB ~ 104 bytes)
    monkeypatch.setattr(settings, "max_dataset_file_size_mb", 0.0001)
    oversized_csv = b"col1,col2\n" + (b"val1,val2\n" * 50)
    res = client.post(
        "/datasets/upload",
        files={"file": ("oversized.csv", oversized_csv, "text/csv")},
        data={"session_id": "session-ds-oversized"},
    )
    assert res.status_code == 413
    assert "exceeds maximum allowed size" in res.json()["detail"].lower()
