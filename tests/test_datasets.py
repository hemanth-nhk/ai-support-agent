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

    # Session 111 list should only contain dataset_a
    list_a = client.get("/datasets", params={"session_id": "session-111"}).json()
    assert len(list_a) == 1
    assert list_a[0]["filename"] == "dataset_a.csv"

    # Session 222 list should only contain dataset_b
    list_b = client.get("/datasets", params={"session_id": "session-222"}).json()
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
    get_res = client.get(f"/datasets/{non_existent}")
    assert get_res.status_code == 404

    del_res = client.delete(f"/datasets/{non_existent}")
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
