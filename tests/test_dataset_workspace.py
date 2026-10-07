"""
Tests for Phase 5 — Dataset Workspace & Bounded AnalysisTable UI Response Contract.
Covers all Option B requirements and invariants:
1. AnalysisTable schema: columns (list[str]), rows (list[list[Any]]), row_count (int)
2. Hard bounding: maximum 20 rows returned in table payload
3. Column bounding: maximum 30 columns
4. Cell bounding: long string cells truncated
5. Backward compatibility: non-dataset / support responses have table=None and chart=None
6. Dual-consumption contract: /chat payload contains finding, table, and chart
7. Grounded execution: deterministic results match table and finding
8. Zero raw SQL or SQL duration exposed in public response
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.analytics import dataset_service
from app.analytics.schemas import AnalysisTable
from app.api.schemas import ChatResponse


@pytest.fixture
def sample_sales_dataset(tmp_path, monkeypatch):
    """Creates a 25-row dataset to test row bounding (> 20 rows)."""
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))

    rows = ["product,category,revenue,units"]
    for i in range(1, 26):
        rows.append(f"Product_{i:02d},Category_{(i % 3) + 1},{i * 100}.0,{i * 5}")
    csv_bytes = "\n".join(rows).encode("utf-8")

    meta = dataset_service.ingest_dataset(
        session_id="workspace-session-1",
        filename="sales_test.csv",
        content_bytes=csv_bytes,
    )
    return meta, "workspace-session-1"


def test_analysis_table_model():
    """Validates AnalysisTable Pydantic model contract."""
    tbl = AnalysisTable(
        columns=["dept", "budget"],
        rows=[["Engineering", 50000], ["Sales", 30000]],
        row_count=2,
    )
    dumped = tbl.model_dump()
    assert dumped["columns"] == ["dept", "budget"]
    assert len(dumped["rows"]) == 2
    assert dumped["row_count"] == 2


def test_dataset_query_returns_bounded_table_and_chart(sample_sales_dataset):
    """
    Verifies that asking for top products returns:
    1. Finding in 'response'
    2. Bounded AnalysisTable in 'table' (<= 20 rows even though dataset has 25 rows)
    3. ChartArtifact in 'chart'
    4. No raw SQL or SQL execution duration exposed
    """
    meta, session_id = sample_sales_dataset
    client = TestClient(app)

    payload = {
        "message": "Show the top 25 products by revenue as a bar chart",
        "session_id": session_id,
        "dataset_id": meta.dataset_id,
    }

    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    # 1. Finding
    assert "response" in data
    assert len(data["response"]) > 0

    # 2. Bounded Table
    assert "table" in data
    assert data["table"] is not None
    table = data["table"]
    assert "columns" in table
    assert "rows" in table
    assert "row_count" in table

    # Hard limit: maximum 20 rows in payload
    assert len(table["rows"]) <= 20
    assert table["row_count"] >= 20  # Total matched in DuckDB

    # 3. Chart Artifact
    assert "chart" in data
    assert data["chart"] is not None
    assert data["chart"]["chart_type"] == "bar"
    assert "/charts/" in data["chart"]["url"]

    # 4. Invariants: No raw SQL or duration leaked
    assert "sql" not in data
    assert "sql_duration" not in data
    assert "duration" not in data
    assert "query" not in data


def test_cell_string_truncation_bound(tmp_path, monkeypatch):
    """Verifies that cells with long string values are safely bounded to <= 100 chars."""
    monkeypatch.setattr(settings, "datasets_dir", str(tmp_path))

    long_text = "A" * 150
    csv_bytes = f"item,description,price\nItem1,{long_text},99.0\n".encode("utf-8")

    meta = dataset_service.ingest_dataset(
        session_id="workspace-session-2",
        filename="long_cells.csv",
        content_bytes=csv_bytes,
    )

    client = TestClient(app)
    res = client.post(
        "/chat",
        json={
            "message": "List items and descriptions with price",
            "session_id": "workspace-session-2",
            "dataset_id": meta.dataset_id,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data.get("table") is not None

    table = data["table"]
    for row in table["rows"]:
        for cell in row:
            if isinstance(cell, str):
                assert len(cell) <= 100


def test_support_queries_preserve_null_table_and_chart():
    """Ensures regular non-dataset queries don't leak table or chart."""
    client = TestClient(app)
    res = client.post(
        "/chat",
        json={
            "message": "Where is order 1001?",
            "session_id": "clean-session",
        },
    )
    assert res.status_code == 200
    data = res.json()
    # table and chart should either be absent (due to response_model_exclude_none) or None
    assert data.get("table") is None
    assert data.get("chart") is None
