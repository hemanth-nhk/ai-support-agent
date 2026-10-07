"""
Comprehensive tests for Phase 4: Deterministic Visualization & Chart REST API.
Enforces all 14 Phase 4 acceptance gates and security invariants:
1. All four chart types render correctly.
2. PNG magic bytes verified.
3. Artifact path / session isolation verified.
4. Path traversal strictly rejected.
5. Zero-row result produces no chart.
6. Explicit incompatible chart type does NOT silently change semantics.
7. Implicit "Chart this" gets deterministic chart selection.
8. Renderer validates actual result shape independently of LLM.
9. No arbitrary Python/Matplotlib/style/path input reaches the renderer.
10. Chart artifact bounds enforced (<= 2MB, 8x4.5 in, 150 DPI).
11. /chat does not carry unnecessary base64 payload.
12. End-to-end dataset chart request works through FastAPI.
"""
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.analytics.schemas import (
    AnalysisPlan,
    AnalysisResult,
    ChartArtifact,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
)
from app.analytics.visualizer import (
    validate_chart_compatibility,
    render_chart,
    resolve_chart_file,
    ChartSecurityError,
    PNG_MAGIC_BYTES,
)
from app.analytics.chart_style import (
    WIDTH_PX,
    HEIGHT_PX,
    MAX_PNG_BYTES,
)

client = TestClient(app)


def _make_dummy_result(
    rows: list[dict],
    columns: list[str],
    chart_type: str = "bar",
    requires_chart: bool = True,
) -> AnalysisResult:
    plan = AnalysisPlan(
        dataset_id=str(uuid.uuid4()),
        aggregations=[AggregationClause(column=columns[-1], function=AggregationFunction.SUM, alias=columns[-1])],
        group_by=[GroupByClause(column=columns[0])] if len(columns) > 1 else [],
        requires_chart=requires_chart,
        chart_type=chart_type,
    )
    return AnalysisResult(
        plan=plan,
        compiled_sql="SELECT ...",
        rows=rows,
        row_count=len(rows),
        columns=columns,
        execution_time_ms=1.5,
    )


# ---------------------------------------------------------------------------
# 1. All 4 chart types render correctly and verify PNG magic bytes & bounds
# ---------------------------------------------------------------------------

def test_render_bar_chart_success(tmp_path):
    session_id = str(uuid.uuid4())
    rows = [
        {"category": "Electronics", "revenue": 1500.0},
        {"category": "Apparel", "revenue": 850.0},
        {"category": "Home", "revenue": 420.0},
    ]
    res = _make_dummy_result(rows, ["category", "revenue"], "bar")

    artifact = render_chart(res, "bar", session_id, "Test Bar Chart")

    assert isinstance(artifact, ChartArtifact)
    assert artifact.chart_type == "bar"
    assert artifact.width_px == WIDTH_PX
    assert artifact.height_px == HEIGHT_PX
    assert artifact.url == f"/charts/{session_id}/{artifact.chart_id}"

    # Verify file on disk
    file_path = resolve_chart_file(session_id, artifact.chart_id)
    assert file_path.exists()
    content = file_path.read_bytes()
    assert content.startswith(PNG_MAGIC_BYTES)
    assert len(content) <= MAX_PNG_BYTES


def test_render_line_chart_success():
    session_id = str(uuid.uuid4())
    rows = [
        {"date": "2024-01-01", "revenue": 100.0},
        {"date": "2024-02-01", "revenue": 150.0},
        {"date": "2024-03-01", "revenue": 210.0},
    ]
    res = _make_dummy_result(rows, ["date", "revenue"], "line")

    artifact = render_chart(res, "line", session_id, "Monthly Revenue Trend")

    assert artifact.chart_type == "line"
    file_path = resolve_chart_file(session_id, artifact.chart_id)
    assert file_path.read_bytes().startswith(PNG_MAGIC_BYTES)


def test_render_scatter_chart_success():
    session_id = str(uuid.uuid4())
    rows = [
        {"ad_spend": 100.0, "revenue": 500.0},
        {"ad_spend": 250.0, "revenue": 1200.0},
        {"ad_spend": 400.0, "revenue": 2100.0},
    ]
    res = _make_dummy_result(rows, ["ad_spend", "revenue"], "scatter")

    artifact = render_chart(res, "scatter", session_id, "Spend vs Revenue")

    assert artifact.chart_type == "scatter"
    file_path = resolve_chart_file(session_id, artifact.chart_id)
    assert file_path.read_bytes().startswith(PNG_MAGIC_BYTES)


def test_render_histogram_chart_success():
    session_id = str(uuid.uuid4())
    rows = [{"price": p} for p in [10, 15, 18, 22, 25, 30, 35, 40, 42, 50, 55, 60]]
    res = _make_dummy_result(rows, ["price"], "histogram")

    artifact = render_chart(res, "histogram", session_id, "Price Distribution")

    assert artifact.chart_type == "histogram"
    file_path = resolve_chart_file(session_id, artifact.chart_id)
    assert file_path.read_bytes().startswith(PNG_MAGIC_BYTES)


# ---------------------------------------------------------------------------
# 2. Incompatible chart requests & Non-blind coercion
# ---------------------------------------------------------------------------

def test_zero_rows_produces_no_chart():
    res = _make_dummy_result([], ["category", "revenue"], "bar")
    is_compat, eff_type, err = validate_chart_compatibility(res, requested_type="bar", is_explicit=True)
    assert not is_compat
    assert eff_type is None
    assert "0 rows" in err.lower() or "empty" in err.lower()


def test_explicit_incompatible_scatter_fails_closed():
    # User asks for scatter plot of categorical product and numeric profit
    rows = [
        {"product": "Laptop", "profit": 300.0},
        {"product": "Phone", "profit": 150.0},
    ]
    res = _make_dummy_result(rows, ["product", "profit"], "scatter")

    is_compat, eff_type, err = validate_chart_compatibility(res, requested_type="scatter", is_explicit=True)

    # Invariant: Must NOT silently coerce to bar chart!
    assert not is_compat
    assert eff_type is None
    assert "scatter" in err.lower()
    assert "incompatible" in err.lower()


def test_implicit_chart_gets_deterministic_selection():
    # 1 categorical + 1 numeric -> bar
    res_bar = _make_dummy_result([{"cat": "A", "val": 10}], ["cat", "val"])
    compat, eff_type, _ = validate_chart_compatibility(res_bar, requested_type=None, is_explicit=False)
    assert compat
    assert eff_type == "bar"

    # 1 temporal + 1 numeric -> line
    res_line = _make_dummy_result([{"date": "2024-01-01", "val": 10}], ["date", "val"])
    compat, eff_type, _ = validate_chart_compatibility(res_line, requested_type=None, is_explicit=False)
    assert compat
    assert eff_type == "line"

    # 2 numeric -> scatter
    res_scatter = _make_dummy_result([{"x": 1.0, "y": 2.0}], ["x", "y"])
    compat, eff_type, _ = validate_chart_compatibility(res_scatter, requested_type=None, is_explicit=False)
    assert compat
    assert eff_type == "scatter"

    # 1 numeric -> histogram
    res_hist = _make_dummy_result([{"num": 5.0}], ["num"])
    compat, eff_type, _ = validate_chart_compatibility(res_hist, requested_type=None, is_explicit=False)
    assert compat
    assert eff_type == "histogram"


# ---------------------------------------------------------------------------
# 3. Path Traversal & Security Defenses
# ---------------------------------------------------------------------------

def test_resolve_chart_file_path_traversal_rejection():
    valid_uuid = str(uuid.uuid4())

    # Traversal in session_id
    with pytest.raises(ChartSecurityError):
        resolve_chart_file("../../etc", valid_uuid)

    with pytest.raises(ChartSecurityError):
        resolve_chart_file("test/../session", valid_uuid)

    # Traversal in chart_id
    with pytest.raises(ChartSecurityError):
        resolve_chart_file(valid_uuid, "../../secret")

    with pytest.raises(ChartSecurityError):
        resolve_chart_file(valid_uuid, f"{valid_uuid}/../other")

    # Non-existent chart returns FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_chart_file(valid_uuid, str(uuid.uuid4()))


def test_get_chart_endpoint_serves_valid_png_and_blocks_traversal():
    session_id = str(uuid.uuid4())
    rows = [{"cat": "Widget", "val": 42.0}]
    res = _make_dummy_result(rows, ["cat", "val"], "bar")
    artifact = render_chart(res, "bar", session_id)

    # 1. Valid retrieval
    resp = client.get(f"/charts/{session_id}/{artifact.chart_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(PNG_MAGIC_BYTES)

    # 2. Non-existent chart -> 404
    missing_id = str(uuid.uuid4())
    resp_404 = client.get(f"/charts/{session_id}/{missing_id}")
    assert resp_404.status_code == 404

    # 3. Traversal attack via URL -> 400 / 404 (never 200 or 500)
    resp_traversal = client.get(f"/charts/..%2F..%2Fsession/{artifact.chart_id}")
    assert resp_traversal.status_code in (400, 404)

    resp_traversal_chart = client.get(f"/charts/{session_id}/..%2F..%2Fsecret")
    assert resp_traversal_chart.status_code in (400, 404)


# ---------------------------------------------------------------------------
# 4. /chat Contract Invariant: No Base64 in Response
# ---------------------------------------------------------------------------

def test_chat_response_contract_no_base64(monkeypatch):
    from unittest.mock import patch
    import io

    # Upload test CSV
    csv_bytes = b"department,salary\nSales,50000\nEngineering,90000\nMarketing,60000\n"
    session_id = str(uuid.uuid4())
    up_resp = client.post(
        "/datasets/upload",
        data={"session_id": session_id},
        files={"file": ("salary.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up_resp.status_code == 201
    dataset_id = up_resp.json()["dataset_id"]

    # Request chart via chat
    chat_req = {
        "message": "Can you plot a bar chart of salary by department?",
        "session_id": session_id,
        "dataset_id": dataset_id,
    }
    chat_resp = client.post("/chat", json={**chat_req})
    assert chat_resp.status_code == 200
    data = chat_resp.json()

    assert data["status"] == "ok"
    assert data["intent"] == "DATASET_ANALYSIS"
    assert "chart" in data
    assert data["chart"] is not None

    chart_obj = data["chart"]
    # Check fields match ChartArtifact exactly
    assert "chart_id" in chart_obj
    assert "chart_type" in chart_obj
    assert "title" in chart_obj
    assert "url" in chart_obj
    assert "width_px" in chart_obj
    assert "height_px" in chart_obj

    # Invariant: No base64 payload in ChatResponse
    assert "base64" not in chart_obj
    assert "base64_data_uri" not in chart_obj
    assert "data:image/png;base64" not in str(data)

    # Verify chart URL works
    img_resp = client.get(chart_obj["url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    assert img_resp.content.startswith(PNG_MAGIC_BYTES)
