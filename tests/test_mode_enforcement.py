"""
Regression tests for workspace mode enforcement and intent routing:
1. Support-mode questions stay in Support workflows even when a dataset is active in the session.
2. Order ID extraction supports alphanumeric formats (e.g. ORD-1001, #1001, order 1001).
3. Spurious dataset filters are prevented by preserving mode boundaries.
4. Missing-source handling (missing order ID, nonexistent order, missing document attachment) behaves predictably.
"""
import io
import uuid
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agent.router import extract_order_id, classify_intent
from app.agent.state import AgentState


@pytest.fixture
def client():
    return TestClient(app)


def test_order_id_extraction_alphanumeric():
    assert extract_order_id("What is the status of order 1001?") == 1001
    assert extract_order_id("What is the status of order ORD-1001?") == 1001
    assert extract_order_id("What is the status of ORD-1001?") == 1001
    assert extract_order_id("Track order #1001 please") == 1001
    assert extract_order_id("order id: 1001") == 1001
    assert extract_order_id("order id ORD-1001") == 1001
    assert extract_order_id("Where is my order?") is None


def test_order_status_known_workflow(client):
    """Test A: Known Support workflow returns order status for order 1001."""
    resp = client.post("/chat", json={
        "message": "What is the status of order 1001?",
        "session_id": "test_known_order_1001",
        "workspace_mode": "support",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "ORDER_STATUS"
    assert "1001" in data["response"]
    assert "Laptop" in data["response"]
    assert data["status"] == "ok"


def test_alphanumeric_order_support_mode(client):
    """Test B: Alphanumeric order format (ORD-1001) resolves correctly in Support mode."""
    resp = client.post("/chat", json={
        "message": "What is the status of order ORD-1001?",
        "session_id": "test_alphanumeric_ord_1001",
        "workspace_mode": "support",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "ORDER_STATUS"
    assert "1001" in data["response"]
    assert "Laptop" in data["response"]
    assert data["status"] == "ok"


def test_support_mode_not_misrouted_when_dataset_attached(client):
    """Verify that Support questions do not get hijacked into DATASET_ANALYSIS."""
    # 1. Upload a dataset with columns that match common support words: Product, Category, Payment, Status
    df = pd.DataFrame({
        "Product": ["Product A", "Product B"],
        "Category": ["Refunds", "Sales"],
        "Payment": ["Deducted", "Pending"],
        "Status": ["Delivered", "Processing"],
        "Amount": [100.0, 50.0],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    session_id = f"test_sess_mode_isolation_{uuid.uuid4().hex[:8]}"
    upload_resp = client.post(
        "/datasets/upload",
        files={"file": ("store_data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"session_id": session_id},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    # 2. Ask an order status question with dataset attached in Support mode
    resp_order = client.post("/chat", json={
        "message": "What is the status of order ORD-1001?",
        "session_id": session_id,
        "dataset_id": dataset_id,
        "workspace_mode": "support",
    })
    assert resp_order.status_code == 200
    data_order = resp_order.json()
    assert data_order["intent"] == "ORDER_STATUS"
    assert "1001" in data_order["response"]
    assert "Laptop" in data_order["response"]

    # 3. Ask a policy/refund question containing 'payment' and 'refund' in Support mode
    resp_refund = client.post("/chat", json={
        "message": "Can I get a refund if payment was deducted?",
        "session_id": session_id,
        "dataset_id": dataset_id,
        "workspace_mode": "support",
    })
    assert resp_refund.status_code == 200
    data_refund = resp_refund.json()
    # Must NOT be routed to DATASET_ANALYSIS or execute_analysis_plan
    assert data_refund["intent"] != "DATASET_ANALYSIS"
    assert data_refund["tool_used"] != "execute_analysis_plan"
    assert data_refund["intent"] == "FAQ"


def test_dataset_mode_routes_to_dataset_analysis(client):
    """Verify that in Dataset Intelligence mode, analytical questions reach DATASET_ANALYSIS."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Mouse"],
        "Category": ["Electronics", "Accessories"],
        "Sales": [1200.0, 30.0],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    session_id = f"test_sess_dataset_mode_{uuid.uuid4().hex[:8]}"
    upload_resp = client.post(
        "/datasets/upload",
        files={"file": ("sales_data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"session_id": session_id},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    resp = client.post("/chat", json={
        "message": "What is the total sales by category?",
        "session_id": session_id,
        "dataset_id": dataset_id,
        "workspace_mode": "dataset",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "DATASET_ANALYSIS"


def test_missing_source_handling(client):
    """Test C: Missing source handling for nonexistent orders, missing IDs, and missing docs."""
    # 1. Nonexistent order ID
    resp_unknown = client.post("/chat", json={
        "message": "What is the status of order 9999?",
        "session_id": "test_missing_order_id",
        "workspace_mode": "support",
    })
    assert resp_unknown.status_code == 200
    data_unknown = resp_unknown.json()
    assert data_unknown["intent"] == "ORDER_STATUS"
    assert data_unknown["status"] == "not_found"
    assert "couldn't find that order" in data_unknown["response"].lower()

    # 2. Missing order ID prompt
    resp_where = client.post("/chat", json={
        "message": "Where is my order?",
        "session_id": "test_missing_order_prompt",
        "workspace_mode": "support",
    })
    assert resp_where.status_code == 200
    data_where = resp_where.json()
    assert data_where["intent"] == "ORDER_STATUS"
    assert data_where["status"] == "not_found"
    assert "share the order id" in data_where["response"].lower()

    # 3. Missing document reference
    resp_doc = client.post("/chat", json={
        "message": "According to the uploaded document, what is the policy?",
        "session_id": "test_missing_doc_session",
        "workspace_mode": "support",
    })
    assert resp_doc.status_code == 200
    data_doc = resp_doc.json()
    assert data_doc["intent"] == "DOCUMENT_QUERY"
    assert data_doc["status"] == "not_found"
    assert "no document is available in this chat" in data_doc["response"].lower()


def test_order_delivery_estimates_formatting():
    """Test F: Distinguish past, future, delivered, and missing estimates without hallucinating."""
    from app.agent.nodes import _format_order_response

    # 1. Past estimated delivery date for Shipped order
    past_order = {
        "id": 1001,
        "product": "Laptop",
        "status": "Shipped",
        "estimated_delivery": "2026-09-15",
    }
    resp_past = _format_order_response(past_order)
    assert "Order 1001 (Laptop) is currently: Shipped" in resp_past
    assert "estimated delivery was 2026-09-15" in resp_past
    assert "date has passed" in resp_past
    assert "Delivered" not in resp_past or "Shipped" in resp_past

    # 2. Future estimated delivery date for Shipped order
    future_order = {
        "id": 1005,
        "product": "Monitor",
        "status": "Shipped",
        "estimated_delivery": "2099-01-01",
    }
    resp_future = _format_order_response(future_order)
    assert "Order 1005 (Monitor) is currently: Shipped, estimated delivery 2099-01-01." == resp_future

    # 3. Delivered order
    delivered_order = {
        "id": 1002,
        "product": "Wireless Mouse",
        "status": "Delivered",
        "estimated_delivery": "2026-09-05",
    }
    resp_deliv = _format_order_response(delivered_order)
    assert "Order 1002 (Wireless Mouse) is currently: Delivered, delivered on 2026-09-05." == resp_deliv

    # 4. Missing delivery date
    missing_date_order = {
        "id": 1006,
        "product": "USB Cable",
        "status": "Processing",
        "estimated_delivery": None,
    }
    resp_missing = _format_order_response(missing_date_order)
    assert resp_missing == "Order 1006 (USB Cable) is currently: Processing."


def test_empty_workspace_analysis_guidance(client):
    """Test E: Requesting analysis in an empty workspace provides actionable guidance without raw UUIDs."""
    sess_id = f"test_empty_ws_guidance_{uuid.uuid4().hex[:8]}"
    resp = client.post("/chat", json={
        "message": "tell me about sales analysis",
        "session_id": sess_id,
        "workspace_mode": "dataset",
        "dataset_id": None,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "DATASET_ANALYSIS"
    assert data["status"] == "not_found"
    assert "no active dataset is selected" in data["response"].lower()
    assert "upload" in data["response"].lower() or "switch dataset" in data["response"].lower()
    assert "not found" not in data["response"] or "dataset" in data["response"].lower()


def test_stale_deleted_dataset_explanation(client):
    """Test E & D: Request referencing a deleted dataset explains availability without raw UUIDs."""
    import io
    import pandas as pd

    df = pd.DataFrame({"Category": ["Electronics"], "Sales": [500]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    session_id = "test_stale_ds_session"

    upload_resp = client.post(
        "/datasets/upload",
        files={"file": ("to_delete.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"session_id": session_id},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    # Delete dataset permanently
    del_res = client.delete(f"/datasets/{dataset_id}?session_id={session_id}")
    assert del_res.status_code == 200

    # Query with the deleted dataset ID
    resp = client.post("/chat", json={
        "message": "tell me about sales analysis",
        "session_id": session_id,
        "workspace_mode": "dataset",
        "dataset_id": dataset_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "DATASET_ANALYSIS"
    assert data["status"] == "not_found"
    assert dataset_id not in data["response"]
    assert "no longer available" in data["response"].lower() or "removed" in data["response"].lower()


def test_remove_dataset_lifecycle_separation(client):
    """Test C: Remove Dataset detaches from conversation without destroying the underlying stored dataset."""
    import io
    import pandas as pd

    df = pd.DataFrame({"Category": ["Electronics"], "Sales": [500]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    session_id = f"test_detach_sess_{uuid.uuid4().hex[:8]}"

    upload_resp = client.post(
        "/datasets/upload",
        files={"file": ("detach_test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"session_id": session_id},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    # Create a conversation bound to this dataset
    cid = f"conv_detach_test_{uuid.uuid4().hex[:8]}"
    client.post("/conversations", json={
        "id": cid,
        "title": "Detach Test Analysis",
        "workspace_mode": "dataset",
        "dataset_id": dataset_id,
        "session_id": session_id,
    })

    # Detach (Remove) dataset from conversation
    patch_resp = client.patch(f"/conversations/{cid}", json={
        "dataset_id": None,
    })
    assert patch_resp.status_code == 200
    assert patch_resp.json()["dataset_id"] is None

    # Underlying dataset must STILL exist in storage
    get_ds = client.get(f"/datasets/{dataset_id}?session_id={session_id}")
    assert get_ds.status_code == 200
    assert get_ds.json()["dataset_id"] == dataset_id
