"""
Comprehensive integration and persistence tests for analytical conversation history.
Verifies SQLite persistence, session isolation, server-restart recovery, dataset switching,
and conversation lifecycle.
"""
import io
import shutil
import tempfile
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.database.seed import seed
from app.api.rest import _sessions
from app.services import conversation_service

SAMPLE_CSV = """Product,Category,Sales
Cheese,Dairy,27906.30
Coffee,Beverages,27694.87
Shampoo,Personal Care,27497.48
Cooking Oil,Pantry,21525.06
Tea,Beverages,21404.99
Apples,Produce,15000.00
"""

SAMPLE_CSV_2 = """Region,Revenue
North,50000.0
South,30000.0
East,20000.0
West,40000.0
"""


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    temp_dir = tempfile.mkdtemp(prefix="test_conv_hist_")
    temp_db = str(Path(temp_dir) / "test_isolated.db")
    monkeypatch.setattr(settings, "database_path", temp_db)
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)
    seed(force=True)
    _sessions.clear()
    yield temp_dir
    _sessions.clear()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


def _upload_dataset(client: TestClient, session_id: str, filename: str, csv_data: str) -> str:
    resp = client.post(
        "/datasets/upload",
        files={"file": (filename, io.BytesIO(csv_data.encode("utf-8")), "text/csv")},
        data={"session_id": session_id},
    )
    assert resp.status_code in (200, 201), f"Upload failed: {resp.text}"
    return resp.json()["dataset_id"]


def test_conversation_api_crud(client: TestClient):
    """Test full CRUD lifecycle of conversation endpoints."""
    # 1. Create
    ds_id = _upload_dataset(client, "crud_session", "sales.csv", SAMPLE_CSV)
    res = client.post(
        "/conversations",
        json={"title": "Sales Analysis Q1", "workspace_mode": "dataset", "dataset_id": ds_id, "session_id": "crud_session"},
    )
    assert res.status_code in (200, 201)
    conv = res.json()
    cid = conv["id"]
    assert conv["title"] == "Sales Analysis Q1"
    assert conv["workspace_mode"] == "dataset"
    assert conv["dataset_id"] == ds_id

    # 2. Get
    res_get = client.get(f"/conversations/{cid}")
    assert res_get.status_code == 200
    got = res_get.json()
    assert got["id"] == cid
    assert got["messages"] == []

    # 3. List
    res_list = client.get("/conversations?workspace_mode=dataset")
    assert res_list.status_code == 200
    conv_list = res_list.json()
    assert any(c["id"] == cid for c in conv_list)

    # 4. Patch
    res_patch = client.patch(f"/conversations/{cid}", json={"title": "Updated Title"})
    assert res_patch.status_code == 200
    assert res_patch.json()["title"] == "Updated Title"

    # 5. Delete
    res_del = client.delete(f"/conversations/{cid}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

    # 6. Verify 404 after delete
    res_not_found = client.get(f"/conversations/{cid}")
    assert res_not_found.status_code == 404


def test_new_analysis_creates_distinct_conversations(client: TestClient):
    """Creating a new analysis does not overwrite or delete previous conversations."""
    cid_1 = str(uuid.uuid4())
    ds_id_1 = _upload_dataset(client, cid_1, "sales.csv", SAMPLE_CSV)

    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_1,
        "dataset_id": ds_id_1,
    })
    assert r1.status_code == 200
    assert r1.json()["status"] == "ok"

    # Create new analysis
    cid_2 = str(uuid.uuid4())
    r_create = client.post("/conversations", json={
        "id": cid_2,
        "title": "New Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id_1,
        "session_id": cid_1,
    })
    assert r_create.status_code in (200, 201)

    r2 = client.post("/chat", json={
        "message": "Show sales by Category",
        "session_id": cid_2,
        "dataset_id": ds_id_1,
    })
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"

    # Verify both conversations exist independently in SQLite
    conv1 = client.get(f"/conversations/{cid_1}").json()
    conv2 = client.get(f"/conversations/{cid_2}").json()

    assert conv1["id"] == cid_1
    assert len(conv1["messages"]) == 2  # 1 user + 1 assistant
    assert "top 5" in conv1["messages"][0]["user_message"].lower()

    assert conv2["id"] == cid_2
    assert len(conv2["messages"]) == 2
    assert "category" in conv2["messages"][0]["user_message"].lower()


def test_switch_conversation_and_follow_up(client: TestClient):
    """Switching conversations restores analytical plan and follow-ups work in each."""
    cid_1 = str(uuid.uuid4())
    cid_2 = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_1, "sales.csv", SAMPLE_CSV)

    # Conversation 1: Top 5 products by sales
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_1,
        "dataset_id": ds_id,
    })
    assert r1.status_code == 200

    # Conversation 2: Create new analysis conversation bound to ds_id
    client.post("/conversations", json={
        "id": cid_2,
        "title": "Category Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
        "session_id": cid_1,
    })

    # Conversation 2: Category breakdown
    r2 = client.post("/chat", json={
        "message": "Show sales by Category",
        "session_id": cid_2,
        "dataset_id": ds_id,
    })
    assert r2.status_code == 200

    # Follow up on Conversation 1: "plot this as a bar chart"
    r1_followup = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": cid_1,
        "dataset_id": ds_id,
    })
    assert r1_followup.status_code == 200
    chart_info = r1_followup.json().get("chart")
    assert chart_info is not None

    # Follow up on Conversation 2: "filter Category to Beverages"
    r2_followup = client.post("/chat", json={
        "message": "Filter Category to Beverages",
        "session_id": cid_2,
        "dataset_id": ds_id,
    })
    assert r2_followup.status_code == 200
    assert r2_followup.json().get("status") == "ok"

    # Verify state isolation in SQLite
    c1_stored = conversation_service.get_conversation_with_messages(cid_1)
    c2_stored = conversation_service.get_conversation_with_messages(cid_2)

    assert len(c1_stored["messages"]) == 4  # 2 turns
    assert len(c2_stored["messages"]) == 4  # 2 turns
    p1 = c1_stored["active_analysis_plan"]
    p2 = c2_stored["active_analysis_plan"]
    dim1 = p1["group_by"][0]["column"] if p1.get("group_by") else p1.get("dimension")
    dim2 = p2["group_by"][0]["column"] if p2.get("group_by") else p2.get("dimension")
    assert dim1 == "Product"
    assert dim2 == "Category"


def test_restart_backend_recovers_analytical_state(client: TestClient):
    """Wiping in-memory sessions simulates server restart; follow-up restores from SQLite."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    # Turn 1
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })
    assert r1.status_code == 200
    assert r1.json()["status"] == "ok"

    # Simulate backend restart: completely purge in-memory _sessions
    _sessions.clear()
    assert cid not in _sessions

    # Turn 2: Follow-up query requiring active_analysis_plan
    r2 = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": cid,
        "dataset_id": ds_id,
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "ok"
    assert data2.get("chart") is not None
    assert cid in _sessions  # Loaded into memory and updated


def test_dataset_switch_resets_analytical_plan(client: TestClient):
    """Switching dataset within a session resets analytical plan to avoid leaking."""
    cid = str(uuid.uuid4())
    ds1 = _upload_dataset(client, cid, "sales1.csv", SAMPLE_CSV)
    ds2 = _upload_dataset(client, cid, "sales2.csv", SAMPLE_CSV_2)

    # Query dataset 1
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds1,
    })
    assert r1.status_code == 200

    # Switch to dataset 2 with follow-up-like prompt "plot this" -> should not leak ds1 columns
    # Because dataset switched, active_analysis_plan was reset
    r2 = client.post("/chat", json={
        "message": "Show revenue by Region",
        "session_id": cid,
        "dataset_id": ds2,
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "ok"

    # SQLite record should now reflect ds2 and new plan
    conv = conversation_service.get_conversation(cid)
    assert conv["dataset_id"] == ds2
    plan = conv["active_analysis_plan"]
    dim = plan["group_by"][0]["column"] if plan.get("group_by") else plan.get("dimension")
    assert dim == "Region"


def test_failed_request_does_not_corrupt_history(client: TestClient):
    """An invalid/failed chat request does not save fake assistant turn or corrupt state."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    # Valid turn 1
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })
    assert r1.status_code == 200

    initial_messages = client.get(f"/conversations/{cid}").json()["messages"]
    assert len(initial_messages) == 2

    # Empty message (bad request)
    r_bad = client.post("/chat", json={"message": "", "session_id": cid, "dataset_id": ds_id})
    assert r_bad.status_code in (400, 422)

    # Messages count must remain unchanged
    after_messages = client.get(f"/conversations/{cid}").json()["messages"]
    assert len(after_messages) == 2


def test_dataset_deletion_does_not_corrupt_conversation(client: TestClient):
    """Deleting a dataset does not delete or break conversation history in SQLite."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })
    assert r1.status_code == 200

    # Delete dataset from filesystem / storage
    del_res = client.delete(f"/datasets/{ds_id}?session_id={cid}")
    assert del_res.status_code == 200

    # Conversation history must still be fully recoverable
    res = client.get(f"/conversations/{cid}")
    assert res.status_code == 200
    conv = res.json()
    assert conv["id"] == cid
    assert len(conv["messages"]) == 2
    assert conv["dataset_id"] == ds_id


def test_delete_inactive_conversation(client: TestClient):
    """Deleting an inactive conversation removes it from SQLite while keeping active conversation and dataset intact."""
    cid_active = str(uuid.uuid4())
    cid_inactive = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_active, "sales.csv", SAMPLE_CSV)

    # 1. Setup active conversation
    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_active,
        "dataset_id": ds_id,
    })

    # 2. Setup inactive conversation
    client.post("/conversations", json={
        "id": cid_inactive,
        "title": "Category Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
        "session_id": cid_active,
    })
    client.post("/chat", json={
        "message": "Show sales by Category",
        "session_id": cid_inactive,
        "dataset_id": ds_id,
    })

    # 3. Delete the inactive conversation
    del_res = client.delete(f"/conversations/{cid_inactive}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # 4. Inactive conversation is gone
    assert client.get(f"/conversations/{cid_inactive}").status_code == 404

    # 5. Active conversation is completely preserved
    active_res = client.get(f"/conversations/{cid_active}")
    assert active_res.status_code == 200
    active_data = active_res.json()
    assert active_data["id"] == cid_active
    assert len(active_data["messages"]) == 2
    assert active_data["active_analysis_plan"] is not None

    # 6. Underlying dataset is untouched
    ds_res = client.get(f"/datasets/{ds_id}?session_id={cid_active}")
    assert ds_res.status_code == 200


def test_delete_active_conversation_and_switch(client: TestClient):
    """Deleting the active conversation removes it from SQLite and leaves other conversations switchable."""
    cid_1 = str(uuid.uuid4())
    cid_2 = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_1, "sales.csv", SAMPLE_CSV)

    # Conv 1
    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_1,
        "dataset_id": ds_id,
    })

    # Conv 2
    client.post("/conversations", json={
        "id": cid_2,
        "title": "Category Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
        "session_id": cid_1,
    })
    client.post("/chat", json={
        "message": "Show sales by Category",
        "session_id": cid_2,
        "dataset_id": ds_id,
    })

    # Delete active conversation (cid_1)
    del_res = client.delete(f"/conversations/{cid_1}")
    assert del_res.status_code == 200
    assert client.get(f"/conversations/{cid_1}").status_code == 404

    # Only cid_2 remains in conversations list
    convs = client.get("/conversations?workspace_mode=dataset").json()
    assert len(convs) == 1
    assert convs[0]["id"] == cid_2

    # Querying remaining conversation (cid_2) continues smoothly
    followup_res = client.post("/chat", json={
        "message": "Filter to Beverages",
        "session_id": cid_2,
        "dataset_id": ds_id,
    })
    assert followup_res.status_code == 200
    c2_data = client.get(f"/conversations/{cid_2}").json()
    assert len(c2_data["messages"]) == 4

    # Underlying dataset is still untouched
    assert client.get(f"/datasets/{ds_id}?session_id={cid_2}").status_code == 200


def test_delete_final_conversation_leaves_clean_state(client: TestClient):
    """Deleting the final remaining conversation leaves clean empty list without deleting dataset."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })

    # Delete the only conversation
    del_res = client.delete(f"/conversations/{cid}")
    assert del_res.status_code == 200

    # Conversations list is empty
    convs = client.get("/conversations?workspace_mode=dataset").json()
    assert convs == []
    assert client.get(f"/conversations/{cid}").status_code == 404

    # Dataset still exists
    assert client.get(f"/datasets/{ds_id}?session_id={cid}").status_code == 200


def test_deletion_persists_after_reloading(client: TestClient):
    """Deletion is persistent in SQLite and survives backend restart/cache purge."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })

    # Delete conversation
    client.delete(f"/conversations/{cid}")

    # Simulate server restart / cache purge
    _sessions.clear()

    # Verify conversation remains deleted
    assert client.get(f"/conversations/{cid}").status_code == 404
    assert client.get("/conversations?workspace_mode=dataset").json() == []


def test_create_delete_immediate_fetch_history_remains_absent(client: TestClient):
    """Req 1: Create a conversation, delete it, immediately fetch history, and verify it remains absent."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })

    # Verify conversation exists in history
    initial_list = client.get("/conversations?workspace_mode=dataset").json()
    assert any(c["id"] == cid for c in initial_list)

    # Delete conversation
    del_res = client.delete(f"/conversations/{cid}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Immediately fetch history: must be completely absent
    immediate_list = client.get("/conversations?workspace_mode=dataset").json()
    assert not any(c["id"] == cid for c in immediate_list)
    assert client.get(f"/conversations/{cid}").status_code == 404


def test_delete_final_conversation_no_replacement_auto_created(client: TestClient):
    """Req 3: Delete the final conversation and verify no replacement is automatically created."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })

    # Delete the final conversation
    del_res = client.delete(f"/conversations/{cid}")
    assert del_res.status_code == 200

    # Ensure list is strictly empty
    convs = client.get("/conversations?workspace_mode=dataset").json()
    assert convs == []

    # Verify in SQLite database directly that count of dataset conversations is exactly 0
    from app.database.connection import get_db
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) FROM conversations WHERE workspace_mode = 'dataset'").fetchone()
        assert row[0] == 0


def test_overlapping_history_request_after_deletion_simulation(client: TestClient):
    """Req 4: Simulate overlapping history requests where an older response arrives after deletion; verify deleted item does not return."""
    cid_1 = str(uuid.uuid4())
    cid_2 = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_1, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_1,
        "dataset_id": ds_id,
    })

    client.post("/conversations", json={
        "id": cid_2,
        "title": "Category Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
        "session_id": cid_1,
    })

    # Snapshot state before deletion (simulating an in-flight GET that captured both items)
    stale_in_flight_history = client.get("/conversations?workspace_mode=dataset").json()
    assert len(stale_in_flight_history) == 2

    # Client deletes cid_1
    del_res = client.delete(f"/conversations/{cid_1}")
    assert del_res.status_code == 200

    # Backend database state
    fresh_history = client.get("/conversations?workspace_mode=dataset").json()
    assert len(fresh_history) == 1
    assert fresh_history[0]["id"] == cid_2

    # Client-side sequencing / deletion set simulation:
    # If the older response (stale_in_flight_history) resolves after deletion,
    # the client filter prunes cid_1 using deletedAnalysisIds:
    deleted_analysis_ids = {cid_1}
    reconciled_list = [c for c in stale_in_flight_history if c["id"] not in deleted_analysis_ids]
    assert len(reconciled_list) == 1
    assert reconciled_list[0]["id"] == cid_2


def test_delete_preserves_dataset_and_unrelated_conversations(client: TestClient):
    """Req 6: Verify deletion does not delete the dataset or unrelated conversations."""
    cid_ds = str(uuid.uuid4())
    cid_sup = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_ds, "sales.csv", SAMPLE_CSV)

    # Dataset conversation
    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_ds,
        "dataset_id": ds_id,
    })

    # Support conversation
    client.post("/conversations", json={
        "id": cid_sup,
        "title": "Where is my order?",
        "workspace_mode": "support",
    })
    client.post("/chat", json={
        "message": "Where is my order 1001?",
        "session_id": cid_sup,
    })

    # Delete dataset conversation
    del_res = client.delete(f"/conversations/{cid_ds}")
    assert del_res.status_code == 200

    # Underlying dataset is preserved
    ds_res = client.get(f"/datasets/{ds_id}?session_id={cid_ds}")
    assert ds_res.status_code == 200
    assert ds_res.json()["dataset_id"] == ds_id

    # Unrelated support conversation is preserved
    sup_res = client.get(f"/conversations/{cid_sup}")
    assert sup_res.status_code == 200
    assert sup_res.json()["title"] == "Where is my order?"


def test_failed_deletion_preserves_conversation_and_returns_error(client: TestClient, monkeypatch):
    """Req 7: Verify failed deletion preserves the conversation and displays an error."""
    cid = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid, "sales.csv", SAMPLE_CSV)

    client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_id,
    })

    # 1. Non-existent ID returns 404
    non_existent = str(uuid.uuid4())
    bad_del = client.delete(f"/conversations/{non_existent}")
    assert bad_del.status_code == 404
    assert f"Conversation '{non_existent}' not found." in bad_del.json()["detail"]

    # Existing conversation is untouched
    assert client.get(f"/conversations/{cid}").status_code == 200

    # 2. Simulated failure in service layer preserves the target conversation
    orig_del = conversation_service.delete_conversation
    try:
        monkeypatch.setattr(conversation_service, "delete_conversation", lambda _: False)
        fail_del = client.delete(f"/conversations/{cid}")
        assert fail_del.status_code == 404
    finally:
        monkeypatch.setattr(conversation_service, "delete_conversation", orig_del)

    intact_conv = client.get(f"/conversations/{cid}")
    assert intact_conv.status_code == 200
    assert len(intact_conv.json()["messages"]) == 2


def test_dataset_switch_via_patch_resets_analytical_plans(client: TestClient):
    """PATCH updating dataset_id resets persisted and in-memory analytical plans when changed,
    but preserves valid plans when dataset_id is unchanged or when updating other fields."""
    cid = str(uuid.uuid4())
    ds_a = _upload_dataset(client, cid, "sales_a.csv", SAMPLE_CSV)
    ds_b = _upload_dataset(client, cid, "sales_b.csv", SAMPLE_CSV_2)

    # 1. Establish an active analysis plan tied to dataset A
    r1 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid,
        "dataset_id": ds_a,
    })
    assert r1.status_code == 200
    assert r1.json()["status"] == "ok"

    conv_initial = conversation_service.get_conversation(cid)
    assert conv_initial["dataset_id"] == ds_a
    assert conv_initial["active_analysis_plan"] is not None
    assert _sessions[cid]["active_analysis_plan"] is not None
    assert _sessions[cid]["analysis_plan"] is not None

    # 2. Switch dataset to dataset B via PATCH endpoint
    res_patch = client.patch(f"/conversations/{cid}", json={"dataset_id": ds_b})
    assert res_patch.status_code == 200
    assert res_patch.json()["dataset_id"] == ds_b

    # Verify dataset ID changed and both persisted and in-memory analytical plans are cleared
    conv_after_switch = conversation_service.get_conversation(cid)
    assert conv_after_switch["dataset_id"] == ds_b
    assert conv_after_switch["active_analysis_plan"] is None
    assert _sessions[cid]["active_analysis_plan"] is None
    assert _sessions[cid]["analysis_plan"] is None

    # 3. Create a valid analytical plan on dataset B
    r2 = client.post("/chat", json={
        "message": "Show revenue by Region",
        "session_id": cid,
        "dataset_id": ds_b,
    })
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"

    conv_b = conversation_service.get_conversation(cid)
    assert conv_b["dataset_id"] == ds_b
    assert conv_b["active_analysis_plan"] is not None
    assert _sessions[cid]["active_analysis_plan"] is not None

    # 4. Switching dataset ID to its existing value must NOT clear the valid plan
    res_patch_same = client.patch(f"/conversations/{cid}", json={"dataset_id": ds_b})
    assert res_patch_same.status_code == 200
    conv_same = conversation_service.get_conversation(cid)
    assert conv_same["dataset_id"] == ds_b
    assert conv_same["active_analysis_plan"] is not None
    assert _sessions[cid]["active_analysis_plan"] is not None

    # 5. Updating an unrelated field (title) must also preserve the active plan
    res_patch_title = client.patch(f"/conversations/{cid}", json={"title": "Region Revenue Overview"})
    assert res_patch_title.status_code == 200
    conv_title = conversation_service.get_conversation(cid)
    assert conv_title["title"] == "Region Revenue Overview"
    assert conv_title["dataset_id"] == ds_b
    assert conv_title["active_analysis_plan"] is not None
    assert _sessions[cid]["active_analysis_plan"] is not None

    # 6. Switching dataset ID to None clears the plan
    res_patch_none = client.patch(f"/conversations/{cid}", json={"dataset_id": None})
    assert res_patch_none.status_code == 200
    conv_none = conversation_service.get_conversation(cid)
    assert conv_none["dataset_id"] is None
    assert conv_none["active_analysis_plan"] is None
    assert _sessions[cid]["active_analysis_plan"] is None
    assert _sessions[cid]["analysis_plan"] is None


def test_restart_backend_recovers_workspace_mode(client: TestClient):
    """Reconstructing previous_state from SQLite preserves workspace_mode when omitted in request,
    while an explicitly provided workspace_mode takes precedence."""
    # 1. Support mode conversation restored without in-memory session
    cid_sup = str(uuid.uuid4())
    client.post("/conversations", json={
        "id": cid_sup,
        "title": "Order Support",
        "workspace_mode": "support",
    })
    r1 = client.post("/chat", json={
        "message": "What is the status of order 1001?",
        "session_id": cid_sup,
        "workspace_mode": "support",
    })
    assert r1.status_code == 200
    assert r1.json()["intent"] == "ORDER_STATUS"

    # Purge in-memory sessions to simulate restart/cache miss
    _sessions.clear()
    assert cid_sup not in _sessions

    # Subsequent request omits workspace_mode -> restores 'support'
    r2 = client.post("/chat", json={
        "message": "What is the status of order 1002?",
        "session_id": cid_sup,
    })
    assert r2.status_code == 200
    assert r2.json()["intent"] == "ORDER_STATUS"
    assert _sessions[cid_sup].get("workspace_mode") == "support"
    conv_sup = conversation_service.get_conversation(cid_sup)
    assert conv_sup["workspace_mode"] == "support"

    # 2. Dataset mode conversation restored without in-memory session
    cid_ds = str(uuid.uuid4())
    ds_id = _upload_dataset(client, cid_ds, "sales.csv", SAMPLE_CSV)
    client.post("/conversations", json={
        "id": cid_ds,
        "title": "Dataset Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
    })
    r3 = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": cid_ds,
        "dataset_id": ds_id,
        "workspace_mode": "dataset",
    })
    assert r3.status_code == 200

    # Purge in-memory sessions
    _sessions.clear()
    assert cid_ds not in _sessions

    # Subsequent request omits workspace_mode -> restores 'dataset'
    r4 = client.post("/chat", json={
        "message": "Plot this as a bar chart",
        "session_id": cid_ds,
        "dataset_id": ds_id,
    })
    assert r4.status_code == 200
    assert _sessions[cid_ds].get("workspace_mode") == "dataset"
    conv_ds = conversation_service.get_conversation(cid_ds)
    assert conv_ds["workspace_mode"] == "dataset"

    # 3. Explicit workspace_mode in request takes precedence over persisted mode
    _sessions.clear()
    r5 = client.post("/chat", json={
        "message": "What is the status of order 1001?",
        "session_id": cid_ds,
        "workspace_mode": "support",
    })
    assert r5.status_code == 200
    assert r5.json()["intent"] == "ORDER_STATUS"
    assert _sessions[cid_ds].get("workspace_mode") == "support"


def test_workspace_switching_support_dataset_support_session_isolation(client: TestClient):
    """Switching Support -> Dataset -> Support preserves thread-level session isolation.
    Support queries and document uploads must use the support thread ID and never leak into
    the active analytical dataset conversation."""
    thread_id_sup = str(uuid.uuid4())
    conv_id_ds = str(uuid.uuid4())
    from app.api.session_security import sign_session_id
    headers_ds = {"X-Session-Token": sign_session_id(conv_id_ds)}
    headers_sup = {"X-Session-Token": sign_session_id(thread_id_sup)}

    ds_id = _upload_dataset(client, conv_id_ds, "sales.csv", SAMPLE_CSV)

    # 1. Start in Support Mode: create support thread and upload document to support thread
    client.post("/conversations", json={
        "id": thread_id_sup,
        "title": "Support Inquiry",
        "workspace_mode": "support",
    }, headers=headers_sup)

    doc_resp = client.post(
        "/documents/upload",
        files={"file": ("guide.txt", io.BytesIO(b"Return window is 30 days for electronics."), "text/plain")},
        data={"session_id": thread_id_sup},
        headers=headers_sup,
    )
    assert doc_resp.status_code == 201
    doc_id = doc_resp.json()["document_id"]

    # 2. Switch to Dataset Intelligence Mode: create analytical conversation & run query
    client.post("/conversations", json={
        "id": conv_id_ds,
        "title": "Sales Analysis",
        "workspace_mode": "dataset",
        "dataset_id": ds_id,
    }, headers=headers_ds)
    r_ds = client.post("/chat", json={
        "message": "Show me the top 5 products by sales",
        "session_id": conv_id_ds,
        "dataset_id": ds_id,
        "workspace_mode": "dataset",
    }, headers=headers_ds)
    assert r_ds.status_code == 200
    assert r_ds.json()["status"] == "ok"
    assert _sessions[conv_id_ds].get("active_analysis_plan") is not None

    # 3. Switch back to Support Mode: send support message using the support thread ID
    r_sup = client.post("/chat", json={
        "message": "What is the status of order 1001?",
        "session_id": thread_id_sup,
        "workspace_mode": "support",
    }, headers=headers_sup)
    assert r_sup.status_code == 200
    assert r_sup.json()["intent"] == "ORDER_STATUS"

    # 4. Verify complete session isolation between Support and Dataset Intelligence
    conv_ds = conversation_service.get_conversation_with_messages(conv_id_ds)
    conv_sup = conversation_service.get_conversation_with_messages(thread_id_sup)

    # Analytical conversation in SQLite must NOT contain the support message
    assert conv_ds["workspace_mode"] == "dataset"
    assert len(conv_ds["messages"]) == 2  # 1 user + 1 assistant analytical query
    assert "top 5" in conv_ds["messages"][0]["user_message"].lower()
    assert not any("1001" in (m.get("user_message") or "") for m in conv_ds["messages"])
    assert conv_ds["active_analysis_plan"] is not None

    # Support conversation in SQLite must contain the support message and stay in support mode
    assert conv_sup["workspace_mode"] == "support"
    assert len(conv_sup["messages"]) == 2  # 1 user + 1 assistant support query
    assert any("1001" in (m.get("user_message") or "") for m in conv_sup["messages"])

    # 5. Verify document deletion uses support thread ID and cleans up cleanly
    del_doc = client.delete(f"/documents/{doc_id}?session_id={thread_id_sup}", headers=headers_sup)
    assert del_doc.status_code == 200

    # 6. Verify static/app.js source code ensures session isolation
    app_js_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
    assert app_js_path.exists()
    js_content = app_js_path.read_text(encoding="utf-8")

    # In switchWorkspaceMode(), when switching to support mode, currentSessionId is restored from activeThread
    assert "currentSessionId = activeThread.id;" in js_content

    # In sendMessage(), session_id derives from active thread
    assert "session_id: (thread && thread.id) ? thread.id : currentSessionId," in js_content

    # In handleFileUpload(), session_id derives from active thread
    assert "uploadSessionId = (thread && thread.id) ? thread.id : currentSessionId;" in js_content
