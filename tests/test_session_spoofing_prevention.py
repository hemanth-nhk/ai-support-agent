"""
Isolated regression tests for server-verified signed session identity and anti-spoofing defense.

Verifies all 8 requirement criteria:
1. Missing tokens rejected on protected endpoints.
2. Forged and tampered tokens rejected.
3. Attacker cannot select existing session identity on session creation paths.
4. Spoofed POST /conversations with victim dataset rejected (403).
5. Spoofed PATCH /conversations ownership mutation & dataset stealing rejected (403) and immutable ownership.
6. Downstream /chat denied before execution for foreign sessions or unauthenticated callers.
7. Legitimate owner access and legitimate dataset switching succeed.
8. Legacy records with NULL ownership are protected (no silent access grant).
9. CSRF defense for cookie-authenticated mutations.
"""
import io
import json
import uuid
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from pathlib import Path
from app.database.connection import get_db
from app.api.session_security import sign_session_id, verify_session_token, COOKIE_NAME, HEADER_NAME


@pytest.fixture(autouse=True)
def isolated_security_env(monkeypatch):
    """Isolates dataset storage and ensures a secure SESSION_SECRET_KEY is configured."""
    temp_dir = tempfile.mkdtemp(prefix="test_security_store_")
    monkeypatch.setattr(settings, "datasets_dir", temp_dir)
    monkeypatch.setattr(settings, "database_path", str(Path(temp_dir) / "test_sec.db"))
    monkeypatch.setattr(settings, "session_secret_key", "d5a5c7e7b925bcbb2e8325c76339192e2a62a8d9ca85d4abcff1a2bf376c7d17")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _upload_test_dataset(client: TestClient, session_id: str, filename: str = "test.csv") -> str:
    """Helper to upload a dataset and return dataset_id."""
    csv_bytes = b"col_a,col_b\n10,20\n30,40"
    res = client.post(
        "/datasets/upload",
        data={"session_id": session_id},
        files={"file": (filename, csv_bytes, "text/csv")},
    )
    assert res.status_code == 201
    return res.json()["dataset_id"]


def test_missing_token_on_protected_endpoints_rejected():
    """Requirement 2 & 7: Protected operations without token or cookie fail with 401."""
    client = TestClient(app)
    unauth_client = TestClient(app)  # has no cookies

    # 1. DELETE /datasets/{id} without token
    res = unauth_client.delete("/datasets/dummy-uuid?session_id=any-session")
    # Missing dataset returns 404, or if dataset exists without token returns 401
    # Create dataset first
    ds_id = _upload_test_dataset(client, "owner-session-1")

    # Now unauthenticated client tries to delete it
    res = unauth_client.delete(f"/datasets/{ds_id}?session_id=owner-session-1")
    assert res.status_code == 401

    # 2. PATCH /conversations/{id} without token
    # Create conversation first
    c_res = client.post("/conversations", json={"title": "Test Conv"})
    assert c_res.status_code == 201
    conv_id = c_res.json()["id"]

    res = unauth_client.patch(f"/conversations/{conv_id}", json={"title": "Hacked Title"})
    assert res.status_code == 401

    # 3. DELETE /conversations/{id} without token
    res = unauth_client.delete(f"/conversations/{conv_id}")
    assert res.status_code == 401


def test_forged_or_tampered_token_rejected():
    """Requirement 1 & 7: Forged, tampered, or invalid tokens fail verification."""
    client = TestClient(app)
    ds_id = _upload_test_dataset(client, "owner-session-2")

    attacker = TestClient(app)
    # Attacker crafts forged token with invalid signature
    tampered_token = f"owner-session-2.{'0' * 64}"
    attacker.headers[HEADER_NAME] = tampered_token

    res = attacker.get(f"/datasets/{ds_id}")
    assert res.status_code in (401, 403)

    res = attacker.delete(f"/datasets/{ds_id}?session_id=owner-session-2")
    assert res.status_code in (401, 403)


def test_cannot_select_existing_identity_on_session_creation_without_token():
    """Requirement 2: Missing token may not select an existing identity."""
    client = TestClient(app)
    victim_session = f"victim-sess-{uuid.uuid4().hex[:6]}"
    _upload_test_dataset(client, victim_session)

    attacker = TestClient(app)
    # Attacker tries to upload dataset asserting victim's existing session_id without holding its token
    csv_bytes = b"x,y\n1,2"
    res = attacker.post(
        "/datasets/upload",
        data={"session_id": victim_session},
        files={"file": ("injected.csv", csv_bytes, "text/csv")},
    )
    assert res.status_code == 403
    assert "Cannot select an existing session identity" in res.json()["detail"]

    # Attacker tries to create conversation asserting victim's session_id
    res = attacker.post(
        "/conversations",
        json={"title": "Spoofed", "session_id": victim_session},
    )
    assert res.status_code == 403
    assert "Cannot select an existing session identity" in res.json()["detail"]


def test_spoofed_post_conversations_with_victim_dataset_rejected():
    """Requirement 3 & 7: POST /conversations with victim's dataset_id is rejected with 403."""
    victim_client = TestClient(app)
    victim_sess = f"victim-{uuid.uuid4().hex[:6]}"
    victim_ds = _upload_test_dataset(victim_client, victim_sess)

    # Attacker with distinct session tries to create conversation binding victim's dataset
    attacker_client = TestClient(app)
    res = attacker_client.post(
        "/conversations",
        json={
            "title": "Attacker Conversation",
            "dataset_id": victim_ds,
            "session_id": victim_sess,
        },
    )
    assert res.status_code == 403
    assert "Access denied: dataset belongs to a different session." in res.json()["detail"]


def test_spoofed_patch_conversations_ownership_and_dataset_rejected():
    """Requirement 3 & 7: PATCH /conversations ownership is immutable and dataset switching is authenticated."""
    victim_client = TestClient(app)
    victim_sess = f"victim-{uuid.uuid4().hex[:6]}"
    victim_ds = _upload_test_dataset(victim_client, victim_sess)

    # Victim creates a conversation
    c_res = victim_client.post("/conversations", json={"title": "Victim Analysis", "dataset_id": victim_ds})
    assert c_res.status_code == 201
    victim_conv_id = c_res.json()["id"]

    # Attacker creates their own session & conversation
    attacker_client = TestClient(app)
    _upload_test_dataset(attacker_client, "attacker-session")
    a_conv_res = attacker_client.post("/conversations", json={"title": "Attacker Conv"})
    assert a_conv_res.status_code == 201
    attacker_conv_id = a_conv_res.json()["id"]

    # 1. Attacker tries to PATCH victim's conversation directly
    patch_res1 = attacker_client.patch(
        f"/conversations/{victim_conv_id}",
        json={"title": "Hacked"},
    )
    assert patch_res1.status_code == 403
    assert "you do not own this conversation" in patch_res1.json()["detail"]

    # 2. Attacker tries to bind victim's dataset to attacker's own conversation
    patch_res2 = attacker_client.patch(
        f"/conversations/{attacker_conv_id}",
        json={"dataset_id": victim_ds},
    )
    assert patch_res2.status_code == 403
    assert "dataset belongs to a different session" in patch_res2.json()["detail"]

    # 3. Attacker tries to mutate ownership by passing a foreign session_id
    patch_res3 = attacker_client.patch(
        f"/conversations/{attacker_conv_id}",
        json={"session_id": victim_sess, "title": "Trying to steal ownership"},
    )
    assert patch_res3.status_code == 403
    assert "session token does not match" in patch_res3.json()["detail"]

    # 4. Legitimate owner updates title: ownership remains immutable
    legit_patch = attacker_client.patch(
        f"/conversations/{attacker_conv_id}",
        json={"title": "Updated Title"},
    )
    assert legit_patch.status_code == 200
    with get_db() as conn:
        cursor = conn.execute("SELECT session_id FROM conversations WHERE id = ?", (attacker_conv_id,))
        row = cursor.fetchone()
        assert row["session_id"] == "attacker-session"


def test_downstream_chat_denied_before_execution():
    """Requirement 4 & 7: /chat denies unauthorized dataset or conversation access before execution."""
    victim_client = TestClient(app)
    victim_sess = f"victim-{uuid.uuid4().hex[:6]}"
    victim_ds = _upload_test_dataset(victim_client, victim_sess)

    v_conv = victim_client.post("/conversations", json={"title": "Victim Conv", "dataset_id": victim_ds})
    assert v_conv.status_code == 201
    victim_conv_id = v_conv.json()["id"]

    attacker_client = TestClient(app)

    # 1. Attacker queries victim's conversation via /chat
    res1 = attacker_client.post(
        "/chat",
        json={"message": "Show total revenue", "session_id": victim_conv_id},
    )
    assert res1.status_code == 403
    assert "conversation belongs to a different session" in res1.json()["detail"]

    # 2. Attacker queries victim's dataset directly via /chat
    res2 = attacker_client.post(
        "/chat",
        json={"message": "Show total revenue", "dataset_id": victim_ds},
    )
    assert res2.status_code == 403
    assert "dataset belongs to a different session" in res2.json()["detail"]


def test_legitimate_owner_access_and_dataset_switching():
    """Requirement 5 & 7: Legitimate owner access and dataset switching succeed normally."""
    owner_client = TestClient(app)
    owner_sess = f"owner-{uuid.uuid4().hex[:6]}"
    ds1 = _upload_test_dataset(owner_client, owner_sess, "ds1.csv")
    ds2 = _upload_test_dataset(owner_client, owner_sess, "ds2.csv")

    # Create conversation with ds1
    conv_res = owner_client.post(
        "/conversations",
        json={"title": "Owner Analysis", "dataset_id": ds1},
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    # Legitimate owner switches to ds2
    patch_res = owner_client.patch(
        f"/conversations/{conv_id}",
        json={"dataset_id": ds2},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["dataset_id"] == ds2

    # Legitimate owner detaches dataset
    unbind_res = owner_client.patch(
        f"/conversations/{conv_id}",
        json={"dataset_id": None},
    )
    assert unbind_res.status_code == 200
    assert unbind_res.json()["dataset_id"] is None


def test_legacy_record_handling_no_silent_grant():
    """Requirement 6 & 7: Legacy records with NULL session_id are not silently granted."""
    legacy_conv_id = f"legacy-conv-{uuid.uuid4().hex[:6]}"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, workspace_mode, dataset_id, session_id, created_at, updated_at)
            VALUES (?, 'Legacy Conv', 'dataset', NULL, NULL, '2026-01-01', '2026-01-01')
            """,
            (legacy_conv_id,),
        )

    # Attacker with unrelated session attempts to access legacy conversation
    attacker = TestClient(app)
    _upload_test_dataset(attacker, "attacker-sess")

    get_res = attacker.get(f"/conversations/{legacy_conv_id}")
    assert get_res.status_code == 403
    assert "legacy conversation requires matching session ownership" in get_res.json()["detail"]

    chat_res = attacker.post(
        "/chat",
        json={"message": "hello", "session_id": legacy_conv_id},
    )
    assert chat_res.status_code == 403
    assert "legacy conversation requires matching session ownership" in chat_res.json()["detail"]


def test_csrf_defense_for_cookie_mutations():
    """Requirement 5: CSRF verification blocks cross-origin cookie mutations."""
    client = TestClient(app)
    _upload_test_dataset(client, "csrf-sess")

    # Make request with cookie but cross-site Sec-Fetch-Site header
    res = client.post(
        "/conversations",
        json={"title": "CSRF Attempt"},
        headers={"sec-fetch-site": "cross-site"},
    )
    assert res.status_code == 403
    assert "CSRF validation failed" in res.json()["detail"]


def test_unauthenticated_get_datasets_returns_empty_and_does_not_leak():
    """Batch 1: GET /datasets must not leak datasets across sessions or allow unverified session queries."""
    client = TestClient(app)
    _upload_test_dataset(client, "dataset-owner-sess")

    unauth = TestClient(app)
    # Unauthenticated GET /datasets returns empty list
    r_empty = unauth.get("/datasets")
    assert r_empty.status_code == 200
    assert r_empty.json() == []

    # Unauthenticated GET /datasets?session_id=victim returns 401
    r_unauth_victim = unauth.get("/datasets?session_id=dataset-owner-sess")
    assert r_unauth_victim.status_code == 401

    # Tampered token returns 401
    tampered = f"dataset-owner-sess.{'0' * 64}"
    r_tampered = unauth.get("/datasets?session_id=dataset-owner-sess", headers={HEADER_NAME: tampered})
    assert r_tampered.status_code == 401

    # Cross-session query returns 403
    other_token = sign_session_id("other-session")
    r_cross = unauth.get("/datasets?session_id=dataset-owner-sess", headers={HEADER_NAME: other_token})
    assert r_cross.status_code == 403

    # Authenticated owner sees their dataset
    r_owner = client.get("/datasets")
    assert r_owner.status_code == 200
    assert len(r_owner.json()) == 1
    assert r_owner.json()[0]["session_id"] == "dataset-owner-sess"


def test_conversation_listing_authorization_enforced():
    """Batch 1: GET /conversations requires verified session and blocks unauthenticated enumeration."""
    client = TestClient(app)
    c_res = client.post("/conversations", json={"title": "Owner Conv", "workspace_mode": "support"})
    assert c_res.status_code == 201

    unauth = TestClient(app)
    # Unauthenticated listing with workspace_mode returns empty list
    assert unauth.get("/conversations?workspace_mode=support").json() == []

    # Unauthenticated listing with target session_id rejected with 401
    assert unauth.get("/conversations?session_id=victim-sess").status_code == 401

    # Cross-session listing rejected with 403
    other_token = sign_session_id("other-sess")
    r_cross = unauth.get("/conversations?session_id=victim-sess", headers={HEADER_NAME: other_token})
    assert r_cross.status_code == 403

    # Authenticated owner sees their conversations
    r_owner = client.get("/conversations?workspace_mode=support")
    assert r_owner.status_code == 200
    assert len(r_owner.json()) >= 1


def test_order_endpoint_authentication_enforced():
    """Batch 1: GET /orders/{order_id} requires API key."""
    from app.database.seed import seed
    seed(force=True)

    client = TestClient(app)
    settings.support_api_key = "test-key-batch1"

    # Missing API key
    assert client.get("/orders/1001").status_code == 401

    # Invalid API key
    assert client.get("/orders/1001", headers={"X-API-Key": "wrong"}).status_code == 401

    # Valid API key
    res = client.get("/orders/1001", headers={"X-API-Key": "test-key-batch1"})
    assert res.status_code == 200
    assert res.json()["id"] == 1001


def test_chart_artifact_session_ownership_enforced():
    """Batch 1: GET /charts/{session_id}/{chart_id} requires verified session ownership."""
    client = TestClient(app)

    # Missing token
    assert client.get("/charts/sess_alpha/some_chart_id").status_code == 401

    # Tampered token
    tampered = f"sess_alpha.{'0' * 64}"
    assert client.get("/charts/sess_alpha/some_chart_id", headers={HEADER_NAME: tampered}).status_code == 401

    # Cross-session token
    beta_token = sign_session_id("sess_beta")
    assert client.get("/charts/sess_alpha/some_chart_id", headers={HEADER_NAME: beta_token}).status_code == 403


def test_document_upload_and_delete_session_ownership():
    """Batch 1: Document upload anti-spoofing and document delete session ownership."""
    client = TestClient(app)
    doc_bytes = b"Sample document text content."

    # 1. First upload creates session-doc-owner
    up1 = client.post(
        "/documents/upload",
        files={"file": ("test.txt", doc_bytes, "text/plain")},
        data={"session_id": "session-doc-owner"},
    )
    assert up1.status_code == 201
    doc_id = up1.json()["document_id"]
    assert up1.json()["session_token"] is not None

    # 2. Attacker without token attempts to upload document targeting existing session identity
    attacker = TestClient(app)
    up_spoofed = attacker.post(
        "/documents/upload",
        files={"file": ("evil.txt", doc_bytes, "text/plain")},
        data={"session_id": "session-doc-owner"},
    )
    assert up_spoofed.status_code == 403
    assert "Cannot select an existing session identity" in up_spoofed.json()["detail"]

    # 3. Attacker without token attempts to delete document
    del_unauth = attacker.delete(f"/documents/{doc_id}?session_id=session-doc-owner")
    assert del_unauth.status_code == 401

    # 4. Attacker with different session token attempts to delete document
    other_token = sign_session_id("attacker-sess")
    del_cross = attacker.delete(
        f"/documents/{doc_id}?session_id=session-doc-owner",
        headers={HEADER_NAME: other_token},
    )
    assert del_cross.status_code == 403

    # 5. Legitimate owner with session cookie deletes document
    del_owner = client.delete(f"/documents/{doc_id}?session_id=session-doc-owner")
    assert del_owner.status_code == 200
    assert del_owner.json()["status"] == "deleted"
