"""
Dataset ingestion, lifecycle management, storage, and session isolation.
Stores uploaded datasets under data/datasets/{session_id}/{dataset_id}/original.{ext}
and records metadata in SQLite and metadata.json.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Optional
import uuid

from app.config import settings
from app.database.connection import get_db
from app.analytics.schemas import (
    DatasetProfile,
    DatasetMetadata,
    DatasetUploadResponse,
    DatasetDeleteResponse,
)
from app.analytics.profiler import profile_dataset_bytes, DatasetProfileError

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".json"}


class DatasetValidationError(Exception):
    """Raised when an uploaded dataset fails validation."""
    pass


class DatasetNotFoundError(Exception):
    """Raised when a dataset does not exist."""
    pass


class DatasetAccessDeniedError(Exception):
    """Raised when cross-session dataset access or modification is attempted."""
    pass


def get_max_file_size_bytes() -> int:
    return settings.max_dataset_file_size_mb * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Strips directory traversal sequences and unsafe characters."""
    raw_name = Path(filename).name
    cleaned = re.sub(r"[^\w.\-_]", "_", raw_name).strip()
    return cleaned or "dataset.csv"


def validate_file(filename: str, content_bytes: bytes) -> tuple[str, str]:
    """
    Validates file extension, non-empty content, and max file size.
    Returns (sanitized_filename, format_extension_without_dot).
    """
    if not content_bytes or len(content_bytes) == 0:
        raise DatasetValidationError("Dataset file is empty.")

    max_size = get_max_file_size_bytes()
    if len(content_bytes) > max_size:
        raise DatasetValidationError(
            f"File exceeds maximum allowed size of {settings.max_dataset_file_size_mb}MB."
        )

    sanitized = sanitize_filename(filename)
    ext = Path(sanitized).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file type '{ext}'. Allowed types are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    file_format = ext.lstrip(".")
    return sanitized, file_format


def _get_dataset_dir(session_id: str, dataset_id: str) -> Path:
    """Computes and validates dataset directory path within datasets_dir."""
    base_dir = Path(settings.datasets_dir).resolve()
    target_dir = (base_dir / session_id / dataset_id).resolve()
    
    # Path traversal check
    if not str(target_dir).startswith(str(base_dir)):
        raise DatasetValidationError("Invalid directory path (path traversal detected).")
        
    return target_dir


def ingest_dataset(session_id: str, filename: str, content_bytes: bytes) -> DatasetUploadResponse:
    """
    Ingests and profiles a dataset:
    1. Validates file size and format.
    2. Runs deterministic fast profiling.
    3. Saves original file to disk without redundant copies.
    4. Records metadata in SQLite and JSON.
    """
    if not session_id or not session_id.strip():
        raise DatasetValidationError("A valid session_id is required.")
    session_id = session_id.strip()

    sanitized_filename, file_format = validate_file(filename, content_bytes)

    # Deterministic profiling
    try:
        profile: DatasetProfile = profile_dataset_bytes(content_bytes, file_format)
    except DatasetProfileError as e:
        raise DatasetValidationError(str(e))

    dataset_id = str(uuid.uuid4())
    dataset_dir = _get_dataset_dir(session_id, dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save original file
    original_path = dataset_dir / f"original.{file_format}"
    with open(original_path, "wb") as f:
        f.write(content_bytes)

    created_at = datetime.now(timezone.utc).isoformat()

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        session_id=session_id,
        filename=sanitized_filename,
        format=file_format,
        row_count=profile.row_count,
        column_count=profile.column_count,
        created_at=created_at,
        profile=profile,
    )

    # Save metadata.json
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    # Save record to SQLite
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO uploaded_datasets (
                dataset_id, session_id, filename, format, row_count, column_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                session_id,
                sanitized_filename,
                file_format,
                profile.row_count,
                profile.column_count,
                created_at,
            ),
        )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        session_id=session_id,
        filename=sanitized_filename,
        format=file_format,
        row_count=profile.row_count,
        column_count=profile.column_count,
        profile=profile,
    )


def get_dataset(dataset_id: str, session_id: Optional[str] = None) -> DatasetMetadata:
    """
    Retrieves dataset metadata. If session_id is supplied, verifies session ownership.
    """
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT dataset_id, session_id, filename, format, row_count, column_count, created_at FROM uploaded_datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise DatasetNotFoundError(f"Dataset with ID '{dataset_id}' not found.")

    row_session_id = (row["session_id"] or "").strip()
    if session_id and row_session_id != session_id.strip():
        from app.repositories import conversation_repository as conv_repo
        is_authorized = False
        with get_db() as c_conn:
            conv = conv_repo.get_conversation(c_conn, session_id.strip())
            if conv and conv.get("dataset_id") == dataset_id:
                conv_owner = (conv.get("session_id") or conv.get("id") or "").strip()
                if conv_owner == row_session_id:
                    is_authorized = True
        if not is_authorized:
            raise DatasetAccessDeniedError("Access denied: dataset belongs to a different session.")

    dataset_dir = _get_dataset_dir(row_session_id, dataset_id)
    metadata_path = dataset_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return DatasetMetadata.model_validate(data)

    raise DatasetNotFoundError(f"Dataset files for ID '{dataset_id}' are missing.")


def list_datasets(session_id: Optional[str] = None) -> list[DatasetMetadata]:
    """Lists all uploaded datasets for a session. Returns empty list if session_id is None."""
    if not session_id or not session_id.strip():
        return []
    clean_session_id = session_id.strip()
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT dataset_id FROM uploaded_datasets WHERE session_id = ? ORDER BY created_at DESC",
            (clean_session_id,),
        )
        rows = cursor.fetchall()

    results = []
    seen = set()
    for row in rows:
        did = row["dataset_id"]
        if did in seen:
            continue
        seen.add(did)
        try:
            results.append(get_dataset(did, session_id=clean_session_id))
        except (DatasetNotFoundError, DatasetAccessDeniedError):
            continue
    return results


def delete_dataset(dataset_id: str, session_id: str) -> DatasetDeleteResponse:
    """
    Deletes dataset from SQLite and removes all filesystem artifacts.
    Enforces non-empty session ownership.
    Invalidates any active in-memory session bindings for this dataset.
    """
    if not dataset_id or not dataset_id.strip():
        raise DatasetValidationError("A valid, non-empty dataset_id is required.")

    if not session_id or not session_id.strip():
        raise DatasetValidationError("A valid, non-empty session_id is required.")

    clean_dataset_id = dataset_id.strip()
    clean_session_id = session_id.strip()

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?",
            (clean_dataset_id,),
        )
        row = cursor.fetchone()

        if not row:
            raise DatasetNotFoundError(f"Dataset with ID '{clean_dataset_id}' not found.")

        row_session_id = row["session_id"]
        if (row_session_id or "").strip() != clean_session_id:
            raise DatasetAccessDeniedError("Access denied: dataset belongs to a different session.")

        # Delete database record
        conn.execute("DELETE FROM uploaded_datasets WHERE dataset_id = ?", (clean_dataset_id,))

    # Invalidate in-memory session caches that hold references to this deleted dataset
    try:
        from app.api.rest import _sessions
        for sess in list(_sessions.values()):
            if isinstance(sess, dict) and sess.get("dataset_id") == clean_dataset_id:
                sess["dataset_id"] = None
                sess["active_analysis_plan"] = None
                sess["analysis_plan"] = None
    except Exception:
        pass

    # Remove filesystem artifacts
    dataset_dir = _get_dataset_dir(row_session_id, clean_dataset_id)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)

    # Clean up parent session directory if empty
    session_dir = dataset_dir.parent
    if session_dir.exists() and not any(session_dir.iterdir()):
        shutil.rmtree(session_dir, ignore_errors=True)

    return DatasetDeleteResponse(dataset_id=clean_dataset_id, deleted=True)


def get_dataset_file_path(dataset_id: str, session_id: str) -> Path:
    """
    Returns the resolved path to the original dataset file.
    Verifies existence and session ownership.
    """
    metadata = get_dataset(dataset_id, session_id=session_id)
    dataset_dir = _get_dataset_dir(metadata.session_id, dataset_id)
    original_path = dataset_dir / f"original.{metadata.format}"
    if not original_path.exists():
        raise DatasetNotFoundError(f"Original dataset file not found for dataset '{dataset_id}'.")
    return original_path
