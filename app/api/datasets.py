"""
API endpoints for dataset upload, profiling, inspection, and deletion.
"""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status, Request, Response

from app.analytics.schemas import (
    DatasetUploadResponse,
    DatasetMetadata,
    DatasetDeleteResponse,
)
from app.config import settings
from app.analytics import dataset_service
from app.api.session_security import (
    get_verified_session,
    require_verified_session,
    set_session_cookie,
    verify_csrf_protection,
    sign_session_id,
)

logger = logging.getLogger("app.analytics.api")

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    """
    Upload and profile a tabular dataset (CSV, XLSX, JSON).
    Deterministic fast profiling provides column types, missing value ratios, and duplicate counts.
    """
    verify_csrf_protection(request)

    verified_session = get_verified_session(request)

    if session_id and session_id.strip():
        candidate = session_id.strip()
        if verified_session and verified_session == candidate:
            resolved_session_id = verified_session
            token = set_session_cookie(response, resolved_session_id, request=request)
        else:
            from app.database.connection import get_db
            with get_db() as conn:
                cursor1 = conn.execute("SELECT 1 FROM uploaded_datasets WHERE session_id = ? LIMIT 1", (candidate,))
                cursor2 = conn.execute("SELECT 1 FROM conversations WHERE session_id = ? LIMIT 1", (candidate,))
                if cursor1.fetchone() or cursor2.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Cannot select an existing session identity without a valid session token.",
                    )
            resolved_session_id = candidate
            token = set_session_cookie(response, resolved_session_id, request=request)
    elif verified_session:
        resolved_session_id = verified_session
        token = set_session_cookie(response, resolved_session_id, request=request)
    else:
        resolved_session_id = str(uuid.uuid4())
        token = set_session_cookie(response, resolved_session_id, request=request)

    max_bytes = settings.max_dataset_file_size_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_dataset_file_size_mb}MB.",
        )

    try:
        chunks = []
        total_read = 0
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {settings.max_dataset_file_size_mb}MB.",
                )
            chunks.append(chunk)
        content_bytes = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read uploaded dataset file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    try:
        result = dataset_service.ingest_dataset(
            session_id=resolved_session_id,
            filename=file.filename or "dataset.csv",
            content_bytes=content_bytes,
        )
        result.session_token = token
        return result
    except dataset_service.DatasetValidationError as e:
        logger.warning(f"Dataset validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error processing dataset upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the dataset: {e}",
        )


@router.get(
    "/{dataset_id}",
    response_model=DatasetMetadata,
)
def get_dataset(
    dataset_id: str,
    request: Request,
    session_id: Optional[str] = Query(default=None),
):
    """Retrieves metadata and fast profile for an uploaded dataset."""
    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT dataset_id, session_id FROM uploaded_datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    row_session_id = (row["session_id"] or "").strip()
    verified_session = get_verified_session(request)

    if session_id and session_id.strip():
        req_sess = session_id.strip()
        if verified_session and verified_session != req_sess:
            is_owned_conv = False
            from app.repositories import conversation_repository as conv_repo
            with get_db() as c_conn:
                conv = conv_repo.get_conversation(c_conn, req_sess)
                if conv and (conv.get("session_id") or conv.get("id") or "").strip() == verified_session:
                    is_owned_conv = True
            if not is_owned_conv:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: session token does not match requested session_id.",
                )
        if not verified_session:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: signed session token is required to access owned datasets.",
            )

    if row_session_id:
        if not verified_session or verified_session != row_session_id:
            is_authorized = False
            if verified_session:
                from app.repositories import conversation_repository as conv_repo
                with get_db() as c_conn:
                    conv = conv_repo.get_conversation(c_conn, verified_session)
                    if conv and conv.get("dataset_id") == dataset_id:
                        conv_owner = (conv.get("session_id") or conv.get("id") or "").strip()
                        if conv_owner == row_session_id:
                            is_authorized = True
            if not is_authorized:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: dataset belongs to a different session.",
                )

    try:
        return dataset_service.get_dataset(dataset_id=dataset_id, session_id=verified_session or row_session_id)
    except dataset_service.DatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except dataset_service.DatasetAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error fetching dataset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching the dataset: {e}",
        )


@router.get(
    "",
    response_model=list[DatasetMetadata],
)
def list_datasets(
    request: Request,
    session_id: Optional[str] = Query(default=None),
):
    """Lists all datasets associated with an active session."""
    verified_session = get_verified_session(request)
    if session_id and session_id.strip():
        req_sess = session_id.strip()
        if not verified_session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: a valid signed session token or cookie is required.",
            )
        if verified_session != req_sess:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: session token does not match requested session_id.",
            )
        return dataset_service.list_datasets(session_id=req_sess)

    if not verified_session:
        return []
    return dataset_service.list_datasets(session_id=verified_session)


@router.delete(
    "/{dataset_id}",
    response_model=DatasetDeleteResponse,
)
def delete_dataset(
    dataset_id: str,
    request: Request,
    session_id: str = Query(..., min_length=1, description="Owning session ID for dataset deletion"),
):
    """Deletes an uploaded dataset and all stored artifacts."""
    verify_csrf_protection(request)
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid, non-empty session_id is required for dataset deletion.",
        )
    clean_sess = session_id.strip()

    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.execute("SELECT session_id FROM uploaded_datasets WHERE dataset_id = ?", (dataset_id,))
        row = cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    row_session_id = (row["session_id"] or "").strip()
    verified_session = get_verified_session(request)

    if not verified_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: a valid signed session token or cookie is required.",
        )

    if verified_session != clean_sess or (row_session_id and verified_session != row_session_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: dataset belongs to a different session.",
        )

    try:
        return dataset_service.delete_dataset(dataset_id=dataset_id, session_id=verified_session)
    except dataset_service.DatasetValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except dataset_service.DatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except dataset_service.DatasetAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error deleting dataset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the dataset: {e}",
        )
