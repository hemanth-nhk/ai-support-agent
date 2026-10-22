"""
API endpoints for document upload and deletion.
"""
import logging
from typing import Optional
import uuid

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status, Request, Response

from app.api.schemas import DocumentUploadResponse, DocumentDeleteResponse
from app.services import document_service
from app.api.session_security import (
    get_verified_session,
    require_verified_session,
    set_session_cookie,
    verify_csrf_protection,
)

logger = logging.getLogger("app.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload and index a document (PDF, DOCX, TXT) for RAG document assistance.
    Isolates document vectors by session_id in the uploaded_documents collection.
    """
    verify_csrf_protection(request)

    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid session_id is required.",
        )

    candidate = session_id.strip()
    verified_session = get_verified_session(request)

    if verified_session and verified_session == candidate:
        resolved_session_id = verified_session
        token = set_session_cookie(response, resolved_session_id, request=request)
    else:
        # Check if candidate session identity already exists in datasets, conversations, or Chroma
        from app.database.connection import get_db
        with get_db() as conn:
            cursor1 = conn.execute("SELECT 1 FROM uploaded_datasets WHERE session_id = ? LIMIT 1", (candidate,))
            cursor2 = conn.execute("SELECT 1 FROM conversations WHERE session_id = ? LIMIT 1", (candidate,))
            has_existing = bool(cursor1.fetchone() or cursor2.fetchone())

        if not has_existing:
            try:
                existing_docs = document_service.get_session_documents(candidate)
                if existing_docs:
                    has_existing = True
            except Exception:
                pass

        if has_existing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot select an existing session identity without a valid session token.",
            )
        resolved_session_id = candidate
        token = set_session_cookie(response, resolved_session_id, request=request)

    max_bytes = document_service.MAX_FILE_SIZE_BYTES
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB.",
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
                    detail=f"File exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB.",
                )
            chunks.append(chunk)
        content_bytes = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    try:
        result = document_service.index_document(
            session_id=resolved_session_id,
            filename=file.filename or "uploaded_document",
            content_bytes=content_bytes,
        )
        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            session_id=resolved_session_id,
            session_token=token,
        )
    except document_service.DocumentValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error processing document upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {e}",
        )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
def delete_document(
    document_id: str,
    request: Request,
    session_id: str = Query(..., min_length=1, description="Owning session ID for document deletion"),
):
    """
    Deletes an uploaded document and its vector chunks.
    Enforces session ownership.
    """
    verify_csrf_protection(request)
    verified_session = require_verified_session(request)

    clean_session_id = session_id.strip() if session_id else ""
    if not clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid, non-empty session_id is required for document deletion.",
        )

    if verified_session != clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: document belongs to a different session.",
        )

    try:
        return document_service.delete_document(
            document_id=document_id,
            session_id=verified_session,
        )
    except document_service.DocumentValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except document_service.DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error deleting document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the document: {e}",
        )
