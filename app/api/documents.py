"""
API endpoints for document upload and deletion.
"""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status
import logging

from app.api.schemas import DocumentUploadResponse, DocumentDeleteResponse
from app.services import document_service

logger = logging.getLogger("app.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload and index a document (PDF, DOCX, TXT) for RAG document assistance.
    Isolates document vectors by session_id in the uploaded_documents collection.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid session_id is required.",
        )

    try:
        content_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    try:
        result = document_service.index_document(
            session_id=session_id.strip(),
            filename=file.filename or "uploaded_document",
            content_bytes=content_bytes,
        )
        return result
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
    session_id: Optional[str] = Query(default=None),
):
    """
    Deletes an uploaded document and its vector chunks.
    Optionally enforces session ownership if session_id is supplied.
    """
    try:
        return document_service.delete_document(
            document_id=document_id,
            session_id=session_id,
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
