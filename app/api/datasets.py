"""
API endpoints for dataset upload, profiling, inspection, and deletion.
"""
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status

from app.analytics.schemas import (
    DatasetUploadResponse,
    DatasetMetadata,
    DatasetDeleteResponse,
)
from app.analytics import dataset_service

logger = logging.getLogger("app.analytics.api")

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload and profile a tabular dataset (CSV, XLSX, JSON).
    Deterministic fast profiling provides column types, missing value ratios, and duplicate counts.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid session_id is required.",
        )

    try:
        content_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded dataset file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    try:
        result = dataset_service.ingest_dataset(
            session_id=session_id.strip(),
            filename=file.filename or "dataset.csv",
            content_bytes=content_bytes,
        )
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
    session_id: Optional[str] = Query(default=None),
):
    """Retrieves metadata and fast profile for an uploaded dataset."""
    try:
        return dataset_service.get_dataset(dataset_id=dataset_id, session_id=session_id)
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
    session_id: str = Query(...),
):
    """Lists all datasets associated with an active session."""
    return dataset_service.list_datasets(session_id=session_id)


@router.delete(
    "/{dataset_id}",
    response_model=DatasetDeleteResponse,
)
def delete_dataset(
    dataset_id: str,
    session_id: Optional[str] = Query(default=None),
):
    """Deletes an uploaded dataset and all stored artifacts."""
    try:
        return dataset_service.delete_dataset(dataset_id=dataset_id, session_id=session_id)
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
