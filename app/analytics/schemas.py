"""
Pydantic schemas for dataset profiling, metadata, and API responses.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str = Field(..., description="Column header name")
    dtype: str = Field(..., description="Inferred data type (e.g. int64, float64, object, datetime64)")
    missing_count: int = Field(..., description="Count of null or missing values")
    missing_percentage: float = Field(..., description="Proportion of missing values (0.0 to 1.0)")
    unique_count: int = Field(..., description="Count of distinct non-null values")


class DatasetProfile(BaseModel):
    row_count: int = Field(..., description="Total number of data rows")
    column_count: int = Field(..., description="Total number of columns")
    columns: list[ColumnProfile] = Field(default_factory=list, description="Profiles for each column")
    missing_cells: int = Field(..., description="Total missing cells across all columns")
    duplicate_rows: int = Field(..., description="Count of duplicate rows")


class DatasetMetadata(BaseModel):
    dataset_id: str
    session_id: str
    filename: str
    format: str
    row_count: int
    column_count: int
    created_at: str
    profile: DatasetProfile


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    format: str
    row_count: int
    column_count: int
    profile: DatasetProfile


class DatasetDeleteResponse(BaseModel):
    dataset_id: str
    deleted: bool = True
    message: str = "Dataset deleted successfully."
