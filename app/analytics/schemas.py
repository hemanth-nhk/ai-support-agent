"""
Pydantic schemas for dataset profiling, metadata, analysis plans, and execution responses.
"""
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# ---------------------------------------------------------------------------
# AnalysisPlan AST Schemas
# ---------------------------------------------------------------------------

class AggregationFunction(str, Enum):
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class DateTrunc(str, Enum):
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"


class FilterOperator(str, Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"


class FilterClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Target dataset column name")
    operator: FilterOperator = Field(..., description="Comparison or membership operator")
    value: Optional[Union[str, int, float, bool, list[Union[str, int, float]]]] = Field(
        default=None, description="Literal value, range, or list of values for filtering"
    )


class AggregationClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: Optional[str] = Field(
        default=None, description="Target column, or None/'*' for general row counts"
    )
    function: AggregationFunction = Field(..., description="Aggregation function")
    alias: Optional[str] = Field(
        default=None, description="Explicit output column alias"
    )


class GroupByClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Column to group by")
    date_trunc: Optional[DateTrunc] = Field(
        default=None, description="Optional date truncation bucket (year, quarter, month, day)"
    )

    @field_validator("date_trunc", mode="before")
    @classmethod
    def normalize_date_trunc(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower() in ("null", "none", ""):
            return None
        return v


class OrderByClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="Group-by column or aggregation alias to sort by")
    descending: bool = Field(default=True, description="True for DESC, False for ASC")

    @model_validator(mode="before")
    @classmethod
    def normalize_order_by(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "column" in data and "target" not in data:
                data["target"] = data.pop("column")
            if "direction" in data and "descending" not in data:
                val = str(data.pop("direction")).lower()
                data["descending"] = val in ("desc", "descending")
        return data


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="UUID of the active dataset")
    filters: list[FilterClause] = Field(default_factory=list, description="Row filter conditions")
    group_by: list[GroupByClause] = Field(default_factory=list, description="Group by dimensions")
    aggregations: list[AggregationClause] = Field(default_factory=list, description="Aggregate metrics")
    order_by: list[OrderByClause] = Field(default_factory=list, description="Sort order clauses")
    limit: int = Field(default=50, ge=1, le=500, description="Max rows to return (clamped 1-500)")
    requires_chart: bool = Field(default=False, description="Whether visualization is requested")
    chart_type: Optional[Literal["bar", "line", "scatter", "histogram"]] = Field(
        default=None, description="Suggested chart visualization type"
    )


class AnalysisResult(BaseModel):
    plan: AnalysisPlan
    compiled_sql: str
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    columns: list[str]
    execution_time_ms: float


class ChartArtifact(BaseModel):
    chart_id: str
    chart_type: Literal["bar", "line", "scatter", "histogram"]
    filename: str
    title: str
    width_px: int
    height_px: int
    url: str


class AnalysisTable(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
