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
    MEDIAN = "median"
    STDDEV = "stddev"
    PERCENTILE_CONT = "percentile_cont"


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
    quantile: Optional[float] = Field(
        default=None, description="Quantile parameter for PERCENTILE_CONT (0.0 to 1.0)"
    )

    @model_validator(mode="after")
    def validate_quantile_usage(self) -> "AggregationClause":
        if self.function == AggregationFunction.PERCENTILE_CONT:
            if self.quantile is None:
                raise ValueError("PERCENTILE_CONT requires a 'quantile' parameter between 0.0 and 1.0.")
            if not (0.0 <= self.quantile <= 1.0):
                raise ValueError(f"quantile must be between 0.0 and 1.0, got {self.quantile}.")
        else:
            if self.quantile is not None:
                raise ValueError(f"function '{self.function.value}' does not accept a 'quantile' parameter.")
        return self


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


class DerivedMetricOperation(str, Enum):
    RATIO = "ratio"
    PERCENTAGE = "percentage"


class DerivedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: DerivedMetricOperation = Field(
        default=DerivedMetricOperation.RATIO, description="Arithmetic ratio or percentage operation"
    )
    numerator: str = Field(..., description="Numerator numeric column")
    denominator: str = Field(..., description="Denominator numeric column")
    multiplier: float = Field(
        default=1.0, description="Multiplier (defaults to 1.0 for ratio, 100.0 for percentage)"
    )
    alias: str = Field(..., description="Output alias for the derived metric")

    @model_validator(mode="before")
    @classmethod
    def normalize_derived_metric(cls, data: Any) -> Any:
        if isinstance(data, dict):
            op = data.get("operation")
            if (op == "percentage" or op == DerivedMetricOperation.PERCENTAGE) and "multiplier" not in data:
                data["multiplier"] = 100.0
        return data


class CorrelationClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_a: str = Field(..., description="First numeric column for Pearson correlation")
    column_b: str = Field(..., description="Second numeric column for Pearson correlation")
    alias: Optional[str] = Field(default=None, description="Output alias for correlation")


class PeriodComparisonMode(str, Enum):
    GROWTH_PCT = "growth_pct"
    DELTA = "delta"


class PeriodComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_column: str = Field(..., description="Date/timestamp column for period ordering")
    metric: str = Field(..., description="Numeric metric column to compare across periods")
    period: DateTrunc = Field(default=DateTrunc.MONTH, description="Period rollup bucket")
    mode: PeriodComparisonMode = Field(
        default=PeriodComparisonMode.GROWTH_PCT, description="Growth percentage or absolute delta"
    )
    alias: Optional[str] = Field(default=None, description="Output alias for period metric")


# ---------------------------------------------------------------------------
# Phase 7: Windowed Analytics, Cumulative Totals, Outliers & Quantiles
# ---------------------------------------------------------------------------

class WindowFunction(str, Enum):
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


class RollingWindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: WindowFunction = Field(..., description="Aggregation function over the window")
    column: str = Field(..., description="Target numeric column")
    order_by: str = Field(..., description="Date/timestamp or sequential order column")
    preceding: int = Field(..., ge=1, le=365, description="Number of preceding rows/periods to include")
    partition_by: list[str] = Field(default_factory=list, description="Categorical dimension columns to partition the window")
    alias: Optional[str] = Field(default=None, description="Output alias for the rolling metric")


class CumulativeFunction(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"


class CumulativeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: CumulativeFunction = Field(default=CumulativeFunction.SUM, description="Cumulative accumulation function")
    column: Optional[str] = Field(default=None, description="Numeric target column (required for sum/avg; optional for count)")
    order_by: str = Field(..., description="Date or chronological sequence column")
    partition_by: list[str] = Field(default_factory=list, description="Optional partition dimensions")
    alias: Optional[str] = Field(default=None, description="Output alias for cumulative metric")

    @model_validator(mode="after")
    def validate_cumulative_column(self) -> "CumulativeSpec":
        if self.function in (CumulativeFunction.SUM, CumulativeFunction.AVG) and not self.column:
            raise ValueError(f"Cumulative function '{self.function.value}' requires a target 'column'.")
        return self


class OutlierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Target numeric column to analyze for anomalies")
    threshold: float = Field(default=3.0, ge=1.0, le=10.0, description="Z-score sigma threshold (1.0 to 10.0)")
    partition_by: list[str] = Field(default_factory=list, description="Optional partition dimensions")
    filter_outliers_only: bool = Field(default=True, description="If True, filters output rows to only outliers (abs(z_score) >= threshold)")
    alias: Optional[str] = Field(default=None, description="Base alias prefix for outlier columns")


class QuantileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Numeric column used to calculate quantiles")
    num_quantiles: int = Field(default=4, ge=2, le=100, description="Number of quantile buckets (2 to 100)")
    order_descending: bool = Field(default=False, description="Whether higher values get bucket 1 (True) or bucket N (False)")
    partition_by: list[str] = Field(default_factory=list, description="Optional partition dimensions")
    alias: Optional[str] = Field(default=None, description="Output alias for quantile bucket number")


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="UUID of the active dataset")
    filters: list[FilterClause] = Field(default_factory=list, description="Row filter conditions")
    group_by: list[GroupByClause] = Field(default_factory=list, description="Group by dimensions")
    aggregations: list[AggregationClause] = Field(default_factory=list, description="Aggregate metrics")
    derived_metrics: list[DerivedMetric] = Field(
        default_factory=list, description="Constrained ratio/percentage derived metrics"
    )
    correlations: list[CorrelationClause] = Field(
        default_factory=list, description="Bivariate Pearson correlation calculations"
    )
    period_comparison: Optional[PeriodComparison] = Field(
        default=None, description="Deterministic period-over-period comparison"
    )
    # Phase 7 Extensions
    rolling_windows: list[RollingWindowSpec] = Field(
        default_factory=list, description="Rolling window aggregations"
    )
    cumulative_metrics: list[CumulativeSpec] = Field(
        default_factory=list, description="Running cumulative totals/averages/counts"
    )
    outlier_analysis: Optional[OutlierSpec] = Field(
        default=None, description="Z-score anomaly & outlier detection"
    )
    quantiles: list[QuantileSpec] = Field(
        default_factory=list, description="Quantile segmentation (NTILE)"
    )
    # Phase 9: Cross-Tab / Pivot Analysis
    pivot: Optional["PivotSpec"] = Field(
        default=None, description="Two-dimensional cross-tabulation / pivot matrix specification"
    )

    order_by: list[OrderByClause] = Field(default_factory=list, description="Sort order clauses")
    limit: int = Field(default=50, ge=1, le=500, description="Max rows to return (clamped 1-500)")
    requires_chart: bool = Field(default=False, description="Whether visualization is requested")
    chart_type: Optional[Literal["bar", "line", "scatter", "histogram", "heatmap"]] = Field(
        default=None, description="Suggested chart visualization type"
    )


# ---------------------------------------------------------------------------
# Phase 9: Cross-Tab / Pivot Analysis & Heatmap Visualization
# ---------------------------------------------------------------------------

class PivotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_dimension: str = Field(..., description="Categorical or discrete column for pivot table rows")
    column_dimension: str = Field(..., description="Categorical or discrete column for pivot table columns")
    metric: str = Field(..., description="Numeric column to aggregate across row and column dimensions")
    aggregation: AggregationFunction = Field(
        default=AggregationFunction.SUM,
        description="Aggregation function (sum, avg, count, min, max)"
    )
    max_rows: int = Field(default=20, ge=1, le=50, description="Max distinct row dimension values (clamped 1-50)")
    max_columns: int = Field(default=15, ge=1, le=30, description="Max distinct column dimension values (clamped 1-30)")
    include_totals: bool = Field(default=True, description="Whether to compute marginal row/column totals")
    alias: Optional[str] = Field(default=None, description="Optional matrix alias")

    @model_validator(mode="after")
    def validate_pivot_dimensions(self) -> "PivotSpec":
        if self.row_dimension.lower() == self.column_dimension.lower():
            raise ValueError("row_dimension and column_dimension must be distinct columns.")
        return self


class PivotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_dimension: str
    column_dimension: str
    metric: str
    aggregation: str
    rows: list[str]
    columns: list[str]
    values: list[list[Optional[float]]]
    row_totals: Optional[list[Optional[float]]] = None
    column_totals: Optional[list[Optional[float]]] = None
    grand_total: Optional[float] = None


class AnalysisResult(BaseModel):
    plan: AnalysisPlan
    compiled_sql: str
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    columns: list[str]
    execution_time_ms: float
    pivot_table: Optional[PivotResult] = None


class ChartArtifact(BaseModel):
    chart_id: str
    chart_type: Literal["bar", "line", "scatter", "histogram", "heatmap"]
    filename: str
    title: str
    width_px: int
    height_px: int
    url: str


class AnalysisTable(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int


# ---------------------------------------------------------------------------
# Phase 8: Multi-Turn Analytical Memory & Incremental AST Mutation
# ---------------------------------------------------------------------------

class PlanMutationType(str, Enum):
    NEW_QUERY = "new_query"
    RESET_ANALYSIS = "reset_analysis"
    ADD_FILTER = "add_filter"
    REMOVE_FILTER = "remove_filter"
    CHANGE_FILTER = "change_filter"
    CHANGE_GROUP_BY = "change_group_by"
    ADD_GROUP_BY = "add_group_by"
    REMOVE_GROUP_BY = "remove_group_by"
    CHANGE_METRIC = "change_metric"
    CHANGE_SORT = "change_sort"
    CHANGE_LIMIT = "change_limit"
    ADD_ROLLING = "add_rolling"
    ADD_CUMULATIVE = "add_cumulative"
    TOGGLE_CHART = "toggle_chart"
    # Phase 9
    TOGGLE_HEATMAP = "toggle_heatmap"
    CHANGE_PIVOT_COLUMNS = "change_pivot_columns"


class FilterDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Target dataset column name")
    operator: Optional[FilterOperator] = Field(
        default=None, description="Comparison or membership operator"
    )
    value: Optional[Union[str, int, float, bool, list[Union[str, int, float]]]] = Field(
        default=None, description="Literal value, range, or list of values for filtering"
    )

    def to_filter_clause(self) -> FilterClause:
        if self.operator is None:
            raise ValueError(f"Cannot convert FilterDelta for column '{self.column}' to FilterClause without an operator.")
        return FilterClause(column=self.column, operator=self.operator, value=self.value)


class GroupByDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., description="Target dimension column name")
    date_trunc: Optional[DateTrunc] = Field(
        default=None, description="Optional date truncation bucket (year, quarter, month, day)"
    )

    @field_validator("date_trunc", mode="before")
    @classmethod
    def normalize_date_trunc(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower() in ("null", "none", ""):
            return None
        return v

    def to_group_by_clause(self) -> GroupByClause:
        return GroupByClause(column=self.column, date_trunc=self.date_trunc)


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: Optional[str] = Field(
        default=None, description="Target column, or None/'*' for general row counts"
    )
    function: AggregationFunction = Field(..., description="Aggregation function")
    alias: Optional[str] = Field(default=None, description="Explicit output column alias")
    quantile: Optional[float] = Field(
        default=None, description="Quantile parameter for PERCENTILE_CONT (0.0 to 1.0)"
    )

    @model_validator(mode="after")
    def validate_quantile_usage(self) -> "MetricDelta":
        if self.function == AggregationFunction.PERCENTILE_CONT:
            if self.quantile is None:
                raise ValueError("PERCENTILE_CONT requires a 'quantile' parameter between 0.0 and 1.0.")
            if not (0.0 <= self.quantile <= 1.0):
                raise ValueError(f"quantile must be between 0.0 and 1.0, got {self.quantile}.")
        else:
            if self.quantile is not None:
                raise ValueError(f"function '{self.function.value}' does not accept a 'quantile' parameter.")
        return self

    def to_aggregation_clause(self) -> AggregationClause:
        return AggregationClause(
            column=self.column,
            function=self.function,
            alias=self.alias,
            quantile=self.quantile,
        )


class SortDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="Group-by column or aggregation alias to sort by")
    descending: bool = Field(default=True, description="True for DESC, False for ASC")

    def to_order_by_clause(self) -> OrderByClause:
        return OrderByClause(target=self.target, descending=self.descending)


class PivotDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_dimension: Optional[str] = Field(default=None, description="New row dimension")
    column_dimension: Optional[str] = Field(default=None, description="New column dimension")
    metric: Optional[str] = Field(default=None, description="New metric column")
    aggregation: Optional[AggregationFunction] = Field(default=None, description="New aggregation function")
    max_rows: Optional[int] = Field(default=None, ge=1, le=50, description="Max rows clamp")
    max_columns: Optional[int] = Field(default=None, ge=1, le=30, description="Max columns clamp")
    include_totals: Optional[bool] = Field(default=None, description="Whether to include totals")


class PlanMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mutation_type: PlanMutationType = Field(..., description="Type of mutation to apply")
    filter_delta: Optional[FilterDelta] = Field(
        default=None, description="Filter modification delta"
    )
    group_by_delta: Optional[GroupByDelta] = Field(
        default=None, description="Group-by modification delta"
    )
    metric_delta: Optional[MetricDelta] = Field(
        default=None, description="Metric modification delta"
    )
    sort_delta: Optional[SortDelta] = Field(
        default=None, description="Sort modification delta"
    )
    limit_delta: Optional[int] = Field(
        default=None, ge=1, le=500, description="New row limit (clamped 1-500)"
    )
    rolling_delta: Optional[RollingWindowSpec] = Field(
        default=None, description="Rolling window specification to add or update"
    )
    cumulative_delta: Optional[CumulativeSpec] = Field(
        default=None, description="Cumulative metric specification to add or update"
    )
    pivot_delta: Optional[PivotDelta] = Field(
        default=None, description="Pivot modification delta"
    )
    chart_type: Optional[Literal["bar", "line", "scatter", "histogram", "heatmap"]] = Field(
        default=None, description="Chart visualization type"
    )
    requires_chart: Optional[bool] = Field(
        default=None, description="Whether visualization is requested"
    )
    new_plan: Optional[AnalysisPlan] = Field(
        default=None, description="Complete replacement plan for NEW_QUERY mutations"
    )
