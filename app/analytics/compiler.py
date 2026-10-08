"""
Deterministic SQL Compiler for AnalysisPlan AST targeting DuckDB.
Performs strict schema validation, type checking, identifier escaping,
and parameter binding. Rejects unvalidated column names and raw SQL strings.
"""
from datetime import datetime, date
from typing import Any, Optional, Union
from pydantic import BaseModel

from app.analytics.schemas import (
    AnalysisPlan,
    AggregationFunction,
    DateTrunc,
    FilterOperator,
    ColumnProfile,
    DatasetMetadata,
    DerivedMetric,
    DerivedMetricOperation,
    CorrelationClause,
    PeriodComparison,
    PeriodComparisonMode,
    WindowFunction,
    RollingWindowSpec,
    CumulativeFunction,
    CumulativeSpec,
    QuantileSpec,
    OutlierSpec,
    PivotSpec,
)


class PlanValidationError(Exception):
    """Raised when the analysis plan contains logically invalid or ill-formed clauses."""
    pass


class ColumnNotFoundError(Exception):
    """Raised when an analysis plan references a column that does not exist in the dataset."""
    pass


class TypeMismatchError(Exception):
    """Raised when an operation or comparison is incompatible with the column's data type."""
    pass


class CompiledQuery(BaseModel):
    sql: str
    parameters: list[Any]


def escape_identifier(name: str) -> str:
    """Escapes SQL identifiers using double quotes and escaped internal double quotes."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def classify_dtype(dtype_str: str, col_name: str = "") -> str:
    """Categorizes column data types into 'numeric', 'datetime', 'boolean', or 'string'."""
    lowered = dtype_str.lower()
    if any(k in lowered for k in ("int", "float", "double", "decimal", "numeric", "real")):
        return "numeric"
    if any(k in lowered for k in ("date", "time", "timestamp")):
        return "datetime"
    if "bool" in lowered:
        return "boolean"
    if col_name and any(k in col_name.lower() for k in ("_date", "date_", "date", "_time", "timestamp")):
        return "datetime"
    return "string"


def validate_date_value(val: Any, col_name: str) -> str:
    """Validates that a date filter value is a valid ISO date/datetime string."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if not isinstance(val, str):
        raise TypeMismatchError(
            f"Expected date string for column '{col_name}', got {type(val).__name__}."
        )
    val_clean = val.strip()
    try:
        # Check standard ISO formats (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        datetime.fromisoformat(val_clean.replace("Z", "+00:00"))
        return val_clean
    except ValueError:
        raise PlanValidationError(
            f"Invalid date format '{val}' for column '{col_name}'. Expected ISO-8601 (YYYY-MM-DD)."
        )


def validate_numeric_value(val: Any, col_name: str) -> Union[int, float]:
    """Validates that a numeric filter value is a valid number."""
    if isinstance(val, bool):
        raise TypeMismatchError(f"Boolean value is not valid for numeric column '{col_name}'.")
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        # Attempt conversion if clean string representation
        val_str = val.strip()
        try:
            if "." in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            raise TypeMismatchError(
                f"Cannot compare numeric column '{col_name}' with non-numeric value '{val}'."
            )
    raise TypeMismatchError(
        f"Cannot compare numeric column '{col_name}' with value of type {type(val).__name__}."
    )


def _format_literal_sql(val: Any, col_dtype_cat: str) -> str:
    """Formats a validated filter value safely as a SQL literal without parameter placeholders."""
    if val is None:
        return "NULL"
    if col_dtype_cat == "numeric":
        return str(val)
    if col_dtype_cat == "boolean":
        return "TRUE" if val else "FALSE"
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _compile_pivot_query(
    plan: AnalysisPlan,
    metadata: DatasetMetadata,
    col_map: dict[str, ColumnProfile],
    col_types: dict[str, str],
) -> CompiledQuery:
    """Compiles a PivotSpec cross-tab AST into deterministic DuckDB PIVOT SQL."""
    pivot = plan.pivot
    if pivot is None:
        raise PlanValidationError("Missing pivot specification.")

    if pivot.row_dimension not in col_map:
        raise ColumnNotFoundError(f"Pivot row dimension '{pivot.row_dimension}' does not exist in dataset.")
    if pivot.column_dimension not in col_map:
        raise ColumnNotFoundError(f"Pivot column dimension '{pivot.column_dimension}' does not exist in dataset.")
    if pivot.row_dimension.lower() == pivot.column_dimension.lower():
        raise PlanValidationError("Pivot row dimension and column dimension cannot be the same column.")
    if pivot.metric not in col_map:
        raise ColumnNotFoundError(f"Pivot metric '{pivot.metric}' does not exist in dataset.")

    metric_type = col_types[pivot.metric]
    if pivot.aggregation != AggregationFunction.COUNT and metric_type != "numeric":
        raise TypeMismatchError(
            f"Pivot metric '{pivot.metric}' must be numeric for aggregation '{pivot.aggregation.value}', but is '{metric_type}'."
        )

    where_clauses: list[str] = []
    for f in plan.filters:
        if f.column not in col_map:
            raise ColumnNotFoundError(f"Column '{f.column}' does not exist in dataset. Available columns: {sorted(col_map.keys())}.")

        col_dtype_cat = col_types[f.column]
        col_ident = escape_identifier(f.column)

        if f.operator == FilterOperator.IS_NULL:
            where_clauses.append(f"{col_ident} IS NULL")
            continue
        elif f.operator == FilterOperator.IS_NOT_NULL:
            where_clauses.append(f"{col_ident} IS NOT NULL")
            continue

        if f.operator in (FilterOperator.CONTAINS, FilterOperator.STARTS_WITH):
            if col_dtype_cat != "string":
                raise TypeMismatchError(
                    f"Operator '{f.operator.value}' is only valid on text/string columns, not '{col_dtype_cat}' column '{f.column}'."
                )
            if not isinstance(f.value, str):
                raise TypeMismatchError(f"Operator '{f.operator.value}' requires a string value, got {type(f.value).__name__}.")
            val_escaped = str(f.value).replace("'", "''")
            if f.operator == FilterOperator.CONTAINS:
                where_clauses.append(f"{col_ident} ILIKE '%{val_escaped}%'")
            else:
                where_clauses.append(f"{col_ident} ILIKE '{val_escaped}%'")
            continue

        if f.operator in (FilterOperator.IN, FilterOperator.NOT_IN):
            if not isinstance(f.value, list) or len(f.value) == 0:
                raise PlanValidationError(f"Operator '{f.operator.value}' requires a non-empty list of values.")
            clean_vals = []
            for item in f.value:
                if col_dtype_cat == "numeric":
                    clean_vals.append(_format_literal_sql(validate_numeric_value(item, f.column), col_dtype_cat))
                elif col_dtype_cat == "datetime":
                    clean_vals.append(_format_literal_sql(validate_date_value(item, f.column), col_dtype_cat))
                else:
                    clean_vals.append(_format_literal_sql(item, col_dtype_cat))
            op_sql = "IN" if f.operator == FilterOperator.IN else "NOT IN"
            where_clauses.append(f"{col_ident} {op_sql} ({', '.join(clean_vals)})")
            continue

        if f.operator == FilterOperator.BETWEEN:
            if not isinstance(f.value, list) or len(f.value) != 2:
                raise PlanValidationError("Operator 'between' requires a list of exactly two boundary values [low, high].")
            if col_dtype_cat == "string":
                raise TypeMismatchError(f"Operator 'between' is not supported on text column '{f.column}'.")
            if col_dtype_cat == "numeric":
                low = _format_literal_sql(validate_numeric_value(f.value[0], f.column), col_dtype_cat)
                high = _format_literal_sql(validate_numeric_value(f.value[1], f.column), col_dtype_cat)
            elif col_dtype_cat == "datetime":
                low = _format_literal_sql(validate_date_value(f.value[0], f.column), col_dtype_cat)
                high = _format_literal_sql(validate_date_value(f.value[1], f.column), col_dtype_cat)
            else:
                low = _format_literal_sql(f.value[0], col_dtype_cat)
                high = _format_literal_sql(f.value[1], col_dtype_cat)
            where_clauses.append(f"{col_ident} BETWEEN {low} AND {high}")
            continue

        op_map = {
            FilterOperator.EQ: "=",
            FilterOperator.NEQ: "!=",
            FilterOperator.GT: ">",
            FilterOperator.GTE: ">=",
            FilterOperator.LT: "<",
            FilterOperator.LTE: "<=",
        }
        op_sql = op_map[f.operator]

        if f.operator in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
            if col_dtype_cat == "string":
                raise TypeMismatchError(f"Inequality operator '{f.operator.value}' is not supported on text column '{f.column}'.")

        if col_dtype_cat == "numeric":
            clean_val = _format_literal_sql(validate_numeric_value(f.value, f.column), col_dtype_cat)
        elif col_dtype_cat == "datetime":
            clean_val = _format_literal_sql(validate_date_value(f.value, f.column), col_dtype_cat)
        elif col_dtype_cat == "boolean":
            if not isinstance(f.value, bool):
                raise TypeMismatchError(f"Boolean column '{f.column}' requires boolean value (True/False).")
            clean_val = "TRUE" if f.value else "FALSE"
        else:
            clean_val = _format_literal_sql(str(f.value), col_dtype_cat)

        where_clauses.append(f"{col_ident} {op_sql} {clean_val}")

    row_dim_ident = escape_identifier(pivot.row_dimension)
    col_dim_ident = escape_identifier(pivot.column_dimension)
    metric_ident = escape_identifier(pivot.metric)
    agg_func = pivot.aggregation.value.upper()

    order_dir = "ASC"
    if plan.order_by:
        for o in plan.order_by:
            if o.target.lower() == pivot.row_dimension.lower():
                order_dir = "DESC" if o.descending else "ASC"
                break

    where_sql = f"\n    WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    clamped_max_rows = max(1, min(pivot.max_rows, 50))

    pivot_sql = f"""PIVOT (
    SELECT {row_dim_ident}, {col_dim_ident}, {metric_ident}
    FROM source_table{where_sql}
)
ON {col_dim_ident}
USING {agg_func}({metric_ident})
GROUP BY {row_dim_ident}
ORDER BY {row_dim_ident} {order_dir}
LIMIT {clamped_max_rows}"""

    return CompiledQuery(sql=pivot_sql.strip(), parameters=[])


def compile_plan(plan: AnalysisPlan, metadata: DatasetMetadata) -> CompiledQuery:
    """
    Compiles an AnalysisPlan into parameterized DuckDB SQL targeting 'source_table'.
    Enforces strict column validation, type checks, and parameter binding.
    """
    if plan.dataset_id != metadata.dataset_id:
        raise PlanValidationError(
            f"Plan dataset_id '{plan.dataset_id}' does not match metadata dataset_id '{metadata.dataset_id}'."
        )

    col_map: dict[str, ColumnProfile] = {c.name: c for c in metadata.profile.columns}
    col_types: dict[str, str] = {c.name: classify_dtype(c.dtype, c.name) for c in metadata.profile.columns}

    # Phase 9: Cross-tab / Pivot compilation branch
    if plan.pivot is not None:
        return _compile_pivot_query(plan, metadata, col_map, col_types)

    # 1. Validate Columns in Filters
    where_clauses: list[str] = []
    parameters: list[Any] = []

    for f in plan.filters:
        if f.column not in col_map:
            raise ColumnNotFoundError(
                f"Column '{f.column}' does not exist in dataset. Available columns: {sorted(col_map.keys())}."
            )

        col_dtype_cat = col_types[f.column]
        col_ident = escape_identifier(f.column)

        # Check operator compatibility by type
        if f.operator in (FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL):
            if f.operator == FilterOperator.IS_NULL:
                where_clauses.append(f"{col_ident} IS NULL")
            else:
                where_clauses.append(f"{col_ident} IS NOT NULL")
            continue

        if f.operator in (FilterOperator.CONTAINS, FilterOperator.STARTS_WITH):
            if col_dtype_cat != "string":
                raise TypeMismatchError(
                    f"Operator '{f.operator.value}' is only valid on text/string columns, not '{col_dtype_cat}' column '{f.column}'."
                )
            if not isinstance(f.value, str):
                raise TypeMismatchError(
                    f"Operator '{f.operator.value}' requires a string value, got {type(f.value).__name__}."
                )
            if f.operator == FilterOperator.CONTAINS:
                where_clauses.append(f"{col_ident} ILIKE ?")
                parameters.append(f"%{f.value}%")
            else:
                where_clauses.append(f"{col_ident} ILIKE ?")
                parameters.append(f"{f.value}%")
            continue

        if f.operator in (FilterOperator.IN, FilterOperator.NOT_IN):
            if not isinstance(f.value, list) or len(f.value) == 0:
                raise PlanValidationError(
                    f"Operator '{f.operator.value}' requires a non-empty list of values."
                )
            # Validate each item
            clean_vals = []
            for item in f.value:
                if col_dtype_cat == "numeric":
                    clean_vals.append(validate_numeric_value(item, f.column))
                elif col_dtype_cat == "datetime":
                    clean_vals.append(validate_date_value(item, f.column))
                else:
                    clean_vals.append(str(item))

            placeholders = ", ".join(["?"] * len(clean_vals))
            op_sql = "IN" if f.operator == FilterOperator.IN else "NOT IN"
            where_clauses.append(f"{col_ident} {op_sql} ({placeholders})")
            parameters.extend(clean_vals)
            continue

        if f.operator == FilterOperator.BETWEEN:
            if not isinstance(f.value, list) or len(f.value) != 2:
                raise PlanValidationError(
                    f"Operator 'between' requires a list of exactly two boundary values [low, high]."
                )
            if col_dtype_cat == "string":
                raise TypeMismatchError(
                    f"Operator 'between' is not supported on text column '{f.column}'."
                )

            if col_dtype_cat == "numeric":
                low = validate_numeric_value(f.value[0], f.column)
                high = validate_numeric_value(f.value[1], f.column)
            elif col_dtype_cat == "datetime":
                low = validate_date_value(f.value[0], f.column)
                high = validate_date_value(f.value[1], f.column)
            else:
                low, high = f.value[0], f.value[1]

            where_clauses.append(f"{col_ident} BETWEEN ? AND ?")
            parameters.extend([low, high])
            continue

        # Standard comparisons (==, !=, >, >=, <, <=)
        op_map = {
            FilterOperator.EQ: "=",
            FilterOperator.NEQ: "!=",
            FilterOperator.GT: ">",
            FilterOperator.GTE: ">=",
            FilterOperator.LT: "<",
            FilterOperator.LTE: "<=",
        }
        op_sql = op_map[f.operator]

        if f.operator in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
            if col_dtype_cat == "string":
                raise TypeMismatchError(
                    f"Inequality operator '{f.operator.value}' is not supported on text column '{f.column}'."
                )

        if col_dtype_cat == "numeric":
            clean_val = validate_numeric_value(f.value, f.column)
        elif col_dtype_cat == "datetime":
            clean_val = validate_date_value(f.value, f.column)
        elif col_dtype_cat == "boolean":
            if not isinstance(f.value, bool):
                raise TypeMismatchError(
                    f"Boolean column '{f.column}' requires boolean value (True/False)."
                )
            clean_val = f.value
        else:
            clean_val = str(f.value)

        where_clauses.append(f"{col_ident} {op_sql} ?")
        parameters.append(clean_val)

    # 2. Validate and Compile GroupBy
    select_items: list[str] = []
    group_by_items: list[str] = []
    available_sort_targets: set[str] = set()

    for g in plan.group_by:
        if g.column not in col_map:
            raise ColumnNotFoundError(
                f"Group by column '{g.column}' does not exist in dataset."
            )

        col_dtype_cat = col_types[g.column]
        col_ident = escape_identifier(g.column)

        if g.date_trunc is not None:
            if col_dtype_cat != "datetime":
                raise TypeMismatchError(
                    f"Cannot apply date_trunc to non-datetime column '{g.column}' (dtype: {col_map[g.column].dtype})."
                )
            alias = f"{g.date_trunc.value}_{g.column}"
            alias_ident = escape_identifier(alias)
            trunc_expr = f"DATE_TRUNC('{g.date_trunc.value}', {col_ident})"
            select_items.append(f"{trunc_expr} AS {alias_ident}")
            group_by_items.append(trunc_expr)
            available_sort_targets.add(alias)
            available_sort_targets.add(g.column)
        else:
            select_items.append(col_ident)
            group_by_items.append(col_ident)
            available_sort_targets.add(g.column)

    # 3. Validate and Compile Aggregations
    for agg in plan.aggregations:
        if agg.column in (None, "*"):
            if agg.function != AggregationFunction.COUNT:
                raise PlanValidationError(
                    f"Aggregation function '{agg.function.value}' requires an explicit column name."
                )
            target_col = None
        else:
            if agg.column not in col_map:
                raise ColumnNotFoundError(
                    f"Aggregation column '{agg.column}' does not exist in dataset."
                )
            target_col = agg.column

        alias = agg.alias or (f"{agg.function.value}_{target_col}" if target_col else "count_rows")
        alias_ident = escape_identifier(alias)
        available_sort_targets.add(alias)

        if target_col is not None:
            col_dtype_cat = col_types[target_col]
            col_ident = escape_identifier(target_col)

            if agg.function in (AggregationFunction.SUM, AggregationFunction.AVG, AggregationFunction.MEDIAN, AggregationFunction.STDDEV, AggregationFunction.PERCENTILE_CONT):
                if col_dtype_cat != "numeric":
                    raise TypeMismatchError(
                        f"Cannot compute {agg.function.value.upper()} on non-numeric column '{target_col}' (dtype: {col_map[target_col].dtype})."
                    )

            if agg.function in (AggregationFunction.MIN, AggregationFunction.MAX):
                if col_dtype_cat not in ("numeric", "datetime"):
                    raise TypeMismatchError(
                        f"Cannot compute {agg.function.value.upper()} on '{col_dtype_cat}' column '{target_col}'."
                    )

            if agg.function == AggregationFunction.COUNT:
                select_items.append(f"COUNT({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.COUNT_DISTINCT:
                select_items.append(f"COUNT(DISTINCT {col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.SUM:
                select_items.append(f"SUM({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.AVG:
                select_items.append(f"AVG({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.MIN:
                select_items.append(f"MIN({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.MAX:
                select_items.append(f"MAX({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.MEDIAN:
                select_items.append(f"MEDIAN({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.STDDEV:
                select_items.append(f"STDDEV_SAMP({col_ident}) AS {alias_ident}")
            elif agg.function == AggregationFunction.PERCENTILE_CONT:
                select_items.append(f"QUANTILE_CONT({col_ident}, {float(agg.quantile or 0.5)}) AS {alias_ident}")
        else:
            select_items.append(f"COUNT(*) AS {alias_ident}")

    # 3.1 Validate and Compile Correlations
    for c in plan.correlations:
        if c.column_a not in col_map:
            raise ColumnNotFoundError(f"Correlation column '{c.column_a}' does not exist in dataset.")
        if c.column_b not in col_map:
            raise ColumnNotFoundError(f"Correlation column '{c.column_b}' does not exist in dataset.")
        if col_types[c.column_a] != "numeric":
            raise TypeMismatchError(f"Correlation requires numeric column, got '{c.column_a}' (dtype: {col_map[c.column_a].dtype}).")
        if col_types[c.column_b] != "numeric":
            raise TypeMismatchError(f"Correlation requires numeric column, got '{c.column_b}' (dtype: {col_map[c.column_b].dtype}).")

        alias = c.alias or f"corr_{c.column_a}_{c.column_b}"
        alias_ident = escape_identifier(alias)
        available_sort_targets.add(alias)
        col_a_ident = escape_identifier(c.column_a)
        col_b_ident = escape_identifier(c.column_b)
        select_items.append(f"NULLIF(CORR({col_a_ident}, {col_b_ident}), 'nan'::DOUBLE) AS {alias_ident}")

    # 3.2 Validate and Compile Derived Metrics (Ratios & Percentages)
    for dm in plan.derived_metrics:
        if dm.numerator not in col_map:
            raise ColumnNotFoundError(f"Derived metric numerator '{dm.numerator}' does not exist in dataset.")
        if dm.denominator not in col_map:
            raise ColumnNotFoundError(f"Derived metric denominator '{dm.denominator}' does not exist in dataset.")
        if col_types[dm.numerator] != "numeric":
            raise TypeMismatchError(f"Derived metric numerator '{dm.numerator}' must be numeric (dtype: {col_map[dm.numerator].dtype}).")
        if col_types[dm.denominator] != "numeric":
            raise TypeMismatchError(f"Derived metric denominator '{dm.denominator}' must be numeric (dtype: {col_map[dm.denominator].dtype}).")

        alias_ident = escape_identifier(dm.alias)
        available_sort_targets.add(dm.alias)
        num_ident = escape_identifier(dm.numerator)
        den_ident = escape_identifier(dm.denominator)
        mult = float(dm.multiplier)
        select_items.append(f"( (SUM({num_ident}) * {mult}) / NULLIF(SUM({den_ident}), 0) ) AS {alias_ident}")

    # 3.3 Validate and Compile Period Comparison (Period-over-Period with LAG)
    if plan.period_comparison is not None:
        pc = plan.period_comparison
        if pc.date_column not in col_map:
            raise ColumnNotFoundError(f"Period comparison date column '{pc.date_column}' not found.")
        if col_types[pc.date_column] != "datetime":
            raise TypeMismatchError(f"Period comparison requires datetime column, got '{pc.date_column}' (dtype: {col_map[pc.date_column].dtype}).")
        if pc.metric not in col_map:
            raise ColumnNotFoundError(f"Period comparison metric column '{pc.metric}' not found.")
        if col_types[pc.metric] != "numeric":
            raise TypeMismatchError(f"Period comparison metric '{pc.metric}' must be numeric (dtype: {col_map[pc.metric].dtype}).")

        date_ident = escape_identifier(pc.date_column)
        metric_ident = escape_identifier(pc.metric)
        alias = pc.alias or (f"{pc.metric}_growth_pct" if pc.mode == PeriodComparisonMode.GROWTH_PCT else f"{pc.metric}_delta")
        alias_ident = escape_identifier(alias)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        clamped_limit = max(1, min(plan.limit, 500))

        if pc.mode == PeriodComparisonMode.GROWTH_PCT:
            comp_expr = "( (current_val - LAG(current_val) OVER (ORDER BY period)) * 100.0 ) / NULLIF(ABS(LAG(current_val) OVER (ORDER BY period)), 0)"
        else:
            comp_expr = "(current_val - LAG(current_val) OVER (ORDER BY period))"

        sql = (
            f"WITH period_summary AS (\n"
            f"  SELECT\n"
            f"    DATE_TRUNC('{pc.period.value}', {date_ident}) AS period,\n"
            f"    SUM({metric_ident}) AS current_val\n"
            f"  FROM source_table\n"
            f"  {where_sql}\n"
            f"  GROUP BY 1\n"
            f")\n"
            f"SELECT\n"
            f"  period,\n"
            f"  current_val,\n"
            f"  LAG(current_val) OVER (ORDER BY period) AS prev_val,\n"
            f"  {comp_expr} AS {alias_ident}\n"
            f"FROM period_summary\n"
            f"ORDER BY period ASC\n"
            f"LIMIT {clamped_limit}"
        )
        return CompiledQuery(sql=sql, parameters=parameters)

    # 3.35 Validate and Compile Outlier Analysis (Z-score anomaly detection CTE)
    if plan.outlier_analysis is not None:
        oa = plan.outlier_analysis
        if oa.column not in col_map:
            raise ColumnNotFoundError(f"Outlier analysis column '{oa.column}' not found in dataset.")
        if col_types[oa.column] != "numeric":
            raise TypeMismatchError(
                f"Outlier analysis requires numeric column, got '{oa.column}' (dtype: {col_map[oa.column].dtype})."
            )
        for p in oa.partition_by:
            if p not in col_map:
                raise ColumnNotFoundError(f"Outlier analysis partition column '{p}' not found in dataset.")

        col_ident = escape_identifier(oa.column)
        part_sql = f"PARTITION BY {', '.join(escape_identifier(p) for p in oa.partition_by)}" if oa.partition_by else ""
        prefix = oa.alias or oa.column
        mean_alias = escape_identifier(f"mean_{prefix}")
        stddev_alias = escape_identifier(f"stddev_{prefix}")
        z_score_alias = escape_identifier(f"z_score_{prefix}")
        is_outlier_alias = escape_identifier("is_outlier")

        threshold_val = float(oa.threshold)
        clamped_limit = max(1, min(plan.limit, 500))
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Outer where condition to filter outliers only
        if oa.filter_outliers_only:
            outer_where = f"WHERE _std_val IS NOT NULL AND _std_val > 0 AND ABS(({col_ident} - _mean_val) / _std_val) >= {threshold_val}"
        else:
            outer_where = ""

        # Projected columns: all source columns + outlier statistical columns
        source_cols = [escape_identifier(c) for c in col_map]
        select_cols = (
            source_cols
            + [
                f"ROUND(_mean_val, 4) AS {mean_alias}",
                f"ROUND(_std_val, 4) AS {stddev_alias}",
                f"CASE WHEN _std_val IS NULL OR _std_val = 0 THEN NULL ELSE ROUND(({col_ident} - _mean_val) / _std_val, 4) END AS {z_score_alias}",
                f"CASE WHEN _std_val IS NULL OR _std_val = 0 THEN FALSE WHEN ABS(({col_ident} - _mean_val) / _std_val) >= {threshold_val} THEN TRUE ELSE FALSE END AS {is_outlier_alias}",
            ]
        )

        # Ordering
        if plan.order_by:
            valid_targets = set(col_map.keys()) | {f"mean_{prefix}", f"stddev_{prefix}", f"z_score_{prefix}", "is_outlier"}
            order_parts = []
            for o in plan.order_by:
                if o.target not in valid_targets:
                    raise PlanValidationError(
                        f"Order by target '{o.target}' is not in selected outlier projection. Available: {sorted(valid_targets)}."
                    )
                target_ident = escape_identifier(o.target)
                direction = "DESC" if o.descending else "ASC"
                order_parts.append(f"{target_ident} {direction} NULLS LAST")
            order_sql = f"ORDER BY {', '.join(order_parts)}"
        else:
            # Default ordering: most extreme outliers first
            order_sql = f"ORDER BY CASE WHEN _std_val IS NULL OR _std_val = 0 THEN 0 ELSE ABS(({col_ident} - _mean_val) / _std_val) END DESC, {col_ident} DESC"

        sql_parts = [
            f"WITH stats_cte AS (",
            f"  SELECT",
            f"    *,",
            f"    AVG({col_ident}) OVER ({part_sql}) AS _mean_val,",
            f"    STDDEV_SAMP({col_ident}) OVER ({part_sql}) AS _std_val",
            f"  FROM source_table",
            f"  {where_sql}",
            f")",
            f"SELECT",
            f"  {', '.join(select_cols)}",
            f"FROM stats_cte",
        ]
        if outer_where:
            sql_parts.append(outer_where)
        sql_parts.append(order_sql)
        sql_parts.append(f"LIMIT {clamped_limit}")

        return CompiledQuery(sql="\n".join(sql_parts), parameters=parameters)

    # 3.4 Validate and Compile Rolling Windows
    for rw in plan.rolling_windows:
        if rw.column not in col_map:
            raise ColumnNotFoundError(f"Rolling window column '{rw.column}' does not exist in dataset.")
        if col_types[rw.column] != "numeric":
            raise TypeMismatchError(
                f"Rolling window function '{rw.function.value}' requires numeric column, got '{rw.column}' (dtype: {col_map[rw.column].dtype})."
            )
        if rw.order_by not in col_map:
            raise ColumnNotFoundError(f"Rolling window order_by column '{rw.order_by}' does not exist in dataset.")
        for p in rw.partition_by:
            if p not in col_map:
                raise ColumnNotFoundError(f"Rolling window partition column '{p}' does not exist in dataset.")

        alias = rw.alias or f"rolling_{rw.function.value}_{rw.column}"
        alias_ident = escape_identifier(alias)
        available_sort_targets.add(alias)
        col_ident = escape_identifier(rw.column)
        order_ident = escape_identifier(rw.order_by)
        part_sql = f"PARTITION BY {', '.join(escape_identifier(p) for p in rw.partition_by)} " if rw.partition_by else ""
        func_sql = rw.function.value.upper()
        preceding_val = int(rw.preceding)
        select_items.append(
            f"{func_sql}({col_ident}) OVER ({part_sql}ORDER BY {order_ident} ASC ROWS BETWEEN {preceding_val} PRECEDING AND CURRENT ROW) AS {alias_ident}"
        )

    # 3.5 Validate and Compile Cumulative Metrics
    for cm in plan.cumulative_metrics:
        if cm.column:
            if cm.column not in col_map:
                raise ColumnNotFoundError(f"Cumulative metric column '{cm.column}' does not exist in dataset.")
            if cm.function in (CumulativeFunction.SUM, CumulativeFunction.AVG) and col_types[cm.column] != "numeric":
                raise TypeMismatchError(
                    f"Cumulative function '{cm.function.value}' requires numeric column, got '{cm.column}' (dtype: {col_map[cm.column].dtype})."
                )
            col_target_sql = escape_identifier(cm.column)
        else:
            col_target_sql = "*"

        if cm.order_by not in col_map:
            raise ColumnNotFoundError(f"Cumulative metric order_by column '{cm.order_by}' does not exist in dataset.")
        for p in cm.partition_by:
            if p not in col_map:
                raise ColumnNotFoundError(f"Cumulative metric partition column '{p}' does not exist in dataset.")

        alias = cm.alias or f"cumulative_{cm.function.value}_{cm.column if cm.column else 'rows'}"
        alias_ident = escape_identifier(alias)
        available_sort_targets.add(alias)
        order_ident = escape_identifier(cm.order_by)
        part_sql = f"PARTITION BY {', '.join(escape_identifier(p) for p in cm.partition_by)} " if cm.partition_by else ""
        func_sql = cm.function.value.upper()
        select_items.append(
            f"{func_sql}({col_target_sql}) OVER ({part_sql}ORDER BY {order_ident} ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {alias_ident}"
        )

    # 3.6 Validate and Compile Quantiles (NTILE)
    for q in plan.quantiles:
        if q.column not in col_map:
            raise ColumnNotFoundError(f"Quantile column '{q.column}' does not exist in dataset.")
        if col_types[q.column] != "numeric":
            raise TypeMismatchError(
                f"Quantile segmentation requires numeric column, got '{q.column}' (dtype: {col_map[q.column].dtype})."
            )
        for p in q.partition_by:
            if p not in col_map:
                raise ColumnNotFoundError(f"Quantile partition column '{p}' does not exist in dataset.")

        alias = q.alias or f"quantile_{q.num_quantiles}_{q.column}"
        alias_ident = escape_identifier(alias)
        available_sort_targets.add(alias)
        col_ident = escape_identifier(q.column)
        part_sql = f"PARTITION BY {', '.join(escape_identifier(p) for p in q.partition_by)} " if q.partition_by else ""
        dir_sql = "DESC" if q.order_descending else "ASC"
        num_buckets = int(q.num_quantiles)
        select_items.append(
            f"NTILE({num_buckets}) OVER ({part_sql}ORDER BY {col_ident} {dir_sql}) AS {alias_ident}"
        )

    # Projection Fallback: If no group_by and no aggregations, select all columns safely
    has_window_funcs = bool(plan.rolling_windows or plan.cumulative_metrics or plan.quantiles)
    if not select_items:
        select_sql = "SELECT *"
        for c in col_map:
            available_sort_targets.add(c)
    elif has_window_funcs and not plan.group_by and not plan.aggregations and not plan.derived_metrics:
        # Row-level window projection includes source table columns
        for c in col_map:
            available_sort_targets.add(c)
        select_sql = f"SELECT *, {', '.join(select_items)}"
    else:
        distinct_kw = "DISTINCT " if (plan.group_by and not plan.aggregations and not plan.derived_metrics and not plan.correlations and not has_window_funcs) else ""
        select_sql = f"SELECT {distinct_kw}{', '.join(select_items)}"

    # 4. Construct FROM and WHERE
    sql_parts = [f"{select_sql} FROM source_table"]

    if where_clauses:
        sql_parts.append(f"WHERE {' AND '.join(where_clauses)}")

    if group_by_items:
        sql_parts.append(f"GROUP BY {', '.join(group_by_items)}")

    # 5. Validate and Compile OrderBy
    if plan.order_by:
        col_to_alias: dict[str, str] = {}
        for agg in plan.aggregations:
            alias = agg.alias or (f"{agg.function.value}_{agg.column}" if agg.column else "count_rows")
            if agg.column:
                col_to_alias[agg.column] = alias
                col_to_alias[agg.column.lower()] = alias
            col_to_alias[alias] = alias
            col_to_alias[alias.lower()] = alias

        for gb in plan.group_by:
            col_to_alias[gb.column] = gb.column
            col_to_alias[gb.column.lower()] = gb.column

        order_parts = []
        for o in plan.order_by:
            target = o.target
            resolved_target = None
            if target in available_sort_targets:
                resolved_target = target
            elif target in col_to_alias and col_to_alias[target] in available_sort_targets:
                resolved_target = col_to_alias[target]
            elif target.lower() in col_to_alias and col_to_alias[target.lower()] in available_sort_targets:
                resolved_target = col_to_alias[target.lower()]
            else:
                matched = next((ast for ast in available_sort_targets if ast.lower() == target.lower()), None)
                if matched:
                    resolved_target = matched
                else:
                    raise PlanValidationError(
                        f"Order by target '{o.target}' is not in selected projection or aliases. Available: {sorted(available_sort_targets)}."
                    )
            target_ident = escape_identifier(resolved_target)
            direction = "DESC" if o.descending else "ASC"
            order_parts.append(f"{target_ident} {direction} NULLS LAST")
        sql_parts.append(f"ORDER BY {', '.join(order_parts)}")

    # 6. Apply Clamped LIMIT
    clamped_limit = max(1, min(plan.limit, 500))
    sql_parts.append(f"LIMIT {clamped_limit}")

    final_sql = "\n".join(sql_parts)
    return CompiledQuery(sql=final_sql, parameters=parameters)
