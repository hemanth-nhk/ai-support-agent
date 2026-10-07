"""
Schema-aware LLM Planner and Deterministic Result Formatter.
Translates natural-language dataset questions into validated AnalysisPlans.
Enforces server-authoritative dataset IDs and deterministic numerical formatting.
"""
import json
import logging
import re
from typing import Any, Optional

from app.agent.llm_client import get_llm
from app.analytics.schemas import (
    AnalysisPlan,
    AnalysisResult,
    DatasetMetadata,
    AggregationClause,
    AggregationFunction,
    GroupByClause,
    OrderByClause,
    FilterClause,
    FilterOperator,
    DateTrunc,
    DerivedMetric,
    DerivedMetricOperation,
    CorrelationClause,
    PeriodComparison,
    PeriodComparisonMode,
    RollingWindowSpec,
    WindowFunction,
    CumulativeSpec,
    CumulativeFunction,
    OutlierSpec,
    QuantileSpec,
    PlanMutation,
    PlanMutationType,
    FilterDelta,
    GroupByDelta,
    MetricDelta,
    SortDelta,
)
from app.analytics.mutator import apply_plan_mutation, MutationError, describe_mutation
from app.analytics.compiler import classify_dtype

logger = logging.getLogger("app.analytics.planner")


def build_schema_context(metadata: DatasetMetadata) -> str:
    """Formats a compact, rich schema context for the LLM planner."""
    lines = [
        f"Dataset: {metadata.filename} (ID: {metadata.dataset_id})",
        f"Total Rows: {metadata.row_count}",
        "Columns:",
    ]
    for col in metadata.profile.columns:
        cat = classify_dtype(col.dtype, col.name)
        lines.append(
            f"  - name: \"{col.name}\" (type: {cat}, dtype: {col.dtype}, unique_values: {col.unique_count}, missing: {col.missing_count})"
        )
    return "\n".join(lines)


def _extract_json_from_llm_output(text: str) -> dict:
    """Extracts JSON object from LLM response, stripping markdown blocks if present."""
    clean = text.strip()
    # Match ```json ... ``` or ``` ... ```
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if md_match:
        clean = md_match.group(1).strip()
    else:
        # Match outermost { ... }
        brace_match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if brace_match:
            clean = brace_match.group(1).strip()

    return json.loads(clean)


PLANNER_SYSTEM_PROMPT = """You are a precise data analysis planner.
Your task is to translate a user's analytical question about a dataset into an AnalysisPlan JSON object.

AnalysisPlan JSON SCHEMA:
{
  "dataset_id": "<active_dataset_id>",
  "filters": [
    {"column": "string", "operator": "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "starts_with", "is_null", "is_not_null", "between", "value": string | number | array}
  ],
  "group_by": [
    {"column": "string", "date_trunc": "year" | "quarter" | "month" | "day" | null}
  ],
  "aggregations": [
    {"column": "string" | null, "function": "count" | "count_distinct" | "sum" | "avg" | "min" | "max" | "median" | "stddev" | "percentile_cont", "quantile": float | null, "alias": "string"}
  ],
  "derived_metrics": [
    {"operation": "ratio" | "percentage", "numerator": "string", "denominator": "string", "multiplier": 100.0, "alias": "string"}
  ],
  "correlations": [
    {"column_a": "string", "column_b": "string", "alias": "string"}
  ],
  "period_comparison": {
    "date_column": "string", "metric": "string", "period": "year" | "quarter" | "month" | "day", "mode": "growth_pct" | "delta", "alias": "string"
  } | null,
  "rolling_windows": [
    {"function": "avg" | "sum" | "min" | "max", "column": "string", "order_by": "string", "preceding": int, "partition_by": ["string"], "alias": "string"}
  ],
  "cumulative_metrics": [
    {"function": "sum" | "avg" | "count", "column": "string" | null, "order_by": "string", "partition_by": ["string"], "alias": "string"}
  ],
  "outlier_analysis": {
    "column": "string", "threshold": float, "partition_by": ["string"], "filter_outliers_only": true | false, "alias": "string"
  } | null,
  "quantiles": [
    {"column": "string", "num_quantiles": int, "order_descending": true | false, "partition_by": ["string"], "alias": "string"}
  ],
  "order_by": [
    {"target": "string", "descending": true | false}
  ],
  "limit": 50,
  "requires_chart": false,
  "chart_type": null | "bar" | "line" | "scatter" | "histogram"
}

CRITICAL RULES:
1. Output ONLY a valid JSON object matching the AnalysisPlan schema. Do NOT output markdown explanations or commentary.
2. ONLY reference column names that exist in the provided schema. NEVER invent column names.
3. For aggregations with 'sum', 'avg', 'min', 'max', 'median', 'stddev', the 'column' MUST be a numeric column (type: numeric). NEVER compute 'sum' or 'avg' on string/categorical columns!
4. For 'percentile_cont', provide 'quantile' as a float between 0.0 and 1.0 (e.g. 0.5 for median, 0.75 for 75th percentile).
5. For derived metrics (ratios, margins, percentages, e.g. profit margin = profit / revenue * 100), use 'derived_metrics' with 'numerator' and 'denominator'.
6. For bivariate correlation between two numeric columns, use 'correlations' with 'column_a' and 'column_b'.
7. For period comparisons (e.g. Month-over-Month revenue growth), use 'period_comparison' with 'date_column', 'metric', 'period', and 'mode' ('growth_pct' or 'delta').
8. For rolling/moving window aggregations (e.g. 7-day rolling average), use 'rolling_windows' with 'function', 'column', 'order_by', and 'preceding' (e.g. 6 for 7-day window including current row).
9. For cumulative/running totals or counts over time, use 'cumulative_metrics' with 'function' ('sum', 'avg', or 'count'), 'order_by', and optional 'column' and 'partition_by'.
10. For statistical outliers/anomalies (e.g. values > 3 stddev from mean), use 'outlier_analysis' with 'column', 'threshold' (default 3.0), and 'filter_outliers_only'.
11. For quantile segmentation (e.g. quartiles, deciles, quintiles), use 'quantiles' with 'column' and 'num_quantiles' (4 for quartiles, 5 for quintiles, 10 for deciles).
12. The group_by 'column' must be the categorical/dimension column (type: string), NOT the metric column.
13. Do NOT add filters unless the user explicitly requested filtering by a specific value (e.g., status == "delivered").
14. In "order_by", the sort field must be named "target" (matching group_by column or aggregation alias), and direction is boolean "descending" (true for highest/top/desc/best, false for lowest/asc/worst).
15. If the question asks to 'chart', 'plot', 'graph', or 'visualize', set 'requires_chart': true, and specify 'chart_type' if a specific chart is requested (use 'line' for moving windows and cumulative trends).
"""


MUTATION_PLANNER_SYSTEM_PROMPT = """You are a precise data analysis mutation planner.
You receive a user's question, the dataset schema, and the PREVIOUS AnalysisPlan JSON currently active in the session.
Your task is to determine whether the user is:
1. Refining, modifying, filtering, or resetting the previous analysis (an incremental mutation), OR
2. Asking an entirely new, unrelated analytical query (NEW_QUERY).

You must output ONLY a valid PlanMutation JSON object matching this schema:
{
  "mutation_type": "new_query" | "reset_analysis" | "add_filter" | "change_filter" | "remove_filter" | "change_group_by" | "add_group_by" | "remove_group_by" | "change_metric" | "change_sort" | "change_limit" | "add_rolling" | "add_cumulative" | "toggle_chart",
  "filter_delta": {"column": "string", "operator": "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not_in" | "contains" | "starts_with" | "is_null" | "is_not_null" | "between" | null, "value": string | number | array | null} | null,
  "group_by_delta": {"column": "string", "date_trunc": "year" | "quarter" | "month" | "day" | null} | null,
  "metric_delta": {"column": "string" | null, "function": "count" | "count_distinct" | "sum" | "avg" | "min" | "max" | "median" | "stddev" | "percentile_cont", "alias": "string" | null, "quantile": float | null} | null,
  "sort_delta": {"target": "string", "descending": true | false} | null,
  "limit_delta": int | null,
  "rolling_delta": {"function": "avg" | "sum" | "min" | "max", "column": "string", "order_by": "string", "preceding": int, "partition_by": ["string"], "alias": "string" | null} | null,
  "cumulative_delta": {"function": "sum" | "avg" | "count", "column": "string" | null, "order_by": "string", "partition_by": ["string"], "alias": "string" | null} | null,
  "chart_type": "bar" | "line" | "scatter" | "histogram" | null,
  "requires_chart": boolean | null,
  "new_plan": AnalysisPlan | null
}

RULES:
1. Output ONLY a valid JSON object matching the PlanMutation schema. Do NOT output markdown explanations or commentary.
2. If the user says "filter to...", "only for...", "where...", or specifies a category/dimension value (e.g. "Filter to Electronics", "only in North"):
   Use "add_filter" or "change_filter" with "filter_delta".
3. If the user says "remove filter", "clear filters", "reset filter", "unfilter":
   Use "remove_filter" (or "reset_analysis" if resetting entire analysis/drill-down).
4. If the user says "group by...", "by <dimension>", "break down by...", "now show by product":
   Use "change_group_by" (to switch dimension) or "add_group_by" (to add a multi-dimension breakdown).
5. If the user says "show profit instead", "switch metric to revenue", "now show the average price":
   Use "change_metric" with "metric_delta".
6. If the user says "sort by...", "order by lowest", "top...", "highest...":
   Use "change_sort" with "sort_delta".
7. If the user specifies row limits (e.g. "show top 5", "limit to 10"):
   Use "change_limit" with "limit_delta", or include "limit_delta" as a secondary field.
8. If the user asks for a rolling/moving window (e.g. "7-day rolling average"):
   Use "add_rolling" with "rolling_delta".
9. If the user asks for running/cumulative totals (e.g. "cumulative sum over time"):
   Use "add_cumulative" with "cumulative_delta".
10. If the user asks to "chart it", "show as bar chart", "visualize":
   Use "toggle_chart" with "chart_type" / "requires_chart".
11. If the user asks a completely new question unrelated to the previous analysis (e.g. "Show total users by country" after analyzing revenue by product):
    Use "new_query" and provide the complete "new_plan".
"""


def plan_mutation(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
    resolved_metric: Optional[str] = None,
) -> PlanMutation:
    """
    Classifies user analytical intent and produces a typed PlanMutation.
    If previous_plan is None, generates a fresh AnalysisPlan and wraps it in NEW_QUERY.
    If previous_plan is provided, prompts the LLM or falls back to deterministic heuristic classification.
    """
    if previous_plan is None:
        fresh_plan = generate_analysis_plan(question, metadata, resolved_metric=resolved_metric)
        return PlanMutation(
            mutation_type=PlanMutationType.NEW_QUERY,
            new_plan=fresh_plan,
        )

    schema_ctx = build_schema_context(metadata)
    prev_plan_json = previous_plan.model_dump_json(indent=2)

    user_prompt = f"""{schema_ctx}

ACTIVE ANALYSIS PLAN (PREVIOUS TURN):
{prev_plan_json}

USER FOLLOW-UP QUESTION:
"{question}"
"""
    if resolved_metric:
        user_prompt += f"\nClarified metric: '{resolved_metric}'."
    user_prompt += "\nOutput the PlanMutation JSON object:"

    llm = get_llm()
    try:
        response = llm.invoke([
            {"role": "system", "content": MUTATION_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw_output = response.content if hasattr(response, "content") else str(response)
        parsed_json = _extract_json_from_llm_output(raw_output)

        if parsed_json.get("mutation_type") == "new_query" and not parsed_json.get("new_plan"):
            parsed_json["new_plan"] = generate_analysis_plan(question, metadata, resolved_metric=resolved_metric).model_dump()

        mutation = PlanMutation.model_validate(parsed_json)
    except Exception as e:
        logger.warning(f"LLM mutation planning failed: {e}. Using deterministic heuristic fallback.")
        mutation = _fallback_heuristic_mutation(question, metadata, previous_plan, resolved_metric)

    return mutation


def generate_analysis_plan(
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
    previous_plan: Optional[AnalysisPlan] = None,
) -> AnalysisPlan:
    """
    Prompts the LLM to generate an AnalysisPlan, validates with Pydantic,
    and enforces server-authoritative dataset_id.
    If previous_plan is provided, applies an incremental PlanMutation.
    """
    if previous_plan is not None:
        mutation = plan_mutation(
            question=question,
            metadata=metadata,
            previous_plan=previous_plan,
            resolved_metric=resolved_metric,
        )
        return apply_plan_mutation(previous_plan, mutation, metadata=metadata)
    schema_ctx = build_schema_context(metadata)

    user_prompt = f"""{schema_ctx}

User Question: "{question}"
"""
    if resolved_metric:
        user_prompt += f"\nNote: The user clarified to analyze/rank by metric '{resolved_metric}'. For ranking 'best' or 'top', sort descending (descending: true)."

    user_prompt += "\nOutput the AnalysisPlan JSON:"

    str_col_names = {c.name.lower() for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"}
    num_col_names = {c.name.lower(): c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"}

    llm = get_llm()
    try:
        response = llm.invoke([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw_output = response.content if hasattr(response, "content") else str(response)
        parsed_json = _extract_json_from_llm_output(raw_output)

        # Sanitize common LLM quirks before Pydantic validation:
        # Remove vacuous filters where value equals column name
        if "filters" in parsed_json and isinstance(parsed_json["filters"], list):
            parsed_json["filters"] = [
                f for f in parsed_json["filters"]
                if isinstance(f, dict) and str(f.get("value")).lower() != str(f.get("column")).lower()
            ]

        # Fix aggregation if LLM put a string column in sum/avg/min/max or used uppercase function
        if "aggregations" in parsed_json and isinstance(parsed_json["aggregations"], list):
            for agg in parsed_json["aggregations"]:
                if isinstance(agg, dict):
                    fn = str(agg.get("function", "")).lower()
                    agg["function"] = fn
                    col_str = str(agg.get("column", "")).lower()
                    if col_str in str_col_names and fn in ("sum", "avg", "min", "max"):
                        matching_metric = resolved_metric
                        if not matching_metric:
                            for nc_low, nc in num_col_names.items():
                                if nc_low in question.lower() or nc_low in str(agg.get("alias", "")).lower():
                                    matching_metric = nc
                                    break
                        agg["column"] = matching_metric or (list(num_col_names.values())[0] if num_col_names else None)

        # Server-Authoritative Override: Force dataset_id to match server's verified dataset
        parsed_json["dataset_id"] = metadata.dataset_id
        plan = AnalysisPlan.model_validate(parsed_json)
    except Exception as e:
        logger.warning(f"LLM planner generation or validation failed: {e}. Attempting heuristic fallback.")
        fallback_json = _fallback_heuristic_plan(question, metadata, resolved_metric)
        fallback_json["dataset_id"] = metadata.dataset_id
        plan = AnalysisPlan.model_validate(fallback_json)

    # Deterministic sanity checks on AST:
    # 1. If question explicitly asks for total/sum, enforce aggregation function=sum
    if any(w in question.lower() for w in ("total", "sum", "highest total", "best", "top")):
        for agg in plan.aggregations:
            if agg.function == AggregationFunction.MAX:
                agg.function = AggregationFunction.SUM

    # 2. If question explicitly mentions a categorical column (e.g. 'products' or 'region'), align group_by
    mentioned_cat_cols = [
        c.name for c in metadata.profile.columns
        if classify_dtype(c.dtype, c.name) == "string" and (c.name.lower() in question.lower() or c.name.lower().rstrip("s") in question.lower())
    ]
    if mentioned_cat_cols and plan.group_by:
        curr_col = plan.group_by[0].column.lower()
        if curr_col not in [mc.lower() for mc in mentioned_cat_cols]:
            plan.group_by[0].column = mentioned_cat_cols[0]

    # 3. If ranking superlative is present, enforce descending order
    if any(w in question.lower() for w in ("best", "top", "highest", "most", "greatest", "leading")):
        for ob in plan.order_by:
            ob.descending = True

    # 4. If order_by targets a metric not in aggregations (or user clarified resolved_metric), ensure aggregation exists
    target_metric_name = (plan.order_by[0].target if plan.order_by else None) or resolved_metric
    if target_metric_name and target_metric_name.lower() in num_col_names and plan.group_by:
        canonical_metric = num_col_names[target_metric_name.lower()]
        is_already_agg = any(
            (agg.column and agg.column.lower() == canonical_metric.lower()) or (agg.alias and agg.alias.lower() == target_metric_name.lower())
            for agg in plan.aggregations
        )
        if not is_already_agg:
            alias_to_use = target_metric_name if not any(agg.alias == target_metric_name for agg in plan.aggregations) else f"total_{canonical_metric}"
            plan.aggregations = [
                AggregationClause(
                    column=canonical_metric,
                    function=AggregationFunction.SUM,
                    alias=alias_to_use,
                )
            ]
            if plan.order_by:
                plan.order_by[0].target = alias_to_use

    # 5. Deterministic chart intent and explicit chart type detection
    q_low = question.lower()
    chart_keywords = ("chart", "plot", "graph", "visualize", "visualization", "histogram", "scatter")
    if any(w in q_low for w in chart_keywords):
        plan.requires_chart = True
        if "scatter" in q_low:
            plan.chart_type = "scatter"
        elif "histogram" in q_low or "distribution" in q_low:
            plan.chart_type = "histogram"
        elif "line" in q_low or "trend" in q_low:
            plan.chart_type = "line"
        elif "bar" in q_low:
            plan.chart_type = "bar"

    return plan


def _fallback_heuristic_plan(
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
) -> dict:
    """
    Deterministic fallback for common analytical patterns if the LLM is offline or non-JSON.
    """
    lowered = question.lower()
    cols = {c.name.lower(): c.name for c in metadata.profile.columns}
    num_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
    str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"]

    target_metric = resolved_metric
    if not target_metric:
        for c in num_cols:
            if c.lower() in lowered:
                target_metric = c
                break
    if not target_metric and num_cols:
        target_metric = num_cols[0]

    date_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "datetime"]

    # Period comparison fallback (growth %, MoM, YoY)
    if any(k in lowered for k in ("growth", "mom", "yoy", "month-over-month", "period")) and date_cols and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        return {
            "dataset_id": metadata.dataset_id,
            "period_comparison": {
                "date_column": date_cols[0],
                "metric": m,
                "period": "month",
                "mode": "growth_pct",
                "alias": f"{m}_mom_growth",
            },
        }

    # Derived metric / ratio fallback (e.g. profit margin)
    if any(k in lowered for k in ("margin", "profit margin", "ratio", "percentage of revenue")):
        num_candidates = [c for c in num_cols if "profit" in c.lower()] or num_cols[:1]
        den_candidates = [c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower()] or num_cols[1:2]
        if num_candidates and den_candidates and num_candidates[0] != den_candidates[0]:
            group_by_list = []
            for sc in str_cols:
                if sc.lower() in lowered:
                    group_by_list.append({"column": sc})
                    break
            plan_dict = {
                "dataset_id": metadata.dataset_id,
                "derived_metrics": [{
                    "operation": "percentage",
                    "numerator": num_candidates[0],
                    "denominator": den_candidates[0],
                    "multiplier": 100.0,
                    "alias": f"{num_candidates[0]}_margin_pct",
                }],
            }
            if group_by_list:
                plan_dict["group_by"] = group_by_list
            return plan_dict

    # Correlation fallback
    if ("correlation" in lowered or "correlated" in lowered or "relationship" in lowered) and len(num_cols) >= 2:
        return {
            "dataset_id": metadata.dataset_id,
            "correlations": [{
                "column_a": num_cols[0],
                "column_b": num_cols[1],
                "alias": f"corr_{num_cols[0]}_{num_cols[1]}",
            }],
        }

    # Phase 7: Outlier / Anomaly detection fallback
    outlier_keywords = (
        "outlier", "outliers", "anomaly", "anomalies", "abnormal", "extreme",
        "z-score", "z score", "deviation from the mean", "deviations from the mean"
    )
    if any(k in lowered for k in outlier_keywords) and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        thresh = 3.0
        thresh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:stddev|standard deviation|sigma|z-?score)", lowered)
        if thresh_match:
            try:
                thresh = float(thresh_match.group(1))
            except ValueError:
                thresh = 3.0

        part_cols = []
        for sc in str_cols:
            if f"by {sc.lower()}" in lowered or f"for each {sc.lower()}" in lowered or f"per {sc.lower()}" in lowered:
                part_cols.append(sc)
                break

        filter_only = not any(k in lowered for k in ("all rows", "don't filter", "flag all", "keep all"))
        return {
            "dataset_id": metadata.dataset_id,
            "outlier_analysis": {
                "column": m,
                "threshold": thresh,
                "partition_by": part_cols,
                "filter_outliers_only": filter_only,
                "alias": m,
            },
        }

    # Phase 7: Rolling window fallback (moving average, rolling sum)
    rolling_keywords = (
        "rolling", "moving average", "moving avg", "moving sum", "moving total",
        "rolling average", "rolling avg", "rolling sum", "rolling max", "rolling min"
    )
    if any(k in lowered for k in rolling_keywords) and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        date_col = date_cols[0] if date_cols else (str_cols[0] if str_cols else [c.name for c in metadata.profile.columns][0])
        fn = "avg"
        if "sum" in lowered or "total" in lowered:
            fn = "sum"
        elif "min" in lowered:
            fn = "min"
        elif "max" in lowered:
            fn = "max"

        preceding = 6
        window_match = re.search(r"(\d+)[- ]?(?:day|period|row|month|week)", lowered)
        if window_match:
            try:
                w_val = int(window_match.group(1))
                preceding = max(1, w_val - 1)
            except ValueError:
                preceding = 6

        part_cols = []
        for sc in str_cols:
            if f"by {sc.lower()}" in lowered or f"for each {sc.lower()}" in lowered or f"per {sc.lower()}" in lowered:
                part_cols.append(sc)
                break

        return {
            "dataset_id": metadata.dataset_id,
            "rolling_windows": [{
                "function": fn,
                "column": m,
                "order_by": date_col,
                "preceding": preceding,
                "partition_by": part_cols,
                "alias": f"rolling_{fn}_{m}",
            }],
        }

    # Phase 7: Cumulative metric fallback
    cum_keywords = ("cumulative", "running total", "running sum", "running count", "running avg", "running average")
    if any(k in lowered for k in cum_keywords) and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        date_col = date_cols[0] if date_cols else (str_cols[0] if str_cols else [c.name for c in metadata.profile.columns][0])
        fn = "sum"
        if "count" in lowered:
            fn = "count"
        elif "avg" in lowered or "average" in lowered:
            fn = "avg"

        part_cols = []
        for sc in str_cols:
            if f"by {sc.lower()}" in lowered or f"for each {sc.lower()}" in lowered or f"per {sc.lower()}" in lowered:
                part_cols.append(sc)
                break

        return {
            "dataset_id": metadata.dataset_id,
            "cumulative_metrics": [{
                "function": fn,
                "column": m if fn != "count" else None,
                "order_by": date_col,
                "partition_by": part_cols,
                "alias": f"cumulative_{fn}_{m if fn != 'count' else 'rows'}",
            }],
        }

    # Phase 7: Quantile segmentation fallback
    quantile_keywords = (
        "quartile", "quartiles", "decile", "deciles", "quintile", "quintiles",
        "quantile", "quantiles", "ntile", "spending tier", "spending tiers"
    )
    if any(k in lowered for k in quantile_keywords) and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        num_q = 4
        if "decile" in lowered:
            num_q = 10
        elif "quintile" in lowered:
            num_q = 5
        else:
            q_match = re.search(r"(\d+)\s*(?:quantile|bucket|tier|segment)", lowered)
            if q_match:
                try:
                    num_q = max(2, min(int(q_match.group(1)), 100))
                except ValueError:
                    num_q = 4

        part_cols = []
        for sc in str_cols:
            if f"by {sc.lower()}" in lowered or f"for each {sc.lower()}" in lowered or f"per {sc.lower()}" in lowered:
                part_cols.append(sc)
                break

        return {
            "dataset_id": metadata.dataset_id,
            "quantiles": [{
                "column": m,
                "num_quantiles": num_q,
                "partition_by": part_cols,
                "alias": f"quantile_{num_q}_{m}",
            }],
        }

    # Statistical fallback (median / stddev)
    if "median" in lowered and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        return {
            "dataset_id": metadata.dataset_id,
            "aggregations": [{"column": m, "function": "median", "alias": f"median_{m}"}],
        }
    if ("stddev" in lowered or "standard deviation" in lowered) and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        return {
            "dataset_id": metadata.dataset_id,
            "aggregations": [{"column": m, "function": "stddev", "alias": f"stddev_{m}"}],
        }

    # Ranking query: e.g. "top products", "which region had highest revenue"
    if any(k in lowered for k in ("top", "highest", "best", "most", "lowest", "worst")) and str_cols and target_metric:
        group_col = str_cols[0]
        for c in str_cols:
            if c.lower() in lowered:
                group_col = c
                break
        desc = not any(k in lowered for k in ("lowest", "worst", "least"))
        top_match = re.search(r"top\s+(\d+)", lowered)
        limit_val = int(top_match.group(1)) if top_match else 10
        return {
            "dataset_id": metadata.dataset_id,
            "group_by": [{"column": group_col}],
            "aggregations": [{"column": target_metric, "function": "sum", "alias": f"sum_{target_metric}"}],
            "order_by": [{"target": f"sum_{target_metric}", "descending": desc}],
            "limit": limit_val,
        }

    # Grouped query by dimension (e.g. "salary by department", "revenue by region")
    group_col = None
    for sc in str_cols:
        if sc.lower() in lowered or f"by {sc.lower()}" in lowered:
            group_col = sc
            break
    if group_col and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        return {
            "dataset_id": metadata.dataset_id,
            "group_by": [{"column": group_col}],
            "aggregations": [{"column": m, "function": "sum", "alias": f"sum_{m}"}],
            "limit": 50,
        }

    # Total / Average query
    if "average" in lowered or "avg" in lowered:
        metric = target_metric
        for c in num_cols:
            if c.lower() in lowered:
                metric = c
                break
        if metric:
            return {
                "dataset_id": metadata.dataset_id,
                "aggregations": [{"column": metric, "function": "avg", "alias": f"avg_{metric}"}],
            }

    if "total" in lowered or "sum" in lowered or "revenue" in lowered or "profit" in lowered:
        metric = target_metric
        for c in num_cols:
            if c.lower() in lowered:
                metric = c
                break
        if metric:
            return {
                "dataset_id": metadata.dataset_id,
                "aggregations": [{"column": metric, "function": "sum", "alias": f"sum_{metric}"}],
            }

    # General row count
    return {
        "dataset_id": metadata.dataset_id,
        "aggregations": [{"column": None, "function": "count", "alias": "row_count"}],
    }


def _fallback_heuristic_mutation(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: AnalysisPlan,
    resolved_metric: Optional[str] = None,
) -> PlanMutation:
    """
    Deterministic fallback for classifying follow-up analytical intent and extracting
    PlanMutation when the LLM is unavailable or produces invalid JSON.
    """
    q_low = question.lower().strip()
    str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"]
    num_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
    date_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "datetime"]

    # 1. RESET_ANALYSIS / REMOVE_FILTER
    if any(k in q_low for k in ("clear filter", "clear filters", "reset filter", "reset filters", "reset analysis", "clear all filters", "start over")):
        top_match = re.search(r"(?:top|limit|first)\s+(\d+)", q_low)
        limit_val = int(top_match.group(1)) if top_match else None

        target_dim = None
        for sc in str_cols:
            if sc.lower() in q_low or sc.lower().rstrip("s") in q_low:
                target_dim = sc
                break

        if target_dim or limit_val or "reset" in q_low or "clear filters and" in q_low:
            gb_delta = GroupByDelta(column=target_dim) if target_dim else None
            return PlanMutation(
                mutation_type=PlanMutationType.RESET_ANALYSIS,
                group_by_delta=gb_delta,
                limit_delta=limit_val,
            )
        col_to_rm = None
        for col in metadata.profile.columns:
            if col.name.lower() in q_low:
                col_to_rm = col.name
                break
        if col_to_rm:
            return PlanMutation(
                mutation_type=PlanMutationType.REMOVE_FILTER,
                filter_delta=FilterDelta(column=col_to_rm),
            )
        return PlanMutation(mutation_type=PlanMutationType.REMOVE_FILTER)

    # 2. ADD_ROLLING (e.g. "show that over time as a 7-day rolling average")
    rolling_keywords = (
        "rolling", "moving average", "moving avg", "moving sum", "rolling average", "rolling avg"
    )
    if any(k in q_low for k in rolling_keywords):
        metric = resolved_metric
        if not metric and previous_plan.aggregations and previous_plan.aggregations[0].column:
            metric = previous_plan.aggregations[0].column
        if not metric and num_cols:
            metric = num_cols[0]

        order_col = date_cols[0] if date_cols else (str_cols[0] if str_cols else metadata.profile.columns[0].name)
        preceding = 6
        window_match = re.search(r"(\d+)[- ]?(?:day|period|row|month|week)", q_low)
        if window_match:
            try:
                preceding = max(1, int(window_match.group(1)) - 1)
            except ValueError:
                preceding = 6

        fn = WindowFunction.AVG
        if "sum" in q_low or "total" in q_low:
            fn = WindowFunction.SUM
        elif "min" in q_low:
            fn = WindowFunction.MIN
        elif "max" in q_low:
            fn = WindowFunction.MAX

        return PlanMutation(
            mutation_type=PlanMutationType.ADD_ROLLING,
            rolling_delta=RollingWindowSpec(
                function=fn,
                column=metric or (num_cols[0] if num_cols else "value"),
                order_by=order_col,
                preceding=preceding,
            ),
        )

    # 3. ADD_CUMULATIVE (e.g. "cumulative total over time")
    cum_keywords = ("cumulative", "running total", "running sum", "running avg", "running count")
    if any(k in q_low for k in cum_keywords):
        metric = resolved_metric or (previous_plan.aggregations[0].column if previous_plan.aggregations else None) or (num_cols[0] if num_cols else None)
        order_col = date_cols[0] if date_cols else (str_cols[0] if str_cols else metadata.profile.columns[0].name)
        cfn = CumulativeFunction.SUM
        if "count" in q_low:
            cfn = CumulativeFunction.COUNT
            metric = None
        elif "avg" in q_low or "average" in q_low:
            cfn = CumulativeFunction.AVG

        return PlanMutation(
            mutation_type=PlanMutationType.ADD_CUMULATIVE,
            cumulative_delta=CumulativeSpec(
                function=cfn,
                column=metric,
                order_by=order_col,
            ),
        )

    # 4. TOGGLE_CHART / VISUALIZATION
    chart_keywords = ("chart", "plot", "graph", "visualize", "visualization")
    if any(k in q_low for k in chart_keywords):
        if any(w in q_low for w in ("hide", "remove", "disable", "no chart")):
            return PlanMutation(mutation_type=PlanMutationType.TOGGLE_CHART, requires_chart=False)
        ctype = "bar"
        if "line" in q_low or "trend" in q_low:
            ctype = "line"
        elif "scatter" in q_low:
            ctype = "scatter"
        elif "histogram" in q_low:
            ctype = "histogram"
        return PlanMutation(
            mutation_type=PlanMutationType.TOGGLE_CHART,
            chart_type=ctype,
            requires_chart=True,
        )

    # 5. CHANGE_SORT
    if any(k in q_low for k in ("sort by", "order by", "highest first", "lowest first", "ascending", "descending")):
        desc = not any(k in q_low for k in ("lowest", "least", "asc", "ascending"))
        target = None
        for col in metadata.profile.columns:
            if col.name.lower() in q_low:
                target = col.name
                break
        if not target and previous_plan.aggregations:
            target = previous_plan.aggregations[0].alias or previous_plan.aggregations[0].column
        if not target and previous_plan.group_by:
            target = previous_plan.group_by[0].column
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_SORT,
            sort_delta=SortDelta(target=target or "total", descending=desc),
        )

    # 6. CHANGE_LIMIT (standalone: e.g. "show top 5", "limit to 10")
    limit_match = re.search(r"^(?:show\s+)?(?:top|limit|first)\s+(\d+)$", q_low)
    if limit_match:
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_LIMIT,
            limit_delta=int(limit_match.group(1)),
        )

    # 7. CHANGE_METRIC (e.g. "show profit instead", "switch to revenue", "now average price")
    if any(k in q_low for k in ("instead", "switch metric", "change metric", "switch to", "now show")):
        target_num = None
        for nc in num_cols:
            if nc.lower() in q_low:
                target_num = nc
                break
        if target_num:
            fn = AggregationFunction.AVG if any(w in q_low for w in ("avg", "average", "mean")) else AggregationFunction.SUM
            return PlanMutation(
                mutation_type=PlanMutationType.CHANGE_METRIC,
                metric_delta=MetricDelta(
                    column=target_num,
                    function=fn,
                    alias=f"{fn.value}_{target_num}",
                ),
            )

    # 8. GROUP_BY MUTATIONS (CHANGE_GROUP_BY, ADD_GROUP_BY, REMOVE_GROUP_BY)
    if any(k in q_low for k in ("also by", "add group", "break down further by")):
        for sc in str_cols:
            if sc.lower() in q_low:
                return PlanMutation(
                    mutation_type=PlanMutationType.ADD_GROUP_BY,
                    group_by_delta=GroupByDelta(column=sc),
                )
    if any(k in q_low for k in ("group by", "by ", "per ", "break down by", "now show by")):
        for sc in str_cols:
            if sc.lower() in q_low or sc.lower().rstrip("s") in q_low:
                return PlanMutation(
                    mutation_type=PlanMutationType.CHANGE_GROUP_BY,
                    group_by_delta=GroupByDelta(column=sc),
                )

    # 9. ADD_FILTER / CHANGE_FILTER (e.g. "Filter to Electronics", "only in North", "where status is delivered")
    filter_trigger = any(k in q_low for k in ("filter", "only", "where", "for "))
    if filter_trigger:
        target_col = None
        filter_val = None

        for sc in str_cols:
            if sc.lower() in q_low:
                target_col = sc
                break

        v_match = re.search(r"(?:filter\s+to|filter\s+by|only|where)\s+([A-Za-z0-9_\s-]+)", question, re.IGNORECASE)
        if v_match:
            raw_v = v_match.group(1).strip()
            raw_v = re.sub(r"^(?:in|the|for|status|category|region|country)\s+", "", raw_v, flags=re.IGNORECASE).strip()
            if raw_v:
                filter_val = raw_v

        if not target_col:
            if previous_plan.group_by:
                target_col = previous_plan.group_by[0].column
            elif str_cols:
                target_col = str_cols[0]

        if target_col and filter_val:
            existing_cols = {f.column for f in previous_plan.filters}
            m_type = PlanMutationType.CHANGE_FILTER if target_col in existing_cols else PlanMutationType.ADD_FILTER
            return PlanMutation(
                mutation_type=m_type,
                filter_delta=FilterDelta(
                    column=target_col,
                    operator=FilterOperator.EQ,
                    value=filter_val,
                ),
            )

    # 10. Default: NEW_QUERY
    fresh = _fallback_heuristic_plan(question, metadata, resolved_metric)
    fresh["dataset_id"] = metadata.dataset_id
    new_plan_obj = AnalysisPlan.model_validate(fresh)
    return PlanMutation(
        mutation_type=PlanMutationType.NEW_QUERY,
        new_plan=new_plan_obj,
    )


def format_deterministic_result(
    result: AnalysisResult,
    question: str,
    previous_plan: Optional[AnalysisPlan] = None,
) -> str:
    """
    Synthesizes a verifiable, deterministic factual statement from DuckDB results.
    Guarantees numerical correctness independently of LLM whims.
    If previous_plan is provided, prefixes the grounded finding with conversational context.
    """
    mutation_prefix = ""
    if previous_plan is not None:
        # 1. RESET_ANALYSIS: Filters cleared
        if previous_plan.filters and not result.plan.filters:
            if not result.plan.group_by and previous_plan.group_by:
                mutation_prefix = "Cleared all active filters and groupings. "
            else:
                mutation_prefix = "Cleared all active filters. "
        elif not result.plan.filters and not result.plan.group_by and (previous_plan.filters or previous_plan.group_by):
            mutation_prefix = "Cleared all active filters and groupings. "

        # 2. ADD_FILTER / CHANGE_FILTER: New or modified filters
        elif result.plan.filters:
            prev_filter_set = {
                (f.column, f.operator.value, str(f.value)) for f in previous_plan.filters
            }
            curr_filter_set = {
                (f.column, f.operator.value, str(f.value)) for f in result.plan.filters
            }
            if curr_filter_set != prev_filter_set or len(result.plan.filters) > len(previous_plan.filters):
                filter_descs = []
                for f in result.plan.filters:
                    op_symbol = "=" if f.operator == FilterOperator.EQ else f.operator.value
                    filter_descs.append(f"{f.column} {op_symbol} '{f.value}'")
                mutation_prefix = f"Filtered by {' and '.join(filter_descs)}. "

        # 3. CHANGE_METRIC: Aggregation target changed
        if not mutation_prefix and result.plan.aggregations and previous_plan.aggregations:
            prev_col = previous_plan.aggregations[0].column
            curr_col = result.plan.aggregations[0].column
            prev_fn = previous_plan.aggregations[0].function
            curr_fn = result.plan.aggregations[0].function
            if prev_col != curr_col or prev_fn != curr_fn:
                target_metric = curr_col or curr_fn.value
                mutation_prefix = f"Switched metric to {target_metric}. "

        # 4. CHANGE_GROUP_BY: Grouping dimension changed
        if not mutation_prefix and result.plan.group_by:
            prev_gb = [g.column for g in previous_plan.group_by]
            curr_gb = [g.column for g in result.plan.group_by]
            if prev_gb != curr_gb:
                dim_name = curr_gb[0]
                filter_ctx = ""
                if result.plan.filters:
                    filter_vals = [f"'{f.value}'" for f in result.plan.filters if f.value]
                    if filter_vals:
                        filter_col = result.plan.filters[0].column
                        filter_ctx = f" in {filter_col} {', '.join(filter_vals)}"
                mutation_prefix = f"Grouped by {dim_name}{filter_ctx}. "

        # 5. CHANGE_SORT: Order target or direction changed
        if not mutation_prefix and result.plan.order_by and previous_plan.order_by:
            prev_ob = previous_plan.order_by[0]
            curr_ob = result.plan.order_by[0]
            if prev_ob.target != curr_ob.target or prev_ob.descending != curr_ob.descending:
                dir_label = "descending" if curr_ob.descending else "ascending"
                mutation_prefix = f"Sorted by {curr_ob.target} ({dir_label}). "

    if result.row_count == 0:
        if result.plan.outlier_analysis:
            oa = result.plan.outlier_analysis
            return f"{mutation_prefix}No statistical outliers detected in '{oa.column}' at threshold {oa.threshold} standard deviations across the dataset."
        return f"{mutation_prefix}The query completed successfully, but no matching records were found in the dataset."

    rows = result.rows
    cols = result.columns

    # Phase 7: Outlier Analysis result
    if result.plan.outlier_analysis:
        oa = result.plan.outlier_analysis
        prefix = oa.alias or oa.column
        mean_col = f"mean_{prefix}"
        stddev_col = f"stddev_{prefix}"
        z_col = f"z_score_{prefix}"

        # Zero variance check
        first_row = rows[0] if rows else {}
        stddev_val = first_row.get(stddev_col)
        if stddev_val is None or stddev_val == 0:
            return f"Unable to calculate outliers: '{oa.column}' has zero variance or insufficient observations across the dataset."

        outlier_count = sum(1 for r in rows if r.get("is_outlier") is True)
        if outlier_count > 0:
            top_row = rows[0]
            top_val = top_row.get(oa.column)
            top_z = top_row.get(z_col)
            mean_val = top_row.get(mean_col)
            z_str = f"{top_z:+.2f}" if isinstance(top_z, (int, float)) else str(top_z)
            return (
                f"Identified {outlier_count} statistical outlier{'s' if outlier_count != 1 else ''} in '{oa.column}' "
                f"exceeding {oa.threshold} standard deviations (most extreme: value {top_val} with z-score {z_str} vs mean {mean_val})."
            )
        else:
            return f"No statistical outliers detected in '{oa.column}' at threshold {oa.threshold} standard deviations across {result.row_count} observed records."

    # Phase 7: Rolling Window result
    if result.plan.rolling_windows and not result.plan.group_by:
        rw = result.plan.rolling_windows[0]
        alias = rw.alias or f"rolling_{rw.function.value}_{rw.column}"
        w_size = rw.preceding + 1
        valid_vals = [
            (r.get(alias), r.get(rw.order_by))
            for r in rows
            if isinstance(r.get(alias), (int, float))
        ]
        if valid_vals:
            peak_val, peak_order = max(valid_vals, key=lambda x: x[0])
            latest_val, latest_order = valid_vals[-1]
            return (
                f"Across {len(rows)} observed intervals, the {w_size}-period rolling {rw.function.value} of '{rw.column}' "
                f"peaked at {peak_val:.2f} (at {peak_order}), with the latest value at {latest_val:.2f}."
            )
        return f"Calculated {w_size}-period rolling {rw.function.value} for '{rw.column}' across {len(rows)} records."

    # Phase 7: Cumulative Metric result
    if result.plan.cumulative_metrics and not result.plan.group_by:
        cm = result.plan.cumulative_metrics[0]
        alias = cm.alias or f"cumulative_{cm.function.value}_{cm.column if cm.column else 'rows'}"
        target_name = cm.column or "records"
        latest_row = rows[-1]
        final_val = latest_row.get(alias)
        end_order = latest_row.get(cm.order_by)
        if final_val is not None:
            val_str = f"{final_val:,.2f}" if isinstance(final_val, float) else f"{final_val:,}" if isinstance(final_val, int) else str(final_val)
            return f"Across {len(rows)} observed intervals up to {end_order}, cumulative {cm.function.value} of {target_name} reached {val_str}."
        return f"Calculated cumulative {cm.function.value} of {target_name} across {len(rows)} records."

    # Phase 7: Quantile Segmentation result
    if result.plan.quantiles and not result.plan.group_by:
        q = result.plan.quantiles[0]
        bucket_label = "quartiles" if q.num_quantiles == 4 else ("quintiles" if q.num_quantiles == 5 else ("deciles" if q.num_quantiles == 10 else f"{q.num_quantiles} quantile tiers"))
        vals = [r.get(q.column) for r in rows if isinstance(r.get(q.column), (int, float))]
        if vals:
            return f"Segmented {len(rows)} records into {q.num_quantiles} {bucket_label} based on '{q.column}' (ranging from {min(vals)} to {max(vals)})."
        return f"Segmented {len(rows)} records into {q.num_quantiles} {bucket_label} based on '{q.column}'."

    # Period comparison result
    if result.plan.period_comparison:
        pc = result.plan.period_comparison
        alias = pc.alias or (f"{pc.metric}_growth_pct" if pc.mode == PeriodComparisonMode.GROWTH_PCT else f"{pc.metric}_delta")
        latest = rows[-1]
        p_val = latest.get("period")
        c_val = latest.get("current_val")
        comp_val = latest.get(alias)
        if comp_val is None:
            comp_str = "baseline / first observed period"
        elif pc.mode == PeriodComparisonMode.GROWTH_PCT:
            comp_str = f"{comp_val:+.1f}% vs previous observed period"
        else:
            comp_str = f"{comp_val:+g} vs previous observed period"
        return f"Across {len(rows)} observed periods, the latest period ({p_val}) recorded {pc.metric} of {c_val} ({alias}: {comp_str})."

    # Correlation result
    if result.plan.correlations and not result.plan.group_by:
        c = result.plan.correlations[0]
        alias = c.alias or f"corr_{c.column_a}_{c.column_b}"
        val = rows[0].get(alias) if rows else None
        if val is None:
            return f"The correlation between '{c.column_a}' and '{c.column_b}' is undefined (constant variance or insufficient observations)."
        desc = "strong positive" if val > 0.7 else ("moderate positive" if val > 0.3 else ("strong negative" if val < -0.7 else ("moderate negative" if val < -0.3 else "weak / near zero")))
        return f"The Pearson correlation between '{c.column_a}' and '{c.column_b}' is {val:.3f} ({desc} relationship)."

    # Derived metric standalone result
    if result.plan.derived_metrics and not result.plan.group_by:
        facts = []
        for dm in result.plan.derived_metrics:
            val = rows[0].get(dm.alias)
            if val is None:
                facts.append(f"{dm.alias}: unavailable (zero denominator)")
            elif dm.operation == DerivedMetricOperation.PERCENTAGE or "pct" in dm.alias:
                facts.append(f"{dm.alias}: {val:.2f}%")
            else:
                facts.append(f"{dm.alias}: {val:.4f}")
        return "Based on the dataset, " + ", ".join(facts) + "."

    # Single scalar aggregation (e.g. total revenue or avg score)
    if result.row_count == 1 and not result.plan.group_by:
        row = rows[0]
        facts = [f"{k}: {v if v is not None else 'null'}" for k, v in row.items()]
        return f"Based on the dataset, " + ", ".join(facts) + "."

    # Grouped / Ranked query (e.g. top regions by revenue)
    if result.plan.group_by:
        top_row = rows[0]
        group_keys = [gb.column for gb in result.plan.group_by]
        metric_keys = [c for c in cols if c not in group_keys]

        lead_entity = top_row.get(group_keys[0]) if group_keys else "First item"
        lead_metrics = [f"{m} is {top_row.get(m)}" for m in metric_keys]

        if mutation_prefix:
            active_filter_ctx = ""
            if result.plan.filters:
                filter_vals = [f"'{f.value}'" for f in result.plan.filters if f.value]
                if filter_vals:
                    active_filter_ctx = f" in {', '.join(filter_vals)}"

            if len(rows) > 1:
                statement = (
                    f"{mutation_prefix}Across {len(rows)} {group_keys[0]}s, "
                    f"leading {group_keys[0]}{active_filter_ctx} is {lead_entity} with " + ", ".join(lead_metrics) + "."
                )
                second_row = rows[1]
                sec_entity = second_row.get(group_keys[0]) if group_keys else "Second item"
                sec_metrics = [f"{m} is {second_row.get(m)}" for m in metric_keys]
                statement += f" Followed by {sec_entity} with " + ", ".join(sec_metrics) + "."
            else:
                statement = (
                    f"{mutation_prefix}Top {group_keys[0]}{active_filter_ctx} is {lead_entity} with " + ", ".join(lead_metrics) + "."
                )
            return statement

        statement = f"The leading {group_keys[0] if group_keys else 'group'} is {lead_entity} with " + ", ".join(lead_metrics) + "."

        if len(rows) > 1:
            second_row = rows[1]
            sec_entity = second_row.get(group_keys[0]) if group_keys else "Second item"
            sec_metrics = [f"{m} is {second_row.get(m)}" for m in metric_keys]
            statement += f" Followed by {sec_entity} with " + ", ".join(sec_metrics) + "."

        return statement

    # Table listing
    if mutation_prefix:
        return f"{mutation_prefix}Displaying full dataset across {result.row_count} records."
    return f"The query returned {result.row_count} rows across {len(cols)} columns."
