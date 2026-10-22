"""
Schema-aware LLM Planner and Deterministic Result Formatter.
Translates natural-language dataset questions into validated AnalysisPlans.
Enforces server-authoritative dataset IDs and deterministic numerical formatting.
"""
import json
import logging
import re
from typing import Any, Optional, Tuple

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
    PivotSpec,
    PivotDelta,
)
from app.analytics.mutator import apply_plan_mutation, MutationError, describe_mutation
from app.analytics.compiler import classify_dtype, PlanValidationError, ColumnNotFoundError

logger = logging.getLogger("app.analytics.planner")

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "twenty-five": 25, "fifty": 50,
}
WORD_NUM_PATTERN = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty)"


def _parse_int_or_word(val: Optional[str]) -> Optional[int]:
    """Parses a digit string or spelled-out number word into an integer."""
    if not val:
        return None
    val_low = val.lower().strip()
    if val_low in WORD_TO_NUM:
        return WORD_TO_NUM[val_low]
    if val_low.isdigit():
        return int(val_low)
    return None


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
10. For statistical outliers/anomalies (e.g. values > 3 stddev from mean), use 'outlier_analysis' with 'column', 'threshold' (default 3.0), and 'filter_outliers_only': true. Do NOT put the outlier threshold in 'filters'. Only set 'partition_by' if the user explicitly asks to partition/group outliers by a specific column; otherwise use [].
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
   Use "toggle_chart" with "chart_type" / "requires_chart": true.
   If the user asks to "show as a table", "switch back to a table", "display as a table", or return to table view:
   Use "toggle_chart" with "requires_chart": false, "chart_type": null.
11. If the user asks a completely new question unrelated to the previous analysis (e.g. "Show total users by country" after analyzing revenue by product):
    Use "new_query" and provide the complete "new_plan".
"""


from app.analytics.chart_style import format_clean_metric_label
_format_clean_metric_label = format_clean_metric_label



def _is_meaningful_filter_value(val: Any) -> bool:
    """
    Determines if a filter value represents a meaningful user-facing context
    (e.g., 'North', 'Electronics') rather than an empty, sentinel, or zero value.
    """
    if val is None:
        return False
    val_str = str(val).strip().strip("'\"")
    if val_str in ("", "0", "0.0", "none", "null", "all", "*", "any"):
        return False
    try:
        if float(val_str) == 0.0:
            return False
    except (ValueError, TypeError):
        pass
    return True


def _clean_filter_repr(f: Any) -> Optional[str]:
    """
    Returns a quoted clean string representation of a filter value if meaningful, else None.
    """
    if not (f.operator in (FilterOperator.EQ, FilterOperator.IN) and _is_meaningful_filter_value(f.value)):
        return None
    val_str = str(f.value).strip().strip("'\"")
    return f"'{val_str}'"


_EXPLICIT_CLEAR_FILTER_PATTERNS = (
    r"\b(?:ignor(?:e|ing)|discard(?:ing)?|disregard(?:ing)?)\s+(?:all\s+|any\s+|the\s+)?(?:previous\s+|prior\s+|earlier\s+|existing\s+)?filters?\b",
    r"\bwithout\s+(?:all\s+|any\s+|the\s+)?(?:previous\s+|prior\s+|earlier\s+|existing\s+)?filters?\b",
    r"\b(?:clear(?:ing)?|remov(?:e|ing)|reset(?:ting)?)\s+(?:all\s+|any\s+|the\s+)?(?:previous\s+|prior\s+|earlier\s+|existing\s+)?filters?\b",
    r"\b(?:across|in|for|over)\s+(?:the\s+)?(?:entire|whole)\s+(?:dataset|data|table|records)\b",
    r"\bacross\s+all\s+(?:records|data)\b",
    r"\ball\s+records\s+across\s+the\s+dataset\b",
    r"\b(?:with\s+)?no\s+filters?\b",
)


def _has_explicit_clear_filter_directive(text: str) -> bool:
    """
    Detects whether the question contains an explicit directive to clear, ignore,
    or reset previous/existing filters, or evaluate across the whole/entire dataset.
    e.g. 'across the entire dataset', 'ignore previous filters', 'without filters',
         'clear all filters', 'reset filters', etc.
    """
    if not text:
        return False
    t_low = text.lower()
    return any(re.search(pat, t_low) for pat in _EXPLICIT_CLEAR_FILTER_PATTERNS)


def _detect_pure_chart_followup(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> Optional[Tuple[bool, Optional[str]]]:
    """
    Detects if a follow-up query is purely requesting to visualize / chart / toggle the chart
    for the current analytical result, without redefining metrics, dimensions, filters, or Top-N limits.
    Returns (requires_chart, chart_type) if matched, or None otherwise.
    """
    q_low = question.lower().strip()
    chart_keywords = ("chart", "plot", "graph", "visualize", "visualization")
    if not any(k in q_low for k in chart_keywords):
        return None

    if _has_explicit_clear_filter_directive(q_low):
        return None

    # Check for disabling/hiding chart
    if any(w in q_low for w in ("hide chart", "remove chart", "disable chart", "no chart", "close chart", "hide the chart")):
        return (False, None)

    # Check for query redirection or conflicting mutation keywords
    conflict_keywords = (
        "instead", "switch to", "switch metric", "change metric",
        "filter to", "filter by", "where ", "only in",
        "group by", "also by", "break down by",
    )
    if any(k in q_low for k in conflict_keywords):
        return None

    # If user specifies a top/bottom limit change, let standard limit mutation handle it
    top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
    if top_match and previous_plan and previous_plan.limit and int(top_match.group(1)) != previous_plan.limit:
        return None

    # Check if a column not already present in the active plan is introduced
    prev_cols = set()
    if previous_plan:
        for gb in previous_plan.group_by:
            prev_cols.add(gb.column.lower())
        for agg in previous_plan.aggregations:
            if agg.column:
                prev_cols.add(agg.column.lower())
                for tok in re.findall(r"\b\w+\b", agg.column.lower()):
                    if len(tok) >= 3:
                        prev_cols.add(tok)
        for ob in previous_plan.order_by:
            if ob.target:
                prev_cols.add(ob.target.lower())

    all_cols = [c.name.lower() for c in metadata.profile.columns]
    for c in all_cols:
        if len(c) >= 3 and re.search(rf"\b{re.escape(c)}\b", q_low):
            if c not in prev_cols:
                return None

    ctype = "bar"
    if "line" in q_low or "trend" in q_low:
        ctype = "line"
    elif "scatter" in q_low:
        ctype = "scatter"
    elif "histogram" in q_low or "distribution" in q_low:
        ctype = "histogram"
    elif "heatmap" in q_low:
        ctype = "heatmap"
    elif "bar" in q_low:
        ctype = "bar"

    return (True, ctype)


def _detect_pure_table_followup(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> bool:
    """
    Detects if a follow-up query is purely requesting to present / display / switch
    the current analytical result back to a table (tabular view), without redefining
    metrics, dimensions, filters, or Top-N limits.
    Returns True if matched, False otherwise.
    """
    if previous_plan is None:
        return False

    if _has_explicit_clear_filter_directive(question):
        return False

    q_low = question.lower().strip()
    q_clean = q_low.rstrip("?.!")

    # Explicit new query indicators (e.g. "what is...", "calculate...", "how many...")
    if any(q_clean.startswith(p) for p in ("what is", "what are", "how many", "calculate", "find", "tell me what")):
        return False

    # Exclude pivot / crosstab queries
    if any(k in q_low for k in ("pivot", "crosstab", "cross-tab")):
        return False

    # Check for table presentation patterns
    table_patterns = (
        r"\b(?:as|to|in|into)\s+(?:a\s+)?(?:table|tabular|grid)\b",
        r"\b(?:switch|change|convert|toggle|return|go)\s+(?:back\s+)?to\s+(?:a\s+)?(?:table|tabular|grid)\b",
        r"\b(?:table|tabular|grid)\s+(?:view|format|mode|presentation)\b",
        r"\b(?:show|display|view|present|render|see|give)\s+(?:me\s+)?(?:this|the\s+results?|it|results?|data)?\s*(?:as|in|into)?\s*(?:a\s+)?(?:table|tabular|grid)\b",
        r"\b(?:show|display|view|present|render)\s+(?:me\s+)?(?:this|the\s+results?|it|results?|data)?\s*(?:table|tabular|grid)\b",
        r"\bback\s+to\s+(?:a\s+)?(?:table|tabular|grid)\b",
        r"^(?:just\s+)?(?:the\s+)?(?:table|tabular|grid)$",
    )
    if not any(re.search(p, q_clean) for p in table_patterns):
        return False

    # Check for query redirection or conflicting mutation keywords
    conflict_keywords = (
        "instead", "switch metric", "change metric",
        "filter to", "filter by", "where ", "only in",
        "group by", "also by", "break down by",
    )
    if any(k in q_low for k in conflict_keywords):
        return False

    # "switch to" is allowed ONLY when switching to table/tabular/grid, not to a new metric or dimension
    if "switch to" in q_low and not re.search(r"\bswitch\s+to\s+(?:a\s+)?(?:table|tabular|grid)\b", q_low):
        return False

    # If user specifies a top/bottom limit change, let standard limit mutation handle it
    top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
    if top_match and previous_plan and previous_plan.limit and int(top_match.group(1)) != previous_plan.limit:
        return False

    # Check if a column not already present in the active plan is introduced
    prev_cols = set()
    if previous_plan:
        for gb in previous_plan.group_by:
            prev_cols.add(gb.column.lower())
        for agg in previous_plan.aggregations:
            if agg.column:
                prev_cols.add(agg.column.lower())
                for tok in re.findall(r"\b\w+\b", agg.column.lower()):
                    if len(tok) >= 3:
                        prev_cols.add(tok)
        for ob in previous_plan.order_by:
            if ob.target:
                prev_cols.add(ob.target.lower())

    all_cols = [c.name.lower() for c in metadata.profile.columns]
    for c in all_cols:
        if len(c) >= 3 and re.search(rf"\b{re.escape(c)}\b", q_low):
            if c not in prev_cols:
                return False

    return True


def _detect_pure_sort_followup(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> Optional[SortDelta]:
    """
    Detects if a follow-up query is purely requesting to sort or change the ordering
    direction/target of the active analysis result (e.g. "Sort descending", "Sort ascending",
    "Sort desc", "Sort asc", "Reverse the order", "Lowest first", "Highest first",
    "Keep the same five products but sort them from smallest to largest").
    """
    if previous_plan is None:
        return None

    q_low = question.lower().strip()
    q_clean = q_low.rstrip("?.!")

    # Explicit new query indicators
    if any(q_clean.startswith(p) for p in ("what is", "what are", "how many", "calculate", "find", "tell me what")):
        return None

    # Exclude filter / group by / pivot directives
    if any(k in q_low for k in ("filter", "where ", "only in", "group by", "break down", "pivot", "crosstab")):
        return None

    sort_patterns = (
        r"^(?:please\s+)?(?:sort|order)\s+(?:in\s+|by\s+)?(?:descending|desc|ascending|asc)(?:\s+order)?$",
        r"^(?:please\s+)?(?:sort|order)\s+(?:by\s+)?(?:highest|lowest|least|most)\s*(?:first)?$",
        r"^(?:reverse|flip|invert)\s+(?:the\s+)?(?:order|sort|sorting|results?)$",
        r"^(?:highest|lowest|least|most)\s+first$",
        r"^(?:descending|ascending|desc|asc)$",
        r"(?:keep\s+the\s+same.*)?(?:sort|order)?\s*.*(?:from\s+)?(?:smallest|lowest|least)\s+to\s+(?:largest|highest|greatest)",
        r"(?:keep\s+the\s+same.*)?(?:sort|order)?\s*.*(?:from\s+)?(?:largest|highest|greatest)\s+to\s+(?:smallest|lowest|least)",
        r"^(?:from\s+)?(?:smallest|lowest|least)\s+to\s+(?:largest|highest|greatest)$",
        r"^(?:from\s+)?(?:largest|highest|greatest)\s+to\s+(?:smallest|lowest|least)$",
    )
    is_pure_sort = any(re.search(p, q_clean) for p in sort_patterns)
    if not is_pure_sort:
        if q_clean.startswith("sort") or q_clean.startswith("order by") or "reverse the order" in q_clean:
            is_pure_sort = True

    if not is_pure_sort:
        return None

    # Determine direction
    is_reverse = any(w in q_clean for w in ("reverse", "flip", "invert"))
    if any(k in q_clean for k in ("smallest to largest", "lowest to highest", "least to most", "ascending", "asc")):
        desc = False
    elif any(k in q_clean for k in ("largest to smallest", "highest to lowest", "most to least", "descending", "desc")):
        desc = True
    elif is_reverse:
        prev_desc = previous_plan.order_by[0].descending if previous_plan.order_by else True
        desc = not prev_desc
    else:
        desc = not any(w in q_clean for w in ("lowest", "least", "asc", "ascending", "lowest first", "smallest"))

    # Determine target column/alias
    target = None
    if previous_plan.aggregations:
        for agg in previous_plan.aggregations:
            if (agg.column and agg.column.lower() in q_clean) or (agg.alias and agg.alias.lower() in q_clean):
                target = agg.alias or (f"{agg.function.value}_{agg.column}" if agg.column else agg.column)
                break
    if not target and previous_plan.group_by:
        for gb in previous_plan.group_by:
            if gb.column.lower() in q_clean:
                target = gb.column
                break
    if not target:
        for col in metadata.profile.columns:
            if col.name.lower() in q_clean:
                target = col.name
                break
    if not target and previous_plan.order_by:
        target = previous_plan.order_by[0].target
    if not target and previous_plan.aggregations:
        agg0 = previous_plan.aggregations[0]
        target = agg0.alias or (f"{agg0.function.value}_{agg0.column}" if agg0.column else agg0.column)
    if not target and previous_plan.group_by:
        target = previous_plan.group_by[0].column

    return SortDelta(target=target or "total", descending=desc)


def _detect_pure_metric_followup(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> Optional[MetricDelta]:
    """
    Detects if a follow-up query is purely requesting to change or switch the metric
    (e.g. "Use profit instead of sales", "Switch to profit", "Show profit instead",
    "Change metric to profit", "Switch from sales to profit").
    Correctly ignores the replaced/discarded metric in 'instead of' or 'from X to Y'.
    """
    if previous_plan is None:
        return None

    q_low = question.lower().strip()
    q_clean = q_low.rstrip("?.!")

    # Exclude explicit new queries, filters, or group-by
    if any(q_clean.startswith(p) for p in ("what is", "what are", "how many", "calculate", "find", "tell me what")):
        return None
    if any(k in q_low for k in ("group by", "break down by", "pivot", "crosstab", "filter to", "where ")):
        return None
    if _has_explicit_clear_filter_directive(question):
        return None

    metric_triggers = ("instead", "switch metric", "change metric", "switch to", "now show", "use ", "show ")
    if not any(k in q_low for k in metric_triggers):
        return None

    num_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
    if not num_cols:
        return None

    # Strip the negated/replaced part (e.g. "instead of sales" or "from sales to")
    cleaned_for_target = re.sub(r"\binstead of\s+[a-z0-9_ -]+", "", q_low)
    cleaned_for_target = re.sub(r"\b(?:switch|change)\s+from\s+[a-z0-9_ -]+\s+to\b", "to", cleaned_for_target)

    target_num = None
    # 1. Exact or substring match in cleaned string
    for nc in num_cols:
        nc_low = nc.lower()
        if re.search(rf"\b{re.escape(nc_low)}\b", cleaned_for_target) or nc_low in cleaned_for_target:
            target_num = nc
            break

    # 2. Token match
    if not target_num:
        for nc in num_cols:
            for tok in re.findall(r"\b\w+\b", nc.lower()):
                if len(tok) >= 3 and re.search(rf"\b{re.escape(tok)}\b", cleaned_for_target):
                    target_num = nc
                    break
            if target_num:
                break

    # 3. Domain synonyms
    if not target_num:
        if any(w in cleaned_for_target for w in ("sales", "sale", "revenue", "turnover", "spend", "amount")):
            for nc in num_cols:
                if any(k in nc.lower() for k in ("sale", "revenue", "amount", "total", "spend")):
                    target_num = nc
                    break
        elif any(w in cleaned_for_target for w in ("profit", "margin", "earnings", "income", "gain")):
            for nc in num_cols:
                if any(k in nc.lower() for k in ("profit", "margin", "earnings", "income")):
                    target_num = nc
                    break

    if not target_num:
        return None

    fn = AggregationFunction.AVG if any(w in q_low for w in ("avg", "average", "mean")) else (
        previous_plan.aggregations[0].function if previous_plan.aggregations else AggregationFunction.SUM
    )
    return MetricDelta(
        column=target_num,
        function=fn,
        alias=f"{fn.value}_{target_num}",
    )


def _has_explicit_breakdown_directive(question: str, target_dim: str) -> bool:
    """
    Checks if the question explicitly requests grouping/breakdown by target_dim
    (e.g. 'by region', 'per region', 'across regions', 'group by region', 'break down by region').
    """
    if not target_dim:
        return False
    q_low = question.lower()
    dim_low = target_dim.lower()
    dim_stems = [re.escape(dim_low)]
    if dim_low.endswith("s"):
        dim_stems.append(re.escape(dim_low[:-1]))
    else:
        dim_stems.append(re.escape(dim_low + "s"))
    if dim_low.endswith("y"):
        dim_stems.append(re.escape(dim_low[:-1] + "ies"))
    dim_regex = "(?:" + "|".join(dim_stems) + ")"

    breakdown_patterns = [
        rf"\b(?:group\s+(?:these|them|it|results?|data)?\s*by|grouped\s+by)\s+(?:the\s+|each\s+)?{dim_regex}\b",
        rf"\b(?:break\s+down\s+(?:these|them|it|results?|data)?\s*by|broken\s+down\s+by)\s+(?:the\s+|each\s+)?{dim_regex}\b",
        rf"(?<!filter\s)(?<!filtered\s)(?<!sort\s)(?<!sorted\s)(?<!order\s)(?<!ordered\s)\b(?:by|per|across\s+(?!all\b|every\b)|for\s+each)\s+(?:the\s+|each\s+)?{dim_regex}\b",
        rf"\b(?:switch|change)\s+(?:grouping|dimension|group|breakdown)?\s*(?:to)\s+(?:the\s+)?{dim_regex}\b",
        rf"\b{dim_regex}\s+instead\b",
        rf"\bnow\s+(?:by\s+)?{dim_regex}\b",
    ]
    return any(re.search(pat, q_low) for pat in breakdown_patterns)


def _has_positive_intent(text: str, keywords: tuple[str, ...]) -> bool:
    """
    Checks if any of the keywords appear with positive intent,
    i.e. NOT negated by 'do not', 'don't', 'never', 'no', 'without', 'not using', etc.
    """
    negation_patterns = [
        r"\b(?:do\s+not|don't|dont|never|not|no|without|avoid|stop)\s+(?:use|using|apply|applying|take|taking|calculate|calculating)?\s*(?:the\s+|an?\s+)?",
    ]
    for kw in keywords:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            start = m.start()
            prefix = text[max(0, start - 40):start].lower()
            is_negated = any(re.search(np + r"$", prefix) for np in negation_patterns)
            if not is_negated:
                return True
    return False


def _is_standalone_new_query(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> bool:
    """
    Determines if question is an independent standalone query rather than an
    intentional conversational refinement of the previous analysis plan.
    """
    if previous_plan is None:
        return True

    q_low = question.lower().strip()
    has_clear_directive = _has_explicit_clear_filter_directive(q_low)

    # When clear filter directive is present, strip it from anaphora check so words like
    # 'previous' in 'ignoring previous filters' or 'clear filters' don't trigger false anaphora
    q_check_anaphora = q_low
    if has_clear_directive:
        for pat in _EXPLICIT_CLEAR_FILTER_PATTERNS:
            q_check_anaphora = re.sub(pat, " ", q_check_anaphora)

    # Intentional follow-up indicators:
    # 1. Anaphora / contextual pronouns referencing previous results
    anaphora = (
        "this", "these", "those", "them", "that", "it", "the results", "the data",
        "previous", "above", "earlier", "same"
    )
    if any(re.search(rf"\b{re.escape(a)}\b", q_check_anaphora) for a in anaphora):
        return False

    # 2. Conversational modifiers / transition words indicating continuation
    if any(q_low.startswith(cs) or f" {cs}" in q_low for cs in ("instead", "further", "furthermore")):
        return False
    continuation_starters = (
        "now ", "also ", "and now ", "then ", "next ", "what about", "how about",
        "keep the", "keep only", "add ", "remove ", "clear ", "reset ", "except "
    )
    if any(q_low.startswith(cs) for cs in continuation_starters):
        if not (has_clear_directive and any(k in q_low for k in ("count", "total", "sum", "average", "avg", "by ", "per "))):
            return False

    # 3. Pure filtering directives
    filter_intents = ("filter to", "filter by", "filtered to", "where ", "only in", "only for")
    if any(q_low.startswith(fi) for fi in filter_intents):
        return False

    # 4. Pure sorting, limit, table presentation, or visualization toggles
    if any(q_low.startswith(s) for s in ("sort by", "order by", "top ", "bottom ", "limit ", "show as table", "as a table")):
        return False

    # 5. Intentional dimension changes and visualization directives:
    # e.g. "Plot sales by region", "Plot sales by category", "Group by category", "Break down by region"
    prev_metric_cols = {
        agg.column.lower() for agg in previous_plan.aggregations if agg.column
    }
    if previous_plan.pivot and previous_plan.pivot.metric:
        prev_metric_cols.add(previous_plan.pivot.metric.lower())

    num_cols = [c.name.lower() for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
    mentioned_metrics = {nc for nc in num_cols if re.search(rf"\b{re.escape(nc)}\b", q_low)}

    has_breakdown_directive = (
        any(k in q_low for k in ("group by", "break down by", "broken down by"))
        or bool(re.search(r"\b(?:by|across|per|for each)\s+([a-z0-9_ -]+)", q_low))
    )
    is_refinement_directive = any(q_low.startswith(p) for p in (
        "plot ", "chart ", "graph ", "visualize ", "group by ", "break down by ", "break down ", "by "
    ))

    # Independent interrogative query starters
    interrogative_starters = (
        "what is", "what are", "how much", "how many", "calculate", "find ", "give me",
        "tell me", "can you", "could you"
    )
    is_interrogative = any(q_low.startswith(s) for s in interrogative_starters)

    if (is_refinement_directive or has_breakdown_directive) and not is_interrogative:
        if has_clear_directive:
            # Explicit instruction to evaluate across entire dataset or clear filters
            # with an aggregation or metric or independent breakdown is a standalone query
            mentions_agg = any(re.search(rf"\b{re.escape(w)}\b", q_low) for w in (
                "total", "sum", "average", "avg", "mean", "count", "number of", "records", "min", "minimum", "max", "maximum"
            ))
            if mentions_agg or mentioned_metrics:
                return True
        if not mentioned_metrics:
            return False
        if prev_metric_cols and mentioned_metrics.issubset(prev_metric_cols):
            # Same metric continues under a new grouping dimension or chart
            return False

    # 6. If explicitly interrogative, it is an independent query
    if is_interrogative:
        return True

    # 7. Incomplete fragments without metric/agg (e.g. "by product", "break down by product")
    mentions_agg = any(re.search(rf"\b{re.escape(w)}\b", q_low) for w in (
        "total", "sum", "average", "avg", "mean", "count", "number of", "records", "min", "minimum", "max", "maximum"
    ))
    if not (mentioned_metrics or mentions_agg):
        return False

    # 8. Otherwise, if it mentions an aggregation and/or a distinct metric, treat as independent query
    return True


def _detect_pure_groupby_followup(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
) -> Optional[PlanMutation]:
    """
    Detects follow-up queries that purely alter or add grouping dimensions
    (e.g. "Group these by category", "Group by category", "Now group by category",
    "Break down by category", "Also by region", "Plot sales by region").
    """
    if previous_plan is None or _is_standalone_new_query(question, metadata, previous_plan):
        return None

    if _has_explicit_clear_filter_directive(question):
        return None

    q_low = question.lower().strip()
    q_clean = q_low.rstrip("?.!")

    # Check for additive grouping
    str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) in ("string", "datetime")]
    if not str_cols:
        return None

    if any(k in q_low for k in ("also by", "add group", "break down further by", "add dimension")):
        for sc in str_cols:
            if re.search(rf"\b{re.escape(sc.lower())}\b", q_low) or sc.lower() in q_low:
                return PlanMutation(
                    mutation_type=PlanMutationType.ADD_GROUP_BY,
                    group_by_delta=GroupByDelta(column=sc),
                )

    filter_intents = ("filter to", "filter by", "filtered to", "where ", "only in", "only for")
    is_pure_filter = any(fi in q_low for fi in filter_intents) and not any(b in q_low for b in ("by ", "per ", "across ", "group by ", "break down by "))
    if not is_pure_filter:
        for sc in str_cols:
            if _has_explicit_breakdown_directive(question, sc):
                return PlanMutation(
                    mutation_type=PlanMutationType.CHANGE_GROUP_BY,
                    group_by_delta=GroupByDelta(column=sc),
                )

    groupby_triggers = (
        r"\b(?:group\s+(?:these|them|it|results?|data)?\s*by)\s+([a-z0-9_ -]+)",
        r"\b(?:break\s+down\s+(?:these|them|it|results?|data)?\s*by)\s+([a-z0-9_ -]+)",
        r"\b(?:now\s+group\s+by)\s+([a-z0-9_ -]+)",
        r"\b(?:group\s+by)\s+([a-z0-9_ -]+)",
        r"\b(?:by\s+([a-z0-9_ -]+)\s+instead)\b",
    )
    for p in groupby_triggers:
        m = re.search(p, q_clean)
        if m:
            matched_phrase = m.group(1).strip()
            for sc in str_cols:
                sc_low = sc.lower()
                if (
                    sc_low in matched_phrase
                    or matched_phrase in sc_low
                    or sc_low.rstrip("s") in matched_phrase
                    or re.search(rf"\b{re.escape(sc_low)}\b", q_clean)
                ):
                    return PlanMutation(
                        mutation_type=PlanMutationType.CHANGE_GROUP_BY,
                        group_by_delta=GroupByDelta(column=sc),
                    )

    return None



def plan_mutation(
    question: str,
    metadata: DatasetMetadata,
    previous_plan: Optional[AnalysisPlan] = None,
    resolved_metric: Optional[str] = None,
) -> PlanMutation:
    """
    Classifies user analytical intent and produces a typed PlanMutation.
    If previous_plan is None or question is an independent query, generates a fresh AnalysisPlan and wraps it in NEW_QUERY.
    If previous_plan is provided, prompts the LLM or falls back to deterministic heuristic classification.
    """
    if previous_plan is None or _is_standalone_new_query(question, metadata, previous_plan):
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
    q_low = question.lower().strip()
    limit_match = re.search(rf"^(?:show\s+)?(?:top|bottom|limit|first|last)\s+({WORD_NUM_PATTERN})$", q_low)
    if not limit_match:
        limit_match = re.search(rf"^(?:show\s+)?({WORD_NUM_PATTERN})\s+(?:top|bottom|limit|first|last)$", q_low)
    if limit_match:
        limit_val = _parse_int_or_word(limit_match.group(1))
        if limit_val is not None:
            return PlanMutation(
                mutation_type=PlanMutationType.CHANGE_LIMIT,
                limit_delta=limit_val,
            )

    # Fast deterministic intercept for pure chart follow-ups
    chart_followup = _detect_pure_chart_followup(question, metadata, previous_plan)
    if chart_followup is not None:
        req_chart, ctype = chart_followup
        m_type = PlanMutationType.TOGGLE_HEATMAP if ctype == "heatmap" else PlanMutationType.TOGGLE_CHART
        return PlanMutation(
            mutation_type=m_type,
            chart_type=ctype if req_chart else None,
            requires_chart=req_chart,
        )

    # Fast deterministic intercept for pure table presentation follow-ups
    if _detect_pure_table_followup(question, metadata, previous_plan):
        return PlanMutation(
            mutation_type=PlanMutationType.TOGGLE_CHART,
            chart_type=None,
            requires_chart=False,
        )

    # Fast deterministic intercept for pure sort follow-ups
    sort_delta = _detect_pure_sort_followup(question, metadata, previous_plan)
    if sort_delta is not None:
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_SORT,
            sort_delta=sort_delta,
        )

    # Fast deterministic intercept for pure metric follow-ups
    metric_delta = _detect_pure_metric_followup(question, metadata, previous_plan)
    if metric_delta is not None:
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_METRIC,
            metric_delta=metric_delta,
        )

    # Fast deterministic intercept for pure group-by follow-ups
    groupby_mutation = _detect_pure_groupby_followup(question, metadata, previous_plan)
    if groupby_mutation is not None:
        return groupby_mutation

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
        if isinstance(parsed_json.get("new_plan"), dict) and parsed_json["new_plan"].get("limit") is None:
            parsed_json["new_plan"]["limit"] = 500

        mutation = PlanMutation.model_validate(parsed_json)
    except Exception as e:
        logger.warning(f"LLM mutation planning failed: {e}. Using deterministic heuristic fallback.")
        mutation = _fallback_heuristic_mutation(question, metadata, previous_plan, resolved_metric)

    return mutation


def _resolve_target_metric(
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
    fallback_to_first: bool = True,
) -> Optional[str]:
    """
    Deterministically resolves the intended numeric metric column from the question
    and dataset metadata, handling token stems, multi-word column names, and synonyms.
    """
    num_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
    if not num_cols:
        return None
    num_col_map = {c.lower(): c for c in num_cols}
    if resolved_metric and resolved_metric.lower() in num_col_map:
        return num_col_map[resolved_metric.lower()]
    if resolved_metric:
        for nc in num_cols:
            if resolved_metric.lower() in nc.lower() or nc.lower() in resolved_metric.lower():
                return nc

    q_low = question.lower()
    # If the user is substituting metrics (e.g. "profit instead of sales" or "switch from sales to profit"),
    # strip the discarded metric clause so we don't accidentally resolve the discarded metric.
    if "instead of" in q_low or "from " in q_low:
        q_low = re.sub(r"\binstead of\s+[a-z0-9_ -]+", "", q_low)
        q_low = re.sub(r"\b(?:switch|change)\s+from\s+[a-z0-9_ -]+\s+to\b", "to", q_low)

    # 1. Look for explicit "by <word>", "of <word>", "on <word>", "in terms of <word>"
    by_match = re.search(r"\b(?:by|of|on|in terms of|ranked by)\s+([a-z0-9_ -]+)", q_low)
    if by_match:
        target_phrase = by_match.group(1).strip()
        for nc in num_cols:
            if target_phrase == nc.lower() or target_phrase in nc.lower() or nc.lower() in target_phrase:
                return nc
        phrase_tokens = set(re.findall(r"[a-z0-9]+", target_phrase.lower()))
        for nc in num_cols:
            nc_tokens = set(re.findall(r"[a-z0-9]+", nc.lower()))
            if phrase_tokens & nc_tokens:
                return nc
            for pt in phrase_tokens:
                for nt in nc_tokens:
                    if pt.rstrip("s") == nt.rstrip("s") and len(pt.rstrip("s")) >= 3:
                        return nc

    # 2. Look for full column name in question (both original and underscore-to-space)
    for nc in num_cols:
        nc_low = nc.lower()
        nc_space = nc_low.replace("_", " ")
        if (
            nc_low in q_low
            or nc_space in q_low
            or re.search(rf"\b{re.escape(nc_low)}\b", q_low)
            or re.search(rf"\b{re.escape(nc_space)}\b", q_low)
        ):
            return nc

    # 3. Look for token stem matches in question (e.g. "sales" or "sale" in "Total Sales", "cost" in "treatment_cost_usd")
    best_nc = None
    best_nc_score = 0
    for nc in num_cols:
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", nc.lower()) if len(tok) >= 3]
        matches = 0
        for tok in tokens:
            stem = tok.rstrip("s")
            if re.search(r"\b" + re.escape(tok) + r"\b", q_low) or re.search(r"\b" + re.escape(stem) + r"\b", q_low):
                matches += 1
        if matches > best_nc_score:
            best_nc_score = matches
            best_nc = nc
    if best_nc:
        return best_nc

    # 4. Domain synonyms (dataset-agnostic heuristics)
    if any(w in q_low for w in ("sales", "sale", "revenue", "turnover", "spend", "spending", "amount", "cost", "price", "charge")):
        for nc in num_cols:
            if any(k in nc.lower() for k in ("sale", "revenue", "amount", "total", "spend", "cost", "price", "charge")):
                return nc
    if any(w in q_low for w in ("profit", "margin", "earnings", "income", "gain")):
        for nc in num_cols:
            if any(k in nc.lower() for k in ("profit", "margin", "income", "gain")):
                return nc
    if any(w in q_low for w in ("unit", "units", "quantity", "volume", "count", "orders", "items", "qty")):
        for nc in num_cols:
            if any(k in nc.lower() for k in ("unit", "qty", "quantity", "count", "vol")):
                return nc

    if fallback_to_first:
        return num_cols[0] if num_cols else None
    return None


def _resolve_target_dimensions(question: str, metadata: DatasetMetadata) -> list[str]:
    """
    Deterministically resolves all intended categorical or temporal dimension columns from the question.
    Preserves multiple dimensions when multi-column grouping is requested (e.g. 'City and Category').
    """
    dim_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) in ("string", "datetime")]
    if not dim_cols:
        return []
    q_low = question.lower()
    matched_spans: list[tuple[int, str]] = []

    for sc in dim_cols:
        sc_low = sc.lower()
        sc_space = sc_low.replace("_", " ")
        patterns = [
            rf"\b{re.escape(sc_low)}\b",
            rf"\b{re.escape(sc_space)}\b",
            rf"\b{re.escape(sc_low.rstrip('s'))}\b",
        ]
        if sc_low.endswith("y"):
            patterns.append(rf"\b{re.escape(sc_low[:-1] + 'ies')}\b")

        for pat in patterns:
            m = re.search(pat, q_low)
            if m:
                preceding = q_low[:m.start()]
                if re.search(r"\b(?:across\s+all|in\s+all|for\s+all|across\s+the\s+entire|across\s+the\s+whole|across\s+every|in\s+every|for\s+every)\s+$", preceding):
                    continue
                matched_spans.append((m.start(), sc))
                break

    # If no exact matches found, check domain synonyms
    if not matched_spans:
        synonym_map = {
            ("item", "items", "product", "products", "good", "goods", "merchandise", "sku", "skus"): ("product", "item", "description"),
            ("city", "cities", "location", "locations", "place", "places", "region", "regions", "branch", "branches"): ("city", "location", "region", "branch", "state"),
            ("customer", "customers", "client", "clients", "buyer", "buyers", "user", "users"): ("customer", "client", "customer type", "user"),
            ("category", "categories", "department", "departments", "type", "types", "group", "groups"): ("category", "department", "type", "genre"),
            ("payment", "payments", "method", "methods"): ("payment", "method", "tender"),
            ("date", "dates", "day", "days", "time", "period"): ("date", "time", "timestamp"),
        }
        for trigger_tokens, target_substrings in synonym_map.items():
            for tok in trigger_tokens:
                m = re.search(rf"\b{re.escape(tok)}\b", q_low)
                if m:
                    preceding = q_low[:m.start()]
                    if re.search(r"\b(?:across\s+all|in\s+all|for\s+all|across\s+the\s+entire|across\s+the\s+whole|across\s+every|in\s+every|for\s+every)\s+$", preceding):
                        continue
                    for sc in dim_cols:
                        if any(sub in sc.lower() for sub in target_substrings):
                            matched_spans.append((m.start(), sc))
                            break
                    break

    # Sort by order of appearance in question and deduplicate
    matched_spans.sort(key=lambda x: x[0])
    seen = set()
    result = []
    for _, col in matched_spans:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


def _resolve_target_dimension(question: str, metadata: DatasetMetadata) -> Optional[str]:
    """
    Deterministically resolves the primary categorical dimension column from the question.
    """
    dims = _resolve_target_dimensions(question, metadata)
    return dims[0] if dims else None


def _extract_comparison_filter(question: str, metadata: DatasetMetadata) -> Optional[FilterClause]:
    """
    Extracts explicit comparison filters like 'at least 100 orders', 'orders >= 100',
    or 'at most 50 quantity' deterministically.
    """
    q_low = question.lower()
    num_cols = {c.name.lower(): c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"}
    if not num_cols:
        return None

    # Pattern 1: at least <number> <col>
    m1 = re.search(r"\bat\s+least\s+([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z0-9_ -]+)?", q_low)
    if m1:
        val = float(m1.group(1)) if "." in m1.group(1) else int(m1.group(1))
        remainder = (m1.group(2) or "").strip()
        matched_col = None
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    matched_col = nc_orig
                    break
            if matched_col:
                break
        if not matched_col:
            before = q_low[:m1.start()]
            for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", before):
                tok_stem = tok.rstrip("s")
                for nc_low, nc_orig in num_cols.items():
                    if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                        matched_col = nc_orig
                        break
                if matched_col:
                    break
        if matched_col:
            return FilterClause(column=matched_col, operator=FilterOperator.GTE, value=val)

    # Pattern 2: <col> ... at least <number>
    m2 = re.search(r"([a-zA-Z0-9_ -]+)\s+(?:of\s+|is\s+|with\s+)?at\s+least\s+([0-9]+(?:\.[0-9]+)?)", q_low)
    if m2:
        val = float(m2.group(2)) if "." in m2.group(2) else int(m2.group(2))
        remainder = (m2.group(1) or "").strip()
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    return FilterClause(column=nc_orig, operator=FilterOperator.GTE, value=val)

    # Pattern 3: >= <number> <col>
    m3 = re.search(r"(?:>=|greater than or equal to|minimum of)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z0-9_ -]+)?", q_low)
    if m3:
        val = float(m3.group(1)) if "." in m3.group(1) else int(m3.group(1))
        remainder = (m3.group(2) or "").strip()
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    return FilterClause(column=nc_orig, operator=FilterOperator.GTE, value=val)

    # Pattern 4: at most <number> <col>
    m4 = re.search(r"\bat\s+most\s+([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z0-9_ -]+)?", q_low)
    if m4:
        val = float(m4.group(1)) if "." in m4.group(1) else int(m4.group(1))
        remainder = (m4.group(2) or "").strip()
        matched_col = None
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    matched_col = nc_orig
                    break
            if matched_col:
                break
        if not matched_col:
            before = q_low[:m4.start()]
            for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", before):
                tok_stem = tok.rstrip("s")
                for nc_low, nc_orig in num_cols.items():
                    if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                        matched_col = nc_orig
                        break
                if matched_col:
                    break
        if matched_col:
            return FilterClause(column=matched_col, operator=FilterOperator.LTE, value=val)

    # Pattern 5: <col> ... at most <number>
    m5 = re.search(r"([a-zA-Z0-9_ -]+)\s+(?:of\s+|is\s+|with\s+)?at\s+most\s+([0-9]+(?:\.[0-9]+)?)", q_low)
    if m5:
        val = float(m5.group(2)) if "." in m5.group(2) else int(m5.group(2))
        remainder = (m5.group(1) or "").strip()
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    return FilterClause(column=nc_orig, operator=FilterOperator.LTE, value=val)

    # Pattern 6: <= <number> <col>
    m6 = re.search(r"(?:<=|less than or equal to|maximum of)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z0-9_ -]+)?", q_low)
    if m6:
        val = float(m6.group(1)) if "." in m6.group(1) else int(m6.group(1))
        remainder = (m6.group(2) or "").strip()
        for tok in re.findall(r"\b[a-zA-Z0-9_]+\b", remainder):
            tok_stem = tok.rstrip("s")
            for nc_low, nc_orig in num_cols.items():
                if tok == nc_low or tok_stem == nc_low.rstrip("s") or tok in nc_low or nc_low in tok:
                    return FilterClause(column=nc_orig, operator=FilterOperator.LTE, value=val)

    return None


_DATASET_COLUMN_VALUES_CACHE: dict[tuple[str, str], set[str]] = {}


def _get_known_column_values(target_dim: str, metadata: DatasetMetadata) -> set[str]:
    """Attempts to retrieve known unique values for target_dim if dataset file is accessible."""
    if not target_dim or not getattr(metadata, "dataset_id", None):
        return set()
    cache_key = (metadata.dataset_id, target_dim)
    if cache_key in _DATASET_COLUMN_VALUES_CACHE:
        return _DATASET_COLUMN_VALUES_CACHE[cache_key]

    values = set()
    try:
        from app.analytics.dataset_service import get_dataset_file_path
        import duckdb
        path = get_dataset_file_path(metadata.dataset_id, metadata.session_id)
        if path.exists():
            con = duckdb.connect(":memory:")
            ext = metadata.format.lower()
            if ext == "csv":
                sql = f"SELECT DISTINCT \"{target_dim}\" FROM read_csv_auto('{path.as_posix()}') WHERE \"{target_dim}\" IS NOT NULL LIMIT 500"
                rows = con.execute(sql).fetchall()
            elif ext == "json":
                sql = f"SELECT DISTINCT \"{target_dim}\" FROM read_json_auto('{path.as_posix()}') WHERE \"{target_dim}\" IS NOT NULL LIMIT 500"
                rows = con.execute(sql).fetchall()
            elif ext in ("parquet", "pq"):
                sql = f"SELECT DISTINCT \"{target_dim}\" FROM read_parquet('{path.as_posix()}') WHERE \"{target_dim}\" IS NOT NULL LIMIT 500"
                rows = con.execute(sql).fetchall()
            elif ext in ("xlsx", "xls"):
                import pandas as pd
                str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"]
                use_cols = [c for c in str_cols if c] or [target_dim]
                try:
                    df = pd.read_excel(path, usecols=use_cols, engine="openpyxl")
                    for c_name in use_cols:
                        if c_name in df.columns:
                            col_vals = set(df[c_name].dropna().astype(str).str.strip().unique())
                            _DATASET_COLUMN_VALUES_CACHE[(metadata.dataset_id, c_name)] = col_vals
                    return _DATASET_COLUMN_VALUES_CACHE.get(cache_key, set())
                except Exception:
                    df = pd.read_excel(path, usecols=[target_dim], engine="openpyxl")
                    con.register("xlsx_src", df)
                    sql = f"SELECT DISTINCT \"{target_dim}\" FROM xlsx_src WHERE \"{target_dim}\" IS NOT NULL LIMIT 500"
                    rows = con.execute(sql).fetchall()
            else:
                rows = []
            for (r,) in rows:
                if r is not None:
                    values.add(str(r).strip())
            con.close()
    except Exception:
        pass
    _DATASET_COLUMN_VALUES_CACHE[cache_key] = values
    return values


def _reconcile_filter_column(
    filter_col: str,
    filter_val: Any,
    metadata: DatasetMetadata,
) -> str:
    """
    Validates and reconciles filter column against known distinct values in the dataset.
    Rebinds a value only when dataset evidence unambiguously identifies exactly one matching column.
    If multiple columns match or none match, preserves the original column without guessing.
    """
    if filter_val is None or not isinstance(filter_val, (str, int, float)):
        return filter_col
    val_str = str(filter_val).strip()
    if not val_str:
        return filter_col

    str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"]
    if not str_cols:
        return filter_col

    # If already verified in the target column, preserve it
    if filter_col in str_cols:
        col_vals = _get_known_column_values(filter_col, metadata)
        if any(val_str.lower() == str(v).lower() for v in col_vals):
            return filter_col

    # Search all string columns for unambiguous match
    matching_cols = []
    for sc in str_cols:
        vals = _get_known_column_values(sc, metadata)
        if any(val_str.lower() == str(v).lower() for v in vals):
            matching_cols.append(sc)

    if len(matching_cols) == 1:
        return matching_cols[0]

    return filter_col


def _is_explicitly_filtered_in_turn(target_dim: str, filter_val: Any, question: str) -> bool:
    """
    Syntactically verifies if a specific filter value on target_dim was explicitly requested
    in this turn rather than loosely mentioned or inherited.
    """
    if filter_val is None:
        return False
    val_str = str(filter_val).strip()
    if not val_str:
        return False
    q_low = question.lower()
    val_low = val_str.lower()
    dim_low = target_dim.lower()

    if not re.search(rf"\b{re.escape(val_low)}\b", q_low):
        return False

    # If preceded by an exclusion keyword, it is an exclusion, not an equality filter
    if re.search(rf"\b(?:except(?:\s+for)?|excluding|other\s+than|without|not\s+including)\s+(?:the\s+)?{re.escape(val_low)}\b", q_low):
        return False

    # 1. Relational / direct assignment syntax
    assign_patterns = [
        rf"\b{re.escape(dim_low)}\s*(?:is|=|==|:|equals?|in)\s*(?:['\"]?){re.escape(val_low)}(?:['\"]?)\b",
        rf"\bwhere\s+(?:the\s+)?{re.escape(dim_low)}\s*(?:is|=|==|:|equals?)\s*(?:['\"]?){re.escape(val_low)}(?:['\"]?)\b",
    ]
    if any(re.search(pat, q_low) for pat in assign_patterns):
        return True

    # 2. Scoped / restrictive phrasing
    restrictive_patterns = [
        rf"\b(?:only\s+(?:in\s+|for\s+)?{re.escape(val_low)}|{re.escape(val_low)}\s+only|just\s+(?:in\s+|for\s+)?{re.escape(val_low)})\b",
        rf"\b(?:for\s+{re.escape(val_low)}\s+only|only\s+the\s+{re.escape(val_low)}\s+{re.escape(dim_low)}|{re.escape(val_low)}\s+{re.escape(dim_low)}\s+only)\b",
        rf"\b(?:filtered\s+to|filter\s+to|filter\s+by)\s+(?:only\s+)?(?:the\s+)?{re.escape(val_low)}\b",
        rf"\b(?:in|for)\s+(?:the\s+)?{re.escape(val_low)}\s+(?:{re.escape(dim_low)}|area|territory|zone|branch)\b",
        rf"\b(?:in|for)\s+(?:the\s+)?{re.escape(dim_low)}\s+(?:of\s+)?{re.escape(val_low)}\b",
    ]
    if any(re.search(pat, q_low) for pat in restrictive_patterns):
        return True

    return False


def _extract_exclusion_filters(
    question: str, target_dim: str, metadata: DatasetMetadata
) -> list[FilterClause]:
    """
    Extracts explicit exclusion filters on target_dim (e.g. 'except North', 'excluding North and South').
    Validates candidate values against known dataset values and prevents capturing trailing clauses
    like 'for 2024' or 'where status = delivered'.
    """
    m = re.search(
        r"\b(?:except(?:\s+for)?|excluding|other\s+than|without|not\s+including)\s+([^,;?\.]+)",
        question,
        re.IGNORECASE,
    )
    if not m:
        return []
    remainder = m.group(1).strip()
    known_values = _get_known_column_values(target_dim, metadata)

    # 1. Authoritative allowlist matching when known_values is available
    if known_values:
        # Sort by length descending to prefer longest match (e.g. 'North America' before 'North')
        sorted_kvs = sorted(known_values, key=lambda x: len(str(x)), reverse=True)
        rem_low = remainder.lower()
        matched_spans: list[tuple[int, int, str]] = []

        for kv in sorted_kvs:
            kv_str = str(kv).strip()
            if not kv_str:
                continue
            pat = rf"\b{re.escape(kv_str.lower())}\b"
            for match in re.finditer(pat, rem_low):
                start, end = match.span()
                # Suppress shorter match that overlaps with an already accepted longer match
                if not any(not (end <= s or start >= e) for s, e, _ in matched_spans):
                    matched_spans.append((start, end, kv_str))

        if matched_spans:
            # Sort by appearance in remainder to preserve user order
            matched_spans.sort(key=lambda x: x[0])
            matched = [kv for _, _, kv in matched_spans]
            if len(matched) == 1:
                return [FilterClause(column=target_dim, operator=FilterOperator.NEQ, value=matched[0])]
            return [FilterClause(column=target_dim, operator=FilterOperator.NOT_IN, value=matched)]

        # Authoritative allowlist: known values exist but no match was found -> return no exclusion filter
        return []

    # 2. Fallback parsing when known_values is unavailable
    # Check for quoted phrases first to protect legitimate values with 'and' or 'or'
    quoted_matches = re.findall(r"['\"]([^'\"]+)['\"]", remainder)
    if quoted_matches:
        cleaned_quoted = [q.strip() for q in quoted_matches if q.strip()]
        if len(cleaned_quoted) == 1:
            return [FilterClause(column=target_dim, operator=FilterOperator.NEQ, value=cleaned_quoted[0])]
        elif len(cleaned_quoted) > 1:
            return [FilterClause(column=target_dim, operator=FilterOperator.NOT_IN, value=cleaned_quoted)]

    # Split before trailing temporal/filter clauses
    parts = re.split(
        r"\b(?:for\b|in\b|during\b|after\b|before\b|where\b|with\b|having\b|filtered\s+by\b|filter\s+by\b|as\s+(?:a\s+)?(?:bar|line|scatter|histogram|chart)|sorted\b|ordered\b)",
        remainder,
        flags=re.IGNORECASE,
    )
    core_phrase = parts[0].strip()
    raw_tokens = re.split(r",|\band\b|\bor\b", core_phrase, flags=re.IGNORECASE)
    res = []
    dim_low = target_dim.lower()
    for tok in raw_tokens:
        clean = tok.strip(" '\"`")
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean, flags=re.IGNORECASE).strip()
        if clean and clean.lower() not in ("for", "in", "by", "with", "where", "during", "after", "before", dim_low):
            # Guard against multi-word sentence fragments (more than 3 words without quotes)
            if len(clean.split()) <= 3:
                res.append(clean)

    if not res:
        return []
    if len(res) == 1:
        return [FilterClause(column=target_dim, operator=FilterOperator.NEQ, value=res[0])]
    return [FilterClause(column=target_dim, operator=FilterOperator.NOT_IN, value=res)]


def _extract_explicit_in_turn_filter(
    question: str, target_dim: str, metadata: DatasetMetadata
) -> Optional[FilterClause]:
    """
    Extracts an explicit equality filter on target_dim requested in this turn
    (e.g. 'for North only', 'where region is North', 'region == North', 'North only').
    """
    if not target_dim:
        return None
    dim_low = target_dim.lower()
    q_low = question.lower()

    # Direct assignment: region is North / where region is North / region = North
    m_assign = re.search(
        rf"\b(?:where\s+)?{re.escape(dim_low)}\s*(?:is|=|==|:|equals?)\s*['\"]?([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)['\"]?(?:\s+(?:only|and|or|for|\.|\?|$)|$)",
        question,
        re.IGNORECASE,
    )
    if m_assign:
        val = m_assign.group(1).strip()
        if val and val.lower() not in ("only", "the", "a", "an"):
            return FilterClause(column=target_dim, operator=FilterOperator.EQ, value=val)

    known_vals = _get_known_column_values(target_dim, metadata)
    if known_vals:
        for kv in sorted(known_vals, key=lambda x: len(str(x)), reverse=True):
            kv_esc = re.escape(str(kv).lower())
            pat = rf"\b(?:for\s+{kv_esc}\s+only|{kv_esc}\s+only|only\s+(?:in\s+|for\s+)?{kv_esc}|just\s+(?:in\s+|for\s+)?{kv_esc})\b"
            if re.search(pat, q_low):
                excl_pat = rf"\b(?:except(?:\s+for)?|excluding|other\s+than|without|not\s+including)\s+(?:the\s+)?{kv_esc}\b"
                if not re.search(excl_pat, q_low):
                    return FilterClause(column=target_dim, operator=FilterOperator.EQ, value=str(kv))

    # Generic restrictive patterns without known_vals
    excluded_only_words = {
        "the", "a", "an", "bar", "chart", "table", "line", "grid",
        "return", "show", "list", "display", "give", "select", "get", "view",
        "sales", "profit", "revenue", "values", "records", "rows", "columns", "names", "products",
        "top", "bottom", "first", "last", "order", "sort", "group", "grouped",
        dim_low,
    }

    m_for_only = re.search(r"\bfor\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\s+only\b", question, re.IGNORECASE)
    if m_for_only:
        val = m_for_only.group(1).strip()
        if val.lower() not in excluded_only_words:
            if not known_vals or val.lower() in {str(k).lower() for k in known_vals}:
                return FilterClause(column=target_dim, operator=FilterOperator.EQ, value=val)

    m_val_only = re.search(r"\b([a-zA-Z0-9_]+)\s+only\b", question, re.IGNORECASE)
    if m_val_only:
        val = m_val_only.group(1).strip()
        if val.lower() not in excluded_only_words:
            if not known_vals or val.lower() in {str(k).lower() for k in known_vals}:
                return FilterClause(column=target_dim, operator=FilterOperator.EQ, value=val)

    m_only_for = re.search(r"\bonly\s+(?:for|in)\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\b", question, re.IGNORECASE)
    if m_only_for:
        val = m_only_for.group(1).strip()
        if val.lower() not in excluded_only_words:
            if not known_vals or val.lower() in {str(k).lower() for k in known_vals}:
                return FilterClause(column=target_dim, operator=FilterOperator.EQ, value=val)

    return None


def _normalize_and_align_plan(
    plan: AnalysisPlan,
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
    is_mutation: bool = False,
) -> AnalysisPlan:
    """
    Enforces deterministic AST invariants on both freshly planned and mutated queries:
    1. Top-N/Bottom-N limit parsing and bounds enforcement.
    2. Correct ORDER BY direction (DESC for top/highest, ASC for bottom/lowest).
    3. Removal of hallucinated aggregations on group dimension (e.g. unwanted top_products).
    4. GroupBy alignment and single-dimension focus for ranking queries.
    5. Preserves un-ordered state for ordinary grouped queries without explicit sorting.
    6. Ensures server-authoritative dataset_id.
    """
    q_low = question.lower()
    superlative_desc = ("top", "highest", "best", "most", "greatest", "leading", "largest", "max", "maximum")
    superlative_asc = ("bottom", "lowest", "least", "worst", "smallest", "min", "minimum")
    all_superlatives = (*superlative_desc, *superlative_asc)
    has_superlative = any(
        (re.search(r"(?<!at\s)\bleast\b", q_low) if w == "least" else re.search(rf"\b{w}\b", q_low))
        for w in all_superlatives
    )

    # 1. Strip hallucinated superlative filters (e.g. filter product == 'best' or 'top 5')
    if plan.filters and has_superlative:
        superlative_words = set(all_superlatives)
        plan.filters = [
            f for f in plan.filters
            if not any(sw in str(f.value).lower().split() for sw in superlative_words)
        ]

    # 2. Parse Top-N / Bottom-N numeric limit and validate bounds
    n_match = re.search(
        rf"(?:\b(?:top|bottom|highest|lowest|best|worst|first|last|limit)\b|(?<!at\s)\bleast\b)\s+({WORD_NUM_PATTERN})\b",
        q_low,
    )
    if not n_match:
        n_match = re.search(
            rf"\b({WORD_NUM_PATTERN})\s+(?:top|bottom|highest|lowest|best|worst|first|last|(?<!at\s)least)\b",
            q_low,
        )
    if n_match:
        parsed_n = _parse_int_or_word(n_match.group(1))
        if parsed_n is not None:
            plan_dict = plan.model_dump()
            plan_dict["limit"] = parsed_n
            plan = AnalysisPlan.model_validate(plan_dict)
    elif has_superlative and plan.limit == 50:
        plan_dict = plan.model_dump()
        plan_dict["limit"] = 10
        plan = AnalysisPlan.model_validate(plan_dict)

    # 3. If question explicitly asks for total/sum or mentions financial metrics, enforce sum
    if _has_positive_intent(q_low, ("total", "sum", "highest total", "best", "top", "sales", "revenue", "profit")):
        for agg in plan.aggregations:
            if agg.function == AggregationFunction.MAX and not _has_positive_intent(q_low, ("max", "maximum")):
                agg.function = AggregationFunction.SUM

    # 4. Categorical grouping & Metric alignment
    raw_triggers = (
        "individual", "raw rows", "raw records", "each row", "without grouping",
        "without aggregation", "list all rows", "list rows", "list transactions",
        "show individual", "list individual", "every row", "every transaction"
    )
    is_raw_row_query = any(k in q_low for k in raw_triggers)
    num_col_names = {c.name.lower(): c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"}
    target_dims = _resolve_target_dimensions(question, metadata)
    target_dim = target_dims[0] if target_dims else None
    target_metric_col = _resolve_target_metric(question, metadata, resolved_metric, fallback_to_first=not is_mutation)

    # Phase 9: Cross-tab / Pivot query alignment
    pivot_keywords = ("cross-tab", "crosstab", "cross tab", "pivot")
    str_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "string"]
    mentioned_cat_cols = [sc for sc in str_cols if sc.lower() in q_low or sc.lower().rstrip("s") in q_low]
    if any(k in q_low for k in pivot_keywords) and len(mentioned_cat_cols) >= 2:
        if plan.pivot is None:
            plan.pivot = PivotSpec(
                row_dimension=mentioned_cat_cols[0],
                column_dimension=mentioned_cat_cols[1],
                metric=target_metric_col or (list(num_col_names.values())[0] if num_col_names else "id"),
                aggregation=AggregationFunction.SUM,
                include_totals=True,
            )
        plan.group_by = []

    has_window = bool(plan.cumulative_metrics or plan.rolling_windows or plan.quantiles or plan.outlier_analysis)

    # Reconcile existing filters against dataset values
    if plan.filters:
        reconciled_filters = []
        for f in plan.filters:
            col_match = next((c for c in metadata.profile.columns if c.name.lower() == str(f.column).lower()), None)
            if col_match and classify_dtype(col_match.dtype, col_match.name) == "numeric":
                try:
                    float(f.value)
                except (ValueError, TypeError):
                    continue
            if f.operator == FilterOperator.EQ and f.value is not None:
                reconciled_col = _reconcile_filter_column(f.column, f.value, metadata)
                reconciled_filters.append(FilterClause(column=reconciled_col, operator=f.operator, value=f.value))
            else:
                reconciled_filters.append(f)
        plan.filters = reconciled_filters
    elif not plan.filters:
        m_for = re.search(r"\bfor\s+([A-Za-z0-9_\s-]+?)(?:\s+only)?\b", question, re.IGNORECASE)
        if m_for:
            cand_val = m_for.group(1).strip()
            if cand_val.lower() not in ("each", "all", "the", "a", "an", "every", "sum", "total", "average", "avg"):
                matched_c = _reconcile_filter_column(str_cols[0] if str_cols else "", cand_val, metadata)
                if matched_c:
                    known_c_vals = _get_known_column_values(matched_c, metadata)
                    if any(cand_val.lower() == str(v).lower() for v in known_c_vals):
                        plan.filters = [FilterClause(column=matched_c, operator=FilterOperator.EQ, value=cand_val)]

    if _has_explicit_clear_filter_directive(q_low):
        # Discard inherited filters when query explicitly requests clearing/ignoring filters
        # or evaluating across the whole/entire dataset. Keep only filters explicitly specified in this turn.
        plan.filters = [
            f for f in plan.filters
            if _is_explicitly_filtered_in_turn(f.column, f.value, question)
        ]

    filter_cols = {str(f.column or "").lower() for f in plan.filters}
    filter_intents = ("filter to", "filter by", "filtered to", "where ", "only in", "only for")
    is_filter_intent = any(fi in q_low for fi in filter_intents)
    has_explicit_breakdown = bool(target_dim and _has_explicit_breakdown_directive(question, target_dim))
    is_universal_total = bool(re.search(r"\b(?:across\s+all|in\s+all|for\s+all|across\s+the\s+(?:entire|whole))\b", q_low)) and not has_explicit_breakdown

    if plan.pivot is not None:
        plan.group_by = []
    elif has_window:
        plan.group_by = []
        plan.aggregations = []
    elif is_raw_row_query:
        plan.group_by = []
        plan.aggregations = []
    elif is_universal_total:
        plan.group_by = []
        plan.order_by = []
        plan.selection_order_by = []
        plan.limit = 500
    elif len(target_dims) > 1 and not is_filter_intent and (
        any(w in q_low for w in ("and", "by", "per", "across", "group"))
    ):
        plan.group_by = [GroupByClause(column=col, date_trunc=None) for col in target_dims]
    elif has_explicit_breakdown and not is_filter_intent:
        # Explicit breakdown intent overrides inherited grouping
        plan.group_by = [GroupByClause(column=target_dim, date_trunc=None)]
    elif is_mutation and plan.group_by and (is_filter_intent or (target_dim and target_dim.lower() in filter_cols and not has_explicit_breakdown)):
        # Preserve active group_by when follow-up intent is filtering on a dimension (e.g. "Filter to Region North")
        pass
    elif target_dim and (not filter_cols or target_dim.lower() not in filter_cols or has_explicit_breakdown):
        plan.group_by = [GroupByClause(column=target_dim, date_trunc=None)]
    elif has_superlative and not plan.group_by and str_cols:
        # Default to primary categorical column if ranking query omitted explicit dimension and didn't ask for time
        no_time_keywords = not any(w in q_low for w in ("date", "day", "month", "year", "time", "trend", "daily", "monthly", "over time"))
        if no_time_keywords:
            primary_dim = None
            for sc in str_cols:
                if sc.lower() not in filter_cols and not any(id_w in sc.lower() for id_w in ("id", "identifier", "code", "key", "number", "num", "hash")):
                    primary_dim = sc
                    break
            primary_dim = primary_dim or (str_cols[0] if str_cols[0].lower() not in filter_cols else (str_cols[1] if len(str_cols) > 1 else str_cols[0]))
            plan.group_by = [GroupByClause(column=primary_dim, date_trunc=None)]
    elif any(w in q_low for w in ("by ", "per ", "for each ", "group by ")) or (re.search(r"\bacross\s+(?!all\b|every\b)", q_low)):
        if target_dim and (not filter_cols or target_dim.lower() not in filter_cols or has_explicit_breakdown):
            plan.group_by = [GroupByClause(column=target_dim, date_trunc=None)]

    # Reconcile filters on target_dim when explicit breakdown is requested:
    if target_dim and has_explicit_breakdown and not is_filter_intent:
        exclusion_filters = _extract_exclusion_filters(question, target_dim, metadata)
        if exclusion_filters:
            # Replace any existing filters on target_dim with exclusion filter(s)
            preserved_filters = [f for f in plan.filters if f.column.lower() != target_dim.lower()]
            plan.filters = preserved_filters + exclusion_filters
        else:
            retained_filters = []
            for f in plan.filters:
                if f.column.lower() == target_dim.lower():
                    if _is_explicitly_filtered_in_turn(target_dim, f.value, question):
                        retained_filters.append(f)
                    # Otherwise, it's an inherited/stale filter on the new grouping dimension -> discard!
                else:
                    # Unrelated filter -> preserve UNLESS query explicitly requests clearing filters
                    if not _has_explicit_clear_filter_directive(q_low):
                        retained_filters.append(f)

            # If target_dim was explicitly filtered in this turn but not yet present in plan.filters
            if not any(f.column.lower() == target_dim.lower() for f in retained_filters):
                in_turn_filter = _extract_explicit_in_turn_filter(question, target_dim, metadata)
                if in_turn_filter is not None:
                    retained_filters.append(in_turn_filter)

            plan.filters = retained_filters

    if plan.group_by:
        for gb in plan.group_by:
            if gb.column in str_cols:
                gb.date_trunc = None

    if not is_raw_row_query and not has_window and plan.pivot is None:
        gb_cols = {gb.column.lower() for gb in plan.group_by}
        # Strip hallucinated aggregations on group dimension (e.g. unwanted top_products, count_distinct)
        cleaned_aggs = []
        for agg in plan.aggregations:
            col_low = str(agg.column or "").lower()
            alias_low = str(agg.alias or "").lower()
            if col_low in gb_cols:
                continue
            if alias_low in ("top_products", "top_product", "count_products", "products"):
                continue
            cleaned_aggs.append(agg)

        is_record_count = (
            any(k in q_low for k in ("record count", "number of records", "count of records", "count records", "how many records", "row count", "count of rows", "number of rows"))
            or (re.search(r"\b(?:count|how many)\b", q_low) and not target_metric_col)
        )
        if is_record_count:
            plan.aggregations = [
                AggregationClause(
                    column=None,
                    function=AggregationFunction.COUNT,
                    alias="count_records",
                )
            ]
            if not has_explicit_breakdown and not target_dims:
                plan.group_by = []
                plan.order_by = []
                plan.selection_order_by = []
                plan.limit = 500
        elif target_metric_col:
            wants_avg = _has_positive_intent(q_low, ("average", "avg", "mean"))
            wants_sum = _has_positive_intent(q_low, ("total", "sum"))
            wants_min = _has_positive_intent(q_low, ("min", "minimum", "lowest"))
            wants_max = _has_positive_intent(q_low, ("max", "maximum", "highest"))

            matching_aggs = []
            seen_funcs = set()
            for agg in cleaned_aggs:
                if agg.column and agg.column.lower() == target_metric_col.lower():
                    agg.column = target_metric_col
                    if agg.function == AggregationFunction.AVG and not wants_avg and wants_sum:
                        agg.function = AggregationFunction.SUM
                    elif agg.function == AggregationFunction.SUM and not wants_sum and wants_avg:
                        agg.function = AggregationFunction.AVG
                    if not agg.alias:
                        agg.alias = f"{agg.function.value}_{target_metric_col}"
                    if agg.function not in seen_funcs:
                        seen_funcs.add(agg.function)
                        matching_aggs.append(agg)

            if wants_min and AggregationFunction.MIN not in seen_funcs:
                matching_aggs.append(
                    AggregationClause(
                        column=target_metric_col,
                        function=AggregationFunction.MIN,
                        alias=f"min_{target_metric_col}",
                    )
                )
                seen_funcs.add(AggregationFunction.MIN)
            if wants_max and AggregationFunction.MAX not in seen_funcs:
                matching_aggs.append(
                    AggregationClause(
                        column=target_metric_col,
                        function=AggregationFunction.MAX,
                        alias=f"max_{target_metric_col}",
                    )
                )
                seen_funcs.add(AggregationFunction.MAX)

            if matching_aggs:
                cleaned_aggs = matching_aggs
            else:
                chosen_fn = AggregationFunction.AVG if (wants_avg and not wants_sum) else AggregationFunction.SUM
                alias_to_use = f"{chosen_fn.value}_{target_metric_col}"
                cleaned_aggs = [
                    AggregationClause(
                        column=target_metric_col,
                        function=chosen_fn,
                        alias=alias_to_use,
                    )
                ]
            plan.aggregations = cleaned_aggs

    # Phase 7: Rolling window AST alignment
    if any(k in q_low for k in ("rolling", "moving average", "moving avg")):
        if not plan.rolling_windows:
            date_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "datetime"]
            order_col = date_cols[0] if date_cols else metadata.profile.columns[0].name
            target_m = target_metric_col or (list(num_col_names.values())[0] if num_col_names else None)
            fn = WindowFunction.AVG
            if any(w in q_low for w in ("sum", "total")):
                fn = WindowFunction.SUM
            elif "min" in q_low:
                fn = WindowFunction.MIN
            elif "max" in q_low:
                fn = WindowFunction.MAX
            w_size = 3
            w_match = re.search(r"(\d+)[-\s]*(?:day|period|month|week|row)?\s*(?:rolling|moving)", q_low)
            if w_match:
                w_size = int(w_match.group(1))
            prec = max(1, w_size - 1)
            if target_m:
                plan.rolling_windows = [
                    RollingWindowSpec(
                        function=fn,
                        column=target_m,
                        order_by=order_col,
                        preceding=prec,
                        alias=f"rolling_{fn.value}_{target_m}",
                    )
                ]
                plan.group_by = []
                plan.aggregations = []
        else:
            w_match = re.search(r"(\d+)[-\s]*(?:day|period|month|week|row)?\s*(?:rolling|moving)", q_low)
            if w_match:
                w_size = int(w_match.group(1))
                prec = max(1, w_size - 1)
                for rw in plan.rolling_windows:
                    rw.preceding = prec
            for rw in plan.rolling_windows:
                rw.alias = f"rolling_{rw.function.value}_{rw.column}"
            plan.group_by = []
            plan.aggregations = []

    # Phase 7: Cumulative metric AST alignment
    if any(k in q_low for k in ("cumulative", "running total", "running sum")) and not plan.cumulative_metrics:
        date_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "datetime"]
        order_col = date_cols[0] if date_cols else metadata.profile.columns[0].name
        target_m = target_metric_col or (list(num_col_names.values())[0] if num_col_names else None)
        cfn = CumulativeFunction.SUM
        if any(w in q_low for w in ("avg", "average", "mean")):
            cfn = CumulativeFunction.AVG
        elif any(w in q_low for w in ("count",)):
            cfn = CumulativeFunction.COUNT
        plan.cumulative_metrics = [
            CumulativeSpec(
                function=cfn,
                column=target_m,
                order_by=order_col,
            )
        ]
        plan.aggregations = []
        plan.group_by = []

    # Phase 7: Quantile segmentation AST alignment
    if any(k in q_low for k in ("quartile", "quintile", "decile", "quantile", "ntile")):
        if not plan.quantiles:
            num_q = 4
            if "quintile" in q_low:
                num_q = 5
            elif "decile" in q_low:
                num_q = 10
            else:
                q_match = re.search(r"(\d+)\s*(?:quartile|quintile|decile|quantile|tier|bucket)", q_low)
                if q_match:
                    num_q = int(q_match.group(1))
            target_m = target_metric_col or (list(num_col_names.values())[0] if num_col_names else None)
            if target_m:
                plan.quantiles = [
                    QuantileSpec(
                        column=target_m,
                        num_quantiles=num_q,
                        alias=f"quantile_{num_q}_{target_m}",
                    )
                ]
        for q in plan.quantiles:
            if not q.alias or not q.alias.startswith("quantile_"):
                q.alias = f"quantile_{q.num_quantiles}_{q.column}"
        plan.aggregations = []
        plan.group_by = []

    # Phase 7: Outlier AST alignment
    if plan.outlier_analysis is not None:
        oa = plan.outlier_analysis
        thresh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:stddev|standard deviation|standard deviations|sigma|z-?score)", q_low)
        if thresh_match:
            try:
                oa.threshold = float(thresh_match.group(1))
            except ValueError:
                pass
        if oa.partition_by:
            oa.partition_by = [
                p for p in oa.partition_by
                if p.lower() in q_low
            ]
        filter_only = not any(k in q_low for k in ("all rows", "don't filter", "flag all", "keep all", "all records", "all transactions"))
        oa.filter_outliers_only = filter_only
        if plan.filters:
            clean_filters = []
            for f in plan.filters:
                if f.column == oa.column and (
                    str(f.value) == str(oa.threshold)
                    or "standard deviation" in q_low
                    or "stddev" in q_low
                ):
                    continue
                clean_filters.append(f)
            plan.filters = clean_filters
        plan.group_by = []
        plan.aggregations = []
    elif any(k in q_low for k in ("outlier", "outliers", "anomaly", "anomalies", "extreme", "deviations from the mean")) and (target_metric_col or num_col_names):
        m = target_metric_col or (list(num_col_names.values())[0] if num_col_names else None)
        thresh = 3.0
        thresh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:stddev|standard deviation|standard deviations|sigma|z-?score)", q_low)
        if thresh_match:
            try:
                thresh = float(thresh_match.group(1))
            except ValueError:
                thresh = 3.0
        part_cols = []
        for sc in str_cols:
            if f"by {sc.lower()}" in q_low or f"for each {sc.lower()}" in q_low or f"per {sc.lower()}" in q_low:
                part_cols.append(sc)
        filter_only = not any(k in q_low for k in ("all rows", "don't filter", "flag all", "keep all"))
        plan.outlier_analysis = OutlierSpec(
            column=m,
            threshold=thresh,
            partition_by=part_cols,
            filter_outliers_only=filter_only,
        )
        plan.group_by = []
        plan.aggregations = []

    # 6. Align OrderBy for superlatives and explicit ordering
    compound_desc = (
        "highest to lowest", "greatest to least", "largest to smallest",
        "best to worst", "most to least", "top to bottom", "descending",
        "desc", "highest first", "latest to earliest", "newest to oldest"
    )
    compound_asc = (
        "lowest to highest", "least to greatest", "smallest to largest",
        "worst to best", "least to most", "bottom to top", "ascending",
        "asc", "lowest first", "earliest to latest", "oldest to newest"
    )

    is_desc = None
    if any(k in q_low for k in compound_desc):
        is_desc = True
    elif any(k in q_low for k in compound_asc):
        is_desc = False
    elif has_superlative:
        is_desc = not any((re.search(r"(?<!at\s)\bleast\b", q_low) if w == "least" else re.search(rf"\b{w}\b", q_low)) for w in superlative_asc)
    elif any((re.search(r"(?<!at\s)\bleast\b", q_low) if w == "least" else re.search(rf"\b{w}\b", q_low)) for w in ("lowest", "least", "asc", "ascending", "lowest first")):
        is_desc = False
    elif any(w in q_low for w in ("highest", "desc", "descending", "highest first")):
        is_desc = True

    has_ranking_intent = bool(
        (has_superlative and n_match)
        or any((re.search(r"(?<!at\s)\bleast\b", q_low) if w == "least" else re.search(rf"\b{w}\b", q_low)) for w in ("top", "bottom", "highest", "lowest", "best", "worst", "least", "most"))
    )
    is_explicit_recompute = bool(
        is_mutation and has_superlative and any(w in q_low for w in ("instead", "now", "switch", "change to", "lowest", "bottom", "highest", "top"))
    )

    if is_desc is not None:
        target_alias = None
        if plan.aggregations:
            target_alias = plan.aggregations[0].alias or (f"{plan.aggregations[0].function.value}_{plan.aggregations[0].column}" if plan.aggregations[0].column else plan.aggregations[0].column)
        elif target_metric_col:
            target_alias = target_metric_col if is_raw_row_query else f"sum_{target_metric_col}"
        if target_alias:
            plan.order_by = [OrderByClause(target=target_alias, descending=is_desc)]
            if has_ranking_intent and (n_match or has_superlative):
                if not is_mutation or is_explicit_recompute:
                    plan.selection_order_by = [OrderByClause(target=target_alias, descending=is_desc)]
                elif not plan.selection_order_by:
                    is_sort_mutation = any(k in q_low for k in ("sort", "order", "reverse", "flip", "invert", "ascending", "descending"))
                    if not is_sort_mutation:
                        plan.selection_order_by = [OrderByClause(target=target_alias, descending=is_desc)]
        elif plan.order_by:
            plan.order_by[0].descending = is_desc
            if has_ranking_intent and (n_match or has_superlative):
                if not is_mutation or is_explicit_recompute:
                    plan.selection_order_by = [OrderByClause(target=plan.order_by[0].target, descending=is_desc)]
                elif not plan.selection_order_by:
                    is_sort_mutation = any(k in q_low for k in ("sort", "order", "reverse", "flip", "invert", "ascending", "descending"))
                    if not is_sort_mutation:
                        plan.selection_order_by = [OrderByClause(target=plan.order_by[0].target, descending=is_desc)]
    else:
        # Ordinary grouped query without explicit sorting
        if not is_mutation and not any(k in q_low for k in ("sort", "order", "rank")):
            plan.order_by = []
        elif is_mutation and has_explicit_breakdown and not any(k in q_low for k in ("sort", "order", "rank")):
            plan.order_by = []

    if has_explicit_breakdown and not has_ranking_intent and not n_match:
        plan.limit = 50
        plan.selection_order_by = []

    # Reconcile sort targets against available projections / aliases to prevent compiler errors
    valid_sort_targets = set()
    for gb in plan.group_by:
        valid_sort_targets.add(gb.column.lower())
    for agg in plan.aggregations:
        if agg.alias:
            valid_sort_targets.add(agg.alias.lower())
        if agg.column:
            valid_sort_targets.add(agg.column.lower())
            valid_sort_targets.add(f"{agg.function.value}_{agg.column}".lower())
    for dm in plan.derived_metrics:
        if dm.alias:
            valid_sort_targets.add(dm.alias.lower())
    if is_raw_row_query:
        for c in metadata.profile.columns:
            valid_sort_targets.add(c.name.lower())

    primary_agg_alias = (
        plan.aggregations[0].alias or (f"{plan.aggregations[0].function.value}_{plan.aggregations[0].column}" if plan.aggregations[0].column else plan.aggregations[0].column)
    ) if plan.aggregations else None

    if plan.order_by:
        aligned_order_by = []
        for ob in plan.order_by:
            if ob.target.lower() in valid_sort_targets:
                aligned_order_by.append(ob)
            elif primary_agg_alias:
                ob.target = primary_agg_alias
                aligned_order_by.append(ob)
        plan.order_by = aligned_order_by

    if plan.selection_order_by:
        aligned_sob = []
        for sob in plan.selection_order_by:
            if sob.target.lower() in valid_sort_targets:
                aligned_sob.append(sob)
            elif primary_agg_alias:
                sob.target = primary_agg_alias
                aligned_sob.append(sob)
        plan.selection_order_by = aligned_sob

    # 7. Deterministic chart intent and explicit chart type detection
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

    if _has_explicit_clear_filter_directive(q_low) and not any(w in q_low for w in chart_keywords):
        plan.requires_chart = False
        plan.chart_type = None

    # 8. Clean hallucinated correlations
    corr_keywords = ("correlat", "relationship between", "association between")
    if not is_mutation and not any(k in q_low for k in corr_keywords):
        plan.correlations = []
    elif any(k in q_low for k in corr_keywords):
        valid_corrs = []
        for c in plan.correlations:
            cx = num_col_names.get(c.column_x.lower())
            cy = num_col_names.get(c.column_y.lower())
            if cx and cy:
                c.column_x = cx
                c.column_y = cy
                valid_corrs.append(c)
        plan.correlations = valid_corrs

    # 9. Clean hallucinated or malformed derived metrics
    ratio_keywords = ("margin", "ratio", "rate", "percentage", "per ", "divided by", "share", "proportion", "%")
    if not is_mutation and not any(k in q_low for k in ratio_keywords):
        plan.derived_metrics = []
    elif any(k in q_low for k in ratio_keywords):
        valid_dms = []
        for dm in plan.derived_metrics:
            num_clean = dm.numerator
            if num_clean.lower() not in num_col_names:
                for prefix in ("total_", "sum_", "avg_"):
                    if num_clean.lower().startswith(prefix) and num_clean[len(prefix):].lower() in num_col_names:
                        num_clean = num_col_names[num_clean[len(prefix):].lower()]
                        break
            else:
                num_clean = num_col_names[num_clean.lower()]

            den_clean = dm.denominator
            if den_clean.lower() not in num_col_names:
                for prefix in ("total_", "sum_", "avg_"):
                    if den_clean.lower().startswith(prefix) and den_clean[len(prefix):].lower() in num_col_names:
                        den_clean = num_col_names[den_clean[len(prefix):].lower()]
                        break
            else:
                den_clean = num_col_names[den_clean.lower()]

            if num_clean.lower() in num_col_names and den_clean.lower() in num_col_names:
                dm.numerator = num_clean
                dm.denominator = den_clean
                if "margin" in q_low:
                    dm.alias = "profit_margin_pct"
                elif any(w in q_low for w in ("pct", "percentage", "%")) and dm.alias and not dm.alias.endswith("_pct"):
                    dm.alias = f"{dm.alias}_pct"
                valid_dms.append(dm)

        if not valid_dms and any(k in q_low for k in ("margin", "profit margin")):
            num_candidates = [c for c in num_col_names.values() if "profit" in c.lower()]
            den_candidates = [c for c in num_col_names.values() if "revenue" in c.lower() or "sales" in c.lower()]
            if num_candidates and den_candidates:
                valid_dms.append(
                    DerivedMetric(
                        operation=DerivedMetricOperation.PERCENTAGE,
                        numerator=num_candidates[0],
                        denominator=den_candidates[0],
                        multiplier=100.0,
                        alias="profit_margin_pct",
                    )
                )
        plan.derived_metrics = valid_dms

    plan.dataset_id = metadata.dataset_id
    return AnalysisPlan.model_validate(plan.model_dump())


def generate_analysis_plan(
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
    previous_plan: Optional[AnalysisPlan] = None,
) -> AnalysisPlan:
    """
    Prompts the LLM to generate an AnalysisPlan, validates with Pydantic,
    and enforces server-authoritative dataset_id and deterministic AST invariants.
    If previous_plan is provided, applies an incremental PlanMutation and normalizes the result.
    """
    q_low = question.lower()

    # Speculative / predictive guard: SQL aggregation engine cannot perform forecasting or predictive models (ANA-12)
    # Querying existing forecast/projection columns is allowed, but requests to generate future predictions
    # or queries referencing non-existent forecast/prediction/stock columns are rejected.
    has_predict_verb = bool(re.search(r"\b(?:predict|extrapolate)\b", q_low)) or "machine learning model" in q_low or "projection for 20" in q_low
    has_forecast_term = bool(re.search(r"\bforecast\b", q_low))
    has_projection_term = bool(re.search(r"\bprojection\b", q_low))
    has_stock_term = "stock price" in q_low or "stock prediction" in q_low

    col_names_low = [c.name.lower() for c in metadata.profile.columns]
    has_forecast_col = any("forecast" in c for c in col_names_low)
    has_predict_col = any("predict" in c for c in col_names_low)
    has_projection_col = any("projection" in c for c in col_names_low)
    has_stock_price_col = any("stock_price" in c or ("stock" in c and "price" in c) for c in col_names_low)

    if has_stock_term and not has_stock_price_col:
        raise PlanValidationError(
            f"Speculative or predictive modeling queries ('{question}') are not supported by the SQL analytics engine."
        )
    if has_predict_verb and not has_predict_col:
        raise PlanValidationError(
            f"Speculative or predictive modeling queries ('{question}') are not supported by the SQL analytics engine."
        )
    if has_forecast_term and not has_forecast_col:
        raise PlanValidationError(
            f"Speculative or predictive modeling queries ('{question}') are not supported by the SQL analytics engine."
        )
    if has_projection_term and not has_projection_col:
        raise PlanValidationError(
            f"Speculative or predictive modeling queries ('{question}') are not supported by the SQL analytics engine."
        )

    # Explicit unsupported metric / column rejection (ANA-11):
    # Detect when user explicitly requests a named metric/measure that is absent from schema
    all_cols_low = {c.name.lower() for c in metadata.profile.columns}
    unsupported_metric_terms = (
        "satisfaction", "rating", "score", "csat", "nps", "sentiment", "churn",
        "retention rate", "conversion rate", "feedback score", "net promoter"
    )
    has_unsupported_term = any(re.search(rf"\b{re.escape(term)}\b", q_low) for term in unsupported_metric_terms)
    if has_unsupported_term and not any(term in c for term in unsupported_metric_terms for c in all_cols_low):
        raise ColumnNotFoundError(
            f"Requested metric in query '{question}' is not available in dataset schema {[c.name for c in metadata.profile.columns]}."
        )

    if previous_plan is not None:
        mutation = plan_mutation(
            question=question,
            metadata=metadata,
            previous_plan=previous_plan,
            resolved_metric=resolved_metric,
        )
        mutated_plan = apply_plan_mutation(previous_plan, mutation, metadata=metadata)
        if mutation.mutation_type in (PlanMutationType.TOGGLE_CHART, PlanMutationType.TOGGLE_HEATMAP):
            target_dim_check = _resolve_target_dimension(question, metadata)
            if not (target_dim_check and _has_explicit_breakdown_directive(question, target_dim_check)):
                mutated_plan.dataset_id = metadata.dataset_id
        is_mut = (mutation.mutation_type != PlanMutationType.NEW_QUERY)
        return _normalize_and_align_plan(mutated_plan, question, metadata, resolved_metric, is_mutation=is_mut)

    raw_triggers = (
        "individual", "raw rows", "raw records", "each row", "without grouping",
        "without aggregation", "list all rows", "list rows", "list transactions",
        "show individual", "list individual", "every row", "every transaction"
    )
    is_raw_row_query = any(k in q_low for k in raw_triggers)

    # Deterministic comparison filter extraction (e.g. "at least 100 orders", "orders >= 100", "where profit is at least 3000"):
    comp_filter = _extract_comparison_filter(question, metadata)

    # Deterministic fast intercept for comparison filter queries (ANA-10):
    complex_triggers = ("margin", "ratio", "correlation", "cross-tab", "crosstab", "pivot", "!=", "and ", "or ")
    has_complex = any(k in q_low for k in complex_triggers)
    if comp_filter is not None and not has_complex and not is_raw_row_query:
        fast_dim = _resolve_target_dimension(question, metadata)
        fast_metric = _resolve_target_metric(question, metadata, resolved_metric, fallback_to_first=False) or comp_filter.column
        if fast_dim and fast_metric:
            fast_plan_dict = _fallback_heuristic_plan(question, metadata, resolved_metric)
            fast_plan_dict["filters"] = [comp_filter.model_dump()]
            fast_plan_dict["dataset_id"] = metadata.dataset_id
            plan = AnalysisPlan.model_validate(fast_plan_dict)
            return _normalize_and_align_plan(plan, question, metadata, resolved_metric)

    # Deterministic fast intercept for unambiguous ranking queries:
    # If the user asks a clear ranking query (e.g. "top 5 products by sales", "bottom 3 departments by score")
    # without complex filtering, resolve it deterministically to guarantee 100% repeatability (DI-01).
    ranking_keywords = ("top", "highest", "best", "most", "lowest", "worst", "bottom", "least", "smallest")
    has_ranking = any(re.search(r"(?<!at\s)\bleast\b" if w == "least" else rf"\b{w}\b", q_low) for w in ranking_keywords)
    filter_triggers = ("where ", "filter to", "filter by", "only in", "only for", "status is", "status ==", "margin", "ratio", "correlation", "cross-tab", "crosstab", "pivot", "at most", "<=", "<", "!=")
    has_filter_or_complex = any(k in q_low for k in filter_triggers)

    if has_ranking and not has_filter_or_complex and not is_raw_row_query:
        fast_dim = _resolve_target_dimension(question, metadata)
        fast_metric = _resolve_target_metric(question, metadata, resolved_metric, fallback_to_first=False)
        if fast_dim and fast_metric:
            fast_plan_dict = _fallback_heuristic_plan(question, metadata, resolved_metric)
            fast_plan_dict["dataset_id"] = metadata.dataset_id
            plan = AnalysisPlan.model_validate(fast_plan_dict)
            return _normalize_and_align_plan(plan, question, metadata, resolved_metric)

    # Deterministic fast intercept for unambiguous profit margin queries:
    # Handles queries like "What is the profit margin percentage by category as a bar chart?"
    # when the dataset contains suitable numeric profit and revenue/sales columns without complex filtering.
    derived_metric_triggers = ("profit margin", "margin percentage", "margin percent", "ratio of profit")
    has_derived_intent = any(k in question.lower() for k in derived_metric_triggers)
    if has_derived_intent and comp_filter is None:
        has_filter_clause = any(k in question.lower() for k in ("where ", "filter to", "filter by", "only in", "only for", "status is", "status ==", "at most", "<=", "<", "!=", ">"))
        if not has_filter_clause:
            num_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "numeric"]
            has_profit = any("profit" in c.lower() for c in num_cols)
            has_revenue = any("revenue" in c.lower() or "sales" in c.lower() for c in num_cols)
            if has_profit and has_revenue:
                fast_plan_dict = _fallback_heuristic_plan(question, metadata, resolved_metric)
                fast_plan_dict["dataset_id"] = metadata.dataset_id
                plan = AnalysisPlan.model_validate(fast_plan_dict)
                return _normalize_and_align_plan(plan, question, metadata, resolved_metric)

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

        # Ensure list fields default to empty lists if LLM provided null
        for list_field in (
            "filters", "group_by", "aggregations", "derived_metrics",
            "correlations", "rolling_windows", "cumulative_metrics",
            "quantiles", "order_by", "selection_order_by"
        ):
            if list_field in parsed_json and parsed_json[list_field] is None:
                parsed_json[list_field] = []

        # Sanitize common LLM quirks before Pydantic validation:
        q_low_norm = question.lower()
        if "filters" in parsed_json and isinstance(parsed_json["filters"], list):
            clean_filters = []
            for f in parsed_json["filters"]:
                if not isinstance(f, dict):
                    continue
                v_str = str(f.get("value", "")).lower().strip()
                c_str = str(f.get("column", "")).lower().strip()
                if v_str == c_str or v_str in ("all", "*", "any"):
                    continue
                # Contextually safe rejection of chart/presentation terms mistakenly extracted as filter values
                if v_str in ("bar chart", "pie chart", "line chart", "scatter plot", "bar graph", "line graph", "histogram"):
                    continue
                if c_str not in q_low_norm and v_str not in q_low_norm:
                    continue
                op = str(f.get("operator", "")).lower().strip()
                if v_str in ("null", "none") or f.get("value") is None:
                    if any(w in q_low_norm for w in ("null", "missing", "empty", "nan", "blank", "none")):
                        if op in ("!=", "<>", "is_not_null"):
                            f["operator"] = "is_not_null"
                            f["value"] = None
                        else:
                            f["operator"] = "is_null"
                            f["value"] = None
                    else:
                        continue
                if "at least" in q_low_norm and op in (">", "gt"):
                    f["operator"] = ">="
                clean_filters.append(f)

            if comp_filter is not None:
                comp_col_low = comp_filter.column.lower()
                existing_comp = False
                for f in clean_filters:
                    if str(f.get("column", "")).lower() == comp_col_low:
                        existing_comp = True
                        if "at least" in q_low_norm:
                            f["operator"] = ">="
                        break
                if not existing_comp:
                    clean_filters.append(comp_filter.model_dump())

            parsed_json["filters"] = clean_filters

        # Fix aggregation if LLM put a string column in sum/avg/min/max or extra fields
        if "aggregations" in parsed_json and isinstance(parsed_json["aggregations"], list):
            valid_agg_keys = {"column", "function", "alias", "quantile"}
            for agg in parsed_json["aggregations"]:
                if isinstance(agg, dict):
                    if "order_by" in agg:
                        ob = agg.pop("order_by")
                        if not parsed_json.get("order_by") and isinstance(ob, str):
                            parsed_json["order_by"] = [{"target": ob, "descending": True}]
                    extra_keys = [k for k in agg.keys() if k not in valid_agg_keys]
                    for k in extra_keys:
                        agg.pop(k, None)

                    fn = str(agg.get("function", "")).lower().strip()
                    if fn == "mean":
                        fn = "avg"
                    agg["function"] = fn
                    col_str = str(agg.get("column", "")).lower()
                    if col_str in str_col_names and fn in ("sum", "avg", "min", "max"):
                        matching_metric = _resolve_target_metric(question, metadata, resolved_metric)
                        agg["column"] = matching_metric or (list(num_col_names.values())[0] if num_col_names else None)

        if "outlier_analysis" in parsed_json and isinstance(parsed_json["outlier_analysis"], dict):
            oa = parsed_json["outlier_analysis"]
            if oa.get("partition_by") is None:
                oa["partition_by"] = []
            elif isinstance(oa.get("partition_by"), list):
                oa["partition_by"] = [
                    p for p in oa["partition_by"]
                    if isinstance(p, str) and p.lower() in q_low_norm
                ]

        parsed_json["dataset_id"] = metadata.dataset_id
        plan = AnalysisPlan.model_validate(parsed_json)
    except Exception as e:
        logger.warning(f"LLM planner generation or validation failed: {e}. Attempting heuristic fallback.")
        fallback_json = _fallback_heuristic_plan(question, metadata, resolved_metric)
        if comp_filter is not None and not fallback_json.get("filters"):
            fallback_json["filters"] = [comp_filter.model_dump()]
        fallback_json["dataset_id"] = metadata.dataset_id
        plan = AnalysisPlan.model_validate(fallback_json)

    return _normalize_and_align_plan(plan, question, metadata, resolved_metric)


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

    target_metric = _resolve_target_metric(question, metadata, resolved_metric)
    if not target_metric and num_cols:
        target_metric = num_cols[0]

    date_cols = [c.name for c in metadata.profile.columns if classify_dtype(c.dtype, c.name) == "datetime"]

    # Phase 9: Cross-tab / Pivot / Matrix / Heatmap fallback
    pivot_keywords = ("cross-tab", "crosstab", "cross tab", "pivot", "matrix", "heatmap")
    is_pivot_intent = any(k in lowered for k in pivot_keywords) or (
        len(str_cols) >= 2 and any(k in lowered for k in ("by ", "across ", " and ")) and sum(1 for sc in str_cols if sc.lower() in lowered) >= 2
    )
    if is_pivot_intent and len(str_cols) >= 2:
        dim_candidates = [sc for sc in str_cols if sc.lower() in lowered]
        if len(dim_candidates) < 2:
            dim_candidates = str_cols[:2]
        row_dim = dim_candidates[0]
        col_dim = dim_candidates[1]

        m = target_metric or (num_cols[0] if num_cols else "id")
        agg_fn = "sum"
        if "avg" in lowered or "average" in lowered or "mean" in lowered:
            agg_fn = "avg"
        elif "count" in lowered or not num_cols:
            agg_fn = "count"
        elif "max" in lowered:
            agg_fn = "max"
        elif "min" in lowered:
            agg_fn = "min"

        is_heatmap = "heatmap" in lowered
        return {
            "dataset_id": metadata.dataset_id,
            "pivot": {
                "row_dimension": row_dim,
                "column_dimension": col_dim,
                "metric": m,
                "aggregation": agg_fn,
                "include_totals": True,
            },
            "requires_chart": is_heatmap,
            "chart_type": "heatmap" if is_heatmap else None,
        }

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
            if any(k in lowered for k in ("chart", "plot", "bar")):
                plan_dict["requires_chart"] = True
                plan_dict["chart_type"] = "bar"
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

    # Raw-row query fallback
    raw_triggers = (
        "individual", "raw rows", "raw records", "each row", "without grouping",
        "without aggregation", "list all rows", "list rows", "list transactions",
        "show individual", "list individual", "every row", "every transaction"
    )
    if any(k in lowered for k in raw_triggers):
        res = {
            "dataset_id": metadata.dataset_id,
            "group_by": [],
            "aggregations": [],
            "limit": 50,
        }
        if target_metric:
            res["order_by"] = [{"target": target_metric, "descending": False}]
        return res

    # Record count fallback
    is_record_count = (
        any(k in lowered for k in ("record count", "number of records", "count of records", "count records", "how many records", "row count", "count of rows", "number of rows"))
        or (re.search(r"\b(?:count|how many)\b", lowered) and not target_metric)
    )
    if is_record_count:
        target_dims = _resolve_target_dimensions(question, metadata)
        if target_dims:
            return {
                "dataset_id": metadata.dataset_id,
                "group_by": [{"column": c} for c in target_dims],
                "aggregations": [{"column": None, "function": "count", "alias": "count_records"}],
                "limit": 50,
            }
        return {
            "dataset_id": metadata.dataset_id,
            "aggregations": [{"column": None, "function": "count", "alias": "count_records"}],
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

    # Ranking query: e.g. "top products", "which region had highest revenue", "bottom 5 products by sales"
    ranking_keywords = ("top", "highest", "best", "most", "lowest", "worst", "bottom", "least", "smallest")
    has_ranking_in_fallback = any(
        (re.search(r"(?<!at\s)\bleast\b", lowered) if k == "least" else k in lowered)
        for k in ranking_keywords
    )
    if has_ranking_in_fallback and str_cols and target_metric:
        group_col = str_cols[0]
        for c in str_cols:
            if c.lower() in lowered or c.lower().rstrip("s") in lowered or (c.lower().endswith("y") and c.lower()[:-1] + "ies" in lowered):
                group_col = c
                break
        compound_desc = ("highest to lowest", "greatest to least", "largest to smallest", "best to worst", "most to least", "top to bottom", "descending", "desc", "highest first", "latest to earliest", "newest to oldest")
        compound_asc = ("lowest to highest", "least to greatest", "smallest to largest", "worst to best", "least to most", "bottom to top", "ascending", "asc", "lowest first", "earliest to latest", "oldest to newest")
        if any(k in lowered for k in compound_desc):
            desc = True
        elif any(k in lowered for k in compound_asc):
            desc = False
        else:
            desc = not any((re.search(r"(?<!at\s)\bleast\b", lowered) if k == "least" else k in lowered) for k in ("lowest", "worst", "least", "bottom", "smallest"))

        top_match = re.search(
            rf"(?:\b(?:top|bottom|highest|lowest|best|worst|first|last)\b|(?<!at\s)\bleast\b)\s+({WORD_NUM_PATTERN})\b",
            lowered,
        )
        if not top_match:
            top_match = re.search(
                rf"\b({WORD_NUM_PATTERN})\s+(?:top|bottom|highest|lowest|best|worst|first|last|(?<!at\s)least)\b",
                lowered,
            )
        limit_val = _parse_int_or_word(top_match.group(1)) if top_match else 10
        wants_avg = _has_positive_intent(lowered, ("average", "avg", "mean"))
        wants_sum = _has_positive_intent(lowered, ("total", "sum"))
        agg_fn = "avg" if (wants_avg and not wants_sum) else "sum"
        alias = f"{agg_fn}_{target_metric}"
        return {
            "dataset_id": metadata.dataset_id,
            "group_by": [{"column": group_col}],
            "aggregations": [{"column": target_metric, "function": agg_fn, "alias": alias}],
            "order_by": [{"target": alias, "descending": desc}],
            "selection_order_by": [{"target": alias, "descending": desc}],
            "limit": limit_val,
        }

    # Grouped query by dimension (e.g. "salary by department", "revenue by region", "average profit by region")
    target_dims = _resolve_target_dimensions(question, metadata)
    group_col = None
    for sc in str_cols:
        if sc.lower() in lowered or f"by {sc.lower()}" in lowered:
            group_col = sc
    group_cols = target_dims if target_dims else ([group_col] if group_col else [])
    if re.search(r"\b(?:across\s+all|in\s+all|for\s+all|across\s+the\s+entire|across\s+the\s+whole)\b", lowered):
        group_cols = []

    if group_cols and (target_metric or num_cols):
        m = target_metric or num_cols[0]
        wants_min = _has_positive_intent(lowered, ("min", "minimum", "lowest"))
        wants_max = _has_positive_intent(lowered, ("max", "maximum", "highest"))
        wants_avg = _has_positive_intent(lowered, ("average", "avg", "mean"))
        wants_sum = _has_positive_intent(lowered, ("total", "sum"))

        if wants_min and wants_max:
            aggs = [
                {"column": m, "function": "min", "alias": f"min_{m}"},
                {"column": m, "function": "max", "alias": f"max_{m}"},
            ]
        else:
            agg_fn = "avg" if (wants_avg and not wants_sum) else "sum"
            alias = f"{agg_fn}_{m}"
            aggs = [{"column": m, "function": agg_fn, "alias": alias}]

        return {
            "dataset_id": metadata.dataset_id,
            "group_by": [{"column": c} for c in group_cols],
            "aggregations": aggs,
            "limit": 50,
        }

    # Total / Average query
    wants_avg = _has_positive_intent(lowered, ("average", "avg", "mean"))
    wants_sum = _has_positive_intent(lowered, ("total", "sum", "revenue", "profit"))
    if wants_avg and not wants_sum:
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

    if wants_sum:
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
        "aggregations": [{"column": None, "function": "count", "alias": "count_rows"}],
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

    # Fast deterministic intercept for pure chart follow-ups
    chart_followup = _detect_pure_chart_followup(question, metadata, previous_plan)
    if chart_followup is not None:
        req_chart, ctype = chart_followup
        m_type = PlanMutationType.TOGGLE_HEATMAP if ctype == "heatmap" else PlanMutationType.TOGGLE_CHART
        return PlanMutation(
            mutation_type=m_type,
            chart_type=ctype if req_chart else None,
            requires_chart=req_chart,
        )

    # Fast deterministic intercept for pure table presentation follow-ups
    if _detect_pure_table_followup(question, metadata, previous_plan):
        return PlanMutation(
            mutation_type=PlanMutationType.TOGGLE_CHART,
            chart_type=None,
            requires_chart=False,
        )

    # Explicit new query check (e.g. "what is...", "can you plot...", "calculate profit margin...")
    ranking_keywords = ("top", "highest", "best", "most", "lowest", "worst", "bottom", "least", "smallest")
    has_ranking_query = any(re.search(r"(?<!at\s)\bleast\b" if w == "least" else rf"\b{w}\b", q_low) for w in ranking_keywords)
    target_dim_found = _resolve_target_dimension(question, metadata)
    prev_dim = previous_plan.group_by[0].column if previous_plan and previous_plan.group_by else None

    is_explicit_new_query = (
        any(q_low.startswith(p) for p in ("what is", "what are", "how many", "can you", "could you", "calculate", "find", "tell me", "show me", "give me", "list", "display", "get"))
        or any(k in q_low for k in ("margin", "profit margin", "ratio", "correlation", "cross-tab", "crosstab", "pivot table"))
        or (has_ranking_query and target_dim_found and target_dim_found != prev_dim)
    ) and not any(k in q_low for k in ("filter that", "filter this", "filter to", "sort that", "order that", "now group", "also by", "instead of", "switch to", "clear filter", "reset"))
    if is_explicit_new_query:
        fresh = _fallback_heuristic_plan(question, metadata, resolved_metric)
        fresh["dataset_id"] = metadata.dataset_id
        new_plan_obj = AnalysisPlan.model_validate(fresh)
        return PlanMutation(
            mutation_type=PlanMutationType.NEW_QUERY,
            new_plan=new_plan_obj,
        )

    # 1. RESET_ANALYSIS / REMOVE_FILTER
    if _has_explicit_clear_filter_directive(q_low) or any(k in q_low for k in ("clear filter", "clear filters", "reset filter", "reset filters", "reset analysis", "clear all filters", "start over")):
        top_match = re.search(r"(?:top|limit|first)\s+(\d+)", q_low)
        limit_val = int(top_match.group(1)) if top_match else None

        target_dim = _resolve_target_dimension(question, metadata)
        target_metric = _resolve_target_metric(question, metadata, resolved_metric, fallback_to_first=False)

        gb_delta = GroupByDelta(column=target_dim) if target_dim else None
        metric_delta = MetricDelta(column=target_metric, function=AggregationFunction.SUM, alias=f"sum_{target_metric}") if target_metric else None

        return PlanMutation(
            mutation_type=PlanMutationType.RESET_ANALYSIS,
            group_by_delta=gb_delta,
            metric_delta=metric_delta,
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

    # Phase 9: TOGGLE_HEATMAP
    if "heatmap" in q_low:
        if any(w in q_low for w in ("hide", "remove", "disable", "no heatmap")):
            return PlanMutation(mutation_type=PlanMutationType.TOGGLE_HEATMAP, requires_chart=False)
        return PlanMutation(
            mutation_type=PlanMutationType.TOGGLE_HEATMAP,
            chart_type="heatmap",
            requires_chart=True,
        )

    # Phase 9: CHANGE_PIVOT_COLUMNS
    if any(k in q_low for k in ("pivot by", "cross-tab by", "switch columns to", "switch rows to", "change columns to", "change rows to")):
        target_dims = [sc for sc in str_cols if sc.lower() in q_low]
        prev_p = previous_plan.pivot
        new_row = prev_p.row_dimension if prev_p else (str_cols[0] if str_cols else None)
        new_col = prev_p.column_dimension if prev_p else (str_cols[1] if len(str_cols) > 1 else None)
        if "switch columns to" in q_low and target_dims:
            new_col = target_dims[0]
        elif "switch rows to" in q_low and target_dims:
            new_row = target_dims[0]
        elif len(target_dims) >= 2:
            new_row, new_col = target_dims[0], target_dims[1]
        elif len(target_dims) == 1:
            new_col = target_dims[0]

        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_PIVOT_COLUMNS,
            pivot_delta=PivotDelta(
                row_dimension=new_row,
                column_dimension=new_col,
                metric=prev_p.metric if prev_p else (num_cols[0] if num_cols else "id"),
            ),
        )

    # 4. TOGGLE_CHART / VISUALIZATION / TABLE
    chart_keywords = ("chart", "plot", "graph", "visualize", "visualization")
    table_presentation_keywords = ("table", "tabular", "grid")
    target_dim_for_breakdown = _resolve_target_dimension(question, metadata)
    has_explicit_breakdown_intent = bool(target_dim_for_breakdown and _has_explicit_breakdown_directive(question, target_dim_for_breakdown))
    if not has_explicit_breakdown_intent and (any(k in q_low for k in chart_keywords) or any(k in q_low for k in table_presentation_keywords)):
        if any(w in q_low for w in ("hide", "remove", "disable", "no chart", "as a table", "as table", "to a table", "to table", "in a table", "in table", "table view", "table format")):
            return PlanMutation(mutation_type=PlanMutationType.TOGGLE_CHART, requires_chart=False)
        if any(k in q_low for k in chart_keywords):
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
        desc = not any((re.search(r"(?<!at\s)\bleast\b", q_low) if k == "least" else k in q_low) for k in ("lowest", "least", "asc", "ascending"))
        target = None
        # Check active aggregations first to align with projected alias (e.g., sum_revenue)
        if previous_plan.aggregations:
            for agg in previous_plan.aggregations:
                if (agg.column and agg.column.lower() in q_low) or (agg.alias and agg.alias.lower() in q_low):
                    target = agg.alias or (f"{agg.function.value}_{agg.column}" if agg.column else agg.column)
                    break
        if not target and previous_plan.group_by:
            for gb in previous_plan.group_by:
                if gb.column.lower() in q_low:
                    target = gb.column
                    break
        if not target:
            for col in metadata.profile.columns:
                if col.name.lower() in q_low:
                    target = col.name
                    break
        if not target and previous_plan.aggregations:
            target = previous_plan.aggregations[0].alias or (f"{previous_plan.aggregations[0].function.value}_{previous_plan.aggregations[0].column}" if previous_plan.aggregations[0].column else previous_plan.aggregations[0].column)
        if not target and previous_plan.group_by:
            target = previous_plan.group_by[0].column
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_SORT,
            sort_delta=SortDelta(target=target or "total", descending=desc),
        )

    # 6. CHANGE_LIMIT (standalone: e.g. "show top 5", "bottom 3", "limit to 10")
    limit_match = re.search(rf"^(?:show\s+)?(?:top|bottom|limit|first|last)\s+({WORD_NUM_PATTERN})$", q_low)
    if not limit_match:
        limit_match = re.search(rf"^(?:show\s+)?({WORD_NUM_PATTERN})\s+(?:top|bottom|limit|first|last)$", q_low)
    if limit_match:
        limit_val = _parse_int_or_word(limit_match.group(1))
        if limit_val is not None:
            return PlanMutation(
                mutation_type=PlanMutationType.CHANGE_LIMIT,
                limit_delta=limit_val,
            )

    # 7. CHANGE_METRIC (e.g. "show profit instead", "switch to revenue", "now average price")
    metric_delta = _detect_pure_metric_followup(question, metadata, previous_plan)
    if metric_delta:
        return PlanMutation(
            mutation_type=PlanMutationType.CHANGE_METRIC,
            metric_delta=metric_delta,
        )

    # 8. GROUP_BY MUTATIONS (CHANGE_GROUP_BY, ADD_GROUP_BY, REMOVE_GROUP_BY)
    groupby_mutation = _detect_pure_groupby_followup(question, metadata, previous_plan)
    if groupby_mutation:
        return groupby_mutation
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

        # Check for quoted value first (e.g. 'North' or "Electronics")
        quote_match = re.search(r"['\"]([^'\"]+)['\"]", question)
        if quote_match:
            filter_val = quote_match.group(1).strip()
        else:
            eq_match = re.search(r"(?:==|=|is)\s*([A-Za-z0-9_\s-]+)", question, re.IGNORECASE)
            if eq_match:
                filter_val = eq_match.group(1).strip()
            else:
                v_match = re.search(r"(?:filter\s+to|filter\s+by|only\s+in|only|where)\s+([A-Za-z0-9_\s-]+)", question, re.IGNORECASE)
                if v_match:
                    raw_v = v_match.group(1).strip()
                    raw_v = re.sub(r"^(?:in|the|for|status|category|region|country)\s+", "", raw_v, flags=re.IGNORECASE).strip()
                    if raw_v:
                        filter_val = raw_v
                if not filter_val:
                    m_for_only = re.search(r"\bfor\s+([A-Za-z0-9_\s-]+?)\s+only\b", question, re.IGNORECASE)
                    if m_for_only:
                        filter_val = m_for_only.group(1).strip()
                if not filter_val:
                    m_only = re.search(r"\b([A-Za-z0-9_\s-]+?)\s+only\b", question, re.IGNORECASE)
                    if m_only:
                        val_candidate = m_only.group(1).strip()
                        if val_candidate.lower() not in ("filter", "show", "sales", "group", "top"):
                            filter_val = val_candidate

        if not target_col:
            if previous_plan.pivot:
                target_col = previous_plan.pivot.row_dimension
            elif previous_plan.group_by:
                target_col = previous_plan.group_by[0].column
            elif str_cols:
                target_col = str_cols[0]

        if filter_val:
            target_col = _reconcile_filter_column(target_col or (str_cols[0] if str_cols else None), filter_val, metadata)

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
                mutation_prefix = f"Switched metric to {_format_clean_metric_label(target_metric)}. "

        # 4. CHANGE_GROUP_BY: Grouping dimension changed
        if not mutation_prefix and result.plan.group_by:
            prev_gb = [g.column for g in previous_plan.group_by]
            curr_gb = [g.column for g in result.plan.group_by]
            if prev_gb != curr_gb:
                dim_name = curr_gb[0]
                filter_ctx = ""
                if result.plan.filters:
                    filter_vals = [r for f in result.plan.filters for r in [_clean_filter_repr(f)] if r is not None]
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
                mutation_prefix = f"Sorted by {_format_clean_metric_label(curr_ob.target)} ({dir_label}). "

    # Phase 9: Cross-Tab / Pivot Table result
    if result.pivot_table is not None:
        pt = result.pivot_table
        if not pt.rows or not pt.columns:
            return f"{mutation_prefix}The cross-tabulation query completed, but no matching records were found."

        max_val = None
        max_cell = None
        for r_idx, r_name in enumerate(pt.rows):
            for c_idx, c_name in enumerate(pt.columns):
                v = pt.values[r_idx][c_idx]
                if v is not None and (max_val is None or v > max_val):
                    max_val = v
                    max_cell = (r_name, c_name)

        total_part = f", with grand total {pt.grand_total:,.2f}" if pt.grand_total is not None else ""
        if max_cell and max_val is not None:
            val_formatted = f"{max_val:,.2f}" if isinstance(max_val, float) else f"{max_val}"
            return (
                f"{mutation_prefix}Cross-tabulation of {pt.aggregation} {pt.metric} across {len(pt.rows)} {pt.row_dimension} values "
                f"and {len(pt.columns)} {pt.column_dimension} values. Leading cell observed in {max_cell[0]} × {max_cell[1]} "
                f"({val_formatted}){total_part}."
            )
        else:
            return (
                f"{mutation_prefix}Cross-tabulation of {pt.aggregation} {pt.metric} across {len(pt.rows)} {pt.row_dimension} values "
                f"and {len(pt.columns)} {pt.column_dimension} values{total_part}."
            )

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
    if result.plan.rolling_windows:
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
                f"{mutation_prefix}Across {len(rows)} observed intervals, the {w_size}-period rolling {rw.function.value} of '{rw.column}' "
                f"peaked at {peak_val:.2f} (at {peak_order}), with the latest value at {latest_val:.2f}."
            )
        return f"{mutation_prefix}Calculated {w_size}-period rolling {rw.function.value} for '{rw.column}' across {len(rows)} records."

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
    # Grouped / Ranked query (e.g. top regions by revenue)
    if result.plan.group_by:
        group_keys = [gb.column for gb in result.plan.group_by]
        metric_keys = [c for c in cols if c not in group_keys]
        group_dim = group_keys[0] if group_keys else "group"
        dim_low = group_dim.lower()

        # Pluralize dimension name
        if dim_low.endswith("y"):
            plural_dim = dim_low[:-1] + "ies"
        elif dim_low.endswith("s"):
            plural_dim = dim_low
        else:
            plural_dim = dim_low + "s"

        active_filter_ctx = ""
        if result.plan.filters:
            filter_vals = [r for f in result.plan.filters for r in [_clean_filter_repr(f)] if r is not None]
            if filter_vals:
                active_filter_ctx = f" in {', '.join(filter_vals)}"

        if metric_keys:
            primary_metric = metric_keys[0]

            def _get_num(r):
                v = r.get(primary_metric)
                if isinstance(v, (int, float)):
                    return v
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return float("-inf")

            def _fmt_val(v):
                if isinstance(v, float):
                    return f"{v:,.2f}"
                if isinstance(v, int):
                    return f"{v:,}"
                return str(v)

            is_lowest_query = False
            if result.plan and result.plan.selection_order_by and result.plan.selection_order_by[0].descending is False:
                is_lowest_query = True
            elif previous_plan and previous_plan.selection_order_by and previous_plan.selection_order_by[0].descending is False and not any(w in question.lower() for w in ("highest", "top", "best", "most")):
                is_lowest_query = True
            elif any((re.search(r"(?<!at\s)\bleast\b", question.lower()) if w == "least" else w in question.lower()) for w in ("lowest", "bottom", "least", "worst", "smallest")):
                is_lowest_query = True

            if is_lowest_query:
                sorted_rows = sorted(rows, key=_get_num)
                rank_word = "lowest"
            else:
                sorted_rows = sorted(rows, key=_get_num, reverse=True)
                rank_word = "highest"

            lead_row = sorted_rows[0]
            lead_entity = lead_row.get(group_dim, "First item")
            lead_val = lead_row.get(primary_metric)
            lead_val_str = _fmt_val(lead_val)

            metric_label = _format_clean_metric_label(primary_metric)

            if len(rows) > 1:
                second_row = sorted_rows[1]
                second_entity = second_row.get(group_dim, "Second item")
                second_val = second_row.get(primary_metric)
                second_val_str = _fmt_val(second_val)

                return (
                    f"{mutation_prefix}Across {len(rows)} {plural_dim}, "
                    f"{lead_entity} has the {rank_word} {metric_label}{active_filter_ctx} at {lead_val_str}, "
                    f"followed by {second_entity} at {second_val_str}."
                )
            else:
                return (
                    f"{mutation_prefix}Across 1 {dim_low}, "
                    f"{lead_entity} has {metric_label}{active_filter_ctx} of {lead_val_str}."
                )

        # Dimension listing without metric
        lead_entity = rows[0].get(group_dim, "First item")
        if len(rows) > 1:
            return f"{mutation_prefix}Across {len(rows)} {plural_dim}, leading {dim_low} is {lead_entity}."
        return f"{mutation_prefix}The query returned 1 {dim_low}: {lead_entity}."

    # Table listing
    if mutation_prefix:
        return f"{mutation_prefix}Displaying full dataset across {result.row_count} records."
    return f"The query returned {result.row_count} rows across {len(cols)} columns."


def verify_finding_result_consistency(finding: str, result: AnalysisResult) -> bool:
    """
    Verifies that a grounded analytical finding does not contradict the deterministic result table.
    Enforces the following invariants:
    1. Referenced leading/highest entity exists in result rows.
    2. Finding does not report a value different from the deterministic result.
    3. Ranking direction is consistent with the result.
    """
    if result.row_count == 0:
        return True
    if not result.plan.group_by or not result.rows:
        return True

    group_col = result.plan.group_by[0].column
    metric_cols = [c for c in result.columns if c != group_col]
    if not metric_cols:
        return True

    m_col = metric_cols[0]

    def _get_num(r):
        v = r.get(m_col)
        if isinstance(v, (int, float)):
            return v
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("-inf")

    max_row = max(result.rows, key=_get_num)
    min_row = min(result.rows, key=_get_num)
    max_entity = str(max_row.get(group_col, ""))
    min_entity = str(min_row.get(group_col, ""))

    finding_lower = finding.lower()

    # Check for "highest" or "leading" claim
    if "highest" in finding_lower or "leading" in finding_lower or "top" in finding_lower:
        if max_entity.lower() not in finding_lower:
            return False
        if _get_num(max_row) > _get_num(min_row):
            if f"{min_entity.lower()} has the highest" in finding_lower or f"leading {group_col.lower()} is {min_entity.lower()}" in finding_lower:
                return False

    # Check for "lowest" or "bottom" claim
    if "lowest" in finding_lower or "bottom" in finding_lower:
        if min_entity.lower() not in finding_lower:
            return False
        if _get_num(max_row) > _get_num(min_row):
            if f"{max_entity.lower()} has the lowest" in finding_lower:
                return False

    return True
