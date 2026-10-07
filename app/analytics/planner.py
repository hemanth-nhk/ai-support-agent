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
)
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
8. The group_by 'column' must be the categorical/dimension column (type: string), NOT the metric column.
9. Do NOT add filters unless the user explicitly requested filtering by a specific value (e.g., status == "delivered").
10. In "order_by", the sort field must be named "target" (matching group_by column or aggregation alias), and direction is boolean "descending" (true for highest/top/desc/best, false for lowest/asc/worst).
11. If the question asks to 'chart', 'plot', 'graph', or 'visualize', set 'requires_chart': true, and specify 'chart_type' if a specific chart is requested.
"""


def generate_analysis_plan(
    question: str,
    metadata: DatasetMetadata,
    resolved_metric: Optional[str] = None,
) -> AnalysisPlan:
    """
    Prompts the LLM to generate an AnalysisPlan, validates with Pydantic,
    and enforces server-authoritative dataset_id.
    """
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
        import re
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


def format_deterministic_result(result: AnalysisResult, question: str) -> str:
    """
    Synthesizes a verifiable, deterministic factual statement from DuckDB results.
    Guarantees numerical correctness independently of LLM whims.
    """
    if result.row_count == 0:
        return "The query completed successfully, but no matching records were found in the dataset."

    rows = result.rows
    cols = result.columns

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

        statement = f"The leading {group_keys[0] if group_keys else 'group'} is {lead_entity} with " + ", ".join(lead_metrics) + "."

        if len(rows) > 1:
            second_row = rows[1]
            sec_entity = second_row.get(group_keys[0]) if group_keys else "Second item"
            sec_metrics = [f"{m} is {second_row.get(m)}" for m in metric_keys]
            statement += f" Followed by {sec_entity} with " + ", ".join(sec_metrics) + "."

        return statement

    # Table listing
    return f"The query returned {result.row_count} rows across {len(cols)} columns."
