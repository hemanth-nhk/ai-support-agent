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
    {"column": "string" | null, "function": "count" | "count_distinct" | "sum" | "avg" | "min" | "max", "alias": "string"}
  ],
  "order_by": [
    {"target": "string", "descending": true | false}
  ],
  "limit": 50,
  "requires_chart": false
}

CRITICAL RULES:
1. Output ONLY a valid JSON object matching the AnalysisPlan schema. Do NOT output markdown explanations or commentary.
2. ONLY reference column names that exist in the provided schema. NEVER invent column names.
3. For aggregations with 'sum', 'avg', 'min', 'max', the 'column' MUST be a numeric column (type: numeric). NEVER compute 'sum' or 'avg' on string/categorical columns!
4. The group_by 'column' must be the categorical/dimension column (type: string), NOT the metric column.
   Example: For "Which region had the highest total revenue?", group_by column is "region", and aggregation is sum of "revenue" with alias "total_revenue".
5. Do NOT add filters unless the user explicitly requested filtering by a specific value (e.g., status == "delivered"). If there is no filter, "filters" MUST be []. NEVER filter a column against its own name.
6. In "order_by", the sort field must be named "target" (matching group_by column or aggregation alias), and direction is boolean "descending" (true for highest/top/desc/best, false for lowest/asc/worst).
7. Do NOT output extra fields like "name", "type", or "aggregation". Use the exact keys shown above.
8. If the question asks for a ranking ("top", "highest", "best"), group by the entity and order by the metric descending (descending: true) with limit (default 5 or 10).
9. If the user asks for a simple total or average without breakdown, use an aggregation without group_by.
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

        # Fix aggregation if LLM put a string column in sum/avg
        if "aggregations" in parsed_json and isinstance(parsed_json["aggregations"], list):
            for agg in parsed_json["aggregations"]:
                if isinstance(agg, dict) and str(agg.get("column")).lower() in str_col_names and agg.get("function") in ("sum", "avg"):
                    # Find candidate numeric metric
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

    # Ranking query: e.g. "top products", "which region had highest revenue"
    if any(k in lowered for k in ("top", "highest", "best", "most", "lowest", "worst")) and str_cols and target_metric:
        group_col = str_cols[0]
        for c in str_cols:
            if c.lower() in lowered:
                group_col = c
                break
        desc = not any(k in lowered for k in ("lowest", "worst", "least"))
        return {
            "dataset_id": metadata.dataset_id,
            "group_by": [{"column": group_col}],
            "aggregations": [{"column": target_metric, "function": "sum", "alias": f"sum_{target_metric}"}],
            "order_by": [{"target": f"sum_{target_metric}", "descending": desc}],
            "limit": 5,
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

    # Single scalar aggregation (e.g. total revenue or avg score)
    if result.row_count == 1 and not result.plan.group_by:
        row = rows[0]
        facts = [f"{k}: {v}" for k, v in row.items()]
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
