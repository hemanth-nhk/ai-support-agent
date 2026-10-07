"""
Deterministic AST mutation engine for multi-turn conversational analytical memory.
Applies incremental PlanMutation deltas to an existing AnalysisPlan, enforcing
strict Pydantic validation and fail-closed rollback invariants.
"""
from copy import deepcopy
from typing import Optional

from pydantic import ValidationError

from app.analytics.schemas import (
    AnalysisPlan,
    DatasetMetadata,
    FilterClause,
    GroupByClause,
    PlanMutation,
    PlanMutationType,
)


class MutationError(ValueError):
    """Raised when an incremental AST mutation fails validation or execution."""
    pass


def apply_plan_mutation(
    previous_plan: AnalysisPlan,
    mutation: PlanMutation,
    metadata: Optional[DatasetMetadata] = None,
) -> AnalysisPlan:
    """
    Applies an incremental PlanMutation to previous_plan, returning an updated AnalysisPlan.

    Guarantees fail-closed semantics:
    - If any transformation, constraint, or Pydantic validation fails, an exception is raised
      and previous_plan remains completely untouched.
    """
    if previous_plan is None:
        raise MutationError("Cannot apply mutation: previous_plan is None.")
    if mutation is None:
        raise MutationError("Cannot apply mutation: mutation is None.")

    mtype = mutation.mutation_type

    # 1. NEW_QUERY: Complete replacement plan
    if mtype == PlanMutationType.NEW_QUERY:
        if mutation.new_plan is None:
            raise MutationError("NEW_QUERY mutation requires 'new_plan'.")
        # Validate replacement plan
        try:
            return AnalysisPlan.model_validate(mutation.new_plan.model_dump())
        except ValidationError as err:
            raise MutationError(f"NEW_QUERY plan failed validation: {err}") from err

    # Clone previous plan dictionary for safe, isolated modification
    plan_dict = previous_plan.model_dump()

    # 2. RESET_ANALYSIS: Clear filters and secondary operators back to baseline
    if mtype == PlanMutationType.RESET_ANALYSIS:
        plan_dict["filters"] = []
        plan_dict["period_comparison"] = None
        plan_dict["rolling_windows"] = []
        plan_dict["cumulative_metrics"] = []
        plan_dict["outlier_analysis"] = None
        plan_dict["correlations"] = []
        plan_dict["quantiles"] = []
        if mutation.group_by_delta:
            plan_dict["group_by"] = [mutation.group_by_delta.to_group_by_clause().model_dump()]
        if mutation.metric_delta:
            plan_dict["aggregations"] = [mutation.metric_delta.to_aggregation_clause().model_dump()]
        if mutation.sort_delta:
            plan_dict["order_by"] = [mutation.sort_delta.to_order_by_clause().model_dump()]
        if mutation.limit_delta is not None:
            plan_dict["limit"] = max(1, min(500, mutation.limit_delta))

    # 3. ADD_FILTER: Add or replace condition on target column
    elif mtype == PlanMutationType.ADD_FILTER:
        if mutation.filter_delta is None:
            raise MutationError("ADD_FILTER requires 'filter_delta'.")
        if mutation.filter_delta.operator is None:
            raise MutationError("ADD_FILTER requires an operator in filter_delta.")
        
        new_clause: FilterClause = mutation.filter_delta.to_filter_clause()
        filters = plan_dict.get("filters", [])
        
        # If filter on same column and operator already exists, replace it; otherwise append
        replaced = False
        for i, existing in enumerate(filters):
            if existing.get("column") == new_clause.column and existing.get("operator") == new_clause.operator.value:
                filters[i] = new_clause.model_dump()
                replaced = True
                break
        if not replaced:
            filters.append(new_clause.model_dump())
        plan_dict["filters"] = filters

    # 4. CHANGE_FILTER: Update filter for specific column
    elif mtype == PlanMutationType.CHANGE_FILTER:
        if mutation.filter_delta is None:
            raise MutationError("CHANGE_FILTER requires 'filter_delta'.")
        if mutation.filter_delta.operator is None:
            raise MutationError("CHANGE_FILTER requires an operator in filter_delta.")

        new_clause = mutation.filter_delta.to_filter_clause()
        filters = plan_dict.get("filters", [])
        
        # Replace first filter targeting this column, or append if none
        target_idx = next((i for i, f in enumerate(filters) if f.get("column") == new_clause.column), None)
        if target_idx is not None:
            filters[target_idx] = new_clause.model_dump()
        else:
            filters.append(new_clause.model_dump())
        plan_dict["filters"] = filters

    # 5. REMOVE_FILTER: Remove filter(s) on column, or clear all if none specified
    elif mtype == PlanMutationType.REMOVE_FILTER:
        if mutation.filter_delta is not None and mutation.filter_delta.column:
            target_col = mutation.filter_delta.column
            plan_dict["filters"] = [
                f for f in plan_dict.get("filters", []) if f.get("column") != target_col
            ]
        else:
            plan_dict["filters"] = []

    # 6. CHANGE_GROUP_BY: Replace entire group_by clause with new dimension
    elif mtype == PlanMutationType.CHANGE_GROUP_BY:
        if mutation.group_by_delta is None:
            raise MutationError("CHANGE_GROUP_BY requires 'group_by_delta'.")
        new_gb: GroupByClause = mutation.group_by_delta.to_group_by_clause()
        plan_dict["group_by"] = [new_gb.model_dump()]

    # 7. ADD_GROUP_BY: Append additional dimension to group_by
    elif mtype == PlanMutationType.ADD_GROUP_BY:
        if mutation.group_by_delta is None:
            raise MutationError("ADD_GROUP_BY requires 'group_by_delta'.")
        new_gb = mutation.group_by_delta.to_group_by_clause()
        existing_gbs = plan_dict.get("group_by", [])
        
        # If dimension already exists, update date_trunc if provided; else append
        exists = False
        for i, eg in enumerate(existing_gbs):
            if eg.get("column") == new_gb.column:
                existing_gbs[i] = new_gb.model_dump()
                exists = True
                break
        if not exists:
            existing_gbs.append(new_gb.model_dump())
        plan_dict["group_by"] = existing_gbs

    # 8. REMOVE_GROUP_BY: Remove specified dimension, or clear all if none specified
    elif mtype == PlanMutationType.REMOVE_GROUP_BY:
        if mutation.group_by_delta is not None and mutation.group_by_delta.column:
            target_col = mutation.group_by_delta.column
            plan_dict["group_by"] = [
                g for g in plan_dict.get("group_by", []) if g.get("column") != target_col
            ]
        else:
            plan_dict["group_by"] = []

    # 9. CHANGE_METRIC: Replace primary aggregation metric
    elif mtype == PlanMutationType.CHANGE_METRIC:
        if mutation.metric_delta is None:
            raise MutationError("CHANGE_METRIC requires 'metric_delta'.")
        new_agg = mutation.metric_delta.to_aggregation_clause()
        plan_dict["aggregations"] = [new_agg.model_dump()]

    # 10. CHANGE_SORT: Replace ordering clauses
    elif mtype == PlanMutationType.CHANGE_SORT:
        if mutation.sort_delta is None:
            raise MutationError("CHANGE_SORT requires 'sort_delta'.")
        new_ob = mutation.sort_delta.to_order_by_clause()
        plan_dict["order_by"] = [new_ob.model_dump()]

    # 11. CHANGE_LIMIT: Modify row limit
    elif mtype == PlanMutationType.CHANGE_LIMIT:
        if mutation.limit_delta is None:
            raise MutationError("CHANGE_LIMIT requires 'limit_delta'.")
        plan_dict["limit"] = max(1, min(500, mutation.limit_delta))

    # 12. ADD_ROLLING: Add or update rolling window specification
    elif mtype == PlanMutationType.ADD_ROLLING:
        if mutation.rolling_delta is None:
            raise MutationError("ADD_ROLLING requires 'rolling_delta'.")
        plan_dict["rolling_windows"] = [mutation.rolling_delta.model_dump()]

    # 13. ADD_CUMULATIVE: Add or update cumulative metric specification
    elif mtype == PlanMutationType.ADD_CUMULATIVE:
        if mutation.cumulative_delta is None:
            raise MutationError("ADD_CUMULATIVE requires 'cumulative_delta'.")
        plan_dict["cumulative_metrics"] = [mutation.cumulative_delta.model_dump()]

    # 14. TOGGLE_CHART: Toggle visualization or set chart type
    elif mtype == PlanMutationType.TOGGLE_CHART:
        if mutation.chart_type is not None:
            plan_dict["chart_type"] = mutation.chart_type
            plan_dict["requires_chart"] = True
        elif mutation.requires_chart is not None:
            plan_dict["requires_chart"] = mutation.requires_chart
            if not mutation.requires_chart:
                plan_dict["chart_type"] = None
        else:
            cur = plan_dict.get("requires_chart", False)
            plan_dict["requires_chart"] = not cur
            if not plan_dict["requires_chart"]:
                plan_dict["chart_type"] = None

    # Handle secondary deltas if specified alongside primary mutation
    if mtype != PlanMutationType.CHANGE_LIMIT and mutation.limit_delta is not None:
        plan_dict["limit"] = max(1, min(500, mutation.limit_delta))
    if mtype != PlanMutationType.TOGGLE_CHART:
        if mutation.chart_type is not None:
            plan_dict["chart_type"] = mutation.chart_type
            plan_dict["requires_chart"] = True
        elif mutation.requires_chart is not None:
            plan_dict["requires_chart"] = mutation.requires_chart
            if not mutation.requires_chart:
                plan_dict["chart_type"] = None
    if mtype not in (PlanMutationType.CHANGE_SORT, PlanMutationType.RESET_ANALYSIS) and mutation.sort_delta is not None:
        plan_dict["order_by"] = [mutation.sort_delta.to_order_by_clause().model_dump()]

    # Pydantic validation chokepoint
    try:
        updated_plan = AnalysisPlan.model_validate(plan_dict)
    except ValidationError as err:
        raise MutationError(f"Mutated plan failed schema validation: {err}") from err

    # Optional dataset column integrity check
    if metadata is not None:
        col_names = {c.name for c in metadata.profile.columns}
        for f in updated_plan.filters:
            if f.column not in col_names:
                raise MutationError(f"Filter column '{f.column}' does not exist in dataset.")
        for g in updated_plan.group_by:
            if g.column not in col_names:
                raise MutationError(f"Group-by column '{g.column}' does not exist in dataset.")
        for a in updated_plan.aggregations:
            if a.column and a.column != "*" and a.column not in col_names:
                raise MutationError(f"Aggregation column '{a.column}' does not exist in dataset.")
        for r in updated_plan.rolling_windows:
            if r.column not in col_names:
                raise MutationError(f"Rolling window target column '{r.column}' does not exist in dataset.")
            if r.order_by not in col_names:
                raise MutationError(f"Rolling window order column '{r.order_by}' does not exist in dataset.")
            for p in r.partition_by:
                if p not in col_names:
                    raise MutationError(f"Rolling partition column '{p}' does not exist in dataset.")
        for c in updated_plan.cumulative_metrics:
            if c.column and c.column not in col_names:
                raise MutationError(f"Cumulative target column '{c.column}' does not exist in dataset.")
            if c.order_by not in col_names:
                raise MutationError(f"Cumulative order column '{c.order_by}' does not exist in dataset.")
            for p in c.partition_by:
                if p not in col_names:
                    raise MutationError(f"Cumulative partition column '{p}' does not exist in dataset.")

    return updated_plan


def describe_mutation(mutation: PlanMutation) -> str:
    """Returns a concise human-readable description of the applied mutation."""
    mtype = mutation.mutation_type
    if mtype == PlanMutationType.NEW_QUERY:
        return "Started new analysis query."
    if mtype == PlanMutationType.RESET_ANALYSIS:
        return "Reset analysis filters and baseline."
    if mtype == PlanMutationType.ADD_FILTER and mutation.filter_delta:
        fd = mutation.filter_delta
        return f"Added filter on '{fd.column}' ({fd.operator.value if fd.operator else '=='} {fd.value})."
    if mtype == PlanMutationType.CHANGE_FILTER and mutation.filter_delta:
        fd = mutation.filter_delta
        return f"Updated filter on '{fd.column}' to {fd.operator.value if fd.operator else '=='} {fd.value}."
    if mtype == PlanMutationType.REMOVE_FILTER:
        if mutation.filter_delta and mutation.filter_delta.column:
            return f"Removed filter on '{mutation.filter_delta.column}'."
        return "Cleared all filters."
    if mtype == PlanMutationType.CHANGE_GROUP_BY and mutation.group_by_delta:
        return f"Changed grouping dimension to '{mutation.group_by_delta.column}'."
    if mtype == PlanMutationType.ADD_GROUP_BY and mutation.group_by_delta:
        return f"Added grouping dimension '{mutation.group_by_delta.column}'."
    if mtype == PlanMutationType.REMOVE_GROUP_BY:
        if mutation.group_by_delta and mutation.group_by_delta.column:
            return f"Removed grouping dimension '{mutation.group_by_delta.column}'."
        return "Removed all grouping dimensions."
    if mtype == PlanMutationType.CHANGE_METRIC and mutation.metric_delta:
        md = mutation.metric_delta
        return f"Changed metric to {md.function.value}({md.column or '*'})."
    if mtype == PlanMutationType.CHANGE_SORT and mutation.sort_delta:
        sd = mutation.sort_delta
        return f"Sorted by '{sd.target}' ({'DESC' if sd.descending else 'ASC'})."
    if mtype == PlanMutationType.CHANGE_LIMIT and mutation.limit_delta:
        return f"Changed row limit to {mutation.limit_delta}."
    if mtype == PlanMutationType.ADD_ROLLING and mutation.rolling_delta:
        rd = mutation.rolling_delta
        return f"Added {rd.preceding + 1}-period rolling {rd.function.value} on '{rd.column}'."
    if mtype == PlanMutationType.ADD_CUMULATIVE and mutation.cumulative_delta:
        cd = mutation.cumulative_delta
        return f"Added cumulative {cd.function.value} on '{cd.column}'."
    if mtype == PlanMutationType.TOGGLE_CHART:
        if mutation.chart_type:
            return f"Set visualization chart to {mutation.chart_type}."
        if mutation.requires_chart is not None:
            return "Enabled visualization." if mutation.requires_chart else "Disabled visualization."
        return "Toggled visualization."
    return f"Applied mutation: {mtype.value}."
