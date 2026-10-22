import uuid
import pytest
from app.analytics.dataset_service import ingest_dataset, get_dataset
from app.analytics.planner import generate_analysis_plan, plan_mutation
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query
from app.analytics.schemas import PlanMutationType, AggregationFunction
from app.agent.router import classify_intent
from app.agent.nodes import ambiguity_node


def test_nlu_multiturn_conversational_progression():
    """
    Comprehensive regression suite for natural language variations and multi-turn
    conversational state progression.
    """
    session_id = f"test_nlu_{uuid.uuid4().hex[:8]}"
    csv_bytes = (
        "Product,Category,Sales,Profit\n"
        "Apples,Fruit,120,40\n"
        "Bananas,Fruit,80,25\n"
        "Carrots,Vegetable,60,15\n"
        "Dates,Fruit,200,90\n"
        "Eggplant,Vegetable,40,10\n"
        "Figs,Fruit,150,70\n"
        "Grapes,Fruit,300,110\n"
    ).encode("utf-8")

    res = ingest_dataset(session_id, "grocery_sales.csv", csv_bytes)
    metadata = get_dataset(res.dataset_id, session_id=session_id)

    # Turn 1: "Top 5 products by sales"
    plan_t1 = generate_analysis_plan("Top 5 products by sales", metadata=metadata)
    assert plan_t1.limit == 5
    assert len(plan_t1.order_by) > 0
    assert plan_t1.order_by[0].descending is True
    compiled_t1 = compile_plan(plan_t1, metadata)
    res_t1 = execute_compiled_query(compiled_t1, res.dataset_id, session_id, plan_t1)
    assert len(res_t1.rows) == 5

    # Turn 2: "Reverse the order."
    mut_t2 = plan_mutation("Reverse the order.", metadata=metadata, previous_plan=plan_t1)
    assert mut_t2.mutation_type == PlanMutationType.CHANGE_SORT
    assert mut_t2.sort_delta.descending is False
    plan_t2 = generate_analysis_plan("Reverse the order.", metadata=metadata, previous_plan=plan_t1)
    assert plan_t2.order_by[0].descending is False

    # Turn 3: "Keep the same five products but sort them from smallest to largest."
    mut_t3 = plan_mutation(
        "Keep the same five products but sort them from smallest to largest.",
        metadata=metadata,
        previous_plan=plan_t1,
    )
    assert mut_t3.mutation_type == PlanMutationType.CHANGE_SORT
    assert mut_t3.sort_delta.descending is False
    plan_t3 = generate_analysis_plan(
        "Keep the same five products but sort them from smallest to largest.",
        metadata=metadata,
        previous_plan=plan_t1,
    )
    assert plan_t3.order_by[0].descending is False

    # Turn 4: "Show the five lowest products by sales." (Testing word 'five' + ascending order)
    plan_t4 = generate_analysis_plan("Show the five lowest products by sales.", metadata=metadata)
    assert plan_t4.limit == 5
    assert plan_t4.order_by[0].descending is False
    compiled_t4 = compile_plan(plan_t4, metadata)
    res_t4 = execute_compiled_query(compiled_t4, res.dataset_id, session_id, plan_t4)
    assert len(res_t4.rows) == 5

    # Turn 5: "Use profit instead of sales." (Testing metric substitution without collision)
    mut_t5 = plan_mutation("Use profit instead of sales.", metadata=metadata, previous_plan=plan_t1)
    assert mut_t5.mutation_type == PlanMutationType.CHANGE_METRIC
    assert mut_t5.metric_delta.column == "Profit"
    plan_t5 = generate_analysis_plan("Use profit instead of sales.", metadata=metadata, previous_plan=plan_t1)
    assert plan_t5.aggregations[0].column == "Profit"

    # Turn 6: "Group these by category." (Testing dimension pivot)
    mut_t6 = plan_mutation("Group these by category.", metadata=metadata, previous_plan=plan_t1)
    assert mut_t6.mutation_type == PlanMutationType.CHANGE_GROUP_BY
    assert mut_t6.group_by_delta.column == "Category"
    plan_t6 = generate_analysis_plan("Group these by category.", metadata=metadata, previous_plan=plan_t1)
    assert plan_t6.group_by[0].column == "Category"

    # Turn 7: "Turn the current results into a bar chart."
    mut_t7 = plan_mutation("Turn the current results into a bar chart.", metadata=metadata, previous_plan=plan_t1)
    assert mut_t7.mutation_type == PlanMutationType.TOGGLE_CHART
    assert mut_t7.requires_chart is True
    assert mut_t7.chart_type == "bar"


def test_nlu_ambiguous_request_clarification():
    """
    Verifies that ambiguous dataset requests with multiple metric candidates
    trigger structured clarification rather than making unverified guesses.
    """
    session_id = f"test_ambig_{uuid.uuid4().hex[:8]}"
    csv_bytes = (
        "Product,Sales,Profit,Units\n"
        "Widget A,100,20,5\n"
        "Widget B,200,40,10\n"
    ).encode("utf-8")

    res = ingest_dataset(session_id, "multi_metric.csv", csv_bytes)

    # User asks "What are the top products?" without specifying whether by Sales, Profit, or Units
    route_result = classify_intent(
        message="What are the top products?",
        dataset_id=res.dataset_id,
    )

    assert route_result["intent"] == "AMBIGUOUS"
    assert route_result["awaiting_entity"] == "dataset_metric"
    assert "candidate_metrics" in route_result["pending_analysis"]
    candidates = route_result["pending_analysis"]["candidate_metrics"]
    assert "Sales" in candidates
    assert "Profit" in candidates

    # Ambiguity node generates clarification prompt
    node_out = ambiguity_node(route_result)
    assert node_out["status"] == "needs_clarification"
    assert "analyze this by" in node_out["final_response"].lower()
