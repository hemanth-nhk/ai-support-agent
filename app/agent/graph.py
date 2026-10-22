"""
Graph assembly. START -> classify -> conditional route -> one tool/response
node -> END. This is the only place the graph topology is defined.
"""
from typing import Optional
from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    classify_node,
    order_status_node,
    customer_orders_node,
    faq_node,
    general_node,
    out_of_scope_node,
    ambiguity_node,
    data_query_node,
    document_query_node,
    dataset_analysis_node,
    order_policy_node,
    route_after_classify,
)

_compiled_graph = None


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_node", classify_node)
    graph.add_node("order_status_node", order_status_node)
    graph.add_node("customer_orders_node", customer_orders_node)
    graph.add_node("faq_node", faq_node)
    graph.add_node("general_node", general_node)
    graph.add_node("out_of_scope_node", out_of_scope_node)
    graph.add_node("ambiguity_node", ambiguity_node)
    graph.add_node("data_query_node", data_query_node)
    graph.add_node("document_query_node", document_query_node)
    graph.add_node("dataset_analysis_node", dataset_analysis_node)
    graph.add_node("order_policy_node", order_policy_node)

    graph.add_edge(START, "classify_node")
    graph.add_conditional_edges(
        "classify_node",
        route_after_classify,
        {
            "order_status_node": "order_status_node",
            "customer_orders_node": "customer_orders_node",
            "faq_node": "faq_node",
            "general_node": "general_node",
            "out_of_scope_node": "out_of_scope_node",
            "ambiguity_node": "ambiguity_node",
            "data_query_node": "data_query_node",
            "document_query_node": "document_query_node",
            "dataset_analysis_node": "dataset_analysis_node",
            "order_policy_node": "order_policy_node",
        },
    )
    for node_name in (
        "order_status_node", "customer_orders_node",
        "faq_node", "general_node", "out_of_scope_node",
        "ambiguity_node", "data_query_node", "document_query_node",
        "dataset_analysis_node", "order_policy_node",
    ):
        graph.add_edge(node_name, END)

    return graph.compile()


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


_UNSET = object()


def run_agent(
    user_message: str,
    previous_state: Optional[dict] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    document_id: Optional[str] = _UNSET,
    dataset_id: Optional[str] = _UNSET,
    workspace_mode: Optional[str] = None,
    reference_date: Optional[object] = None,
) -> AgentState:
    graph = get_graph()
    initial_state = dict(previous_state or {})
    initial_state["user_message"] = user_message
    if session_id is not None:
        initial_state["session_id"] = session_id
    if document_id is not _UNSET:
        initial_state["document_id"] = document_id
    if dataset_id is not _UNSET:
        if dataset_id != initial_state.get("dataset_id"):
            initial_state["active_analysis_plan"] = None
            initial_state["analysis_plan"] = None
        initial_state["dataset_id"] = dataset_id
    if workspace_mode is not None:
        initial_state["workspace_mode"] = workspace_mode
    if api_key is not None:
        initial_state["api_key"] = api_key
    if reference_date is not None:
        initial_state["reference_date"] = reference_date
    elif "reference_date" not in initial_state and previous_state and "reference_date" in previous_state:
        initial_state["reference_date"] = previous_state["reference_date"]
    # Propagate and maintain conversation history across turns
    history = list(initial_state.get("conversation_history") or [])
    if not history and previous_state and previous_state.get("user_message"):
        history.append({"role": "user", "content": previous_state["user_message"]})
        if previous_state.get("final_response"):
            history.append({"role": "assistant", "content": previous_state["final_response"]})
    initial_state["conversation_history"] = history

    result = graph.invoke(initial_state)

    # Append current turn to conversation history
    updated_history = list(result.get("conversation_history") or history)
    updated_history.append({"role": "user", "content": user_message})
    if result.get("final_response"):
        updated_history.append({"role": "assistant", "content": result["final_response"]})
    result["conversation_history"] = updated_history

    return result
