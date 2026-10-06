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
        },
    )
    for node_name in (
        "order_status_node", "customer_orders_node",
        "faq_node", "general_node", "out_of_scope_node",
        "ambiguity_node", "data_query_node",
    ):
        graph.add_edge(node_name, END)

    return graph.compile()


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    user_message: str,
    previous_state: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> AgentState:
    graph = get_graph()
    initial_state = dict(previous_state or {})
    initial_state["user_message"] = user_message
    if api_key is not None:
        initial_state["api_key"] = api_key
    result = graph.invoke(initial_state)
    return result
