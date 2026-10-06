"""
Service layer: business logic. This is the ONE place order-related rules
live. REST, GraphQL, and the LangGraph order tool all call into this module
— none of them talk to the repository directly. This is what "no duplicate
business logic" means in practice.
"""
from typing import Optional
from app.repositories import order_repository, customer_repository
from app.agent.filters import OrderQueryFilter


class OrderNotFoundError(Exception):
    """Raised when a requested order does not exist. Callers must not
    invent a substitute answer when this is raised — see guardrails."""


class CustomerNotFoundError(Exception):
    pass


def get_order(order_id: int) -> dict:
    order = order_repository.find_order_by_id(order_id)
    if order is None:
        raise OrderNotFoundError(f"Order {order_id} was not found.")
    return order


def get_customer_orders(customer_id: int) -> list[dict]:
    customer = customer_repository.find_customer_by_id(customer_id)
    if customer is None:
        raise CustomerNotFoundError(f"Customer {customer_id} was not found.")
    return order_repository.find_orders_by_customer_id(customer_id)


def resolve_customer_name(customer_name: str) -> tuple[str, list[dict]]:
    """
    Resolves customer_name against the customer database.
    Returns (status, matches) where status is:
    - 'resolved': exactly 1 match (matches[0]['id'])
    - 'not_found': 0 matches
    - 'needs_clarification': >1 matches (ambiguous)
    """
    matches = customer_repository.find_customers_by_name(customer_name)
    if len(matches) == 0:
        return "not_found", []
    elif len(matches) == 1:
        return "resolved", matches
    else:
        return "needs_clarification", matches


def search_orders(filters: OrderQueryFilter) -> tuple[list[dict], int]:
    """
    Executes parameterized order search through the repository.
    Enforces that customer_name is never used directly in search_orders.
    """
    return order_repository.search_orders(
        customer_id=filters.customer_id,
        product=filters.product,
        status=filters.status,
        date_from=filters.date_from,
        date_to=filters.date_to,
        time_from=filters.time_from,
        time_to=filters.time_to,
        amount_min=filters.amount_min,
        amount_max=filters.amount_max,
    )
