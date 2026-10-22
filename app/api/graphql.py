"""
GraphQL layer. Resolvers call the SAME service layer as REST — see
app/services/order_service.py. No business logic is duplicated here.
"""
from typing import Optional
import strawberry
from strawberry.types import Info
from strawberry.fastapi import BaseContext

from app.services import order_service
from app.api.auth import check_api_key


@strawberry.type
class Order:
    id: int
    customer_id: int
    product: str
    status: str
    estimated_delivery: Optional[str]
    order_time: Optional[str] = None
    amount: Optional[float] = None
    created_at: str


def _to_order_type(order_dict: dict) -> Order:
    return Order(**order_dict)


@strawberry.type
class Query:
    @strawberry.field
    def order(self, info: Info, id: int) -> Optional[Order]:
        request = info.context["request"]
        provided_key = request.headers.get("x-api-key")
        if not check_api_key(provided_key):
            raise Exception("Invalid or missing API key for order.")

        try:
            order_dict = order_service.get_order(id)
        except order_service.OrderNotFoundError:
            return None
        return _to_order_type(order_dict)

    @strawberry.field
    def customer_orders(self, info: Info, customer_id: int) -> list[Order]:
        # Demo-level protection only, mirrors the REST endpoint's guard --
        # see app/api/auth.py docstring. Reads the same X-API-Key header.
        request = info.context["request"]
        provided_key = request.headers.get("x-api-key")
        if not check_api_key(provided_key):
            raise Exception("Invalid or missing API key for customerOrders.")

        try:
            orders = order_service.get_customer_orders(customer_id)
        except order_service.CustomerNotFoundError:
            return []
        return [_to_order_type(o) for o in orders]


schema = strawberry.Schema(query=Query)
