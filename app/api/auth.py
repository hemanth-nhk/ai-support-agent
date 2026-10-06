"""
DEMO-LEVEL access control only. This protects the customer-orders lookup
(which exposes another person's order history) from being queried by
anyone who simply guesses a customer ID.

This is NOT real authentication. A real system needs per-customer
identity (session/JWT) and an ownership check -- "is the caller allowed
to see customer 5's orders" -- not a single shared key that grants access
to every customer's data equally. This shared-key gate exists only so
the demo isn't wide open by default; replace it before any non-demo use.
"""
from fastapi import Header, HTTPException
from app.config import settings


def require_api_key(x_api_key: str = Header(default=None)) -> None:
    if not settings.support_api_key:
        # No key configured -- fail closed rather than silently allowing
        # everything through if someone forgets to set one.
        raise HTTPException(
            status_code=503,
            detail="Customer-order lookup is not configured (SUPPORT_API_KEY unset).",
        )
    if x_api_key != settings.support_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def check_api_key(provided_key: str | None) -> bool:
    """Same check, usable outside FastAPI's dependency system (e.g. GraphQL)."""
    if not settings.support_api_key:
        return False
    return provided_key == settings.support_api_key
