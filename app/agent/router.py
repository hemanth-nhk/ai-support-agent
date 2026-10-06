"""
Deterministic intent classification and entity extraction.
This is application-controlled logic, NOT an LLM call -- per project design,
the LLM never owns routing decisions. Using keyword/pattern rules keeps
behavior reproducible, explainable, and unit-testable.
"""
import re
from typing import Optional

INTENT_ORDER_STATUS = "ORDER_STATUS"
INTENT_CUSTOMER_ORDERS = "CUSTOMER_ORDERS"
INTENT_FAQ = "FAQ"
INTENT_GENERAL = "GENERAL"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"
INTENT_AMBIGUOUS = "AMBIGUOUS"
INTENT_DATA_QUERY = "DATA_QUERY"

_ORDER_ID_EXPLICIT_PATTERN = re.compile(r"\border\s*(?:id\s*)?#?\s*(\d{3,6})\b", re.IGNORECASE)
_ORDER_ID_STANDALONE_PATTERN = re.compile(r"(?:^|[^\d])#?(\d{3,6})(?:$|[^\d])")
_CUSTOMER_ID_EXPLICIT_PATTERN = re.compile(r"\bcustomer\s*(?:id\s*)?#?\s*(\d{1,6})\b", re.IGNORECASE)
_CUSTOMER_ID_STANDALONE_PATTERN = re.compile(r"(?:^|[^\d])#?(\d{1,6})(?:$|[^\d])")

_ORDER_KEYWORDS = ("order", "package", "shipment", "delivery status", "track", "delivery")
_CUSTOMER_ORDERS_KEYWORDS = ("my orders", "previous orders", "order history", "orders does customer")
_DATA_QUERY_KEYWORDS = (
    "find orders", "search orders", "all orders", "orders between",
    "orders from", "orders on", "filter orders", "list orders",
)
_DATA_QUERY_PATTERN = re.compile(
    r"\b(?:find|search|filter|list)\s+orders?\b|\borders?\s+(?:between|from|on|after|before|since|dated)\b",
    re.IGNORECASE,
)
_FAQ_KEYWORDS = (
    "shipping", "return", "refund", "cancel", "cancellation", "policy",
    "how long does", "how do i return", "money back",
)
_GENERAL_KEYWORDS = ("hello", "hi", "hey", "who are you", "what can you", "help me", "thanks", "thank you")

# Explicit deny-list to reject out-of-scope queries
_OUT_OF_SCOPE_KEYWORDS = (
    "write code", "write a python", "capital of", "stock price",
    "weather", "translate", "poem", "joke", "investment advice",
    "who is the president", "solve this equation",
)


def extract_order_id(text: str, standalone_allowed: bool = False) -> Optional[int]:
    """
    Extracts an order ID from text. If standalone_allowed is True (e.g. during a
    follow-up turn when an order ID was requested), bare numbers like '1001' or '#1001'
    are recognized.
    """
    explicit_match = _ORDER_ID_EXPLICIT_PATTERN.search(text)
    if explicit_match:
        return int(explicit_match.group(1))

    if standalone_allowed:
        # Check for pure number or '#1001' or 'it is 1001'
        standalone_match = _ORDER_ID_STANDALONE_PATTERN.search(text.strip())
        if standalone_match:
            return int(standalone_match.group(1))

    return None


def extract_customer_id(text: str, standalone_allowed: bool = False) -> Optional[int]:
    """
    Extracts a customer ID from text.
    """
    explicit_match = _CUSTOMER_ID_EXPLICIT_PATTERN.search(text)
    if explicit_match:
        return int(explicit_match.group(1))

    if standalone_allowed:
        standalone_match = _CUSTOMER_ID_STANDALONE_PATTERN.search(text.strip())
        if standalone_match:
            return int(standalone_match.group(1))

    return None


def classify_intent(message: str, previous_state: dict | None = None) -> dict:
    """
    Returns a dict:
        {
            "intent": str,
            "order_id": Optional[int],
            "customer_id": Optional[int],
            "confidence": float,
            "candidate_intents": list[str]
        }
    Deterministic -- same inputs always yield the same output.
    """
    text = message.lower().strip()

    if not text:
        return {
            "intent": INTENT_OUT_OF_SCOPE,
            "order_id": None,
            "customer_id": None,
            "confidence": 1.0,
            "candidate_intents": [INTENT_OUT_OF_SCOPE],
        }

    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in text:
            return {
                "intent": INTENT_OUT_OF_SCOPE,
                "order_id": None,
                "customer_id": None,
                "confidence": 1.0,
                "candidate_intents": [INTENT_OUT_OF_SCOPE],
            }

    # Step 4: Check if previous turn was actively awaiting an entity
    if previous_state and previous_state.get("awaiting_entity"):
        awaiting = previous_state.get("awaiting_entity")
        if awaiting == "order_id":
            found_order_id = extract_order_id(text, standalone_allowed=True)
            if found_order_id:
                return {
                    "intent": INTENT_ORDER_STATUS,
                    "order_id": found_order_id,
                    "customer_id": None,
                    "confidence": 1.0,
                    "candidate_intents": [INTENT_ORDER_STATUS],
                }
        elif awaiting == "customer_id":
            found_customer_id = extract_customer_id(text, standalone_allowed=True)
            if found_customer_id:
                return {
                    "intent": INTENT_CUSTOMER_ORDERS,
                    "order_id": None,
                    "customer_id": found_customer_id,
                    "confidence": 1.0,
                    "candidate_intents": [INTENT_CUSTOMER_ORDERS],
                }

    # Standard entity extraction
    order_id = extract_order_id(text, standalone_allowed=False)
    customer_id = extract_customer_id(text, standalone_allowed=False)

    has_order_keywords = any(kw in text for kw in _ORDER_KEYWORDS)
    has_customer_orders_keywords = any(kw in text for kw in _CUSTOMER_ORDERS_KEYWORDS)
    has_data_query_keywords = any(kw in text for kw in _DATA_QUERY_KEYWORDS)
    has_data_query_pattern = bool(_DATA_QUERY_PATTERN.search(text))
    has_faq_keywords = any(kw in text for kw in _FAQ_KEYWORDS)
    has_general_keywords = any(kw in text for kw in _GENERAL_KEYWORDS)

    # Step 5: Ambiguity Detection
    # When query mentions both order tracking / order ID and policy / refund / return keywords
    # E.g. "Can I return order 1001?", "My order hasn't arrived and I want a refund."
    has_order_signal = (order_id is not None) or (has_order_keywords and not has_customer_orders_keywords)
    if has_order_signal and has_faq_keywords:
        return {
            "intent": INTENT_AMBIGUOUS,
            "order_id": order_id,
            "customer_id": None,
            "confidence": 0.5,
            "candidate_intents": [INTENT_FAQ, INTENT_ORDER_STATUS],
        }

    # Customer orders routing (explicit customer id or dedicated customer keywords)
    if customer_id and has_customer_orders_keywords:
        return {
            "intent": INTENT_CUSTOMER_ORDERS,
            "order_id": None,
            "customer_id": customer_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_CUSTOMER_ORDERS],
        }

    if has_customer_orders_keywords and not order_id and not has_data_query_keywords:
        return {
            "intent": INTENT_CUSTOMER_ORDERS,
            "order_id": None,
            "customer_id": customer_id,
            "confidence": 0.9,
            "candidate_intents": [INTENT_CUSTOMER_ORDERS],
        }

    # DATA_QUERY routing - checked BEFORE generic ORDER_STATUS
    if has_data_query_keywords or has_data_query_pattern:
        return {
            "intent": INTENT_DATA_QUERY,
            "order_id": order_id,
            "customer_id": customer_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_DATA_QUERY],
        }

    # Generic order status routing
    if order_id or has_order_keywords:
        return {
            "intent": INTENT_ORDER_STATUS,
            "order_id": order_id,
            "customer_id": None,
            "confidence": 0.95 if order_id else 0.8,
            "candidate_intents": [INTENT_ORDER_STATUS],
        }

    # FAQ routing
    if has_faq_keywords:
        return {
            "intent": INTENT_FAQ,
            "order_id": None,
            "customer_id": None,
            "confidence": 0.9,
            "candidate_intents": [INTENT_FAQ],
        }

    # General greeting routing
    if has_general_keywords:
        return {
            "intent": INTENT_GENERAL,
            "order_id": None,
            "customer_id": None,
            "confidence": 0.9,
            "candidate_intents": [INTENT_GENERAL],
        }

    # Default fallback: fail-closed out-of-scope
    return {
        "intent": INTENT_OUT_OF_SCOPE,
        "order_id": None,
        "customer_id": None,
        "confidence": 1.0,
        "candidate_intents": [INTENT_OUT_OF_SCOPE],
    }
