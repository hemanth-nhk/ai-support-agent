"""
Order-Policy Cross-Referencing Service.
Combines verified order database records with uploaded company policies (or store FAQ fallback)
to deterministically calculate return and refund eligibility.

Strict Safeguards:
1. Deterministic Math: Date arithmetic and eligibility verdicts are computed strictly in Python.
   The LLM never performs date math or decides eligibility.
2. Verified Delivery Date: Only orders with status == "Delivered" and a validated past delivery date
   are evaluated for elapsed days. Non-delivered orders (Shipped/Processing) are reported as not yet delivered.
3. Strict Policy Interpretation: If policy text has conflicting return windows, ambiguous wording,
   or missing return window rules, it abstains with an INDETERMINATE or CONFLICTING verdict instead of guessing.
4. Source Attribution & Isolation: Prioritizes session-uploaded policies; uses store FAQ fallback
   only when no session document exists. Never fabricates a default return window.
"""
from dataclasses import dataclass
from datetime import date, datetime
import logging
import re
from typing import Optional

from app.services import order_service, document_service
from app.rag.retriever import retrieve_faq

logger = logging.getLogger("app.services.order_policy_service")

# Outcome Verdicts
VERDICT_ELIGIBLE = "ELIGIBLE"
VERDICT_EXPIRED = "EXPIRED"
VERDICT_NOT_YET_DELIVERED = "NOT_YET_DELIVERED"
VERDICT_CANCELLED = "CANCELLED"
VERDICT_NON_RETURNABLE = "NON_RETURNABLE"
VERDICT_UNKNOWN_ORDER = "UNKNOWN_ORDER"
VERDICT_MISSING_POLICY = "MISSING_POLICY"
VERDICT_CONFLICTING_POLICY = "CONFLICTING_POLICY"
VERDICT_INDETERMINATE_POLICY = "INDETERMINATE_POLICY"
VERDICT_INDETERMINATE_DELIVERY_DATE = "INDETERMINATE_DELIVERY_DATE"


@dataclass
class EligibilityAssessment:
    verdict: str
    order_id: Optional[int]
    product: Optional[str]
    order_status: Optional[str]
    delivery_date: Optional[date]
    reference_date: date
    days_since_delivery: Optional[int]
    return_window_days: Optional[int]
    policy_source: Optional[str]
    policy_citation: Optional[str]
    conditions: list[str]
    explanation: str


# Regex patterns for return windows
_WINDOW_PATTERNS = [
    # "within 30 days of delivery", "within 14 calendar days"
    re.compile(r"\bwithin\s+(\d+)\s*(?:calendar\s*)?days?\b", re.IGNORECASE),
    # "30-day return window", "14 day return policy", "30 days return"
    re.compile(r"\b(\d+)\s*[- ]day\s*(?:return|refund|exchange)?\s*(?:window|period|policy)?\b", re.IGNORECASE),
    # "accepted up to 30 days"
    re.compile(r"\b(?:accepted|eligible|allowed)\s+up\s+to\s+(\d+)\s*days?\b", re.IGNORECASE),
    # "returnable within 30 days"
    re.compile(r"\breturnable\s+within\s+(\d+)\s*days?\b", re.IGNORECASE),
    # "returns must be made within 30 days"
    re.compile(r"\breturns?\s+(?:must\s+be\s+made|can\s+be\s+made|allowed|accepted)?\s*within\s+(\d+)\s*days?\b", re.IGNORECASE),
]

_NON_RETURNABLE_PATTERNS = [
    re.compile(r"\b(?:all\s+sales\s+(?:are\s+)?final|non[- ]refundable|non[- ]returnable|no\s+returns?\s+(?:allowed|accepted))\b", re.IGNORECASE),
]

_COMMON_CONDITIONS = [
    (re.compile(r"\b(?:original\s+packaging|unopened\s+packaging|in\s+(?:their|its)?\s*original\s+box)\b", re.IGNORECASE), "original packaging"),
    (re.compile(r"\b(?:unused|unworn|unopened|undamaged)\b", re.IGNORECASE), "unused condition"),
    (re.compile(r"\b(?:receipt|proof\s+of\s+purchase)\b", re.IGNORECASE), "proof of purchase required"),
    (re.compile(r"\b(?:tags?\s+attached)\b", re.IGNORECASE), "original tags attached"),
]


def extract_policy_return_rules(policy_text: str, product_name: Optional[str] = None) -> tuple[str, Optional[int], list[str], Optional[str]]:
    """
    Extracts return window days, conditions, and non-returnable status from policy text.
    Returns:
        (status, window_days, conditions, citation_snippet)
    where status is one of:
        - "ok": Single clear return window found
        - "non_returnable": Explicitly non-returnable
        - "conflicting": Multiple conflicting return windows found without product resolution
        - "missing": No return window or return rules found
    """
    if not policy_text or not policy_text.strip():
        return "missing", None, [], None

    clean_text = policy_text.strip()

    # 1. Check for product-specific or universal non-returnable policy
    for pat in _NON_RETURNABLE_PATTERNS:
        match = pat.search(clean_text)
        if match:
            # Check if this applies generally or to this product
            snippet = clean_text[max(0, match.start() - 30):min(len(clean_text), match.end() + 50)].strip()
            return "non_returnable", 0, [], snippet

    # 2. Extract return window numbers
    found_windows = set()
    first_match_snippet = None

    for pat in _WINDOW_PATTERNS:
        for m in pat.finditer(clean_text):
            try:
                days = int(m.group(1))
                if 1 <= days <= 365:
                    found_windows.add(days)
                    if first_match_snippet is None:
                        start_idx = max(0, m.start() - 25)
                        end_idx = min(len(clean_text), m.end() + 35)
                        first_match_snippet = clean_text[start_idx:end_idx].strip()
            except (ValueError, IndexError):
                continue

    # 3. Detect conflicting return windows
    if len(found_windows) > 1:
        # Multiple different windows found in the policy text without disambiguation
        logger.warning(f"Conflicting return windows found in policy text: {found_windows}")
        return "conflicting", None, [], f"Conflicting return windows found: {sorted(found_windows)} days"

    if len(found_windows) == 0:
        return "missing", None, [], None

    window_days = next(iter(found_windows))

    # 4. Extract conditions
    conditions = []
    for cond_pat, label in _COMMON_CONDITIONS:
        if cond_pat.search(clean_text):
            conditions.append(label)

    return "ok", window_days, conditions, first_match_snippet


def retrieve_policy_context(
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    fallback_query: str = "return policy return window refund eligibility",
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Retrieves policy text following strict source hierarchy:
    1. Uploaded document in active session (if documents exist).
    2. Fallback to store FAQ if no uploaded document exists.
    Returns:
        (policy_text, policy_source_type, policy_source_name)
    """
    # Check session uploaded documents first
    if session_id and session_id.strip():
        session_docs = document_service.get_session_documents(session_id.strip())
        if session_docs:
            chunks = document_service.retrieve_document_chunks(
                session_id=session_id.strip(),
                query=fallback_query,
                document_id=document_id,
                top_k=3,
            )
            if chunks:
                combined_text = "\n\n".join(c.get("text", "") for c in chunks if c.get("text"))
                doc_name = chunks[0].get("metadata", {}).get("filename") or session_docs[0].get("filename", "Uploaded Document")
                return combined_text, "uploaded_document", doc_name

    # Fallback: Store FAQ
    faq_res = retrieve_faq("Items can be returned within 30 days of delivery for a full refund.")
    if not (faq_res.get("found") and faq_res.get("chunks")):
        faq_res = retrieve_faq(fallback_query)
    if faq_res.get("found") and faq_res.get("chunks"):
        faq_text = "\n\n".join(faq_res["chunks"])
        return faq_text, "store_faq", "Store Returns FAQ"

    return None, None, None


def assess_return_eligibility(
    order_id: int,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    reference_date: Optional[date] = None,
    custom_policy_text: Optional[str] = None,
    custom_policy_source: Optional[str] = None,
) -> EligibilityAssessment:
    """
    Main evaluation function:
    Combines verified order data with retrieved policy rules to deterministically
    compute return eligibility.
    """
    ref_date = reference_date or date.today()

    # Step 1: Look up order in database
    try:
        order = order_service.get_order(order_id)
    except order_service.OrderNotFoundError:
        return EligibilityAssessment(
            verdict=VERDICT_UNKNOWN_ORDER,
            order_id=order_id,
            product=None,
            order_status=None,
            delivery_date=None,
            reference_date=ref_date,
            days_since_delivery=None,
            return_window_days=None,
            policy_source=None,
            policy_citation=None,
            conditions=[],
            explanation=f"Order {order_id} was not found in our system. Please double-check the order ID.",
        )

    product = order.get("product", "Item")
    status = (order.get("status") or "Unknown").strip()
    status_lower = status.lower()

    # Step 2: Handle non-delivered order states immediately
    if status_lower in ("processing", "shipped", "out for delivery"):
        est_delivery = order.get("estimated_delivery")
        delivery_note = f" (estimated delivery: {est_delivery})" if est_delivery else ""
        return EligibilityAssessment(
            verdict=VERDICT_NOT_YET_DELIVERED,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=None,
            reference_date=ref_date,
            days_since_delivery=None,
            return_window_days=None,
            policy_source=None,
            policy_citation=None,
            conditions=[],
            explanation=(
                f"Order {order_id} ({product}) is currently '{status}'{delivery_note} and has not yet been delivered. "
                "Items can only be returned after delivery has occurred."
            ),
        )

    if status_lower == "cancelled":
        return EligibilityAssessment(
            verdict=VERDICT_CANCELLED,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=None,
            reference_date=ref_date,
            days_since_delivery=None,
            return_window_days=None,
            policy_source=None,
            policy_citation=None,
            conditions=[],
            explanation=f"Order {order_id} ({product}) was cancelled and is not eligible for return.",
        )

    # Step 3: For delivered orders, verify confirmed delivery date
    if status_lower == "delivered":
        raw_delivery = order.get("estimated_delivery")
        if not raw_delivery or not isinstance(raw_delivery, str) or not raw_delivery.strip():
            return EligibilityAssessment(
                verdict=VERDICT_INDETERMINATE_DELIVERY_DATE,
                order_id=order_id,
                product=product,
                order_status=status,
                delivery_date=None,
                reference_date=ref_date,
                days_since_delivery=None,
                return_window_days=None,
                policy_source=None,
                policy_citation=None,
                conditions=[],
                explanation=(
                    f"Order {order_id} ({product}) is marked Delivered, but lacks a verified delivery date record. "
                    "Eligibility cannot be calculated without confirmed delivery date."
                ),
            )
        try:
            deliv_date = datetime.strptime(raw_delivery.strip(), "%Y-%m-%d").date()
        except ValueError:
            return EligibilityAssessment(
                verdict=VERDICT_INDETERMINATE_DELIVERY_DATE,
                order_id=order_id,
                product=product,
                order_status=status,
                delivery_date=None,
                reference_date=ref_date,
                days_since_delivery=None,
                return_window_days=None,
                policy_source=None,
                policy_citation=None,
                conditions=[],
                explanation=(
                    f"Order {order_id} ({product}) has an unparseable delivery date '{raw_delivery}'. "
                    "Eligibility cannot be reliably determined."
                ),
            )

        if deliv_date > ref_date:
            return EligibilityAssessment(
                verdict=VERDICT_INDETERMINATE_DELIVERY_DATE,
                order_id=order_id,
                product=product,
                order_status=status,
                delivery_date=deliv_date,
                reference_date=ref_date,
                days_since_delivery=None,
                return_window_days=None,
                policy_source=None,
                policy_citation=None,
                conditions=[],
                explanation=(
                    f"Order {order_id} ({product}) is marked Delivered with delivery date {deliv_date}, "
                    f"which is in the future relative to {ref_date}. Delivery verification is required."
                ),
            )
    else:
        # Unknown status
        return EligibilityAssessment(
            verdict=VERDICT_INDETERMINATE_DELIVERY_DATE,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=None,
            reference_date=ref_date,
            days_since_delivery=None,
            return_window_days=None,
            policy_source=None,
            policy_citation=None,
            conditions=[],
            explanation=f"Order {order_id} ({product}) has status '{status}', which cannot be evaluated for return.",
        )

    # Step 4: Retrieve policy text
    if custom_policy_text is not None:
        policy_text = custom_policy_text
        policy_source = custom_policy_source or "Provided Policy"
    else:
        policy_text, source_type, source_name = retrieve_policy_context(
            session_id=session_id,
            document_id=document_id,
        )
        policy_source = source_name or "Policy Documentation"

    if not policy_text or not policy_text.strip():
        return EligibilityAssessment(
            verdict=VERDICT_MISSING_POLICY,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=(ref_date - deliv_date).days,
            return_window_days=None,
            policy_source=None,
            policy_citation=None,
            conditions=[],
            explanation=(
                f"Order {order_id} ({product}) was delivered on {deliv_date}, but no return policy document "
                "or FAQ was found to evaluate return eligibility."
            ),
        )

    # Step 5: Extract return rules from policy
    rule_status, window_days, conditions, citation = extract_policy_return_rules(policy_text, product_name=product)

    if rule_status == "non_returnable":
        return EligibilityAssessment(
            verdict=VERDICT_NON_RETURNABLE,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=(ref_date - deliv_date).days,
            return_window_days=0,
            policy_source=policy_source,
            policy_citation=citation,
            conditions=conditions,
            explanation=(
                f"Order {order_id} ({product}) was delivered on {deliv_date}, but the applicable policy "
                f"({policy_source}) states that items are non-refundable/all sales are final."
            ),
        )

    if rule_status == "conflicting":
        return EligibilityAssessment(
            verdict=VERDICT_CONFLICTING_POLICY,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=(ref_date - deliv_date).days,
            return_window_days=None,
            policy_source=policy_source,
            policy_citation=citation,
            conditions=conditions,
            explanation=(
                f"The applicable return policy ({policy_source}) contains conflicting return windows ({citation}). "
                "Eligibility cannot be safely decided without human clarification."
            ),
        )

    if rule_status == "missing" or window_days is None:
        return EligibilityAssessment(
            verdict=VERDICT_INDETERMINATE_POLICY,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=(ref_date - deliv_date).days,
            return_window_days=None,
            policy_source=policy_source,
            policy_citation=None,
            conditions=conditions,
            explanation=(
                f"The policy document ({policy_source}) does not specify a valid return window. "
                "Cannot determine return eligibility."
            ),
        )

    # Step 6: Deterministic date math
    days_elapsed = (ref_date - deliv_date).days

    cond_str = f" The item must be in {', '.join(conditions)}." if conditions else ""

    if days_elapsed <= window_days:
        days_remaining = window_days - days_elapsed
        explanation = (
            f"Yes, Order {order_id} ({product}) qualifies for a return. "
            f"It was delivered on {deliv_date} ({days_elapsed} day{'s' if days_elapsed != 1 else ''} ago), "
            f"which is within the {window_days}-day return window from {policy_source}. "
            f"You have {days_remaining} day{'s' if days_remaining != 1 else ''} remaining to initiate the return.{cond_str}"
        )
        return EligibilityAssessment(
            verdict=VERDICT_ELIGIBLE,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=days_elapsed,
            return_window_days=window_days,
            policy_source=policy_source,
            policy_citation=citation,
            conditions=conditions,
            explanation=explanation,
        )
    else:
        days_expired = days_elapsed - window_days
        explanation = (
            f"No, Order {order_id} ({product}) does not qualify for a return. "
            f"It was delivered on {deliv_date} ({days_elapsed} days ago), exceeding the {window_days}-day return window "
            f"from {policy_source} by {days_expired} day{'s' if days_expired != 1 else ''}."
        )
        return EligibilityAssessment(
            verdict=VERDICT_EXPIRED,
            order_id=order_id,
            product=product,
            order_status=status,
            delivery_date=deliv_date,
            reference_date=ref_date,
            days_since_delivery=days_elapsed,
            return_window_days=window_days,
            policy_source=policy_source,
            policy_citation=citation,
            conditions=conditions,
            explanation=explanation,
        )
