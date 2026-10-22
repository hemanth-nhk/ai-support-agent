"""
Unit tests for app.services.order_policy_service.
Verifies all eligibility outcomes, deterministic date calculations,
safeguards against missing or conflicting policies, and reference_date injection.
"""
from datetime import date
from unittest.mock import patch
import pytest

from app.database.seed import seed
from app.services import order_policy_service as ops


@pytest.fixture(autouse=True)
def setup_db():
    seed(force=True)


def test_assess_eligibility_eligible_within_window():
    """Order 1002 delivered on 2026-09-05 evaluated on 2026-09-15 with 30-day window is ELIGIBLE."""
    custom_policy = (
        "Return Policy: Items can be returned within 30 days of delivery for a full refund. "
        "Products must be in their original packaging and unused condition."
    )
    res = ops.assess_return_eligibility(
        order_id=1002,
        reference_date=date(2026, 9, 15),
        custom_policy_text=custom_policy,
        custom_policy_source="Corporate Policy",
    )
    assert res.verdict == ops.VERDICT_ELIGIBLE
    assert res.days_since_delivery == 10
    assert res.return_window_days == 30
    assert "original packaging" in res.conditions
    assert "unused condition" in res.conditions
    assert "qualifies for a return" in res.explanation
    assert "20 days remaining" in res.explanation


def test_assess_eligibility_expired_window():
    """Order 1002 delivered on 2026-09-05 evaluated on 2026-10-15 (40 days later) with 30-day window is EXPIRED."""
    custom_policy = "Items can be returned within 30 days of delivery."
    res = ops.assess_return_eligibility(
        order_id=1002,
        reference_date=date(2026, 10, 15),
        custom_policy_text=custom_policy,
        custom_policy_source="Store Policy",
    )
    assert res.verdict == ops.VERDICT_EXPIRED
    assert res.days_since_delivery == 40
    assert res.return_window_days == 30
    assert "does not qualify for a return" in res.explanation
    assert "exceeding the 30-day return window" in res.explanation


def test_assess_eligibility_not_yet_delivered_shipped():
    """Order 1001 is Shipped and has not been delivered yet."""
    res = ops.assess_return_eligibility(
        order_id=1001,
        reference_date=date(2026, 9, 15),
    )
    assert res.verdict == ops.VERDICT_NOT_YET_DELIVERED
    assert res.order_status == "Shipped"
    assert "has not yet been delivered" in res.explanation
    assert res.delivery_date is None


def test_assess_eligibility_not_yet_delivered_processing():
    """Order 1003 is Processing."""
    res = ops.assess_return_eligibility(
        order_id=1003,
        reference_date=date(2026, 9, 15),
    )
    assert res.verdict == ops.VERDICT_NOT_YET_DELIVERED
    assert res.order_status == "Processing"
    assert "has not yet been delivered" in res.explanation


def test_assess_eligibility_cancelled_order():
    """Order marked Cancelled."""
    with patch("app.services.order_service.get_order", return_value={
        "id": 9001, "product": "Headphones", "status": "Cancelled", "estimated_delivery": None,
    }):
        res = ops.assess_return_eligibility(
            order_id=9001,
            reference_date=date(2026, 9, 15),
        )
        assert res.verdict == ops.VERDICT_CANCELLED
        assert "was cancelled and is not eligible for return" in res.explanation


def test_assess_eligibility_unknown_order():
    """Non-existent order produces UNKNOWN_ORDER verdict without crashing."""
    res = ops.assess_return_eligibility(
        order_id=9999,
        reference_date=date(2026, 9, 15),
    )
    assert res.verdict == ops.VERDICT_UNKNOWN_ORDER
    assert "was not found in our system" in res.explanation


def test_assess_eligibility_missing_policy():
    """Delivered order with no policy found returns MISSING_POLICY."""
    with patch("app.services.order_policy_service.retrieve_policy_context", return_value=(None, None, None)):
        res = ops.assess_return_eligibility(
            order_id=1002,
            reference_date=date(2026, 9, 15),
        )
        assert res.verdict == ops.VERDICT_MISSING_POLICY
        assert "no return policy document or FAQ was found" in res.explanation


def test_assess_eligibility_conflicting_policy():
    """Policy containing multiple conflicting return windows (e.g. 14 days and 45 days) returns CONFLICTING_POLICY."""
    conflicting_policy = (
        "General items have a 14-day return window. However, special items can be returned within 45 days."
    )
    res = ops.assess_return_eligibility(
        order_id=1002,
        reference_date=date(2026, 9, 15),
        custom_policy_text=conflicting_policy,
        custom_policy_source="Draft Policy",
    )
    assert res.verdict == ops.VERDICT_CONFLICTING_POLICY
    assert "conflicting return windows" in res.explanation


def test_assess_eligibility_non_returnable():
    """Policy explicitly stating all sales final or non-refundable returns NON_RETURNABLE."""
    final_sale_policy = "All clearance and promotional sales are final. Items are non-refundable."
    res = ops.assess_return_eligibility(
        order_id=1002,
        reference_date=date(2026, 9, 15),
        custom_policy_text=final_sale_policy,
        custom_policy_source="Clearance Policy",
    )
    assert res.verdict == ops.VERDICT_NON_RETURNABLE
    assert "non-refundable/all sales are final" in res.explanation


def test_assess_eligibility_indeterminate_delivery_date():
    """Delivered order with missing or unparseable delivery date returns INDETERMINATE_DELIVERY_DATE."""
    with patch("app.services.order_service.get_order", return_value={
        "id": 1002, "product": "Wireless Mouse", "status": "Delivered", "estimated_delivery": "",
    }):
        res = ops.assess_return_eligibility(
            order_id=1002,
            reference_date=date(2026, 9, 15),
        )
        assert res.verdict == ops.VERDICT_INDETERMINATE_DELIVERY_DATE
        assert "lacks a verified delivery date record" in res.explanation


def test_assess_eligibility_future_delivery_date_rejected():
    """Delivered order claiming a future delivery date returns INDETERMINATE_DELIVERY_DATE."""
    with patch("app.services.order_service.get_order", return_value={
        "id": 1002, "product": "Wireless Mouse", "status": "Delivered", "estimated_delivery": "2099-01-01",
    }):
        res = ops.assess_return_eligibility(
            order_id=1002,
            reference_date=date(2026, 9, 15),
        )
        assert res.verdict == ops.VERDICT_INDETERMINATE_DELIVERY_DATE
        assert "in the future" in res.explanation


def test_assess_eligibility_indeterminate_policy_no_window():
    """Policy text discussing returns without any window number returns INDETERMINATE_POLICY."""
    vague_policy = "Returns must be sent to our central warehouse in good condition."
    res = ops.assess_return_eligibility(
        order_id=1002,
        reference_date=date(2026, 9, 15),
        custom_policy_text=vague_policy,
        custom_policy_source="Shipping Info",
    )
    assert res.verdict == ops.VERDICT_INDETERMINATE_POLICY
    assert "does not specify a valid return window" in res.explanation


def test_policy_context_precedence_uploaded_over_faq():
    """Uploaded session document takes precedence over FAQ fallback."""
    with patch("app.services.document_service.get_session_documents", return_value=[{"document_id": "doc-1", "filename": "custom_rules.txt"}]), \
         patch("app.services.document_service.retrieve_document_chunks", return_value=[{"text": "Custom policy: 60-day return window", "metadata": {"filename": "custom_rules.txt"}}]), \
         patch("app.rag.retriever.retrieve_faq") as mock_faq:
        text, src_type, src_name = ops.retrieve_policy_context(session_id="test_sess_123")
        assert src_type == "uploaded_document"
        assert src_name == "custom_rules.txt"
        assert "60-day return window" in text
        mock_faq.assert_not_called()
