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
INTENT_DOCUMENT_QUERY = "DOCUMENT_QUERY"
INTENT_DATASET_ANALYSIS = "DATASET_ANALYSIS"
INTENT_ORDER_POLICY_CHECK = "ORDER_POLICY_CHECK"

_ORDER_POLICY_CHECK_PATTERNS = re.compile(
    r"\b(?:"
    r"qualif(?:y|ies|ied|ication)|"
    r"eligib(?:le|ility)|"
    r"within\s+(?:the\s+)?(?:return\s+)?window|"
    r"(?:return|refund)\s+(?:policy|window|rules?)\s+(?:for|on)\s+(?:order|the\s+order)|"
    r"can\s+(?:order\s*#?\s*\d+|it)\s+be\s+returned\s+(?:under|according\s+to|based\s+on)|"
    r"does\s+(?:order\s*#?\s*\d+|it)\s+meet\s+(?:the\s+)?(?:return\s+)?policy|"
    r"check\s+(?:if\s+(?:order\s*#?\s*\d+|it)\s+can\s+be\s+returned|return\s+eligibility)|"
    r"return\s+window\s+(?:for|on)\s+(?:order|the\s+order)"
    r")\b",
    re.IGNORECASE,
)

_ORDER_ID_EXPLICIT_PATTERN = re.compile(
    r"\b(?:order\s*(?:id\s*)?[:#]?\s*(?:ord[-_]?)?|ord[-_])(\d{3,6})\b",
    re.IGNORECASE,
)
_ORDER_ID_STANDALONE_PATTERN = re.compile(
    r"(?:^|[^\w])(?:ord[-_]?)?#?(\d{3,6})(?:$|[^\w])",
    re.IGNORECASE,
)
_CUSTOMER_ID_EXPLICIT_PATTERN = re.compile(r"\bcustomer\s*(?:id\s*)?#?\s*(\d{1,6})\b", re.IGNORECASE)
_CUSTOMER_ID_STANDALONE_PATTERN = re.compile(r"(?:^|[^\d])#?(\d{1,6})(?:$|[^\d])")
_DOCUMENT_ID_PATTERN = re.compile(
    r"\b(?:document\s*(?:id)?\s*[:#]?\s*|doc\s*[:#]?\s*)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b|"
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_DATASET_ID_PATTERN = re.compile(
    r"\b(?:dataset\s*(?:id)?\s*[:#]?\s*)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_DATASET_EXPLICIT_PATTERN = re.compile(
    r"\b(?:in|from|about)\s+(?:the|this|my|uploaded)\s+(?:dataset|data|table|csv|file)\b|"
    r"\b(?:dataset|uploaded\s+csv|uploaded\s+table)\b|"
    r"\b(?:analyze|query|plot|chart)\s+(?:the\s+)?(?:dataset|data|csv|table)\b",
    re.IGNORECASE,
)
_DATASET_ANALYTIC_PATTERNS = re.compile(
    r"\b(?:"
    r"average|averages|avg|mean|"
    r"total|totals|sum|sums|count|counts|"
    r"minimum|min|maximum|max|highest|lowest|top|bottom|most|least|"
    r"trend|trends|"
    r"group|groups|grouping|grouped|"
    r"compare|comparing|comparison|comparative|"
    r"percentage|percentages|percent|%"
    r"metric|metrics|"
    r"row|rows|column|columns|"
    r"distribution|distributions|breakdown|breakdowns|"
    r"pivot|pivots|cross-?tabs?|cross-?tabulations?|"
    r"correlation|correlations|"
    r"rolling|cumulative|"
    r"outlier|outliers|anomaly|anomalies|deviation|deviations|quantile|quantiles|quartile|quartiles|"
    r"filter|filters|filtered|filtering|clear|reset|instead|"
    r"sort|sorted|sorting|rank|ranks|ranking|"
    r"plot|plots|plotting|chart|charts|graph|graphs|heatmap|heatmaps|"
    r"revenue|sales|profit|quantity|units|"
    r"per"
    r")\b|"
    r"\b(?:in|from|about)\s+(?:the|this|my|uploaded)\s+(?:dataset|data|table|csv|file)\b|"
    r"\b(?:dataset|data|table|csv)\b",
    re.IGNORECASE,
)
_SUPERLATIVE_PATTERNS = re.compile(
    r"\b(?:best|top|highest|greatest|leading|most\s+popular|most\s+successful|worst|lowest)\b",
    re.IGNORECASE,
)

_ORDER_TRACKING_PHRASES = (
    "where is my order", "where is my package", "where is my shipment", "where is my delivery",
    "track my order", "track order", "track package", "track my package", "track shipment", "track my shipment",
    "order status", "delivery status", "check order", "package status", "check my shipment", "check my package",
    "status of order", "status of my order", "status of the order", "status of shipment", "status of my shipment",
    "status of package", "status of my package", "status of delivery", "status of my delivery",
)
_ORDER_TRACKING_PATTERN = re.compile(
    r"\bstatus\s+of\s+(?:my\s+|the\s+)?(?:order|package|shipment|delivery)\b",
    re.IGNORECASE,
)
_ORDER_KEYWORDS = ("order", "package", "shipment", "delivery status", "track")
_CUSTOMER_ORDERS_KEYWORDS = (
    "my orders", "previous orders", "order history", "orders does customer",
    "ordered previously", "what have i ordered", "past orders",
    "how many orders", "orders do i have",
)
_DATA_QUERY_KEYWORDS = (
    "find orders", "search orders", "all orders", "orders between",
    "orders from", "orders on", "filter orders", "list orders",
)
_DATA_QUERY_PATTERN = re.compile(
    r"\b(?:find|search|filter|list)\s+orders?\b|\borders?\s+(?:between|from|on|after|before|since|dated)\b",
    re.IGNORECASE,
)
_DOCUMENT_PATTERN = re.compile(
    r"\b(?:uploaded|attached)\s+(?:document|file|pdf|docx|txt)\b|"
    r"\b(?:document|file|pdf|docx|txt)\s+i\s+uploaded\b|"
    r"\b(?:in|from|about)\s+(?:the|this|my|uploaded)\s+(?:document|file|pdf|docx|txt)\b|"
    r"\baccording\s+to\s+(?:the|this|my|uploaded)\s+(?:document|file|pdf|docx|txt)\b|"
    r"\bsummarize\s+(?:this|the|my|uploaded)?\s*(?:document|file|pdf|docx|txt)\b|"
    r"\bwhat\s+does\s+(?:the|this|my|the\s+uploaded)\s+(?:document|file|pdf)\s+say\b|"
    r"\b(?:this|my)\s+(?:document|file|pdf)\b",
    re.IGNORECASE,
)
_FAQ_KEYWORDS = (
    "shipping", "return", "refund", "cancel", "cancellation", "policy",
    "how long does", "how do i return", "money back", "warranty", "period", "express delivery",
    "send back", "exchange",
)
_GENERAL_PATTERN = re.compile(
    r"\b(?:hello|hi|hey|who are you|what can you|help me|thanks|thank you)\b",
    re.IGNORECASE,
)

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


def extract_document_id(text: str) -> Optional[str]:
    """
    Extracts a document UUID from text if explicitly mentioned.
    """
    match = _DOCUMENT_ID_PATTERN.search(text)
    if match:
        return match.group(1) or match.group(2)
    return None


def extract_dataset_id(text: str) -> Optional[str]:
    """
    Extracts a dataset UUID from text if explicitly mentioned.
    """
    match = _DATASET_ID_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def classify_intent(
    message: str,
    previous_state: dict | None = None,
    document_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    workspace_mode: Optional[str] = None,
) -> dict:
    """
    Returns a dict:
        {
            "intent": str,
            "order_id": Optional[int],
            "customer_id": Optional[int],
            "document_id": Optional[str],
            "dataset_id": Optional[str],
            "confidence": float,
            "candidate_intents": list[str]
        }
    Deterministic -- same inputs always yield the same output.
    """
    text = message.lower().strip()

    active_mode = workspace_mode
    if not active_mode and previous_state:
        active_mode = previous_state.get("workspace_mode")
    if active_mode:
        active_mode = active_mode.lower().strip()

    active_document_id = (
        document_id
        or extract_document_id(text)
        or (previous_state.get("document_id") if previous_state else None)
    )
    if isinstance(active_document_id, str):
        active_document_id = active_document_id.strip() or None

    active_dataset_id = (
        dataset_id
        or extract_dataset_id(text)
        or (previous_state.get("dataset_id") if previous_state else None)
    )
    if isinstance(active_dataset_id, str):
        active_dataset_id = active_dataset_id.strip() or None

    if not text:
        return {
            "intent": INTENT_OUT_OF_SCOPE,
            "order_id": None,
            "customer_id": None,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 1.0,
            "candidate_intents": [INTENT_OUT_OF_SCOPE],
        }

    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in text:
            return {
                "intent": INTENT_OUT_OF_SCOPE,
                "order_id": None,
                "customer_id": None,
                "document_id": active_document_id,
                "dataset_id": active_dataset_id,
                "confidence": 1.0,
                "candidate_intents": [INTENT_OUT_OF_SCOPE],
            }

    # Step 4: Check if previous turn was actively awaiting an entity
    if previous_state and previous_state.get("awaiting_entity"):
        awaiting = previous_state.get("awaiting_entity")
        if awaiting == "order_id":
            found_order_id = extract_order_id(text, standalone_allowed=True)
            if found_order_id:
                target_intent = (
                    INTENT_ORDER_POLICY_CHECK
                    if previous_state.get("intent") == INTENT_ORDER_POLICY_CHECK
                    else INTENT_ORDER_STATUS
                )
                return {
                    "intent": target_intent,
                    "order_id": found_order_id,
                    "customer_id": None,
                    "document_id": active_document_id,
                    "dataset_id": active_dataset_id,
                    "confidence": 1.0,
                    "candidate_intents": [target_intent],
                }
        elif awaiting == "customer_id":
            found_customer_id = extract_customer_id(text, standalone_allowed=True)
            if found_customer_id:
                return {
                    "intent": INTENT_CUSTOMER_ORDERS,
                    "order_id": None,
                    "customer_id": found_customer_id,
                    "document_id": active_document_id,
                    "dataset_id": active_dataset_id,
                    "confidence": 1.0,
                    "candidate_intents": [INTENT_CUSTOMER_ORDERS],
                }
        elif awaiting == "dataset_metric":
            pending = previous_state.get("pending_analysis") or {}
            candidate_metrics = pending.get("candidate_metrics", [])
            matched_metric = None
            for m in candidate_metrics:
                if m.lower() in text:
                    matched_metric = m
                    break
            if matched_metric:
                return {
                    "intent": INTENT_DATASET_ANALYSIS,
                    "order_id": None,
                    "customer_id": None,
                    "document_id": active_document_id,
                    "dataset_id": active_dataset_id,
                    "confidence": 1.0,
                    "candidate_intents": [INTENT_DATASET_ANALYSIS],
                    "resolved_metric": matched_metric,
                    "pending_analysis": pending,
                }
        elif awaiting == "document_topic":
            return {
                "intent": INTENT_DOCUMENT_QUERY,
                "order_id": None,
                "customer_id": None,
                "document_id": active_document_id,
                "dataset_id": active_dataset_id,
                "confidence": 1.0,
                "candidate_intents": [INTENT_DOCUMENT_QUERY],
            }

    # Check if previous turn was clarification from order/policy ambiguity
    if previous_state and previous_state.get("intent") == INTENT_AMBIGUOUS:
        prev_cand = previous_state.get("candidate_intents") or []
        if INTENT_FAQ in prev_cand and INTENT_ORDER_STATUS in prev_cand:
            prev_order_id = previous_state.get("order_id")
            policy_reply_keywords = (
                "policy", "return policy", "check policy", "review policy",
                "eligib", "qualif", "can i return", "return rule", "check return", "refund policy",
            )
            tracking_reply_keywords = ("track", "status", "delivery", "where is", "delivery status", "check status")
            if any(kw in text for kw in policy_reply_keywords):
                return {
                    "intent": INTENT_ORDER_POLICY_CHECK,
                    "order_id": prev_order_id,
                    "customer_id": None,
                    "document_id": active_document_id,
                    "dataset_id": active_dataset_id,
                    "confidence": 0.95,
                    "candidate_intents": [INTENT_ORDER_POLICY_CHECK],
                }
            elif any(kw in text for kw in tracking_reply_keywords):
                return {
                    "intent": INTENT_ORDER_STATUS,
                    "order_id": prev_order_id,
                    "customer_id": None,
                    "document_id": active_document_id,
                    "dataset_id": active_dataset_id,
                    "confidence": 0.95,
                    "candidate_intents": [INTENT_ORDER_STATUS],
                }

    # Standard entity extraction
    order_id = extract_order_id(text, standalone_allowed=False)
    customer_id = extract_customer_id(text, standalone_allowed=False)

    has_data_query_keywords = any(kw in text for kw in _DATA_QUERY_KEYWORDS)
    has_data_query_pattern = bool(_DATA_QUERY_PATTERN.search(text))
    has_customer_orders_keywords = any(kw in text for kw in _CUSTOMER_ORDERS_KEYWORDS)
    has_order_tracking_phrase = any(phrase in text for phrase in _ORDER_TRACKING_PHRASES) or bool(_ORDER_TRACKING_PATTERN.search(text))
    has_order_keywords = any(kw in text for kw in _ORDER_KEYWORDS)
    has_faq_keywords = any(kw in text for kw in _FAQ_KEYWORDS)
    has_general_keywords = bool(_GENERAL_PATTERN.search(text))
    has_document_pattern = bool(_DOCUMENT_PATTERN.search(text))
    has_order_policy_check = bool(_ORDER_POLICY_CHECK_PATTERNS.search(text))

    # Priority 1: DATA_QUERY (never overridden by attached document or dataset)
    if has_data_query_keywords or has_data_query_pattern:
        return {
            "intent": INTENT_DATA_QUERY,
            "order_id": order_id,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_DATA_QUERY],
        }

    # Priority 2: CUSTOMER_ORDERS (never overridden by attached document or dataset)
    if customer_id and has_customer_orders_keywords:
        return {
            "intent": INTENT_CUSTOMER_ORDERS,
            "order_id": None,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_CUSTOMER_ORDERS],
        }

    if has_customer_orders_keywords and not order_id and not has_data_query_keywords:
        return {
            "intent": INTENT_CUSTOMER_ORDERS,
            "order_id": None,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.9,
            "candidate_intents": [INTENT_CUSTOMER_ORDERS],
        }

    # Priority 2.5: Explicit ORDER_POLICY_CHECK (order return eligibility under policy)
    if has_order_policy_check and not has_data_query_keywords:
        return {
            "intent": INTENT_ORDER_POLICY_CHECK,
            "order_id": order_id,
            "customer_id": None,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95 if order_id else 0.85,
            "candidate_intents": [INTENT_ORDER_POLICY_CHECK],
        }

    # Priority 3: Ambiguity between Order Tracking and Policy/FAQ
    # (e.g. "Can I return order 1001?", "My order hasn't arrived and I want a refund.")
    has_order_signal = (order_id is not None) or has_order_tracking_phrase or (has_order_keywords and not has_customer_orders_keywords)
    if (order_id is not None or has_order_tracking_phrase or (has_order_keywords and not active_document_id and not active_dataset_id)) and has_order_signal and has_faq_keywords:
        return {
            "intent": INTENT_AMBIGUOUS,
            "order_id": order_id,
            "customer_id": None,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.5,
            "candidate_intents": [INTENT_FAQ, INTENT_ORDER_STATUS],
        }

    # Priority 4: Explicit ORDER_STATUS (explicit order ID or dedicated tracking phrase)
    if order_id is not None or has_order_tracking_phrase:
        return {
            "intent": INTENT_ORDER_STATUS,
            "order_id": order_id,
            "customer_id": None,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95 if order_id else 0.85,
            "candidate_intents": [INTENT_ORDER_STATUS],
        }

    # Priority 5: Pure general greetings (do not route pure greetings to document or dataset)
    if has_general_keywords and not has_document_pattern and not has_faq_keywords:
        return {
            "intent": INTENT_GENERAL,
            "order_id": None,
            "customer_id": None,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.9,
            "candidate_intents": [INTENT_GENERAL],
        }

    # Check dataset signals
    has_dataset_signal = False
    candidate_num_metrics = []
    has_active_plan = False

    has_explicit_dataset = bool(_DATASET_EXPLICIT_PATTERN.search(text))
    has_active_plan = bool(previous_state and (previous_state.get("active_analysis_plan") or previous_state.get("analysis_plan")))
    has_analytic_signal = bool(_DATASET_ANALYTIC_PATTERNS.search(text)) or bool(_SUPERLATIVE_PATTERNS.search(text))

    if active_dataset_id is not None:
        dataset_meta = None
        try:
            from app.analytics import dataset_service
            dataset_meta = dataset_service.get_dataset(active_dataset_id)
        except Exception:
            pass

        if active_mode == "support":
            # In Support mode, dataset analysis is strictly opt-in via explicit dataset references
            # (e.g. 'in the dataset', 'analyze dataset', 'plot dataset').
            # Ordinary support questions mentioning words like 'product', 'status', 'payment', 'category'
            # MUST NEVER be misrouted to dataset analysis!
            if has_faq_keywords or order_id is not None or has_order_tracking_phrase:
                has_dataset_signal = False
            else:
                has_dataset_signal = has_explicit_dataset
        elif active_mode == "dataset":
            if dataset_meta:
                from app.analytics.compiler import classify_dtype
                col_names = [c.name.lower() for c in dataset_meta.profile.columns if len(c.name) >= 3]
                has_col_signal = any(re.search(rf"\b{re.escape(c)}\b", text) for c in col_names)
                candidate_num_metrics = [
                    c.name for c in dataset_meta.profile.columns
                    if classify_dtype(c.dtype, c.name) == "numeric"
                ]
                has_dataset_signal = has_col_signal or has_analytic_signal or has_active_plan or has_explicit_dataset
            else:
                has_dataset_signal = has_analytic_signal or has_active_plan or has_explicit_dataset
        else:
            # Mode not specified (e.g. direct API/unit test callers without workspace_mode):
            # If the query is a Support question (FAQ, order lookup, etc.) without explicit dataset reference or analytic query,
            # preserve Support intent!
            if (has_faq_keywords or order_id is not None or has_order_tracking_phrase) and not (has_explicit_dataset or has_analytic_signal):
                has_dataset_signal = False
            else:
                if dataset_meta:
                    from app.analytics.compiler import classify_dtype
                    col_names = [c.name.lower() for c in dataset_meta.profile.columns if len(c.name) >= 3]
                    has_col_signal = any(re.search(rf"\b{re.escape(c)}\b", text) for c in col_names)
                    candidate_num_metrics = [
                        c.name for c in dataset_meta.profile.columns
                        if classify_dtype(c.dtype, c.name) == "numeric"
                    ]
                    has_dataset_signal = has_col_signal or has_analytic_signal or has_active_plan or has_explicit_dataset
                else:
                    has_dataset_signal = has_analytic_signal or has_active_plan or has_explicit_dataset
    else:
        # No dataset attached:
        # If in Dataset Intelligence mode, analytical/dataset requests are routed to DATASET_ANALYSIS
        # so the agent can provide actionable guidance (prompt to upload/select a dataset).
        if active_mode == "dataset":
            if has_explicit_dataset or has_analytic_signal or not (has_faq_keywords or order_id is not None or has_order_tracking_phrase or has_general_keywords or has_document_pattern):
                has_dataset_signal = True
        elif has_explicit_dataset:
            # Mode absent or not support, but explicitly mentioning dataset
            has_dataset_signal = True

    # Priority 6A: Explicit document reference without analytical dataset signal -> DOCUMENT_QUERY
    # (e.g. "According to the uploaded document, what is the policy?", "Summarize this PDF")
    if has_document_pattern and not (active_dataset_id is not None and has_dataset_signal):
        return {
            "intent": INTENT_DOCUMENT_QUERY,
            "order_id": order_id,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_DOCUMENT_QUERY],
        }

    # Priority 6B: Tabular Dataset Query -> DATASET_ANALYSIS or AMBIGUOUS
    if has_dataset_signal:
        if active_dataset_id is None:
            return {
                "intent": INTENT_DATASET_ANALYSIS,
                "order_id": order_id,
                "customer_id": customer_id,
                "document_id": active_document_id,
                "dataset_id": None,
                "confidence": 0.95,
                "candidate_intents": [INTENT_DATASET_ANALYSIS],
            }
        has_superlative = bool(_SUPERLATIVE_PATTERNS.search(text))
        def _check_metric_specified(q: str, metrics: list[str]) -> bool:
            for m in metrics:
                m_low = m.lower()
                if m_low in q:
                    return True
                # Check tokens in metric name with len >= 3 (e.g. 'sales' or 'sale' in 'Total Sales')
                for tok in re.findall(r"\b\w+\b", m_low):
                    if len(tok) >= 3 and (
                        re.search(rf"\b{re.escape(tok)}\b", q)
                        or re.search(rf"\b{re.escape(tok.rstrip('s'))}\b", q)
                    ):
                        return True
            return False

        metric_specified = _check_metric_specified(text, candidate_num_metrics)

        if not has_active_plan and has_superlative and len(candidate_num_metrics) >= 2 and not metric_specified:
            return {
                "intent": INTENT_AMBIGUOUS,
                "order_id": None,
                "customer_id": None,
                "document_id": active_document_id,
                "dataset_id": active_dataset_id,
                "confidence": 0.5,
                "candidate_intents": [INTENT_DATASET_ANALYSIS],
                "awaiting_entity": "dataset_metric",
                "pending_analysis": {
                    "dataset_id": active_dataset_id,
                    "question": message,
                    "candidate_metrics": candidate_num_metrics,
                },
                "ambiguity_details": {
                    "type": "dataset_metric",
                    "question": message,
                    "candidate_metrics": candidate_num_metrics,
                },
            }

        return {
            "intent": INTENT_DATASET_ANALYSIS,
            "order_id": order_id,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_DATASET_ANALYSIS],
        }

    # Priority 6C: Active Document Attachment or explicit document pattern -> DOCUMENT_QUERY
    # (Handles document-specific factual/policy questions like DAT_11)
    if active_document_id is not None or has_document_pattern:
        return {
            "intent": INTENT_DOCUMENT_QUERY,
            "order_id": order_id,
            "customer_id": customer_id,
            "document_id": active_document_id,
            "dataset_id": active_dataset_id,
            "confidence": 0.95,
            "candidate_intents": [INTENT_DOCUMENT_QUERY],
        }

    # Priority 8 (No active attachment): Generic order keywords
    if has_order_keywords:
        return {
            "intent": INTENT_ORDER_STATUS,
            "order_id": order_id,
            "customer_id": None,
            "document_id": None,
            "dataset_id": active_dataset_id,
            "confidence": 0.8,
            "candidate_intents": [INTENT_ORDER_STATUS],
        }

    # Priority 9 (No active attachment): FAQ routing
    if has_faq_keywords:
        return {
            "intent": INTENT_FAQ,
            "order_id": None,
            "customer_id": None,
            "document_id": None,
            "dataset_id": active_dataset_id,
            "confidence": 0.9,
            "candidate_intents": [INTENT_FAQ],
        }

    # Priority 10 (No active attachment): General conversation
    if has_general_keywords:
        return {
            "intent": INTENT_GENERAL,
            "order_id": None,
            "customer_id": None,
            "document_id": None,
            "dataset_id": active_dataset_id,
            "confidence": 0.9,
            "candidate_intents": [INTENT_GENERAL],
        }

    # Default fallback: fail-closed out-of-scope
    return {
        "intent": INTENT_OUT_OF_SCOPE,
        "order_id": None,
        "customer_id": None,
        "document_id": None,
        "dataset_id": active_dataset_id,
        "confidence": 1.0,
        "candidate_intents": [INTENT_OUT_OF_SCOPE],
    }
