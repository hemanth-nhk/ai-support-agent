"""
Data models and taxonomy for the Real-World File Reliability & Adversarial Benchmark.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class BenchmarkCaseResult(BaseModel):
    case_id: str
    category: str
    file_type: str
    scenario: str
    expected_behavior: str
    actual_behavior: str
    passed: bool
    failure_class: Optional[str] = None
    severity: Optional[str] = None
    latency_ms: Optional[float] = None
    notes: Optional[str] = None


class BenchmarkSummary(BaseModel):
    baseline_regression_tests: int = 239
    reliability_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0

    # Core Reliability & Adversarial Metrics (%)
    ingestion_success_rate: float = 0.0
    extraction_fidelity: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    groundedness: float = 0.0
    unsupported_question_refusal_accuracy: float = 0.0
    contradiction_detection_rate: float = 0.0
    prompt_injection_resistance: float = 0.0
    cross_session_leakage_rate: float = 0.0
    dataset_accuracy: float = 0.0

    # Severity counts
    p0_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    p3_count: int = 0

    failure_class_breakdown: dict[str, int] = Field(default_factory=dict)
    cases: list[BenchmarkCaseResult] = Field(default_factory=list)
