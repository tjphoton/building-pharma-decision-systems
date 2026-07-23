"""Typed contracts for the Chapter 16 commercial decision system."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SignalEvent(BaseModel):
    """A typed candidate event emitted by the deterministic monitor.

    The monitor reports an unusual pattern. It never diagnoses a cause, selects a tool, or
    recommends a budget move; those fields live downstream in the request and recommendation.
    """

    signal_id: str
    brand: str = "Roventra"
    metric: str
    observed_value: float
    expected_low: float
    expected_high: float
    measurement_window: str
    evidence_date: _date
    population: str
    geography: str
    source: str
    status: Literal["candidate", "confirmed", "dismissed", "superseded"] = "candidate"
    trigger_rule_version: str


class DecisionRequest(BaseModel):
    """The bounded commercial question recorded when a case opens.

    The requested ``proposed_move`` is kept separate from the recommendation the agent selects
    later. New case-link fields default so earlier construction paths keep working during the
    staged migration.
    """

    question: str
    requesting_role: str
    outcome: str = "incremental NRx"
    audience: str
    geography: str
    budget_source: str
    budget_destination: str
    proposed_move: str
    decision_date: str
    deadline: str = "10 business days"
    risk_tolerance: Literal["low", "moderate", "high"] = "moderate"
    reversibility_required: bool
    approvers: list[str] = Field(min_length=1)
    # Case-link and provenance fields (Section 18.2). Optional for backward compatibility.
    case_id: Optional[str] = None
    signal_id: Optional[str] = None
    business_reason: Optional[str] = None
    evidence_date: Optional[_date] = None
    decision_on: Optional[_date] = None
    data_access_classification: Literal["approved", "restricted"] = "approved"
    requested_by: Optional[str] = None
    confirmed_at: Optional[str] = None
    capability_scope: Literal["commercial_allocation", "outside"] = "commercial_allocation"
    boundary_exception_requested: bool = False
    generated_python_requested: bool = False


class EvidenceItem(BaseModel):
    """One sourced result returned by a governed analytical tool."""

    evidence_id: str
    claim: str
    source: str
    entity_level: str
    window: str
    estimate: str
    uncertainty: str
    method: str
    causal_status: Literal["descriptive", "associational", "causal"]
    data_quality: str
    citation: str
    population: str = "Roventra commercial population"
    geography: str = "US"
    availability_date: str = "current decision date"


class Hypothesis(BaseModel):
    id: str
    statement: str
    status: Literal["open", "supported", "refuted", "isolated"]


class AdHocQuery(BaseModel):
    """A read-only query written by the investigator for an uncovered question."""

    purpose: str
    sql: str


class InvestigatorFraming(BaseModel):
    decision_summary: str
    hypotheses: list[Hypothesis]
    requested_tools: list[str]
    ad_hoc_queries: list[AdHocQuery] = Field(default_factory=list)
    expected_information: str


class InvestigatorIntegration(BaseModel):
    evidence_conflicts: list[str]
    marginal_return_read: str
    sufficiency: Literal["sufficient_for_test", "sufficient_for_scale", "insufficient"]
    open_questions: list[str]
    remaining_uncertainty: str


class DecisionOption(BaseModel):
    """An agent-generated action assembled from approved building blocks."""

    name: str
    description: str
    budget_moved_usd: int = Field(ge=0)
    audience: Literal["community_stable", "all_endocrinologists", "hold"]
    geography: Literal["matched_markets", "stable_access_dmas", "all_dmas", "none"]
    duration_weeks: int = Field(ge=0, le=26)
    reversibility: Literal["high", "staged", "low"]
    is_experiment: bool
    measurement_design: Literal["matched_market", "outcome_monitor", "none"]


class OptionSet(BaseModel):
    options: list[DecisionOption]
    assumptions: list[str]

    @model_validator(mode="after")
    def require_choice(self) -> "OptionSet":
        if len(self.options) < 2:
            raise ValueError("The analyst must propose at least two options.")
        names = [option.name for option in self.options]
        if len(names) != len(set(names)):
            raise ValueError("Option names must be unique.")
        return self


class ScenarioResult(BaseModel):
    option_name: str
    feasible: bool
    budget_moved_usd: int
    expected_incr_nrx_low: int
    expected_incr_nrx_mid: int
    expected_incr_nrx_high: int
    downside_nrx: int
    audience_hcp_count: int
    learning_value: Literal["none", "moderate", "high"]
    constraint_notes: list[str]
    calculation: str
    durability_assumption: str


class ExperimentDesign(BaseModel):
    eligible_dmas: int
    test_dmas: int
    holdout_dmas: int
    duration_weeks: int
    primary_outcome: str
    minimum_detectable_effect: float
    estimated_cost_usd: int
    readout_date: str


class AnalystOutput(BaseModel):
    selected_option_name: str
    rationale: str
    evidence_ids: list[str]
    evidence_that_would_change_selection: str
    measurement_plan: str


class ReviewerOutput(BaseModel):
    findings: list[str]
    unsupported_claims: list[str]
    causal_overstatement: Optional[str] = None
    disposition: Literal["pass", "revise_options", "revise_investigation", "escalate"]
    required_revision: Optional[str] = None


class ValidationResult(BaseModel):
    status: Literal["pass", "fail"]
    checks: list[str]
    issues: list[str]


class HumanDisposition(BaseModel):
    decision: Literal["approve", "edit", "reject", "request_more"]
    reviewer: str
    reason: str
    timestamp: str = "2026-07-14T16:00:00Z"
    # For the edit path: the reviewer's replacement option, re-simulated and revalidated.
    edited_option: Optional["DecisionOption"] = None


class GroundedInsight(BaseModel):
    observation: str
    inference: str
    recommendation: str
    uncertainty: str
    evidence_ids: list[str]
    revisit_date: str


class RoleView(BaseModel):
    role: Literal["brand leadership", "measurement", "omnichannel execution", "finance"]
    headline: str
    detail: str


class LearningSummary(BaseModel):
    expected_range: str
    observed_result: str
    changed_evidence: str
    decision_implication: str
    remaining_uncertainty: str


class EvaluationResult(BaseModel):
    scenario: str
    recommendation: str
    passed: bool
    reason: str


# --- Runtime, metering, and case-link contracts (Section 18) ------------------------------


class RuntimeLimits(BaseModel):
    """Bounded budgets the runtime meters on every LLM or tool call (Section 18.3)."""

    max_tool_calls: int = 10
    max_ad_hoc_queries: int = 3
    max_revisions: int = 2
    max_llm_steps: int = 11
    max_output_tokens_per_call: int = 4000
    max_elapsed_seconds: int = 180
    max_cost_usd: float = 1.00
    max_structured_repairs_per_call: int = 1


class PricingSnapshot(BaseModel):
    """Injected pricing used to estimate cost. Kept in run metadata for reproducibility."""

    model_id: str
    input_per_mtok_usd: float
    output_per_mtok_usd: float
    currency: str = "USD"
    effective_date: str
    pricing_version: str = "ch16-pricing-v1"


class RunMetadata(BaseModel):
    """Provenance and metering for one execution of the graph (Section 18.4)."""

    run_id: str
    case_id: str
    thread_id: str
    started_at: str
    updated_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    current_node: Optional[str] = None
    pause_reason: Optional[str] = None
    llm_steps: int = 0
    tool_calls: int = 0
    ad_hoc_queries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    elapsed_ms: int = 0
    retry_count: int = 0
    model_id: str = ""
    prompt_version: str = ""
    schema_version: str = "ch16-schema-v2"
    graph_version: str = ""
    tool_version: str = ""
    data_version: str = ""
    pricing_version: str = ""
    benchmark_version: Optional[str] = None


class ToolCallRecord(BaseModel):
    """One governed tool or ad hoc query, accounted for auditing and metering (Section 21.4)."""

    tool_name: str
    purpose: str
    started_at: str
    ended_at: str
    status: Literal["success", "blocked", "failed"]
    accessed_sources: list[str] = Field(default_factory=list)
    row_count: int = 0
    result_hash: str = ""
    elapsed_ms: float = 0.0
    error_class: Optional[str] = None
    sql: Optional[str] = None


class RunInterrupt(BaseModel):
    """A recorded pause: a human gate, an exhausted budget, or a provider failure."""

    kind: Literal[
        "human_approval", "budget_exhausted", "provider_failure",
        "tool_failure", "restricted_data", "boundary_exception", "generated_python",
    ]
    reason: str
    node: str
    at: str


class OutcomeEvent(BaseModel):
    """A later observed result that reopens the original case (Section 18.5)."""

    outcome_id: str
    case_id: str
    decision_id: str
    available_date: _date
    measurement_window: str
    observed_incremental_nrx: int
    confidence_low: int
    confidence_high: int
    population: str
    geography: str
    source: str
    source_version: str
    maturity_status: Literal["immature", "maturing", "mature"]
    quality_notes: str = ""


class PriorDecisionRecord(BaseModel):
    """The persisted first-date decision the later run loads by case ID (Section 20.4)."""

    decision_id: str
    case_id: str
    selected_option_name: str
    action_description: str
    budget_moved_usd: int
    expected_incr_nrx_low: int
    expected_incr_nrx_high: int
    evidence_ids: list[str]
    measurement_plan: str
    approval_reviewer: str
    approval_reason: str
    approved_at: str
    expected_outcome_window: str
    next_review_date: str


class CaseRecord(BaseModel):
    """The durable case that spans the first and later Roventra decisions (Section 20.1)."""

    case_id: str
    signal: SignalEvent
    request: DecisionRequest
    created_at: str
    status: Literal["open", "decided", "reopened", "closed"] = "open"
    prior_decision: Optional[PriorDecisionRecord] = None
    outcome_event: Optional[OutcomeEvent] = None


class RunSummary(BaseModel):
    run_id: str
    case_id: str
    mode: Literal["mock", "saved", "live"]
    status: str
    current_node: Optional[str] = None
    started_at: str
    updated_at: str


class RunStatus(BaseModel):
    """A compact view of a run returned by the runtime service and the HTTP layer."""

    run_id: str
    case_id: str
    status: str
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    pause_reason: Optional[str] = None
    awaiting_disposition: bool = False
    message: str = ""
