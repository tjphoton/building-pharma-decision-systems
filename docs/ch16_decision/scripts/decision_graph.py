"""LangGraph implementation for the bounded two-date decision system.

The graph keeps a fixed topology (frame, gather, integrate, propose, simulate, select,
validate, review, human approval, deliver). :func:`build_graph` takes its dependencies by
injection so tests can supply a fixed clock, a mock model, a temporary checkpoint database, a
failing tool runner, and low runtime limits. Every completed node is checkpointed, the run
pauses before human approval, and runtime budgets turn into explicit interrupts rather than
silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import agents
from agents import LLM
from config import (
    DATA_VERSION,
    DEFAULT_LIMITS,
    GRAPH_VERSION,
    LATER_DECISION_DATE,
    PROMPT_VERSION,
    TOOL_VERSION,
    pricing_for,
)
from data_access import SqlNotAllowed, query_approved_data
from decision_services import design_experiment, simulate_options
from delivery import build_grounded_insight, build_learning_summary, build_role_views
from models import (
    AdHocQuery,
    AnalystOutput,
    CaseRecord,
    DecisionOption,
    DecisionRequest,
    EvidenceItem,
    ExperimentDesign,
    GroundedInsight,
    HumanDisposition,
    Hypothesis,
    InvestigatorFraming,
    InvestigatorIntegration,
    LearningSummary,
    OutcomeEvent,
    PriorDecisionRecord,
    RunInterrupt,
    RunMetadata,
    RuntimeLimits,
    OptionSet,
    ReviewerOutput,
    RoleView,
    ScenarioResult,
    ToolCallRecord,
    ValidationResult,
    SignalEvent,
)
from tools import TOOL_CATALOG, run_tool
from validation import normalize_tool_requests, validate_recommendation

DecisionStatus = Literal[
    "open", "running", "pending_review", "approved", "edited", "rejected",
    "request_more", "escalated", "interrupted", "failed", "closed",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="local", case_id="local", thread_id="local",
        started_at=_now_iso(), updated_at=_now_iso(),
        model_id=agents.active_model_id(), prompt_version=PROMPT_VERSION,
        graph_version=GRAPH_VERSION, tool_version=TOOL_VERSION, data_version=DATA_VERSION,
    )


class DecisionState(BaseModel):
    request: DecisionRequest
    date_phase: str = "first"
    signal: Optional[SignalEvent] = None
    case_id: Optional[str] = None
    runtime_limits: RuntimeLimits = Field(default_factory=lambda: DEFAULT_LIMITS)
    run_metadata: RunMetadata = Field(default_factory=_local_metadata)
    framing: Optional[InvestigatorFraming] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_records: list[ToolCallRecord] = Field(default_factory=list)
    # Completed graph nodes in execution order. Repeated nodes are retained because a
    # revision loop is part of the observed trajectory, not an implementation detail.
    node_trace: list[str] = Field(default_factory=list)
    integration: Optional[InvestigatorIntegration] = None
    option_set: Optional[OptionSet] = None
    scenarios: list[ScenarioResult] = Field(default_factory=list)
    experiment: Optional[ExperimentDesign] = None
    analyst: Optional[AnalystOutput] = None
    validation: Optional[ValidationResult] = None
    review: Optional[ReviewerOutput] = None
    revision_count: int = 0
    prior_decision: Optional[PriorDecisionRecord] = None
    outcome_event: Optional[OutcomeEvent] = None
    human: Optional[HumanDisposition] = None
    insight: Optional[GroundedInsight] = None
    role_views: list[RoleView] = Field(default_factory=list)
    learning: Optional[LearningSummary] = None
    errors: list[str] = Field(default_factory=list)
    interrupts: list[RunInterrupt] = Field(default_factory=list)
    status: DecisionStatus = "open"


_STATE_MODELS = [
    DecisionRequest, SignalEvent, RuntimeLimits, RunMetadata, EvidenceItem, ToolCallRecord,
    Hypothesis, AdHocQuery, InvestigatorFraming, InvestigatorIntegration, DecisionOption,
    OptionSet, ScenarioResult, ExperimentDesign, AnalystOutput, ValidationResult,
    ReviewerOutput, HumanDisposition, GroundedInsight, RoleView, LearningSummary,
    PriorDecisionRecord, OutcomeEvent, RunInterrupt, CaseRecord, DecisionState,
]


@dataclass
class DecisionService:
    """Deterministic services injected into the graph (Section 21.1)."""

    simulate: Callable[[list[DecisionOption], str], list[ScenarioResult]] = simulate_options
    design_experiment: Callable[[DecisionOption], Optional[ExperimentDesign]] = design_experiment


DEFAULT_SERVICE = DecisionService()


def phase_for(state: DecisionState) -> str:
    """The two-date teaching switch, derived from the evidence date when one is present.

    A stable compatibility adapter: it maps a real ``evidence_date`` to the ``first`` or
    ``later`` label the analytical core still keys off, so callers pass dates, not labels.
    """
    evidence_date = state.request.evidence_date
    if evidence_date is not None:
        return "later" if evidence_date >= LATER_DECISION_DATE else "first"
    return state.date_phase


def check_budget(metadata: RunMetadata, limits: RuntimeLimits) -> Optional[str]:
    """Return the name of the first exhausted budget, or ``None`` when all are within limit."""
    if metadata.llm_steps >= limits.max_llm_steps:
        return "max_llm_steps"
    if metadata.tool_calls > limits.max_tool_calls:
        return "max_tool_calls"
    if metadata.ad_hoc_queries > limits.max_ad_hoc_queries:
        return "max_ad_hoc_queries"
    if metadata.estimated_cost_usd >= limits.max_cost_usd:
        return "max_cost_usd"
    if metadata.elapsed_ms / 1000.0 >= limits.max_elapsed_seconds:
        return "max_elapsed_seconds"
    return None


def _run_ad_hoc(query: AdHocQuery, records: list[ToolCallRecord]) -> EvidenceItem:
    started = _now_iso()
    try:
        result = query_approved_data(query.sql)
        preview = "; ".join(str(row) for row in result.rows[:3])
        if result.row_count > 3:
            preview += " ..."
        records.append(ToolCallRecord(
            tool_name="ad_hoc_sql", purpose=query.purpose, started_at=started, ended_at=_now_iso(),
            status="success", accessed_sources=result.accessed_objects, row_count=result.row_count,
            result_hash=result.result_hash, elapsed_ms=result.elapsed_ms, sql=result.sql,
        ))
        return EvidenceItem(
            evidence_id=f"SQL-{result.result_hash[:6]}",
            claim=f"{query.purpose}: {preview}" if preview else f"{query.purpose}: no rows",
            source="ad hoc SQL", entity_level="query result", window="as queried",
            estimate=f"{result.row_count} rows", uncertainty="raw query result",
            method="agent-written governed SQL", causal_status="descriptive",
            data_quality="fresh",
            citation=f"sql:{result.result_hash} objects={result.accessed_objects}",
        )
    except SqlNotAllowed as error:
        records.append(ToolCallRecord(
            tool_name="ad_hoc_sql", purpose=query.purpose, started_at=started, ended_at=_now_iso(),
            status="blocked", error_class="SqlNotAllowed", sql=query.sql,
        ))
        return EvidenceItem(
            evidence_id="SQL-REJECTED",
            claim=f"Query rejected: {query.purpose}. {error}", source="ad hoc SQL",
            entity_level="n/a", window="n/a", estimate="rejected", uncertainty="n/a",
            method="agent-written governed SQL", causal_status="descriptive",
            data_quality="blocked", citation="sql:rejected",
        )
    except Exception as error:
        records.append(ToolCallRecord(
            tool_name="ad_hoc_sql", purpose=query.purpose, started_at=started, ended_at=_now_iso(),
            status="failed", error_class=type(error).__name__, sql=query.sql,
        ))
        return EvidenceItem(
            evidence_id="SQL-ERROR",
            claim=f"Query failed: {query.purpose}. {type(error).__name__}", source="ad hoc SQL",
            entity_level="n/a", window="n/a", estimate="error", uncertainty="n/a",
            method="agent-written governed SQL", causal_status="descriptive",
            data_quality="error", citation="sql:error",
        )


def build_graph(
    llm: LLM,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    limits: Optional[RuntimeLimits] = None,
    tool_runner: Callable[[str, str], list[EvidenceItem]] = run_tool,
    decision_service: DecisionService = DEFAULT_SERVICE,
    clock: Callable[[], str] = _now_iso,
    on_event=None,
):
    """Compile the fixed graph with an injected checkpointer, budgets, and human approval."""

    limits = limits or DEFAULT_LIMITS
    pricing = pricing_for(llm.model)

    def emit(node: str, update: dict, state: DecisionState):
        update["node_trace"] = state.node_trace + [node]
        if on_event:
            on_event(node, update)
        return update

    def meter(state: DecisionState) -> RunMetadata:
        """Fold drained model usage into the run metadata and recompute cost."""
        meta = state.run_metadata.model_copy(deep=True)
        for usage in llm.drain_usage():
            meta.llm_steps += 1
            meta.input_tokens += usage.input_tokens
            meta.output_tokens += usage.output_tokens
        meta.estimated_cost_usd = round(
            meta.input_tokens / 1_000_000 * pricing.input_per_mtok_usd
            + meta.output_tokens / 1_000_000 * pricing.output_per_mtok_usd,
            6,
        )
        meta.updated_at = clock()
        return meta

    def frame(state: DecisionState) -> dict:
        phase = phase_for(state)
        history = agents.format_case_history(state.prior_decision, state.outcome_event)
        framing = agents.frame_decision(llm, state.request, phase, history)
        framing.requested_tools = normalize_tool_requests(framing.requested_tools)
        # The tool allow list is enforced in gather(), which blocks and logs any name that
        # is still unrecognized after normalization, rather than aborting the whole run here.
        meta = meter(state)
        meta.current_node = "frame"
        return emit(
            "frame", {"framing": framing, "run_metadata": meta, "status": "running"}, state
        )

    def gather(state: DecisionState) -> dict:
        phase = phase_for(state)
        collected: list[EvidenceItem] = []
        records = list(state.tool_records)
        meta = state.run_metadata.model_copy(deep=True)
        tool_budget = limits.max_tool_calls
        query_budget = limits.max_ad_hoc_queries
        for tool in state.framing.requested_tools:
            if tool not in TOOL_CATALOG:
                records.append(ToolCallRecord(
                    tool_name=tool, purpose="requested tool", started_at=clock(),
                    ended_at=clock(), status="blocked", error_class="UnknownTool",
                ))
                continue
            if meta.tool_calls >= tool_budget:
                records.append(ToolCallRecord(
                    tool_name=tool, purpose="requested tool", started_at=clock(),
                    ended_at=clock(), status="blocked", error_class="max_tool_calls",
                ))
                continue
            started = clock()
            try:
                items = tool_runner(tool, phase)
                collected.extend(items)
                records.append(ToolCallRecord(
                    tool_name=tool, purpose="requested tool", started_at=started,
                    ended_at=clock(), status="success",
                    accessed_sources=[tool], row_count=len(items),
                ))
            except Exception as error:
                records.append(ToolCallRecord(
                    tool_name=tool, purpose="requested tool", started_at=started,
                    ended_at=clock(), status="failed", error_class=type(error).__name__,
                ))
            meta.tool_calls += 1
        for query in state.framing.ad_hoc_queries:
            if meta.ad_hoc_queries >= query_budget or meta.tool_calls >= tool_budget:
                records.append(ToolCallRecord(
                    tool_name="ad_hoc_sql", purpose=query.purpose, started_at=clock(),
                    ended_at=clock(), status="blocked", error_class="max_ad_hoc_queries",
                    sql=query.sql,
                ))
                continue
            collected.append(_run_ad_hoc(query, records))
            meta.ad_hoc_queries += 1
            meta.tool_calls += 1
        meta.current_node = "gather"
        meta.updated_at = clock()
        return emit("gather", {"evidence": collected, "tool_records": records,
                               "run_metadata": meta}, state)

    def integrate(state: DecisionState) -> dict:
        history = agents.format_case_history(state.prior_decision, state.outcome_event)
        integration = agents.integrate_evidence(
            llm, state.request, state.evidence, phase_for(state), history
        )
        meta = meter(state)
        meta.current_node = "integrate"
        return emit("integrate", {"integration": integration, "run_metadata": meta}, state)

    def propose(state: DecisionState) -> dict:
        revision_note = state.review.required_revision if state.review else None
        history = agents.format_case_history(state.prior_decision, state.outcome_event)
        option_set = agents.propose_options(
            llm, state.request, state.evidence, state.integration,
            phase_for(state), revision_note, history,
        )
        revision_count = state.revision_count + (1 if state.review else 0)
        meta = meter(state)
        meta.current_node = "propose_options"
        return emit("propose_options", {
            "option_set": option_set, "revision_count": revision_count,
            "review": None, "run_metadata": meta,
        }, state)

    def simulate(state: DecisionState) -> dict:
        phase = phase_for(state)
        scenarios = decision_service.simulate(state.option_set.options, phase)
        experiment = next(
            (decision_service.design_experiment(option)
             for option in state.option_set.options if option.is_experiment),
            None,
        )
        meta = state.run_metadata.model_copy(deep=True)
        meta.current_node = "simulate_options"
        meta.updated_at = clock()
        return emit("simulate_options", {"scenarios": scenarios, "experiment": experiment,
                                         "run_metadata": meta}, state)

    def select(state: DecisionState) -> dict:
        history = agents.format_case_history(state.prior_decision, state.outcome_event)
        analyst = agents.select_recommendation(
            llm, state.evidence, state.option_set.options, state.scenarios,
            phase_for(state), history,
        )
        meta = meter(state)
        meta.current_node = "select_recommendation"
        return emit(
            "select_recommendation", {"analyst": analyst, "run_metadata": meta}, state
        )

    def validate(state: DecisionState) -> dict:
        result = validate_recommendation(
            state.option_set.options, state.scenarios, state.analyst, state.evidence
        )
        return emit("validate", {"validation": result}, state)

    def review_node(state: DecisionState) -> dict:
        review = agents.review(
            llm, state.evidence, state.analyst, state.scenarios, phase_for(state)
        )
        meta = meter(state)
        meta.current_node = "review"
        interrupts = list(state.interrupts)
        exhausted = check_budget(meta, state.runtime_limits)
        if exhausted:
            review = ReviewerOutput(
                findings=review.findings + [f"Runtime budget exhausted: {exhausted}."],
                unsupported_claims=review.unsupported_claims,
                causal_overstatement=review.causal_overstatement,
                disposition="escalate",
                required_revision=None,
            )
            interrupts.append(RunInterrupt(
                kind="budget_exhausted", reason=exhausted, node="review", at=clock(),
            ))
        elif any(
            record.status == "failed" and record.tool_name != "ad_hoc_sql"
            for record in state.tool_records
        ):
            failed_tools = sorted({
                record.tool_name for record in state.tool_records
                if record.status == "failed" and record.tool_name != "ad_hoc_sql"
            })
            review = ReviewerOutput(
                findings=review.findings + [
                    "Required evidence failed: " + ", ".join(failed_tools) + "."
                ],
                unsupported_claims=review.unsupported_claims,
                causal_overstatement=review.causal_overstatement,
                disposition="escalate",
                required_revision=None,
            )
            interrupts.append(RunInterrupt(
                kind="tool_failure",
                reason=", ".join(failed_tools),
                node="review",
                at=clock(),
            ))
        elif state.validation.status == "fail":
            review = ReviewerOutput(
                findings=review.findings + state.validation.issues,
                unsupported_claims=review.unsupported_claims,
                causal_overstatement=review.causal_overstatement,
                disposition="revise_options",
                required_revision="Resolve deterministic validation failures.",
            )
        return emit(
            "review",
            {"review": review, "run_metadata": meta, "interrupts": interrupts},
            state,
        )

    def human_approval(state: DecisionState) -> dict:
        interrupts = list(state.interrupts)
        if state.review.disposition != "pass":
            status = "escalated"
        elif state.human and state.human.decision == "approve":
            status = "approved"
        elif state.human and state.human.decision == "edit":
            status = "edited"
        elif state.human and state.human.decision == "reject":
            status = "rejected"
        elif state.human and state.human.decision == "request_more":
            status = "request_more"
        else:
            status = "pending_review"
        meta = state.run_metadata.model_copy(deep=True)
        meta.status = status
        meta.pause_reason = None if state.human else "awaiting human approval"
        if not state.human:
            interrupts = interrupts + [RunInterrupt(
                kind="human_approval", reason="final recommendation requires approval",
                node="human_approval", at=clock(),
            )]
        return emit("human_approval", {"status": status, "run_metadata": meta,
                                       "interrupts": interrupts}, state)

    def deliver(state: DecisionState) -> dict:
        meta = state.run_metadata.model_copy(deep=True)
        meta.current_node = "deliver"
        meta.completed_at = clock()
        meta.status = state.status
        return emit("deliver", {
            "insight": build_grounded_insight(state),
            "role_views": build_role_views(state),
            "learning": build_learning_summary(state),
            "run_metadata": meta,
        }, state)

    def route_after_review(state: DecisionState) -> str:
        disposition = state.review.disposition
        if disposition == "revise_options":
            return "revise_options" if state.revision_count < limits.max_revisions else "escalate"
        if disposition == "revise_investigation":
            return "revise_investigation" if state.revision_count < limits.max_revisions \
                else "escalate"
        if disposition == "escalate":
            return "escalate"
        return "approve"

    graph = StateGraph(DecisionState)
    graph.add_node("frame", frame)
    graph.add_node("gather", gather)
    graph.add_node("integrate", integrate)
    graph.add_node("propose_options", propose)
    graph.add_node("simulate_options", simulate)
    graph.add_node("select_recommendation", select)
    graph.add_node("validate", validate)
    graph.add_node("review", review_node)
    graph.add_node("human_approval", human_approval)
    graph.add_node("deliver", deliver)

    graph.add_edge(START, "frame")
    graph.add_edge("frame", "gather")
    graph.add_edge("gather", "integrate")
    graph.add_edge("integrate", "propose_options")
    graph.add_edge("propose_options", "simulate_options")
    graph.add_edge("simulate_options", "select_recommendation")
    graph.add_edge("select_recommendation", "validate")
    graph.add_edge("validate", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "revise_options": "propose_options",
            "revise_investigation": "frame",
            "approve": "human_approval",
            "escalate": "human_approval",
        },
    )
    graph.add_edge("human_approval", "deliver")
    graph.add_edge("deliver", END)

    if checkpointer is None:
        serializer = JsonPlusSerializer(allowed_msgpack_modules=()).with_msgpack_allowlist(
            _STATE_MODELS
        )
        checkpointer = MemorySaver(serde=serializer)
    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])


def make_sqlite_checkpointer(path):
    """A durable SQLite checkpointer sharing the state-model allowlist used in memory mode."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    serializer = JsonPlusSerializer(allowed_msgpack_modules=()).with_msgpack_allowlist(
        _STATE_MODELS
    )
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=serializer)
