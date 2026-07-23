"""The shared runtime service for the Chapter 16 decision system.

Both the command-line runner and the FastAPI workbench call :class:`AgentRuntime`. They never
build their own graph, manufacture state, or write their own approval logic. The runtime owns
case creation, run start, durable checkpointing, human dispositions, resume after restart,
outcome ingestion, and case reopening under one stable case ID.
"""

from __future__ import annotations

import uuid
import time
from datetime import date, datetime, timezone
from typing import Callable, Optional

from config import (
    BENCHMARK_VERSION,
    DATA_VERSION,
    DEFAULT_LIMITS,
    FIRST_DECISION_DATE,
    GRAPH_VERSION,
    LATER_DECISION_DATE,
    PROMPT_VERSION,
    TOOL_VERSION,
    pricing_for,
)
from decision_graph import (
    DecisionState,
    DecisionService,
    DEFAULT_SERVICE,
    build_graph,
    make_sqlite_checkpointer,
)
from tools import run_tool
from agents import LLM, active_model_id
from memory import CHECKPOINT_PATH, CaseStore
from models import (
    CaseRecord,
    DecisionRequest,
    HumanDisposition,
    OutcomeEvent,
    PriorDecisionRecord,
    RunInterrupt,
    RunMetadata,
    RunStatus,
    RuntimeLimits,
    RunSummary,
    SignalEvent,
    ToolCallRecord,
)
from validation import validate_recommendation


class RuntimeError_(RuntimeError):
    """Base runtime error."""


class UnknownRun(RuntimeError_):
    pass


class UnknownCase(RuntimeError_):
    pass


class InvalidTransition(RuntimeError_):
    """Raised when a disposition or resume is requested for a run not in the right state."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_decision_request(
    phase: str,
    *,
    case_id: str,
    signal_id: Optional[str] = None,
    evidence_date: Optional[date] = None,
) -> DecisionRequest:
    """The two Roventra requests, tagged with the case and evidence date."""
    if phase == "later":
        return DecisionRequest(
            question="How far should we scale HCP digital now that the test has read out?",
            requesting_role="Brand lead",
            audience="Community and academic endocrinologists",
            geography="US DMAs",
            budget_source="DTC paid media",
            budget_destination="HCP digital",
            proposed_move="Scale HCP digital where the matched-market test proved an effect",
            decision_date="2026-10-06 (later date, mature evidence)",
            reversibility_required=True,
            approvers=["Brand lead", "Commercial finance"],
            case_id=case_id,
            signal_id=signal_id,
            business_reason="A completed test can recalibrate the segment response.",
            evidence_date=evidence_date or LATER_DECISION_DATE,
            decision_on=LATER_DECISION_DATE,
            requested_by="Brand lead",
        )
    return DecisionRequest(
        question="Should we move a quarter of DTC paid media budget into HCP digital to grow "
        "incremental NRx?",
        requesting_role="Brand lead",
        audience="Community and academic endocrinologists",
        geography="US DMAs",
        budget_source="DTC paid media",
        budget_destination="HCP digital",
        proposed_move="Shift about a quarter of quarterly DTC budget into HCP digital",
        decision_date="2026-07-14 (first date, incomplete evidence)",
        reversibility_required=True,
        approvers=["Brand lead", "Commercial finance"],
        case_id=case_id,
        signal_id=signal_id,
        business_reason="DTC paid media is near saturation while HCP digital has headroom.",
        evidence_date=evidence_date or FIRST_DECISION_DATE,
        decision_on=FIRST_DECISION_DATE,
        requested_by="Brand lead",
    )


class AgentRuntime:
    """Start, checkpoint, resume, meter, interrupt, and persist decision runs."""

    def __init__(
        self,
        mock: bool = True,
        store: Optional[CaseStore] = None,
        checkpoint_path=CHECKPOINT_PATH,
        limits: Optional[RuntimeLimits] = None,
        decision_service: DecisionService = DEFAULT_SERVICE,
        tool_runner: Callable[[str, str], list] = run_tool,
        on_event: Optional[Callable[[str, dict], None]] = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ):
        self.mock = mock
        self.store = store or CaseStore()
        self.checkpoint_path = checkpoint_path
        self.limits = limits or DEFAULT_LIMITS
        self.decision_service = decision_service
        self.tool_runner = tool_runner
        self.on_event = on_event
        self.monotonic = monotonic

    # --- graph construction --------------------------------------------------
    def _new_llm(self) -> LLM:
        return LLM(
            mock=self.mock,
            max_output_tokens=self.limits.max_output_tokens_per_call,
            max_repairs=self.limits.max_structured_repairs_per_call,
        )

    def _graph(self, llm: LLM, on_event=None):
        checkpointer = make_sqlite_checkpointer(self.checkpoint_path)
        return build_graph(
            llm,
            checkpointer=checkpointer,
            limits=self.limits,
            tool_runner=self.tool_runner,
            decision_service=self.decision_service,
            on_event=on_event or self.on_event,
        )

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    # --- case lifecycle ------------------------------------------------------
    def create_case(self, signal: SignalEvent, request: DecisionRequest) -> CaseRecord:
        case_id = request.case_id or "CASE-ROVENTRA-HCP-2026"
        request = request.model_copy(update={"case_id": case_id, "signal_id": signal.signal_id})
        case = CaseRecord(
            case_id=case_id, signal=signal, request=request, created_at=_now_iso(),
        )
        self.store.save_case(case)
        return case

    def _metadata(self, run_id: str, case_id: str, thread_id: str, mode: str) -> RunMetadata:
        active_model = active_model_id()
        pricing = pricing_for(active_model)
        return RunMetadata(
            run_id=run_id, case_id=case_id, thread_id=thread_id,
            started_at=_now_iso(), updated_at=_now_iso(),
            status="running", model_id=active_model, prompt_version=PROMPT_VERSION,
            graph_version=GRAPH_VERSION, tool_version=TOOL_VERSION, data_version=DATA_VERSION,
            pricing_version=pricing.pricing_version,
            benchmark_version=BENCHMARK_VERSION if mode == "live" else None,
        )

    def _initial_state(
        self, case: CaseRecord, request: DecisionRequest, run_id: str, thread_id: str, mode: str,
        prior: Optional[PriorDecisionRecord], outcome: Optional[OutcomeEvent],
    ) -> DecisionState:
        phase = "later" if (request.evidence_date and request.evidence_date >= LATER_DECISION_DATE) \
            else "first"
        request_text = " ".join((
            request.question,
            request.proposed_move,
            request.business_reason or "",
        )).upper()
        prohibited_sql = next(
            (keyword for keyword in ("DELETE ", "UPDATE ", "INSERT ", "DROP ", "ALTER ",
                                     "TRUNCATE ", "CREATE ") if keyword in request_text),
            None,
        )
        input_control_records = []
        if prohibited_sql:
            now = _now_iso()
            input_control_records.append(ToolCallRecord(
                tool_name="ad_hoc_sql",
                purpose="Prohibited SQL supplied in the decision request",
                started_at=now,
                ended_at=now,
                status="blocked",
                error_class="InputSqlGuard",
                sql=f"{prohibited_sql.strip()} statement blocked at intake",
            ))
        return DecisionState(
            request=request,
            date_phase=phase,
            signal=case.signal,
            case_id=case.case_id,
            runtime_limits=self.limits,
            run_metadata=self._metadata(run_id, case.case_id, thread_id, mode),
            prior_decision=prior,
            outcome_event=outcome,
            tool_records=input_control_records,
            status="running",
        )

    def _persist(self, run_id: str, mode: str, state: DecisionState) -> None:
        self.store.save_run(state.run_metadata, mode, state.model_dump_json())

    def _status_from_state(self, state: DecisionState, next_node) -> RunStatus:
        awaiting = next_node == ("human_approval",)
        pause_reason = None
        if awaiting:
            pause_reason = "final recommendation requires human approval"
            if state.review and state.review.disposition != "pass":
                pause_reason = f"reviewer disposition {state.review.disposition}; escalated"
        return RunStatus(
            run_id=state.run_metadata.run_id,
            case_id=state.case_id or "",
            status=state.status,
            current_node=state.run_metadata.current_node,
            next_node=next_node[0] if next_node else None,
            pause_reason=pause_reason,
            awaiting_disposition=awaiting,
            message="",
        )

    # --- run start -----------------------------------------------------------
    def register_run(
        self, case_id: str, mode: str = "mock",
        *, prior: Optional[PriorDecisionRecord] = None,
        outcome: Optional[OutcomeEvent] = None,
        request: Optional[DecisionRequest] = None,
    ) -> str:
        """Create and persist a pending run without executing the graph.

        The web layer registers a run (fast), then executes it on a background worker so the
        browser can poll ``get_run`` for node progress.
        """
        case = self.store.get_case(case_id)
        if case is None:
            raise UnknownCase(case_id)
        run_id = f"RUN-{uuid.uuid4().hex[:10]}"
        thread_id = f"{case_id}:{run_id}"
        request = request or case.request
        initial = self._initial_state(case, request, run_id, thread_id, mode, prior, outcome)
        self._persist(run_id, mode, initial)
        return run_id

    def execute_run(self, run_id: str) -> RunStatus:
        """Run the graph for a registered run to its human-approval interrupt."""
        initial = self.get_run(run_id)
        mode = self.store.get_run_mode(run_id) or "mock"
        thread_id = initial.run_metadata.thread_id
        case_id = initial.case_id or ""
        started = self.monotonic()

        interrupt_kind = None
        interrupt_reason = None
        if initial.request.data_access_classification == "restricted":
            interrupt_kind = "restricted_data"
            interrupt_reason = "The request requires restricted data approval."
        elif initial.request.capability_scope == "outside" \
                or initial.request.boundary_exception_requested:
            interrupt_kind = "boundary_exception"
            interrupt_reason = "The request is outside the approved allocation capability."
        elif initial.request.generated_python_requested:
            interrupt_kind = "generated_python"
            interrupt_reason = "Generated Python requires a separate human-approved sandbox."
        if interrupt_kind:
            meta = initial.run_metadata.model_copy(update={
                "status": "interrupted",
                "current_node": "preflight",
                "updated_at": _now_iso(),
                "elapsed_ms": 0,
                "pause_reason": interrupt_reason,
            })
            interrupted = initial.model_copy(update={
                "status": "interrupted",
                "run_metadata": meta,
                "node_trace": initial.node_trace + ["preflight"],
                "interrupts": initial.interrupts + [RunInterrupt(
                    kind=interrupt_kind,
                    reason=interrupt_reason,
                    node="preflight",
                    at=_now_iso(),
                )],
            })
            self._persist(run_id, mode, interrupted)
            return RunStatus(
                run_id=run_id,
                case_id=case_id,
                status="interrupted",
                current_node="preflight",
                pause_reason=interrupt_reason,
                message="Run paused at the capability boundary.",
            )

        current = initial.model_dump()

        def persist_event(node: str, update: dict) -> None:
            current.update(update)
            meta = RunMetadata.model_validate(current["run_metadata"])
            meta.elapsed_ms = int((self.monotonic() - started) * 1000)
            meta.updated_at = _now_iso()
            update["run_metadata"] = meta
            current["run_metadata"] = meta
            self._persist(run_id, mode, DecisionState.model_validate(current))
            if self.on_event:
                self.on_event(node, update)

        llm = self._new_llm()
        graph = self._graph(llm, on_event=persist_event)
        config = self._config(thread_id)
        try:
            graph.invoke(initial, config)
        except Exception as error:
            latest = self.get_run(run_id)
            meta = latest.run_metadata.model_copy(update={
                "status": "interrupted", "updated_at": _now_iso(),
                "elapsed_ms": int((self.monotonic() - started) * 1000),
                "pause_reason": f"{type(error).__name__}: {error}",
            })
            failed = latest.model_copy(update={
                "status": "interrupted", "run_metadata": meta,
                "errors": latest.errors + [f"{type(error).__name__}: {error}"],
                "interrupts": latest.interrupts + [RunInterrupt(
                    kind="provider_failure",
                    reason=f"{type(error).__name__}: {error}",
                    node=meta.current_node or "runtime",
                    at=_now_iso(),
                )],
            })
            self._persist(run_id, mode, failed)
            return RunStatus(
                run_id=run_id, case_id=case_id, status="interrupted",
                pause_reason=meta.pause_reason, message="Run interrupted; state preserved.",
            )
        snap = graph.get_state(config)
        state = DecisionState.model_validate(snap.values)
        meta = state.run_metadata.model_copy(update={
            "elapsed_ms": int((self.monotonic() - started) * 1000),
            "updated_at": _now_iso(),
        })
        state = state.model_copy(update={"run_metadata": meta})
        self._persist(run_id, mode, state)
        return self._status_from_state(state, snap.next)

    def start_run(
        self, case_id: str, mode: str = "mock",
        *, prior: Optional[PriorDecisionRecord] = None,
        outcome: Optional[OutcomeEvent] = None,
        request: Optional[DecisionRequest] = None,
    ) -> RunStatus:
        """Register and execute a run synchronously (used by the CLI and evaluation)."""
        run_id = self.register_run(
            case_id, mode, prior=prior, outcome=outcome, request=request)
        return self.execute_run(run_id)

    # --- read ----------------------------------------------------------------
    def get_run(self, run_id: str) -> DecisionState:
        payload = self.store.get_run_state(run_id)
        if payload is None:
            raise UnknownRun(run_id)
        return DecisionState.model_validate_json(payload)

    def list_case_runs(self, case_id: str) -> list[RunSummary]:
        return self.store.list_runs(case_id)

    def _reopen_graph(self, run_id: str):
        meta = self.store.get_run_meta(run_id)
        if meta is None:
            raise UnknownRun(run_id)
        llm = self._new_llm()
        graph = self._graph(llm)
        config = self._config(meta.thread_id)
        return graph, config, meta

    # --- human dispositions --------------------------------------------------
    def submit_disposition(self, run_id: str, disposition: HumanDisposition) -> RunStatus:
        graph, config, meta = self._reopen_graph(run_id)
        mode = self.store.get_run_mode(run_id) or "mock"
        snap = graph.get_state(config)
        if snap.next != ("human_approval",):
            # Already resolved: idempotent, no duplicate delivery record.
            state = self.get_run(run_id)
            status = self._status_from_state(state, snap.next)
            status.message = "Run already resolved; disposition ignored (idempotent)."
            raise InvalidTransition(status.message)
        state = DecisionState.model_validate(snap.values)
        update: dict = {"human": disposition}
        # Edit path: re-simulate and revalidate the reviewer's replacement option.
        if disposition.decision == "edit" and disposition.edited_option is not None:
            option = disposition.edited_option
            scenario = self.decision_service.simulate([option], state.date_phase)[0]
            analyst = state.analyst.model_copy(update={"selected_option_name": option.name})
            revalidated = validate_recommendation(
                state.option_set.options + [option], state.scenarios + [scenario],
                analyst, state.evidence,
            )
            update.update({
                "option_set": state.option_set.model_copy(update={
                    "options": state.option_set.options + [option]}),
                "scenarios": state.scenarios + [scenario],
                "analyst": analyst,
                "validation": revalidated,
            })
            if revalidated.status == "fail":
                raise InvalidTransition(
                    "Edited option fails deterministic validation: " + "; ".join(revalidated.issues)
                )
        graph.update_state(config, update)
        graph.invoke(None, config)  # resume through the human-approval interrupt
        final = DecisionState.model_validate(graph.get_state(config).values)
        self._persist(run_id, mode, final)
        self._maybe_record_prior_decision(final)
        return self._status_from_state(final, graph.get_state(config).next)

    def resume_run(self, run_id: str) -> RunStatus:
        """Recover a run after a process restart. Finish delivery when a disposition exists."""
        graph, config, meta = self._reopen_graph(run_id)
        mode = self.store.get_run_mode(run_id) or "mock"
        snap = graph.get_state(config)
        state = DecisionState.model_validate(snap.values)
        if snap.next == ("human_approval",) and state.human is not None:
            graph.invoke(None, config)
            state = DecisionState.model_validate(graph.get_state(config).values)
            self._persist(run_id, mode, state)
            self._maybe_record_prior_decision(state)
        return self._status_from_state(state, graph.get_state(config).next)

    def _maybe_record_prior_decision(self, state: DecisionState) -> None:
        if state.status != "approved" or state.analyst is None:
            return
        if state.request.evidence_date and state.request.evidence_date >= LATER_DECISION_DATE:
            return  # only the first-date decision seeds the later run
        scenario = next(
            (s for s in state.scenarios if s.option_name == state.analyst.selected_option_name),
            None,
        )
        option = next(
            (o for o in state.option_set.options
             if o.name == state.analyst.selected_option_name),
            None,
        )
        if scenario is None or option is None:
            return
        record = PriorDecisionRecord(
            decision_id="DEC-2026-0714-A1",
            case_id=state.case_id or "",
            selected_option_name=state.analyst.selected_option_name,
            action_description=option.description,
            budget_moved_usd=option.budget_moved_usd,
            expected_incr_nrx_low=scenario.expected_incr_nrx_low,
            expected_incr_nrx_high=scenario.expected_incr_nrx_high,
            evidence_ids=state.analyst.evidence_ids,
            measurement_plan=state.analyst.measurement_plan,
            approval_reviewer=state.human.reviewer if state.human else "",
            approval_reason=state.human.reason if state.human else "",
            approved_at=_now_iso(),
            expected_outcome_window="2026-W35..W40",
            next_review_date="2026-10-06",
        )
        self.store.save_prior_decision(record)
        case = self.store.get_case(state.case_id or "")
        if case is not None:
            case.prior_decision = record
            case.status = "decided"
            self.store.save_case(case)

    # --- outcomes and reopening ---------------------------------------------
    def ingest_outcome(self, case_id: str, outcome: OutcomeEvent) -> RunStatus:
        case = self.store.get_case(case_id)
        if case is None:
            raise UnknownCase(case_id)
        self.store.save_outcome(outcome)
        case.outcome_event = outcome
        case.status = "reopened"
        self.store.save_case(case)
        return RunStatus(
            run_id="", case_id=case_id, status="reopened",
            message=f"Outcome {outcome.outcome_id} ingested; case reopened.",
        )

    def reopen_case(self, case_id: str, mode: str = "mock") -> RunStatus:
        case = self.store.get_case(case_id)
        if case is None:
            raise UnknownCase(case_id)
        prior = self.store.get_prior_decision(case_id)
        outcome = self.store.get_outcome(case_id)
        later_request = build_decision_request(
            "later", case_id=case_id, signal_id=case.signal.signal_id,
            evidence_date=LATER_DECISION_DATE,
        )
        return self.start_run(
            case_id, mode, prior=prior, outcome=outcome, request=later_request,
        )
