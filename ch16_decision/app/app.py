"""The Chapter 16 decision workbench as a local FastAPI service.

The service boundary is FastAPI; the existing HTML, CSS, and JavaScript are retained. Every
run flows through the shared :class:`AgentRuntime`, so the browser and the command-line runner
share one runtime, one durable checkpoint, and one approval path. The saved no-key mode renders
committed live traces so a reader can open the workbench without an API key. Live and mock runs
execute on a bounded background worker; the browser polls run status.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError

CHAPTER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CHAPTER_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import FIRST_DECISION_DATE, LATER_DECISION_DATE  # noqa: E402
from decision_services import approved_action_components, simulate_budget_scenario  # noqa: E402
from memory import CaseStore  # noqa: E402
from models import (  # noqa: E402
    CaseRecord,
    DecisionOption,
    HumanDisposition,
    OutcomeEvent,
    RunStatus,
    RunSummary,
    ScenarioResult,
    SignalEvent,
)
from runtime import (  # noqa: E402
    AgentRuntime,
    InvalidTransition,
    UnknownCase,
    UnknownRun,
    build_decision_request,
)
from signal_monitor import (  # noqa: E402
    confirm_signal,
    default_case_id,
    evaluate_hcp_digital_signal,
    read_trigger,
)

TEMPLATES = Jinja2Templates(directory=str(CHAPTER_DIR / "app" / "templates"))
STATIC_DIR = CHAPTER_DIR / "app" / "static"
TRACE_DIR = CHAPTER_DIR / "assets" / "traces"
TRACE_FILES = {
    "first": TRACE_DIR / "roventra_first_live_claude_haiku_4_5.json",
    "later": TRACE_DIR / "roventra_later_live_claude_haiku_4_5.json",
}
RUNTIME_DIR = CHAPTER_DIR / "assets" / "runtime"
EVAL_SUMMARY = CHAPTER_DIR / "assets" / "generated_outputs" / "ch16_eval_release_summary.csv"


# --- saved-trace helpers (no-key mode) ----------------------------------------------------

def _load_trace(phase: str) -> dict:
    if phase not in TRACE_FILES:
        raise KeyError(phase)
    state = json.loads(TRACE_FILES[phase].read_text())
    options = [DecisionOption.model_validate(item) for item in state["option_set"]["options"]]
    state["scenarios"] = [
        simulate_budget_scenario(option, phase).model_dump(mode="json") for option in options
    ]
    return state


def _display_text(value):
    if isinstance(value, str):
        return value.replace("—", ": ").replace("--", ": ")
    if isinstance(value, list):
        return [_display_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _display_text(item) for key, item in value.items()}
    return value


def _selected_scenario(state: dict) -> dict:
    selected = state["analyst"]["selected_option_name"]
    return next(item for item in state["scenarios"] if item["option_name"] == selected)


def _load_release_summary() -> dict:
    """Return the latest persisted evaluation summary for the workbench header."""
    if not EVAL_SUMMARY.exists():
        return {"status": "not run", "task_completion": "", "critical_pass": ""}
    with EVAL_SUMMARY.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {"status": "not run", "task_completion": "", "critical_pass": ""}
    if set(rows[0]) == {"metric", "value"}:
        row = {item["metric"]: item["value"] for item in rows}
    else:
        row = rows[-1]
    passed = str(row.get("critical_pass", "")).lower() == "true"
    return {
        "status": "pass" if passed else "blocked",
        "task_completion": row.get("task_completion", ""),
        "critical_pass": row.get("critical_pass", ""),
        "mode": row.get("mode", ""),
        "suite": row.get("suite", ""),
    }


# --- request/response contracts -----------------------------------------------------------

class ScenarioRequest(DecisionOption):
    date_phase: Literal["first", "later"] = "first"


class DispositionRequest(BaseModel):
    decision: Literal["approve", "edit", "reject", "request_more"]
    reviewer: str
    reason: str
    edited_option: Optional[DecisionOption] = None


class RunRequest(BaseModel):
    mode: Literal["mock", "live"] = "mock"


class CaseRequest(BaseModel):
    evidence_date: date = FIRST_DECISION_DATE


class OutcomeRequest(BaseModel):
    observed_incremental_nrx: int
    confidence_low: int
    confidence_high: int
    decision_id: str = "DEC-2026-0714-A1"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: bool
    checkpoint: bool
    model_configured: bool
    live_enabled: bool
    traces: list[str]


class SignalListResponse(BaseModel):
    signals: list[SignalEvent]


class SignalEvaluationResponse(BaseModel):
    trigger: list[str]
    signal: Optional[SignalEvent]


class CaseDetailResponse(BaseModel):
    case: CaseRecord
    runs: list[RunSummary]


class RunStartResponse(BaseModel):
    run_id: str
    status: Literal["running"]
    mode: Literal["mock", "live"]


def _api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _live_enabled() -> bool:
    return os.environ.get("CH16_ALLOW_LIVE", "true").lower() in {"1", "true", "yes"}


def _live_model_available() -> bool:
    return _live_enabled() and bool(os.environ.get("ANTHROPIC_API_KEY"))


def create_app(store_path: Optional[Path] = None,
               checkpoint_path: Optional[Path] = None) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=2)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        app.state.store = CaseStore(store_path or RUNTIME_DIR / "web_case_store.sqlite")
        app.state.checkpoint_path = checkpoint_path or RUNTIME_DIR / "web_checkpoints.sqlite"
        app.state.executor = executor
        yield
        executor.shutdown(wait=False)
        app.state.store.close()

    app = FastAPI(title="Roventra Decision Workbench", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def runtime(mode: str = "mock") -> AgentRuntime:
        return AgentRuntime(
            mock=(mode != "live"), store=app.state.store,
            checkpoint_path=app.state.checkpoint_path)

    # --- workbench page (saved no-key mode) -----------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, phase: str = "first"):
        if phase not in TRACE_FILES:
            phase = "first"
        state = _display_text(_load_trace(phase))
        signal = evaluate_hcp_digital_signal(FIRST_DECISION_DATE)
        return TEMPLATES.TemplateResponse(request, "index.html", {
            "state": state,
            "selected": _selected_scenario(state),
            "phase": phase,
            "controls": approved_action_components(),
            "candidate_signal": signal.model_dump(mode="json") if signal else None,
            "model_configured": _live_model_available(),
            "release": _load_release_summary(),
        })

    @app.get("/health", response_model=HealthResponse)
    def health():
        return {
            "status": "ok",
            "database": (CHAPTER_DIR / "assets" / "data" / "analytics.duckdb").exists(),
            "checkpoint": True,
            "model_configured": _live_model_available(),
            "live_enabled": _live_enabled(),
            "traces": list(TRACE_FILES),
        }

    # --- signals and cases ----------------------------------------------------
    @app.get("/api/signals", response_model=SignalListResponse)
    def list_signals():
        cases = app.state.store.list_cases()
        return {"signals": [c.signal.model_dump(mode="json") for c in cases]}

    @app.post("/api/signals/evaluate", response_model=SignalEvaluationResponse)
    def evaluate_signal(body: CaseRequest):
        readout = read_trigger(body.evidence_date)
        signal = evaluate_hcp_digital_signal(body.evidence_date)
        return {
            "trigger": readout.as_lines(),
            "signal": signal.model_dump(mode="json") if signal else None,
        }

    @app.post("/api/cases", response_model=CaseRecord)
    def create_case(body: CaseRequest):
        signal = evaluate_hcp_digital_signal(body.evidence_date)
        if signal is None:
            raise _api_error(400, "no_candidate_signal", "No candidate signal for that date.")
        case_id = default_case_id()
        request = confirm_signal(signal, build_decision_request(
            "first", case_id=case_id, signal_id=signal.signal_id,
            evidence_date=body.evidence_date))
        case = runtime().create_case(signal, request)
        return case.model_dump(mode="json")

    @app.get("/api/cases/{case_id}", response_model=CaseDetailResponse)
    def get_case(case_id: str):
        case = app.state.store.get_case(case_id)
        if case is None:
            raise _api_error(404, "unknown_case", "Unknown case.")
        runs = [r.model_dump(mode="json") for r in app.state.store.list_runs(case_id)]
        return {"case": case.model_dump(mode="json"), "runs": runs}

    # --- runs -----------------------------------------------------------------
    @app.post("/api/cases/{case_id}/runs", response_model=RunStartResponse)
    def start_case_run(case_id: str, body: RunRequest):
        if body.mode == "live" and not _live_enabled():
            raise _api_error(
                403,
                "live_mode_disabled",
                "Live mode is disabled for this deployment.",
            )
        if body.mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise _api_error(
                400,
                "live_model_not_configured",
                "Live mode requires a configured ANTHROPIC_API_KEY.",
            )
        rt = runtime(body.mode)
        try:
            run_id = rt.register_run(case_id, body.mode)
        except UnknownCase:
            raise _api_error(404, "unknown_case", "Unknown case.")
        # Execute on the bounded background worker; the browser polls /api/runs/{run_id}.
        app.state.executor.submit(runtime(body.mode).execute_run, run_id)
        return {"run_id": run_id, "status": "running", "mode": body.mode}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            state = runtime().get_run(run_id)
        except UnknownRun:
            raise _api_error(404, "unknown_run", "Unknown run.")
        return _display_text(state.model_dump(mode="json"))

    @app.post("/api/runs/{run_id}/disposition", response_model=RunStatus)
    def submit_disposition(run_id: str, body: DispositionRequest):
        if body.decision in {"edit", "reject", "request_more"} and not body.reason.strip():
            raise _api_error(
                400, "disposition_reason_required", "A reason is required for this decision."
            )
        rt = runtime(app.state.store.get_run_mode(run_id) or "mock")
        disposition = HumanDisposition(
            decision=body.decision, reviewer=body.reviewer, reason=body.reason,
            edited_option=body.edited_option)
        try:
            status = rt.submit_disposition(run_id, disposition)
        except UnknownRun:
            raise _api_error(404, "unknown_run", "Unknown run.")
        except InvalidTransition as error:
            raise _api_error(409, "invalid_transition", str(error))
        return status.model_dump(mode="json")

    @app.post("/api/runs/{run_id}/resume", response_model=RunStatus)
    def resume_run(run_id: str):
        try:
            status = runtime(app.state.store.get_run_mode(run_id) or "mock").resume_run(run_id)
        except UnknownRun:
            raise _api_error(404, "unknown_run", "Unknown run.")
        return status.model_dump(mode="json")

    # --- outcomes and reopening ----------------------------------------------
    @app.post("/api/cases/{case_id}/outcomes", response_model=RunStatus)
    def ingest_outcome(case_id: str, body: OutcomeRequest):
        outcome = OutcomeEvent(
            outcome_id="OUT-2026-1006-A1", case_id=case_id, decision_id=body.decision_id,
            available_date=LATER_DECISION_DATE, measurement_window="2026-W35..W40",
            observed_incremental_nrx=body.observed_incremental_nrx,
            confidence_low=body.confidence_low, confidence_high=body.confidence_high,
            population="community endocrinologists in stable-access DMAs", geography="US DMAs",
            source="prior_decisions + experiment_results", source_version="outcome-monitor v1",
            maturity_status="mature")
        try:
            status = runtime().ingest_outcome(case_id, outcome)
        except UnknownCase:
            raise _api_error(404, "unknown_case", "Unknown case.")
        return status.model_dump(mode="json")

    @app.post("/api/cases/{case_id}/reopen", response_model=RunStartResponse)
    def reopen_case(case_id: str, body: RunRequest):
        rt = runtime(body.mode)
        try:
            run_id = rt.register_run(
                case_id, body.mode,
                prior=app.state.store.get_prior_decision(case_id),
                outcome=app.state.store.get_outcome(case_id),
                request=build_decision_request(
                    "later", case_id=case_id, evidence_date=LATER_DECISION_DATE))
        except UnknownCase:
            raise _api_error(404, "unknown_case", "Unknown case.")
        app.state.executor.submit(runtime(body.mode).execute_run, run_id)
        return {"run_id": run_id, "status": "running", "mode": body.mode}

    # --- shared what-if service (kept separate from the released recommendation) --
    @app.post("/api/scenarios", response_model=ScenarioResult)
    def scenario(body: ScenarioRequest):
        phase = body.date_phase
        option = DecisionOption(**body.model_dump(exclude={"date_phase"}))
        return simulate_budget_scenario(option, phase).model_dump(mode="json")

    @app.get("/api/saved/{phase}")
    def saved_trace(phase: str):
        try:
            return _display_text(_load_trace(phase))
        except KeyError:
            raise _api_error(404, "unknown_decision_date", "Unknown decision date.")

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "detail": {
                "code": "validation_error",
                "message": "The request does not match the endpoint contract.",
                "issues": exc.errors(),
            }
        })

    @app.exception_handler(ValidationError)
    async def _validation_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={
            "detail": {
                "code": "validation_error",
                "message": "The request does not match the endpoint contract.",
                "issues": exc.errors(),
            }
        })

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5016)
