"""Run the Chapter 16 decision system end to end through the shared runtime service.

Usage (from the repository root):

    uv run python ch16_decision/scripts/run_skeleton.py            # live, first date
    uv run python ch16_decision/scripts/run_skeleton.py --date later
    uv run python ch16_decision/scripts/run_skeleton.py --mock     # no API key needed
    uv run python ch16_decision/scripts/run_skeleton.py --reject   # human rejects

The monitor opens a candidate case, a marketer confirms the request, the runtime starts a
durable run that pauses at the human-approval interrupt, and a recorded disposition resumes it.
``--date later`` runs the first decision, ingests the observed outcome, and reopens the same
case so the later decision loads the earlier one by ID. Mock mode is a plumbing check only.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_env() -> None:
    """Load KEY=VALUE lines from ch16_decision/.env and the repo root .env, without
    overwriting anything already set in the environment. We never write these files."""
    candidates = [SCRIPT_DIR.parent / ".env", SCRIPT_DIR.parents[1] / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env()

from data_access import ANALYTICS_DB, query_approved_data  # noqa: E402
from decision_graph import DecisionState  # noqa: E402
from delivery import persist_state  # noqa: E402
from memory import CaseStore  # noqa: E402
from models import HumanDisposition, OutcomeEvent  # noqa: E402
from runtime import AgentRuntime, build_decision_request  # noqa: E402
from signal_monitor import (  # noqa: E402
    confirm_signal,
    default_case_id,
    evaluate_hcp_digital_signal,
    read_trigger,
)
from config import FIRST_DECISION_DATE, LATER_DECISION_DATE  # noqa: E402

RULE = "=" * 78


def make_request(date_phase: str):
    """Compatibility shim for tests and notebooks: build a request by two-date label."""
    case_id = default_case_id()
    evidence = LATER_DECISION_DATE if date_phase == "later" else FIRST_DECISION_DATE
    return build_decision_request(
        date_phase, case_id=case_id, signal_id=None, evidence_date=evidence,
    )


def ensure_database() -> None:
    if not ANALYTICS_DB.exists():
        from build_database import build

        build()


def later_outcome(case_id: str, decision_id: str) -> OutcomeEvent:
    """Build the observed later outcome from the approved prior-decisions table."""
    result = query_approved_data(
        "SELECT decision_id, observed FROM prior_decisions LIMIT 1"
    )
    observed = int(result.rows[0][1]) if result.rows else 248
    return OutcomeEvent(
        outcome_id="OUT-2026-1006-A1",
        case_id=case_id,
        decision_id=decision_id,
        available_date=LATER_DECISION_DATE,
        measurement_window="2026-W35..W40",
        observed_incremental_nrx=observed,
        confidence_low=max(0, observed - 68),
        confidence_high=observed + 52,
        population="community endocrinologists in stable-access DMAs",
        geography="US DMAs",
        source="prior_decisions + experiment_results",
        source_version="outcome-monitor v1",
        maturity_status="mature",
        quality_notes="Reconciled claims and a completed matched-market read.",
    )


def print_brief(state: DecisionState) -> None:
    a = state.analyst
    selected = next((o for o in state.option_set.options if o.name == a.selected_option_name), None)
    scenario = next((s for s in state.scenarios if s.option_name == a.selected_option_name), None)
    print(f"\n{RULE}\nDRAFT DECISION BRIEF (paused for human approval)\n{RULE}")
    print(f"Decision: {state.request.question}")
    print(f"Recommendation: {a.selected_option_name}")
    if selected and scenario:
        print(f"  {selected.description}")
        print(
            f"  budget moved: ${selected.budget_moved_usd:,} | expected "
            f"+{scenario.expected_incr_nrx_low}..{scenario.expected_incr_nrx_high} "
            f"incremental NRx | reversibility: {selected.reversibility}"
        )
    print(f"Rationale: {a.rationale}")
    print(f"Measurement plan: {a.measurement_plan}")
    print(f"Would change the call: {a.evidence_that_would_change_selection}")
    print(f"Reviewer: {state.review.disposition}")
    m = state.run_metadata
    print(
        f"Metering: {m.llm_steps} LLM steps | {m.tool_calls} tool calls "
        f"({m.ad_hoc_queries} ad hoc) | est ${m.estimated_cost_usd:.4f}"
    )


def _approve_first(runtime: AgentRuntime, case_id: str, reject: bool):
    status = runtime.start_run(case_id, mode="mock" if runtime.mock else "live")
    state = runtime.get_run(status.run_id)
    print_brief(state)
    print(f"\n[paused] awaiting human; next node: {status.next_node}  (no budget released)")
    passed = state.review.disposition == "pass"
    disposition = HumanDisposition(
        decision="reject" if reject else "approve" if passed else "request_more",
        reviewer="Brand lead",
        reason=(
            "Rejected in skeleton run." if reject
            else "Bounded, reversible, and cites the evidence." if passed
            else "Independent review remains open after the bounded revision cycle."
        ),
    )
    final_status = runtime.submit_disposition(status.run_id, disposition)
    return status.run_id, final_status, runtime.get_run(status.run_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["first", "later"], default="first")
    parser.add_argument("--mock", action="store_true", help="run without an API key")
    parser.add_argument("--reject", action="store_true", help="human rejects the recommendation")
    args = parser.parse_args()

    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found. Add it to ch16_decision/.env, or run with --mock.")
        return 1

    ensure_database()

    # A throwaway runtime store and checkpoint file so the skeleton run is self-contained.
    tmp = Path(tempfile.mkdtemp(prefix="ch16_skeleton_"))
    store = CaseStore(tmp / "case_store.sqlite")
    runtime = AgentRuntime(mock=args.mock, store=store, checkpoint_path=tmp / "checkpoints.sqlite")

    mode = "MOCK (plumbing check, not intelligence)" if args.mock else f"LIVE ({runtime._new_llm().model})"
    print(f"{RULE}\nChapter 16 decision system  |  {mode}  |  date: {args.date}\n{RULE}")

    readout = read_trigger(FIRST_DECISION_DATE)
    print("[monitor] deterministic trigger:")
    for line in readout.as_lines():
        print(f"  {line}")
    signal = evaluate_hcp_digital_signal(FIRST_DECISION_DATE)
    if signal is None:
        print("[monitor] no candidate signal; nothing to decide.")
        return 0
    print(f"[monitor] candidate signal {signal.signal_id} opened.")

    case_id = default_case_id()
    request = confirm_signal(signal, build_decision_request(
        "first", case_id=case_id, signal_id=signal.signal_id, evidence_date=FIRST_DECISION_DATE,
    ))
    persist_state(signal, "signal_event.json")
    persist_state(request, "confirmed_decision_request.json")
    runtime.create_case(signal, request)
    print(f"[marketer] confirmed decision request for case {case_id}.")

    run_id, final_status, final = _approve_first(runtime, case_id, args.reject)
    print(f"\n{RULE}\nHUMAN DISPOSITION: {final.human.decision.upper()}  |  {final.human.reason}")
    print(f"First run {run_id} complete. Status: {final_status.status}. "
          f"Revision cycles used: {final.revision_count}.")

    if args.date == "first":
        output_path = persist_state(final, "first_decision_state.json")
        print(f"Persisted: {output_path.relative_to(SCRIPT_DIR.parents[1])}\n{RULE}")
        store.close()
        return 0

    # Later decision: ingest the observed outcome and reopen the same case ID.
    print(f"\n{RULE}\n[outcome] ingesting the observed result and reopening the case\n{RULE}")
    outcome = later_outcome(case_id, "DEC-2026-0714-A1")
    persist_state(outcome, "later_outcome_event.json")
    runtime.ingest_outcome(case_id, outcome)
    later_status = runtime.reopen_case(case_id, mode="mock" if args.mock else "live")
    later_state = runtime.get_run(later_status.run_id)
    print_brief(later_state)
    disposition = HumanDisposition(
        decision="approve", reviewer="Brand lead",
        reason="The completed test supports scale in the proven segment.",
    )
    runtime.submit_disposition(later_status.run_id, disposition)
    later_final = runtime.get_run(later_status.run_id)
    if later_final.learning:
        print(f"\n[learning] expected {later_final.learning.expected_range}; "
              f"observed {later_final.learning.observed_result}.")
    output_path = persist_state(later_final, "later_decision_state.json")
    print(f"Later run {later_status.run_id} loaded prior decision "
          f"{later_state.prior_decision.decision_id if later_state.prior_decision else 'n/a'}.")
    print(f"Persisted: {output_path.relative_to(SCRIPT_DIR.parents[1])}\n{RULE}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
