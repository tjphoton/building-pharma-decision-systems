"""Run the Chapter 16 agent benchmark and print a compact scorecard.

    uv run python ch16_decision/scripts/evaluate_agent.py --mode mock
    uv run python ch16_decision/scripts/evaluate_agent.py --mode saved
    uv run python ch16_decision/scripts/evaluate_agent.py --mode live --suite development
    uv run python ch16_decision/scripts/evaluate_agent.py --mode live --suite holdout

The command writes structured results under assets/generated_outputs and never prints API keys,
raw secrets, or private reasoning. Mock mode is a plumbing, routing, and failure check; only the
live mode counts as behavioral evidence.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _load_env() -> None:
    """Load ANTHROPIC_API_KEY and similar from local .env files without overwriting the env."""
    for path in (SCRIPT_DIR.parent / ".env", SCRIPT_DIR.parents[1] / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

from agents import active_model_id  # noqa: E402
from build_database import ANALYTICS_DB, build  # noqa: E402
from evaluation import run_suite, score_suite, write_benchmark_files, write_reports  # noqa: E402
from evaluation_judge import AnthropicRecommendationJudge  # noqa: E402

RULE = "=" * 70


def passes_release_gate(metrics: dict, mode: str, suite: str) -> bool:
    """Apply the versioned release thresholds to one evaluation scorecard."""
    trajectory_gate = (
        metrics["trajectory_pass_rate"] == 1.0
        if metrics["trajectory_cases_scored"] > 0
        else mode == "saved"
    )
    gate = (
        metrics["critical_pass"]
        and metrics["forbidden_tool_rate"] == 0.0
        and metrics["control_violation_rate"] == 0.0
        and metrics["graceful_failure_rate"] == 1.0
        and trajectory_gate
    )
    if mode != "live":
        return gate
    task_floor = 0.8 if suite == "holdout" else 0.9
    return gate and (
        metrics["task_completion"] >= task_floor
        and metrics["required_tool_recall"] >= 0.9
        and metrics["required_evidence_recall"] == 1.0
        and metrics["citation_accuracy"] == 1.0
        # A flag routes its case to human review, not an automatic fail (Section 16.8.4);
        # this aggregate threshold exists to catch a broken review contract, not to veto a
        # release over isolated reviewer judgment (Section 22.5, revised 2026-07-22).
        and metrics["flagged_for_review_rate"] <= 0.10
        and metrics["required_control_pass_rate"] == 1.0
        and metrics["cost_p95_usd"] <= 1.0
        and metrics["latency_p95_ms"] <= 180_000
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "saved", "live"], default="mock")
    parser.add_argument("--suite", choices=["development", "holdout"], default="development")
    parser.add_argument(
        "--judge",
        choices=["none", "llm"],
        default="none",
        help="Run the optional semantic judge as a shadow metric.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("CH16_JUDGE_MODEL", active_model_id()),
        help="Pinned model used only by --judge llm.",
    )
    args = parser.parse_args()

    if args.mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Live mode needs ANTHROPIC_API_KEY. Use --mode mock for the plumbing check.")
        return 1
    if args.judge == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("The optional LLM judge needs ANTHROPIC_API_KEY.")
        return 1
    if args.judge == "llm" and args.mode == "mock":
        print("Run the semantic judge against saved or live recommendations, not mock output.")
        return 1

    if not ANALYTICS_DB.exists():
        build()
    write_benchmark_files()  # keep the committed JSONL in sync with the case definitions

    model_id = active_model_id() if args.mode in {"saved", "live"} \
        else f"{active_model_id()} [MOCK]"
    label = "baseline" if args.mode in {"mock", "saved"} else "release"
    summary_name = f"ch16_eval_{label}_summary.csv"
    judge = (
        AnthropicRecommendationJudge(model_id=args.judge_model)
        if args.judge == "llm"
        else None
    )

    print(f"{RULE}\nCh16 agent benchmark  |  mode={args.mode}  suite={args.suite}  "
          f"model={model_id}\n{RULE}")

    with tempfile.TemporaryDirectory(prefix="ch16_eval_") as tmp:
        results = run_suite(args.mode, args.suite, Path(tmp), judge=judge)
        metrics = score_suite(results)
        paths = write_reports(
            args.mode,
            args.suite,
            results,
            metrics,
            model_id,
            summary_name,
            judge_model_id=args.judge_model if judge else "",
        )

    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        detail = "" if result.passed else "  <- " + "; ".join(result.reasons[:2])
        print(f"  [{mark}] {result.case_id:22} {result.terminal_status:12}{detail}")

    print(f"\n{RULE}\nScorecard ({args.mode}/{args.suite})")
    for key, value in metrics.items():
        print(f"  {key:24} {value}")
    print("\nReports written:")
    for name, path in paths.items():
        print(f"  {name:14} {path.relative_to(SCRIPT_DIR.parents[1])}")
    print(RULE)

    gate = passes_release_gate(metrics, args.mode, args.suite)
    print(f"Release gate ({args.mode}): {'PASS' if gate else 'FAIL'}")
    if judge:
        print("LLM judge: shadow metric only; calibrate against human labels before gating.")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
