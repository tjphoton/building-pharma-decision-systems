"""Verify printed Chapter 7 outputs and required assets against current results."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ch07_competitive.scripts.run_analysis import run_analysis  # noqa: E402


def main() -> None:
    results = run_analysis(ROOT)
    chapter = (
        ROOT / "ch07_competitive" / "ch07_competitive_intelligence_market_access.md"
    ).read_text()
    headline = results["headline"].iloc[0]
    event = results["formulary_event_effect"].iloc[0]
    required_text = [
        f"New-to-therapy patients: {int(headline.new_to_therapy_patients):,}",
        f"Roventra new starts: {int(headline.roventra_new_starts):,}",
        "Materially restricted lives: 6,740,000 of 10,926,000 (61.7%)",
        "New to therapy      3415              2798",
        "Continuing after washout check       513",
        "PAY005     South         Non-covered",
        "Dual workstream",
        f"Immediate level effect: {event.immediate_effect:+.1%}",
        (
            f"Week {int(event.effect_week)} effect: {event.effect_at_week:+.1%} "
            f"(95% CI {event.effect_at_week_lower_95:+.1%} "
            f"to {event.effect_at_week_upper_95:+.1%})"
        ),
        "Roventra      2798              0                0           Not reached",
    ]
    missing = [text for text in required_text if text not in chapter]
    if missing:
        raise AssertionError("Chapter output drift:\n" + "\n".join(missing))

    figure_dir = ROOT / "ch07_competitive" / "assets" / "figures"
    required_figures = [
        "figure_7_1_evidence_chain",
        "figure_7_2_access_lives",
        "figure_7_3_payer_region_matrix",
        "figure_7_4_attempt_trace",
        "figure_7_5_decision_map",
        "figure_7_6_formulary_event",
        "figure_7_7_account_actions",
        "figure_7_8_switch_support",
    ]
    absent = [
        f"{name}.{suffix}"
        for name in required_figures
        for suffix in ("svg", "png")
        if not (figure_dir / f"{name}.{suffix}").exists()
    ]
    if absent:
        raise AssertionError("Missing figure assets:\n" + "\n".join(absent))

    notebooks = sorted(
        path.name for path in (ROOT / "ch07_competitive").glob("*.ipynb")
    )
    expected = ["chapter7_walkthrough.ipynb", "exercise_solutions.ipynb"]
    if notebooks != expected:
        raise AssertionError(f"Expected exactly {expected}; found {notebooks}")
    if "—" in chapter:
        raise AssertionError("Chapter manuscript contains an em dash")
    print("Chapter 7 printed outputs, figures, and notebook inventory verified.")


if __name__ == "__main__":
    main()
