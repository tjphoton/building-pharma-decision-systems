"""Deterministic calculations and action controls used by the agent graph."""

from __future__ import annotations

import math

from models import DecisionOption, ExperimentDesign, ScenarioResult


MAX_MOVE_USD = 750_000
MIN_TEST_USD = 120_000
MAX_TEST_USD = 250_000


def approved_action_components() -> dict[str, list[str] | int]:
    return {
        "audiences": ["community_stable", "all_endocrinologists", "hold"],
        "geographies": ["matched_markets", "stable_access_dmas", "all_dmas", "none"],
        "measurement_designs": ["matched_market", "outcome_monitor", "none"],
        "maximum_move_usd": MAX_MOVE_USD,
    }


def _response_parameters(option: DecisionOption, date_phase: str) -> tuple[float, float, float]:
    if option.audience == "hold":
        return 0.0, 1.0, 0.0
    if date_phase == "later" and option.audience == "community_stable":
        return 1_700.0, 520_000.0, 0.18
    if date_phase == "later":
        return 800.0, 450_000.0, 0.34
    if option.audience == "community_stable":
        return 400.0, 210_000.0, 0.48
    return 260.0, 180_000.0, 0.68


def simulate_budget_scenario(option: DecisionOption, date_phase: str) -> ScenarioResult:
    """Calculate one option on a saturating response curve with explicit constraints."""
    notes: list[str] = []
    feasible = True
    if option.budget_moved_usd > MAX_MOVE_USD:
        feasible = False
        notes.append(f"Move exceeds the ${MAX_MOVE_USD:,} approved ceiling.")
    if option.is_experiment and not MIN_TEST_USD <= option.budget_moved_usd <= MAX_TEST_USD:
        feasible = False
        notes.append(f"Matched-market tests require ${MIN_TEST_USD:,} to ${MAX_TEST_USD:,}.")
    if option.is_experiment and option.measurement_design != "matched_market":
        feasible = False
        notes.append("An experimental option requires a matched-market design.")
    if option.audience == "hold" and option.budget_moved_usd != 0:
        feasible = False
        notes.append("A hold option cannot move budget.")

    if option.audience == "hold":
        hcp_count, low, mid, high = 0, 0, 0, 0
        calculation = "0 budget × 0 exposed HCPs = 0 incremental NRx"
        durability = "No effect assumed because no action is released."
    elif date_phase == "later" and option.audience == "community_stable":
        hcp_count = 8_000
        exposure = min(1.0, option.budget_moved_usd / 375_000)
        low = round(0.9 * hcp_count / 100 * exposure)
        mid = round(2.1 * hcp_count / 100 * exposure)
        high = round(3.3 * hcp_count / 100 * exposure)
        calculation = (
            f"8,000 eligible HCPs / 100 × {exposure:.2f} exposure × "
            "0.9, 2.1, 3.3 NRx per 100 HCPs"
        )
        durability = (
            "The 10-week test effect is held through the 13-week action; review persistence at week 6."
        )
    elif date_phase == "later":
        hcp_count = 12_000
        exposure = min(1.0, option.budget_moved_usd / 600_000)
        low, mid, high = [round(value * exposure) for value in (40, 172, 304)]
        calculation = (
            f"12,000 eligible HCPs × {exposure:.2f} exposure; community and academic "
            "experiment intervals weighted by audience mix"
        )
        durability = (
            "The 10-week segment effects are held through the 13-week action; review persistence at week 6."
        )
    else:
        hcp_count = 1_600 if option.audience == "community_stable" else 2_400
        ceiling, scale, downside_share = _response_parameters(option, date_phase)
        mid = round(ceiling * (1 - math.exp(-option.budget_moved_usd / scale)))
        low = max(0, round(mid * (1 - downside_share)))
        high = round(mid * (1 + downside_share))
        calculation = (
            f"planning response curve: ceiling={ceiling:.0f}, scale={int(scale)}, "
            f"uncertainty={downside_share:.2f}"
        )
        durability = "Planning prior only; the matched-market test supplies the durability read."
    downside = max(0, mid - low)
    learning = "high" if option.is_experiment else "moderate" if option.budget_moved_usd else "none"
    if not notes:
        notes.append("Inside the approved budget, audience, duration, and measurement boundaries.")
    return ScenarioResult(
        option_name=option.name,
        feasible=feasible,
        budget_moved_usd=option.budget_moved_usd,
        expected_incr_nrx_low=low,
        expected_incr_nrx_mid=mid,
        expected_incr_nrx_high=high,
        downside_nrx=downside,
        audience_hcp_count=hcp_count,
        learning_value=learning,
        constraint_notes=notes,
        calculation=calculation,
        durability_assumption=durability,
    )


def simulate_options(options: list[DecisionOption], date_phase: str) -> list[ScenarioResult]:
    return [simulate_budget_scenario(option, date_phase) for option in options]


def design_experiment(option: DecisionOption) -> ExperimentDesign | None:
    if not option.is_experiment:
        return None
    return ExperimentDesign(
        eligible_dmas=22,
        test_dmas=11,
        holdout_dmas=11,
        duration_weeks=option.duration_weeks,
        primary_outcome="incremental NRx per 100 targeted HCPs",
        minimum_detectable_effect=1.4,
        estimated_cost_usd=option.budget_moved_usd,
        readout_date="2026-10-01",
    )
