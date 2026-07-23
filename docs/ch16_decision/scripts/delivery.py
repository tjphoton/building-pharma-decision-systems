"""Decision views, learning summaries, and persisted generated outputs."""

from __future__ import annotations

import json
from pathlib import Path

from models import (
    GroundedInsight,
    LearningSummary,
    RoleView,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "generated_outputs"


def build_grounded_insight(state) -> GroundedInsight:
    scenario = next(
        item for item in state.scenarios if item.option_name == state.analyst.selected_option_name
    )
    return GroundedInsight(
        observation=state.integration.marginal_return_read,
        inference=state.analyst.rationale,
        recommendation=(
            f"{state.analyst.selected_option_name}: move ${scenario.budget_moved_usd:,}; "
            f"expected {scenario.expected_incr_nrx_low} to {scenario.expected_incr_nrx_high} "
            "incremental NRx."
        ),
        uncertainty=state.integration.remaining_uncertainty,
        evidence_ids=state.analyst.evidence_ids,
        revisit_date="2026-10-06" if state.date_phase == "first" else "2027-01-05",
    )


def build_role_views(state) -> list[RoleView]:
    scenario = next(
        item for item in state.scenarios if item.option_name == state.analyst.selected_option_name
    )
    common = f"{state.analyst.selected_option_name}; ${scenario.budget_moved_usd:,} moved."
    return [
        RoleView(role="brand leadership", headline=common, detail=state.analyst.rationale),
        RoleView(role="measurement", headline=state.analyst.measurement_plan,
                 detail=f"Citations: {', '.join(state.analyst.evidence_ids)}"),
        RoleView(role="omnichannel execution", headline=common,
                 detail="Release only to the audience and geography named in the selected option."),
        RoleView(role="finance", headline=common,
                 detail=f"Expected range: {scenario.expected_incr_nrx_low} to "
                        f"{scenario.expected_incr_nrx_high} incremental NRx."),
    ]


def build_learning_summary(state) -> LearningSummary | None:
    """Compare the expected range from the prior decision record with the observed outcome.

    The numbers come from the durable ``prior_decision`` and ``outcome_event`` loaded when the
    case reopens, not from literals. Without both records (for example a standalone later-date
    graph run that never went through the runtime) there is no learning to report yet.
    """
    prior = getattr(state, "prior_decision", None)
    outcome = getattr(state, "outcome_event", None)
    if prior is None or outcome is None:
        return None
    return LearningSummary(
        expected_range=(
            f"{prior.expected_incr_nrx_low} to {prior.expected_incr_nrx_high} incremental NRx"
        ),
        observed_result=f"{outcome.observed_incremental_nrx} incremental NRx",
        changed_evidence=(
            "Mature claims and the matched-market read confirmed a community-practice effect "
            f"in stable-access DMAs ({outcome.maturity_status} outcome from {outcome.source})."
        ),
        decision_implication=(
            "Scale stable-access community practices and keep the academic segment out of the move."
        ),
        remaining_uncertainty=(
            "The effect outside stable-access community practices remains unproven."
        ),
    )


def persist_state(state, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2) + "\n")
    return path
