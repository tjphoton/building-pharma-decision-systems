"""Deterministic checks around the language-model judgments."""

from __future__ import annotations

from models import (
    AnalystOutput,
    DecisionOption,
    EvidenceItem,
    ScenarioResult,
    ValidationResult,
)
from tools import TOOL_CATALOG


def normalize_tool_requests(requested: list[str]) -> list[str]:
    """Repair only an omitted ``get_`` prefix when the catalog match is unique."""
    aliases = {name.removeprefix("get_"): name for name in TOOL_CATALOG if name.startswith("get_")}
    return [name if name in TOOL_CATALOG else aliases.get(name, name) for name in requested]


def validate_tool_requests(requested: list[str]) -> ValidationResult:
    unknown = sorted(set(requested) - set(TOOL_CATALOG))
    return ValidationResult(
        status="fail" if unknown else "pass",
        checks=["Every requested tool is in the approved catalog."],
        issues=[f"Unknown tools: {', '.join(unknown)}"] if unknown else [],
    )


def validate_recommendation(
    options: list[DecisionOption],
    scenarios: list[ScenarioResult],
    analyst: AnalystOutput,
    evidence: list[EvidenceItem],
) -> ValidationResult:
    issues: list[str] = []
    option_names = {option.name for option in options}
    option_by_name = {option.name: option for option in options}
    scenario_by_name = {scenario.option_name: scenario for scenario in scenarios}
    evidence_ids = {item.evidence_id for item in evidence}

    if analyst.selected_option_name not in option_names:
        issues.append("The selected option is absent from the generated option set.")
    selected = scenario_by_name.get(analyst.selected_option_name)
    if selected is None:
        issues.append("The selected option has no deterministic scenario result.")
    elif not selected.feasible:
        issues.append("The selected option violates an approved action boundary.")
    selected_option = option_by_name.get(analyst.selected_option_name)
    overlapping_market_event = any(
        item.source == "market_events" and "overlap" in item.claim.lower()
        for item in evidence
    )
    if (
        overlapping_market_event
        and selected_option is not None
        and selected_option.geography == "all_dmas"
    ):
        issues.append(
            "An overlapping market event blocks an all-DMA release for this decision."
        )
    missing = sorted(set(analyst.evidence_ids) - evidence_ids)
    if missing:
        issues.append(f"Recommendation cites unknown evidence: {', '.join(missing)}")
    if not analyst.evidence_ids:
        issues.append("Recommendation has no evidence citations.")

    return ValidationResult(
        status="fail" if issues else "pass",
        checks=[
            "Selected option exists.",
            "Selected option has a feasible deterministic scenario.",
            "Recommendation citations resolve to collected evidence.",
            "An overlapping market event does not enter an all-DMA release.",
        ],
        issues=issues,
    )
