"""Market access and formulary generators for Chapter 3 synthetic data."""
from __future__ import annotations

import random
from datetime import date

from .entities import (
    EntityBundle,
    REGIONS,
)

ACCESS_STATUS = ["Covered", "Covered with PA", "Covered with Step Edit", "Non-covered"]
FORMULARY_TIER = ["Tier 1", "Tier 2", "Tier 3", "Specialty"]
PLAN_FIELDS = [
    "tier",
    "prior_authorization",
    "step_therapy",
    "quantity_limit",
    "specialty_pharmacy",
]
HISTORY_FIELDS = PLAN_FIELDS
CANONICAL_ACCESS_KEY = ("PAY002", "Roventra")
CANONICAL_ACCESS_STATE = {
    "tier": "Specialty",
    "prior_authorization": "Yes",
    "step_therapy": "Yes",
    "quantity_limit": "Yes",
    "specialty_pharmacy": "Yes",
}


def _initial_state(rng: random.Random) -> dict[str, str]:
    return {
        "tier": rng.choice(FORMULARY_TIER),
        "prior_authorization": rng.choice(["Yes", "No", "Yes"]),
        "step_therapy": rng.choice(["Yes", "No", "No"]),
        "quantity_limit": rng.choice(["Yes", "No", "Yes"]),
        "specialty_pharmacy": rng.choice(["Yes", "No"]),
    }


def _state_to_access_status(state: dict[str, str]) -> str:
    if state["tier"] == "Specialty":
        return "Non-covered"
    if state["prior_authorization"] == "Yes":
        return "Covered with PA"
    if state["step_therapy"] == "Yes":
        return "Covered with Step Edit"
    return "Covered"


def _build_formulary_states(
    rng: random.Random,
    bundle: EntityBundle,
) -> tuple[list[dict], dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], list[dict]]]:
    quarters = [
        (date(2024, 1, 1), "2024-Q1"),
        (date(2024, 4, 1), "2024-Q2"),
        (date(2024, 7, 1), "2024-Q3"),
        (date(2024, 10, 1), "2024-Q4"),
    ]
    history_rows: list[dict] = []
    final_states: dict[tuple[str, str], dict[str, str]] = {}
    quarterly_states: dict[tuple[str, str], list[dict]] = {}
    record_id = 1

    for payer in bundle.payers:
        for product in bundle.products:
            key = (payer["payer_id"], product["product_name"])
            current = _initial_state(rng)
            states = []
            for quarter_start, quarter_label in quarters:
                if rng.random() < 0.20:
                    candidate = dict(current)
                    while True:
                        field = rng.choice(PLAN_FIELDS)
                        if field == "tier":
                            candidate[field] = rng.choice(FORMULARY_TIER)
                        else:
                            candidate[field] = "Yes" if current[field] == "No" else "No"
                        if any(candidate[name] != current[name] for name in PLAN_FIELDS):
                            break
                    change_type = (
                        "Tier change"
                        if candidate["tier"] != current["tier"]
                        else "Restriction change"
                    )
                    history_rows.append(
                        {
                            "history_id": f"FH{record_id:06d}",
                            "plan_id": payer["payer_id"],
                            "plan_name": payer["payer_name"],
                            "product_name": product["product_name"],
                            "quarter": quarter_label,
                            "effective_date": quarter_start.isoformat(),
                            "prior_tier": current["tier"],
                            "new_tier": candidate["tier"],
                            "prior_prior_authorization": current["prior_authorization"],
                            "new_prior_authorization": candidate["prior_authorization"],
                            "prior_step_therapy": current["step_therapy"],
                            "new_step_therapy": candidate["step_therapy"],
                            "prior_quantity_limit": current["quantity_limit"],
                            "new_quantity_limit": candidate["quantity_limit"],
                            "prior_specialty_pharmacy": current["specialty_pharmacy"],
                            "new_specialty_pharmacy": candidate["specialty_pharmacy"],
                            "change_type": change_type,
                        }
                    )
                    current = candidate
                    record_id += 1
                states.append(
                    {
                        "quarter": quarter_label,
                        "effective_date": quarter_start.isoformat(),
                        **current,
                    }
                )
            final_states[key] = dict(current)
            quarterly_states[key] = states

    if CANONICAL_ACCESS_KEY not in quarterly_states:
        raise ValueError("Canonical PAY002/Roventra access state is missing.")
    final_states[CANONICAL_ACCESS_KEY] = dict(CANONICAL_ACCESS_STATE)

    history_rows = [
        row
        for row in history_rows
        if (row["plan_id"], row["product_name"]) != CANONICAL_ACCESS_KEY
    ]
    return history_rows, final_states, quarterly_states


def generate_formulary_bundle(
    rng: random.Random,
    bundle: EntityBundle,
) -> tuple[list[dict], list[dict], list[dict], dict[tuple[str, str], list[dict]]]:
    history_rows, final_states, quarterly_states = _build_formulary_states(rng, bundle)
    rows: list[dict] = []
    formulary_rows: list[dict] = []
    for payer in bundle.payers:
        for product in bundle.products:
            state = final_states[(payer["payer_id"], product["product_name"])]
            for region in REGIONS:
                rows.append(
                    {
                        "access_rule_id": f"AR{len(rows)+1:06d}",
                        "payer_id": payer["payer_id"],
                        "payer_type": payer["payer_type"],
                        "region": region,
                        "product_name": product["product_name"],
                        "coverage_status": _state_to_access_status(state),
                        "prior_authorization": state["prior_authorization"],
                        "step_edit": state["step_therapy"],
                        "specialty_pharmacy_required": state["specialty_pharmacy"],
                        "effective_start": date(2024, 1, 1).isoformat(),
                        "effective_end": date(2025, 12, 31).isoformat(),
                    }
                )
            formulary_rows.append(
                {
                    "formulary_id": f"F{len(formulary_rows)+1:06d}",
                    "plan_id": payer["payer_id"],
                    "plan_name": payer["payer_name"],
                    "product_name": product["product_name"],
                    "tier": state["tier"],
                    "prior_authorization": state["prior_authorization"],
                    "step_therapy": state["step_therapy"],
                    "quantity_limit": state["quantity_limit"],
                    "specialty_pharmacy": state["specialty_pharmacy"],
                    "effective_start": date(2024, 1, 1).isoformat(),
                    "effective_end": date(2025, 12, 31).isoformat(),
                }
            )
    return rows, formulary_rows, history_rows, quarterly_states


def generate_access(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    access_rows, _, _, _ = generate_formulary_bundle(rng, bundle)
    return access_rows


def generate_formulary(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    _, formulary_rows, _, _ = generate_formulary_bundle(rng, bundle)
    return formulary_rows


def generate_formulary_history(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    """Generate dated formulary state changes for 2024 from one state model."""
    _, _, history_rows, _ = generate_formulary_bundle(rng, bundle)
    return history_rows
