"""Specialty pharmacy event generator for ch03 synthetic data."""
from __future__ import annotations

import random
from datetime import date

from .entities import EntityBundle


SP_EVENT_FIELDS = [
    "event_id", "patient_id", "prescriber_npi", "product_ndc", "referral_date",
    "hub_status", "status_date", "ship_date", "days_supply", "dispense_status",
    "copay_assistance", "discontinue_reason",
]


def generate_specialty_pharmacy(
    rng: random.Random,
    bundle: EntityBundle,
    treatment_attempts: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    sp_id = 1
    treatment_attempts = treatment_attempts or {}
    global_start = date(2024, 1, 1)
    global_end = date(2025, 1, 15)

    for patient in bundle.patients:
        attempt = treatment_attempts.get(patient["patient_id"])
        if not attempt or not attempt["specialty_case"]:
            continue
        patient_start = max(global_start, date.fromisoformat(patient["_coverage_start"]))
        patient_end = min(global_end, date.fromisoformat(patient["_coverage_end"]))
        if patient_start > patient_end:
            continue

        prescriber_npi = patient["prescriber_npi"]
        referral = max(patient_start, date.fromisoformat(attempt["referral_date"]))
        status_date = max(referral, date.fromisoformat(attempt["auth_decision_date"]))
        hub_status = attempt["auth_status"]

        if attempt["resolved_status"] == "Paid":
            dispense_status = "Shipped"
            ship_value = (
                min(date.fromisoformat(attempt["ship_date"]), patient_end).isoformat()
                if attempt["ship_date"]
                else ""
            )
            discontinue_reason = ""
        elif attempt["resolved_status"] == "Pending":
            dispense_status = "Pending"
            ship_value = ""
            discontinue_reason = ""
        else:
            dispense_status = "Abandoned"
            ship_value = ""
            discontinue_reason = attempt.get("abandonment_reason", "")

        rows.append(
            {
                "event_id": f"SP{sp_id:07d}",
                "patient_id": patient["patient_id"],
                "prescriber_npi": prescriber_npi,
                "product_ndc": attempt["product_ndc"],
                "referral_date": referral.isoformat(),
                "hub_status": hub_status,
                "status_date": status_date.isoformat(),
                "ship_date": ship_value,
                "days_supply": attempt.get("days_supply", 28),
                "dispense_status": dispense_status,
                "copay_assistance": attempt["copay_assistance"],
                "discontinue_reason": discontinue_reason,
            }
        )
        sp_id += 1
    return rows
