"""Shared treatment-attempt scenario logic for Chapter 3 synthetic data."""
from __future__ import annotations

import random
from datetime import date, timedelta

from .entities import EntityBundle, rand_date


def _quarter_state_as_of(states: list[dict], as_of: date) -> dict[str, str]:
    current = states[0]
    for state in states:
        effective = date.fromisoformat(state["effective_date"])
        if effective <= as_of:
            current = state
    return current


def build_launch_treatment_attempts(
    rng: random.Random,
    bundle: EntityBundle,
    quarterly_states: dict[tuple[str, str], list[dict]],
) -> dict[str, dict]:
    """Create one coherent Roventra treatment-attempt path per eligible patient.

    The attempt is latent. Downstream generators turn it into EHR orders,
    specialty-pharmacy cases, and pharmacy transaction chains.
    """
    attempts: dict[str, dict] = {}
    launch_ndc = bundle.ndc_codes[0]["NDC"]
    global_start = date(2024, 1, 1)
    global_end = date(2025, 1, 31)

    for patient in bundle.patients:
        if patient["condition_bucket"] != "Launch condition":
            continue
        if patient["patient_id"] != "PAT02034" and rng.random() >= 0.68:
            continue
        patient_start = max(global_start, date.fromisoformat(patient["_coverage_start"]))
        patient_end = min(global_end, date.fromisoformat(patient["_coverage_end"]))
        if patient_start > patient_end:
            continue

        order_window_start = max(patient_start, date(2024, 2, 1))
        order_window_end = min(patient_end, date(2024, 12, 20))
        if order_window_start > order_window_end:
            order_window_start = patient_start
            order_window_end = patient_end
        order_date = rand_date(rng, order_window_start, order_window_end)

        policy = _quarter_state_as_of(
            quarterly_states[(patient["payer_id"], "Roventra")],
            order_date,
        )
        prior_auth = policy["prior_authorization"] == "Yes"
        step_therapy = policy["step_therapy"] == "Yes"
        quantity_limit = policy["quantity_limit"] == "Yes"
        specialty_required = policy["specialty_pharmacy"] == "Yes"
        specialty_case = specialty_required or rng.random() < 0.82

        friction_score = sum(
            int(flag)
            for flag in (prior_auth, step_therapy, quantity_limit, specialty_case)
        )

        referral_date = (
            min(order_date + timedelta(days=rng.randint(0, 6)), patient_end)
            if specialty_case
            else None
        )
        auth_needed = specialty_case or prior_auth
        if auth_needed:
            pending_prob = 0.03 if order_date.month >= 11 else 0.01
            approve_prob = max(0.58, 0.90 - 0.05 * friction_score)
            draw = rng.random()
            if draw < pending_prob:
                auth_status = "Pending"
            elif draw < pending_prob + approve_prob:
                auth_status = "Approved"
            else:
                auth_status = "Denied"
            decision_anchor = referral_date or order_date
            auth_decision_date = min(
                decision_anchor + timedelta(days=rng.randint(1, 8)),
                patient_end,
            )
        else:
            auth_status = "Approved"
            auth_decision_date = min(order_date + timedelta(days=1), patient_end)

        first_submission_date = min(
            auth_decision_date + timedelta(days=rng.randint(0, 5)),
            patient_end,
        )
        if auth_status == "Denied":
            first_status = "Rejected"
            resolved_status = "Abandoned"
            paid_after_reject = False
        elif auth_status == "Pending":
            first_status = "Rejected"
            resolved_status = "Pending"
            paid_after_reject = False
        else:
            reject_prob = 0.04
            reject_prob += 0.10 if prior_auth else 0.0
            reject_prob += 0.12 if quantity_limit else 0.0
            reject_prob += 0.05 if step_therapy else 0.0
            reject_prob += 0.03 if specialty_case else 0.0
            if rng.random() < min(reject_prob, 0.42):
                first_status = "Rejected"
                paid_after_reject = rng.random() < max(0.55, 0.92 - 0.06 * friction_score)
                resolved_status = "Paid" if paid_after_reject else "Abandoned"
            else:
                first_status = "Paid"
                resolved_status = "Paid"
                paid_after_reject = False

        first_paid_date = None
        if resolved_status == "Paid":
            if first_status == "Paid":
                first_paid_date = first_submission_date
            else:
                first_paid_date = min(
                    first_submission_date + timedelta(days=rng.randint(2, 9)),
                    patient_end,
                )

        ship_date = None
        if specialty_case and auth_status == "Approved" and resolved_status == "Paid":
            ship_anchor = max(
                auth_decision_date,
                first_paid_date or auth_decision_date,
            )
            original_delay_draw = rng.randint(2, 12)
            ship_delay_days = 1 + (original_delay_draw - 2) % 3
            ship_date = min(
                ship_anchor + timedelta(days=ship_delay_days),
                patient_end,
            )

        abandonment_reason = ""
        if resolved_status == "Abandoned":
            if auth_status == "Denied":
                abandonment_reason = rng.choice(["Coverage", "Cost", "Documentation"])
            else:
                abandonment_reason = rng.choice(
                    ["Patient decision", "Cost", "Documentation", "Lost follow-up"]
                )

        attempt = {
            "patient_id": patient["patient_id"],
            "prescriber_npi": patient["prescriber_npi"],
            "payer_id": patient["payer_id"],
            "order_date": order_date.isoformat(),
            "product_name": "Roventra",
            "product_ndc": launch_ndc,
            "tier": policy["tier"],
            "prior_authorization": policy["prior_authorization"],
            "step_therapy": policy["step_therapy"],
            "quantity_limit": policy["quantity_limit"],
            "specialty_pharmacy_required": policy["specialty_pharmacy"],
            "specialty_case": specialty_case,
            "referral_date": referral_date.isoformat() if referral_date else "",
            "auth_status": auth_status,
            "auth_decision_date": auth_decision_date.isoformat(),
            "first_submission_date": first_submission_date.isoformat(),
            "first_status": first_status,
            "first_paid_date": first_paid_date.isoformat() if first_paid_date else "",
            "resolved_status": resolved_status,
            "ship_date": ship_date.isoformat() if ship_date else "",
            "abandonment_reason": abandonment_reason,
            "channel": "Hub" if specialty_case else "Retail",
            "copay_assistance": "Yes" if friction_score >= 2 or rng.random() < 0.55 else "No",
        }
        if patient["patient_id"] == "PAT02034":
            attempt.update(
                {
                    "order_date": "2024-06-12",
                    "specialty_case": True,
                    "referral_date": "2024-06-13",
                    "auth_status": "Approved",
                    "auth_decision_date": "2024-06-18",
                    "tier": "Specialty",
                    "prior_authorization": "Yes",
                    "step_therapy": "Yes",
                    "quantity_limit": "Yes",
                    "specialty_pharmacy_required": "Yes",
                    "first_submission_date": "2024-07-02",
                    "first_status": "Rejected",
                    "reject_reason": (
                        "New-to-market review; exception documentation required"
                    ),
                    "exception_path": "Clinical exception",
                    "first_paid_date": "2024-07-09",
                    "resolved_status": "Paid",
                    "ship_date": "2024-07-10",
                    "abandonment_reason": "",
                    "channel": "Hub",
                    "copay_assistance": "Yes",
                }
            )
        else:
            attempt["reject_reason"] = (
                "Utilization management requirement"
                if first_status == "Rejected"
                else ""
            )
            attempt["exception_path"] = ""
        attempts[patient["patient_id"]] = attempt
    return attempts
