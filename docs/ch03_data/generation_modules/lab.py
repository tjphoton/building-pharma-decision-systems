"""Lab results generator for ch03 synthetic data.

Lab data comes from the same longitudinal claims vendor via their specialty lab
partnerships (Quest, LabCorp, etc.). Results are LOINC-coded and linked to the
same patient population as claims.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .entities import (
    EntityBundle,
    LAB_REFERENCE_RANGES,
    LAB_TESTS,
    rand_date,
)

# LOINC codes used in this dataset: (test_name → loinc_code)
_LOINC_BY_NAME = {name: loinc for name, loinc, _ in LAB_TESTS}
_UNIT_BY_NAME = {name: unit for name, _, unit in LAB_TESTS}

LAB_RESULT_FIELDS = [
    "lab_id", "patient_id", "service_date", "loinc_code", "test_name",
    "result", "result_unit", "ref_low", "ref_high", "abnormal_flag",
    "ordering_npi", "diagnosis_1",
]

# Internal short keys → readable test names (for condition-based value generation)
_SHORT_TO_FULL = {
    "A1C": "Hemoglobin A1c",
    "LDL": "LDL Cholesterol",
    "BP_SYSTOLIC": "Systolic BP",
    "BP_DIASTOLIC": "Diastolic BP",
    "AST": "AST",
    "ALT": "ALT",
    "PSA": "PSA",
    "PD_L1_PERCENT": "PD-L1 TPS",
}
_SHORT_KEYS = list(_SHORT_TO_FULL.keys())


def _lab_value(rng: random.Random, short_key: str, bucket: str) -> float:
    if short_key == "A1C":
        if bucket == "Launch condition":
            return round(rng.triangular(6.0, 11.5, 7.9), 1)
        return round(rng.triangular(4.6, 7.2, 5.4), 1)
    if short_key == "LDL":
        if bucket == "Cardiology":
            return round(rng.triangular(70, 215, 150), 0)
        return round(rng.triangular(45, 190, 105), 0)
    if short_key == "BP_SYSTOLIC":
        return float(rng.randint(118, 185) if bucket == "Cardiology" else rng.randint(102, 165))
    return {
        "BP_DIASTOLIC": lambda: float(rng.randint(62, 108)),
        "AST":           lambda: float(rng.randint(12, 88)),
        "ALT":           lambda: float(rng.randint(10, 96)),
        "PSA":           lambda: round(rng.lognormvariate(1.0, 0.8), 2),
        "PD_L1_PERCENT": lambda: round(rng.uniform(0, 80), 1),
    }[short_key]()


def _abnormal_flag(short_key: str, value: float) -> str:
    lo, hi = LAB_REFERENCE_RANGES[short_key]
    if value > hi:
        return "H"
    if value < lo:
        return "L"
    return ""


def _append_lab(
    rows: list[dict],
    lab_id_ref: list[int],
    patient: dict,
    ordering_npi: str,
    short_key: str,
    service_date: date,
    value: float,
    dx_code: str = "",
) -> None:
    full_name = _SHORT_TO_FULL[short_key]
    ref_lo, ref_hi = LAB_REFERENCE_RANGES[short_key]
    rows.append(
        {
            "lab_id": f"LAB{lab_id_ref[0]:07d}",
            "patient_id": patient["patient_id"],
            "service_date": service_date.isoformat(),
            "loinc_code": _LOINC_BY_NAME.get(full_name, ""),
            "test_name": full_name,
            "result": value,
            "result_unit": _UNIT_BY_NAME.get(full_name, ""),
            "ref_low": ref_lo,
            "ref_high": ref_hi,
            "abnormal_flag": _abnormal_flag(short_key, value),
            "ordering_npi": ordering_npi,
            "diagnosis_1": dx_code,
        }
    )
    lab_id_ref[0] += 1


def generate_lab_results(
    rng: random.Random,
    bundle: EntityBundle,
    treatment_attempts: dict[str, dict] | None = None,
) -> list[dict]:
    """Generate lab results for all patients.

    Produces LOINC-coded rows with numeric results and abnormal flags.
    PAT02034 gets a pinned A1C timeline to support the teaching chain.
    """
    treatment_attempts = treatment_attempts or {}

    start_dt = date(2024, 1, 1)
    end_dt = date(2025, 1, 31)

    rows: list[dict] = []
    lab_id_ref = [1]
    pat02034_rows: list[dict] = []

    for patient in bundle.patients:
        bucket = patient["condition_bucket"]
        ordering_npi = patient["prescriber_npi"]
        patient_start = max(start_dt, date.fromisoformat(patient["_coverage_start"]))
        patient_end = min(end_dt, date.fromisoformat(patient["_coverage_end"]))
        if patient_start > patient_end:
            continue

        attempt = treatment_attempts.get(patient["patient_id"])

        if patient["patient_id"] == "PAT02034":
            # Canonical teaching patient — pinned A1C timeline
            _build_pat02034_labs(pat02034_rows, lab_id_ref, patient, ordering_npi)
            continue

        # Patients on the launch drug get two A1C measurements bracketing their start
        if attempt and bucket == "Launch condition":
            order_date = date.fromisoformat(attempt.get("order_date", "2024-06-01"))
            baseline = max(patient_start, order_date - timedelta(days=rng.randint(60, 110)))
            a1c_base = round(rng.triangular(6.0, 11.5, 7.7), 1)
            _append_lab(rows, lab_id_ref, patient, ordering_npi, "A1C", baseline, a1c_base, "E11.9")
            follow_up = min(order_date + timedelta(days=rng.randint(7, 21)), patient_end)
            a1c_follow = round(rng.triangular(6.4, 11.8, 8.4), 1)
            _append_lab(rows, lab_id_ref, patient, ordering_npi, "A1C", follow_up, a1c_follow, "E11.9")

        # Background labs — 1–3 tests per patient, condition-skewed selection
        base_keys = [k for k in _SHORT_KEYS if k not in ("PSA", "PD_L1_PERCENT")]
        selected = rng.sample(base_keys, k=rng.randint(1, 3))
        if bucket == "Oncology":
            if patient["sex"] == "M" and rng.random() < 0.7:
                selected.append("PSA")
            if rng.random() < 0.6:
                selected.append("PD_L1_PERCENT")

        for short_key in selected:
            svc_date = rand_date(rng, patient_start, patient_end)
            value = _lab_value(rng, short_key, bucket)
            dx = "E11.9" if bucket == "Launch condition" else ""
            _append_lab(rows, lab_id_ref, patient, ordering_npi, short_key, svc_date, value, dx)

    # Insert PAT02034 rows at roughly the same position (just append to end; sort handled downstream)
    rows.extend(pat02034_rows)
    return rows


def _build_pat02034_labs(
    rows: list[dict],
    lab_id_ref: list[int],
    patient: dict,
    ordering_npi: str,
) -> None:
    """Append the fixed A1C and LDL timeline for PAT02034."""
    pinned: list[tuple[str, str, float, str]] = [
        ("A1C",  "2024-06-13", 10.6, "E11.40"),
        ("A1C",  "2024-06-22",  8.9, "E11.40"),
        ("LDL",  "2024-12-30", 170.0, "E78.5"),
        ("A1C",  "2025-01-01",  8.4, "E11.40"),
    ]
    for short_key, svc_date_str, value, dx_code in pinned:
        svc_date = date.fromisoformat(svc_date_str)
        _append_lab(rows, lab_id_ref, patient, ordering_npi, short_key, svc_date, value, dx_code)
