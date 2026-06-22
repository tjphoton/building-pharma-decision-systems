"""Medical and pharmacy claims generators for ch03 synthetic data."""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta

from .data_noise import apply_ndc_variation, claim_lag_days
from .entities import EntityBundle, PROBLEM_LIST_CODES, rand_date


CLAIM_CODES_BY_BUCKET = {
    "Launch condition": ["E11.9", "E11.65", "E11.40"],
    "Oncology": ["C34.10", "C34.90", "Z85.118"],
    "Cardiology": ["I50.9", "I10", "I25.10"],
    "Rheumatology": ["M06.9", "M05.79", "M32.9"],
    "Other": ["F41.9", "G89.3", "K21.9"],
}

LAUNCH_DIAGNOSIS_DOCUMENTATION_RATE = 0.86
LAUNCH_DIAGNOSIS_FALSE_POSITIVE_RATE = 0.015

# Comorbidity ICD codes appended as secondary diagnoses
COMORBIDITY_POOL = ["I10", "E78.5", "K21.9", "F41.9", "N18.9", "J45.909"]

# CPT / HCPCS codes for service lines
PROCEDURE_CODES = ["99213", "99214", "99215", "96413", "J9999", "93000", "80053", "99232"]

# Place of service codes (CMS standard numeric codes)
POS_BY_CLAIM_TYPE = {
    "Institutional": ["21", "22"],
    "Professional": ["11", "11", "11", "02"],
}

# ICD procedure codes (Institutional only)
ICD_PROC_CODES = ["0BH17EZ", "5A1935Z", "3E0736Z", ""]


def _stable_patient_fraction(patient_id: str, salt: str) -> float:
    """Return a reproducible number in [0, 1) without consuming the RNG stream."""
    digest = hashlib.sha256(f"{salt}:{patient_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _payer_type_from_id(payer_id: str, bundle: EntityBundle) -> str:
    for p in bundle.payers:
        if p["payer_id"] == payer_id:
            return p["payer_type"]
    return "Commercial"


def generate_medical_headers(
    rng: random.Random, bundle: EntityBundle
) -> list[dict]:
    """Generate medical claim headers (encounter grain) with internal lag metadata.

    Each row includes a private `_lag_days` field used by filter_by_snapshot_date()
    to produce two snapshot files. That field is stripped before writing to CSV.
    """
    payer_type_map: dict[str, str] = {p["payer_id"]: p["payer_type"] for p in bundle.payers}

    start = date(2024, 1, 1)
    end = date(2025, 1, 31)

    rows: list[dict] = []
    encounter_id = 1

    for patient in bundle.patients:
        patient_start = max(start, date.fromisoformat(patient["_coverage_start"]))
        patient_end = min(end, date.fromisoformat(patient["_coverage_end"]))
        if patient_start > patient_end:
            continue

        rendering_npi = patient["prescriber_npi"]
        payer_type = payer_type_map.get(patient["payer_id"], "Commercial")

        true_launch_condition = patient["condition_bucket"] == "Launch condition"
        launch_diagnosis_documented = patient["patient_id"] == "PAT02034" or (
            true_launch_condition
            and _stable_patient_fraction(patient["patient_id"], "launch-diagnosis-documented")
            < LAUNCH_DIAGNOSIS_DOCUMENTATION_RATE
        )
        false_positive_launch_diagnosis = (
            not true_launch_condition
            and _stable_patient_fraction(
                patient["patient_id"], "launch-diagnosis-false-positive"
            )
            < LAUNCH_DIAGNOSIS_FALSE_POSITIVE_RATE
        )

        observed_bucket = patient["condition_bucket"]
        diagnosis_pool = CLAIM_CODES_BY_BUCKET.get(
            observed_bucket, CLAIM_CODES_BY_BUCKET["Other"]
        )
        diagnosis_rows = [
            row for row in bundle.diagnosis_codes if row["CODE"] in diagnosis_pool
        ] or [
            row
            for row in bundle.diagnosis_codes
            if row["CODE"] in PROBLEM_LIST_CODES.get(observed_bucket, [])
        ] or bundle.diagnosis_codes

        launch_diagnosis_rows = [
            row
            for row in bundle.diagnosis_codes
            if row["CODE"] in CLAIM_CODES_BY_BUCKET["Launch condition"]
        ]
        diagnosis_by_code = {row["CODE"]: row for row in bundle.diagnosis_codes}

        claim_count = rng.randint(1, 4)
        for claim_index in range(claim_count):
            diag = rng.choice(diagnosis_rows)
            if true_launch_condition and not launch_diagnosis_documented:
                diag = diagnosis_by_code.get("F41.9", rng.choice(diagnosis_rows))
            if false_positive_launch_diagnosis and claim_index == 0:
                diag = launch_diagnosis_rows[0] if launch_diagnosis_rows else diag

            claim_date = rand_date(rng, patient_start, patient_end)
            claim_type = rng.choice(["Institutional", "Professional", "Professional", "Professional"])

            # Secondary diagnoses (comorbidities)
            secondary_count = rng.randint(0, 3)
            secondary_codes = rng.sample(
                [c for c in COMORBIDITY_POOL if c != diag["CODE"]],
                k=min(secondary_count, len(COMORBIDITY_POOL) - 1),
            )

            dx: list[str] = [diag["CODE"]] + secondary_codes
            while len(dx) < 10:
                dx.append("")

            # ICD procedure codes (Institutional only)
            icd_procs = ["", "", ""]
            if claim_type == "Institutional" and rng.random() < 0.6:
                icd_procs[0] = rng.choice([c for c in ICD_PROC_CODES if c])

            # Admitting diagnosis: institutional-only, populated ~4% of institutional
            # claims so the overall rate is ~1% of all encounters — matching Komodo data.
            # When populated it equals D1 ~62% of the time; the other ~38% it holds a
            # comorbidity code not already listed in D1-D10 (mimicking cases where the
            # admitting reason differs from the principal discharge diagnosis).
            admitting_dx = ""
            if claim_type == "Institutional" and rng.random() < 0.04:
                if rng.random() < 0.62:
                    admitting_dx = diag["CODE"]
                else:
                    alt = [c for c in COMORBIDITY_POOL if c not in dx]
                    admitting_dx = rng.choice(alt) if alt else diag["CODE"]

            lag = claim_lag_days(rng)
            rows.append(
                {
                    "encounter_id": f"ENC{encounter_id:07d}",
                    "patient_id": patient["patient_id"],
                    "claim_type": claim_type,
                    "claim_date": claim_date.isoformat(),
                    "admitting_diagnosis": admitting_dx,
                    "diagnosis_1": dx[0],
                    "diagnosis_2": dx[1],
                    "diagnosis_3": dx[2],
                    "diagnosis_4": dx[3],
                    "diagnosis_5": dx[4],
                    "diagnosis_6": dx[5],
                    "diagnosis_7": dx[6],
                    "diagnosis_8": dx[7],
                    "diagnosis_9": dx[8],
                    "diagnosis_10": dx[9],
                    "icd_procedure_1": icd_procs[0],
                    "icd_procedure_2": icd_procs[1],
                    "icd_procedure_3": icd_procs[2],
                    "patient_gender": patient["sex"],
                    "patient_state": patient["state"],
                    "coverage_type": payer_type,
                    "rendering_npi": rendering_npi,
                    "attending_npi": rendering_npi if claim_type == "Institutional" else "",
                    "referring_npi": "",
                    "facility_npi": "",
                    "payer_id": patient["payer_id"],
                    # Internal field used for snapshot filtering — not written to CSV
                    "_lag_days": lag,
                    "_claim_date_obj": claim_date,
                }
            )
            encounter_id += 1

    return rows


def generate_service_lines(
    headers: list[dict], rng: random.Random
) -> list[dict]:
    """Generate one or two service lines per encounter header."""
    rows: list[dict] = []
    for header in headers:
        claim_type = header["claim_type"]
        claim_date = header["_claim_date_obj"]
        pos_options = POS_BY_CLAIM_TYPE.get(claim_type, ["11"])
        pos = rng.choice(pos_options)

        line_count = 1 if rng.random() < 0.75 else 2
        for line_num in range(1, line_count + 1):
            proc = rng.choice(PROCEDURE_CODES)
            service_to = claim_date if claim_type == "Professional" else (
                claim_date + timedelta(days=rng.randint(0, 5))
            )
            line_dx1 = header["diagnosis_1"]
            line_dx2 = header["diagnosis_2"] if line_num == 1 else header["diagnosis_3"]
            rows.append(
                {
                    "encounter_id": header["encounter_id"],
                    "patient_id": header["patient_id"],
                    "line_number": line_num,
                    "service_from": claim_date.isoformat(),
                    "service_to": service_to.isoformat(),
                    "procedure_code": proc,
                    "place_of_service": pos,
                    "line_diagnosis_1": line_dx1,
                    "line_diagnosis_2": line_dx2,
                    "units": 1,
                    "line_charge": round(rng.uniform(80, 1_200), 2),
                }
            )
    return rows


def filter_by_snapshot_date(headers: list[dict], cutoff_date: date) -> list[dict]:
    """Return only encounters whose claim_date + lag_days <= cutoff_date.

    Used to produce the early (less complete) snapshot file.
    All encounters are included in the mature snapshot (no filtering needed).
    """
    result = []
    for h in headers:
        claim_date = h["_claim_date_obj"]
        lag = h["_lag_days"]
        if claim_date + timedelta(days=lag) <= cutoff_date:
            result.append(h)
    return result


def strip_internal_fields(headers: list[dict]) -> list[dict]:
    """Remove private underscore-prefixed fields before writing to CSV."""
    internal = {"_lag_days", "_claim_date_obj"}
    return [{k: v for k, v in row.items() if k not in internal} for row in headers]


MEDICAL_HEADER_FIELDS = [
    "encounter_id", "patient_id", "claim_type", "claim_date",
    "admitting_diagnosis",
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5",
    "diagnosis_6", "diagnosis_7", "diagnosis_8", "diagnosis_9", "diagnosis_10",
    "icd_procedure_1", "icd_procedure_2", "icd_procedure_3",
    "patient_gender", "patient_state", "coverage_type",
    "rendering_npi", "attending_npi", "referring_npi", "facility_npi",
    "payer_id",
]

SERVICE_LINE_FIELDS = [
    "encounter_id", "patient_id", "line_number",
    "service_from", "service_to", "procedure_code", "place_of_service",
    "line_diagnosis_1", "line_diagnosis_2", "units", "line_charge",
]

PHARMACY_CLAIM_FIELDS = [
    "claim_id", "patient_id", "patient_state", "prescriber_npi", "primary_care_npi",
    "date_of_service", "rx_written_date", "transaction_type",
    "ndc", "ndc_prescribed",
    "refills_authorized", "diagnosis_code",
    "qty_prescribed", "qty_dispensed", "fill_number", "days_supply",
    "patient_pay", "plan_pay", "reject_code", "payer_id",
]


def generate_pharmacy_claims(
    rng: random.Random,
    bundle: EntityBundle,
    treatment_attempts: dict[str, dict] | None = None,
) -> list[dict]:
    """Generate pharmacy transactions with realistic chains and NDC crosswalk gaps.

    Structure preserved on purpose:
    - A PENDED submission may be re-submitted and become PAID a few days later.
    - A PAID transaction can be REVERSED, sometimes followed by a corrected PAID.
    - ~5% of dispensed fills use an unmapped pack-size NDC variant; PAT02034 is pinned.
    - Reversal rows carry negative patient_pay and plan_pay.
    - The prescription_number field does not exist in this vendor's schema; refills
      share fill_number progression (0 = original, 1 = first refill, etc.).
    """
    rows: list[dict] = []
    claim_counter = 1
    start = date(2024, 1, 1)
    end = date(2025, 1, 31)

    launch_ndc = bundle.ndc_codes[0]["NDC"]
    competitor_ndcs = [row["NDC"] for row in bundle.ndc_codes[1:]]

    def _plan_pay(patient_pay: float, days_supply: int) -> float:
        base = days_supply * rng.uniform(2.0, 8.5)
        return round(base * (1 if patient_pay >= 0 else -1), 2)

    def emit(
        patient: dict,
        prescriber_npi: str,
        ndc_prescribed: str,
        ndc_dispensed: str,
        txn_date: date,
        transaction_type: str,
        fill_number: int,
        days_supply: int,
        patient_pay: float,
        reject_code: str = "",
    ) -> None:
        nonlocal claim_counter
        patient_end = min(end, date.fromisoformat(patient["_coverage_end"]))
        txn_date = min(txn_date, patient_end)
        written_date = max(start, txn_date - timedelta(days=rng.randint(0, 5)))
        plan = _plan_pay(patient_pay, days_supply) if transaction_type != "PENDED" else 0.0
        dx_code = "E11.9" if patient["condition_bucket"] == "Launch condition" else ""
        rows.append(
            {
                "claim_id": f"RXCL{claim_counter:07d}",
                "patient_id": patient["patient_id"],
                "patient_state": patient["state"],
                "prescriber_npi": prescriber_npi,
                "primary_care_npi": "",
                "date_of_service": txn_date.isoformat(),
                "rx_written_date": written_date.isoformat(),
                "transaction_type": transaction_type,
                "ndc": ndc_dispensed,
                "ndc_prescribed": ndc_prescribed,
                "refills_authorized": rng.randint(1, 5),
                "diagnosis_code": dx_code,
                "qty_prescribed": days_supply,
                "qty_dispensed": days_supply if transaction_type != "PENDED" else 0,
                "fill_number": fill_number,
                "days_supply": days_supply,
                "patient_pay": patient_pay,
                "plan_pay": plan,
                "reject_code": reject_code,
                "payer_id": patient["payer_id"],
            }
        )
        claim_counter += 1

    treatment_attempts = treatment_attempts or {}

    for patient in bundle.patients:
        patient_start = max(start, date.fromisoformat(patient["_coverage_start"]))
        patient_end = min(end, date.fromisoformat(patient["_coverage_end"]))
        if patient_start > patient_end:
            continue

        prescriber_npi = patient["prescriber_npi"]
        attempt = treatment_attempts.get(patient["patient_id"])

        if patient["patient_id"] == "PAT02034" and attempt:
            # PAT02034 canonical pharmacy chain per redesign plan §5.3
            _emit_pat02034_chain(patient, prescriber_npi, launch_ndc, emit)
        elif attempt:
            _emit_launch_chain(rng, patient, prescriber_npi, launch_ndc, patient_end, attempt, emit)
        else:
            _emit_background_fills(rng, patient, prescriber_npi, competitor_ndcs, patient_start, patient_end, emit)


    return rows


def _emit_pat02034_chain(
    patient: dict,
    prescriber_npi: str,
    launch_ndc: str,
    emit_fn: object,
) -> None:
    """Emit the fixed teaching chain for PAT02034 per §5.3 of the redesign plan."""
    if not callable(emit_fn):
        return
    chain = [
        (date(2024, 7, 2), "PENDED", 0, 28, 0.00, "70"),
        (date(2024, 7, 9), "PAID",    0, 28, 45.06, ""),
        (date(2024, 8, 9), "PAID",    1, 28, 64.88, ""),
        (date(2024, 8, 10), "REVERSED", 1, 28, -64.88, ""),
        (date(2024, 8, 15), "PAID",   1, 28, 64.88, ""),
        (date(2024, 9, 9), "PAID",    2, 28, 45.13, ""),
        (date(2024, 10, 9), "PAID",   3, 28, 59.90, ""),
    ]
    for svc_date, txn_type, fill_num, days, pat_pay, rej_code in chain:
        emit_fn(
            patient=patient,
            prescriber_npi=prescriber_npi,
            ndc_prescribed=launch_ndc,
            ndc_dispensed=launch_ndc,
            txn_date=svc_date,
            transaction_type=txn_type,
            fill_number=fill_num,
            days_supply=days,
            patient_pay=pat_pay,
            reject_code=rej_code,
        )


def _emit_launch_chain(
    rng: random.Random,
    patient: dict,
    prescriber_npi: str,
    launch_ndc: str,
    patient_end: date,
    attempt: dict,
    emit_fn: object,
) -> None:
    """Emit a realistic launch product therapy chain for non-teaching patients."""
    if not callable(emit_fn):
        return

    first_date = date.fromisoformat(attempt.get("first_submission_date", "2024-07-01"))
    # Map old scenario vocabulary to new PAID/PENDED/REVERSED
    _status_map = {"Paid": "PAID", "Rejected": "PENDED", "Reversed": "REVERSED"}
    first_type = _status_map.get(attempt.get("first_status", "Paid"), "PAID")
    days_supply = 28
    base_copay = round(rng.uniform(20, 80), 2)

    emit_fn(
        patient=patient,
        prescriber_npi=prescriber_npi,
        ndc_prescribed=launch_ndc,
        ndc_dispensed=launch_ndc,
        txn_date=first_date,
        transaction_type=first_type,
        fill_number=0,
        days_supply=days_supply,
        patient_pay=0.0 if first_type == "PENDED" else base_copay,
        reject_code="70" if first_type == "PENDED" else "",
    )

    first_paid_str = attempt.get("first_paid_date", "")
    if first_type == "PENDED":
        if first_paid_str:
            first_paid = date.fromisoformat(first_paid_str)
        else:
            first_paid = min(patient_end, first_date + timedelta(days=rng.randint(3, 9)))
        emit_fn(
            patient=patient,
            prescriber_npi=prescriber_npi,
            ndc_prescribed=launch_ndc,
            ndc_dispensed=launch_ndc,
            txn_date=first_paid,
            transaction_type="PAID",
            fill_number=0,
            days_supply=days_supply,
            patient_pay=base_copay,
        )
    else:
        first_paid = first_date

    if attempt.get("resolved_status") not in ("Paid",):
        return

    refill_date = first_paid + timedelta(days=days_supply + rng.randint(-3, 8))
    refill_count = rng.randint(1, 4)
    for fill_idx in range(refill_count):
        if refill_date > patient_end:
            break
        fill_num = fill_idx + 1
        refill_copay = round(base_copay + rng.uniform(-5, 25), 2)
        ndc_p, ndc_d = apply_ndc_variation(rng, launch_ndc)
        emit_fn(
            patient=patient,
            prescriber_npi=prescriber_npi,
            ndc_prescribed=ndc_p,
            ndc_dispensed=ndc_d,
            txn_date=refill_date,
            transaction_type="PAID",
            fill_number=fill_num,
            days_supply=days_supply,
            patient_pay=refill_copay,
        )
        refill_date = refill_date + timedelta(days=days_supply + rng.randint(-3, 10))


def _emit_background_fills(
    rng: random.Random,
    patient: dict,
    prescriber_npi: str,
    competitor_ndcs: list[str],
    patient_start: date,
    patient_end: date,
    emit_fn: object,
) -> None:
    """Emit competitor or no-treatment fills for background patients."""
    if not callable(emit_fn) or not competitor_ndcs:
        return
    intended_fills = rng.randint(0, 4)
    if intended_fills == 0:
        return

    ndc_prescribed = rng.choice(competitor_ndcs)
    days_supply = rng.choice([28, 30, 30, 30, 60, 90])
    fill_date = rand_date(rng, patient_start, patient_end)
    fill_num = 0

    for _ in range(intended_fills):
        if fill_date > patient_end:
            break
        if fill_num > 0 and rng.random() < 0.15:
            ndc_prescribed = rng.choice(competitor_ndcs)
            fill_num = 0
        copay = round(rng.uniform(5, 120), 2)
        ndc_p, ndc_d = apply_ndc_variation(rng, ndc_prescribed)
        chain = rng.random()
        if chain < 0.14:
            emit_fn(
                patient=patient,
                prescriber_npi=prescriber_npi,
                ndc_prescribed=ndc_p,
                ndc_dispensed=ndc_p,
                txn_date=fill_date,
                transaction_type="PENDED",
                fill_number=fill_num,
                days_supply=days_supply,
                patient_pay=0.0,
                reject_code="70",
            )
            if rng.random() < 0.75:
                emit_fn(
                    patient=patient,
                    prescriber_npi=prescriber_npi,
                    ndc_prescribed=ndc_p,
                    ndc_dispensed=ndc_d,
                    txn_date=fill_date + timedelta(days=rng.randint(2, 8)),
                    transaction_type="PAID",
                    fill_number=fill_num,
                    days_supply=days_supply,
                    patient_pay=copay,
                )
            else:
                break
        elif chain < 0.18:
            emit_fn(
                patient=patient,
                prescriber_npi=prescriber_npi,
                ndc_prescribed=ndc_p,
                ndc_dispensed=ndc_d,
                txn_date=fill_date,
                transaction_type="PAID",
                fill_number=fill_num,
                days_supply=days_supply,
                patient_pay=copay,
            )
            emit_fn(
                patient=patient,
                prescriber_npi=prescriber_npi,
                ndc_prescribed=ndc_p,
                ndc_dispensed=ndc_d,
                txn_date=fill_date + timedelta(days=rng.randint(0, 3)),
                transaction_type="REVERSED",
                fill_number=fill_num,
                days_supply=days_supply,
                patient_pay=-copay,
            )
            if rng.random() < 0.5:
                emit_fn(
                    patient=patient,
                    prescriber_npi=prescriber_npi,
                    ndc_prescribed=ndc_p,
                    ndc_dispensed=ndc_d,
                    txn_date=fill_date + timedelta(days=rng.randint(3, 10)),
                    transaction_type="PAID",
                    fill_number=fill_num,
                    days_supply=days_supply,
                    patient_pay=copay,
                )
        else:
            emit_fn(
                patient=patient,
                prescriber_npi=prescriber_npi,
                ndc_prescribed=ndc_p,
                ndc_dispensed=ndc_d,
                txn_date=fill_date,
                transaction_type="PAID",
                fill_number=fill_num,
                days_supply=days_supply,
                patient_pay=copay,
            )
        fill_num += 1
        fill_date = fill_date + timedelta(days=days_supply + rng.randint(-3, 10))
