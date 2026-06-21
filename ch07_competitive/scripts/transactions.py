"""Prescription-attempt construction and access-friction measures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_attempts(
    pharmacy: pd.DataFrame,
    ndc_codes: pd.DataFrame,
    patient_context: pd.DataFrame,
    hub: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse pharmacy transactions into final prescription attempts."""

    rx = pharmacy.copy()
    ndc_map = ndc_codes.set_index("ndc")["drug_name"]
    rx["product_name"] = rx["ndc_prescribed"].map(ndc_map)
    rx = rx.loc[rx["product_name"].notna()].copy()
    rx = rx.sort_values(["patient_id", "date_of_service", "claim_id"])
    keys = ["patient_id", "payer_id", "prescriber_npi", "ndc_prescribed", "fill_number"]
    attempts = (
        rx.groupby(keys, dropna=False, sort=False)
        .agg(
            product_name=("product_name", "first"),
            first_submission_date=("date_of_service", "min"),
            last_transaction_date=("date_of_service", "max"),
            final_transaction=("transaction_type", "last"),
            transaction_rows=("claim_id", "size"),
            had_pend=("transaction_type", lambda s: s.eq("PENDED").any()),
            had_reversal=("transaction_type", lambda s: s.eq("REVERSED").any()),
            first_reject_code=("reject_code", "first"),
            first_paid_date=(
                "date_of_service",
                lambda s: (
                    s.loc[rx.loc[s.index, "transaction_type"].eq("PAID")].min()
                    if rx.loc[s.index, "transaction_type"].eq("PAID").any()
                    else pd.NaT
                ),
            ),
        )
        .reset_index()
    )
    attempts["final_outcome"] = np.select(
        [
            attempts["final_transaction"].eq("PAID"),
            attempts["final_transaction"].eq("PENDED"),
            attempts["final_transaction"].eq("REVERSED"),
        ],
        ["Completed", "Unresolved", "Reversed"],
        default="Other",
    )
    attempts["days_to_paid"] = (
        pd.to_datetime(attempts["first_paid_date"])
        - pd.to_datetime(attempts["first_submission_date"])
    ).dt.days
    attempts = attempts.merge(
        patient_context[["patient_id", "region"]].drop_duplicates("patient_id"),
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    hub_summary = hub.groupby("patient_id", as_index=False).agg(
        hub_status=("hub_status", "last"),
        dispense_status=("dispense_status", "last"),
        discontinue_reason=("discontinue_reason", "last"),
    )
    return attempts.merge(hub_summary, on="patient_id", how="left")


def friction_summary(
    attempts: pd.DataFrame,
    brand: str = "Roventra",
) -> pd.DataFrame:
    """Aggregate attempt outcomes by payer-region using attempts as denominator."""

    b = attempts.loc[attempts["product_name"].eq(brand)].copy()
    out = b.groupby(["payer_id", "region"], as_index=False).agg(
        submitted_attempts=("patient_id", "size"),
        patients_with_attempt=("patient_id", "nunique"),
        completed_attempts=("final_outcome", lambda s: s.eq("Completed").sum()),
        unresolved_attempts=("final_outcome", lambda s: s.eq("Unresolved").sum()),
        reversed_attempts=("final_outcome", lambda s: s.eq("Reversed").sum()),
        attempts_with_pend=("had_pend", "sum"),
        median_days_to_paid=("days_to_paid", "median"),
    )
    out["completion_rate"] = out["completed_attempts"] / out["submitted_attempts"]
    out["unresolved_rate"] = out["unresolved_attempts"] / out["submitted_attempts"]
    out["pend_exposure_rate"] = out["attempts_with_pend"] / out["submitted_attempts"]
    return out


def patient_friction(attempts: pd.DataFrame, brand: str = "Roventra") -> pd.DataFrame:
    """Create patient-level attempt evidence for account aggregation."""

    b = attempts.loc[attempts["product_name"].eq(brand)].copy()
    return b.groupby(["patient_id", "payer_id", "region"], as_index=False).agg(
        submitted_attempts=("patient_id", "size"),
        unresolved_attempts=("final_outcome", lambda s: s.eq("Unresolved").sum()),
        attempts_with_pend=("had_pend", "sum"),
        median_days_to_paid=("days_to_paid", "median"),
    )
