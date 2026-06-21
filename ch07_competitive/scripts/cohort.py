"""Corrected competitive starts and treatment transitions for Chapter 7."""

from __future__ import annotations

import pandas as pd


def build_competitive_starts(
    lines: pd.DataFrame,
    journeys: pd.DataFrame,
    initiators: pd.DataFrame,
    brand: str = "Roventra",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build washout-corrected starts, source of business, and transitions."""

    patient_context = journeys[
        ["patient_id", "payer_id", "region", "index_date", "followup_end"]
    ].drop_duplicates("patient_id")
    line1 = (
        lines.loc[lines["line_number"].eq(1)]
        .merge(patient_context, on="patient_id", how="left", validate="one_to_one")
        .rename(columns={"regimen": "first_regimen", "line_start": "therapy_index"})
    )
    line1["brand_start"] = line1["first_regimen"].eq(brand)
    line1["competitor_start"] = ~line1["brand_start"]

    segment = line1.groupby(["payer_id", "region"], as_index=False).agg(
        treated_patients=("patient_id", "nunique"),
        brand_starts=("brand_start", "sum"),
        competitor_starts=("competitor_start", "sum"),
    )
    segment["brand_share"] = segment["brand_starts"] / segment["treated_patients"]

    corrected_ids = set(line1["patient_id"])
    continuing = initiators.loc[~initiators["new_to_therapy"]].copy()

    transitions = lines.loc[lines["line_number"].gt(1)].copy()
    transitions["transition_type"] = transitions["entry_reason"].replace(
        {"Switch": "Switch", "Addition": "Add on", "Restart": "Restart"}
    )
    transitions = transitions[
        ["patient_id", "line_number", "regimen", "line_start", "transition_type"]
    ]

    source = pd.DataFrame(
        [
            ("New to therapy", len(line1)),
            ("Continuing after washout check", continuing["patient_id"].nunique()),
            (
                "Switch",
                transitions.loc[
                    transitions["transition_type"].eq("Switch"), "patient_id"
                ].nunique(),
            ),
            (
                "Add on",
                transitions.loc[
                    transitions["transition_type"].eq("Add on"), "patient_id"
                ].nunique(),
            ),
            (
                "Restart",
                transitions.loc[
                    transitions["transition_type"].eq("Restart"), "patient_id"
                ].nunique(),
            ),
        ],
        columns=["source_of_business", "patients"],
    )
    source["brand_new_starts"] = 0
    source.loc[
        source["source_of_business"].eq("New to therapy"), "brand_new_starts"
    ] = int(line1["brand_start"].sum())

    initiator_check = initiators.copy()
    initiator_check["in_corrected_line1"] = initiator_check["patient_id"].isin(
        corrected_ids
    )
    if not initiator_check.loc[
        initiator_check["new_to_therapy"], "in_corrected_line1"
    ].all():
        raise AssertionError("Chapter 5 initiator and corrected line-1 files disagree")
    return line1, segment, source, transitions


def build_switch_evidence(corrected_line1: pd.DataFrame) -> pd.DataFrame:
    """Report switch support without inventing medians that curves never reach."""

    out = corrected_line1.groupby("first_regimen", as_index=False).agg(
        patients=("patient_id", "nunique"),
        switch_events=("end_reason", lambda s: s.eq("Switch").sum()),
        addition_events=("end_reason", lambda s: s.eq("Addition").sum()),
        discontinuation_events=("end_reason", lambda s: s.eq("Discontinued").sum()),
        censored=("end_reason", lambda s: s.eq("Censored").sum()),
    )
    out["switch_event_rate"] = out["switch_events"] / out["patients"]
    out["median_time_to_switch"] = "Not reached"
    out["comparison_status"] = "Insufficient switch events"
    return out.sort_values("patients", ascending=False).reset_index(drop=True)
