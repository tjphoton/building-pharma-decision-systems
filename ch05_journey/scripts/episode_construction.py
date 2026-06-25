"""Core episode construction functions for Chapter 5 treatment patterns."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STUDY_END = pd.Timestamp("2024-12-31")

# Wide diagnosis column names on medical claims
DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]

# ICD-10 codes for the launch condition (T2D)
LAUNCH_CONDITION_CODES = ["E11.9", "E11.65", "E11.40"]


def load_chapter3_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the longitudinal source tables used in Chapter 5."""
    ndc_codes = pd.read_csv(data_dir / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    ndc_map = ndc_codes.set_index("ndc")["drug_name"]

    pharmacy = pd.read_csv(
        data_dir / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
        parse_dates=["date_of_service"],
    )
    # Derive drug name by joining on ndc_prescribed for stable product attribution
    pharmacy["product_name"] = pharmacy["ndc_prescribed"].map(ndc_map)

    # Build coverage windows from enrollment table
    enrollments = pd.read_csv(
        data_dir / "reference" / "patient_enrollments.csv",
        parse_dates=["eligibility_start_date", "eligibility_end_date"],
    )
    coverage = (
        enrollments
        .groupby("patient_id")
        .agg(
            coverage_start=("eligibility_start_date", "min"),
            coverage_end=("eligibility_end_date", "max"),
        )
        .reset_index()
    )
    patients = pd.read_csv(data_dir / "reference" / "patients.csv")
    # add payer_id from enrollments (patients.csv holds demographics only)
    payer_map = enrollments[["patient_id", "payer_id"]].drop_duplicates("patient_id")
    patients = patients.merge(payer_map, on="patient_id", how="left")
    patients = patients.merge(coverage, on="patient_id", how="left")

    return {
        "patients": patients,
        "products": pd.read_csv(data_dir / "reference" / "products.csv"),
        "providers": pd.read_csv(data_dir / "reference" / "providers.csv"),
        "hcp_targets": pd.read_csv(data_dir / "reference" / "hcp_targets.csv"),
        "accounts": pd.read_csv(data_dir / "reference" / "accounts.csv"),
        "ndc_codes": ndc_codes,
        "medical_claims": pd.read_csv(
            data_dir / "claims_medical" / "medical_claims_mature.csv",
            parse_dates=["claim_date"],
        ),
        "pharmacy_claims": pharmacy,
        "specialty_pharmacy": pd.read_csv(
            data_dir / "specialty_pharmacy" / "sp_events.csv",
            parse_dates=["referral_date", "status_date", "ship_date"],
        ),
    }


def build_newly_observed_cohort(
    tables: Mapping[str, pd.DataFrame],
    icd_code_prefixes: list[str] | None = None,
    minimum_lookback_days: int = 90,
    minimum_followup_days: int = 90,
    study_end: pd.Timestamp = DEFAULT_STUDY_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a diagnosis-indexed cohort and return its attrition table.

    Parameters
    ----------
    icd_code_prefixes:
        ICD-10 prefixes to use for identifying qualifying encounters. Defaults
        to the T2D launch-condition codes from the Chapter 3 generator.
    """
    if icd_code_prefixes is None:
        icd_code_prefixes = list(LAUNCH_CONDITION_CODES)

    patients = tables["patients"].copy()
    medical = tables["medical_claims"].copy()

    # All rows in medical_claims_mature.csv are completed encounters; no status filter needed.
    # Check any of the ten wide diagnosis columns for qualifying ICD codes.
    dx_mask = medical[DX_COLS].isin(set(icd_code_prefixes)).any(axis=1)
    qualifying_diagnoses = medical.loc[dx_mask]

    index_dates = (
        qualifying_diagnoses.groupby("patient_id", as_index=False)["claim_date"]
        .min()
        .rename(columns={"claim_date": "index_date"})
    )

    indexed = patients.merge(index_dates, on="patient_id", how="inner")
    indexed["followup_end"] = indexed["coverage_end"].clip(upper=study_end)
    indexed["lookback_days"] = (indexed["index_date"] - indexed["coverage_start"]).dt.days
    indexed["followup_days"] = (indexed["followup_end"] - indexed["index_date"]).dt.days

    lookback_mask = indexed["lookback_days"].ge(minimum_lookback_days)
    followup_mask = indexed["followup_days"].ge(minimum_followup_days)
    cohort = indexed.loc[lookback_mask & followup_mask].copy()
    cohort["minimum_lookback_days"] = minimum_lookback_days
    cohort["minimum_followup_days"] = minimum_followup_days
    cohort["study_end"] = study_end

    condition_label = f"ICD prefix {'|'.join(icd_code_prefixes[:3])}"
    attrition = pd.DataFrame(
        [
            {
                "stage": "Patients in source population",
                "patients": patients["patient_id"].nunique(),
                "rule": "One row in the patient reference table",
            },
            {
                "stage": "Observed qualifying diagnosis",
                "patients": indexed["patient_id"].nunique(),
                "rule": f"At least one encounter with {condition_label}",
            },
            {
                "stage": "Sufficient lookback",
                "patients": indexed.loc[lookback_mask, "patient_id"].nunique(),
                "rule": f"At least {minimum_lookback_days} covered days before index",
            },
            {
                "stage": "Analysis cohort",
                "patients": cohort["patient_id"].nunique(),
                "rule": (
                    f"Lookback plus at least {minimum_followup_days} observable "
                    "days after index"
                ),
            },
        ]
    )
    return cohort.reset_index(drop=True), attrition


def prepare_pharmacy_events(
    pharmacy_claims: pd.DataFrame,
    cohort: pd.DataFrame,
    products: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate treatment exposure from pended and reversed transactions."""
    product_set = set(products)
    events = pharmacy_claims.merge(
        cohort[["patient_id", "index_date", "followup_end"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    events = events.loc[
        events["product_name"].isin(product_set)
        & events["date_of_service"].between(
            events["index_date"],
            events["followup_end"],
            inclusive="both",
        )
    ].copy()
    events["days_from_index"] = (events["date_of_service"] - events["index_date"]).dt.days

    exposure = events.loc[events["transaction_type"].eq("PAID")].copy()
    nonpaid = events.loc[events["transaction_type"].isin(["PENDED", "REVERSED"])].copy()
    return (
        exposure.sort_values(["patient_id", "date_of_service", "claim_id"]).reset_index(drop=True),
        nonpaid.sort_values(["patient_id", "date_of_service", "claim_id"]).reset_index(drop=True),
    )


def construct_treatment_episodes(
    paid_events: pd.DataFrame,
    cohort: pd.DataFrame,
    permissible_gap_days: int = 30,
) -> pd.DataFrame:
    """Join paid fills of the same product into treatment episodes."""
    episode_rows: list[dict] = []
    episode_id = 1

    for patient_id, patient_events in paid_events.groupby("patient_id", sort=True):
        open_episodes: dict[str, dict] = {}
        for row in patient_events.sort_values(
            ["date_of_service", "product_name", "claim_id"]
        ).itertuples(index=False):
            fill_end = row.date_of_service + pd.Timedelta(days=max(int(row.days_supply), 1) - 1)
            current = open_episodes.get(row.product_name)
            if (
                current is not None
                and row.date_of_service
                <= current["episode_end"] + pd.Timedelta(days=permissible_gap_days)
            ):
                current["episode_end"] = max(current["episode_end"], fill_end)
                current["fill_count"] += 1
                current["paid_days"] += int(row.days_supply)
                current["last_fill_date"] = row.date_of_service
            else:
                current = {
                    "episode_id": f"EP{episode_id:06d}",
                    "patient_id": patient_id,
                    "product_name": row.product_name,
                    "episode_start": row.date_of_service,
                    "episode_end": fill_end,
                    "last_fill_date": row.date_of_service,
                    "fill_count": 1,
                    "paid_days": int(row.days_supply),
                    "permissible_gap_days": permissible_gap_days,
                }
                episode_rows.append(current)
                open_episodes[row.product_name] = current
                episode_id += 1

    columns = [
        "episode_id", "patient_id", "product_name", "episode_start", "episode_end",
        "last_fill_date", "fill_count", "paid_days", "permissible_gap_days",
    ]
    episodes = pd.DataFrame(episode_rows, columns=columns)
    if episodes.empty:
        return episodes.assign(
            line_number=pd.Series(dtype="int64"),
            transition_in=pd.Series(dtype="object"),
            transition_out=pd.Series(dtype="object"),
        )

    episodes = episodes.sort_values(
        ["patient_id", "episode_start", "episode_end", "product_name"]
    ).reset_index(drop=True)
    episodes["line_number"] = episodes.groupby("patient_id").cumcount() + 1
    episodes["transition_in"] = "Initial treatment"
    episodes["transition_out"] = "Censored on treatment"

    followup_end = cohort.set_index("patient_id")["followup_end"]
    for patient_id, positions in episodes.groupby("patient_id", sort=False).groups.items():
        ordered_positions = list(positions)
        for offset, position in enumerate(ordered_positions):
            current = episodes.loc[position]
            if offset < len(ordered_positions) - 1:
                next_position = ordered_positions[offset + 1]
                following = episodes.loc[next_position]
                if following["product_name"] == current["product_name"]:
                    transition = "Restart"
                elif following["episode_start"] <= current["episode_end"]:
                    transition = "Add-on"
                else:
                    transition = "Switch"
                episodes.loc[position, "transition_out"] = transition
                episodes.loc[next_position, "transition_in"] = transition
            else:
                observable_end = followup_end.loc[patient_id]
                discontinuation_threshold = current["episode_end"] + pd.Timedelta(
                    days=permissible_gap_days
                )
                if discontinuation_threshold < observable_end:
                    episodes.loc[position, "transition_out"] = "Discontinuation"

    episodes["episode_days"] = (episodes["episode_end"] - episodes["episode_start"]).dt.days + 1
    return episodes


def summarize_patient_journeys(
    cohort: pd.DataFrame,
    episodes: pd.DataFrame,
    nonpaid_events: pd.DataFrame,
) -> pd.DataFrame:
    """Create one decision-facing journey record per cohort patient."""
    base = cohort[
        ["patient_id", "payer_id", "region",
         "index_date", "followup_end", "followup_days"]
    ].copy()
    if episodes.empty:
        base["initiated_treatment"] = False
        base["journey_state"] = "No observed treatment"
        return base

    episode_summary = (
        episodes.groupby("patient_id", as_index=False)
        .agg(
            first_treatment_date=("episode_start", "min"),
            first_product=("product_name", "first"),
            treatment_episodes=("episode_id", "nunique"),
            products_used=("product_name", "nunique"),
            last_episode_end=("episode_end", "max"),
            final_transition=("transition_out", "last"),
        )
    )
    sequences = (
        episodes.sort_values(["patient_id", "line_number"])
        .groupby("patient_id")["product_name"]
        .agg(lambda values: " > ".join(values))
        .rename("treatment_sequence")
        .reset_index()
    )
    summary = base.merge(episode_summary, on="patient_id", how="left").merge(
        sequences, on="patient_id", how="left"
    )
    summary["initiated_treatment"] = summary["first_treatment_date"].notna()
    summary["days_to_treatment"] = (
        summary["first_treatment_date"] - summary["index_date"]
    ).dt.days

    nonpaid_summary = (
        nonpaid_events.groupby("patient_id", as_index=False)
        .agg(
            pended_transactions=("transaction_type", lambda v: int((v == "PENDED").sum())),
            reversed_transactions=("transaction_type", lambda v: int((v == "REVERSED").sum())),
        )
    )
    summary = summary.merge(nonpaid_summary, on="patient_id", how="left")
    summary[["pended_transactions", "reversed_transactions"]] = summary[
        ["pended_transactions", "reversed_transactions"]
    ].fillna(0).astype(int)
    summary["journey_state"] = np.select(
        [
            summary["final_transition"].eq("Discontinuation"),
            summary["treatment_episodes"].fillna(0).gt(1),
            summary["initiated_treatment"],
            summary["pended_transactions"].gt(0),
        ],
        [
            "Initiated, then discontinued",
            "Multiple treatment episodes",
            "Initiated and observed on treatment",
            "Pended without observed treatment",
        ],
        default="No observed treatment transaction",
    )
    return summary
