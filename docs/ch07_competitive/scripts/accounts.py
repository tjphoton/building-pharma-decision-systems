"""Translate payer evidence into Chapter 6 account actions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from ch07_competitive.scripts.decomposition import wilson_interval


def build_account_actions(
    patient_hcp: pd.DataFrame,
    account_targets: pd.DataFrame,
    corrected_line1: pd.DataFrame,
    brand_policy: pd.DataFrame,
    patient_friction: pd.DataFrame,
    *,
    brand: str,
    benchmark: float,
    min_patients: int,
    min_treated: int,
    posterior_threshold: float,
    restricted_threshold: float,
    friction_threshold: float,
    analysis_date: str,
    rule_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build account evidence from Chapter 6 attribution and payer-specific rows."""

    line = corrected_line1[["patient_id", "first_regimen"]].copy()
    line["brand_start"] = line["first_regimen"].eq(brand)
    pat = patient_hcp[
        ["patient_id", "payer_id", "region", "account_id"]
    ].drop_duplicates("patient_id")
    pat = pat.merge(line, on="patient_id", how="left")
    pat["treated"] = pat["first_regimen"].notna()
    pat["brand_start"] = pat["brand_start"].fillna(False)
    policy_cols = [
        "payer_id",
        "region",
        "unrestricted",
        "access_state",
        "material_access_barrier",
    ]
    pat = pat.merge(
        brand_policy[policy_cols],
        on=["payer_id", "region"],
        how="left",
        validate="many_to_one",
    )
    pat = pat.merge(
        patient_friction[["patient_id", "submitted_attempts", "unresolved_attempts"]],
        on="patient_id",
        how="left",
    )
    pat[["submitted_attempts", "unresolved_attempts"]] = pat[
        ["submitted_attempts", "unresolved_attempts"]
    ].fillna(0)
    pat["restricted_patient"] = pat["material_access_barrier"].fillna(False)

    payer_queue = pat.groupby(
        ["account_id", "payer_id", "region", "access_state"], as_index=False
    ).agg(
        attributed_patients=("patient_id", "nunique"),
        treated_patients=("treated", "sum"),
        brand_starts=("brand_start", "sum"),
        restricted_patients=("restricted_patient", "sum"),
        submitted_attempts=("submitted_attempts", "sum"),
        unresolved_attempts=("unresolved_attempts", "sum"),
    )
    account = pat.groupby("account_id", as_index=False).agg(
        attributed_patients=("patient_id", "nunique"),
        treated_patients=("treated", "sum"),
        brand_starts=("brand_start", "sum"),
        restricted_patients=("restricted_patient", "sum"),
        submitted_attempts=("submitted_attempts", "sum"),
        unresolved_attempts=("unresolved_attempts", "sum"),
        payer_count=("payer_id", "nunique"),
    )
    account["competitor_starts"] = account["treated_patients"] - account["brand_starts"]
    for column in [
        "attributed_patients",
        "treated_patients",
        "brand_starts",
        "competitor_starts",
        "restricted_patients",
        "submitted_attempts",
        "unresolved_attempts",
    ]:
        account[column] = pd.to_numeric(account[column], errors="coerce").astype(float)
    account["brand_share"] = account["brand_starts"] / account[
        "treated_patients"
    ].replace(0, np.nan)
    account["share_lower_95"], account["share_upper_95"] = wilson_interval(
        account["brand_starts"], account["treated_patients"]
    )
    prior_strength = 30.0
    alpha0 = benchmark * prior_strength
    beta0 = (1 - benchmark) * prior_strength
    account["probability_below_benchmark"] = beta.cdf(
        benchmark,
        account["brand_starts"] + alpha0,
        account["competitor_starts"] + beta0,
    )
    account["restricted_patient_rate"] = (
        account["restricted_patients"] / account["attributed_patients"]
    )
    account["unresolved_rate"] = (
        account["unresolved_attempts"]
        / account["submitted_attempts"].replace(0, np.nan)
    ).fillna(0)
    account["evidence_sufficient"] = account["attributed_patients"].ge(
        min_patients
    ) & account["treated_patients"].ge(min_treated)
    account["access_flag"] = account["restricted_patient_rate"].ge(
        restricted_threshold
    ) | account["unresolved_rate"].ge(friction_threshold)
    account["adoption_flag"] = account["evidence_sufficient"] & account[
        "probability_below_benchmark"
    ].ge(posterior_threshold)
    account["action"] = np.select(
        [
            ~account["evidence_sufficient"],
            account["access_flag"] & account["adoption_flag"],
            account["access_flag"] & ~account["adoption_flag"],
            ~account["access_flag"] & account["adoption_flag"],
        ],
        ["Monitor", "Dual workstream", "Access review", "Adoption review"],
        default="Sustain",
    )
    account["reason_code"] = np.select(
        [
            ~account["evidence_sufficient"],
            account["access_flag"] & account["adoption_flag"],
            account["access_flag"] & ~account["adoption_flag"],
            ~account["access_flag"] & account["adoption_flag"],
        ],
        [
            "MONITOR_ACCOUNT_EVIDENCE",
            "DUAL_ACCOUNT_ACCESS_AND_ADOPTION",
            "ACCOUNT_ACCESS_BARRIER",
            "ACCOUNT_ADOPTION_GAP",
        ],
        default="ACCOUNT_DEFEND_SUPPORTED_ADOPTION",
    )
    metadata = account_targets[
        [
            "account_id",
            "account_name",
            "account_type",
            "state",
            "region",
            "territory",
            "allowed_hcps",
        ]
    ]
    account = account.merge(
        metadata, on="account_id", how="left", validate="one_to_one"
    )
    account["analysis_date"] = analysis_date
    account["decision_rule_version"] = rule_version
    return (
        account.sort_values(
            ["action", "restricted_patients", "attributed_patients"],
            ascending=[True, False, False],
        ).reset_index(drop=True),
        payer_queue.sort_values(
            ["account_id", "restricted_patients"], ascending=[True, False]
        ).reset_index(drop=True),
    )
