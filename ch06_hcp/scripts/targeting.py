"""Transparent HCP and account targeting for Chapter 6."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


ANALYSIS_DATE = pd.Timestamp("2024-12-31")
RECENT_DAYS = 90
MIN_ACCOUNT_PATIENTS = 10
MIN_OPPORTUNITY_PATIENTS = 5
MIN_ACCESS_SIGNAL_PATIENTS = 2
SATURATION_CONTACTS_PER_HCP = 2.0
LAUNCH_CONDITION_CODES = {"E11.9", "E11.65", "E11.40"}
DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]


def attribute_index_hcp(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each patient to the rendering HCP on the diagnosis index date."""

    qualifying = medical_claims.loc[
        medical_claims[DX_COLS].isin(LAUNCH_CONDITION_CODES).any(axis=1)
    ].merge(
        journeys[["patient_id", "index_date"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    on_index = qualifying.loc[qualifying["claim_date"].eq(qualifying["index_date"])]
    return (
        on_index.sort_values(["patient_id", "encounter_id"])
        .drop_duplicates("patient_id")
        [["patient_id", "rendering_npi"]]
        .rename(columns={"rendering_npi": "npi"})
        .reset_index(drop=True)
    )


def summarize_hcp_engagement(
    crm: pd.DataFrame,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
    recent_days: int = RECENT_DAYS,
) -> pd.DataFrame:
    """Create HCP-level contact, recency, outcome, and consent features."""

    history = crm.loc[crm["interaction_date"].le(analysis_date)].copy()
    history["recent_contact"] = (
        analysis_date - history["interaction_date"]
    ).dt.days.between(0, recent_days)
    history["productive_contact"] = history["call_outcome"].isin(
        ["Positive", "Follow-up"]
    )
    summary = history.groupby(["hcp_npi", "account_id"], as_index=False).agg(
        lifetime_contacts=("interaction_id", "nunique"),
        recent_contacts=("recent_contact", "sum"),
        productive_contacts=("productive_contact", "sum"),
        last_contact_date=("interaction_date", "max"),
    )
    latest = (
        history.sort_values("interaction_date")
        .groupby(["hcp_npi", "account_id"], as_index=False)
        .tail(1)[["hcp_npi", "account_id", "consent_status"]]
    )
    return summary.merge(latest, on=["hcp_npi", "account_id"], how="left")


def build_hcp_features(
    inputs: Mapping[str, pd.DataFrame],
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one evidence row per target-roster HCP."""

    journeys = inputs["journeys"].copy()
    attribution = attribute_index_hcp(journeys, inputs["medical_claims"])
    patient_hcp = journeys.merge(
        attribution, on="patient_id", how="left", validate="one_to_one"
    ).merge(
        inputs["hcp_roster"][["npi", "account_id"]],
        on="npi",
        how="inner",
        validate="many_to_one",
    )
    patient_hcp["roventra_start"] = patient_hcp["first_product"].eq("Roventra")
    patient_hcp["competitor_start"] = (
        patient_hcp["initiated_treatment"] & ~patient_hcp["roventra_start"]
    )
    patient_hcp["untreated"] = ~patient_hcp["initiated_treatment"]
    patient_hcp["access_signal"] = (
        patient_hcp["untreated"] & patient_hcp["pended_transactions"].gt(0)
    )

    evidence = patient_hcp.groupby(["npi", "account_id"], as_index=False).agg(
        cohort_patients=("patient_id", "nunique"),
        treated_patients=("initiated_treatment", "sum"),
        roventra_starts=("roventra_start", "sum"),
        competitor_starts=("competitor_start", "sum"),
        untreated_patients=("untreated", "sum"),
        access_signal_patients=("access_signal", "sum"),
    )
    engagement = summarize_hcp_engagement(inputs["crm"], analysis_date)
    providers = inputs["providers"][
        ["npi", "specialty_1", "provider_state", "credential"]
    ].drop_duplicates("npi")
    hcp = (
        evidence.merge(providers, on="npi", how="left", validate="many_to_one")
        .merge(
            engagement,
            left_on=["npi", "account_id"],
            right_on=["hcp_npi", "account_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            inputs["accounts"][["account_id", "account_name", "territory"]],
            on="account_id",
            how="left",
            validate="many_to_one",
        )
    )
    numeric = [
        "lifetime_contacts",
        "recent_contacts",
        "productive_contacts",
    ]
    hcp[numeric] = hcp[numeric].fillna(0).astype(int)
    hcp["contact_permitted"] = hcp["consent_status"].eq("Allowed")
    hcp["opportunity_patients"] = (
        hcp["competitor_starts"] + hcp["untreated_patients"]
    )
    hcp["roventra_share"] = np.where(
        hcp["treated_patients"].gt(0),
        hcp["roventra_starts"] / hcp["treated_patients"],
        np.nan,
    )
    hcp["days_since_contact"] = (
        analysis_date - hcp["last_contact_date"]
    ).dt.days
    return hcp.drop(columns=["hcp_npi"]), patient_hcp


def build_decile_summary(hcp_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign equal-sized HCP volume deciles and summarize their evidence."""

    hcp = hcp_features.copy()
    hcp["volume_decile"] = pd.qcut(
        hcp["cohort_patients"].rank(method="first"),
        q=10,
        labels=range(1, 11),
    ).astype(int)
    summary = hcp.groupby("volume_decile", as_index=False).agg(
        hcps=("npi", "nunique"),
        cohort_patients=("cohort_patients", "sum"),
        opportunity_patients=("opportunity_patients", "sum"),
        roventra_starts=("roventra_starts", "sum"),
        treated_patients=("treated_patients", "sum"),
        permitted_hcps=("contact_permitted", "sum"),
        recent_contacts=("recent_contacts", "sum"),
    )
    summary["roventra_share"] = (
        summary["roventra_starts"] / summary["treated_patients"]
    )
    summary["contactable_share"] = summary["permitted_hcps"] / summary["hcps"]
    return hcp, summary


def build_account_features(
    hcp_features: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate HCP evidence and attach account ownership and capacity."""

    account = hcp_features.groupby("account_id", as_index=False).agg(
        cohort_patients=("cohort_patients", "sum"),
        treated_patients=("treated_patients", "sum"),
        roventra_starts=("roventra_starts", "sum"),
        competitor_starts=("competitor_starts", "sum"),
        untreated_patients=("untreated_patients", "sum"),
        access_signal_patients=("access_signal_patients", "sum"),
        target_hcps=("npi", "nunique"),
        contactable_hcps=("contact_permitted", "sum"),
        recent_contacts=("recent_contacts", "sum"),
    )
    account = account.merge(accounts, on="account_id", how="left", validate="one_to_one")
    account["opportunity_patients"] = (
        account["competitor_starts"] + account["untreated_patients"]
    )
    account["roventra_share"] = np.where(
        account["treated_patients"].gt(0),
        account["roventra_starts"] / account["treated_patients"],
        np.nan,
    )
    account["recent_contacts_per_hcp"] = (
        account["recent_contacts"] / account["target_hcps"]
    )
    return account


def apply_account_policy(
    account_features: pd.DataFrame,
    *,
    min_account_patients: int = MIN_ACCOUNT_PATIENTS,
    min_opportunity_patients: int = MIN_OPPORTUNITY_PATIENTS,
    min_access_signal_patients: int = MIN_ACCESS_SIGNAL_PATIENTS,
    saturation_contacts_per_hcp: float = SATURATION_CONTACTS_PER_HCP,
) -> pd.DataFrame:
    """Apply gates first, then assign a reviewable account action."""

    result = account_features.copy()
    treated = result["treated_patients"].sum()
    launch_benchmark = result["roventra_starts"].sum() / treated
    result["evidence_gate"] = result["cohort_patients"].ge(min_account_patients)
    result["opportunity_gate"] = result["opportunity_patients"].ge(
        min_opportunity_patients
    )
    result["access_review"] = result["access_signal_patients"].ge(
        min_access_signal_patients
    )
    result["permission_gate"] = result["contactable_hcps"].gt(0)
    result["ownership_gate"] = result["territory"].notna()
    result["capacity_gate"] = result["capacity"].gt(0)
    result["recently_saturated"] = result["recent_contacts_per_hcp"].ge(
        saturation_contacts_per_hcp
    )
    result["field_eligible"] = (
        result["evidence_gate"]
        & result["opportunity_gate"]
        & ~result["access_review"]
        & result["permission_gate"]
        & result["ownership_gate"]
        & result["capacity_gate"]
    )
    sparse = ~result["evidence_gate"] | ~result["opportunity_gate"]
    below_benchmark = result["roventra_share"].lt(launch_benchmark)
    result["account_action"] = np.select(
        [
            sparse,
            result["access_review"],
            ~result["permission_gate"],
            result["field_eligible"] & result["recently_saturated"],
            result["field_eligible"] & below_benchmark,
            result["field_eligible"],
        ],
        [
            "Monitor",
            "Access review",
            "Hold contact",
            "Maintain",
            "Increase priority",
            "Maintain",
        ],
        default="Monitor",
    )
    result["action_reason"] = np.select(
        [
            sparse,
            result["access_review"],
            ~result["permission_gate"],
            result["field_eligible"] & result["recently_saturated"],
            result["field_eligible"] & below_benchmark,
            result["field_eligible"],
        ],
        [
            "Evidence or opportunity is below the review threshold",
            "At least 2 untreated patients have unresolved transaction signals",
            "No HCP at the account has current contact permission",
            "Opportunity exists, but recent contact is already high",
            "Contactable opportunity and Roventra share below the cohort benchmark",
            "Contactable opportunity with Roventra share at or above benchmark",
        ],
        default="No current action passed the policy gates",
    )
    result["launch_share_benchmark"] = launch_benchmark
    order = {
        "Increase priority": 1,
        "Maintain": 2,
        "Access review": 3,
        "Hold contact": 4,
        "Monitor": 5,
    }
    result["action_order"] = result["account_action"].map(order)
    return result.sort_values(
        ["action_order", "opportunity_patients", "cohort_patients"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_hcp_actions(
    hcp_features: pd.DataFrame,
    account_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Apply account context and HCP permission to the HCP action."""

    context = account_targets[
        ["account_id", "account_action", "action_reason", "launch_share_benchmark"]
    ]
    result = hcp_features.merge(context, on="account_id", how="left", validate="many_to_one")
    result["hcp_action"] = np.select(
        [
            ~result["contact_permitted"],
            result["account_action"].eq("Access review"),
            result["account_action"].eq("Increase priority")
            & result["recent_contacts"].lt(2),
            result["account_action"].eq("Maintain")
            & result["recent_contacts"].lt(1),
        ],
        ["Hold contact", "Access follow-up", "Prioritize", "Maintain"],
        default="Monitor",
    )
    return result.sort_values(
        ["hcp_action", "opportunity_patients", "cohort_patients"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_call_plan(
    account_targets: pd.DataFrame,
    hcp_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Translate account actions into a bounded HCP call plan."""

    accounts = account_targets.copy()
    accounts["suggested_calls"] = np.select(
        [
            accounts["account_action"].eq("Increase priority"),
            accounts["account_action"].eq("Maintain")
            & accounts["recent_contacts_per_hcp"].lt(1),
        ],
        [
            np.minimum(
                accounts["capacity"],
                np.maximum(1, np.ceil(accounts["opportunity_patients"] / 8)),
            ),
            1,
        ],
        default=0,
    ).astype(int)
    planned = []
    for account in accounts.loc[accounts["suggested_calls"].gt(0)].itertuples():
        eligible_action = (
            "Prioritize" if account.account_action == "Increase priority" else "Maintain"
        )
        candidates = hcp_targets.loc[
            hcp_targets["account_id"].eq(account.account_id)
            & hcp_targets["hcp_action"].eq(eligible_action)
        ].sort_values(
            ["opportunity_patients", "recent_contacts", "cohort_patients"],
            ascending=[False, True, False],
        )
        if candidates.empty:
            continue
        allocation = [0] * len(candidates)
        for call in range(account.suggested_calls):
            allocation[call % len(candidates)] += 1
        for row, calls in zip(candidates.itertuples(), allocation, strict=True):
            if calls:
                planned.append(
                    {
                        "territory": account.territory,
                        "account_id": account.account_id,
                        "account_name": account.account_name,
                        "npi": row.npi,
                        "specialty": row.specialty_1,
                        "account_action": account.account_action,
                        "hcp_action": row.hcp_action,
                        "recommended_calls": calls,
                        "hcp_opportunity_patients": row.opportunity_patients,
                        "recent_contacts": row.recent_contacts,
                        "reason": account.action_reason,
                    }
                )
    return pd.DataFrame(planned).sort_values(
        ["territory", "account_action", "hcp_opportunity_patients"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_gate_summary(account_targets: pd.DataFrame) -> pd.DataFrame:
    """Count accounts remaining after each sequential targeting gate."""

    rows = [{"stage": "Target-roster accounts", "accounts": len(account_targets)}]
    mask = pd.Series(True, index=account_targets.index)
    for label, condition in [
        ("Evidence floor", account_targets["evidence_gate"]),
        ("Opportunity floor", account_targets["opportunity_gate"]),
        ("No access-review flag", ~account_targets["access_review"]),
        ("Contact permission", account_targets["permission_gate"]),
        ("Ownership and capacity", account_targets["ownership_gate"] & account_targets["capacity_gate"]),
    ]:
        mask &= condition
        rows.append({"stage": label, "accounts": int(mask.sum())})
    return pd.DataFrame(rows)


def build_territory_summary(
    account_targets: pd.DataFrame,
    call_plan: pd.DataFrame,
) -> pd.DataFrame:
    """Compare actionable opportunity and planned calls by territory."""

    territory = account_targets.groupby("territory", as_index=False).agg(
        accounts=("account_id", "nunique"),
        priority_accounts=("account_action", lambda s: int(s.eq("Increase priority").sum())),
        opportunity_patients=("opportunity_patients", "sum"),
        actionable_opportunity=(
            "opportunity_patients",
            lambda s: int(s[account_targets.loc[s.index, "field_eligible"]].sum()),
        ),
    )
    calls = call_plan.groupby("territory", as_index=False).agg(
        planned_hcps=("npi", "nunique"),
        recommended_calls=("recommended_calls", "sum"),
    )
    territory = territory.merge(calls, on="territory", how="left")
    territory[["planned_hcps", "recommended_calls"]] = territory[
        ["planned_hcps", "recommended_calls"]
    ].fillna(0).astype(int)
    opportunity_total = territory["actionable_opportunity"].sum()
    call_total = territory["recommended_calls"].sum()
    territory["opportunity_share"] = np.where(
        opportunity_total > 0,
        territory["actionable_opportunity"] / opportunity_total,
        0,
    )
    territory["call_share"] = np.where(
        call_total > 0, territory["recommended_calls"] / call_total, 0
    )
    territory["allocation_gap"] = territory["call_share"] - territory["opportunity_share"]
    return territory.sort_values("actionable_opportunity", ascending=False).reset_index(drop=True)


def compare_naive_and_gated(
    hcp_targets: pd.DataFrame,
    call_plan: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Compare a volume-only list with the gated HCP call plan."""

    naive = hcp_targets.nlargest(top_n, "cohort_patients")
    gated_npis = set(
        hcp_targets.loc[hcp_targets["npi"].isin(call_plan["npi"])]
        .nlargest(top_n, "opportunity_patients")["npi"]
    )
    rows = [
        {
            "plan": "Top 30 by patient volume",
            "selected_hcps": len(naive),
            "contact_permitted": int(naive["contact_permitted"].sum()),
            "opted_out": int((~naive["contact_permitted"]).sum()),
            "opportunity_patients": int(naive["opportunity_patients"].sum()),
            "recent_contacts": int(naive["recent_contacts"].sum()),
        },
        {
            "plan": "Gated near-term plan",
            "selected_hcps": len(gated_npis),
            "contact_permitted": int(
                hcp_targets.loc[hcp_targets["npi"].isin(gated_npis), "contact_permitted"].sum()
            ),
            "opted_out": 0,
            "opportunity_patients": int(
                hcp_targets.loc[
                    hcp_targets["npi"].isin(gated_npis), "opportunity_patients"
                ].sum()
            ),
            "recent_contacts": int(
                hcp_targets.loc[hcp_targets["npi"].isin(gated_npis), "recent_contacts"].sum()
            ),
        },
    ]
    return pd.DataFrame(rows)
