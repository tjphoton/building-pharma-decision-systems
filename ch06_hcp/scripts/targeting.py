"""Transparent HCP and account targeting for Chapter 6."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product

import numpy as np
import pandas as pd


ANALYSIS_DATE = pd.Timestamp("2024-12-31")
CYCLE_START = pd.Timestamp("2025-01-01")
CYCLE_END = pd.Timestamp("2025-01-28")
RECENT_DAYS = 90
ATTRIBUTION_WINDOW_DAYS = 180
MIN_ACCOUNT_PATIENTS = 10
MIN_OPPORTUNITY_PATIENTS = 5
MIN_TREATED_PATIENTS = 8
MIN_ACCESS_SIGNAL_PATIENTS = 2
ADOPTION_THRESHOLD = 0.65
SATURATION_CONTACTS_PER_HCP = 2.0
MAX_CALLS_PER_HCP = 2
TERRITORY_CALL_CAPACITY = 25
MATURE_FOLLOWUP_DAYS = 60
LAUNCH_CONDITION_CODES = {"E11.9", "E11.65", "E11.40"}
RELEVANT_SPECIALTIES = {"Primary Care", "Endocrinology", "Cardiology"}
DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]


def build_target_universe(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the eligible HCP-site universe before ranking or modeling."""

    affiliations = inputs["hcp_account_affiliations"].copy()
    affiliations["npi"] = affiliations["npi"].astype(str)
    affiliations["effective_start"] = pd.to_datetime(affiliations["effective_start"])
    affiliations["effective_end"] = pd.to_datetime(
        affiliations["effective_end"], errors="coerce"
    )
    affiliations = affiliations.rename(columns={"site_account_id": "account_id"})
    providers = inputs["providers"][["npi", "credential"]].copy()
    providers["npi"] = providers["npi"].astype(str)
    universe = (
        affiliations.merge(providers, on="npi", how="left", validate="one_to_one")
        .merge(
            inputs["accounts"][["account_id", "account_name", "account_type", "capacity"]],
            on="account_id",
            how="left",
            validate="many_to_one",
        )
    )
    universe["specialty_eligible"] = universe["specialty_1"].isin(
        RELEVANT_SPECIALTIES
    )
    universe["geography_eligible"] = universe["territory"].notna()
    universe["affiliation_active"] = universe["effective_start"].le(ANALYSIS_DATE) & (
        universe["effective_end"].isna()
        | universe["effective_end"].ge(ANALYSIS_DATE)
    )
    universe["target_eligible"] = universe[
        ["specialty_eligible", "geography_eligible", "affiliation_active"]
    ].all(axis=1)
    universe["eligibility_reason"] = np.select(
        [
            ~universe["specialty_eligible"],
            ~universe["geography_eligible"],
            ~universe["affiliation_active"],
        ],
        [
            "Specialty outside T2D targeting scope",
            "No assigned territory",
            "HCP-account affiliation inactive",
        ],
        default="Eligible",
    )
    return universe.sort_values(["target_eligible", "npi"], ascending=[False, True])


def _qualifying_claims(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
) -> pd.DataFrame:
    if {"event_id", "event_date", "npi", "condition_code"} <= set(
        medical_claims.columns
    ):
        events = medical_claims.loc[
            medical_claims["event_date"].le(ANALYSIS_DATE)
            & medical_claims["condition_code"].eq("E11")
        ].rename(
            columns={
                "event_id": "encounter_id",
                "event_date": "claim_date",
                "npi": "rendering_npi",
            }
        )
        return events.merge(
            journeys[["patient_id", "index_date"]],
            on="patient_id",
            how="inner",
            validate="many_to_one",
        )
    claims = medical_claims.copy()
    claims["rendering_npi"] = claims["rendering_npi"].astype(str)
    claims = claims.loc[
        claims["claim_date"].le(ANALYSIS_DATE)
        & claims[DX_COLS].isin(LAUNCH_CONDITION_CODES).any(axis=1)
    ]
    return claims.merge(
        journeys[["patient_id", "index_date"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )


def attribute_index_hcp(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the rendering HCP on the diagnosis index date."""

    qualifying = _qualifying_claims(journeys, medical_claims)
    return (
        qualifying.loc[qualifying["claim_date"].eq(qualifying["index_date"])]
        .sort_values(["patient_id", "encounter_id", "rendering_npi"])
        .drop_duplicates("patient_id")
        [["patient_id", "rendering_npi"]]
        .rename(columns={"rendering_npi": "npi"})
        .reset_index(drop=True)
    )


def attribute_plurality_hcp(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
    *,
    window_days: int = ATTRIBUTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Assign the most frequent relevant HCP around diagnosis."""

    qualifying = _qualifying_claims(journeys, medical_claims)
    qualifying["days_from_index"] = (
        qualifying["claim_date"] - qualifying["index_date"]
    ).dt.days
    qualifying = qualifying.loc[
        qualifying["days_from_index"].between(-window_days, window_days)
    ]
    counts = qualifying.groupby(["patient_id", "rendering_npi"], as_index=False).agg(
        relevant_encounters=("encounter_id", "nunique"),
        latest_encounter=("claim_date", "max"),
    )
    return (
        counts.sort_values(
            ["patient_id", "relevant_encounters", "latest_encounter", "rendering_npi"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("patient_id")
        [["patient_id", "rendering_npi"]]
        .rename(columns={"rendering_npi": "npi"})
        .reset_index(drop=True)
    )


def attribute_latest_hcp(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the latest relevant treating HCP before the analysis date."""

    qualifying = _qualifying_claims(journeys, medical_claims)
    return (
        qualifying.sort_values(
            ["patient_id", "claim_date", "encounter_id", "rendering_npi"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("patient_id")
        [["patient_id", "rendering_npi"]]
        .rename(columns={"rendering_npi": "npi"})
        .reset_index(drop=True)
    )


def compare_attribution_rules(
    journeys: pd.DataFrame,
    medical_claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare patient ownership under 3 defensible attribution rules."""

    comparison = journeys[["patient_id"]].copy()
    rules = {
        "index_npi": attribute_index_hcp(journeys, medical_claims),
        "plurality_npi": attribute_plurality_hcp(journeys, medical_claims),
        "latest_npi": attribute_latest_hcp(journeys, medical_claims),
    }
    for column, values in rules.items():
        comparison = comparison.merge(
            values.rename(columns={"npi": column}),
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
    comparison["all_rules_agree"] = comparison[
        ["index_npi", "plurality_npi", "latest_npi"]
    ].nunique(axis=1, dropna=True).eq(1)
    summary = pd.DataFrame(
        [
            {
                "comparison": "Index vs plurality",
                "patients_with_both": comparison[["index_npi", "plurality_npi"]]
                .notna()
                .all(axis=1)
                .sum(),
                "same_hcp": comparison["index_npi"].eq(comparison["plurality_npi"]).sum(),
            },
            {
                "comparison": "Index vs latest",
                "patients_with_both": comparison[["index_npi", "latest_npi"]]
                .notna()
                .all(axis=1)
                .sum(),
                "same_hcp": comparison["index_npi"].eq(comparison["latest_npi"]).sum(),
            },
            {
                "comparison": "All 3 rules",
                "patients_with_both": comparison[
                    ["index_npi", "plurality_npi", "latest_npi"]
                ]
                .notna()
                .all(axis=1)
                .sum(),
                "same_hcp": comparison["all_rules_agree"].sum(),
            },
        ]
    )
    summary["agreement_rate"] = (
        summary["same_hcp"] / summary["patients_with_both"].clip(lower=1)
    )
    return comparison, summary


def summarize_hcp_engagement(
    crm: pd.DataFrame,
    *,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
) -> pd.DataFrame:
    """Create dated field-contact and 3-state permission evidence."""

    history = crm.loc[crm["interaction_date"].le(analysis_date)].copy()
    history["recent_contact"] = (
        analysis_date - history["interaction_date"]
    ).dt.days.between(0, RECENT_DAYS)
    history["productive_contact"] = history["call_outcome"].isin(
        ["Positive", "Follow-up"]
    )
    summary = history.groupby(["hcp_npi", "account_id"], as_index=False).agg(
        lifetime_contacts=("interaction_id", "nunique"),
        recent_contacts=("recent_contact", "sum"),
        productive_contacts=("productive_contact", "sum"),
        last_contact_date=("interaction_date", "max"),
    )
    return summary


def build_hcp_features(
    inputs: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one dated evidence row per eligible target HCP."""

    journeys = inputs["journeys"].copy()
    attribution_source = inputs.get("attribution_events", inputs["medical_claims"])
    attribution = attribute_index_hcp(journeys, attribution_source)
    universe = build_target_universe(inputs).loc[lambda frame: frame["target_eligible"]]
    patient_hcp = (
        journeys.merge(attribution, on="patient_id", how="left", validate="one_to_one")
        .merge(
            inputs["current_treatment_state"],
            on="patient_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_state"),
        )
        .merge(
            universe[
                [
                    "npi",
                    "account_id",
                    "parent_account_id",
                    "territory",
                    "specialty_1",
                ]
            ],
            on="npi",
            how="inner",
            validate="many_to_one",
        )
    )
    patient_hcp["roventra_start"] = patient_hcp["first_product"].eq("Roventra")
    patient_hcp["roventra_current"] = patient_hcp["current_treatment_state"].eq(
        "Roventra treated"
    )
    patient_hcp["competitor_treated"] = (
        patient_hcp["current_treatment_state"].eq("Competitor treated")
    )
    patient_hcp["untreated_mature"] = (
        patient_hcp["current_treatment_state"].eq("Untreated")
        & patient_hcp["maturity_status"].eq("Mature")
    )
    patient_hcp["immature_or_unknown"] = (
        patient_hcp["maturity_status"].ne("Mature")
    )
    patient_hcp["access_signal"] = (
        patient_hcp["untreated_mature"] & patient_hcp["unresolved_access_signal"]
    )
    evidence = patient_hcp.groupby(
        ["npi", "account_id", "parent_account_id"], as_index=False
    ).agg(
        cohort_patients=("patient_id", "nunique"),
        treated_patients=(
            "current_treatment_state",
            lambda values: int(values.isin(["Roventra treated", "Competitor treated"]).sum()),
        ),
        roventra_starts=("roventra_current", "sum"),
        competitor_treated=("competitor_treated", "sum"),
        untreated_mature=("untreated_mature", "sum"),
        immature_or_unknown=("immature_or_unknown", "sum"),
        access_signal_patients=("access_signal", "sum"),
    )
    engagement = summarize_hcp_engagement(inputs["crm"])
    permissions = inputs["contact_permissions"].rename(
        columns={
            "status": "contact_permission_status",
            "channel": "permission_channel",
            "purpose": "permission_purpose",
            "effective_start": "permission_effective_date",
        }
    )
    hcp = (
        evidence.merge(
            universe[
                [
                    "npi",
                    "account_id",
                    "account_name",
                    "territory",
                    "specialty_1",
                    "state",
                    "region",
                    "credential",
                ]
            ],
            on=["npi", "account_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            engagement,
            left_on=["npi", "account_id"],
            right_on=["hcp_npi", "account_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            permissions[
                [
                    "npi",
                    "account_id",
                    "contact_permission_status",
                    "permission_channel",
                    "permission_purpose",
                    "permission_effective_date",
                    "source",
                ]
            ],
            on=["npi", "account_id"],
            how="left",
            validate="one_to_one",
        )
    )
    for column in ["lifetime_contacts", "recent_contacts", "productive_contacts"]:
        hcp[column] = hcp[column].fillna(0).astype(int)
    hcp["contact_permission_status"] = hcp["contact_permission_status"].fillna("Unknown")
    hcp["permission_channel"] = hcp["permission_channel"].fillna("Field")
    hcp["permission_purpose"] = hcp["permission_purpose"].fillna("Promotional")
    hcp["contact_permitted"] = hcp["contact_permission_status"].eq("Allowed")
    hcp["review_opportunity"] = hcp["competitor_treated"] + hcp["untreated_mature"]
    hcp["roventra_share"] = np.where(
        hcp["treated_patients"].gt(0),
        hcp["roventra_starts"] / hcp["treated_patients"],
        np.nan,
    )
    hcp["days_since_contact"] = (
        ANALYSIS_DATE - hcp["last_contact_date"]
    ).dt.days
    hcp["analysis_date"] = ANALYSIS_DATE.date().isoformat()
    hcp["data_cutoff"] = ANALYSIS_DATE.date().isoformat()
    hcp["rule_version"] = "targeting-v2.0"
    return hcp.drop(columns=["hcp_npi"]), patient_hcp


def build_coverage_summary(
    journeys: pd.DataFrame,
    patient_hcp: pd.DataFrame,
) -> pd.DataFrame:
    """Show selective coverage by region and payer."""

    covered = journeys[["patient_id", "region", "payer_id"]].assign(
        covered=lambda frame: frame["patient_id"].isin(patient_hcp["patient_id"])
    )
    return (
        covered.groupby(["region", "payer_id"], as_index=False)
        .agg(journey_patients=("patient_id", "nunique"), covered_patients=("covered", "sum"))
        .assign(coverage_rate=lambda frame: frame["covered_patients"] / frame["journey_patients"])
    )


def build_decile_summary(hcp_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative opportunity capture for contactable HCPs ranked by opportunity."""

    contactable = hcp_features.loc[
        hcp_features["contact_permitted"]
    ].sort_values(
        ["review_opportunity", "npi"], ascending=[False, True]
    ).reset_index(drop=True)
    contactable["opportunity_decile"] = pd.qcut(
        contactable.index.to_series(),
        q=10,
        labels=range(1, 11),
    ).astype(int)
    summary = contactable.groupby("opportunity_decile", as_index=False).agg(
        hcps=("npi", "nunique"),
        cohort_patients=("cohort_patients", "sum"),
        review_opportunity=("review_opportunity", "sum"),
    )
    summary["cumulative_hcp_share"] = summary["hcps"].cumsum() / summary["hcps"].sum()
    summary["cumulative_opportunity_share"] = (
        summary["review_opportunity"].cumsum() / summary["review_opportunity"].sum()
    )
    return contactable, summary


def build_account_features(
    hcp_features: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate evidence to site accounts while preserving parent ownership."""

    account = hcp_features.groupby(
        ["account_id", "parent_account_id"], as_index=False
    ).agg(
        cohort_patients=("cohort_patients", "sum"),
        treated_patients=("treated_patients", "sum"),
        roventra_starts=("roventra_starts", "sum"),
        competitor_treated=("competitor_treated", "sum"),
        untreated_mature=("untreated_mature", "sum"),
        immature_or_unknown=("immature_or_unknown", "sum"),
        access_signal_patients=("access_signal_patients", "sum"),
        target_hcps=("npi", "nunique"),
        allowed_hcps=("contact_permission_status", lambda values: int(values.eq("Allowed").sum())),
        opted_out_hcps=("contact_permission_status", lambda values: int(values.eq("Opt-out").sum())),
        unknown_permission_hcps=(
            "contact_permission_status",
            lambda values: int(values.eq("Unknown").sum()),
        ),
        recent_contacts=("recent_contacts", "sum"),
    )
    account = account.merge(accounts, on="account_id", how="left", validate="one_to_one")
    account["review_opportunity"] = account["competitor_treated"] + account["untreated_mature"]
    account["roventra_share"] = np.where(
        account["treated_patients"].gt(0),
        account["roventra_starts"] / account["treated_patients"],
        np.nan,
    )
    account["recent_contacts_per_hcp"] = account["recent_contacts"] / account["target_hcps"]
    return account


def apply_account_policy(
    account_features: pd.DataFrame,
    *,
    min_account_patients: int = MIN_ACCOUNT_PATIENTS,
    min_opportunity_patients: int = MIN_OPPORTUNITY_PATIENTS,
    min_treated_patients: int = MIN_TREATED_PATIENTS,
    adoption_threshold: float = ADOPTION_THRESHOLD,
) -> pd.DataFrame:
    """Apply prespecified evidence and operational gates in visible order."""

    result = account_features.copy()
    result["evidence_gate"] = result["cohort_patients"].ge(min_account_patients)
    result["treated_denominator_gate"] = result["treated_patients"].ge(min_treated_patients)
    result["opportunity_gate"] = result["review_opportunity"].ge(min_opportunity_patients)
    result["access_review"] = result["access_signal_patients"].ge(MIN_ACCESS_SIGNAL_PATIENTS)
    result["permission_gate"] = result["allowed_hcps"].gt(0)
    result["ownership_gate"] = result["territory"].notna()
    result["capacity_gate"] = result["capacity"].gt(0)
    result["recently_saturated"] = result["recent_contacts_per_hcp"].ge(
        SATURATION_CONTACTS_PER_HCP
    )
    result["field_eligible"] = (
        result["evidence_gate"]
        & result["treated_denominator_gate"]
        & result["opportunity_gate"]
        & ~result["access_review"]
        & result["permission_gate"]
        & result["ownership_gate"]
        & result["capacity_gate"]
    )
    sparse = ~result["evidence_gate"] | ~result["opportunity_gate"]
    uncertain_share = ~result["treated_denominator_gate"]
    below_threshold = result["roventra_share"].lt(adoption_threshold)
    result["account_action"] = np.select(
        [
            sparse,
            result["access_review"],
            ~result["permission_gate"],
            uncertain_share,
            result["field_eligible"] & result["recently_saturated"],
            result["field_eligible"] & below_threshold,
            result["field_eligible"],
        ],
        [
            "Monitor",
            "Access review",
            "Hold contact",
            "Monitor",
            "Maintain",
            "Increase priority",
            "Maintain",
        ],
        default="Monitor",
    )
    result["reason_code"] = np.select(
        [
            sparse,
            result["access_review"],
            ~result["permission_gate"],
            uncertain_share,
            result["field_eligible"] & result["recently_saturated"],
            result["field_eligible"] & below_threshold,
            result["field_eligible"],
        ],
        [
            "MONITOR_LOW_EVIDENCE",
            "ROUTE_ACCESS_REVIEW",
            "HOLD_NO_PERMISSION",
            "MONITOR_SMALL_TREATED_DENOMINATOR",
            "MAINTAIN_SATURATED",
            "PRIORITIZE_REVIEW_OPPORTUNITY",
            "MAINTAIN_ESTABLISHED",
        ],
        default="MONITOR_NO_RULE",
    )
    reason_text = {
        "MONITOR_LOW_EVIDENCE": "Evidence or review opportunity is below the threshold",
        "ROUTE_ACCESS_REVIEW": "Repeated unresolved access evidence requires separate review",
        "HOLD_NO_PERMISSION": "No HCP has current field contact permission",
        "MONITOR_SMALL_TREATED_DENOMINATOR": "Treated denominator is too small for an adoption action",
        "MAINTAIN_SATURATED": "Review opportunity exists, but recent field contact is high",
        "PRIORITIZE_REVIEW_OPPORTUNITY": "Permitted review opportunity and adoption below the scenario threshold",
        "MAINTAIN_ESTABLISHED": "Permitted evidence with adoption at or above the scenario threshold",
        "MONITOR_NO_RULE": "No current action passed the policy gates",
    }
    result["action_reason"] = result["reason_code"].map(reason_text)
    result["adoption_threshold"] = adoption_threshold
    result["analysis_date"] = ANALYSIS_DATE.date().isoformat()
    result["cycle_start"] = CYCLE_START.date().isoformat()
    result["cycle_end"] = CYCLE_END.date().isoformat()
    action_order = {
        "Increase priority": 1,
        "Maintain": 2,
        "Access review": 3,
        "Hold contact": 4,
        "Monitor": 5,
    }
    result["action_order"] = result["account_action"].map(action_order)
    return result.sort_values(
        ["action_order", "review_opportunity", "cohort_patients", "account_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_hcp_actions(
    hcp_features: pd.DataFrame,
    account_targets: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """Apply account context and use segments only for eligible engagement form."""

    context = account_targets[
        ["account_id", "account_action", "action_reason", "reason_code", "adoption_threshold"]
    ]
    segment_columns = [
        "npi",
        "segment_id",
        "segment_name",
        "assignment_stability",
        "distance_to_centroid",
        "segment_model_version",
        "engagement_pattern",
    ]
    result = (
        hcp_features.merge(context, on="account_id", how="left", validate="many_to_one")
        .merge(segments[segment_columns], on="npi", how="left", validate="one_to_one")
    )
    result["segment_name"] = result["segment_name"].fillna("Not clustered")
    result["segment_model_version"] = result["segment_model_version"].fillna(
        "Not applicable"
    )
    result["hcp_action"] = np.select(
        [
            ~result["contact_permission_status"].eq("Allowed"),
            result["account_action"].eq("Access review"),
            result["account_action"].eq("Increase priority") & result["recent_contacts"].lt(2),
            result["account_action"].eq("Maintain") & result["recent_contacts"].lt(1),
        ],
        ["Hold contact", "Access follow-up", "Prioritize", "Maintain"],
        default="Monitor",
    )
    result["engagement_pattern"] = result["engagement_pattern"].fillna("Field review")
    result.loc[
        ~result["hcp_action"].isin(["Prioritize", "Maintain"]),
        "engagement_pattern",
    ] = "No promotional sequence"
    return result.sort_values(
        ["hcp_action", "review_opportunity", "cohort_patients", "npi"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_call_plan(
    account_targets: pd.DataFrame,
    hcp_targets: pd.DataFrame,
    territory_capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Translate account actions into a 4-week, capacity-bounded field plan."""

    planned = []
    for account in account_targets.itertuples():
        if account.account_action == "Increase priority":
            suggested = min(account.capacity, max(1, int(np.ceil(account.review_opportunity / 8))))
            eligible_action = "Prioritize"
        elif account.account_action == "Maintain" and account.recent_contacts_per_hcp < 1:
            suggested = 1
            eligible_action = "Maintain"
        else:
            continue
        candidates = hcp_targets.loc[
            hcp_targets["account_id"].eq(account.account_id)
            & hcp_targets["hcp_action"].eq(eligible_action)
        ].sort_values(
            ["review_opportunity", "recent_contacts", "cohort_patients", "npi"],
            ascending=[False, True, False, True],
        )
        if candidates.empty:
            continue
        allocation = {str(npi): 0 for npi in candidates["npi"]}
        for _ in range(int(suggested)):
            available = [npi for npi, calls in allocation.items() if calls < MAX_CALLS_PER_HCP]
            if not available:
                break
            chosen = min(available, key=lambda npi: (allocation[npi], available.index(npi)))
            allocation[chosen] += 1
        for row in candidates.itertuples():
            calls = allocation[str(row.npi)]
            if calls == 0:
                continue
            planned.append(
                {
                    "cycle_start": CYCLE_START.date().isoformat(),
                    "cycle_end": CYCLE_END.date().isoformat(),
                    "territory": account.territory,
                    "account_id": account.account_id,
                    "parent_account_id": account.parent_account_id,
                    "account_name": account.account_name,
                    "npi": row.npi,
                    "specialty": row.specialty_1,
                    "account_action": account.account_action,
                    "hcp_action": row.hcp_action,
                    "engagement_pattern": row.engagement_pattern,
                    "segment_name": row.segment_name,
                    "recommended_calls": calls,
                    "hcp_review_opportunity": row.review_opportunity,
                    "recent_contacts": row.recent_contacts,
                    "permission_status": row.contact_permission_status,
                    "reason_code": account.reason_code,
                    "reason": account.action_reason,
                }
            )
    if not planned:
        return pd.DataFrame()
    plan = pd.DataFrame(planned).sort_values(
        ["territory", "account_action", "hcp_review_opportunity", "npi"],
        ascending=[True, True, False, True],
    )
    capacity = territory_capacity.set_index("territory")["available_calls"]
    plan["territory_cycle_capacity"] = plan["territory"].map(capacity)
    plan["territory_running_calls"] = plan.groupby("territory")["recommended_calls"].cumsum()
    plan = plan.loc[
        plan["territory_running_calls"].le(plan["territory_cycle_capacity"])
    ]
    return plan.drop(columns="territory_running_calls").reset_index(drop=True)


def build_override_template(call_plan: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "account_id",
        "npi",
        "original_action",
        "override_action",
        "override_reason",
        "approver",
        "approval_date",
        "expiration_date",
    ]
    template = call_plan[["account_id", "npi", "hcp_action"]].rename(
        columns={"hcp_action": "original_action"}
    )
    for column in columns[3:]:
        template[column] = pd.NA
    return template[columns]


def build_gate_summary(account_targets: pd.DataFrame) -> pd.DataFrame:
    rows = [{"stage": "Eligible site accounts", "accounts": len(account_targets)}]
    mask = pd.Series(True, index=account_targets.index)
    for label, condition in [
        ("Evidence floor", account_targets["evidence_gate"]),
        ("Treated denominator", account_targets["treated_denominator_gate"]),
        ("Opportunity floor", account_targets["opportunity_gate"]),
        ("No access-review route", ~account_targets["access_review"]),
        ("Field permission", account_targets["permission_gate"]),
        ("Ownership and capacity", account_targets["ownership_gate"] & account_targets["capacity_gate"]),
    ]:
        mask &= condition
        rows.append({"stage": label, "accounts": int(mask.sum())})
    return pd.DataFrame(rows)


def build_coverage_funnel(
    journeys: pd.DataFrame,
    attribution: pd.DataFrame,
    patient_hcp: pd.DataFrame,
    hcp_targets: pd.DataFrame,
    call_plan: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage": "Journey cohort", "count": journeys["patient_id"].nunique()},
            {"stage": "Attributed patients", "count": attribution["patient_id"].nunique()},
            {"stage": "Eligible-roster patients", "count": patient_hcp["patient_id"].nunique()},
            {"stage": "Permitted HCPs", "count": hcp_targets.loc[hcp_targets["contact_permitted"], "npi"].nunique()},
            {"stage": "Planned HCPs", "count": call_plan["npi"].nunique()},
        ]
    )


def build_policy_sensitivity(account_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evidence, opportunity, threshold in product(
        (8, 10, 12),
        (4, 8, 12),
        (0.45, 0.60, 0.75),
    ):
        targets = apply_account_policy(
            account_features,
            min_account_patients=evidence,
            min_opportunity_patients=opportunity,
            adoption_threshold=threshold,
        )
        rows.append(
            {
                "minimum_account_patients": evidence,
                "minimum_opportunity_patients": opportunity,
                "adoption_threshold": threshold,
                "priority_accounts": int(targets["account_action"].eq("Increase priority").sum()),
                "changed_from_default": pd.NA,
            }
        )
    result = pd.DataFrame(rows)
    default_actions = apply_account_policy(account_features).set_index("account_id")["account_action"]
    changes = []
    for row in result.itertuples():
        actions = apply_account_policy(
            account_features,
            min_account_patients=row.minimum_account_patients,
            min_opportunity_patients=row.minimum_opportunity_patients,
            adoption_threshold=row.adoption_threshold,
        ).set_index("account_id")["account_action"]
        changes.append(int(actions.ne(default_actions).sum()))
    result["changed_from_default"] = changes
    return result


def build_territory_summary(
    account_targets: pd.DataFrame,
    call_plan: pd.DataFrame,
    territory_capacity: pd.DataFrame,
) -> pd.DataFrame:
    territory = account_targets.groupby("territory", as_index=False).agg(
        accounts=("account_id", "nunique"),
        priority_accounts=("account_action", lambda values: int(values.eq("Increase priority").sum())),
        review_opportunity=("review_opportunity", "sum"),
        actionable_opportunity=(
            "review_opportunity",
            lambda values: int(values[account_targets.loc[values.index, "field_eligible"]].sum()),
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
    territory = territory.merge(
        territory_capacity[["territory", "available_calls"]],
        on="territory",
        how="left",
        validate="one_to_one",
    )
    territory["cycle_capacity"] = territory["available_calls"]
    territory["unused_capacity"] = territory["cycle_capacity"] - territory["recommended_calls"]
    opportunity_total = territory["actionable_opportunity"].sum()
    call_total = territory["recommended_calls"].sum()
    territory["opportunity_share"] = territory["actionable_opportunity"] / max(opportunity_total, 1)
    territory["call_share"] = territory["recommended_calls"] / max(call_total, 1)
    territory["allocation_gap"] = territory["call_share"] - territory["opportunity_share"]
    return territory.sort_values("actionable_opportunity", ascending=False).reset_index(drop=True)


def compare_naive_and_gated(
    hcp_targets: pd.DataFrame,
    call_plan: pd.DataFrame,
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    naive = hcp_targets.nlargest(top_n, "cohort_patients")
    gated = hcp_targets.loc[hcp_targets["npi"].isin(call_plan["npi"])]
    return pd.DataFrame(
        [
            {
                "plan": "Top 30 by patient volume",
                "selected_hcps": len(naive),
                "contact_permitted": int(naive["contact_permitted"].sum()),
                "held_or_unknown": int((~naive["contact_permitted"]).sum()),
                "review_opportunity": int(naive["review_opportunity"].sum()),
                "recent_contacts": int(naive["recent_contacts"].sum()),
            },
            {
                "plan": "Gated 4-week field plan",
                "selected_hcps": gated["npi"].nunique(),
                "contact_permitted": int(gated["contact_permitted"].sum()),
                "held_or_unknown": 0,
                "review_opportunity": int(gated["review_opportunity"].sum()),
                "recent_contacts": int(gated["recent_contacts"].sum()),
            },
        ]
    )
