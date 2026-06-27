"""Reason-coded 4-week omnichannel policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ch08_omnichannel.generation_modules.ch08_config import (
    FIELD_CAPACITY_PER_TERRITORY,
    HIGH_PRESSURE_MIN,
    PREDICTED_RESPONSE_FLOOR,
)


def build_channel_plan(
    scored_snapshots: pd.DataFrame,
    analysis_date: pd.Timestamp,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    refresh_date: pd.Timestamp,
    policy_version: str,
) -> pd.DataFrame:
    """Apply permission, account, pressure, and response rules in order."""

    plan = scored_snapshots.loc[
        scored_snapshots["snapshot_date"].eq(analysis_date)
    ].copy()
    suppressed = (
        ~plan["contact_permitted"]
        | plan["account_action"].eq("Hold contact")
    )
    access_route = (
        plan["account_action"].eq("Access review")
        | plan["competitive_action"].isin(["Access work", "Dual workstream"])
    )
    priority = plan["account_action"].eq("Increase priority")
    high_pressure = plan["total_pressure_30"].ge(HIGH_PRESSURE_MIN)
    digital_response = (
        plan["email_clicks_90"] + plan["web_actions_90"] + plan["paid_clicks_90"]
    ).gt(0)
    peer_response = plan["peer_attendance_180"].gt(0)
    speaker_response = plan["speaker_attendance_180"].gt(0)
    live_program_response = plan["live_program_attendance_180"].gt(0)
    field_response = plan["field_responses_90"].gt(0)
    predicted_high = plan["predicted_response"].ge(PREDICTED_RESPONSE_FLOOR)

    plan["promotion_eligible"] = (
        ~suppressed
        & ~access_route
        & ~high_pressure
        & (
            priority
            | predicted_high
            | digital_response
            | field_response
            | live_program_response
        )
    )
    plan["capacity_rank"] = np.nan
    eligible = plan.loc[plan["promotion_eligible"]].copy()
    eligible["capacity_rank"] = (
        eligible.sort_values(
            ["territory", "predicted_response", "total_pressure_30"],
            ascending=[True, False, True],
        )
        .groupby("territory")
        .cumcount()
        + 1
    )
    plan.loc[eligible.index, "capacity_rank"] = eligible["capacity_rank"]
    plan["capacity_selected"] = plan["capacity_rank"].le(FIELD_CAPACITY_PER_TERRITORY)

    plan["recommended_action"] = "Observe"
    plan["reason_code"] = "OBSERVE_LIMITED_EVIDENCE"
    plan.loc[suppressed, ["recommended_action", "reason_code"]] = [
        "Suppress",
        "SUPPRESS_PERMISSION",
    ]
    plan.loc[~suppressed & access_route, ["recommended_action", "reason_code"]] = [
        "Access coordination",
        "ROUTE_ACCESS_BOUNDARY",
    ]
    plan.loc[
        ~suppressed & ~access_route & high_pressure,
        ["recommended_action", "reason_code"],
    ] = ["Observe", "PAUSE_HIGH_PRESSURE"]

    selected = plan["capacity_selected"] & ~suppressed & ~access_route & ~high_pressure
    plan.loc[selected & speaker_response, ["recommended_action", "reason_code"]] = [
        "Speaker-program invitation",
        "CAPACITY_RANKED_SPEAKER_RESPONSE",
    ]
    plan.loc[
        selected & ~speaker_response & peer_response,
        ["recommended_action", "reason_code"],
    ] = [
        "Peer-program invitation",
        "CAPACITY_RANKED_PEER_RESPONSE",
    ]
    plan.loc[
        selected & ~speaker_response & ~peer_response & digital_response,
        ["recommended_action", "reason_code"],
    ] = ["Email follow-up", "CAPACITY_RANKED_DIGITAL_RESPONSE"]
    plan.loc[
        selected & ~speaker_response & ~peer_response & ~digital_response & field_response,
        ["recommended_action", "reason_code"],
    ] = ["Field follow-up", "CAPACITY_RANKED_FIELD_RESPONSE"]
    plan.loc[
        selected
        & ~speaker_response
        & ~peer_response
        & ~digital_response
        & ~field_response,
        ["recommended_action", "reason_code"],
    ] = ["Field follow-up", "CAPACITY_RANKED_PREDICTED_RESPONSE"]
    plan.loc[
        plan["promotion_eligible"] & ~plan["capacity_selected"] & plan["reason_code"].eq("OBSERVE_LIMITED_EVIDENCE"),
        "reason_code",
    ] = "OBSERVE_BELOW_CAPACITY_CUTOFF"
    plan["recommended_channel"] = plan["recommended_action"].map(
        {
            "Suppress": "None",
            "Access coordination": "Access support",
            "Observe": "None",
            "Peer-program invitation": "Peer program",
            "Speaker-program invitation": "Speaker program",
            "Email follow-up": "Email",
            "Field follow-up": "Field",
        }
    )
    plan["recommended_topic"] = np.select(
        [
            plan["recommended_action"].eq("Access coordination"),
            plan["last_response_channel"].ne("None"),
            plan["segment_name"].str.contains("evidence", case=False),
        ],
        [
            "Coverage support",
            "Continue latest responsive topic",
            "Clinical evidence",
        ],
        default="Patient identification",
    )
    plan["timing"] = plan["recommended_action"].map(
        {
            "Suppress": "No contact",
            "Access coordination": "Week 1 review",
            "Observe": "Refresh after cycle",
            "Peer-program invitation": "Week 1 invitation",
            "Speaker-program invitation": "Week 1 invitation",
            "Email follow-up": "Week 1 email",
            "Field follow-up": "Week 2 field review",
        }
    )
    plan["maximum_cycle_frequency"] = plan["recommended_action"].map(
        {
            "Suppress": 0,
            "Access coordination": 1,
            "Observe": 0,
            "Peer-program invitation": 1,
            "Speaker-program invitation": 1,
            "Email follow-up": 2,
            "Field follow-up": 2,
        }
    ).astype(int)
    plan["evidence_note"] = (
        "Pressure "
        + plan["pressure_band"].str.lower()
        + "; last response "
        + plan["last_response_channel"]
        + "; predicted response "
        + plan["predicted_response"].map(lambda value: f"{value:.0%}")
    )
    plan["measurement_hook"] = plan["recommended_action"].map(
        {
            "Suppress": "Audit suppression compliance",
            "Access coordination": "Track access status and resolved attempts",
            "Observe": "Refresh pressure and response state",
            "Peer-program invitation": "Track invitation, attendance, and follow-up",
            "Speaker-program invitation": "Track invitation, registration, attendance, and follow-up",
            "Email follow-up": "Track delivery and click",
            "Field follow-up": "Track completed interaction and outcome",
        }
    )
    plan["analysis_date"] = analysis_date
    plan["cycle_start"] = cycle_start
    plan["cycle_end"] = cycle_end
    plan["policy_version"] = policy_version
    plan["refresh_date"] = refresh_date
    columns = [
        "analysis_date",
        "cycle_start",
        "cycle_end",
        "npi",
        "account_id",
        "territory",
        "account_action",
        "competitive_action",
        "contact_permission_status",
        "pressure_band",
        "total_pressure_30",
        "last_response_channel",
        "predicted_response",
        "promotion_eligible",
        "capacity_rank",
        "capacity_selected",
        "recommended_action",
        "recommended_channel",
        "recommended_topic",
        "timing",
        "maximum_cycle_frequency",
        "reason_code",
        "evidence_note",
        "measurement_hook",
        "policy_version",
        "refresh_date",
    ]
    return plan[columns].sort_values(
        ["recommended_action", "predicted_response"],
        ascending=[True, False],
    ).reset_index(drop=True)


def plan_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Count relationships and planned frequency by action."""

    return (
        plan.groupby("recommended_action", as_index=False)
        .agg(
            relationships=("npi", "size"),
            planned_contacts=("maximum_cycle_frequency", "sum"),
            mean_predicted_response=("predicted_response", "mean"),
        )
        .sort_values("relationships", ascending=False)
        .reset_index(drop=True)
    )
