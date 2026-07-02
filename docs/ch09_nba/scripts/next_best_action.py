"""Governed next-best-action engine for Chapter 9.

The engine reads the omnichannel channel-plan state and releases one executable
recommendation per HCP-account row. Hard gates run before any score can act.
The released row carries action, channel, content, timing, measurement,
expiration, and version metadata. Later sections add constrained ranking,
bucketed Thompson exploration, off-policy evaluation, and execution feedback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_DATE = pd.Timestamp("2025-02-28")
POLICY_VERSION = "nba_policy_2025_02_v2"
RULE_SET_VERSION = "nba_rules_2025_02_v2"
MODEL_VERSION = "omni_response_2025_02_v1"
CONTENT_CATALOG_VERSION = "content_catalog_2025_02_v1"

DEFAULT_VALIDITY_DAYS = 14
HIGH_PRESSURE_MIN = 5
EMAIL_FREQUENCY_CAP = 2
FIELD_CAPACITY_LIMIT = 3
PROGRAM_CAPACITY_LIMIT = 10
RESPONSE_VALUE = 4_000.0
EXPLORE_EPSILON = 0.10
SEED = 20260701
THOMPSON_DRAWS = 1_000_000
THOMPSON_SEED = 20260701
MINIMUM_DETECTABLE_EFFECT = 0.05
POWER_TARGET = 0.80
ALPHA = 0.05

PRECEDENCE = {
    "No action": 1,
    "Access follow-up": 10,
    "Field conversation": 20,
    "Program invitation": 25,
    "Approved email": 30,
    "Continue responsive content": 40,
    "Monitor": 80,
}

ACTION_CHANNEL = {
    "No action": "None",
    "Access follow-up": "Access team",
    "Field conversation": "Field",
    "Program invitation": "Program",
    "Approved email": "Email",
    "Continue responsive content": "Web",
    "Monitor": "None",
}

ACTION_TTL_DAYS = {
    "No action": 14,
    "Access follow-up": 7,
    "Field conversation": 21,
    "Program invitation": 10,
    "Approved email": 14,
    "Continue responsive content": 14,
    "Monitor": 30,
}

ACTION_UNIT_COST = {
    "No action": 0.0,
    "Access follow-up": 130.0,
    "Field conversation": 225.0,
    "Program invitation": 340.0,
    "Approved email": 0.25,
    "Continue responsive content": 0.12,
    "Monitor": 0.0,
}

ACTION_EXPECTED_RESULT = {
    "No action": "Maintain compliance and refresh the row",
    "Access follow-up": "Resolve or document the access barrier",
    "Field conversation": "Complete a relevant field conversation",
    "Program invitation": "Secure attendance at an approved program",
    "Approved email": "Deliver approved content and earn a click",
    "Continue responsive content": "Maintain relevant digital continuity",
    "Monitor": "Wait for a material evidence change",
}

ACTION_MEASUREMENT_HOOK = {
    "No action": "Suppression compliance; refresh state",
    "Access follow-up": "Access-state change; resolved attempts",
    "Field conversation": "Completed interaction; field outcome",
    "Program invitation": "Invitation; attendance; follow-up",
    "Approved email": "Delivery; click; qualified follow-up",
    "Continue responsive content": "Delivery; web qualified action",
    "Monitor": "Evidence refresh; recommendation change",
}

ACTION_REQUIRES_CONTENT = {
    "Field conversation",
    "Program invitation",
    "Approved email",
    "Continue responsive content",
}

ACTION_REVIEW_REQUIRED = {
    "Access follow-up",
    "Field conversation",
    "Program invitation",
}

PROMOTIONAL_ACTIONS = [
    "Field conversation",
    "Program invitation",
    "Approved email",
]

ACTION_VALUE_BONUS = {
    "Access follow-up": 0.18,
    "Field conversation": 0.11,
    "Program invitation": 0.10,
    "Approved email": 0.08,
    "Continue responsive content": 0.05,
    "Monitor": 0.0,
    "No action": 0.0,
}

UPLIFT_COVARIATES = [
    "review_opportunity",
    "evidence_need_score",
    "access_resource_score",
    "digital_response_rate",
    "field_response_rate",
    "total_pressure_30",
    "total_pressure_90",
    "shrunken_response_rate_90",
]


def _clip_prob(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(value, 0.01, 0.99)


def build_content_catalog() -> pd.DataFrame:
    """Return a compact synthetic catalog of approved and blocked content."""

    rows = [
        {
            "content_id": "CNT_EMAIL_ACCESS_01",
            "candidate_action": "Approved email",
            "content_family": "Access support",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Email",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-15"),
            "approval_expires_on": pd.Timestamp("2025-04-30"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 1,
        },
        {
            "content_id": "CNT_EMAIL_EXPIRED_02",
            "candidate_action": "Approved email",
            "content_family": "Efficacy follow-up",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Email",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2024-10-01"),
            "approval_expires_on": pd.Timestamp("2025-02-15"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 2,
        },
        {
            "content_id": "CNT_EMAIL_DRAFT_03",
            "candidate_action": "Approved email",
            "content_family": "Safety update",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Email",
            "mlr_status": "Draft",
            "approval_starts_on": pd.Timestamp("2025-02-01"),
            "approval_expires_on": pd.Timestamp("2025-05-31"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 3,
        },
        {
            "content_id": "CNT_WEB_RESP_01",
            "candidate_action": "Continue responsive content",
            "content_family": "Responsive follow-up",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Web",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-10"),
            "approval_expires_on": pd.Timestamp("2025-04-15"),
            "risk_information_required": False,
            "promotional_claim_type": "Branded",
            "content_priority": 1,
        },
        {
            "content_id": "CNT_WEB_ACCOUNT_02",
            "candidate_action": "Continue responsive content",
            "content_family": "Account support page",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "Account",
            "approved_channel": "Web",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-10"),
            "approval_expires_on": pd.Timestamp("2025-04-15"),
            "risk_information_required": False,
            "promotional_claim_type": "Operational",
            "content_priority": 2,
        },
        {
            "content_id": "CNT_FIELD_GUIDE_01",
            "candidate_action": "Field conversation",
            "content_family": "Field guide",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Field",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-01"),
            "approval_expires_on": pd.Timestamp("2025-05-31"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 1,
        },
        {
            "content_id": "CNT_FIELD_EXPIRED_02",
            "candidate_action": "Field conversation",
            "content_family": "Field guide",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Field",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2024-10-01"),
            "approval_expires_on": pd.Timestamp("2025-02-20"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 2,
        },
        {
            "content_id": "CNT_PROGRAM_INVITE_01",
            "candidate_action": "Program invitation",
            "content_family": "Program invitation",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "HCP",
            "approved_channel": "Program",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-20"),
            "approval_expires_on": pd.Timestamp("2025-04-10"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 1,
        },
        {
            "content_id": "CNT_PROGRAM_WRONG_AUD_02",
            "candidate_action": "Program invitation",
            "content_family": "Program invitation",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "Account",
            "approved_channel": "Program",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-20"),
            "approval_expires_on": pd.Timestamp("2025-04-10"),
            "risk_information_required": True,
            "promotional_claim_type": "Branded",
            "content_priority": 2,
        },
        {
            "content_id": "CNT_ACCESS_PLAYBOOK_01",
            "candidate_action": "Access follow-up",
            "content_family": "Access support",
            "product": "Roventra",
            "indication": "Relapsing neuromuscular syndrome",
            "audience": "Account",
            "approved_channel": "Access team",
            "mlr_status": "Approved",
            "approval_starts_on": pd.Timestamp("2025-01-01"),
            "approval_expires_on": pd.Timestamp("2025-12-31"),
            "risk_information_required": False,
            "promotional_claim_type": "Operational",
            "content_priority": 1,
        },
    ]
    return pd.DataFrame(rows)


def build_policy_registry() -> pd.DataFrame:
    """Return one row describing the current release policy."""

    return pd.DataFrame(
        [
            {
                "policy_version": POLICY_VERSION,
                "rule_set_version": RULE_SET_VERSION,
                "model_version": MODEL_VERSION,
                "content_catalog_version": CONTENT_CATALOG_VERSION,
                "recommendation_date": RECOMMENDATION_DATE,
                "validity_days_default": DEFAULT_VALIDITY_DAYS,
                "high_pressure_min": HIGH_PRESSURE_MIN,
                "email_frequency_cap": EMAIL_FREQUENCY_CAP,
                "field_capacity_limit": FIELD_CAPACITY_LIMIT,
                "program_capacity_limit": PROGRAM_CAPACITY_LIMIT,
                "exploration_budget": EXPLORE_EPSILON,
            }
        ]
    )


def nba_object_model() -> pd.DataFrame:
    """Describe the production objects used by the governed release layer."""

    return pd.DataFrame(
        [
            {
                "object": "state",
                "record_level": "HCP-account-date",
                "job": "Current facts available to the engine",
                "example_fields": "permission, access state, response score",
            },
            {
                "object": "candidate",
                "record_level": "HCP-account-action-content",
                "job": "Every action the policy considered",
                "example_fields": "action, channel, content ID, gate",
            },
            {
                "object": "contract",
                "record_level": "released recommendation",
                "job": "Executable row sent to downstream systems",
                "example_fields": "reason, timing, measurement, expiration",
            },
            {
                "object": "policy log",
                "record_level": "historical decision",
                "job": "Evidence for learning and replay",
                "example_fields": "logged action, probability, outcome",
            },
            {
                "object": "execution log",
                "record_level": "released recommendation",
                "job": "Last-mile adoption and override record",
                "example_fields": "status, override reason, feedback time",
            },
        ]
    )


def load_state(ch08_results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the channel-plan governance state with scored snapshot features."""

    plan = ch08_results["channel_plan"].copy()
    panel = ch08_results["scored_snapshots"].copy()
    plan["npi"] = plan["npi"].astype(str)
    panel["npi"] = panel["npi"].astype(str)
    snapshot = panel.loc[panel["snapshot_date"].eq(RECOMMENDATION_DATE)].copy()
    feature_columns = [
        "npi",
        "account_id",
        "future_response",
        "review_opportunity",
        "evidence_need_score",
        "access_resource_score",
        "digital_response_rate",
        "field_response_rate",
        "total_pressure_30",
        "total_pressure_90",
        "field_responses_90",
        "email_clicks_90",
        "web_actions_90",
        "paid_clicks_90",
        "live_program_attendance_180",
        "email_frequency_30",
        "shrunken_response_rate_90",
        "segment_name",
        "last_channel",
        "last_response_channel",
    ]
    state = plan.merge(
        snapshot[feature_columns],
        on=["npi", "account_id"],
        how="left",
        suffixes=("", "_snapshot"),
        validate="one_to_one",
    )
    if "total_pressure_30_snapshot" in state.columns:
        state["total_pressure_30"] = state["total_pressure_30_snapshot"].fillna(
            state["total_pressure_30"]
        )
    if "pressure_band_snapshot" in state.columns:
        state["pressure_band"] = state["pressure_band_snapshot"].fillna(state["pressure_band"])
    state["digital_signal"] = (
        state["email_clicks_90"] + state["web_actions_90"] + state["paid_clicks_90"]
    ).gt(0)
    state["field_signal"] = state["field_responses_90"].gt(0)
    state["live_program_signal"] = state["live_program_attendance_180"].gt(0)
    state["permitted"] = state["contact_permission_status"].eq("Allowed")
    state["priority_flag"] = state["account_action"].eq("Increase priority")
    state["high_pressure_flag"] = state["total_pressure_30"].ge(HIGH_PRESSURE_MIN)
    state = assign_context_bucket(state)
    return state


def assign_context_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign a compact context label for the exploration section."""

    work = frame.copy()
    if "digital_signal" not in work.columns:
        work["digital_signal"] = (
            work["email_clicks_90"] + work["web_actions_90"] + work["paid_clicks_90"]
        ).gt(0)
    work["context_bucket"] = np.select(
        [
            work["live_program_attendance_180"].gt(0),
            work["digital_signal"],
            work["field_responses_90"].gt(0),
            work["access_resource_score"].ge(0.65),
            work["evidence_need_score"].ge(0.65),
        ],
        [
            "Program-history",
            "Digital-responsive",
            "Field-responsive",
            "Access-need",
            "Evidence-need",
        ],
        default="Routine-monitor",
    )
    return work


def _gates(record: pd.Series) -> dict[str, bool]:
    suppressed = (not bool(record["permitted"])) or record["account_action"] == "Hold contact"
    access_route = (
        record["account_action"] == "Access review"
        or record["competitive_action"] in {"Access work", "Dual workstream"}
    )
    high_pressure = int(record["total_pressure_30"]) >= HIGH_PRESSURE_MIN
    priority = bool(record["priority_flag"])
    return {
        "suppressed": suppressed,
        "access_route": access_route,
        "high_pressure": high_pressure,
        "priority": priority,
    }


def _action_gate_outcome(
    *,
    action: str,
    suppressed: bool,
    access_route: bool,
    high_pressure: bool,
    priority: bool,
    field_capacity: bool,
    live_program: bool,
    digital: bool,
    email_at_cap: bool,
) -> tuple[bool, str, str]:
    if action == "No action":
        if suppressed:
            return True, "suppressed", "Permission or policy suppresses contact"
        return False, "not_suppressed", "Row is reachable"
    if action == "Access follow-up":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return True, "access_route", "Account evidence points to access friction"
        return False, "no_access_route", "No access route"
    if action == "Field conversation":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return False, "access_route", "Access route runs before promotion"
        if high_pressure:
            return False, "high_pressure", "Recent pressure is already high"
        if not priority:
            return False, "not_priority", "Account is not in the priority tier"
        if not field_capacity:
            return False, "capacity", "No field capacity remains"
        return True, "passed", "Priority HCP-account row with field capacity"
    if action == "Program invitation":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return False, "access_route", "Access route runs before promotion"
        if high_pressure:
            return False, "high_pressure", "Recent pressure is already high"
        if not live_program:
            return False, "program_signal", "No prior live-program signal"
        return True, "passed", "Prior live-program attendance supports a repeat invitation"
    if action == "Approved email":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return False, "access_route", "Access route runs before promotion"
        if high_pressure:
            return False, "high_pressure", "Recent pressure is already high"
        if not (priority or digital):
            return False, "no_email_signal", "No priority or digital signal"
        if email_at_cap:
            return False, "email_cap", "Email is at the cycle cap"
        return True, "passed", "Available email frequency with a qualifying signal"
    if action == "Continue responsive content":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return False, "access_route", "Access route runs before promotion"
        if not digital:
            return False, "no_digital_signal", "No recent digital response signal"
        if priority:
            return False, "priority_reroute", "Priority rows are handled by higher tiers"
        return True, "passed", "Meaningful digital response without a higher-priority action"
    if action == "Monitor":
        if suppressed:
            return False, "suppressed", "Permission or policy suppresses contact"
        if access_route:
            return False, "access_route", "Access route runs before monitoring"
        return True, "passed", "Eligible row without a stronger action signal"
    raise ValueError(f"Unknown action: {action}")


def generate_action_menu(state: pd.DataFrame) -> pd.DataFrame:
    """Build the 7-action menu per HCP-account row before content expansion."""

    rows: list[dict[str, object]] = []
    for record in state.to_dict("records"):
        gates = _gates(pd.Series(record))
        field_capacity = pd.notna(record.get("capacity_rank")) and (
            bool(record.get("capacity_selected")) or float(record.get("capacity_rank", 99)) <= FIELD_CAPACITY_LIMIT
        )
        email_at_cap = int(record["email_frequency_30"]) >= EMAIL_FREQUENCY_CAP
        for action, precedence in PRECEDENCE.items():
            passed, binding_gate, binding_reason = _action_gate_outcome(
                action=action,
                suppressed=gates["suppressed"],
                access_route=gates["access_route"],
                high_pressure=gates["high_pressure"],
                priority=gates["priority"],
                field_capacity=field_capacity,
                live_program=bool(record["live_program_signal"]),
                digital=bool(record["digital_signal"]),
                email_at_cap=email_at_cap,
            )
            rows.append(
                {
                    "npi": record["npi"],
                    "account_id": record["account_id"],
                    "territory": record["territory"],
                    "candidate_action": action,
                    "candidate_channel": ACTION_CHANNEL[action],
                    "policy_precedence": precedence,
                    "action_ttl_days": ACTION_TTL_DAYS[action],
                    "action_requires_content": action in ACTION_REQUIRES_CONTENT,
                    "eligible_before_content": bool(passed),
                    "suppressed_gate": gates["suppressed"],
                    "access_route_gate": gates["access_route"],
                    "high_pressure_gate": gates["high_pressure"],
                    "priority_gate": gates["priority"],
                    "field_capacity_available": bool(field_capacity),
                    "live_program_signal": bool(record["live_program_signal"]),
                    "digital_signal": bool(record["digital_signal"]),
                    "field_signal": bool(record["field_signal"]),
                    "email_at_cap": bool(email_at_cap),
                    "binding_gate": binding_gate,
                    "binding_reason": binding_reason,
                    "reason_code": binding_reason if passed else f"Blocked: {binding_reason}",
                    "expected_result": ACTION_EXPECTED_RESULT[action],
                    "measurement_hook": ACTION_MEASUREMENT_HOOK[action],
                    "predicted_response": float(record["predicted_response"]),
                    "review_opportunity": float(record["review_opportunity"]),
                    "evidence_need_score": float(record["evidence_need_score"]),
                    "access_resource_score": float(record["access_resource_score"]),
                    "digital_response_rate": float(record["digital_response_rate"]),
                    "field_response_rate": float(record["field_response_rate"]),
                    "total_pressure_30": int(record["total_pressure_30"]),
                    "total_pressure_90": int(record["total_pressure_90"]),
                    "shrunken_response_rate_90": float(record["shrunken_response_rate_90"]),
                    "segment_name": record["segment_name"],
                    "context_bucket": record["context_bucket"],
                    "recommended_topic": record["recommended_topic"],
                    "timing": record["timing"],
                    "last_response_channel": record["last_response_channel"],
                    "state_refresh_date": record["refresh_date"],
                }
            )
    return pd.DataFrame(rows)


def gate_summary(menu: pd.DataFrame) -> pd.DataFrame:
    """Summarize the first blocking reason on the action menu."""

    blocked = menu.loc[~menu["eligible_before_content"]]
    summary = (
        blocked.groupby("binding_reason", as_index=False)
        .agg(blocked_candidates=("candidate_action", "size"))
        .sort_values("blocked_candidates", ascending=False)
        .reset_index(drop=True)
    )
    passed = pd.DataFrame(
        [{"binding_reason": "Passed", "blocked_candidates": int(menu["eligible_before_content"].sum())}]
    )
    return pd.concat([summary, passed], ignore_index=True)


def attach_content_candidates(menu: pd.DataFrame, content_catalog: pd.DataFrame) -> pd.DataFrame:
    """Expand content-requiring actions into content-specific candidate rows."""

    rows: list[dict[str, object]] = []
    for record in menu.to_dict("records"):
        action = record["candidate_action"]
        if action not in ACTION_REQUIRES_CONTENT:
            rows.append(
                {
                    **record,
                    "content_id": pd.NA,
                    "content_family": pd.NA,
                    "product": pd.NA,
                    "indication": pd.NA,
                    "audience": pd.NA,
                    "approved_channel": pd.NA,
                    "mlr_status": "Not required",
                    "approval_starts_on": pd.NaT,
                    "approval_expires_on": pd.NaT,
                    "risk_information_required": False,
                    "promotional_claim_type": pd.NA,
                    "content_priority": 0,
                    "content_gate_passed": True,
                    "content_gate_reason": "Not required",
                    "eligible": bool(record["eligible_before_content"]),
                    "selected_content_rank": 0,
                }
            )
            continue
        matches = content_catalog.loc[content_catalog["candidate_action"].eq(action)]
        if matches.empty:
            rows.append(
                {
                    **record,
                    "content_id": pd.NA,
                    "content_family": pd.NA,
                    "product": pd.NA,
                    "indication": pd.NA,
                    "audience": pd.NA,
                    "approved_channel": pd.NA,
                    "mlr_status": pd.NA,
                    "approval_starts_on": pd.NaT,
                    "approval_expires_on": pd.NaT,
                    "risk_information_required": pd.NA,
                    "promotional_claim_type": pd.NA,
                    "content_priority": 99,
                    "content_gate_passed": False,
                    "content_gate_reason": "No matching content template",
                    "eligible": False,
                    "selected_content_rank": 99,
                }
            )
            continue
        for content in matches.to_dict("records"):
            rows.append(
                {
                    **record,
                    **content,
                    "selected_content_rank": int(content["content_priority"]),
                }
            )
    return pd.DataFrame(rows)


def apply_content_gates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply content approval, date, audience, and channel gates."""

    work = candidates.copy()
    if "content_gate_passed" not in work.columns:
        work["content_gate_passed"] = True
        work["content_gate_reason"] = "Not required"

    requires_content = work["action_requires_content"]
    work.loc[requires_content, "content_gate_passed"] = True
    work.loc[requires_content, "content_gate_reason"] = "Passed"

    no_content = requires_content & work["content_id"].isna()
    work.loc[no_content, "content_gate_passed"] = False
    work.loc[no_content, "content_gate_reason"] = "No approved content"

    wrong_status = requires_content & work["mlr_status"].ne("Approved") & ~work["content_id"].isna()
    work.loc[wrong_status, "content_gate_passed"] = False
    work.loc[wrong_status, "content_gate_reason"] = "Content not approved"

    before_start = (
        requires_content
        & work["content_gate_passed"]
        & work["approval_starts_on"].notna()
        & work["approval_starts_on"].gt(RECOMMENDATION_DATE)
    )
    work.loc[before_start, "content_gate_passed"] = False
    work.loc[before_start, "content_gate_reason"] = "Approval not active"

    expired = (
        requires_content
        & work["content_gate_passed"]
        & work["approval_expires_on"].notna()
        & work["approval_expires_on"].lt(RECOMMENDATION_DATE)
    )
    work.loc[expired, "content_gate_passed"] = False
    work.loc[expired, "content_gate_reason"] = "Content expired"

    wrong_audience = (
        requires_content
        & work["content_gate_passed"]
        & work["audience"].notna()
        & ~work["audience"].isin(["HCP", "HCP-account row"])
    )
    work.loc[wrong_audience, "content_gate_passed"] = False
    work.loc[wrong_audience, "content_gate_reason"] = "Audience mismatch"

    wrong_channel = (
        requires_content
        & work["content_gate_passed"]
        & work["approved_channel"].notna()
        & work["approved_channel"].ne(work["candidate_channel"])
    )
    work.loc[wrong_channel, "content_gate_passed"] = False
    work.loc[wrong_channel, "content_gate_reason"] = "Channel mismatch"

    work["eligible"] = work["eligible_before_content"] & work["content_gate_passed"]
    work["binding_gate"] = np.where(
        work["eligible_before_content"],
        np.where(work["content_gate_passed"], "passed", "content"),
        work["binding_gate"],
    )
    work["binding_reason"] = np.where(
        work["eligible_before_content"],
        np.where(work["content_gate_passed"], work["binding_reason"], work["content_gate_reason"]),
        work["binding_reason"],
    )
    return work


def gate_summary_by_gate(candidates: pd.DataFrame) -> pd.DataFrame:
    """Count blocked candidates by gate family on the expanded candidate table."""

    blocked = candidates.loc[~candidates["eligible"]].copy()
    summary = (
        blocked.groupby("binding_gate", as_index=False)
        .agg(
            blocked_candidates=("candidate_action", "size"),
            affected_hcp_account_rows=("npi", lambda values: values.astype(str).nunique()),
            example_action=("candidate_action", "first"),
        )
        .sort_values("blocked_candidates", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def expiration_policy_table() -> pd.DataFrame:
    """Return action-specific TTL and the event that invalidates the row."""

    refresh_trigger = {
        "No action": "Permission or access state changes",
        "Access follow-up": "Access-state change or resolved case",
        "Field conversation": "Completed call or territory refresh",
        "Program invitation": "Seat date, attendance, or new access barrier",
        "Approved email": "Content expiry, opt-out, or new engagement",
        "Continue responsive content": "Content expiry or new digital response",
        "Monitor": "Material evidence change or cycle refresh",
    }
    return pd.DataFrame(
        [
            {
                "candidate_action": action,
                "default_ttl_days": ACTION_TTL_DAYS[action],
                "stale_when": "TTL reached",
                "refresh_trigger": refresh_trigger[action],
            }
            for action in PRECEDENCE
        ]
    )


def score_action_values(candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach action-specific baseline, action response, uplift, and value."""

    work = candidates.copy()
    base = (
        work["predicted_response"]
        - 0.06 * work["digital_signal"].astype(float)
        - 0.03 * work["field_signal"].astype(float)
        - 0.02 * work["live_program_signal"].astype(float)
    )
    work["p_no_action"] = _clip_prob(base)

    review_scaled = work["review_opportunity"] / 10.0
    digital_rate = work["digital_response_rate"]
    field_rate = work["field_response_rate"]
    access_need = work["access_resource_score"]
    pressure_penalty = 0.02 * work["high_pressure_gate"].astype(float)

    action_bonus = pd.Series(0.0, index=work.index)
    action = work["candidate_action"]
    action_bonus = np.where(
        action.eq("Access follow-up"),
        ACTION_VALUE_BONUS["Access follow-up"] * (0.6 + access_need),
        action_bonus,
    )
    action_bonus = np.where(
        action.eq("Field conversation"),
        ACTION_VALUE_BONUS["Field conversation"]
        + 0.06 * review_scaled
        + 0.04 * field_rate
        - 0.03 * digital_rate,
        action_bonus,
    )
    action_bonus = np.where(
        action.eq("Program invitation"),
        ACTION_VALUE_BONUS["Program invitation"]
        + 0.07 * work["live_program_signal"].astype(float)
        + 0.05 * review_scaled
        - 0.02 * work["total_pressure_30"].ge(4).astype(float),
        action_bonus,
    )
    action_bonus = np.where(
        action.eq("Approved email"),
        ACTION_VALUE_BONUS["Approved email"]
        + 0.09 * work["digital_signal"].astype(float)
        + 0.03 * digital_rate
        - 0.02 * work["priority_gate"].astype(float),
        action_bonus,
    )
    action_bonus = np.where(
        action.eq("Continue responsive content"),
        ACTION_VALUE_BONUS["Continue responsive content"]
        + 0.07 * work["digital_signal"].astype(float)
        + 0.02 * digital_rate,
        action_bonus,
    )
    action_bonus = np.where(action.eq("Monitor"), 0.0, action_bonus)
    action_bonus = np.where(action.eq("No action"), 0.0, action_bonus)

    work["p_action"] = _clip_prob(work["p_no_action"] + action_bonus - pressure_penalty)
    work["estimated_uplift_action"] = np.where(
        action.eq("No action"),
        0.0,
        work["p_action"] - work["p_no_action"],
    )
    work["unit_cost"] = work["candidate_action"].map(ACTION_UNIT_COST).astype(float)
    work["fatigue_risk"] = np.where(
        work["candidate_action"].isin(["Approved email", "Continue responsive content"]),
        0.02 + 0.03 * work["email_at_cap"].astype(float) + 0.02 * work["high_pressure_gate"].astype(float),
        np.where(
            work["candidate_action"].eq("Field conversation"),
            0.03 + 0.02 * work["high_pressure_gate"].astype(float),
            0.01 + 0.01 * work["high_pressure_gate"].astype(float),
        ),
    )
    work["expected_incremental_value"] = (
        work["estimated_uplift_action"] * RESPONSE_VALUE
        - work["unit_cost"]
        - work["fatigue_risk"] * 200.0
    )
    return work


def reward_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return one best content row per HCP-account-action for valued actions."""

    work = score_action_values(candidates)
    view = (
        work.loc[
            work["eligible"]
            & work["candidate_action"].isin(["Field conversation", "Program invitation", "Approved email"])
        ]
        .sort_values(
            ["npi", "candidate_action", "selected_content_rank", "content_priority"],
            ascending=[True, True, True, True],
        )
        .groupby(["npi", "account_id", "candidate_action"], as_index=False)
        .first()
    )
    view["rank_by_response"] = (
        view["predicted_response"].rank(ascending=False, method="first").astype(int)
    )
    view["rank_by_value"] = (
        view["expected_incremental_value"].rank(ascending=False, method="first").astype(int)
    )
    view["rank_by_uplift"] = (
        view["estimated_uplift_action"].rank(ascending=False, method="first").astype(int)
    )
    return view.sort_values("rank_by_response").reset_index(drop=True)


def reward_overlap(reward: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Summarize how response ranking differs from value ranking."""

    from scipy.stats import spearmanr

    rho = float(
        spearmanr(reward["predicted_response"], reward["expected_incremental_value"]).statistic
    )
    top_response = set(reward.nsmallest(top_k, "rank_by_response")["npi"])
    top_value = set(reward.nsmallest(top_k, "rank_by_value")["npi"])
    return pd.DataFrame(
        [
            {"metric": "Promotional-eligible HCP-account rows", "value": float(len(reward))},
            {"metric": "Spearman response vs value", "value": round(rho, 3)},
            {"metric": f"Top-{top_k} shared by both rankings", "value": float(len(top_response & top_value))},
            {"metric": f"Top-{top_k} only in response ranking", "value": float(len(top_response - top_value))},
        ]
    )


def constrained_allocation_comparison(
    reward: pd.DataFrame,
    action: str = "Program invitation",
    capacity: int = PROGRAM_CAPACITY_LIMIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare response-ranked and value-ranked allocations in one action tier."""

    tier = reward.loc[reward["candidate_action"].eq(action)].copy()
    response_sel = tier.nsmallest(capacity, "rank_by_response").copy()
    response_sel["allocation_rule"] = "response_ranked"
    value_sel = tier.nsmallest(capacity, "rank_by_value").copy()
    value_sel["allocation_rule"] = "value_ranked"
    detail = pd.concat([response_sel, value_sel], ignore_index=True)
    summary = (
        detail.groupby("allocation_rule", as_index=False)
        .agg(
            released_slots=("npi", "nunique"),
            mean_predicted_response=("predicted_response", "mean"),
            mean_estimated_uplift=("estimated_uplift_action", "mean"),
            expected_incremental_value=("expected_incremental_value", "sum"),
        )
        .sort_values("allocation_rule")
        .reset_index(drop=True)
    )
    shared_rows = len(set(response_sel["npi"]) & set(value_sel["npi"]))
    summary["shared_rows"] = [shared_rows, shared_rows]
    summary["candidate_action"] = action
    return detail, summary


def value_components_trace(reward: pd.DataFrame) -> pd.DataFrame:
    """Return teaching rows that show how value differs from response ranking."""

    program = reward.loc[reward["candidate_action"].eq("Program invitation")].copy()
    response_row = program.nsmallest(1, "rank_by_response").copy()
    value_row = program.nsmallest(1, "rank_by_value").copy()
    view = pd.concat(
        [
            response_row.assign(example="Highest response"),
            value_row.assign(example="Highest value"),
        ],
        ignore_index=True,
    )
    return view[
        [
            "example",
            "npi",
            "account_id",
            "predicted_response",
            "p_no_action",
            "p_action",
            "estimated_uplift_action",
            "unit_cost",
            "fatigue_risk",
            "expected_incremental_value",
        ]
    ]


def multi_action_uplift_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-action incremental value for 3 teaching rows: HCP0280 plus two diverse examples."""
    scored = score_action_values(candidates)
    baseline = (
        scored.drop_duplicates(["npi", "account_id"])[["npi", "account_id", "p_no_action"]]
        .copy()
        .reset_index(drop=True)
    )
    for short, full in [
        ("email", "Approved email"),
        ("field", "Field conversation"),
        ("program", "Program invitation"),
    ]:
        src = (
            scored.loc[scored["candidate_action"].eq(full)]
            .groupby(["npi", "account_id"], as_index=False)
            .agg(**{f"p_{short}": ("p_action", "first"), f"uplift_{short}": ("estimated_uplift_action", "first")})
        )
        baseline = baseline.merge(src, on=["npi", "account_id"], how="left")
    for col in ["p_email", "uplift_email", "p_field", "uplift_field", "p_program", "uplift_program"]:
        baseline[col] = baseline[col].fillna(0.0)

    def _best(row: pd.Series) -> str:
        scores = {"Approved email": row["uplift_email"], "Field conversation": row["uplift_field"],
                  "Program invitation": row["uplift_program"]}
        return max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]

    baseline["best_incremental_action"] = baseline.apply(_best, axis=1)
    hcp0280 = baseline.loc[baseline["npi"].eq("9000000280")].assign(example="HCP0280").iloc[0:1]
    other = baseline.loc[~baseline["npi"].eq("9000000280")]
    high_email = other.nlargest(1, "uplift_email").assign(example="High-email")
    high_field = other.loc[~other["npi"].isin(high_email["npi"])].nlargest(1, "uplift_field").assign(
        example="High-field"
    )
    result = pd.concat([hcp0280, high_email, high_field], ignore_index=True)
    cols = ["example", "p_no_action", "p_email", "p_field", "p_program",
            "uplift_email", "uplift_field", "uplift_program", "best_incremental_action"]
    return result[cols].reset_index(drop=True)


def select_recommendations(
    state: pd.DataFrame,
    candidates: pd.DataFrame,
    policy_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Select one eligible candidate row per HCP-account pair."""

    eligible = score_action_values(candidates.loc[candidates["eligible"]].copy())
    selected = (
        eligible.sort_values(
            [
                "npi",
                "account_id",
                "policy_precedence",
                "selected_content_rank",
                "expected_incremental_value",
                "predicted_response",
            ],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(["npi", "account_id"], as_index=False)
        .first()
    )

    registry = policy_registry.iloc[0]
    recommendations = selected[
        [
            "npi",
            "account_id",
            "territory",
            "candidate_action",
            "candidate_channel",
            "content_id",
            "content_family",
            "reason_code",
            "expected_result",
            "measurement_hook",
            "policy_precedence",
            "predicted_response",
            "p_no_action",
            "p_action",
            "estimated_uplift_action",
            "expected_incremental_value",
            "fatigue_risk",
            "unit_cost",
            "binding_gate",
            "binding_reason",
            "context_bucket",
            "segment_name",
            "timing",
            "action_ttl_days",
            "approval_expires_on",
        ]
    ].rename(
        columns={
            "candidate_action": "recommended_action",
            "candidate_channel": "recommended_channel",
        }
    )
    recommendations["recommendation_date"] = RECOMMENDATION_DATE
    recommendations["expires_on"] = RECOMMENDATION_DATE + pd.to_timedelta(
        recommendations["action_ttl_days"], unit="D"
    )
    content_cap = recommendations["approval_expires_on"].where(
        recommendations["approval_expires_on"].notna(),
        recommendations["expires_on"],
    )
    recommendations["expires_on"] = recommendations[["expires_on"]].join(
        content_cap.rename("content_cap")
    )[["expires_on", "content_cap"]].min(axis=1)
    recommendations["review_required"] = recommendations["recommended_action"].isin(
        ACTION_REVIEW_REQUIRED
    )
    recommendations["hold_flag"] = recommendations["recommended_action"].eq("No action")
    recommendations["policy_version"] = registry["policy_version"]
    recommendations["rule_set_version"] = registry["rule_set_version"]
    recommendations["model_version"] = registry["model_version"]
    recommendations["content_catalog_version"] = registry["content_catalog_version"]
    recommendations["recommendation_id"] = [
        f"NBA{i:05d}" for i in range(1, len(recommendations) + 1)
    ]
    return recommendations.sort_values(
        ["policy_precedence", "expected_incremental_value", "predicted_response"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Count released actions and review burden."""

    return (
        recommendations.groupby("recommended_action", as_index=False)
        .agg(
            recommendations=("recommendation_id", "nunique"),
            review_required=("review_required", "sum"),
            mean_predicted_response=("predicted_response", "mean"),
            mean_estimated_uplift=("estimated_uplift_action", "mean"),
        )
        .sort_values("recommendations", ascending=False)
        .reset_index(drop=True)
    )


def candidate_audit(candidates: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    """Mark each candidate row as selected, blocked, or lower precedence."""

    selected = recommendations[["npi", "account_id", "content_id", "recommended_action"]].copy()
    selected = selected.rename(columns={"recommended_action": "candidate_action"})
    audit = candidates.merge(
        selected.assign(selected=True),
        on=["npi", "account_id", "candidate_action", "content_id"],
        how="left",
    )
    audit["selected"] = audit["selected"].fillna(False).astype(bool)
    audit["candidate_status"] = np.select(
        [audit["selected"], ~audit["eligible"]],
        ["Selected", "Ineligible"],
        default="Eligible but lower precedence",
    )
    return audit


def audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby("candidate_status", as_index=False)
        .agg(candidates=("candidate_action", "size"))
        .sort_values("candidates", ascending=False)
        .reset_index(drop=True)
    )


def candidate_status_by_action(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby(["candidate_action", "candidate_status"], as_index=False)
        .agg(candidates=("candidate_status", "size"))
        .sort_values(["candidate_action", "candidate_status"])
        .reset_index(drop=True)
    )


def content_gate_trace(audit: pd.DataFrame, focus_npi: str = "9000000280") -> pd.DataFrame:
    """Return content decisions for the carried HCP-account row."""

    view = audit.loc[
        audit["npi"].eq(focus_npi)
        & audit["candidate_action"].isin(ACTION_REQUIRES_CONTENT),
        [
            "candidate_action",
            "content_id",
            "content_family",
            "mlr_status",
            "audience",
            "approved_channel",
            "approval_expires_on",
            "content_gate_reason",
            "eligible",
        ],
    ].copy()
    return view.sort_values(["candidate_action", "content_id"]).reset_index(drop=True)


def recommendation_contract_dictionary() -> pd.DataFrame:
    """Describe the fields in the recommendation contract."""

    return pd.DataFrame(
        [
            {
                "field": "recommended_action",
                "question_answered": "What should happen next?",
                "why_it_matters": "Execution role and workload",
            },
            {
                "field": "recommended_channel",
                "question_answered": "Where should it happen?",
                "why_it_matters": "CRM, email, program, or access routing",
            },
            {
                "field": "content_id",
                "question_answered": "Which approved asset is used?",
                "why_it_matters": "MLR, indication, audience, and expiry control",
            },
            {
                "field": "reason_code",
                "question_answered": "Why was this action selected?",
                "why_it_matters": "User trust and audit",
            },
            {
                "field": "measurement_hook",
                "question_answered": "What must be observed later?",
                "why_it_matters": "Learning and accountability",
            },
            {
                "field": "expires_on",
                "question_answered": "When is the row stale?",
                "why_it_matters": "Prevents outdated action release",
            },
            {
                "field": "policy_version",
                "question_answered": "Which rule set produced it?",
                "why_it_matters": "Rollback and model-risk review",
            },
        ]
    )


def hcp0280_rejected_alternatives(audit: pd.DataFrame) -> pd.DataFrame:
    """Return the compact rejected-alternative trace for the carried case."""

    view = audit.loc[
        audit["npi"].eq("9000000280"),
        [
            "candidate_action",
            "content_id",
            "policy_precedence",
            "candidate_status",
            "binding_gate",
            "binding_reason",
        ],
    ].copy()
    view["content_id"] = view["content_id"].fillna("")
    return view.sort_values(["policy_precedence", "content_id"]).reset_index(drop=True)


def expiration_analysis(event_ledger: pd.DataFrame) -> pd.DataFrame:
    """Measure how fast evidence refreshes for the same HCP-account row."""

    gaps = inter_event_gaps(event_ledger)
    return pd.DataFrame(
        [
            {"metric": "Median days between events", "value": round(float(gaps.median()), 1)},
            {"metric": "Mean days between events", "value": round(float(gaps.mean()), 1)},
            {"metric": "Share of gaps within 14 days", "value": round(float((gaps <= 14).mean()), 3)},
            {"metric": "Share of gaps within 30 days", "value": round(float((gaps <= 30).mean()), 3)},
        ]
    )


def inter_event_gaps(event_ledger: pd.DataFrame) -> pd.Series:
    """Return inter-event gaps in days for figure use."""

    ledger = event_ledger.copy()
    ledger["event_date"] = pd.to_datetime(ledger["event_date"])
    ledger = ledger.sort_values(["npi", "account_id", "event_date"])
    ledger["previous"] = ledger.groupby(["npi", "account_id"])["event_date"].shift(1)
    return (ledger["event_date"] - ledger["previous"]).dt.days.dropna()


def _historical_state_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Approximate the recommendation state on each historical snapshot."""

    work = panel.copy()
    work["npi"] = work["npi"].astype(str)
    work["digital_signal"] = (
        work["email_clicks_90"] + work["web_actions_90"] + work["paid_clicks_90"]
    ).gt(0)
    work["priority_flag"] = work["account_action"].eq("Increase priority")
    work["live_program_signal"] = work["live_program_attendance_180"].gt(0)
    work["permitted"] = work["contact_permitted"].astype(bool)
    work["capacity_rank_hist"] = (
        work.groupby(["snapshot_date", "territory"])["predicted_response"]
        .rank(ascending=False, method="first")
    )
    work["field_capacity_available"] = work["capacity_rank_hist"].le(FIELD_CAPACITY_LIMIT)
    work = assign_context_bucket(work)
    return work


def build_logged_policy_history(panel: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Create a stored synthetic behavior-policy log with propensities."""

    history = _historical_state_from_panel(panel).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, object]] = []
    for idx, record in history.iterrows():
        gates = _gates(record)
        field_capacity = bool(record["field_capacity_available"])
        email_at_cap = int(record["email_frequency_30"]) >= EMAIL_FREQUENCY_CAP
        eligible_actions: list[str] = []
        scores: list[float] = []
        for action in PRECEDENCE:
            passed, _, _ = _action_gate_outcome(
                action=action,
                suppressed=gates["suppressed"],
                access_route=gates["access_route"],
                high_pressure=gates["high_pressure"],
                priority=gates["priority"],
                field_capacity=field_capacity,
                live_program=bool(record["live_program_signal"]),
                digital=bool(record["digital_signal"]),
                email_at_cap=email_at_cap,
            )
            if not passed:
                continue
            eligible_actions.append(action)
            if action == "No action":
                score = 0.05 + 0.10 * (1.0 - float(record["predicted_response"]))
            elif action == "Access follow-up":
                score = 0.25 + float(record["access_resource_score"])
            elif action == "Field conversation":
                score = 0.40 + 0.7 * float(record["predicted_response"])
            elif action == "Program invitation":
                score = 0.35 + 0.7 * float(record["predicted_response"])
            elif action == "Approved email":
                score = 0.30 + 0.9 * float(record["predicted_response"])
            elif action == "Continue responsive content":
                score = 0.20 + 0.5 * float(record["predicted_response"])
            else:
                score = 0.10 + 0.2 * float(record["predicted_response"])
            scores.append(score)

        score_arr = np.array(scores, dtype=float)
        score_arr = np.exp(score_arr - score_arr.max())
        greedy = score_arr / score_arr.sum()
        top = int(np.argmax(greedy))
        base_policy_action = eligible_actions[top]
        probs = np.full(len(eligible_actions), EXPLORE_EPSILON / len(eligible_actions))
        probs[top] += 1.0 - EXPLORE_EPSILON
        probs = probs / probs.sum()
        choice = int(rng.choice(len(eligible_actions), p=probs))
        logged_action = eligible_actions[choice]
        logged_probability = float(probs[choice])
        candidate_action = base_policy_action
        if (
            base_policy_action == "Field conversation"
            and "Approved email" in eligible_actions
            and float(record["predicted_response"]) >= 0.50
        ):
            candidate_action = "Approved email"

        rows.append(
            {
                "snapshot_id": f"SNAP{idx + 1:05d}",
                "snapshot_date": record["snapshot_date"],
                "npi": record["npi"],
                "account_id": record["account_id"],
                "territory": record["territory"],
                "context_bucket": record["context_bucket"],
                "eligible_actions": "; ".join(eligible_actions),
                "base_policy_action": base_policy_action,
                "logged_action": logged_action,
                "logged_probability": logged_probability,
                "candidate_action": candidate_action,
                "future_response": int(record["future_response"]),
                "predicted_response": float(record["predicted_response"]),
                "review_opportunity": float(record["review_opportunity"]),
                "evidence_need_score": float(record["evidence_need_score"]),
                "access_resource_score": float(record["access_resource_score"]),
                "digital_response_rate": float(record["digital_response_rate"]),
                "field_response_rate": float(record["field_response_rate"]),
                "total_pressure_30": int(record["total_pressure_30"]),
                "total_pressure_90": int(record["total_pressure_90"]),
                "shrunken_response_rate_90": float(record["shrunken_response_rate_90"]),
                "policy_version": POLICY_VERSION,
                "exploration_flag": bool(choice != top),
            }
        )
    return pd.DataFrame(rows)


def thompson_exploration(
    history: pd.DataFrame,
    seed: int = THOMPSON_SEED,
    months: int | None = None,
    focus_context: str | None = None,
) -> pd.DataFrame:
    """Report bucketed Thompson posterior summaries from the stored history."""

    work = history.copy()
    if focus_context is not None:
        work = work.loc[work["context_bucket"].eq(focus_context)].copy()
    if months is not None:
        earliest = sorted(work["snapshot_date"].unique())[:months]
        work = work.loc[work["snapshot_date"].isin(earliest)]
    arms = (
        work.groupby("logged_action", as_index=False)
        .agg(snapshots=("future_response", "size"), successes=("future_response", "sum"))
    )
    arms.insert(0, "context_bucket", focus_context or "All contexts")
    arms["failures"] = arms["snapshots"] - arms["successes"]
    arms["alpha"] = arms["successes"] + 1
    arms["beta"] = arms["failures"] + 1
    arms["posterior_mean"] = arms["alpha"] / (arms["alpha"] + arms["beta"])
    arms["posterior_sd"] = np.sqrt(
        arms["alpha"] * arms["beta"]
        / ((arms["alpha"] + arms["beta"]) ** 2 * (arms["alpha"] + arms["beta"] + 1))
    )
    rng = np.random.default_rng(seed)
    draws = rng.beta(
        arms["alpha"].to_numpy()[:, None],
        arms["beta"].to_numpy()[:, None],
        size=(len(arms), THOMPSON_DRAWS),
    )
    arms["explore_share"] = (draws == draws.max(axis=0)).mean(axis=1)
    arms["posterior_mean"] = arms["posterior_mean"].round(3)
    arms["posterior_sd"] = arms["posterior_sd"].round(3)
    arms["explore_share"] = arms["explore_share"].round(6)
    return arms.sort_values("posterior_mean", ascending=False).reset_index(drop=True)


def simulate_bucketed_thompson_decision(
    recommendations: pd.DataFrame,
    thompson_full: pd.DataFrame,
    eligible_arms: Sequence[str],
    focus_npi: str = "9000000280",
    seed: int = THOMPSON_SEED,
) -> pd.DataFrame:
    """Simulate one Thompson draw for the carried case."""

    row = recommendations.loc[recommendations["npi"].eq(focus_npi)].iloc[0]
    arms = thompson_full.loc[thompson_full["logged_action"].isin(eligible_arms)].copy()
    rng = np.random.default_rng(seed)
    draws = rng.beta(arms["alpha"], arms["beta"])
    arms = arms.assign(draw=float(0.0))
    arms["draw"] = draws
    selected = arms.loc[arms["draw"].idxmax()]
    selected_action = str(selected["logged_action"])
    selected_prob = float(selected["explore_share"])
    return pd.DataFrame(
        [
            {
                "npi": row["npi"],
                "account_id": row["account_id"],
                "context_bucket": row["context_bucket"],
                "eligible_arms": "; ".join(sorted(eligible_arms)),
                "base_policy_action": row["recommended_action"],
                "selected_arm": selected_action,
                "logged_probability": selected_prob,
                "exploration_flag": selected_action != row["recommended_action"],
                "policy_version": POLICY_VERSION,
            }
        ]
    )


def off_policy_evaluation(history: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Estimate a digital-first policy from stored logged probabilities."""

    actions = ["No action", "Access follow-up", "Field conversation", "Program invitation", "Approved email", "Monitor"]
    design = pd.concat(
        [
            history[UPLIFT_COVARIATES].reset_index(drop=True),
            pd.get_dummies(history["logged_action"]).reindex(columns=actions, fill_value=0),
        ],
        axis=1,
    )
    reward_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=seed)
    reward_model.fit(design, history["future_response"])

    def _predict(action_series: pd.Series) -> np.ndarray:
        frame = pd.concat(
            [
                history[UPLIFT_COVARIATES].reset_index(drop=True),
                pd.get_dummies(action_series.reset_index(drop=True)).reindex(
                    columns=actions, fill_value=0
                ),
            ],
            axis=1,
        )
        return reward_model.predict_proba(frame)[:, 1]

    matched = history["candidate_action"].eq(history["logged_action"]).to_numpy()
    weight = np.where(matched, 1.0 / history["logged_probability"].to_numpy(), 0.0)
    reward = history["future_response"].to_numpy()
    n = len(history)
    match_rate = float(matched.mean())
    ess = float((weight.sum() ** 2) / np.square(weight).sum()) if np.square(weight).sum() else 0.0
    max_weight = float(weight.max()) if len(weight) else 0.0

    logged_value = float(reward.mean())
    ips_value = float((weight * reward).sum() / n)
    snips_value = float((weight * reward).sum() / weight.sum()) if weight.sum() else np.nan
    direct_value = float(_predict(history["candidate_action"]).mean())
    q_candidate = _predict(history["candidate_action"])
    q_logged = _predict(history["logged_action"])
    dr_value = float((q_candidate + weight * (reward - q_logged)).mean())
    overlap_warning = (match_rate < 0.70) or (ess < 0.50 * max(int(matched.sum()), 1))
    warning_text = "Review overlap" if overlap_warning else "Overlap acceptable"

    return pd.DataFrame(
        [
            {
                "policy": "logged_policy",
                "estimator": "on_policy_mean",
                "estimated_response_rate": logged_value,
                "matched_snapshots": n,
                "match_rate": 1.0,
                "effective_sample_size": float(n),
                "max_weight": 1.0,
                "overlap_warning": "Reference policy",
            },
            {
                "policy": "digital_first",
                "estimator": "ips",
                "estimated_response_rate": ips_value,
                "matched_snapshots": int(matched.sum()),
                "match_rate": match_rate,
                "effective_sample_size": ess,
                "max_weight": max_weight,
                "overlap_warning": warning_text,
            },
            {
                "policy": "digital_first",
                "estimator": "snips",
                "estimated_response_rate": snips_value,
                "matched_snapshots": int(matched.sum()),
                "match_rate": match_rate,
                "effective_sample_size": ess,
                "max_weight": max_weight,
                "overlap_warning": warning_text,
            },
            {
                "policy": "digital_first",
                "estimator": "direct_method",
                "estimated_response_rate": direct_value,
                "matched_snapshots": int(matched.sum()),
                "match_rate": match_rate,
                "effective_sample_size": ess,
                "max_weight": max_weight,
                "overlap_warning": warning_text,
            },
            {
                "policy": "digital_first",
                "estimator": "doubly_robust",
                "estimated_response_rate": dr_value,
                "matched_snapshots": int(matched.sum()),
                "match_rate": match_rate,
                "effective_sample_size": ess,
                "max_weight": max_weight,
                "overlap_warning": warning_text,
            },
        ]
    )


def ope_replay_trace(history: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return a small row-level trace for teaching off-policy replay."""

    actions = [
        "No action",
        "Access follow-up",
        "Field conversation",
        "Program invitation",
        "Approved email",
        "Monitor",
    ]
    design = pd.concat(
        [
            history[UPLIFT_COVARIATES].reset_index(drop=True),
            pd.get_dummies(history["logged_action"]).reindex(columns=actions, fill_value=0),
        ],
        axis=1,
    )
    reward_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=seed)
    reward_model.fit(design, history["future_response"])

    def _predict(action_series: pd.Series) -> np.ndarray:
        frame = pd.concat(
            [
                history[UPLIFT_COVARIATES].reset_index(drop=True),
                pd.get_dummies(action_series.reset_index(drop=True)).reindex(
                    columns=actions, fill_value=0
                ),
            ],
            axis=1,
        )
        return reward_model.predict_proba(frame)[:, 1]

    work = history.copy()
    work["matched"] = work["candidate_action"].eq(work["logged_action"])
    work["inverse_weight"] = np.where(work["matched"], 1.0 / work["logged_probability"], 0.0)
    work["model_candidate_response"] = _predict(work["candidate_action"])
    work["dr_contribution"] = (
        work["model_candidate_response"]
        + work["inverse_weight"]
        * (work["future_response"] - _predict(work["logged_action"]))
    )
    examples = pd.concat(
        [
            work.loc[work["matched"]].head(3),
            work.loc[~work["matched"]].head(2),
        ],
        ignore_index=True,
    )
    return examples[
        [
            "snapshot_id",
            "context_bucket",
            "logged_action",
            "candidate_action",
            "logged_probability",
            "future_response",
            "matched",
            "inverse_weight",
            "model_candidate_response",
            "dr_contribution",
        ]
    ]


def precedence_experiment_design(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Power a recommendation-level test of a precedence change."""

    baseline = float(recommendations["p_action"].mean())
    treated = min(baseline + MINIMUM_DETECTABLE_EFFECT, 0.99)
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    pooled = (baseline + treated) / 2
    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_power * np.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
    ) ** 2
    per_arm = int(np.ceil(numerator / (MINIMUM_DETECTABLE_EFFECT**2)))
    eligible = int((~recommendations["recommended_action"].eq("No action")).sum())
    cycles = int(np.ceil((2 * per_arm) / max(eligible, 1)))
    return pd.DataFrame(
        [
            {"parameter": "Randomization unit", "value": "HCP-account row"},
            {"parameter": "Control policy", "value": "Current precedence"},
            {"parameter": "Candidate policy", "value": "Digital-first precedence"},
            {"parameter": "Primary outcome", "value": "Meaningful response"},
            {"parameter": "Measurement window days", "value": 14},
            {"parameter": "Guardrail outcomes", "value": "Opt-out; stale row; field burden; access delay"},
            {"parameter": "Baseline response rate", "value": round(baseline, 3)},
            {"parameter": "Minimum detectable effect", "value": MINIMUM_DETECTABLE_EFFECT},
            {"parameter": "Power", "value": POWER_TARGET},
            {"parameter": "Two-sided alpha", "value": ALPHA},
            {"parameter": "Required HCP-account rows per arm", "value": per_arm},
            {"parameter": "Eligible HCP-account rows this cycle", "value": eligible},
            {"parameter": "Cycles needed", "value": cycles},
        ]
    )


def simulate_execution_feedback(recommendations: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Simulate execution status and override reasons for released rows."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for record in recommendations.to_dict("records"):
        action = record["recommended_action"]
        if action == "No action":
            status = "Executed"
            reason = ""
        elif action == "Access follow-up":
            status = rng.choice(
                ["Executed", "Viewed not executed", "Overridden"],
                p=[0.70, 0.15, 0.15],
            )
            reason = "" if status != "Overridden" else "Access issue resolved"
        elif action == "Field conversation":
            status = rng.choice(
                ["Executed", "Viewed not executed", "Overridden", "Expired"],
                p=[0.58, 0.12, 0.20, 0.10],
            )
            reason = "" if status != "Overridden" else "Field judgment"
        elif action == "Program invitation":
            status = rng.choice(
                ["Executed", "Viewed not executed", "Overridden", "Expired"],
                p=[0.55, 0.15, 0.15, 0.15],
            )
            reason = "" if status != "Overridden" else "HCP unavailable"
        elif action in {"Approved email", "Continue responsive content"}:
            status = rng.choice(
                ["Executed", "Viewed not executed", "Overridden", "Suppressed after release"],
                p=[0.72, 0.12, 0.10, 0.06],
            )
            reason = "" if status != "Overridden" else "Content mismatch"
        else:
            status = rng.choice(["Executed", "Expired"], p=[0.65, 0.35])
            reason = ""
        rows.append(
            {
                "recommendation_id": record["recommendation_id"],
                "npi": record["npi"],
                "account_id": record["account_id"],
                "recommended_action": action,
                "execution_status": status,
                "override_reason": reason,
                "feedback_timestamp": RECOMMENDATION_DATE + pd.Timedelta(days=7),
            }
        )
    return pd.DataFrame(rows)


def execution_feedback_summary(feedback: pd.DataFrame) -> pd.DataFrame:
    work = (
        feedback.groupby("execution_status", as_index=False)
        .agg(recommendations=("recommendation_id", "nunique"))
        .sort_values("recommendations", ascending=False)
        .reset_index(drop=True)
    )
    work["share"] = work["recommendations"] / work["recommendations"].sum()
    return work


def override_reason_summary(feedback: pd.DataFrame) -> pd.DataFrame:
    override = feedback.loc[feedback["execution_status"].eq("Overridden")].copy()
    if override.empty:
        return pd.DataFrame(columns=["override_reason", "recommendations", "example_action"])
    return (
        override.groupby("override_reason", as_index=False)
        .agg(
            recommendations=("recommendation_id", "nunique"),
            example_action=("recommended_action", "first"),
        )
        .sort_values("recommendations", ascending=False)
        .reset_index(drop=True)
    )


def model_risk_controls() -> pd.DataFrame:
    """Return a compact production-control checklist for the NBA release layer."""

    return pd.DataFrame(
        [
            {
                "control": "Policy versioning",
                "failure_it_catches": "Unknown rule set in execution",
                "release_requirement": "policy_version and rule_set_version on every row",
            },
            {
                "control": "Content audit",
                "failure_it_catches": "Expired or wrong-audience asset",
                "release_requirement": "approved content ID with active dates",
            },
            {
                "control": "Propensity logging",
                "failure_it_catches": "Unusable offline replay",
                "release_requirement": "logged action probability when exploration runs",
            },
            {
                "control": "Overlap review",
                "failure_it_catches": "Candidate policy outside historical support",
                "release_requirement": "match rate, ESS, and max weight before rollout",
            },
            {
                "control": "Execution feedback",
                "failure_it_catches": "Recommendations ignored or overridden",
                "release_requirement": "status and override reason after release",
            },
        ]
    )


def run_analysis(repo_root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Return the full Chapter 9 package."""

    import sys

    sys.path.insert(0, str(repo_root))
    from ch08_omnichannel.generation_modules.synthetic import generate as generate_ch08
    from ch08_omnichannel.scripts.run_analysis import run_analysis as run_ch08

    generate_ch08(repo_root, repo_root / "ch08_omnichannel" / "data" / "generated")
    ch08 = run_ch08(repo_root)
    panel = ch08["scored_snapshots"].copy()
    panel["npi"] = panel["npi"].astype(str)

    state = load_state(ch08)
    content_catalog = build_content_catalog()
    policy_registry = build_policy_registry()
    action_menu = generate_action_menu(state)
    candidates = apply_content_gates(attach_content_candidates(action_menu, content_catalog))
    recommendations = select_recommendations(state, candidates, policy_registry)
    reward = reward_candidates(candidates)
    allocation_detail, allocation_summary = constrained_allocation_comparison(reward)
    audit = candidate_audit(candidates, recommendations)
    logged_policy_history = build_logged_policy_history(panel)
    focus_context = str(
        recommendations.loc[recommendations["npi"].eq("9000000280"), "context_bucket"].iloc[0]
    )
    thompson_cold = thompson_exploration(
        logged_policy_history, months=1, focus_context=focus_context
    )
    thompson_full = thompson_exploration(logged_policy_history, focus_context=focus_context)
    feedback = simulate_execution_feedback(recommendations)
    recommendation_contract = recommendations.loc[
        recommendations["npi"].eq("9000000280")
    ].reset_index(drop=True)
    hcp0280_audit = audit.loc[audit["npi"].eq("9000000280")].copy()
    hcp0280_eligible_arms = (
        hcp0280_audit.loc[hcp0280_audit["eligible"], "candidate_action"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return {
        "nba_object_model": nba_object_model(),
        "state": state,
        "policy_registry": policy_registry,
        "content_catalog": content_catalog,
        "action_menu": action_menu,
        "action_candidates": candidates,
        "gate_summary": gate_summary(action_menu),
        "gate_summary_by_gate": gate_summary_by_gate(candidates),
        "recommendations": recommendations,
        "recommendation_summary": recommendation_summary(recommendations),
        "recommendation_contract": recommendation_contract,
        "candidate_audit": audit,
        "audit_summary": audit_summary(audit),
        "candidate_status_by_action": candidate_status_by_action(audit),
        "hcp0280_audit": hcp0280_audit,
        "hcp0280_content_trace": content_gate_trace(audit),
        "recommendation_contract_dictionary": recommendation_contract_dictionary(),
        "hcp0280_rejected_alternatives": hcp0280_rejected_alternatives(audit),
        "expiration_policy": expiration_policy_table(),
        "expiration_analysis": expiration_analysis(ch08["event_ledger"]),
        "reward_candidates": reward,
        "reward_overlap": reward_overlap(reward),
        "value_components_trace": value_components_trace(reward),
        "multi_action_uplift": multi_action_uplift_table(candidates),
        "constrained_allocation": allocation_detail,
        "constrained_allocation_summary": allocation_summary,
        "logged_policy_history": logged_policy_history,
        "thompson_cold_start": thompson_cold,
        "thompson_exploration": thompson_full,
        "thompson_decision_log": simulate_bucketed_thompson_decision(
            recommendations, thompson_full, hcp0280_eligible_arms
        ),
        "off_policy_evaluation": off_policy_evaluation(logged_policy_history),
        "ope_replay_trace": ope_replay_trace(logged_policy_history),
        "experiment_design": precedence_experiment_design(recommendations),
        "execution_feedback": feedback,
        "execution_feedback_summary": execution_feedback_summary(feedback),
        "override_reason_summary": override_reason_summary(feedback),
        "model_risk_controls": model_risk_controls(),
        "event_ledger": ch08["event_ledger"],
    }


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write chapter outputs as CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    results = run_analysis(ROOT)
    output = ROOT / "ch09_nba" / "assets" / "generated_outputs"
    write_outputs(results, output)
    registry = results["policy_registry"].iloc[0]
    print("Next best action")
    print(f"  Recommendations: {len(results['recommendations']):,}")
    print(f"  Candidates: {len(results['action_candidates']):,}")
    print(f"  Content records: {len(results['content_catalog']):,}")
    print(f"  Policy version: {registry['policy_version']}")
    print(f"Wrote outputs to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
