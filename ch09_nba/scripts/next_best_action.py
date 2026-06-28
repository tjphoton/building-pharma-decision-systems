"""Governed next-best-action engine for Chapter 9.

The engine reads the omnichannel channel-plan state and turns it into one
auditable recommendation per HCP-account relationship. Eligibility gates and
policy precedence run first; response and uplift scores rank only inside the
eligible set. Later sections add exploration, off-policy evaluation, lifecycle,
and experiment design.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_DATE = pd.Timestamp("2025-02-28")
VALIDITY_DAYS = 14
HIGH_PRESSURE_MIN = 5
EMAIL_FREQUENCY_CAP = 2
EXPLORE_EPSILON = 0.10
SEED = 20260625

# Minimum detectable effect, power, and baseline for the precedence experiment.
MINIMUM_DETECTABLE_EFFECT = 0.05
POWER_TARGET = 0.80
ALPHA = 0.05

# Pre-action context shared by the uplift and reward models.
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

# The next-best-action menu and its governed precedence. Lower runs first.
PRECEDENCE = {
    "No action": 1,            # under suppression
    "Access follow-up": 10,
    "Field conversation": 20,
    "Program invitation": 25,
    "Approved email": 30,
    "Continue responsive content": 40,
    "Monitor": 80,
    "No action fallback": 90,  # nothing eligible cleared a stronger signal
}

ACTION_CHANNEL = {
    "No action": "None",
    "Access follow-up": "Access team",
    "Field conversation": "Field",
    "Program invitation": "Peer or speaker program",
    "Approved email": "Email",
    "Continue responsive content": "Web or email",
    "Monitor": "None",
    "No action fallback": "None",
}

ACTION_EXPECTED_RESULT = {
    "No action": "Avoid ineligible contact",
    "Access follow-up": "Clarify or resolve the access barrier",
    "Field conversation": "Complete a relevant account conversation",
    "Program invitation": "Secure attendance at a peer or speaker program",
    "Approved email": "Deliver approved content and earn a click",
    "Continue responsive content": "Maintain relevant content continuity",
    "Monitor": "Wait for a material change in evidence",
    "No action fallback": "Wait for a stronger eligible signal",
}

ACTION_MEASUREMENT_HOOK = {
    "No action": "Suppression compliance",
    "Access follow-up": "Access-state change and resolved attempts",
    "Field conversation": "Completed interaction and outcome",
    "Program invitation": "Invitation, attendance, and follow-up",
    "Approved email": "Delivery and click",
    "Continue responsive content": "Delivery and qualified action",
    "Monitor": "Evidence refresh and recommendation change",
    "No action fallback": "Evidence refresh and recommendation change",
}


# --- State assembly ------------------------------------------------------------


def load_state(ch08_results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the channel-plan governance state with snapshot scoring features."""

    plan = ch08_results["channel_plan"].copy()
    plan["npi"] = plan["npi"].astype(str)
    panel = ch08_results["scored_snapshots"].copy()
    panel["npi"] = panel["npi"].astype(str)
    snapshot = panel.loc[panel["snapshot_date"].eq(RECOMMENDATION_DATE)].copy()
    uplift = _score_uplift(panel)
    snapshot = snapshot.merge(
        uplift[["npi", "account_id", "snapshot_date", "estimated_uplift"]],
        on=["npi", "account_id", "snapshot_date"],
        how="left",
        validate="one_to_one",
    )
    feature_columns = [
        "npi",
        "account_id",
        "future_response",
        "live_program_attendance_180",
        "peer_frequency_90",
        "speaker_frequency_90",
        "evidence_need_score",
        "access_resource_score",
        "digital_response_rate",
        "field_response_rate",
        "field_responses_90",
        "email_clicks_90",
        "web_actions_90",
        "paid_clicks_90",
        "email_frequency_30",
        "segment_name",
        "estimated_uplift",
    ]
    state = plan.merge(
        snapshot[feature_columns],
        on=["npi", "account_id"],
        how="left",
        validate="one_to_one",
    )
    state["digital_response"] = (
        state["email_clicks_90"] + state["web_actions_90"] + state["paid_clicks_90"]
    ).gt(0)
    state["field_response"] = state["field_responses_90"].gt(0)
    state["live_program_response"] = state["live_program_attendance_180"].gt(0)
    state["permitted"] = state["contact_permission_status"].eq("Allowed")
    state = assign_context_bucket(state)
    return state


def assign_context_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign a compact context label for bandit summaries."""

    work = frame.copy()
    if "digital_response" not in work.columns:
        work["digital_response"] = (
            work["email_clicks_90"] + work["web_actions_90"] + work["paid_clicks_90"]
        ).gt(0)
    work["context_bucket"] = np.select(
        [
            work["live_program_attendance_180"].gt(0),
            work["digital_response"],
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


def _score_uplift(panel: pd.DataFrame) -> pd.DataFrame:
    """T-learner uplift for the live-program action, reused for reward design."""

    work = panel.copy()
    work["treated"] = work["live_program_attendance_180"].gt(0).astype(int)
    treated = work.loc[work["treated"].eq(1)]
    control = work.loc[work["treated"].eq(0)]
    treated_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=SEED)
    control_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=SEED)
    treated_model.fit(treated[UPLIFT_COVARIATES], treated["future_response"])
    control_model.fit(control[UPLIFT_COVARIATES], control["future_response"])
    work["estimated_uplift"] = (
        treated_model.predict_proba(work[UPLIFT_COVARIATES])[:, 1]
        - control_model.predict_proba(work[UPLIFT_COVARIATES])[:, 1]
    )
    return work


# --- Candidate generation and eligibility -------------------------------------


def _gates(record: pd.Series) -> dict[str, bool]:
    suppressed = (
        not bool(record["permitted"])
        or record["account_action"] == "Hold contact"
    )
    access_route = (
        record["account_action"] == "Access review"
        or record["competitive_action"] in {"Access work", "Dual workstream"}
    )
    high_pressure = int(record["total_pressure_30"]) >= HIGH_PRESSURE_MIN
    priority = record["account_action"] == "Increase priority"
    return {
        "suppressed": suppressed,
        "access_route": access_route,
        "high_pressure": high_pressure,
        "priority": priority,
    }


def generate_candidates(state: pd.DataFrame) -> pd.DataFrame:
    """Build the full candidate menu per relationship with eligibility flags."""

    rows: list[dict[str, object]] = []
    for record in state.to_dict("records"):
        series = pd.Series(record)
        gate = _gates(series)
        suppressed = gate["suppressed"]
        access_route = gate["access_route"]
        promo_ok = not suppressed and not access_route and not gate["high_pressure"]
        field_capacity = pd.notna(record.get("capacity_rank"))
        email_at_cap = int(record["email_frequency_30"]) >= EMAIL_FREQUENCY_CAP
        base = {"npi": record["npi"], "account_id": record["account_id"]}

        menu = [
            (
                "No action",
                True,
                PRECEDENCE["No action"] if suppressed else PRECEDENCE["No action fallback"],
                "Permission or policy suppresses contact"
                if suppressed
                else "No higher-precedence eligible action passed",
            ),
            (
                "Access follow-up",
                not suppressed and access_route,
                PRECEDENCE["Access follow-up"],
                "Account evidence points to access friction",
            ),
            (
                "Field conversation",
                promo_ok and gate["priority"] and field_capacity,
                PRECEDENCE["Field conversation"],
                "Priority relationship with permitted field capacity",
            ),
            (
                "Program invitation",
                promo_ok and bool(record["live_program_response"]),
                PRECEDENCE["Program invitation"],
                "Prior live-program attendance supports a repeat invitation",
            ),
            (
                "Approved email",
                promo_ok
                and (gate["priority"] or bool(record["digital_response"]))
                and not email_at_cap,
                PRECEDENCE["Approved email"],
                "Available email frequency with a priority or digital signal",
            ),
            (
                "Continue responsive content",
                not suppressed
                and not access_route
                and bool(record["digital_response"])
                and not gate["priority"],
                PRECEDENCE["Continue responsive content"],
                "Meaningful digital response without a higher-priority action",
            ),
            (
                "Monitor",
                not suppressed and not access_route,
                PRECEDENCE["Monitor"],
                "Eligible relationship without a stronger action signal",
            ),
        ]
        for action, eligible, precedence, reason in menu:
            rows.append(
                {
                    **base,
                    "candidate_action": action,
                    "candidate_channel": ACTION_CHANNEL.get(action, "None"),
                    "eligible": bool(eligible),
                    "suppressed_gate": suppressed,
                    "access_route_gate": access_route,
                    "high_pressure_gate": gate["high_pressure"],
                    "priority_gate": gate["priority"],
                    "field_capacity_available": field_capacity,
                    "live_program_signal": bool(record["live_program_response"]),
                    "digital_signal": bool(record["digital_response"]),
                    "email_at_cap": email_at_cap,
                    "ineligibility_reason": _candidate_ineligibility_reason(
                        action=action,
                        eligible=bool(eligible),
                        suppressed=suppressed,
                        access_route=access_route,
                        high_pressure=gate["high_pressure"],
                        priority=gate["priority"],
                        field_capacity=field_capacity,
                        live_program=bool(record["live_program_response"]),
                        digital=bool(record["digital_response"]),
                        email_at_cap=email_at_cap,
                    ),
                    "policy_precedence": precedence,
                    "reason_code": reason,
                    "expected_result": ACTION_EXPECTED_RESULT.get(action, ""),
                    "measurement_hook": ACTION_MEASUREMENT_HOOK.get(action, ""),
                }
            )
    candidates = pd.DataFrame(rows)
    scores = state[["npi", "account_id", "predicted_response", "estimated_uplift"]]
    candidates = candidates.merge(
        scores, on=["npi", "account_id"], how="left", validate="many_to_one"
    )
    promotional = candidates["candidate_action"].isin(
        ["Field conversation", "Program invitation", "Approved email", "Continue responsive content"]
    )
    candidates["decision_score"] = np.where(
        promotional, candidates["predicted_response"], np.nan
    )
    return candidates


def _candidate_ineligibility_reason(
    *,
    action: str,
    eligible: bool,
    suppressed: bool,
    access_route: bool,
    high_pressure: bool,
    priority: bool,
    field_capacity: bool,
    live_program: bool,
    digital: bool,
    email_at_cap: bool,
) -> str:
    """Return the first policy reason that blocks a candidate action."""

    if eligible:
        return "Passed"
    if action == "Access follow-up":
        if suppressed:
            return "Suppressed"
        return "No access route"
    if action == "Field conversation":
        if suppressed:
            return "Suppressed"
        if access_route:
            return "Access route first"
        if high_pressure:
            return "High pressure"
        if not priority:
            return "Not priority"
        if not field_capacity:
            return "No field capacity rank"
    if action == "Program invitation":
        if suppressed:
            return "Suppressed"
        if access_route:
            return "Access route first"
        if high_pressure:
            return "High pressure"
        if not live_program:
            return "No live-program signal"
    if action == "Approved email":
        if suppressed:
            return "Suppressed"
        if access_route:
            return "Access route first"
        if high_pressure:
            return "High pressure"
        if not (priority or digital):
            return "No priority or digital signal"
        if email_at_cap:
            return "Email cap"
    if action == "Continue responsive content":
        if suppressed:
            return "Suppressed"
        if access_route:
            return "Access route first"
        if not digital:
            return "No digital signal"
        if priority:
            return "Priority handled by higher tier"
    if action == "Monitor":
        if suppressed:
            return "Suppressed"
        if access_route:
            return "Access route first"
    return "Policy gate"


def gate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate eligibility and the first blocking reason."""

    blocked = candidates.loc[~candidates["eligible"]]
    summary = (
        blocked.groupby("ineligibility_reason", as_index=False)
        .agg(blocked_candidates=("candidate_action", "size"))
        .sort_values("blocked_candidates", ascending=False)
        .reset_index(drop=True)
    )
    passed = pd.DataFrame(
        [{"ineligibility_reason": "Passed", "blocked_candidates": int(candidates["eligible"].sum())}]
    )
    return pd.concat([summary, passed], ignore_index=True)


# --- Precedence selection and the recommendation contract ----------------------


def select_recommendations(
    state: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Select the highest-precedence eligible candidate per relationship."""

    eligible = candidates.loc[candidates["eligible"]].copy()
    selected = (
        eligible.sort_values(
            ["npi", "account_id", "policy_precedence", "candidate_action"]
        )
        .groupby(["npi", "account_id"], as_index=False)
        .first()
    )
    selected["recommended_action"] = selected["candidate_action"].replace(
        {"No action fallback": "No action"}
    )
    context = state[
        [
            "npi",
            "account_id",
            "territory",
            "capacity_rank",
            "capacity_selected",
            "account_action",
            "competitive_action",
            "contact_permission_status",
            "pressure_band",
            "predicted_response",
            "estimated_uplift",
            "segment_name",
            "context_bucket",
        ]
    ]
    recommendations = context.merge(
        selected[
            [
                "npi",
                "account_id",
                "recommended_action",
                "candidate_channel",
                "policy_precedence",
                "reason_code",
                "expected_result",
                "measurement_hook",
            ]
        ].rename(columns={"candidate_channel": "recommended_channel"}),
        on=["npi", "account_id"],
        how="left",
        validate="one_to_one",
    )
    recommendations["recommendation_date"] = RECOMMENDATION_DATE
    recommendations["expires_on"] = RECOMMENDATION_DATE + pd.Timedelta(days=VALIDITY_DAYS)
    recommendations = recommendations.sort_values(
        ["policy_precedence", "predicted_response"], ascending=[True, False]
    ).reset_index(drop=True)
    recommendations["recommendation_id"] = [
        f"NBA{i:05d}" for i in range(1, len(recommendations) + 1)
    ]
    recommendations["hold_flag"] = recommendations["recommended_action"].eq("No action")
    recommendations["review_required"] = recommendations["recommended_action"].isin(
        ["Access follow-up", "Field conversation"]
    )
    return recommendations


def recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Count recommendations and review volume by action."""

    return (
        recommendations.groupby("recommended_action", as_index=False)
        .agg(
            recommendations=("recommendation_id", "nunique"),
            review_required=("review_required", "sum"),
            mean_predicted_response=("predicted_response", "mean"),
        )
        .sort_values("recommendations", ascending=False)
        .reset_index(drop=True)
    )


# --- Reward design: response vs uplift -----------------------------------------


PROMOTIONAL_ACTIONS = ["Field conversation", "Program invitation", "Approved email"]


def reward_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-relationship promotional rewards, ranked by response and by uplift."""

    promotional = (
        candidates.loc[
            candidates["eligible"]
            & candidates["candidate_action"].isin(PROMOTIONAL_ACTIONS)
        ]
        .drop_duplicates(["npi", "account_id"])
        .copy()
    )
    promotional["rank_by_response"] = (
        promotional["predicted_response"].rank(ascending=False, method="first").astype(int)
    )
    promotional["rank_by_uplift"] = (
        promotional["estimated_uplift"].rank(ascending=False, method="first").astype(int)
    )
    return promotional.sort_values("rank_by_response").reset_index(drop=True)


def reward_overlap(candidates: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Quantify how much the uplift ranking disagrees with the response ranking."""

    from scipy.stats import spearmanr

    promotional = reward_candidates(candidates)
    rho = float(
        spearmanr(promotional["predicted_response"], promotional["estimated_uplift"]).statistic
    )
    top_response = set(
        promotional.nsmallest(top_k, "rank_by_response")["npi"]
    )
    top_uplift = set(promotional.nsmallest(top_k, "rank_by_uplift")["npi"])
    return pd.DataFrame(
        [
            {"metric": "Promotional-eligible relationships", "value": float(len(promotional))},
            {"metric": "Spearman response vs uplift", "value": round(rho, 3)},
            {"metric": f"Top-{top_k} shared by both rankings", "value": float(len(top_response & top_uplift))},
            {"metric": f"Top-{top_k} only in response ranking", "value": float(len(top_response - top_uplift))},
        ]
    )


# --- Candidate audit -----------------------------------------------------------


def candidate_audit(
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Mark every candidate as selected, ineligible, or lower precedence."""

    selected = recommendations[["npi", "account_id", "recommended_action"]].copy()
    selected["recommended_action"] = selected["recommended_action"].replace(
        {"No action": "No action"}
    )
    audit = candidates.copy()
    audit["candidate_action_norm"] = audit["candidate_action"].replace(
        {"No action fallback": "No action"}
    )
    audit = audit.merge(
        selected.assign(_selected=True).rename(
            columns={"recommended_action": "candidate_action_norm"}
        ),
        on=["npi", "account_id", "candidate_action_norm"],
        how="left",
    )
    audit["selected"] = audit["_selected"].eq(True)
    audit["candidate_status"] = np.select(
        [audit["selected"], ~audit["eligible"]],
        ["Selected", "Ineligible"],
        default="Eligible but lower precedence",
    )
    return audit.drop(columns=["_selected", "candidate_action_norm"])


def audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Count candidates by status."""

    return (
        audit.groupby("candidate_status", as_index=False)
        .agg(candidates=("candidate_action", "size"))
        .sort_values("candidates", ascending=False)
        .reset_index(drop=True)
    )


def candidate_status_by_action(audit: pd.DataFrame) -> pd.DataFrame:
    """Count selected and rejected candidates by action."""

    return (
        audit.groupby(["candidate_action", "candidate_status"], as_index=False)
        .agg(candidates=("candidate_status", "size"))
        .sort_values(["candidate_action", "candidate_status"])
        .reset_index(drop=True)
    )


# --- Lifecycle and expiration --------------------------------------------------


def expiration_analysis(event_ledger: pd.DataFrame) -> pd.DataFrame:
    """Justify the validity window from how fast relationship evidence refreshes."""

    ledger = event_ledger.copy()
    ledger["event_date"] = pd.to_datetime(ledger["event_date"])
    ledger = ledger.sort_values(["npi", "account_id", "event_date"])
    ledger["previous"] = ledger.groupby(["npi", "account_id"])["event_date"].shift(1)
    gaps = (ledger["event_date"] - ledger["previous"]).dt.days.dropna()
    return pd.DataFrame(
        [
            {"metric": "Median days between events", "value": round(float(gaps.median()), 1)},
            {"metric": "Mean days between events", "value": round(float(gaps.mean()), 1)},
            {"metric": "Share of gaps within 14 days", "value": round(float((gaps <= 14).mean()), 3)},
            {"metric": "Share of gaps within 30 days", "value": round(float((gaps <= 30).mean()), 3)},
        ]
    )


def inter_event_gaps(event_ledger: pd.DataFrame) -> pd.Series:
    """Return the per-relationship inter-event gap in days, for the figure."""

    ledger = event_ledger.copy()
    ledger["event_date"] = pd.to_datetime(ledger["event_date"])
    ledger = ledger.sort_values(["npi", "account_id", "event_date"])
    ledger["previous"] = ledger.groupby(["npi", "account_id"])["event_date"].shift(1)
    return (ledger["event_date"] - ledger["previous"]).dt.days.dropna()


# --- Exploration: Thompson sampling over action arms ---------------------------


def _base_logged_actions(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each historical snapshot with the action the base policy implies."""

    work = assign_context_bucket(panel)
    work["digital_response"] = (
        work["email_clicks_90"] + work["web_actions_90"] + work["paid_clicks_90"]
    ).gt(0)
    work["logged_action"] = np.select(
        [
            ~work["contact_permitted"],
            work["live_program_attendance_180"].gt(0),
            work["field_responses_90"].gt(0),
            work["digital_response"],
        ],
        ["No action", "Program invitation", "Field conversation", "Approved email"],
        default="Monitor",
    )
    return work


def thompson_exploration(
    panel: pd.DataFrame,
    seed: int = SEED,
    months: int | None = None,
    focus_context: str | None = None,
) -> pd.DataFrame:
    """Seed a Beta arm per action from history and report posterior summaries.

    With ``months`` set, only the earliest snapshot months seed the arms, which
    reproduces the cold-start regime where posteriors are wide and exploration
    spreads across arms.
    """

    history = _base_logged_actions(panel)
    if focus_context is not None:
        history = history.loc[history["context_bucket"].eq(focus_context)].copy()
    if months is not None:
        earliest = sorted(history["snapshot_date"].unique())[:months]
        history = history.loc[history["snapshot_date"].isin(earliest)]
    arms = (
        history.groupby("logged_action", as_index=False)
        .agg(
            snapshots=("future_response", "size"),
            successes=("future_response", "sum"),
        )
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
        size=(len(arms), 2_000),
    )
    arms["explore_share"] = (draws == draws.max(axis=0)).mean(axis=1)
    arms["posterior_mean"] = arms["posterior_mean"].round(3)
    arms["posterior_sd"] = arms["posterior_sd"].round(3)
    arms["explore_share"] = arms["explore_share"].round(3)
    return arms.sort_values("posterior_mean", ascending=False).reset_index(drop=True)


def thompson_beta_params(panel: pd.DataFrame) -> pd.DataFrame:
    """Return alpha/beta per arm for the posterior-density figure."""

    history = _base_logged_actions(panel)
    arms = (
        history.groupby("logged_action", as_index=False)
        .agg(successes=("future_response", "sum"), snapshots=("future_response", "size"))
    )
    arms["alpha"] = arms["successes"] + 1
    arms["beta"] = arms["snapshots"] - arms["successes"] + 1
    return arms


# --- Off-policy evaluation of an alternative precedence ------------------------


def off_policy_evaluation(panel: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Estimate a digital-first precedence variant from logged history.

    The logged policy is the base action with a high epsilon-greedy logging
    propensity that varies slightly by context. The candidate policy is
    digital-first: it elevates the email action above the field action when
    predicted response is high. IPS,
    self-normalized IPS, and a doubly-robust estimator value the candidate.
    """

    history = _base_logged_actions(panel).reset_index(drop=True)
    actions = ["No action", "Program invitation", "Field conversation", "Approved email", "Monitor"]
    base = history["logged_action"]
    base_probability = 1 - EXPLORE_EPSILON + EXPLORE_EPSILON / len(actions)
    adjustment = (
        0.03 * history["live_program_attendance_180"].gt(0).astype(float)
        - 0.04 * history["digital_response"].astype(float)
        - 0.02 * history["total_pressure_30"].ge(HIGH_PRESSURE_MIN).astype(float)
        + 0.02 * history["field_responses_90"].gt(0).astype(float)
    )
    logged_probability = np.clip(base_probability + adjustment, 0.82, 0.96)
    history["logged_probability"] = logged_probability

    candidate = base.copy()
    digital_first = (
        base.eq("Field conversation") & history["predicted_response"].ge(0.5)
    )
    candidate = candidate.mask(digital_first, "Approved email")
    history["candidate_action"] = candidate

    reward_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=seed)
    design = pd.concat(
        [
            history[UPLIFT_COVARIATES].reset_index(drop=True),
            pd.get_dummies(history["logged_action"]).reindex(columns=actions, fill_value=0),
        ],
        axis=1,
    )
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

    logged_value = float(reward.mean())
    ips_value = float((weight * reward).sum() / n)
    snips_value = float((weight * reward).sum() / weight.sum()) if weight.sum() else np.nan
    q_candidate = _predict(history["candidate_action"])
    q_logged = _predict(history["logged_action"])
    dr_value = float((q_candidate + weight * (reward - q_logged)).mean())
    ess = float((weight.sum() ** 2) / np.square(weight).sum()) if np.square(weight).sum() else 0.0
    return pd.DataFrame(
        [
            {"policy": "logged_policy", "estimator": "on_policy_mean",
             "estimated_response_rate": logged_value,
             "matched_snapshots": n, "effective_sample_size": float(n)},
            {"policy": "digital_first", "estimator": "ips",
             "estimated_response_rate": ips_value,
             "matched_snapshots": int(matched.sum()), "effective_sample_size": ess},
            {"policy": "digital_first", "estimator": "snips",
             "estimated_response_rate": snips_value,
             "matched_snapshots": int(matched.sum()), "effective_sample_size": ess},
            {"policy": "digital_first", "estimator": "doubly_robust",
             "estimated_response_rate": dr_value,
             "matched_snapshots": int(matched.sum()), "effective_sample_size": ess},
        ]
    )


# --- Experiment design for the precedence question -----------------------------


def precedence_experiment_design(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Power a recommendation-level test of an alternative precedence ordering."""

    baseline = float(recommendations["predicted_response"].mean())
    treated = min(baseline + MINIMUM_DETECTABLE_EFFECT, 0.99)
    z_alpha = 1.959963984540054   # two-sided alpha 0.05
    z_power = 0.8416212335729143  # power 0.80
    pooled = (baseline + treated) / 2
    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_power * np.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
    ) ** 2
    per_arm = int(np.ceil(numerator / (MINIMUM_DETECTABLE_EFFECT ** 2)))
    eligible = int((~recommendations["recommended_action"].eq("No action")).sum())
    cycles = int(np.ceil((2 * per_arm) / max(eligible, 1)))
    return pd.DataFrame(
        [
            {"parameter": "Baseline response rate", "value": round(baseline, 3)},
            {"parameter": "Minimum detectable effect", "value": MINIMUM_DETECTABLE_EFFECT},
            {"parameter": "Power", "value": POWER_TARGET},
            {"parameter": "Two-sided alpha", "value": ALPHA},
            {"parameter": "Required relationships per arm", "value": per_arm},
            {"parameter": "Eligible relationships this cycle", "value": eligible},
            {"parameter": "Cycles to reach both arms", "value": cycles},
        ]
    )


# --- Orchestration -------------------------------------------------------------


def run_analysis(repo_root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Return the complete Chapter 9 next-best-action package."""

    import sys

    sys.path.insert(0, str(repo_root))
    from ch08_omnichannel.generation_modules.synthetic import generate as generate_ch08
    from ch08_omnichannel.scripts.run_analysis import run_analysis as run_ch08

    # Regenerate the omnichannel data so Chapter 9 always reconciles to the
    # current omnichannel source rather than a stale CSV snapshot on disk.
    generate_ch08(repo_root, repo_root / "ch08_omnichannel" / "data" / "generated")
    ch08 = run_ch08(repo_root)
    panel = ch08["scored_snapshots"].copy()
    panel["npi"] = panel["npi"].astype(str)
    state = load_state(ch08)
    focus_context = str(
        state.loc[state["npi"].eq("9000000280"), "context_bucket"].iloc[0]
    )
    candidates = generate_candidates(state)
    recommendations = select_recommendations(state, candidates)
    audit = candidate_audit(candidates, recommendations)
    return {
        "state": state,
        "action_candidates": candidates,
        "gate_summary": gate_summary(candidates),
        "recommendations": recommendations,
        "recommendation_summary": recommendation_summary(recommendations),
        "reward_candidates": reward_candidates(candidates),
        "reward_overlap": reward_overlap(candidates),
        "candidate_audit": audit,
        "audit_summary": audit_summary(audit),
        "candidate_status_by_action": candidate_status_by_action(audit),
        "expiration_analysis": expiration_analysis(ch08["event_ledger"]),
        "thompson_exploration": thompson_exploration(panel, focus_context=focus_context),
        "thompson_cold_start": thompson_exploration(
            panel, months=2, focus_context=focus_context
        ),
        "thompson_beta_params": thompson_beta_params(panel),
        "off_policy_evaluation": off_policy_evaluation(panel),
        "experiment_design": precedence_experiment_design(recommendations),
        "event_ledger": ch08["event_ledger"],
    }


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write the analysis outputs as CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    results = run_analysis(ROOT)
    output = ROOT / "ch09_nba" / "assets" / "generated_outputs"
    write_outputs(results, output)
    print("Next best action")
    print(f"  Recommendations: {len(results['recommendations']):,}")
    print(f"  Candidates: {len(results['action_candidates']):,}")
    actions = results["recommendation_summary"]["recommended_action"].tolist()
    print(f"  Actions: {', '.join(actions)}")
    print(f"Wrote outputs to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
