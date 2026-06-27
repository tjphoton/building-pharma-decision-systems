"""Practitioner extensions: uplift, off-policy evaluation, sequence features.

Each method has one builder for its shared inputs so the same feature engineering
is never copied across the summary and diagnostic functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Pre-action context used as controls for the uplift and reward models.
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

# Logging-policy selection probabilities for the off-policy example.
LOGGED_ACTION_PROBABILITY = {
    "Field": 0.45,
    "Email": 0.35,
    "Live program": 0.25,
    "Observe": 0.60,
}
POLICY_ACTIONS = ["Observe", "Field", "Email", "Live program", "Account support"]


# --- Uplift (T-learner on the planted live-program effect) ---------------------


def _score_uplift(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit a T-learner once and return every snapshot with its estimated uplift.

    The treatment is a live-program meaningful response in the prior 180 days. A
    separate response model is fit on treated and on untreated snapshots, and the
    per-snapshot uplift is the difference in predicted response between the two.
    """

    work = panel.copy()
    work["live_program_action"] = work["live_program_attendance_180"].gt(0).astype(int)
    treated = work.loc[work["live_program_action"].eq(1)]
    control = work.loc[work["live_program_action"].eq(0)]
    treated_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=20260622)
    control_model = LogisticRegression(C=0.3, max_iter=1_000, random_state=20260622)
    treated_model.fit(treated[UPLIFT_COVARIATES], treated["future_response"])
    control_model.fit(control[UPLIFT_COVARIATES], control["future_response"])
    work["response_if_action"] = treated_model.predict_proba(
        work[UPLIFT_COVARIATES]
    )[:, 1]
    work["response_if_no_action"] = control_model.predict_proba(
        work[UPLIFT_COVARIATES]
    )[:, 1]
    work["estimated_uplift"] = (
        work["response_if_action"] - work["response_if_no_action"]
    )
    return work


def uplift_segment_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank snapshots into uplift quintiles and show that uplift is not response."""

    work = _score_uplift(panel)
    work["uplift_segment"] = pd.qcut(
        work["estimated_uplift"],
        q=5,
        labels=["Low", "Mid-low", "Mid", "Mid-high", "High"],
        duplicates="drop",
    )
    result = (
        work.groupby("uplift_segment", observed=True, as_index=False)
        .agg(
            snapshots=("npi", "size"),
            response_rate=("future_response", "mean"),
            mean_baseline_response=("response_if_no_action", "mean"),
            mean_predicted_response_if_contacted=("response_if_action", "mean"),
            mean_uplift=("estimated_uplift", "mean"),
        )
        .sort_values("mean_uplift", ascending=False)
        .reset_index(drop=True)
    )
    return result[
        [
            "uplift_segment",
            "snapshots",
            "response_rate",
            "mean_baseline_response",
            "mean_predicted_response_if_contacted",
            "mean_uplift",
        ]
    ]


def uplift_response_contrast(panel: pd.DataFrame) -> pd.DataFrame:
    """Show that the highest baseline responders gain the least from the action."""

    work = _score_uplift(panel)
    work["response_band"] = pd.qcut(
        work["response_if_no_action"],
        q=5,
        labels=["B1 lowest", "B2", "B3", "B4", "B5 highest"],
        duplicates="drop",
    )
    return (
        work.groupby("response_band", observed=True, as_index=False)
        .agg(
            snapshots=("npi", "size"),
            mean_baseline_response=("response_if_no_action", "mean"),
            mean_uplift=("estimated_uplift", "mean"),
        )
        .sort_values("mean_baseline_response")
        .reset_index(drop=True)
    )


def uplift_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize the planted-effect recovery: observed uplift in top vs bottom band."""

    work = _score_uplift(panel)
    treated = work.loc[work["live_program_action"].eq(1), "future_response"].mean()
    control = work.loc[work["live_program_action"].eq(0), "future_response"].mean()
    top = work.loc[work["estimated_uplift"] >= work["estimated_uplift"].quantile(0.75)]
    bottom = work.loc[work["estimated_uplift"] <= work["estimated_uplift"].quantile(0.25)]
    return pd.DataFrame(
        [
            {
                "treated_snapshots": int(work["live_program_action"].sum()),
                "control_snapshots": int((1 - work["live_program_action"]).sum()),
                "naive_treated_minus_control": float(treated - control),
                "mean_estimated_uplift": float(work["estimated_uplift"].mean()),
                "observed_uplift_top_quartile": float(
                    top.loc[top["live_program_action"].eq(1), "future_response"].mean()
                    - top.loc[top["live_program_action"].eq(0), "future_response"].mean()
                ),
                "observed_uplift_bottom_quartile": float(
                    bottom.loc[
                        bottom["live_program_action"].eq(1), "future_response"
                    ].mean()
                    - bottom.loc[
                        bottom["live_program_action"].eq(0), "future_response"
                    ].mean()
                ),
            }
        ]
    )


def uplift_scatter_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-snapshot p0, p1, and estimated uplift for scatter visualization."""

    work = _score_uplift(panel)
    return work[
        ["response_if_no_action", "response_if_action", "estimated_uplift"]
    ].copy()


def uplift_ranking_comparison(
    panel: pd.DataFrame, top_fraction: float = 0.20
) -> pd.DataFrame:
    """Compare the rows selected by response ranking vs uplift ranking.

    Both rankings select the same number of rows. Response ranking fills its
    top slice with Sure Things (high p0, low uplift). Uplift ranking fills it
    with Persuadables (moderate p0, high uplift). The overlap is near zero.
    """

    work = _score_uplift(panel)
    n_select = int(round(top_fraction * len(work)))
    response_top = work.nlargest(n_select, "response_if_no_action")
    uplift_top = work.nlargest(n_select, "estimated_uplift")
    overlap = len(set(response_top.index) & set(uplift_top.index))
    return pd.DataFrame(
        [
            {
                "ranking": "response_ranked",
                "selected": n_select,
                "mean_baseline_response": response_top["response_if_no_action"].mean(),
                "mean_estimated_uplift": response_top["estimated_uplift"].mean(),
                "rows_shared_with_other_ranking": overlap,
            },
            {
                "ranking": "uplift_ranked",
                "selected": n_select,
                "mean_baseline_response": uplift_top["response_if_no_action"].mean(),
                "mean_estimated_uplift": uplift_top["estimated_uplift"].mean(),
                "rows_shared_with_other_ranking": overlap,
            },
        ]
    )


# --- Off-policy evaluation (IPS, self-normalized IPS, doubly robust) -----------


def _label_policies(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach logged action, logging probability, and candidate action."""

    work = panel.copy()
    work["logged_action"] = np.select(
        [
            work["field_frequency_30"].gt(0),
            work["email_frequency_30"].gt(0),
            work["live_program_attendance_180"].gt(0),
        ],
        ["Field", "Email", "Live program"],
        default="Observe",
    )
    work["logged_probability"] = work["logged_action"].map(LOGGED_ACTION_PROBABILITY)
    work["candidate_action"] = np.select(
        [
            work["access_resource_score"].gt(0.70),
            work["evidence_need_score"].gt(0.65),
            work["digital_response_rate"].gt(0.55),
        ],
        ["Account support", "Field", "Email"],
        default="Observe",
    )
    return work


def _reward_model(panel: pd.DataFrame) -> LogisticRegression:
    """Fit a reward model q(context, action) over all snapshots for DR correction."""

    work = _label_policies(panel)
    design = pd.concat(
        [
            work[UPLIFT_COVARIATES].reset_index(drop=True),
            pd.get_dummies(work["logged_action"]).reindex(
                columns=POLICY_ACTIONS, fill_value=0
            ).reset_index(drop=True),
        ],
        axis=1,
    )
    model = LogisticRegression(C=0.3, max_iter=1_000, random_state=20260622)
    model.fit(design, work["future_response"].reset_index(drop=True))
    return model


def _reward_predictions(
    model: LogisticRegression,
    frame: pd.DataFrame,
    action: pd.Series,
) -> np.ndarray:
    """Predicted reward for each row under a chosen action column."""

    design = pd.concat(
        [
            frame[UPLIFT_COVARIATES].reset_index(drop=True),
            pd.get_dummies(action.reset_index(drop=True)).reindex(
                columns=POLICY_ACTIONS, fill_value=0
            ),
        ],
        axis=1,
    )
    return model.predict_proba(design)[:, 1]


def off_policy_evaluation(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate the candidate policy value with IPS, self-normalized IPS, and DR."""

    latest = _label_policies(
        panel.loc[panel["snapshot_date"].eq(panel["snapshot_date"].max())]
    ).reset_index(drop=True)
    model = _reward_model(panel)
    n = len(latest)
    matched = latest["candidate_action"].eq(latest["logged_action"])
    weight = np.where(matched, 1.0 / latest["logged_probability"], 0.0)
    reward = latest["future_response"].to_numpy()

    logged_value = float(reward.mean())
    ips_value = float((weight * reward).sum() / n)
    snips_value = (
        float((weight * reward).sum() / weight.sum()) if weight.sum() else np.nan
    )
    effective_sample_size = (
        float((weight.sum() ** 2) / np.square(weight).sum())
        if np.square(weight).sum()
        else 0.0
    )
    q_candidate = _reward_predictions(model, latest, latest["candidate_action"])
    q_logged = _reward_predictions(model, latest, latest["logged_action"])
    dr_value = float((q_candidate + weight * (reward - q_logged)).mean())
    return pd.DataFrame(
        [
            {
                "policy": "logged_policy",
                "estimator": "on_policy_mean",
                "estimated_response_rate": logged_value,
                "matched_snapshots": n,
                "effective_sample_size": float(n),
            },
            {
                "policy": "candidate_policy",
                "estimator": "ips",
                "estimated_response_rate": ips_value,
                "matched_snapshots": int(matched.sum()),
                "effective_sample_size": effective_sample_size,
            },
            {
                "policy": "candidate_policy",
                "estimator": "snips",
                "estimated_response_rate": snips_value,
                "matched_snapshots": int(matched.sum()),
                "effective_sample_size": effective_sample_size,
            },
            {
                "policy": "candidate_policy",
                "estimator": "doubly_robust",
                "estimated_response_rate": dr_value,
                "matched_snapshots": int(matched.sum()),
                "effective_sample_size": effective_sample_size,
            },
        ]
    )


def off_policy_support(panel: pd.DataFrame) -> pd.DataFrame:
    """Show overlap between logged and candidate actions on the latest snapshot."""

    latest = _label_policies(
        panel.loc[panel["snapshot_date"].eq(panel["snapshot_date"].max())]
    )
    return (
        latest.groupby(["logged_action", "candidate_action"], as_index=False)
        .agg(
            snapshots=("npi", "size"),
            responses=("future_response", "sum"),
            response_rate=("future_response", "mean"),
        )
        .sort_values(["snapshots", "response_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )


# --- Sequence-derived features -------------------------------------------------


def _add_sequence_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the two order-aware features used by the sequence comparison."""

    work = panel.copy()
    work["field_then_digital"] = (
        work["last_channel"].isin(["Email", "Web"])
        & work["field_responses_90"].gt(0)
    ).astype(int)
    work["repeated_email"] = work["email_frequency_90"].ge(3).astype(int)
    return work


SEQUENCE_BASE_FEATURES = [
    "total_pressure_90",
    "shrunken_response_rate_90",
    "evidence_need_score",
    "access_resource_score",
]
SEQUENCE_ORDER_FEATURES = ["field_then_digital", "repeated_email"]


def field_then_digital_contrast(panel: pd.DataFrame) -> pd.DataFrame:
    """Show the planted order effect: a recent field response lifts a later digital touch."""

    digital = panel.loc[panel["last_channel"].isin(["Email", "Web"])].copy()
    digital["recent_field_response"] = np.where(
        digital["field_responses_90"].gt(0),
        "Field response in prior 90 days",
        "No recent field response",
    )
    return (
        digital.groupby("recent_field_response", as_index=False)
        .agg(
            snapshots=("npi", "size"),
            future_responses=("future_response", "sum"),
            future_response_rate=("future_response", "mean"),
        )
        .sort_values("future_response_rate", ascending=False)
        .reset_index(drop=True)
    )


def sequence_feature_model(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare aggregate features with aggregate-plus-sequence features."""

    work = _add_sequence_features(panel)
    train, test = train_test_split(work, test_size=0.30, shuffle=False)
    rows = []
    for name, features in [
        ("aggregate_only", SEQUENCE_BASE_FEATURES),
        ("aggregate_plus_sequence", SEQUENCE_BASE_FEATURES + SEQUENCE_ORDER_FEATURES),
    ]:
        model = LogisticRegression(C=0.2, max_iter=1_000, random_state=20260622)
        model.fit(train[features], train["future_response"])
        probability = model.predict_proba(test[features])[:, 1]
        rows.append(
            {
                "model": name,
                "test_snapshots": len(test),
                "roc_auc": roc_auc_score(test["future_response"], probability),
                "average_precision": average_precision_score(
                    test["future_response"], probability
                ),
            }
        )
    return pd.DataFrame(rows)


def sequence_feature_effects(panel: pd.DataFrame) -> pd.DataFrame:
    """Expose the sequence-feature coefficients and their support."""

    work = _add_sequence_features(panel)
    features = SEQUENCE_BASE_FEATURES + SEQUENCE_ORDER_FEATURES
    train, _ = train_test_split(work, test_size=0.30, shuffle=False)
    model = LogisticRegression(C=0.2, max_iter=1_000, random_state=20260622)
    model.fit(train[features], train["future_response"])
    rows = []
    for feature, coefficient in zip(features, model.coef_[0], strict=True):
        present = train[feature].gt(0)
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "odds_ratio": float(np.exp(coefficient)),
                "snapshots_with_feature": int(present.sum()),
                "response_rate_when_present": float(
                    train.loc[present, "future_response"].mean()
                )
                if present.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
