"""Temporal response modeling for Chapter 8."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "total_pressure_30",
    "total_pressure_90",
    "days_since_response",
    "field_responses_90",
    "email_clicks_90",
    "web_actions_90",
    "paid_clicks_90",
    "live_program_attendance_180",
    "account_support_resolutions_90",
    "funnel_engagements_90",
    "evidence_need_score",
    "access_resource_score",
    "digital_response_rate",
    "field_response_rate",
    "shrunken_response_rate_90",
    "decayed_total_pressure_90",
    "decayed_response_90",
    "review_opportunity",
]

CATEGORICAL_FEATURES = [
    "last_response_channel",
]


def temporal_split(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split snapshots into earlier training and later evaluation periods."""

    return {
        "train": panel.loc[
            panel["snapshot_date"].le(pd.Timestamp("2024-11-30"))
        ].copy(),
        "validation": panel.loc[
            panel["snapshot_date"].eq(pd.Timestamp("2024-12-31"))
        ].copy(),
        "test": panel.loc[panel["snapshot_date"].ge("2025-01-31")].copy(),
    }


def _pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    max_iter=1_000,
                    random_state=20260622,
                ),
            ),
        ]
    )


def fit_response_model(
    panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit on earlier snapshots and evaluate on later snapshots."""

    splits = temporal_split(panel)
    model = _pipeline()
    model.fit(
        splits["train"][NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        splits["train"]["future_response"],
    )
    scored_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    train_rate = float(splits["train"]["future_response"].mean())
    for split_name, frame in splits.items():
        scored = frame.copy()
        scored["raw_predicted_response"] = model.predict_proba(
            frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )[:, 1]
        scored["predicted_response"] = scored["raw_predicted_response"]
        scored["split"] = split_name
        scored_parts.append(scored)
        y = scored["future_response"]
        probability = scored["predicted_response"]
        metrics_rows.append(
            {
                "split": split_name,
                "snapshots": len(scored),
                "responses": int(y.sum()),
                "response_rate": y.mean(),
                "roc_auc": roc_auc_score(y, probability),
                "average_precision": average_precision_score(y, probability),
                "brier_score": brier_score_loss(y, probability),
                "base_rate_brier": brier_score_loss(
                    y, np.repeat(train_rate, len(y))
                ),
            }
        )

    scored_panel = pd.concat(scored_parts, ignore_index=True)
    test = scored_panel.loc[scored_panel["split"].eq("test")].copy()
    calibration = _calibration_table(test)
    lift = _lift_table(test)
    coefficients = _coefficient_table(model)
    model_card = pd.DataFrame(
        [
            {
                "model_version": "ch08-logit-v1.0",
                "training_end": splits["train"]["snapshot_date"].max(),
                "validation_date": splits["validation"]["snapshot_date"].max(),
                "test_start": splits["test"]["snapshot_date"].min(),
                "test_end": splits["test"]["snapshot_date"].max(),
                "numeric_features": len(NUMERIC_FEATURES),
                "categorical_features": len(CATEGORICAL_FEATURES),
            }
        ]
    )
    return {
        "scored_snapshots": scored_panel,
        "model_metrics": pd.DataFrame(metrics_rows),
        "calibration": calibration,
        "lift": lift,
        "model_coefficients": coefficients,
        "model_card": model_card,
        "leakage_check": leakage_check(panel),
    }


def leakage_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Quantify how a same-window response feature inflates apparent fit."""

    splits = temporal_split(panel)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    leaked_feature = "future_response_count"
    model = _pipeline()
    model.fit(
        splits["train"][feature_columns],
        splits["train"]["future_response"],
    )
    train_probability = model.predict_proba(splits["train"][feature_columns])[:, 1]
    test_probability = model.predict_proba(splits["test"][feature_columns])[:, 1]

    leaked = _pipeline()
    leaked_numeric = NUMERIC_FEATURES + [leaked_feature]
    leaked_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), leaked_numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    leaked.set_params(preprocess=leaked_preprocessor)
    leaked_features = leaked_numeric + CATEGORICAL_FEATURES
    leaked.fit(
        splits["train"][leaked_features],
        splits["train"]["future_response"],
    )
    leaked_train_probability = leaked.predict_proba(
        splits["train"][leaked_features]
    )[:, 1]
    leaked_test_probability = leaked.predict_proba(
        splits["test"][leaked_features]
    )[:, 1]
    return pd.DataFrame(
        [
            {
                "model": "past_only",
                "train_auc": roc_auc_score(
                    splits["train"]["future_response"], train_probability
                ),
                "test_auc": roc_auc_score(
                    splits["test"]["future_response"], test_probability
                ),
            },
            {
                "model": "same_window_leak",
                "train_auc": roc_auc_score(
                    splits["train"]["future_response"], leaked_train_probability
                ),
                "test_auc": roc_auc_score(
                    splits["test"]["future_response"], leaked_test_probability
                ),
            },
        ]
    )


def _calibration_table(test: pd.DataFrame) -> pd.DataFrame:
    work = test.copy()
    work["probability_bin"] = pd.qcut(
        work["predicted_response"],
        q=5,
        duplicates="drop",
    )
    result = (
        work.groupby("probability_bin", observed=True, as_index=False)
        .agg(
            snapshots=("npi", "size"),
            mean_predicted=("predicted_response", "mean"),
            observed_rate=("future_response", "mean"),
        )
    )
    result["bin_order"] = np.arange(1, len(result) + 1)
    return result[
        ["bin_order", "snapshots", "mean_predicted", "observed_rate"]
    ]


def _lift_table(test: pd.DataFrame) -> pd.DataFrame:
    work = test.sort_values("predicted_response", ascending=False).copy()
    work["quintile"] = pd.qcut(
        np.arange(len(work)),
        q=5,
        labels=["Q1 highest", "Q2", "Q3", "Q4", "Q5 lowest"],
    )
    overall = work["future_response"].mean()
    result = (
        work.groupby("quintile", observed=True, as_index=False)
        .agg(
            snapshots=("npi", "size"),
            responses=("future_response", "sum"),
            response_rate=("future_response", "mean"),
            mean_predicted=("predicted_response", "mean"),
        )
    )
    result["lift_vs_test_average"] = result["response_rate"] / overall
    result["quintile_order"] = np.arange(1, len(result) + 1)
    return result


def _coefficient_table(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    result = pd.DataFrame(
        {
            "feature": [
                name.replace("numeric__", "").replace("categorical__", "")
                for name in feature_names
            ],
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    return result.sort_values(
        "coefficient", ascending=False
    ).reset_index(drop=True)
