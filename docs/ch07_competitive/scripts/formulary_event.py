"""Controlled interrupted time series and changepoint monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _hac_covariance(X: np.ndarray, residuals: np.ndarray, lag: int = 4) -> np.ndarray:
    """Newey-West covariance for an OLS design matrix."""

    n = len(residuals)
    bread = np.linalg.inv(X.T @ X)
    scores = X * residuals[:, None]
    meat = scores.T @ scores
    for distance in range(1, min(lag, n - 1) + 1):
        weight = 1 - distance / (lag + 1)
        gamma = scores[distance:].T @ scores[:-distance]
        meat += weight * (gamma + gamma.T)
    return bread @ meat @ bread


def controlled_its(
    panel: pd.DataFrame,
    treated_payer: str,
    change_week: int,
    effect_week: int,
    hac_lag: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a centered controlled ITS with HAC standard errors."""

    pivot = panel.pivot(
        index="week", columns="payer_id", values="brand_share"
    ).sort_index()
    donors = [column for column in pivot.columns if column != treated_payer]
    control = pivot[donors].mean(axis=1)
    treated = pivot[treated_payer]
    week = pivot.index.to_numpy(dtype=float)
    centered = week - change_week
    post = (week >= change_week).astype(float)
    time_after = np.maximum(0, centered)
    X = np.column_stack(
        [np.ones(len(week)), centered, post, time_after, control.values]
    )
    y = treated.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    residuals = y - fitted
    covariance = _hac_covariance(X, residuals, lag=hac_lag)
    se = np.sqrt(np.diag(covariance))
    terms = [
        "intercept",
        "pretrend",
        "immediate_level",
        "slope_change",
        "control_share",
    ]
    coefficients = pd.DataFrame({"term": terms, "estimate": beta, "std_error_hac": se})
    coefficients["lower_95"] = (
        coefficients["estimate"] - 1.96 * coefficients["std_error_hac"]
    )
    coefficients["upper_95"] = (
        coefficients["estimate"] + 1.96 * coefficients["std_error_hac"]
    )

    X_counter = X.copy()
    X_counter[:, 2] = 0
    X_counter[:, 3] = 0
    counterfactual = X_counter @ beta
    fitted_table = pd.DataFrame(
        {
            "week": week.astype(int),
            "actual": y,
            "control_mean": control.values,
            "fitted": fitted,
            "counterfactual": counterfactual,
            "effect": fitted - counterfactual,
            "post": post.astype(bool),
        }
    )
    effect_row = fitted_table.loc[fitted_table["week"].eq(effect_week)].iloc[0]
    contrast = np.array([0, 0, 1, effect_week - change_week, 0], dtype=float)
    effect_se = float(np.sqrt(contrast @ covariance @ contrast))
    summary = pd.DataFrame(
        [
            {
                "change_week": change_week,
                "effect_week": effect_week,
                "immediate_effect": beta[2],
                "slope_change_per_week": beta[3],
                "effect_at_week": effect_row["effect"],
                "effect_at_week_se": effect_se,
                "effect_at_week_lower_95": effect_row["effect"] - 1.96 * effect_se,
                "effect_at_week_upper_95": effect_row["effect"] + 1.96 * effect_se,
                "pre_period_weeks": int((week < change_week).sum()),
                "post_period_weeks": int((week >= change_week).sum()),
            }
        ]
    )
    return coefficients, fitted_table, summary


def synthetic_control(
    panel: pd.DataFrame,
    treated_payer: str,
    change_week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit convex donor weights in the pre-period and report treated-control gaps."""

    pivot = panel.pivot(
        index="week", columns="payer_id", values="brand_share"
    ).sort_index()
    donor_names = [name for name in pivot.columns if name != treated_payer]
    y = pivot[treated_payer].to_numpy()
    D = pivot[donor_names].to_numpy()
    pre = pivot.index.to_numpy() < change_week

    def objective(weights: np.ndarray) -> float:
        residual = y[pre] - D[pre] @ weights
        return float(residual @ residual)

    initial = np.repeat(1 / len(donor_names), len(donor_names))
    fit = minimize(
        objective,
        initial,
        bounds=[(0, 1)] * len(donor_names),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
    )
    if not fit.success:
        raise RuntimeError(f"Synthetic-control fit failed: {fit.message}")
    weights = fit.x
    synthetic = D @ weights
    result = pd.DataFrame(
        {
            "week": pivot.index.to_numpy(dtype=int),
            "actual": y,
            "synthetic_control": synthetic,
            "gap": y - synthetic,
            "post": ~pre,
        }
    )
    diagnostics = pd.DataFrame(
        [
            {
                "pre_rmspe": float(np.sqrt(np.mean((y[pre] - synthetic[pre]) ** 2))),
                "post_mean_gap": float((y[~pre] - synthetic[~pre]).mean()),
                **{
                    f"weight_{name}": weight
                    for name, weight in zip(donor_names, weights, strict=True)
                },
            }
        ]
    )
    return result, diagnostics


def standardized_cusum(
    series: pd.Series,
    baseline_periods: int = 12,
    slack: float = 0.5,
    threshold: float = 4.0,
    recovery_z: float = 0.5,
    recovery_periods: int = 4,
) -> pd.DataFrame:
    """Open an episode at the first sustained shift and suppress repeats."""

    baseline = series.iloc[:baseline_periods]
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=1))
    if std <= 0:
        raise ValueError("CUSUM baseline standard deviation must be positive")
    positive = negative = 0.0
    increase_episode_open = False
    decrease_episode_open = False
    recovery_streak = 0
    rows: list[dict] = []
    for position, value in enumerate(series, start=1):
        z = (float(value) - mean) / std
        if abs(z) <= recovery_z:
            recovery_streak += 1
        else:
            recovery_streak = 0
        if recovery_streak >= recovery_periods:
            increase_episode_open = False
            decrease_episode_open = False

        positive = max(0.0, positive + z - slack)
        negative = min(0.0, negative + z + slack)
        increase_alarm = positive > threshold
        decrease_alarm = negative < -threshold
        if increase_alarm and not increase_episode_open:
            rows.append(
                {
                    "week": position,
                    "direction": "Increase",
                    "standardized_cusum": positive,
                    "episode_status": "Opened",
                }
            )
            increase_episode_open = True
            recovery_streak = 0
        elif decrease_alarm and not decrease_episode_open:
            rows.append(
                {
                    "week": position,
                    "direction": "Decrease",
                    "standardized_cusum": negative,
                    "episode_status": "Opened",
                }
            )
            decrease_episode_open = True
            recovery_streak = 0
        if increase_alarm or decrease_alarm:
            positive = negative = 0.0
    return pd.DataFrame(rows)
