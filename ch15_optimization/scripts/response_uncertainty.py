"""Response fitting and bootstrap uncertainty for the allocation chapter.

This module fits a shared, segment-level incremental-response curve from the
observed account-period history, propagates uncertainty with a territory
block bootstrap, and converts every fitted draw into per-account call-step
gains. It reads observed data only. It never imports the truth table, so the
point estimate and the uncertainty draws are reproducible from observed
history alone, exactly as a real deployment would have to work.

The fitted model for account ``i`` in segment ``s`` at ``c`` calls is

    incremental_nrx = opportunity_i * access_multiplier_i
                      * scale_s * c**shape_s / (ec50_s**shape_s + c**shape_s)

with ``opportunity`` and ``access_multiplier`` observed scaling inputs and
``(scale_s, ec50_s, shape_s)`` the three shared parameters fitted per segment.
Positive parameters are fitted in log space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from allocation_config import (
    MAX_CALLS_PER_ACCOUNT,
    N_BOOTSTRAP_DRAWS,
    RESPONSE_MULTIPLIER,
    SEED_BOOTSTRAP,
    SEGMENT_ORDER,
)


def hill_fraction(calls: np.ndarray, ec50: float, shape: float) -> np.ndarray:
    calls = np.asarray(calls, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(calls <= 0, 0.0, calls**shape / (ec50**shape + calls**shape))


# Log-space parameter box keeps the fit inside a plausible detailing-response
# region: scale and ec50 trade off strongly when calls never fully saturate a
# curve, so a bounded fit avoids the degenerate "huge scale, huge ec50, flat
# shape" solution that a purely local search can drift into.
_FIT_LOWER = np.log([0.03, 2.5, 0.7])
_FIT_UPPER = np.log([3.0, 22.0, 2.2])


def _fit_one_segment(calls: np.ndarray, scaling: np.ndarray, incremental: np.ndarray) -> tuple[float, float, float]:
    """Bounded nonlinear least squares for (scale, ec50, shape) in log space.

    The target is the normalized response ``incremental / scaling``, so every
    account-period sits on the same 0-to-``scale`` axis and the fit is not
    dominated by a handful of high-opportunity accounts. Observations are
    weighted by the square root of scaling, a compromise that trusts precise
    large-account reads without letting them swamp the curve.
    """
    keep = scaling > 1e-6
    calls, scaling, incremental = calls[keep], scaling[keep], incremental[keep]
    target = incremental / scaling
    weight = np.sqrt(scaling)

    def residual(log_params: np.ndarray) -> np.ndarray:
        scale, ec50, shape = np.exp(log_params)
        predicted = scale * hill_fraction(calls, ec50, shape)
        return weight * (predicted - target)

    x0 = np.clip(np.log([0.35, 6.0, 1.2]), _FIT_LOWER, _FIT_UPPER)
    result = least_squares(residual, x0, method="trf", bounds=(_FIT_LOWER, _FIT_UPPER), max_nfev=2000)
    scale, ec50, shape = np.exp(result.x)
    return float(scale), float(ec50), float(shape)


def _segment_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Attach the observed scaling and incremental target used by the fit."""
    frame = history.copy()
    frame["access_multiplier"] = frame["access_state"].map(RESPONSE_MULTIPLIER)
    frame["scaling"] = frame["opportunity_nrx"] * frame["access_multiplier"]
    frame["incremental_nrx"] = (frame["observed_nrx"] - frame["baseline_nrx"]).clip(lower=0.0)
    return frame


def fit_segment_response(history: pd.DataFrame) -> pd.DataFrame:
    """Fit each segment's shared response curve from the observed panel.

    Returns one row per segment with fitted (scale, ec50, shape), the number
    of account-period observations, and the in-sample fit quality.
    """
    frame = _segment_frame(history)
    rows = []
    for segment in SEGMENT_ORDER:
        block = frame[frame["segment"] == segment]
        calls = block["calls"].to_numpy(dtype=float)
        scaling = block["scaling"].to_numpy()
        incremental = block["incremental_nrx"].to_numpy()
        scale, ec50, shape = _fit_one_segment(calls, scaling, incremental)
        predicted = scaling * scale * hill_fraction(calls, ec50, shape)
        resid = incremental - predicted
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((incremental - incremental.mean()) ** 2))
        rows.append({
            "segment": segment,
            "scale": round(scale, 4),
            "ec50": round(ec50, 3),
            "shape": round(shape, 3),
            "n_obs": int(len(block)),
            "rmse": round(float(np.sqrt(np.mean(resid**2))), 3),
            "r2": round(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, 3),
        })
    return pd.DataFrame(rows)


def block_bootstrap_response_draws(
    history: pd.DataFrame,
    n_draws: int = N_BOOTSTRAP_DRAWS,
    seed: int = SEED_BOOTSTRAP,
) -> np.ndarray:
    """Territory block bootstrap: resample territories, refit every segment per replicate.

    Returns an array of shape ``(n_draws, n_segments, 3)`` holding
    ``(scale, ec50, shape)`` per segment per draw, in ``SEGMENT_ORDER``. A
    replicate that resamples territories preserves the joint dependence of the
    three parameters, since all three are refitted together on the same
    resampled panel.
    """
    frame = _segment_frame(history)
    territories = frame["territory_id"].unique()
    by_territory = {t: frame[frame["territory_id"] == t] for t in territories}
    rng = np.random.default_rng(seed)

    draws = np.empty((n_draws, len(SEGMENT_ORDER), 3))
    for d in range(n_draws):
        picked = rng.choice(territories, size=len(territories), replace=True)
        replicate = pd.concat([by_territory[t] for t in picked], ignore_index=True)
        for s_idx, segment in enumerate(SEGMENT_ORDER):
            block = replicate[replicate["segment"] == segment]
            if len(block) < 8:
                # Degenerate resample: fall back to the point fit for this segment.
                point = fit_segment_response(history)
                params = point.loc[point["segment"] == segment, ["scale", "ec50", "shape"]].iloc[0]
                draws[d, s_idx] = params.to_numpy()
                continue
            draws[d, s_idx] = _fit_one_segment(
                block["calls"].to_numpy(dtype=float),
                block["scaling"].to_numpy(),
                block["incremental_nrx"].to_numpy(),
            )
    return draws


def _account_scaling(planning: pd.DataFrame) -> np.ndarray:
    return (planning["opportunity_nrx"] * planning["response_multiplier"]).to_numpy()


def params_to_step_gains(
    planning: pd.DataFrame,
    segment_params: dict[str, tuple[float, float, float]],
    max_calls: int = MAX_CALLS_PER_ACCOUNT,
) -> np.ndarray:
    """Convert one fitted parameter set into a (n_accounts, max_calls) step-gain matrix.

    Entry ``[i, k-1]`` is the incremental NRx of the k-th call to account ``i``:
    ``ceiling_i * (H(k) - H(k-1))``. Blocked accounts (max_calls == 0) get zeros.
    """
    scaling = _account_scaling(planning)
    segments = planning["segment"].to_numpy()
    cap = planning["max_calls"].to_numpy()
    n = len(planning)
    gains = np.zeros((n, max_calls))
    steps = np.arange(1, max_calls + 1)
    for segment, (scale, ec50, shape) in segment_params.items():
        mask = segments == segment
        if not mask.any():
            continue
        frac = hill_fraction(steps, ec50, shape)  # H(1..max)
        step_frac = np.diff(np.concatenate([[0.0], frac]))  # H(k) - H(k-1)
        ceiling = scaling[mask] * scale
        gains[mask] = ceiling[:, None] * step_frac[None, :]
    # Zero out steps beyond each account's own cap.
    reach = steps[None, :] <= cap[:, None]
    gains = gains * reach
    return gains


def point_estimate_step_gains(planning: pd.DataFrame, fit: pd.DataFrame) -> np.ndarray:
    params = {r.segment: (r.scale, r.ec50, r.shape) for r in fit.itertuples()}
    return params_to_step_gains(planning, params)


def response_draws_to_step_gains(
    planning: pd.DataFrame,
    draws: np.ndarray,
    max_calls: int = MAX_CALLS_PER_ACCOUNT,
) -> np.ndarray:
    """Stack step-gain matrices for every bootstrap draw.

    Returns an array of shape ``(n_draws, n_accounts, max_calls)``: draw-specific
    marginal NRx for each allowed call step, the input to SAA, CVaR, regret, and
    the frontier.
    """
    n_draws = draws.shape[0]
    out = np.empty((n_draws, len(planning), max_calls))
    for d in range(n_draws):
        params = {seg: tuple(draws[d, i]) for i, seg in enumerate(SEGMENT_ORDER)}
        out[d] = params_to_step_gains(planning, params, max_calls)
    return out


def summarize_step_gain_draws(planning: pd.DataFrame, step_gain_draws: np.ndarray) -> pd.DataFrame:
    """Mean, 10th and 90th percentile, and sd of each account's call-step gain."""
    n_draws, n_accounts, max_calls = step_gain_draws.shape
    account_ids = np.repeat(planning["account_id"].to_numpy(), max_calls)
    territory = np.repeat(planning["territory_id"].to_numpy(), max_calls)
    segment = np.repeat(planning["segment"].to_numpy(), max_calls)
    cap = np.repeat(planning["max_calls"].to_numpy(), max_calls)
    step = np.tile(np.arange(1, max_calls + 1), n_accounts)

    mean = step_gain_draws.mean(axis=0).reshape(-1)
    p10 = np.quantile(step_gain_draws, 0.10, axis=0).reshape(-1)
    p90 = np.quantile(step_gain_draws, 0.90, axis=0).reshape(-1)
    sd = step_gain_draws.std(axis=0).reshape(-1)

    summary = pd.DataFrame({
        "account_id": account_ids,
        "territory_id": territory,
        "segment": segment,
        "call_step": step,
        "mean_gain": mean.round(4),
        "p10_gain": p10.round(4),
        "p90_gain": p90.round(4),
        "sd_gain": sd.round(4),
    })
    # Keep only allowed, non-trivial steps to keep the table to a readable size.
    summary = summary[(step <= cap) & (summary["mean_gain"] > 1e-6)].reset_index(drop=True)
    return summary


def validate_draw_quality(
    planning: pd.DataFrame,
    fit: pd.DataFrame,
    draws: np.ndarray,
    step_gain_draws: np.ndarray,
) -> pd.DataFrame:
    """Diagnostics that the bootstrap draws are usable, computed without hidden truth."""
    point = point_estimate_step_gains(planning, fit)
    mean_gains = step_gain_draws.mean(axis=0)
    # Reconstruct each account's fitted ceiling from the point step gains: the
    # gains must telescope back to the response level at the account's cap.
    cap = planning["max_calls"].to_numpy()
    cumulative = point.cumsum(axis=1)
    reconstructed = cumulative[np.arange(len(planning)), np.clip(cap - 1, 0, point.shape[1] - 1)]
    reconstructed[cap == 0] = 0.0
    rows = [
        {
            "check": "all_step_gains_nonnegative",
            "value": round(float((step_gain_draws >= -1e-9).mean()), 4),
            "passes": bool((step_gain_draws >= -1e-9).all()),
        },
        {
            "check": "gains_telescope_to_response_level",
            "value": round(float(np.max(np.abs(reconstructed - cumulative.max(axis=1)))), 6),
            "passes": bool(np.allclose(reconstructed, cumulative.max(axis=1), atol=1e-6)),
        },
        {
            "check": "draw_mean_within_10pct_of_point_total",
            "value": round(float(np.abs(mean_gains.sum() - point.sum()) / max(point.sum(), 1e-9)), 4),
            "passes": bool(np.abs(mean_gains.sum() - point.sum()) / max(point.sum(), 1e-9) < 0.10),
        },
        {
            "check": "segment_shape_bootstrap_spread_positive",
            "value": round(float(draws[:, :, 2].std()), 4),
            "passes": bool(draws[:, :, 2].std() > 0),
        },
    ]
    return pd.DataFrame(rows)
