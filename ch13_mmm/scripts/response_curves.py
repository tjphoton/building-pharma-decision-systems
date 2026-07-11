"""Response curve analysis for the marketing-mix-modeling chapter.

Builds posterior response curves, computes marginal ROI, and finds saturation
points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import exposure_column, exposure_to_spend, spend_to_exposure
from model import CHANNELS


# ── Local copies of transformation primitives (no sibling imports) ────────────

def _adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock transform: carries forward a fraction of prior spend."""
    ads = np.zeros_like(spend)
    ads[0] = spend[0]
    for t in range(1, len(spend)):
        ads[t] = spend[t] + decay * ads[t - 1]
    return ads


def _hill(x: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """Hill saturation function on raw spend units, mapping spend to response in [0, 1]."""
    return x ** slope / (ec50 ** slope + x ** slope)


# ── Response curve functions ──────────────────────────────────────────────────

def build_response_curves(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    """Compute spend vs. NRx response curves with posterior credible intervals.

    For each channel, sweeps spend from 0 to the larger of 2x observed mean
    spend and 1.25x observed max spend, computing the expected marginal NRx
    contribution across posterior draws.
    """
    channels = CHANNELS
    rows = []
    for ch in channels:
        exposure_obs = df[exposure_column(ch)].to_numpy(dtype=float)
        observed_mean_spend = exposure_to_spend(ch, float(exposure_obs.mean()))
        observed_max_spend = exposure_to_spend(ch, float(exposure_obs.max()))
        spend_ceiling = max(2 * observed_mean_spend, 1.25 * observed_max_spend)
        spend_grid = np.linspace(0, spend_ceiling, n_points)

        for spend_val in spend_grid:
            # Approximate: replace observed with a constant spend_val series
            exposure_val = spend_to_exposure(ch, float(spend_val))
            spend_const = np.full(len(exposure_obs), exposure_val)
            contrib_samples = []
            for j in range(min(500, len(draws))):  # subset of draws for speed
                row = draws.iloc[j]
                ads = _adstock(spend_const, float(row[f"{ch}_decay"]))
                sat = _hill(ads, float(row[f"{ch}_ec50"]), float(row[f"{ch}_slope"]))
                contrib = float(row[f"{ch}_coef"]) * sat.mean()
                contrib_samples.append(contrib)
            contrib_arr = np.array(contrib_samples)
            rows.append({
                "channel": ch,
                "weekly_spend": round(spend_val, 1),
                "weekly_input_value": round(exposure_val, 2),
                "input_unit": "calls" if ch == "field" else "dollars",
                "mean_nrx_contribution": round(float(contrib_arr.mean()), 2),
                "p10_nrx_contribution": round(float(np.percentile(contrib_arr, 10)), 2),
                "p90_nrx_contribution": round(float(np.percentile(contrib_arr, 90)), 2),
                "current_spend_flag": abs(spend_val - exposure_to_spend(ch, float(exposure_obs.mean()))) < (max(exposure_to_spend(ch, float(exposure_obs.mean())), 1.0) * 0.1),
            })
    return pd.DataFrame(rows)


def compute_marginal_roi(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    delta_spend: float = 1.0,
) -> pd.DataFrame:
    """Compute marginal ROI (NRx per $) at the current spend level for each channel."""
    channels = CHANNELS
    rows = []
    for ch in channels:
        exposure_obs = df[exposure_column(ch)].to_numpy(dtype=float)
        spend_mean = exposure_to_spend(ch, float(exposure_obs.mean()))
        exposure_mean = float(exposure_obs.mean())
        delta_exposure = spend_to_exposure(ch, delta_spend)
        spend_lo = np.full(len(exposure_obs), max(exposure_mean - delta_exposure / 2, 0))
        spend_hi = np.full(len(exposure_obs), exposure_mean + delta_exposure / 2)

        mroi_samples = []
        for j in range(min(500, len(draws))):
            row = draws.iloc[j]
            decay = float(row[f"{ch}_decay"])
            ec50 = float(row[f"{ch}_ec50"])
            slope = float(row[f"{ch}_slope"])
            coef = float(row[f"{ch}_coef"])

            ads_lo = _adstock(spend_lo, decay)
            ads_hi = _adstock(spend_hi, decay)
            contrib_lo = coef * _hill(ads_lo, ec50, slope).mean()
            contrib_hi = coef * _hill(ads_hi, ec50, slope).mean()
            mroi_samples.append((contrib_hi - contrib_lo) / delta_spend)

        mroi_arr = np.array(mroi_samples)
        rows.append({
            "channel": ch,
            "current_weekly_spend": round(spend_mean, 1),
            "current_input_value": round(exposure_mean, 2),
            "input_unit": "calls" if ch == "field" else "dollars",
            "marginal_roi_mean": round(float(mroi_arr.mean()), 4),
            "marginal_roi_p10": round(float(np.percentile(mroi_arr, 10)), 4),
            "marginal_roi_p90": round(float(np.percentile(mroi_arr, 90)), 4),
        })
    return pd.DataFrame(rows).sort_values("marginal_roi_mean", ascending=False)


def find_saturation_points(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Find the spend level at which each channel reaches `threshold` of its maximum response.

    Uses the posterior median rather than the mean: a rare draw with a
    near-zero slope sends the closed-form solve to an enormous value, and the
    median is robust to that single-draw blowup in a way the mean is not.
    """
    channels = CHANNELS
    rows = []
    for ch in channels:
        exposure_obs = df[exposure_column(ch)].to_numpy(dtype=float)
        sat_spends = []
        for j in range(min(300, len(draws))):
            row = draws.iloc[j]
            ec50 = float(row[f"{ch}_ec50"])
            slope = float(row[f"{ch}_slope"])
            decay = float(row[f"{ch}_decay"])
            # Solve hill(ads, ec50, slope) = threshold for the steady-state
            # adstock level: ads^s = threshold * ec50^s / (1 - threshold)
            ratio = threshold / (1 - threshold)
            ads_solution = (ratio * ec50 ** slope) ** (1 / slope)
            # A constant weekly input of x converges to ads = x / (1 - decay)
            # under geometric adstock, so invert that to get the raw input
            # level (calls or dollars) that produces this steady-state ads.
            x_solution = ads_solution * (1 - decay)
            if np.isfinite(x_solution):
                sat_spends.append(x_solution)
        sat_arr = np.array(sat_spends)
        rows.append({
            "channel": ch,
            "saturation_threshold_pct": int(threshold * 100),
            "saturation_spend_median": round(float(exposure_to_spend(ch, float(np.median(sat_arr)))), 1),
            "saturation_spend_p10": round(float(exposure_to_spend(ch, float(np.percentile(sat_arr, 10)))), 1),
            "saturation_spend_p90": round(float(exposure_to_spend(ch, float(np.percentile(sat_arr, 90)))), 1),
            "saturation_input_median": round(float(np.median(sat_arr)), 2),
            "current_weekly_spend": round(float(exposure_to_spend(ch, float(exposure_obs.mean()))), 1),
            "current_input_value": round(float(exposure_obs.mean()), 2),
            "input_unit": "calls" if ch == "field" else "dollars",
            "at_or_above_saturation": float(exposure_obs.mean()) >= float(np.median(sat_arr)),
        })
    return pd.DataFrame(rows)
