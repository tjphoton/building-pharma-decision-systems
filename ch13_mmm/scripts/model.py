"""Bayesian MMM model primitives and a multi-chain Metropolis-Hastings sampler.

Priors are set from category-level beliefs a brand analyst could plausibly
hold before looking at the 104-week series. They do not import the
ground-truth generating parameters from `data.py`. The only fact this module
is allowed to know in advance is the calendar week of the formulary access
event (`KNOWN_EVENT_WEEK`), because that date is an operational fact, not a
statistical unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import CHANNELS, KNOWN_EVENT_WEEK, RNG_SEED, exposure_column

# Category-level prior beliefs, deliberately not equal to the true generating
# values in data.py. Some are close, some are off, mirroring how well a
# practitioner's category intuition usually tracks reality.
PRIOR_DECAY = {
    "field": (0.30, 0.15),        # personal selling: moderate, assumed carryover
    "email": (0.15, 0.10),        # short shelf life
    "digital": (0.20, 0.12),      # assumed short; true value is shorter still
    "paid_media": (0.55, 0.15),   # brand advertising: assumed longer memory
}
PRIOR_SLOPE = (1.5, 0.6)          # generic S-curve steepness guess, same for all channels
PRIOR_COEF = (60.0, 50.0)         # generic magnitude guess, same for all channels


@dataclass(frozen=True)
class MMMPriors:
    """The tunable surface of this model: category-level prior beliefs.

    Where a regression MMM or Robyn would grid-search adstock decay and
    saturation shape outside the fit, this model samples them jointly in the
    posterior, guided by these priors instead. Passing a different
    `MMMPriors` instance to `fit_bayesian_mmm()` is how the prior-sensitivity
    refit (`build_prior_sensitivity()` in `run_analysis.py`) reruns the
    sampler under a shifted belief without touching the model's structure.
    """

    decay: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(PRIOR_DECAY))
    slope: tuple[float, float] = PRIOR_SLOPE
    coef: tuple[float, float] = PRIOR_COEF


DEFAULT_PRIORS = MMMPriors()


def _adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock transform: carries forward a fraction of prior spend."""
    ads = np.zeros_like(spend)
    ads[0] = spend[0]
    for t in range(1, len(spend)):
        ads[t] = spend[t] + decay * ads[t - 1]
    return ads


def _hill(x: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """Hill saturation function on raw spend units. x and ec50 share units."""
    x = np.clip(x, 0, None)
    return x ** slope / (ec50 ** slope + x ** slope)


def _seasonal_basis(week_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sine/cosine pair at a 52-week period. Linear-in-parameters avoids phase wrapping."""
    angle = 2 * np.pi * week_index / 52
    return np.sin(angle), np.cos(angle)


def _param_layout(use_controls: bool) -> dict[str, int]:
    """Index of each scalar parameter within the flat parameter vector."""
    idx: dict[str, int] = {"baseline0": 0, "noise_sd": 1}
    offset = 2
    if use_controls:
        for name in ("trend", "seasonal_sin", "seasonal_cos", "event_size"):
            idx[name] = offset
            offset += 1
    for ch in CHANNELS:
        for param in ("coef", "decay", "ec50", "slope"):
            idx[f"{ch}_{param}"] = offset
            offset += 1
    return idx


def _mean_baseline_prior(df: pd.DataFrame) -> float:
    """A generic guess at pre-promotion NRx: the observed minimum smoothed level."""
    return float(df["nrx"].rolling(8, min_periods=1).mean().min())


def _log_likelihood(
    params_vec: np.ndarray,
    idx: dict[str, int],
    use_controls: bool,
    week_index: np.ndarray,
    spend_mat: dict[str, np.ndarray],
    nrx: np.ndarray,
) -> float:
    """Gaussian log-likelihood for the MMM given a parameter vector."""
    baseline0 = params_vec[idx["baseline0"]]
    noise_sd = params_vec[idx["noise_sd"]]
    if noise_sd <= 0:
        return -1e12

    mu = np.full(len(nrx), baseline0)
    if use_controls:
        trend = params_vec[idx["trend"]]
        sin_coef = params_vec[idx["seasonal_sin"]]
        cos_coef = params_vec[idx["seasonal_cos"]]
        event_size = params_vec[idx["event_size"]]
        sin_basis, cos_basis = _seasonal_basis(week_index)
        event_indicator = (week_index >= KNOWN_EVENT_WEEK).astype(float)
        mu = mu + trend * week_index + sin_coef * sin_basis + cos_coef * cos_basis + event_size * event_indicator

    for ch in CHANNELS:
        coef = params_vec[idx[f"{ch}_coef"]]
        decay = params_vec[idx[f"{ch}_decay"]]
        ec50 = params_vec[idx[f"{ch}_ec50"]]
        slope = params_vec[idx[f"{ch}_slope"]]
        if not (0 < decay < 1 and ec50 > 0 and slope > 0 and coef > 0):
            return -1e12
        ads = _adstock(spend_mat[ch], decay)
        mu = mu + coef * _hill(ads, ec50, slope)

    resid = nrx - mu
    return float(-0.5 * np.sum((resid / noise_sd) ** 2) - len(nrx) * np.log(noise_sd))


def _log_prior(
    params_vec: np.ndarray,
    idx: dict[str, int],
    use_controls: bool,
    baseline_prior_mean: float,
    spend_means: dict[str, float],
    priors: MMMPriors,
) -> float:
    """Log-prior over MMM parameters, set from category-level belief only."""
    lp = 0.0
    baseline0 = params_vec[idx["baseline0"]]
    noise_sd = params_vec[idx["noise_sd"]]
    lp += -0.5 * ((baseline0 - baseline_prior_mean) / 25) ** 2
    lp += -0.5 * ((np.log(max(noise_sd, 1e-6)) - np.log(10)) / 0.7) ** 2

    if use_controls:
        lp += -0.5 * (params_vec[idx["trend"]] / 0.5) ** 2
        lp += -0.5 * (params_vec[idx["seasonal_sin"]] / 15) ** 2
        lp += -0.5 * (params_vec[idx["seasonal_cos"]] / 15) ** 2
        lp += -0.5 * (params_vec[idx["event_size"]] / 20) ** 2

    slope_mean, slope_sd = priors.slope
    coef_mean, coef_sd = priors.coef
    for ch in CHANNELS:
        decay_mean, decay_sd = priors.decay[ch]
        ec50_mean = spend_means[ch]
        ec50_sd = spend_means[ch] * 0.6
        lp += -0.5 * ((params_vec[idx[f"{ch}_coef"]] - coef_mean) / coef_sd) ** 2
        lp += -0.5 * ((params_vec[idx[f"{ch}_decay"]] - decay_mean) / decay_sd) ** 2
        lp += -0.5 * ((params_vec[idx[f"{ch}_ec50"]] - ec50_mean) / ec50_sd) ** 2
        lp += -0.5 * ((params_vec[idx[f"{ch}_slope"]] - slope_mean) / slope_sd) ** 2
    return lp


def implied_field_experiment_prior(
    control_mean: float,
    adjusted_effect: float,
    effect_se: float,
    baseline_weekly_nrx: float,
    averaging_weeks: int = 52,
) -> dict[str, float]:
    """Translate the experiment into a prior on field's mean weekly contribution.

    The experiment identifies a relative lift on a controlled baseline. We map
    that lift onto the chapter's own baseline weekly NRx scale, then shrink the
    standard error to the uncertainty of a weekly average over one seasonal
    cycle. This keeps the prior independent of the controlled MMM fit while
    staying on visible NRx units.
    """
    relative_lift = adjusted_effect / control_mean
    relative_lift_se = effect_se / control_mean
    return {
        "channel": "field",
        "mean": baseline_weekly_nrx * relative_lift,
        "sd": max(baseline_weekly_nrx * relative_lift_se / np.sqrt(averaging_weeks), 1e-6),
        "baseline_weekly_nrx": baseline_weekly_nrx,
        "relative_lift": relative_lift,
        "relative_lift_se": relative_lift_se,
        "averaging_weeks": averaging_weeks,
    }


def _geo_prior_penalty(
    params_vec: np.ndarray,
    idx: dict[str, int],
    spend_mat: dict[str, np.ndarray],
    geo_prior: dict[str, float],
) -> float:
    """Extra log-prior term anchoring one channel to an experiment-derived read."""
    ch = geo_prior["channel"]
    coef = params_vec[idx[f"{ch}_coef"]]
    decay = params_vec[idx[f"{ch}_decay"]]
    ec50 = params_vec[idx[f"{ch}_ec50"]]
    slope = params_vec[idx[f"{ch}_slope"]]
    if not (0 < decay < 1 and ec50 > 0 and slope > 0 and coef > 0):
        return -1e12
    if {"input_level", "delta_input", "mean_incremental_nrx", "sd_incremental_nrx"}.issubset(geo_prior):
        input_level = float(geo_prior["input_level"])
        delta_input = float(geo_prior["delta_input"])
        lo = np.full(20, max(input_level - delta_input / 2, 0.0))
        hi = np.full(20, input_level + delta_input / 2)
        incr = coef * (_hill(_adstock(hi, decay), ec50, slope).mean() - _hill(_adstock(lo, decay), ec50, slope).mean())
        return -0.5 * ((incr - geo_prior["mean_incremental_nrx"]) / geo_prior["sd_incremental_nrx"]) ** 2
    ads = _adstock(spend_mat[ch], decay)
    contribution = coef * _hill(ads, ec50, slope).mean()
    return -0.5 * ((contribution - geo_prior["mean"]) / geo_prior["sd"]) ** 2


def _mean_prediction_for_row(
    row: pd.Series,
    use_controls: bool,
    week_index: np.ndarray,
    spend_mat: dict[str, np.ndarray],
) -> np.ndarray:
    """Expected NRx path for one posterior draw."""
    mu = np.full(len(week_index), float(row["baseline0"]))
    if use_controls:
        sin_basis, cos_basis = _seasonal_basis(week_index)
        event_indicator = (week_index >= KNOWN_EVENT_WEEK).astype(float)
        mu = (
            mu
            + float(row["trend"]) * week_index
            + float(row["seasonal_sin"]) * sin_basis
            + float(row["seasonal_cos"]) * cos_basis
            + float(row["event_size"]) * event_indicator
        )
    for ch in CHANNELS:
        ads = _adstock(spend_mat[ch], float(row[f"{ch}_decay"]))
        mu = mu + float(row[f"{ch}_coef"]) * _hill(ads, float(row[f"{ch}_ec50"]), float(row[f"{ch}_slope"]))
    return mu


def posterior_mean_prediction(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    n_draws: int = 500,
) -> np.ndarray:
    """Posterior-mean expected NRx for each week in `df`."""
    use_controls = bool(draws.attrs.get("use_controls", True))
    week_index = df["week_index"].to_numpy(dtype=float)
    spend_mat = {ch: df[exposure_column(ch)].to_numpy(dtype=float) for ch in CHANNELS}
    preds = np.zeros(len(df))
    use_n = min(n_draws, len(draws))
    for j in range(use_n):
        preds += _mean_prediction_for_row(draws.iloc[j], use_controls, week_index, spend_mat)
    return preds / use_n


def _init_from_prior(
    idx: dict[str, int],
    use_controls: bool,
    baseline_prior_mean: float,
    spend_means: dict[str, float],
    rng: np.random.Generator,
    priors: MMMPriors,
) -> np.ndarray:
    """Draw a starting point from the prior, never from the truth."""
    vec = np.zeros(len(idx))
    vec[idx["baseline0"]] = baseline_prior_mean + rng.normal(0, 10)
    vec[idx["noise_sd"]] = 10.0
    if use_controls:
        vec[idx["trend"]] = 0.0
        vec[idx["seasonal_sin"]] = 0.0
        vec[idx["seasonal_cos"]] = 0.0
        vec[idx["event_size"]] = 0.0
    for ch in CHANNELS:
        decay_mean, _ = priors.decay[ch]
        vec[idx[f"{ch}_coef"]] = max(priors.coef[0] + rng.normal(0, 10), 5.0)
        vec[idx[f"{ch}_decay"]] = float(np.clip(decay_mean, 0.02, 0.98))
        vec[idx[f"{ch}_ec50"]] = max(spend_means[ch], 5.0)
        vec[idx[f"{ch}_slope"]] = max(priors.slope[0], 0.3)
    return vec


def _step_sizes(idx: dict[str, int], use_controls: bool, spend_means: dict[str, float]) -> np.ndarray:
    step = np.zeros(len(idx))
    step[idx["baseline0"]] = 4.0
    step[idx["noise_sd"]] = 0.8
    if use_controls:
        step[idx["trend"]] = 0.03
        step[idx["seasonal_sin"]] = 1.5
        step[idx["seasonal_cos"]] = 1.5
        step[idx["event_size"]] = 2.0
    for ch in CHANNELS:
        step[idx[f"{ch}_coef"]] = 8.0
        step[idx[f"{ch}_decay"]] = 0.05
        step[idx[f"{ch}_ec50"]] = max(spend_means[ch] * 0.15, 0.5)
        step[idx[f"{ch}_slope"]] = 0.2
    return step


def _run_one_chain(
    idx: dict[str, int],
    use_controls: bool,
    week_index: np.ndarray,
    spend_mat: dict[str, np.ndarray],
    nrx: np.ndarray,
    baseline_prior_mean: float,
    spend_means: dict[str, float],
    geo_prior: dict[str, float] | None,
    n_samples: int,
    warmup: int,
    seed: int,
    priors: MMMPriors,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    current = _init_from_prior(idx, use_controls, baseline_prior_mean, spend_means, rng, priors)
    step = _step_sizes(idx, use_controls, spend_means)

    def log_posterior(v: np.ndarray) -> float:
        ll = _log_likelihood(v, idx, use_controls, week_index, spend_mat, nrx)
        lp = _log_prior(v, idx, use_controls, baseline_prior_mean, spend_means, priors)
        if geo_prior is not None:
            lp += _geo_prior_penalty(v, idx, spend_mat, geo_prior)
        return ll + lp

    current_lp = log_posterior(current)
    total_iters = warmup + n_samples
    samples = np.zeros((n_samples, len(idx)))
    n_accept = 0
    scale = 1.0
    block_accept = 0
    block_size = 50
    for i in range(total_iters):
        proposal = current + rng.normal(0, step * scale)
        proposal_lp = log_posterior(proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current = proposal
            current_lp = proposal_lp
            n_accept += 1
            block_accept += 1
        if i < warmup and (i + 1) % block_size == 0:
            block_rate = block_accept / block_size
            if block_rate < 0.15:
                scale *= 0.8
            elif block_rate > 0.40:
                scale *= 1.25
            block_accept = 0
        if i >= warmup:
            samples[i - warmup] = current
    return samples, n_accept / total_iters


def _split_rhat_ess(chains: np.ndarray) -> tuple[float, float]:
    """Gelman-Rubin R-hat and a simple autocorrelation-based ESS for one parameter.

    `chains` has shape (n_chains, n_samples).
    """
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    between = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    within = chain_vars.mean()
    var_hat = ((n - 1) / n) * within + between / n
    rhat = float(np.sqrt(var_hat / within)) if within > 0 else float("nan")

    pooled = chains.reshape(-1)
    pooled = pooled - pooled.mean()
    max_lag = min(50, n - 1)
    autocorr_sum = 0.0
    denom = np.sum(pooled ** 2)
    for lag in range(1, max_lag):
        acf = np.sum(pooled[:-lag] * pooled[lag:]) / denom if denom > 0 else 0.0
        if acf < 0.05:
            break
        autocorr_sum += acf
    ess = float(m * n / (1 + 2 * autocorr_sum))
    return rhat, ess


def fit_bayesian_mmm(
    df: pd.DataFrame,
    n_samples: int = 16_000,
    warmup: int = 12_000,
    seed: int = RNG_SEED,
    n_chains: int = 4,
    use_controls: bool = True,
    geo_prior: dict[str, float] | None = None,
    priors: MMMPriors | None = None,
) -> pd.DataFrame:
    """Fit MMM with a multi-chain random-walk Metropolis-Hastings sampler.

    `priors` defaults to `DEFAULT_PRIORS`, the chapter's own category-level
    beliefs. Passing a different `MMMPriors` reruns the identical sampler
    under a shifted belief, which is how `build_prior_sensitivity()` in
    `run_analysis.py` tests whether a channel's answer belongs to the data
    or to the prior, instead of grid-searching hyperparameters outside the fit.

    Returns a DataFrame of posterior draws with one row per accepted sample
    per chain (a `chain` column identifies the source chain). Diagnostics
    (acceptance rate, split R-hat, effective sample size per parameter) are
    attached via `draws.attrs`.
    """
    priors = priors if priors is not None else DEFAULT_PRIORS
    idx = _param_layout(use_controls)
    week_index = df["week_index"].to_numpy(dtype=float)
    spend_mat = {ch: df[exposure_column(ch)].to_numpy(dtype=float) for ch in CHANNELS}
    nrx = df["nrx"].to_numpy(dtype=float)
    baseline_prior_mean = _mean_baseline_prior(df)
    spend_means = {ch: float(spend_mat[ch].mean()) for ch in CHANNELS}

    all_samples = []
    acceptance_rates = []
    for c in range(n_chains):
        samples, rate = _run_one_chain(
            idx, use_controls, week_index, spend_mat, nrx,
            baseline_prior_mean, spend_means, geo_prior,
            n_samples, warmup, seed=seed + 1000 * c, priors=priors,
        )
        all_samples.append(samples)
        acceptance_rates.append(rate)

    col_names = sorted(idx, key=idx.get)
    rows = []
    for c, samples in enumerate(all_samples):
        chain_df = pd.DataFrame(samples, columns=col_names)
        chain_df.insert(0, "chain", c)
        rows.append(chain_df)
    draws = pd.concat(rows, ignore_index=True)

    if not use_controls:
        for name in ("trend", "seasonal_sin", "seasonal_cos", "event_size"):
            draws[name] = 0.0

    stacked = np.stack(all_samples)  # (n_chains, n_samples, n_params)
    rhat = {}
    ess = {}
    for j, name in enumerate(col_names):
        rhat[name], ess[name] = _split_rhat_ess(stacked[:, :, j])

    # Interleave chains so any downstream `.iloc[:k]` subsample draws from all of them.
    draws = draws.sample(frac=1, random_state=seed).reset_index(drop=True)

    draws.attrs["acceptance_rate"] = float(np.mean(acceptance_rates))
    draws.attrs["acceptance_rate_by_chain"] = acceptance_rates
    draws.attrs["rhat"] = rhat
    draws.attrs["ess"] = ess
    draws.attrs["use_controls"] = use_controls
    return draws


def posterior_summary(draws: pd.DataFrame, true_lookup: dict[str, float] | None = None) -> pd.DataFrame:
    """Summarise posterior means, 90% credible intervals, and diagnostics."""
    rhat = draws.attrs.get("rhat", {})
    ess = draws.attrs.get("ess", {})
    rows = []
    for col in draws.columns:
        if col == "chain":
            continue
        vals = draws[col].to_numpy()
        true_val = true_lookup.get(col) if true_lookup else None
        rows.append({
            "parameter": col,
            "posterior_mean": round(float(vals.mean()), 3),
            "posterior_sd": round(float(vals.std()), 3),
            "p5": round(float(np.percentile(vals, 5)), 3),
            "p95": round(float(np.percentile(vals, 95)), 3),
            "true_value": true_val,
            "rhat": round(rhat.get(col, float("nan")), 3),
            "ess": round(ess.get(col, float("nan")), 0),
        })
    return pd.DataFrame(rows)
