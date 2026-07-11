"""Entry point for the marketing-mix-modeling and unified measurement chapter.

Fits three versions of the model on the same 104-week series: naive (no
baseline controls), controlled (trend, seasonality, and the formulary-event
control added), and calibrated (controlled plus an experiment-anchored prior
on the field coefficient). Scores all three against the known ground truth,
builds response curves and a budget optimization off the calibrated model,
and writes the reconciliation table that compares field's attribution
credit, experiment lift, and MMM contribution share side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from data import (  # noqa: E402
    DIGITAL_PRE_EVENT_TEST_WEEKS,
    KNOWN_EVENT_WEEK,
    PAID_MEDIA_DARK_WEEKS,
    _TRUE_PARAMS,
    exposure_column,
    generate_field_geo_holdout,
    generate_mmm_data,
    true_channel_share,
)
from model import (  # noqa: E402
    CHANNELS,
    PRIOR_COEF,
    MMMPriors,
    _adstock,
    _hill,
    _mean_prediction_for_row,
    fit_bayesian_mmm,
    implied_field_experiment_prior,
    posterior_mean_prediction,
    posterior_summary,
)
from optimization import (  # noqa: E402
    DECISION_READY_INCREASE_CAP,
    DIRECTIONAL_BAND,
    WEAK_DIRECTIONAL_BAND,
    build_mmm_budget_recommendation,
    build_unified_budget_recommendation,
    budget_optimisation,
    evaluate_reallocation,
    optimal_allocation_at_budget,
    optimal_allocation_by_draw,
)
from figure_system_map import write_system_map  # noqa: E402
from response_curves import (  # noqa: E402
    build_response_curves,
    compute_marginal_roi,
    find_saturation_points,
)

ROOT = Path(__file__).resolve().parents[2]
CH08_OUTPUT_DIR = ROOT / "ch08_omnichannel" / "assets" / "generated_outputs"
CH10_OUTPUT_DIR = ROOT / "ch10_experiments" / "assets" / "generated_outputs"
CH11_OUTPUT_DIR = ROOT / "ch11_natural_experiments" / "assets" / "generated_outputs"

# Gate thresholds for the model-health and channel-identification checks in
# `measurement_decision_record()`. These are illustrative, category-level
# thresholds (not tuned to this dataset's answer), the same spirit as the
# category-level priors in model.py. None of the 3 checks reads
# `true_channel_share()`, so the gate stays usable on real data, where that
# ground truth never exists.
GATE_MAX_RHAT_DECISION_READY = 1.20
GATE_MAX_CONFOUND_CORR_DECISION_READY = 0.60
GATE_MIN_CV_DECISION_READY = 0.15
GATE_MIN_CV_NOT_USABLE = 0.05
GATE_MAX_CONFOUND_CORR_NOT_USABLE = 0.95
GATE_MAX_RHAT_NOT_USABLE = 1.50

CHANNEL_PALETTE = {
    "field": "#2F6B9A",
    "email": "#2A9D8F",
    "digital": "#7A68A6",
    "paid_media": "#C77D2B",
}
FIT_PALETTE = {
    "naive": "#E69F00",
    "controlled": "#0072B2",
    "calibrated": "#009E73",
}


def _channel_contributions(df: pd.DataFrame, draws: pd.DataFrame) -> dict[str, float]:
    """Posterior-mean weekly NRx contribution per channel, at posterior-mean parameters."""
    spend_mat = {ch: df[exposure_column(ch)].to_numpy(dtype=float) for ch in CHANNELS}
    contributions = {}
    for ch in CHANNELS:
        coef = float(draws[f"{ch}_coef"].mean())
        decay = float(draws[f"{ch}_decay"].mean())
        ec50 = float(draws[f"{ch}_ec50"].mean())
        slope = float(draws[f"{ch}_slope"].mean())
        ads = _adstock(spend_mat[ch], decay)
        contributions[ch] = coef * float(_hill(ads, ec50, slope).mean())
    return contributions


def load_cross_chapter_measurements() -> dict[str, float]:
    """Read the attribution and experiment numbers from the upstream chapter outputs."""
    adjusted = pd.read_csv(CH10_OUTPUT_DIR / "adjusted_itt.csv").iloc[0]
    crude = pd.read_csv(CH10_OUTPUT_DIR / "crude_itt.csv").iloc[0]
    markov = pd.read_csv(CH08_OUTPUT_DIR / "markov_attribution.csv")
    did = pd.read_csv(CH11_OUTPUT_DIR / "did.csv")
    its = pd.read_csv(CH11_OUTPUT_DIR / "its_summary.csv").iloc[0]
    field_markov = markov.loc[markov["channel"] == "Field"].iloc[0]
    email_markov = markov.loc[markov["channel"] == "Email"].iloc[0]
    web_markov = markov.loc[markov["channel"] == "Web"].iloc[0]
    paid_markov = markov.loc[markov["channel"] == "Paid media"].iloc[0]
    did_effect = float(did.loc[did["quantity"] == "DiD effect (treated x post)", "value"].iloc[0])
    did_se = float(did.loc[did["quantity"] == "DiD standard error", "value"].iloc[0])
    return {
        "field_experiment_control_mean": float(crude["control_mean"]),
        "field_experiment_adjusted_effect": float(adjusted["effect"]),
        "field_experiment_crude_se": float(crude["standard_error"]),
        "field_experiment_relative_lift": float(adjusted["effect"] / crude["control_mean"]),
        "field_experiment_relative_lift_se": float(crude["standard_error"] / crude["control_mean"]),
        "field_attribution_markov_credit": float(field_markov["markov_credit"]) / 100.0,
        "email_attribution_markov_credit": float(email_markov["markov_credit"]) / 100.0,
        "digital_proxy_attribution_markov_credit": float(web_markov["markov_credit"]) / 100.0,
        "paid_media_attribution_markov_credit": float(paid_markov["markov_credit"]) / 100.0,
        "natural_experiment_did_effect": did_effect,
        "natural_experiment_did_se": did_se,
        "natural_experiment_its_effect": float(its["effect_at_week"]),
        "natural_experiment_its_se": float(its["effect_at_week_se"]),
    }


def observed_baseline_weekly_nrx(df: pd.DataFrame) -> float:
    """A simple baseline proxy: the lowest 8-week rolling mean in the series."""
    return float(df["nrx"].rolling(8, min_periods=1).mean().min())


def build_scorecard(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    fits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Truth-vs-posterior contribution scorecard across naive/controlled/calibrated fits."""
    truth_by_channel = truth.set_index("channel")["true_mean_weekly_contribution"]
    rows = []
    for ch in CHANNELS:
        row = {"channel": ch, "true_weekly_contribution": round(float(truth_by_channel[ch]), 2)}
        for label, draws in fits.items():
            contrib = _channel_contributions(df, draws)[ch]
            row[f"{label}_weekly_contribution"] = round(contrib, 2)
            row[f"{label}_pct_error"] = round(
                (contrib - truth_by_channel[ch]) / truth_by_channel[ch] * 100, 1
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_reconciliation_table(measurements: dict[str, float], calibrated_share: float) -> pd.DataFrame:
    """Field's three measurement-family readouts, with what each one answers."""
    return pd.DataFrame([
        {
            "measurement_family": "Path attribution (Markov removal effect)",
            "source": "ch08_omnichannel: markov_attribution() in sequences.py",
            "field_estimate": f"{measurements['field_attribution_markov_credit']:.1%} of conversion credit",
            "what_it_answers": "Share of recorded converting paths that pass through field, relative to all ten channels",
        },
        {
            "measurement_family": "Randomized experiment (account-cycle ITT)",
            "source": "ch10_experiments: adjusted_itt.csv from analysis.py",
            "field_estimate": f"+{measurements['field_experiment_relative_lift']:.1%} incremental patient starts vs. control",
            "what_it_answers": "Causal lift from one incremental coordinated field/digital action on the accounts that received it",
        },
        {
            "measurement_family": "Marketing mix model (calibrated posterior)",
            "source": "ch13_mmm: fit_bayesian_mmm() in model.py, geo_prior applied",
            "field_estimate": f"{calibrated_share:.1%} of decomposed weekly NRx",
            "what_it_answers": "Average share of weekly NRx decomposed to field across the full 104-week series, net of trend/seasonality/the formulary event",
        },
    ])


def build_holdout_validation(
    df: pd.DataFrame,
    holdout_weeks: int,
    measurements: dict[str, float],
) -> pd.DataFrame:
    """Compare predictive holdout fit across MMM variants and a seasonal naive baseline."""
    train = df.iloc[:-holdout_weeks].copy()
    holdout = df.iloc[-holdout_weeks:].copy()

    naive_draws = fit_bayesian_mmm(train, use_controls=False)
    controlled_draws = fit_bayesian_mmm(train, use_controls=True)
    geo_holdout = generate_field_geo_holdout()
    calibrated_draws = fit_bayesian_mmm(train, use_controls=True, geo_prior=geo_holdout)

    observed = holdout["nrx"].to_numpy(dtype=float)
    seasonal_naive_pred = df["nrx"].shift(52).iloc[-holdout_weeks:].to_numpy(dtype=float)
    prediction_map = {
        "seasonal_naive": seasonal_naive_pred,
        "naive_mmm": posterior_mean_prediction(holdout, naive_draws),
        "controlled_mmm": posterior_mean_prediction(holdout, controlled_draws),
        "calibrated_mmm": posterior_mean_prediction(holdout, calibrated_draws),
    }

    rows = []
    scale = max(observed.mean(), 1e-6)
    for model_name, pred in prediction_map.items():
        errors = observed - pred
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        rows.append({
            "model": model_name,
            "holdout_weeks": holdout_weeks,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "nrmse_vs_holdout_mean": round(rmse / scale, 4),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def channel_identifiability_diagnostics(df: pd.DataFrame, controlled_draws: pd.DataFrame) -> pd.DataFrame:
    """Per-channel checks on whether the 104-week series can identify that channel.

    `weeks_near_zero` counts weeks below 10% of the channel's own mean exposure.
    `spend_cv` is the coefficient of variation of the native exposure unit.
    `corr_with_event`/`corr_with_seasonal_sin`/`corr_with_seasonal_cos` measure
    how entangled the channel's own variation is with the baseline controls a
    real decomposition has to separate it from. `posterior_interval_width_pct`
    is the controlled fit's 90% credible interval for that channel's
    coefficient, as a percent of its posterior mean: a wide interval means
    the series leaves that channel's scale poorly pinned down even before
    calibration.
    """
    week_index = df["week_index"].to_numpy(dtype=float)
    event_indicator = (week_index >= KNOWN_EVENT_WEEK).astype(float)
    sin_basis = np.sin(2 * np.pi * week_index / 52)
    cos_basis = np.cos(2 * np.pi * week_index / 52)

    rows = []
    for ch in CHANNELS:
        exposure = df[exposure_column(ch)].to_numpy(dtype=float)
        coef_draws = controlled_draws[f"{ch}_coef"].to_numpy()
        coef_mean = float(coef_draws.mean())
        interval_width_pct = (
            (np.percentile(coef_draws, 95) - np.percentile(coef_draws, 5)) / coef_mean * 100
            if coef_mean > 0 else float("nan")
        )
        rows.append({
            "channel": ch,
            "weeks_near_zero": int((exposure < 0.10 * exposure.mean()).sum()),
            "spend_cv": round(float(exposure.std() / exposure.mean()), 4),
            "corr_with_event": round(float(np.corrcoef(exposure, event_indicator)[0, 1]), 4),
            "corr_with_seasonal_sin": round(float(np.corrcoef(exposure, sin_basis)[0, 1]), 4),
            "corr_with_seasonal_cos": round(float(np.corrcoef(exposure, cos_basis)[0, 1]), 4),
            "posterior_interval_width_pct": round(interval_width_pct, 1),
        })
    diagnostics = pd.DataFrame(rows)
    confound_cols = ["corr_with_event", "corr_with_seasonal_sin", "corr_with_seasonal_cos"]
    diagnostics["seasonality_corr"] = diagnostics[
        ["corr_with_seasonal_sin", "corr_with_seasonal_cos"]
    ].abs().max(axis=1).round(4)
    diagnostics["max_abs_confound_corr"] = diagnostics[confound_cols].abs().max(axis=1).round(4)
    return diagnostics


def model_health_scorecard(
    fits: dict[str, pd.DataFrame],
    scorecard: pd.DataFrame,
    holdout_validation: pd.DataFrame,
) -> pd.DataFrame:
    """One row per fit: R-hat, ESS, holdout fit, and truth recovery, all in one place.

    `decision_status` here describes the fit as a whole, not any one channel:
    naive and controlled are teaching fits that were never meant to move
    budget; the calibrated fit is the only one considered even for a
    constrained decision, and only for the channels `measurement_decision_record()`
    separately clears.
    """
    holdout_by_model = holdout_validation.set_index("model")
    holdout_name = {"naive": "naive_mmm", "controlled": "controlled_mmm", "calibrated": "calibrated_mmm"}
    status = {
        "naive": "teaching only: no baseline controls, not a decision input",
        "controlled": "teaching only: controls added, still uncalibrated and unconstrained",
        "calibrated": "channel-gated: see measurement_decision_record.csv for which channels clear the gate",
    }

    rows = []
    for label, draws in fits.items():
        rhat = draws.attrs["rhat"]
        ess = draws.attrs["ess"]
        worst_param = max(rhat, key=rhat.get)
        channel_ess = {k: v for k, v in ess.items() if any(k.startswith(f"{ch}_") for ch in CHANNELS)}
        error_cols = [f"{label}_pct_error"]
        worst_row = scorecard.loc[scorecard[error_cols[0]].abs().idxmax()]
        holdout_row = holdout_by_model.loc[holdout_name[label]]
        total_abs_error = float(
            (scorecard[f"{label}_weekly_contribution"] - scorecard["true_weekly_contribution"]).abs().sum()
        )
        rows.append({
            "fit_name": label,
            "max_rhat": round(max(rhat.values()), 3),
            "worst_rhat_parameter": worst_param,
            "min_ess_decision_params": round(min(channel_ess.values()), 0),
            "holdout_mae": float(holdout_row["mae"]),
            "holdout_nrmse": float(holdout_row["nrmse_vs_holdout_mean"]),
            "total_abs_contribution_error": round(total_abs_error, 2),
            "worst_channel": worst_row["channel"],
            "worst_channel_pct_error": float(worst_row[error_cols[0]]),
            "decision_status": status[label],
        })
    return pd.DataFrame(rows)


def calibration_trace(
    df: pd.DataFrame,
    controlled_draws: pd.DataFrame,
    calibrated_draws: pd.DataFrame,
    geo_holdout: dict[str, float],
) -> pd.DataFrame:
    """Show what the geo-holdout measured and what the model implied before/after.

    The model-implied increment uses the same steady-state segment convention
    as `_geo_prior_penalty()` in model.py: field's posterior-mean parameters,
    evaluated at `input_level` +/- `delta_input`/2.
    """
    input_level = float(geo_holdout["input_level"])
    delta_input = float(geo_holdout["delta_input"])

    def implied_increment(draws: pd.DataFrame) -> float:
        coef = float(draws["field_coef"].mean())
        decay = float(draws["field_decay"].mean())
        ec50 = float(draws["field_ec50"].mean())
        slope = float(draws["field_slope"].mean())
        lo = np.full(20, max(input_level - delta_input / 2, 0.0))
        hi = np.full(20, input_level + delta_input / 2)
        return coef * (_hill(_adstock(hi, decay), ec50, slope).mean() - _hill(_adstock(lo, decay), ec50, slope).mean())

    return pd.DataFrame([{
        "measurement_source": "synthetic field geo-holdout (generate_field_geo_holdout)",
        "n_geos": geo_holdout["n_geos"],
        "input_level_calls": input_level,
        "delta_input_calls": delta_input,
        "experiment_mean_incremental_nrx": geo_holdout["mean_incremental_nrx"],
        "experiment_sd_incremental_nrx": geo_holdout["sd_incremental_nrx"],
        "model_implied_increment_before_calibration": round(implied_increment(controlled_draws), 2),
        "model_implied_increment_after_calibration": round(implied_increment(calibrated_draws), 2),
    }])


def build_prior_sensitivity(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    controlled_draws: pd.DataFrame,
    calibrated_draws: pd.DataFrame,
    geo_holdout: dict[str, float],
) -> pd.DataFrame:
    """Refit under a halved and a doubled channel-coefficient prior mean.

    `PRIOR_COEF` is one generic (mean, sd) applied identically to every
    channel (model.py never gives itself a per-channel head start). Shifting
    its mean and refitting, both before and after the geo-holdout
    calibration, tests directly whether a channel's answer belongs to the
    weekly data or to the prior: a channel with strong independent variation
    should barely move; a weakly identified channel should swing with the
    prior until an outside measurement anchors it. `controlled_draws` and
    `calibrated_draws` supply the default-prior (mean=60) rung so only two
    additional refits per fit stage are needed.
    """
    variants = {
        "coef_halved": MMMPriors(coef=(PRIOR_COEF[0] / 2, PRIOR_COEF[1])),
        "coef_doubled": MMMPriors(coef=(PRIOR_COEF[0] * 2, PRIOR_COEF[1])),
    }
    controlled_by_variant = {"default": controlled_draws}
    calibrated_by_variant = {"default": calibrated_draws}
    for label, priors in variants.items():
        controlled_by_variant[label] = fit_bayesian_mmm(df, use_controls=True, priors=priors)
        calibrated_by_variant[label] = fit_bayesian_mmm(df, use_controls=True, geo_prior=geo_holdout, priors=priors)

    truth_by_channel = truth.set_index("channel")["true_mean_weekly_contribution"]
    variant_order = ["coef_halved", "default", "coef_doubled"]
    rows = []
    for ch in CHANNELS:
        row: dict[str, float | str] = {"channel": ch, "true_weekly_contribution": round(float(truth_by_channel[ch]), 2)}
        controlled_vals: list[float] = []
        calibrated_vals: list[float] = []
        for label in variant_order:
            controlled_contrib = _channel_contributions(df, controlled_by_variant[label])[ch]
            calibrated_contrib = _channel_contributions(df, calibrated_by_variant[label])[ch]
            row[f"controlled_{label}_contribution"] = round(controlled_contrib, 2)
            row[f"calibrated_{label}_contribution"] = round(calibrated_contrib, 2)
            controlled_vals.append(controlled_contrib)
            calibrated_vals.append(calibrated_contrib)
        # Swing is reported against each rung's own default-prior contribution
        # (the middle entry of variant_order) so it reads on the same NRx
        # scale as the contribution columns above.
        controlled_default = controlled_vals[variant_order.index("default")]
        calibrated_default = calibrated_vals[variant_order.index("default")]
        row["controlled_swing_pct_of_default"] = round(
            (max(controlled_vals) - min(controlled_vals)) / controlled_default * 100 if controlled_default else float("nan"), 1
        )
        row["calibrated_swing_pct_of_default"] = round(
            (max(calibrated_vals) - min(calibrated_vals)) / calibrated_default * 100 if calibrated_default else float("nan"), 1
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pareto_front(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    calibrated_draws: pd.DataFrame,
    n_draws: int = 2_400,
    focus_channel: str = "digital",
) -> pd.DataFrame:
    """Score each calibrated-fit posterior draw on in-sample fit and decomposition.

    `nrmse` is in-sample fit error, the number a tuning loop that selects on
    predictive fit alone would optimize. `decomp_rssd` is Robyn's own second
    objective (see Table 13.7): the root-sum-square distance between each
    channel's dollar-spend share and its NRx-effect share in that draw, using
    the same native-units-to-dollars spend each channel already reports via
    `spend_{channel}`. The geo-holdout calibration only disciplines field's own
    parameters, so field's own contribution is tight in every draw here; the
    remaining decomposition disagreement is tracked instead on `focus_channel`
    (digital by default), the channel whose 0.90 event correlation means it is
    still uncalibrated and still not cleanly identified. A draw's
    `{focus_channel}_pct_error` against ground truth is only checkable because
    this chapter has synthetic ground truth; decomp_rssd alone is the portable
    half of this diagnostic. Draws are ranked by NRMSE and flagged
    `pareto_efficient` where no earlier (lower-NRMSE) draw has an equal or
    lower decomp_rssd, tracing the same two-objective front Robyn returns
    instead of a single tuned model.
    """
    week_index = df["week_index"].to_numpy(dtype=float)
    spend_mat = {ch: df[exposure_column(ch)].to_numpy(dtype=float) for ch in CHANNELS}
    nrx = df["nrx"].to_numpy(dtype=float)
    scale = max(float(nrx.mean()), 1e-6)

    dollar_spend = {ch: float(df[f"spend_{ch}"].mean()) for ch in CHANNELS}
    total_dollar_spend = sum(dollar_spend.values())
    spend_share = {ch: dollar_spend[ch] / total_dollar_spend for ch in CHANNELS}
    true_focus = float(truth.set_index("channel").loc[focus_channel, "true_mean_weekly_contribution"])

    use_n = min(n_draws, len(calibrated_draws))
    rows = []
    for j in range(use_n):
        row = calibrated_draws.iloc[j]
        mu = _mean_prediction_for_row(row, True, week_index, spend_mat)
        nrmse = float(np.sqrt(np.mean((nrx - mu) ** 2))) / scale

        contributions = {}
        for ch in CHANNELS:
            ads = _adstock(spend_mat[ch], float(row[f"{ch}_decay"]))
            contributions[ch] = float(row[f"{ch}_coef"]) * float(
                _hill(ads, float(row[f"{ch}_ec50"]), float(row[f"{ch}_slope"])).mean()
            )
        total_contribution = sum(contributions.values())
        effect_share = {ch: contributions[ch] / total_contribution for ch in CHANNELS}
        decomp_rssd = float(np.sqrt(sum((spend_share[ch] - effect_share[ch]) ** 2 for ch in CHANNELS)))
        focus_pct_error = (contributions[focus_channel] - true_focus) / true_focus * 100

        rows.append({
            "draw": j,
            "nrmse": round(nrmse, 5),
            "decomp_rssd": round(decomp_rssd, 5),
            f"{focus_channel}_contribution": round(contributions[focus_channel], 2),
            f"{focus_channel}_pct_error": round(focus_pct_error, 1),
        })

    result = pd.DataFrame(rows).sort_values("nrmse").reset_index(drop=True)
    best_rssd_so_far = float("inf")
    is_efficient = []
    for rssd in result["decomp_rssd"]:
        efficient = rssd < best_rssd_so_far
        is_efficient.append(efficient)
        if efficient:
            best_rssd_so_far = rssd
    result["pareto_efficient"] = is_efficient
    return result


def measurement_decision_record(
    diagnostics: pd.DataFrame,
    scorecard: pd.DataFrame,
    calibrated_draws: pd.DataFrame,
) -> pd.DataFrame:
    """Classify each channel as decision-ready, directional, or not usable.

    A channel clears the gate only if its calibrated-fit R-hat, channel
    variation, and confound correlation all clear the thresholds above. None
    of these 3 checks reads `true_channel_share()`, so the gate is the one a
    real analyst can run without an answer key; the calibrated-vs-truth
    error stays in the output only as a teaching column (this synthetic
    case's answer key) and as an input to `build_next_measurement_agenda()`'s
    priority score, not as a gate criterion. A channel that fails on
    identifying variation alone (low CV, high correlation with a control)
    but still has a well-converged calibrated estimate is marked directional
    and flagged as calibration-dependent, since that estimate rests on the
    external geo-holdout rather than on the channel's own time-series
    variation.
    """
    rhat = calibrated_draws.attrs["rhat"]
    diagnostics_by_channel = diagnostics.set_index("channel")
    scorecard_by_channel = scorecard.set_index("channel")

    rows = []
    for ch in CHANNELS:
        worst_channel_rhat = max(rhat[f"{ch}_{p}"] for p in ("coef", "decay", "ec50", "slope"))
        abs_pct_error = abs(float(scorecard_by_channel.loc[ch, "calibrated_pct_error"]))
        cv = float(diagnostics_by_channel.loc[ch, "spend_cv"])
        max_confound_corr = float(diagnostics_by_channel.loc[ch, "max_abs_confound_corr"])
        calibration_dependent = cv < GATE_MIN_CV_DECISION_READY

        clears_rhat = worst_channel_rhat < GATE_MAX_RHAT_DECISION_READY
        clears_corr = max_confound_corr < GATE_MAX_CONFOUND_CORR_DECISION_READY
        clears_cv = cv > GATE_MIN_CV_DECISION_READY

        fails_hard = (
            cv < GATE_MIN_CV_NOT_USABLE
            or max_confound_corr > GATE_MAX_CONFOUND_CORR_NOT_USABLE
            or worst_channel_rhat > GATE_MAX_RHAT_NOT_USABLE
        )

        if clears_rhat and clears_corr and clears_cv:
            status = "decision-ready"
        elif fails_hard:
            status = "not usable"
        else:
            status = "directional"

        reasons = []
        if not clears_rhat:
            if worst_channel_rhat > GATE_MAX_RHAT_NOT_USABLE:
                reasons.append(f"worst channel R-hat {worst_channel_rhat:.3f} > not-usable {GATE_MAX_RHAT_NOT_USABLE:.2f}")
            else:
                reasons.append(
                    f"worst channel R-hat {worst_channel_rhat:.3f} "
                    f"(directional: {GATE_MAX_RHAT_DECISION_READY:.2f}-{GATE_MAX_RHAT_NOT_USABLE:.2f})"
                )
        if not clears_corr:
            if max_confound_corr > GATE_MAX_CONFOUND_CORR_NOT_USABLE:
                reasons.append(f"correlation with a baseline control {max_confound_corr:.2f} > not-usable {GATE_MAX_CONFOUND_CORR_NOT_USABLE:.2f}")
            else:
                reasons.append(
                    f"correlation with a baseline control {max_confound_corr:.2f} "
                    f"(directional: {GATE_MAX_CONFOUND_CORR_DECISION_READY:.2f}-{GATE_MAX_CONFOUND_CORR_NOT_USABLE:.2f})"
                )
        if not clears_cv:
            if cv < GATE_MIN_CV_NOT_USABLE:
                reasons.append(f"own-series coefficient of variation {cv:.2f} < not-usable {GATE_MIN_CV_NOT_USABLE:.2f}")
            else:
                reasons.append(
                    f"own-series coefficient of variation {cv:.2f} "
                    f"(directional: {GATE_MIN_CV_NOT_USABLE:.2f}-{GATE_MIN_CV_DECISION_READY:.2f})"
                )
        if not reasons:
            reasons.append("clears R-hat, correlation, and variation thresholds")

        rows.append({
            "channel": ch,
            "decision_status": status,
            "calibration_dependent": calibration_dependent,
            "worst_channel_rhat": round(worst_channel_rhat, 3),
            "calibrated_abs_pct_error": round(abs_pct_error, 1),
            "spend_cv": round(cv, 3),
            "max_abs_confound_corr": round(max_confound_corr, 3),
            "reasons": "; ".join(reasons),
            "eligible_for_unconstrained_optimization": status == "decision-ready",
        })
    return pd.DataFrame(rows)


def model_ladder_scorecard(scorecard: pd.DataFrame, decision_record: pd.DataFrame) -> pd.DataFrame:
    """Narrow ladder view: truth, estimate, and error for each channel at each rung.

    `decision_consequence` is only populated for the calibrated rung, since
    that is the fit the chapter considers for any budget conversation.
    """
    consequence_by_channel = decision_record.set_index("channel")["decision_status"]
    rung_labels = {"naive": "1. Naive (no controls)", "controlled": "2. Controlled (trend/seasonality/event)", "calibrated": "3. Calibrated (geo-holdout on field)"}

    rows = []
    for label, rung_name in rung_labels.items():
        contrib_col = f"{label}_weekly_contribution"
        error_col = f"{label}_pct_error"
        total_abs_error = float((scorecard[contrib_col] - scorecard["true_weekly_contribution"]).abs().sum())
        worst_channel = scorecard.loc[scorecard[error_col].abs().idxmax(), "channel"]
        row = {"fit_name": rung_name}
        for ch in CHANNELS:
            ch_row = scorecard.set_index("channel").loc[ch]
            row[f"{ch}_true_contribution"] = float(ch_row["true_weekly_contribution"])
            row[f"{ch}_estimated_contribution"] = float(ch_row[contrib_col])
            row[f"{ch}_pct_error"] = float(ch_row[error_col])
        row["total_abs_nrx_error"] = round(total_abs_error, 2)
        row["largest_error_channel"] = worst_channel
        row["decision_consequence"] = (
            f"Used only as a teaching diagnostic; {worst_channel} carries the largest error."
            if label != "calibrated"
            else "; ".join(
                f"{ch}: {consequence_by_channel[ch]}" for ch in CHANNELS
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_measurement_method_map() -> pd.DataFrame:
    """What each measurement family answers, at what unit, and what it should
    not decide alone.

    A static reference table, not derived from this chapter's own fit: it is
    the chapter's answer to "which tool controls which decision," the frame
    the rest of the chapter's generated evidence (the evidence record, the
    agenda, the unified budget recommendation) has to fit inside.
    """
    return pd.DataFrame([
        {
            "method": "Attribution",
            "unit_of_analysis": "recorded touchpoint sequence, per patient/account",
            "cadence": "near-real-time to weekly",
            "causal_strength": "observational: credit, not incrementality",
            "best_business_decision": "fast, granular channel-mix or creative optimization",
            "informs_other_methods": "flags candidate channels for an incrementality test; sanity-checks MMM channel ranking",
            "should_not_decide_alone": "portfolio budget moves; whether a channel is causal at all",
        },
        {
            "method": "Uplift / response modeling",
            "unit_of_analysis": "individual or segment propensity",
            "cadence": "weekly to monthly refresh",
            "causal_strength": "observational, model-based counterfactual",
            "best_business_decision": "targeting and next-best-action within a channel",
            "informs_other_methods": "supplies candidate segments for randomized experiments",
            "should_not_decide_alone": "cross-channel portfolio allocation",
        },
        {
            "method": "Randomized experiment",
            "unit_of_analysis": "one channel or action, one treated population",
            "cadence": "per test, weeks to a quarter",
            "causal_strength": "strong: randomized causal lift",
            "best_business_decision": "whether one specific action causes lift, and how much",
            "informs_other_methods": "anchors or calibrates MMM priors; disciplines attribution credit for that channel",
            "should_not_decide_alone": "channels or populations outside the tested scope",
        },
        {
            "method": "Geo-holdout / causal-impact read",
            "unit_of_analysis": "geography or market, aggregate response",
            "cadence": "per test, expressed as a response-curve segment",
            "causal_strength": "strong within tested geos; quasi-experimental",
            "best_business_decision": "calibrating one channel's response curve at a spend level",
            "informs_other_methods": "anchors the MMM posterior for the tested channel",
            "should_not_decide_alone": "channels never included in a geo test",
        },
        {
            "method": "Natural experiment / quasi-experimental read",
            "unit_of_analysis": "population exposed to an external policy or access event",
            "cadence": "opportunistic, tied to the event",
            "causal_strength": "moderate to strong, conditional on identification assumptions",
            "best_business_decision": "estimating the effect of an event neither team controlled",
            "informs_other_methods": "flags confounds MMM must control for (e.g., a formulary event)",
            "should_not_decide_alone": "routine channel budget decisions absent a comparable event",
        },
        {
            "method": "Marketing mix model (MMM)",
            "unit_of_analysis": "channel, aggregated across the full portfolio and time series",
            "cadence": "refit for planning cycles, quarterly to annual",
            "causal_strength": "observational: assumes no unmeasured confounders unless calibrated",
            "best_business_decision": "portfolio-level budget allocation across all channels at once",
            "informs_other_methods": "flags channels with weak identification or high error that need a test",
            "should_not_decide_alone": "moving an unconstrained channel that has not cleared the model-health gate",
        },
        {
            "method": "Unified measurement decision record",
            "unit_of_analysis": "channel, across every available measurement family",
            "cadence": "continuous, updated as each pillar refreshes",
            "causal_strength": "inherits the strongest available evidence per channel, not averaged",
            "best_business_decision": "how much budget movement each channel's evidence currently supports",
            "informs_other_methods": "sets the next-measurement agenda that each pillar's next cycle should close",
            "should_not_decide_alone": "nothing: it is the record, not a fifth independent estimate",
        },
    ])


def build_method_comparability_checks(
    measurements: dict[str, float],
    geo_holdout: dict[str, float],
) -> pd.DataFrame:
    """Structured comparability record across measurement families."""
    rows = [
        {
            "channel": "field",
            "method_family": "Attribution",
            "metric": "conversion credit share",
            "population": "converting paths across all ten channels",
            "window": "Jan 2024 to Mar 2025 path history",
            "intervention": "observed field touches on recorded paths",
            "spend_level_or_exposure": "path occurrence, not spend calibrated",
            "comparable_to_mmm": "partial",
            "allowed_use": "sanity check only",
            "source": "ch08_omnichannel/assets/generated_outputs/markov_attribution.csv",
            "summary": f"{measurements['field_attribution_markov_credit']:.1%} conversion credit",
        },
        {
            "channel": "email",
            "method_family": "Attribution",
            "metric": "conversion credit share",
            "population": "converting paths across all ten channels",
            "window": "Jan 2024 to Mar 2025 path history",
            "intervention": "observed email touches on recorded paths",
            "spend_level_or_exposure": "path occurrence, not spend calibrated",
            "comparable_to_mmm": "partial",
            "allowed_use": "sanity check only",
            "source": "ch08_omnichannel/assets/generated_outputs/markov_attribution.csv",
            "summary": f"{measurements['email_attribution_markov_credit']:.1%} conversion credit",
        },
        {
            "channel": "digital",
            "method_family": "Attribution",
            "metric": "conversion credit share",
            "population": "authenticated web paths, used here as a digital proxy",
            "window": "Jan 2024 to Mar 2025 path history",
            "intervention": "observed web touches on recorded paths",
            "spend_level_or_exposure": "path occurrence, not spend calibrated",
            "comparable_to_mmm": "partial",
            "allowed_use": "proxy sanity check only",
            "source": "ch08_omnichannel/assets/generated_outputs/markov_attribution.csv",
            "summary": f"{measurements['digital_proxy_attribution_markov_credit']:.1%} conversion credit (web proxy)",
        },
        {
            "channel": "paid_media",
            "method_family": "Attribution",
            "metric": "conversion credit share",
            "population": "converting paths across all ten channels",
            "window": "Jan 2024 to Mar 2025 path history",
            "intervention": "observed paid-media touches on recorded paths",
            "spend_level_or_exposure": "path occurrence, not spend calibrated",
            "comparable_to_mmm": "partial",
            "allowed_use": "sanity check only",
            "source": "ch08_omnichannel/assets/generated_outputs/markov_attribution.csv",
            "summary": f"{measurements['paid_media_attribution_markov_credit']:.1%} conversion credit",
        },
        {
            "channel": "field",
            "method_family": "Randomized experiment",
            "metric": "incremental patient starts vs. control",
            "population": "treated and control accounts in one account cycle",
            "window": "one account-cycle experiment",
            "intervention": "coordinated field and digital action",
            "spend_level_or_exposure": "tested action bundle, not a full response curve",
            "comparable_to_mmm": "partial",
            "allowed_use": "fallback calibration or scope check",
            "source": "ch10_experiments/assets/generated_outputs/adjusted_itt.csv",
            "summary": f"+{measurements['field_experiment_relative_lift']:.1%} relative lift",
        },
        {
            "channel": "field",
            "method_family": "Geo-holdout",
            "metric": "incremental weekly NRx",
            "population": f"{geo_holdout['n_geos']} geos in the synthetic holdout",
            "window": "tested spend segment at the planning cadence",
            "intervention": "field call increase in the holdout geos",
            "spend_level_or_exposure": f"{geo_holdout['input_level']:.2f} calls plus {geo_holdout['delta_input']:.2f} incremental calls",
            "comparable_to_mmm": "yes",
            "allowed_use": "calibrate prior",
            "source": "ch13_mmm/scripts/data.py generate_field_geo_holdout()",
            "summary": f"{geo_holdout['mean_incremental_nrx']:.2f} incremental weekly NRx",
        },
        {
            "channel": "portfolio_baseline",
            "method_family": "Natural experiment",
            "metric": "brand-share lift from the formulary event",
            "population": "treated vs. control geographies around the access change",
            "window": "event window from the natural-experiment chapter",
            "intervention": "formulary access win",
            "spend_level_or_exposure": "not a channel spend measure",
            "comparable_to_mmm": "no",
            "allowed_use": "baseline control only",
            "source": "ch11_natural_experiments/assets/generated_outputs/did.csv and its_summary.csv",
            "summary": f"DiD {measurements['natural_experiment_did_effect']:.4f}; ITS {measurements['natural_experiment_its_effect']:.4f}",
        },
    ]
    for ch in CHANNELS:
        rows.append({
            "channel": ch,
            "method_family": "MMM",
            "metric": "average weekly NRx contribution",
            "population": "national weekly portfolio time series",
            "window": "104-week planning series",
            "intervention": "all observed spend over the full period",
            "spend_level_or_exposure": "full weekly response curve",
            "comparable_to_mmm": "yes",
            "allowed_use": "portfolio allocation",
            "source": "ch13_mmm/scripts/model.py fit_bayesian_mmm()",
            "summary": "portfolio contribution estimate",
        })
    return pd.DataFrame(rows)


def build_channel_evidence_record(
    measurements: dict[str, float],
    geo_holdout: dict[str, float],
    decision_record: pd.DataFrame,
    scorecard: pd.DataFrame,
    comparability_checks: pd.DataFrame,
) -> pd.DataFrame:
    """Per-channel evidence record: what each measurement family says about
    this channel today, and what that combination allows for budget.

    `evidence_tier` is derived, not hand-assigned: a channel is
    "causal-anchored" only if it has an experiment or geo-holdout read;
    otherwise its tier falls out of `measurement_decision_record()`'s own
    decision_status, so this record cannot rate a channel higher than the
    MMM-health gate already allows.
    """
    decision_by_channel = decision_record.set_index("channel")
    scorecard_by_channel = scorecard.set_index("channel")
    checks_by_channel = comparability_checks.groupby("channel")

    next_action_by_tier = {
        "causal-anchored": "refresh the calibration periodically (repeat the geo-holdout or experiment); monitor for drift",
        "mmm-only decision-ready": "add a low-cost incrementality or holdout test to anchor this channel before scaling meaningfully beyond current spend",
        "mmm-only directional": "run a targeted holdout or geo experiment to sharpen decomposition before requesting a wider bound",
        "mmm-only not usable": "run an event-specific or geo experiment before any budget move; the MMM read alone is not sufficient",
    }

    rows = []
    for ch in CHANNELS:
        decision_status = decision_by_channel.loc[ch, "decision_status"]
        mmm_contribution = float(scorecard_by_channel.loc[ch, "calibrated_weekly_contribution"])

        has_attribution = ch in {"field", "email", "digital", "paid_media"}
        has_experiment = ch == "field"
        has_causal = ch == "field"

        if ch == "field":
            attribution_signal = f"{measurements['field_attribution_markov_credit']:.1%} of conversion credit (Markov removal effect)"
            attribution_scope = "path-level credit pool shared across all ten book channels"
        elif ch == "email":
            attribution_signal = f"{measurements['email_attribution_markov_credit']:.1%} of conversion credit (Markov removal effect)"
            attribution_scope = "path-level credit pool shared across all ten book channels"
        elif ch == "digital":
            attribution_signal = f"{measurements['digital_proxy_attribution_markov_credit']:.1%} of conversion credit (authenticated web proxy)"
            attribution_scope = "partial digital proxy from authenticated web paths; compare scope before using it against MMM"
        elif ch == "paid_media":
            attribution_signal = f"{measurements['paid_media_attribution_markov_credit']:.1%} of conversion credit (Markov removal effect)"
            attribution_scope = "path-level credit pool shared across all ten book channels"
        else:
            attribution_signal = "not available in current attribution pull"
            attribution_scope = "n/a"

        if has_experiment:
            experiment_signal = f"+{measurements['field_experiment_relative_lift']:.1%} incremental patient starts vs. control (account-cycle ITT)"
            experiment_scope = "one coordinated incremental action, treated accounts only"
        else:
            experiment_signal = "not available in current experiment pull"
            experiment_scope = "n/a"

        if has_causal:
            causal_signal = (
                f"{geo_holdout['mean_incremental_nrx']:.2f} incremental weekly NRx across "
                f"{geo_holdout['n_geos']} geos (field geo-holdout, response-curve segment)"
            )
        else:
            causal_signal = "not available: no geo-holdout or causal-impact read run on this channel yet"

        has_causal_or_experiment = has_causal or has_experiment
        if has_causal_or_experiment:
            evidence_tier = "causal-anchored"
        elif decision_status == "decision-ready":
            evidence_tier = "mmm-only decision-ready"
        elif decision_status == "directional":
            evidence_tier = "mmm-only directional"
        else:
            evidence_tier = "mmm-only not usable"

        channel_checks = checks_by_channel.get_group(ch) if ch in checks_by_channel.groups else pd.DataFrame()
        non_mmm_checks = channel_checks.loc[channel_checks["method_family"] != "MMM"] if not channel_checks.empty else channel_checks
        if not non_mmm_checks.empty and (non_mmm_checks["comparable_to_mmm"] == "yes").any():
            comparability_status = "fully comparable calibration or budget guardrail available"
        elif not non_mmm_checks.empty and (non_mmm_checks["comparable_to_mmm"] == "partial").any():
            comparability_status = "partial cross-check available; compare metric, scope, and population before using it against MMM"
        elif has_attribution or has_experiment or has_causal:
            comparability_status = "multiple estimands available; compare scope before blending, do not average"
        else:
            comparability_status = "single-method read; no independent cross-check available yet"

        rows.append({
            "channel": ch,
            "attribution_signal": attribution_signal,
            "attribution_scope": attribution_scope,
            "experiment_signal": experiment_signal,
            "experiment_scope": experiment_scope,
            "causal_signal": causal_signal,
            "mmm_contribution": round(mmm_contribution, 2),
            "mmm_decision_status": decision_status,
            "comparability_status": comparability_status,
            "evidence_tier": evidence_tier,
            "budget_rule": "",
            "next_measurement_action": next_action_by_tier[evidence_tier],
        })
    return pd.DataFrame(rows)


def build_measurement_guardrails(
    evidence_record: pd.DataFrame,
    decision_record: pd.DataFrame,
    current_spends: np.ndarray,
) -> pd.DataFrame:
    """Turn the evidence record into channel guardrails the optimizer can enforce."""
    evidence_record_by_channel = evidence_record.set_index("channel")
    decision_by_channel = decision_record.set_index("channel")
    current_budget = float(current_spends.sum())

    rows = []
    for i, ch in enumerate(CHANNELS):
        tier = str(evidence_record_by_channel.loc[ch, "evidence_tier"])
        decision_status = str(decision_by_channel.loc[ch, "decision_status"])
        current_spend = float(current_spends[i])
        comparability_status = str(evidence_record_by_channel.loc[ch, "comparability_status"])

        if tier == "causal-anchored" and decision_status == "decision-ready":
            move_permission = "full-range"
            max_move_pct = 0.0
            spend_floor = 0.0
            spend_ceiling = current_budget
            allowed_budget_move = "full range within optimizer bounds; channel is causal-anchored and MMM decision-ready"
            action_type = "refresh_anchor"
            anchor_status = "current"
            reason = "causal anchor in place and MMM gate cleared"
        elif tier == "causal-anchored":
            move_permission = "bounded"
            max_move_pct = DIRECTIONAL_BAND
            spend_floor = max(current_spend * (1 - max_move_pct), 0.0)
            spend_ceiling = current_spend * (1 + max_move_pct)
            allowed_budget_move = f"bounded to +/-{max_move_pct:.0%}; causal anchor exists but MMM diagnostics still limit movement"
            action_type = "refresh_anchor"
            anchor_status = "current"
            reason = "causal anchor exists, but MMM diagnostics remain directional"
        elif tier == "mmm-only decision-ready":
            move_permission = "increase-capped"
            max_move_pct = DECISION_READY_INCREASE_CAP
            spend_floor = 0.0
            spend_ceiling = current_spend * (1 + max_move_pct)
            allowed_budget_move = f"increase capped at +{max_move_pct:.0%}; causal anchor required before sustained scaling"
            action_type = "new_anchor"
            anchor_status = "not anchored"
            reason = "MMM diagnostics clear, but no causal anchor yet"
        elif tier == "mmm-only directional":
            move_permission = "bounded"
            max_move_pct = WEAK_DIRECTIONAL_BAND
            spend_floor = max(current_spend * (1 - max_move_pct), 0.0)
            spend_ceiling = current_spend * (1 + max_move_pct)
            allowed_budget_move = f"bounded to +/-{max_move_pct:.0%}; no causal anchor and MMM remains directional"
            action_type = "new_anchor"
            anchor_status = "not anchored"
            reason = "MMM remains directional and no causal anchor is available"
        else:
            move_permission = "frozen"
            max_move_pct = 0.0
            spend_floor = current_spend
            spend_ceiling = current_spend
            allowed_budget_move = "frozen at current spend until a new experiment or event-specific read is available"
            action_type = "new_anchor"
            anchor_status = "not anchored"
            reason = "MMM is not usable for this channel on its own"

        if "partial cross-check" in comparability_status and move_permission == "full-range":
            move_permission = "increase-capped"
            max_move_pct = DECISION_READY_INCREASE_CAP
            spend_floor = 0.0
            spend_ceiling = current_spend * (1 + max_move_pct)
            allowed_budget_move = f"increase capped at +{max_move_pct:.0%}; cross-check is only partial"
            reason = "cross-method support exists, but it is only partially comparable to MMM"

        rows.append({
            "channel": ch,
            "decision_status": decision_status,
            "evidence_tier": tier,
            "move_permission": move_permission,
            "max_move_pct": round(max_move_pct, 3),
            "spend_floor": round(spend_floor, 1),
            "spend_ceiling": round(spend_ceiling, 1),
            "allowed_budget_move": allowed_budget_move,
            "new_anchor_required": tier != "causal-anchored",
            "refresh_required": tier == "causal-anchored",
            "next_measurement_action_type": action_type,
            "anchor_staleness_status": anchor_status,
            "guardrail_reason": reason,
        })
    return pd.DataFrame(rows)


def build_next_measurement_agenda(
    evidence_record: pd.DataFrame,
    decision_record: pd.DataFrame,
    current_spends: np.ndarray,
    guardrails: pd.DataFrame,
) -> pd.DataFrame:
    """Rank channels by measurement gap so the next test budget goes where it
    matters most: channels whose MMM read cannot be trusted alone, weighted by
    spend at risk, residual confounding, and how much a new anchor could unlock.
    """
    gap_severity_by_tier = {
        "mmm-only not usable": 100.0,
        "mmm-only directional": 70.0,
        "mmm-only decision-ready": 40.0,
        "causal-anchored": 10.0,
    }
    current_budget = float(current_spends.sum())
    decision_by_channel = decision_record.set_index("channel")
    guardrail_by_channel = guardrails.set_index("channel")
    rows = []
    for i, row in evidence_record.iterrows():
        ch = row["channel"]
        spend_share_pct = float(current_spends[i]) / current_budget * 100
        confound_score = float(decision_by_channel.loc[ch, "max_abs_confound_corr"]) * 20
        error_score = float(decision_by_channel.loc[ch, "calibrated_abs_pct_error"]) * 0.25
        if bool(guardrail_by_channel.loc[ch, "new_anchor_required"]):
            unlock_score = 25.0 if decision_by_channel.loc[ch, "decision_status"] == "decision-ready" else 20.0
        elif bool(guardrail_by_channel.loc[ch, "refresh_required"]):
            unlock_score = 5.0
        else:
            unlock_score = 0.0
        priority_score = (
            gap_severity_by_tier[row["evidence_tier"]]
            + spend_share_pct * 0.6
            + confound_score
            + error_score
            + unlock_score
        )
        rows.append({
            "channel": ch,
            "evidence_tier": row["evidence_tier"],
            "gap_severity": gap_severity_by_tier[row["evidence_tier"]],
            "recommended_next_test": row["next_measurement_action"],
            "target_metric": "incremental weekly NRx",
            "spend_share_pct": round(spend_share_pct, 1),
            "confound_score": round(confound_score, 1),
            "error_score": round(error_score, 1),
            "decision_unlock_score": round(unlock_score, 1),
            "priority_score": round(priority_score, 1),
        })
    agenda = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    agenda["priority_rank"] = agenda.index + 1
    return agenda


def run_analysis() -> dict[str, pd.DataFrame]:
    df = generate_mmm_data()
    truth = true_channel_share(df)
    measurements = load_cross_chapter_measurements()

    naive_draws = fit_bayesian_mmm(df, use_controls=False)
    controlled_draws = fit_bayesian_mmm(df, use_controls=True)

    # Main calibration path: a synthetic field-only geo-holdout, expressed as
    # a response-curve segment (current call level, the incremental calls
    # tested, and the incremental NRx measured), not an average-contribution
    # prior. See CH13_13_3_13_4_FITTING_CALIBRATION_REBUILD_PLAN section 4.3.
    geo_holdout = generate_field_geo_holdout()
    calibrated_draws = fit_bayesian_mmm(df, use_controls=True, geo_prior=geo_holdout)

    # Fallback/appendix path: the upstream account-cycle experiment translated
    # onto an average-weekly-contribution scale. Kept for the reconciliation
    # table and as a documented alternative, not fed into the main fit.
    itt_prior = implied_field_experiment_prior(
        measurements["field_experiment_control_mean"],
        measurements["field_experiment_adjusted_effect"],
        measurements["field_experiment_crude_se"],
        baseline_weekly_nrx=observed_baseline_weekly_nrx(df),
    )

    true_lookup: dict[str, float] = {}
    for ch, params in _TRUE_PARAMS.items():
        true_lookup[f"{ch}_coef"] = params["coefficient"]
        true_lookup[f"{ch}_decay"] = params["adstock_decay"]
        true_lookup[f"{ch}_ec50"] = params["hill_ec50"]
        true_lookup[f"{ch}_slope"] = params["hill_slope"]

    fits = {"naive": naive_draws, "controlled": controlled_draws, "calibrated": calibrated_draws}
    scorecard = build_scorecard(df, truth, fits)
    calibrated_share = (
        scorecard.set_index("channel").loc["field", "calibrated_weekly_contribution"]
        / scorecard["calibrated_weekly_contribution"].sum()
    )

    diagnostics = channel_identifiability_diagnostics(df, controlled_draws)
    holdout_validation = build_holdout_validation(df, holdout_weeks=12, measurements=measurements)
    health = model_health_scorecard(fits, scorecard, holdout_validation)
    decision_record = measurement_decision_record(diagnostics, scorecard, calibrated_draws)
    ladder = model_ladder_scorecard(scorecard, decision_record)
    method_map = build_measurement_method_map()
    comparability_checks = build_method_comparability_checks(measurements, geo_holdout)
    evidence_record = build_channel_evidence_record(measurements, geo_holdout, decision_record, scorecard, comparability_checks)
    current_spends = np.array([df[f"spend_{ch}"].mean() for ch in CHANNELS])
    current_budget = float(current_spends.sum())
    guardrails = build_measurement_guardrails(evidence_record, decision_record, current_spends)
    guardrail_by_channel = guardrails.set_index("channel")
    evidence_record["budget_rule"] = evidence_record["channel"].map(guardrail_by_channel["allowed_budget_move"])
    next_agenda = build_next_measurement_agenda(evidence_record, decision_record, current_spends, guardrails)
    calib_trace = calibration_trace(df, controlled_draws, calibrated_draws, geo_holdout)
    prior_sensitivity = build_prior_sensitivity(df, truth, controlled_draws, calibrated_draws, geo_holdout)
    pareto_front = build_pareto_front(df, truth, calibrated_draws)

    geo_calibration = pd.DataFrame([{
        "geo_holdout_n_geos": geo_holdout["n_geos"],
        "geo_holdout_input_level_calls": geo_holdout["input_level"],
        "geo_holdout_delta_input_calls": geo_holdout["delta_input"],
        "geo_holdout_mean_incremental_nrx": geo_holdout["mean_incremental_nrx"],
        "geo_holdout_sd_incremental_nrx": geo_holdout["sd_incremental_nrx"],
        "current_field_calls_per_week": round(float(df["calls_field"].mean()), 2),
        "current_field_spend_per_week": round(float(df["spend_field"].mean()), 1),
        "controlled_posterior_field_contribution": round(_channel_contributions(df, controlled_draws)["field"], 2),
        "calibrated_posterior_field_contribution": round(_channel_contributions(df, calibrated_draws)["field"], 2),
        "true_field_contribution": round(float(truth.set_index("channel").loc["field", "true_mean_weekly_contribution"]), 2),
        "fallback_itt_relative_lift": round(itt_prior["relative_lift"], 4),
        "fallback_itt_relative_lift_se": round(itt_prior["relative_lift_se"], 4),
        "fallback_itt_baseline_weekly_nrx_proxy": round(float(itt_prior["baseline_weekly_nrx"]), 2),
        "fallback_itt_average_contribution_prior_mean": round(float(itt_prior["mean"]), 2),
        "fallback_itt_average_contribution_prior_sd": round(float(itt_prior["sd"]), 2),
        "fallback_itt_note": "Average-contribution translation of the ch10 account-cycle ITT; documented appendix path, not used to fit calibrated_draws.",
    }])

    response_curves_df = build_response_curves(df, calibrated_draws)
    mroi = compute_marginal_roi(df, calibrated_draws)
    saturation = find_saturation_points(df, calibrated_draws)
    decision_status = dict(zip(decision_record["channel"], decision_record["decision_status"]))
    budget_table = budget_optimisation(df, calibrated_draws, decision_status=decision_status, guardrails=guardrails)

    mean_opt_spends_same_budget = optimal_allocation_at_budget(
        current_spends, current_budget, calibrated_draws, CHANNELS, decision_status, guardrails,
    )
    allocation_summary, allocation_by_draw = optimal_allocation_by_draw(
        current_spends,
        current_budget,
        calibrated_draws,
        CHANNELS,
        decision_status=decision_status,
        guardrails=guardrails,
    )
    median_opt_spends_same_budget = allocation_summary.set_index("channel").loc[CHANNELS, "optimized_weekly_spend_median"].to_numpy(dtype=float)
    decision = pd.DataFrame([evaluate_reallocation(current_spends, median_opt_spends_same_budget, calibrated_draws, CHANNELS)])
    mmm_recommendation = build_mmm_budget_recommendation(current_spends, mean_opt_spends_same_budget, allocation_summary, mroi, decision_status, guardrails)
    unified_recommendation = build_unified_budget_recommendation(mmm_recommendation, evidence_record, guardrails)

    reconciliation = build_reconciliation_table(measurements, float(calibrated_share))

    return {
        "synthetic_weekly_data": df,
        "true_channel_contribution": truth,
        "posterior_summary_naive": posterior_summary(naive_draws, true_lookup),
        "posterior_summary_controlled": posterior_summary(controlled_draws, true_lookup),
        "posterior_summary_calibrated": posterior_summary(calibrated_draws, true_lookup),
        "contribution_scorecard": scorecard,
        "channel_identifiability_diagnostics": diagnostics,
        "model_health_scorecard": health,
        "model_ladder_scorecard": ladder,
        "calibration_trace": calib_trace,
        "prior_sensitivity": prior_sensitivity,
        "pareto_front": pareto_front,
        "measurement_decision_record": decision_record,
        "measurement_method_map": method_map,
        "method_comparability_checks": comparability_checks,
        "channel_evidence_record": evidence_record,
        "measurement_guardrails": guardrails,
        "next_measurement_agenda": next_agenda,
        "geo_holdout_calibration": geo_calibration,
        "reconciliation_three_numbers": reconciliation,
        "holdout_validation": holdout_validation,
        "response_curves": response_curves_df,
        "marginal_roi": mroi,
        "saturation_points": saturation,
        "budget_optimisation": budget_table,
        "allocation_distribution": allocation_summary,
        "allocation_distribution_draws": allocation_by_draw,
        "reallocation_decision": decision,
        "mmm_budget_recommendation": mmm_recommendation,
        "unified_budget_recommendation": unified_recommendation,
        "_naive_draws": naive_draws,
        "_controlled_draws": controlled_draws,
        "_calibrated_draws": calibrated_draws,
        "_current_spends": current_spends,
        "_mean_opt_spends_same_budget": mean_opt_spends_same_budget,
        "_median_opt_spends_same_budget": median_opt_spends_same_budget,
        "_measurements": pd.DataFrame([measurements]),
    }


def write_weekly_series_figure(df: pd.DataFrame, figures_dir: Path) -> None:
    """Write Figure 13.1 with observed NRx above channel spend."""
    spend_columns = [c for c in df.columns if c.startswith("spend_")]
    channel_names = [c.removeprefix("spend_") for c in spend_columns]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.42, 0.58],
        subplot_titles=("Observed weekly NRx", "Weekly channel spend"),
    )
    fig.add_trace(
        go.Scatter(
            x=df["week_index"],
            y=df["nrx"],
            mode="lines",
            name="Observed NRx",
            line={"color": "#222222", "width": 3},
        ),
        row=1,
        col=1,
    )
    for spend_col, channel in zip(spend_columns, channel_names, strict=True):
        fig.add_trace(
            go.Scatter(
                x=df["week_index"],
                y=df[spend_col],
                mode="lines",
                name=channel.replace("_", " ").title(),
                line={"color": CHANNEL_PALETTE.get(channel, "#666666"), "width": 2.5},
            ),
            row=2,
            col=1,
        )
    for row in (1, 2):
        fig.add_vline(x=KNOWN_EVENT_WEEK, line_dash="dash", line_color="#666666", row=row, col=1)
    fig.add_annotation(
        x=KNOWN_EVENT_WEEK,
        y=260,
        text="Formulary access win",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#666666",
        ax=90,
        ay=20,
        font={"size": 12, "color": "#444444"},
        row=1,
        col=1,
    )
    digital_start, digital_end = DIGITAL_PRE_EVENT_TEST_WEEKS[0]
    paid_start, paid_end = PAID_MEDIA_DARK_WEEKS[0]
    fig.add_vrect(
        x0=digital_start,
        x1=digital_end - 1,
        fillcolor=CHANNEL_PALETTE["digital"],
        opacity=0.10,
        line_width=0,
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=(digital_start + digital_end - 1) / 2,
        y=250,
        text="Digital test flight",
        showarrow=False,
        font={"size": 12, "color": CHANNEL_PALETTE["digital"]},
        row=2,
        col=1,
    )
    fig.add_vrect(
        x0=paid_start,
        x1=paid_end - 1,
        fillcolor=CHANNEL_PALETTE["paid_media"],
        opacity=0.12,
        line_width=0,
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=(paid_start + paid_end - 1) / 2,
        y=250,
        text="Paid-media dark interval",
        showarrow=False,
        font={"size": 12, "color": CHANNEL_PALETTE["paid_media"]},
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        title="Weekly NRx and channel spend",
        legend_title_text="Series",
        hovermode="x unified",
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    fig.update_yaxes(title_text="NRx", row=1, col=1)
    fig.update_yaxes(title_text="Weekly spend ($)", row=2, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.write_image(str(figures_dir / "figure_13_2_weekly_spend.png"), width=1100, height=720)
    fig.write_image(str(figures_dir / "figure_13_2_weekly_spend.svg"), width=1100, height=720)


def write_outputs(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    from build_channel_permission_idea import write_channel_permission_from_frame

    for name, frame in results.items():
        if name.startswith("_"):
            continue
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    df = results["synthetic_weekly_data"]
    write_system_map(figures_dir)
    write_weekly_series_figure(df, figures_dir)

    week_grid = np.arange(15)
    fig_adstock = go.Figure()
    decay_colors = {0.1: "#009E73", 0.4: "#0072B2", 0.7: "#D55E00"}
    for decay, color in decay_colors.items():
        impulse = np.zeros(15)
        impulse[0] = 1.0
        ads = _adstock(impulse, decay)
        fig_adstock.add_trace(go.Scatter(x=week_grid, y=ads, mode="lines+markers", name=f"decay={decay:.1f}",
                                          line={"color": color}))
    fig_adstock.update_layout(template="plotly_white", title="Adstock: how one week's spend echoes forward",
                               xaxis_title="Weeks after a single $1 of spend", yaxis_title="Remaining adstock effect")
    fig_adstock.write_image(str(figures_dir / "figure_13_3_adstock_decay.png"), width=1000, height=500)
    fig_adstock.write_image(str(figures_dir / "figure_13_3_adstock_decay.svg"), width=1000, height=500)

    spend_grid = np.linspace(0, 300, 200)
    fig_hill = go.Figure()
    slope_colors = {1.0: "#009E73", 2.0: "#0072B2", 4.0: "#D55E00"}
    for slope, color in slope_colors.items():
        response = _hill(spend_grid, 100.0, slope)
        fig_hill.add_trace(go.Scatter(x=spend_grid, y=response, mode="lines", name=f"slope={slope:.0f}",
                                       line={"color": color}))
    fig_hill.add_vline(x=100, line_dash="dash", line_color="gray")
    fig_hill.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_hill.update_layout(template="plotly_white", title="Hill saturation: diminishing returns, EC50 = 100",
                            xaxis_title="Spend", yaxis_title="Fraction of maximum response")
    fig_hill.write_image(str(figures_dir / "figure_13_4_hill_saturation.png"), width=1000, height=500)
    fig_hill.write_image(str(figures_dir / "figure_13_4_hill_saturation.svg"), width=1000, height=500)

    scorecard = results["contribution_scorecard"]
    channel_labels = [c.replace("_", " ").title() for c in scorecard["channel"]]
    fit_styles = {
        "naive": ("Naive fit", FIT_PALETTE["naive"]),
        "controlled": ("Controlled fit", FIT_PALETTE["controlled"]),
        "calibrated": ("Calibrated fit", FIT_PALETTE["calibrated"]),
    }

    fig_fit = go.Figure()
    y_positions = {ch: i for i, ch in enumerate(scorecard["channel"])}
    y_tickvals = [y_positions[ch] for ch in scorecard["channel"]]
    y_ticktext = channel_labels
    y_offsets = {"naive": 0.16, "controlled": 0.00, "calibrated": -0.16}

    # Directional step arrows (naive -> controlled -> calibrated) replace a
    # plain connecting line so the fix-by-fix improvement reads as a path,
    # not just three unordered dots on a row.
    step_order = [("naive", "controlled"), ("controlled", "calibrated")]
    for row in scorecard.itertuples(index=False):
        y = y_positions[str(row.channel)]
        for from_step, to_step in step_order:
            x_from = getattr(row, f"{from_step}_pct_error")
            x_to = getattr(row, f"{to_step}_pct_error")
            _, to_color = fit_styles[to_step]
            fig_fit.add_annotation(
                x=x_to,
                y=y + y_offsets[to_step],
                ax=x_from,
                ay=y + y_offsets[from_step],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=2.0,
                arrowcolor=to_color,
                opacity=0.55,
            )
    for fit_name, (fit_label, color) in fit_styles.items():
        error_col = f"{fit_name}_pct_error"
        fig_fit.add_trace(go.Scatter(
            x=scorecard[error_col],
            y=[y_positions[ch] + y_offsets[fit_name] for ch in scorecard["channel"]],
            mode="markers",
            name=fit_label,
            marker={"color": color, "size": 17, "line": {"color": "white", "width": 1.5}},
            text=[f"{fit_label}: {v:+.1f}%" for v in scorecard[error_col]],
            cliponaxis=False,
            showlegend=False,
        ))
    value_columns = [("naive", 0.79), ("controlled", 0.90), ("calibrated", 1.02)]
    for fit_name, x_paper in value_columns:
        fit_label, color = fit_styles[fit_name]
        fig_fit.add_annotation(
            x=x_paper,
            y=-0.63,
            xref="paper",
            yref="y",
            text=f"<b>{fit_label.replace(' fit', '')}</b>",
            showarrow=False,
            font={"size": 16, "color": color},
            align="center",
        )
        for row in scorecard.itertuples(index=False):
            value = getattr(row, f"{fit_name}_pct_error")
            fig_fit.add_annotation(
                x=x_paper,
                y=y_positions[str(row.channel)],
                xref="paper",
                yref="y",
                text=f"{value:+.1f}%",
                showarrow=False,
                font={"size": 16, "color": color},
                align="center",
            )
    fig_fit.add_vline(x=0, line_color="#333333", line_width=2)
    fig_fit.add_annotation(
        x=0,
        y=1.08,
        xref="x",
        yref="paper",
        text="correct recovery",
        showarrow=False,
        font={"size": 13, "color": "#333333"},
    )
    max_abs_error = float(np.ceil(np.abs(scorecard[["naive_pct_error", "controlled_pct_error", "calibrated_pct_error"]]).to_numpy().max() / 10) * 10)
    fig_fit.update_layout(
        template="plotly_white",
        title={"text": "Contribution Error Ladder Against Ground Truth", "font": {"size": 22}},
        xaxis_title="Percent error against true weekly NRx contribution",
        yaxis_title="Channel",
        xaxis_range=[-max_abs_error - 8, max_abs_error + 8],
        yaxis_range=[len(scorecard) - 0.55, -0.72],
        legend_title_text="Fit step",
        font={"size": 15, "family": "Arial, Helvetica, sans-serif", "color": "#1f2937"},
        margin={"l": 130, "r": 300, "t": 100, "b": 80},
    )
    fig_fit.update_xaxes(domain=[0.0, 0.72], title_font={"size": 16}, tickfont={"size": 14})
    fig_fit.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext, title_font={"size": 16}, tickfont={"size": 15})
    fig_fit.write_image(str(figures_dir / "figure_13_5_naive_vs_truth.png"), width=1300, height=680, scale=2)
    fig_fit.write_image(str(figures_dir / "figure_13_5_naive_vs_truth.svg"), width=1300, height=680)

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"

    curves = results["response_curves"]
    weekly = results["synthetic_weekly_data"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="Observed min-max",
        line={"color": "rgba(75,85,99,0.45)", "width": 9},
        hoverinfo="skip",
    ))
    fig3.add_trace(go.Scatter(
        x=[None, None],
        y=[None, None],
        mode="lines",
        name="Observed p10-p90",
        line={"color": "#4b5563", "width": 13},
        hoverinfo="skip",
    ))
    fig3.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="80% saturation point",
        marker={"symbol": "x", "size": 15, "color": "#111827", "line": {"color": "white", "width": 1.5}},
        hoverinfo="skip",
    ))
    fig3.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="Current weekly mean",
        marker={"symbol": "diamond", "size": 16, "color": "#111827", "line": {"color": "white", "width": 2}},
        hoverinfo="skip",
    ))
    for idx, ch in enumerate(CHANNELS):
        sub = curves[curves["channel"] == ch]
        fig3.add_trace(go.Scatter(x=sub["weekly_spend"], y=sub["p90_nrx_contribution"], line={"width": 0},
                                   showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=sub["weekly_spend"], y=sub["p10_nrx_contribution"], fill="tonexty",
                                   line={"width": 0}, showlegend=False, hoverinfo="skip",
                                   fillcolor=_hex_to_rgba(CHANNEL_PALETTE[ch], 0.15)))
        sat_row = results["saturation_points"].set_index("channel").loc[ch]
        sat_spend = float(sat_row["saturation_spend_median"])
        spend_col = "spend_field" if ch == "field" else f"spend_{ch}"
        observed_spend = weekly[spend_col].to_numpy(dtype=float)
        observed_min = float(observed_spend.min())
        observed_p10 = float(np.percentile(observed_spend, 10))
        observed_p90 = float(np.percentile(observed_spend, 90))
        observed_max = float(observed_spend.max())
        mean_curve = sub["mean_nrx_contribution"].to_numpy(dtype=float)
        spend_axis = sub["weekly_spend"].to_numpy(dtype=float)
        fig3.add_trace(go.Scatter(
            x=sub["weekly_spend"],
            y=sub["mean_nrx_contribution"],
            name=ch.replace("_", " ").title(),
            line={"color": CHANNEL_PALETTE[ch], "width": 3.2},
            showlegend=False,
        ))
        min_mask = (spend_axis >= observed_min) & (spend_axis <= observed_max)
        core_mask = (spend_axis >= observed_p10) & (spend_axis <= observed_p90)
        fig3.add_trace(go.Scatter(
            x=spend_axis[min_mask],
            y=mean_curve[min_mask],
            mode="lines",
            line={"color": _hex_to_rgba(CHANNEL_PALETTE[ch], 0.42), "width": 10},
            showlegend=False,
            hovertemplate=(
                f"{ch.replace('_', ' ').title()} observed min-max"
                "<br>weekly spend: %{x:.1f}<br>expected contribution: %{y:.1f}<extra></extra>"
            ),
        ))
        fig3.add_trace(go.Scatter(
            x=spend_axis[core_mask],
            y=mean_curve[core_mask],
            mode="lines",
            line={"color": CHANNEL_PALETTE[ch], "width": 15},
            showlegend=False,
            hovertemplate=(
                f"{ch.replace('_', ' ').title()} observed p10-p90"
                "<br>weekly spend: %{x:.1f}<br>expected contribution: %{y:.1f}<extra></extra>"
            ),
        ))
        # Current-spend diamond and 80% saturation X are drawn last, on top
        # of the observed-range bands, in a neutral dark color so they read
        # clearly against every curve instead of blending into its own hue.
        current_spend = float(results["_current_spends"][CHANNELS.index(ch)])
        current_response = float(np.interp(current_spend, spend_axis, mean_curve))
        fig3.add_trace(go.Scatter(
            x=[current_spend, current_spend],
            y=[current_response - 4.0, current_response + 4.0],
            mode="lines",
            line={"color": "white", "width": 6},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig3.add_trace(go.Scatter(
            x=[current_spend],
            y=[current_response],
            mode="markers",
            name=f"{ch.replace('_', ' ').title()} current spend",
            marker={"symbol": "diamond", "size": 20, "color": "#111827", "line": {"color": "white", "width": 2.5}},
            showlegend=False,
        ))
        sat_marker_x = min(sat_spend, float(spend_axis.max()))
        sat_response = float(np.interp(sat_marker_x, spend_axis, mean_curve))
        fig3.add_trace(go.Scatter(
            x=[sat_marker_x],
            y=[sat_response],
            mode="markers",
            marker={"symbol": "x", "size": 18, "color": "#111827", "line": {"color": "white", "width": 1.5}},
            showlegend=False,
            hovertemplate=(
                f"{ch.replace('_', ' ').title()} 80% saturation"
                "<br>weekly spend: %{x:.1f}<br>expected contribution: %{y:.1f}<extra></extra>"
            ),
        ))
        label_x_map = {"field": 305.0, "paid_media": 92.0, "digital": 142.0, "email": 9.0}
        label_x = min(label_x_map.get(ch, float(observed_p90)), float(spend_axis.max() * 0.92))
        label_y = float(np.interp(label_x, spend_axis, mean_curve))
        x_shift_map = {"field": 34, "paid_media": -26, "digital": 52, "email": -18}
        y_shift_map = {"field": 64, "paid_media": 82, "digital": -28, "email": 90}
        fig3.add_annotation(
            x=label_x,
            y=label_y,
            text=ch.replace("_", " ").title(),
            showarrow=False,
            xshift=x_shift_map.get(ch, 18),
            yshift=y_shift_map.get(ch, 8),
            font={"size": 20, "color": CHANNEL_PALETTE[ch]},
            bgcolor="rgba(255,255,255,0.80)",
        )
    fig3.update_layout(
        template="plotly_white",
        title={"text": "Calibrated Response Curves With Observed Spend Ranges", "x": 0.02, "xanchor": "left"},
        xaxis_title="Weekly spend ($)",
        yaxis_title="Expected weekly NRx contribution",
        font={"family": "Arial, Helvetica, sans-serif", "size": 18, "color": "#1f2937"},
        title_font={"size": 30, "color": "#111827"},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.965,
            "xanchor": "left",
            "x": 0.025,
            "font": {"size": 14},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(203,213,225,0.9)",
            "borderwidth": 1,
            "traceorder": "normal",
        },
        margin={"l": 90, "r": 40, "t": 115, "b": 80},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig3.update_xaxes(
        showline=True,
        linewidth=1.2,
        linecolor="#9ca3af",
        gridcolor="rgba(203,213,225,0.45)",
        tickfont={"size": 18},
        title_font={"size": 24},
        zeroline=False,
    )
    fig3.update_yaxes(
        showline=True,
        linewidth=1.2,
        linecolor="#9ca3af",
        gridcolor="rgba(203,213,225,0.45)",
        tickfont={"size": 18},
        title_font={"size": 24},
        zeroline=False,
    )
    fig3.write_image(str(figures_dir / "figure_13_7_response_curves.png"), width=1280, height=760)
    fig3.write_image(str(figures_dir / "figure_13_7_response_curves.svg"), width=1280, height=760)
    fig3.write_html(str(output_dir / "response_curves.html"), include_plotlyjs="cdn")

    pareto = results["pareto_front"]
    fig4b = go.Figure()
    non_front = pareto[~pareto["pareto_efficient"]]
    front = pareto[pareto["pareto_efficient"]]
    best_fit_cutoff = float(pareto["nrmse"].quantile(0.20))
    fig4b.add_vrect(
        x0=float(pareto["nrmse"].min()),
        x1=best_fit_cutoff,
        fillcolor="#EEF2F6",
        opacity=0.9,
        line_width=0,
        layer="below",
    )
    fig4b.add_trace(go.Scatter(
        x=non_front["nrmse"], y=non_front["decomp_rssd"], mode="markers", name="Posterior draws",
        marker={"color": "#C7D0DA", "size": 6, "opacity": 0.55, "line": {"width": 0}},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig4b.add_trace(go.Scatter(
        x=front["nrmse"], y=front["decomp_rssd"], mode="markers+lines", name="Pareto front",
        marker={"color": "#1F2937", "size": 9, "symbol": "diamond"},
        line={"color": "#1F2937", "dash": "dot", "width": 1.6},
        text=[f"digital error {v:+.1f}%" for v in front["digital_pct_error"]], hoverinfo="x+y+text",
        showlegend=False,
    ))

    # Two "twin" draws: essentially identical in-sample fit, opposite stories
    # about digital's true contribution (field's own contribution is tight in
    # every draw here because the geo-holdout calibration already anchors it;
    # digital carries the remaining disagreement because it is still
    # uncalibrated). That contradiction is the whole point of the chart, so
    # it is called out directly instead of left for the reader to spot inside
    # a cloud of 2,400 points.
    twins = pareto[pareto["draw"].isin([117, 672])].set_index("draw")
    fig4b.add_trace(go.Scatter(
        x=twins["nrmse"], y=twins["decomp_rssd"], mode="markers",
        marker={"color": "#D55E00", "size": 16, "line": {"color": "white", "width": 2}},
        hoverinfo="skip", showlegend=False,
    ))
    fig4b.add_annotation(
        x=float(twins.loc[117, "nrmse"]), y=float(twins.loc[117, "decomp_rssd"]),
        text="<b>Draw 117</b><br>Digital over-credited 31.7%",
        showarrow=True, arrowhead=2, arrowcolor="#D55E00", ax=75, ay=-60,
        font={"size": 13, "color": "#D55E00"}, align="left",
        bgcolor="white", bordercolor="#D55E00", borderwidth=1, borderpad=4,
    )
    fig4b.add_annotation(
        x=float(twins.loc[672, "nrmse"]), y=float(twins.loc[672, "decomp_rssd"]),
        text="<b>Draw 672</b><br>Digital under-credited 19.8%",
        showarrow=True, arrowhead=2, arrowcolor="#D55E00", ax=150, ay=50,
        font={"size": 13, "color": "#D55E00"}, align="left",
        bgcolor="white", bordercolor="#D55E00", borderwidth=1, borderpad=4,
    )
    fig4b.add_annotation(
        x=(float(pareto["nrmse"].min()) + best_fit_cutoff) / 2,
        y=float(pareto["decomp_rssd"].max()),
        text="best-fitting 20%",
        showarrow=False,
        font={"size": 12, "color": "#52606D"},
    )
    # Label the Pareto front as a line-style swatch (short dotted segment +
    # diamond, matching the front's own line style) rather than an arrow
    # pointing at one marker — the front is the whole frontier, not any
    # single draw on it.
    nrmse_min = float(pareto["nrmse"].min())
    swatch_x0 = nrmse_min + 0.05 * (best_fit_cutoff - nrmse_min)
    swatch_x1 = nrmse_min + 0.55 * (best_fit_cutoff - nrmse_min)
    swatch_y = float(pareto["decomp_rssd"].min()) + 0.02 * (
        float(pareto["decomp_rssd"].max()) - float(pareto["decomp_rssd"].min())
    )
    fig4b.add_shape(
        type="line", xref="x", yref="y",
        x0=swatch_x0, x1=swatch_x1, y0=swatch_y, y1=swatch_y,
        line={"color": "#1F2937", "dash": "dot", "width": 2},
    )
    fig4b.add_trace(go.Scatter(
        x=[(swatch_x0 + swatch_x1) / 2], y=[swatch_y], mode="markers",
        marker={"color": "#1F2937", "size": 8, "symbol": "diamond"},
        hoverinfo="skip", showlegend=False,
    ))
    fig4b.add_annotation(
        x=swatch_x1, y=swatch_y, xanchor="left", yanchor="middle",
        text="  Pareto front", showarrow=False,
        font={"size": 13, "color": "#1F2937"},
    )
    fig4b.add_annotation(
        x=0, y=-0.21, xref="paper", yref="paper", xanchor="left",
        text="Across all 2,400 draws, digital's contribution error ranges from −94% to +64%.",
        showarrow=False,
        font={"size": 12, "color": "#52606D"},
    )
    fig4b.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>Same Fit Score, Different Story</b><br>"
                "<span style='font-size:14px;color:#6B7280'>2,400 posterior draws fit weekly NRx almost equally well "
                "— where they disagree is which channel gets the credit</span>"
            ),
            "x": 0.02, "xanchor": "left", "font": {"size": 23},
        },
        xaxis_title="In-sample NRMSE (lower = better fit)",
        yaxis_title="decomp.RSSD (lower = spend share matches effect share)",
        font={"size": 15, "family": "Arial, Helvetica, sans-serif", "color": "#1f2937"},
        margin={"l": 85, "r": 60, "t": 115, "b": 110},
    )
    fig4b.update_xaxes(title_font={"size": 15}, tickfont={"size": 13})
    fig4b.update_yaxes(title_font={"size": 15}, tickfont={"size": 13})
    fig4b.write_image(str(figures_dir / "figure_13_6_pareto_front.png"), width=1150, height=650, scale=2)
    fig4b.write_image(str(figures_dir / "figure_13_6_pareto_front.svg"), width=1150, height=650)

    write_channel_permission_from_frame(
        results["unified_budget_recommendation"],
        figures_dir,
        stem="figure_13_8_channel_permission",
    )

    # ── Figure 13.9: measurement loop calendar ────────────────────────────
    row_order = [
        "Planning decision", "MMM refresh", "Field calibration refresh",
        "Paid media holdout", "Digital geo test", "Attribution monitoring",
    ]
    fig9 = go.Figure()
    for q, (q_start, q_end) in enumerate([(0, 13), (13, 26), (26, 39), (39, 52)]):
        fig9.add_shape(
            type="rect", xref="x", yref="paper", x0=q_start, x1=q_end, y0=0, y1=1,
            fillcolor="#F7F8FA" if q % 2 == 0 else "white", line={"width": 0}, layer="below",
        )
        fig9.add_annotation(
            x=(q_start + q_end) / 2, y=0, yref="paper", yanchor="top", yshift=-10,
            text=f"<b>Q{q + 1}</b>", showarrow=False, font={"size": 14, "color": "#9CA3AF"},
        )
    # Attribution runs every week: a burst of weekly ticks reads as a
    # continuous weekly cadence, where one solid bar reads as a single block.
    fig9.add_trace(go.Scatter(
        x=list(range(1, 53)), y=["Attribution monitoring"] * 52, mode="markers",
        marker={"symbol": "line-ns", "size": 14, "line": {"color": "#9CA3AF", "width": 2.2}},
        name="Attribution (weekly)",
    ))
    # The field-refresh cycle sits later in the year than the digital and
    # paid-media cycles, so it is shifted earliest of the three toward the
    # front of Q4: its readout must clear before the final MMM refresh and
    # planning decision, which in turn must both land before the week-51
    # budget lock, not after it. Colors match CHANNEL_PALETTE used everywhere
    # else in the chapter, so this calendar reads as the same channels, not a
    # new color language.
    planned_windows = [
        ("Digital geo test", 9, 7, CHANNEL_PALETTE["digital"], "Digital geo test", "digital band decision"),
        ("Paid media holdout", 25, 7, CHANNEL_PALETTE["paid_media"], "Paid media holdout", "paid-media band decision"),
        ("Field calibration refresh", 40, 5, CHANNEL_PALETTE["field"], "Field refresh", "field band refresh"),
    ]
    readout_x, readout_y = [], []
    for label, start, length, color, legend_name, consequence in planned_windows:
        fig9.add_trace(go.Bar(
            y=[label], x=[length], base=[start], orientation="h", width=0.6,
            marker={"color": color, "line": {"color": "white", "width": 1.5}},
            name=legend_name, showlegend=True,
        ))
        readout_week = start + length + 2
        readout_x.append(readout_week)
        readout_y.append(label)
        fig9.add_annotation(
            x=readout_week + 1.1,
            y=label,
            text=consequence,
            showarrow=False,
            xanchor="left",
            font={"size": 11.5, "color": "#4B5563"},
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(209,213,219,0.9)",
            borderwidth=1,
            borderpad=3,
        )
    fig9.add_trace(go.Scatter(
        x=readout_x, y=readout_y, mode="markers",
        marker={"symbol": "circle-open", "size": 15, "color": "#374151", "line": {"width": 2.5}},
        name="Test readout",
    ))
    fig9.add_trace(go.Scatter(
        x=[18, 35, 47], y=["MMM refresh"] * 3, mode="markers",
        marker={"symbol": "diamond", "size": 16, "color": "#E69F00", "line": {"color": "white", "width": 1.5}},
        name="MMM refresh",
    ))
    fig9.add_trace(go.Scatter(
        x=[20, 37, 49], y=["Planning decision"] * 3, mode="markers",
        marker={"symbol": "star", "size": 21, "color": "#D55E00", "line": {"color": "white", "width": 1.5}},
        name="Planning decision",
    ))
    fig9.add_vline(x=51, line_dash="dash", line_color="#8A3E3E", line_width=2)
    fig9.add_annotation(
        x=51, y=1.0, yref="paper", yanchor="bottom", xanchor="right",
        text="<b>annual budget lock</b> (wk 51)", showarrow=False,
        font={"size": 14, "color": "#8A3E3E"},
    )

    fig9.update_xaxes(
        range=[-1, 55], showticklabels=False, showgrid=False, title=None, zeroline=False,
    )
    fig9.update_yaxes(
        tickmode="array", tickvals=row_order, ticktext=row_order,
        categoryorder="array", categoryarray=row_order,
        tickfont={"size": 15.5}, showgrid=False,
    )
    fig9.update_layout(
        template="plotly_white", barmode="overlay",
        title={
            "text": (
                "<b>The Measurement Agenda</b><br>"
                "<span style='font-size:14px;color:#6B7280'>Three tests reset a channel's budget permission each "
                "quarter, on a rhythm that repeats every planning year</span>"
            ),
            "x": 0.02, "xanchor": "left", "font": {"size": 23},
        },
        font={"family": "Arial, Helvetica, sans-serif", "size": 14, "color": "#1f2937"},
        legend={
            "orientation": "h", "yanchor": "top", "y": -0.14, "x": 0.5, "xanchor": "center",
            "font": {"size": 13}, "title": None, "traceorder": "normal",
        },
        margin={"l": 190, "r": 30, "t": 110, "b": 90},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig9.write_image(str(figures_dir / "figure_13_9_measurement_loop_calendar.png"), width=1150, height=580, scale=2)
    fig9.write_image(str(figures_dir / "figure_13_9_measurement_loop_calendar.svg"), width=1150, height=580)


def write_table_outputs(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write analysis tables only. This is the model-fitting output step."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        if name.startswith("_"):
            continue
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def load_results_for_figures(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load existing CSV outputs needed for figures, without refitting models."""
    required = [
        "synthetic_weekly_data",
        "contribution_scorecard",
        "response_curves",
        "saturation_points",
        "pareto_front",
        "measurement_guardrails",
        "allocation_distribution",
        "unified_budget_recommendation",
    ]
    results = {name: pd.read_csv(output_dir / f"{name}.csv") for name in required}
    weekly = results["synthetic_weekly_data"]
    results["_current_spends"] = np.array([weekly[f"spend_{ch}"].mean() for ch in CHANNELS])
    return results


def write_figures_from_outputs(output_dir: Path) -> None:
    """Write figures from saved CSV outputs only. This does not fit models."""
    results = load_results_for_figures(output_dir)
    write_outputs(results, output_dir)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "assets" / "generated_outputs"
    results = run_analysis()
    write_table_outputs(results, out_dir)
    write_figures_from_outputs(out_dir)
    print(f"Wrote MMM chapter outputs to {out_dir}")
