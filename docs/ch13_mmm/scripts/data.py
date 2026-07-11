"""Synthetic data generation for the MMM and unified measurement chapter.

Contains ground-truth parameters (used only to generate data and to score
posterior recovery, never to inform priors), the weekly spend and NRx
generator, and the planted digital/formulary-event confound used in the
naive-vs-controlled demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 20260610
CHANNELS = ["field", "email", "digital", "paid_media"]

# Ground-truth structural parameters used to generate the dataset.
# Hill EC50 is in raw weekly-spend dollars (not normalised), so it sits
# near a channel's typical spend level. Never imported by model.py.
_TRUE_PARAMS = {
    "field": {
        "adstock_decay": 0.40,
        "hill_ec50": 2.7,
        "hill_slope": 2.0,
        "coefficient": 120.0,
    },
    "email": {
        "adstock_decay": 0.20,
        "hill_ec50": 32.0,
        "hill_slope": 1.5,
        "coefficient": 40.0,
    },
    "digital": {
        "adstock_decay": 0.10,
        "hill_ec50": 95.0,
        "hill_slope": 1.8,
        "coefficient": 60.0,
    },
    "paid_media": {
        "adstock_decay": 0.50,
        "hill_ec50": 135.0,
        "hill_slope": 2.5,
        "coefficient": 90.0,
    },
}

_TRUE_BASE0 = 50.0        # baseline NRx at week 0 with zero promotional spend
_TRUE_TREND = 0.12        # weekly linear drift (post-launch uptake)
_TRUE_SEASONAL_AMP = 6.0  # amplitude of the 52-week condition-seasonality cycle
_TRUE_SEASONAL_PHASE = -np.pi / 4
_TRUE_EVENT_WEEK = 60     # formulary access win (national payer coverage expansion)
_TRUE_EVENT_SIZE = 14.0   # permanent step-up in baseline NRx after the event
_TRUE_NOISE_SD = 8.0

# The calendar week of the formulary access win is operational knowledge a brand
# team has on hand (a payer decision date), unlike the channel structural
# parameters above. The model is allowed to condition on this constant; it may
# not condition on _TRUE_PARAMS, _TRUE_EVENT_SIZE, or any other underscored value.
KNOWN_EVENT_WEEK = _TRUE_EVENT_WEEK

FIELD_COST_PER_CALL = 85.0  # dollars per field call, used to convert call volume to spend


def exposure_column(channel: str) -> str:
    """Native model-input column for one channel."""
    return "calls_field" if channel == "field" else f"spend_{channel}"


def spend_to_exposure(channel: str, spend_value: float) -> float:
    """Convert budget dollars into the native model unit."""
    if channel == "field":
        return spend_value / FIELD_COST_PER_CALL
    return spend_value


def exposure_to_spend(channel: str, exposure_value: float) -> float:
    """Convert native model units back into budget dollars."""
    if channel == "field":
        return exposure_value * FIELD_COST_PER_CALL
    return exposure_value


def _adstock_transform(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock transform (internal helper for data generation)."""
    ads = np.zeros_like(spend)
    ads[0] = spend[0]
    for t in range(1, len(spend)):
        ads[t] = spend[t] + decay * ads[t - 1]
    return ads


def _hill_transform(x: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """Hill saturation function on raw spend units. x and ec50 share units."""
    x = np.clip(x, 0, None)
    return x ** slope / (ec50 ** slope + x ** slope)


def _baseline_series(weeks: np.ndarray) -> np.ndarray:
    """Baseline NRx before any promotional effect: trend + seasonality + one event."""
    trend = _TRUE_BASE0 + _TRUE_TREND * weeks
    seasonal = _TRUE_SEASONAL_AMP * np.sin(2 * np.pi * weeks / 52 + _TRUE_SEASONAL_PHASE)
    event = np.where(weeks >= _TRUE_EVENT_WEEK, _TRUE_EVENT_SIZE, 0.0)
    return trend + seasonal + event


def _window_indicator(weeks: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    """1.0 for weeks inside any half-open [start, end) window, else 0.0."""
    indicator = np.zeros(len(weeks), dtype=float)
    for start, end in windows:
        indicator = np.where((weeks >= start) & (weeks < end), 1.0, indicator)
    return indicator


# Independent campaign windows, chosen to avoid lining up with the formulary
# event (week 60), each other, or the channels' own seasonal cycles, so every
# channel has variation the model can use to learn its own response shape.
EMAIL_BURST_WEEKS = [(8, 11), (24, 27), (38, 41), (72, 75), (88, 91), (100, 103)]
PAID_MEDIA_DARK_WEEKS = [(45, 51)]
PAID_MEDIA_OFF_SEASON_PULSE_WEEKS = [(30, 34)]
DIGITAL_PRE_EVENT_TEST_WEEKS = [(18, 29)]


def generate_mmm_data(
    n_weeks: int = 104,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate weekly synthetic spend and NRx with a planted confound.

    Digital spend ramps up starting a few weeks before the formulary access
    win at week `_TRUE_EVENT_WEEK` and stays elevated after it, mirroring how
    a real brand team leans into digital once access improves. Baseline NRx
    also steps up at the same week because of the access change itself, not
    because of digital media. A model that omits the baseline event control
    will read part of that step as digital's contribution.

    Each channel also carries variation that is independent of that event and
    of the other channels' cycles, so the fitting sequence in 13.3-13.4 has a
    real chance at recovering each channel's response shape:

    - Email runs a low, near-flat base rate punctuated by short campaign
      bursts (`EMAIL_BURST_WEEKS`) at weeks that do not coincide with the
      event, digital's pre-event test, or paid media's flight cycle.
    - Paid media gets one dark interval (near-zero spend) and one off-season
      pulse, so it is not a pure function of the annual flight calendar.
    - Digital gets an early test flight (`DIGITAL_PRE_EVENT_TEST_WEEKS`) well
      before the event-linked ramp, giving the model a look at digital's own
      shape that is not entangled with the formulary confound.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

    field_seasonal = 1 + 0.40 * np.sin(2 * np.pi * weeks / 26)  # Q1/Q3 territory cycles
    digital_ramp = np.where(
        weeks < _TRUE_EVENT_WEEK - 4,
        0.0,
        np.clip((weeks - (_TRUE_EVENT_WEEK - 4)) / 10, 0, 1),
    )
    digital_pre_event_test = _window_indicator(weeks, DIGITAL_PRE_EVENT_TEST_WEEKS)
    paid_media_flight = np.where((weeks % 52) < 20, 1.35, 0.85)  # spring/summer flight
    paid_media_dark = _window_indicator(weeks, PAID_MEDIA_DARK_WEEKS)
    paid_media_off_season_pulse = _window_indicator(weeks, PAID_MEDIA_OFF_SEASON_PULSE_WEEKS)
    email_burst = _window_indicator(weeks, EMAIL_BURST_WEEKS)

    exposure_raw = {
        "field": 3.2 + 1.1 * field_seasonal + rng.normal(0, 0.16, n_weeks),
        "email": 12 + 46 * email_burst + rng.normal(0, 2.5, n_weeks),
        "digital": 55 + 70 * digital_ramp + 30 * digital_pre_event_test + rng.normal(0, 8, n_weeks),
        "paid_media": (
            110 * paid_media_flight * (1 - paid_media_dark)
            + 5 * paid_media_dark
            + 95 * paid_media_off_season_pulse
            + rng.normal(0, 12, n_weeks)
        ),
    }
    exposure_raw = {ch: np.clip(arr, 0, None) for ch, arr in exposure_raw.items()}

    nrx = _baseline_series(weeks)
    for channel, params in _TRUE_PARAMS.items():
        ads = _adstock_transform(exposure_raw[channel], params["adstock_decay"])
        saturated = _hill_transform(ads, params["hill_ec50"], params["hill_slope"])
        nrx += params["coefficient"] * saturated

    nrx += rng.normal(0, _TRUE_NOISE_SD, n_weeks)
    nrx = np.clip(nrx, 0, None)

    df = pd.DataFrame({"week": dates, "week_index": weeks, "nrx": nrx.round(1)})
    for ch, exposure in exposure_raw.items():
        if ch == "field":
            df["calls_field"] = exposure.round(2)
            df["spend_field"] = (exposure * FIELD_COST_PER_CALL).round(0)
        else:
            df[f"spend_{ch}"] = exposure.round(0)
    return df


def true_channel_share(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth average weekly NRx contribution and share, by channel.

    Used only to score posterior recovery and the naive/controlled/calibrated
    scorecard against the mechanism that actually generated the data. Never
    read by the fitting code.
    """
    weeks = df["week_index"].to_numpy()
    rows = []
    contributions = {}
    for channel, params in _TRUE_PARAMS.items():
        ads = _adstock_transform(df[exposure_column(channel)].to_numpy(dtype=float), params["adstock_decay"])
        sat = _hill_transform(ads, params["hill_ec50"], params["hill_slope"])
        contributions[channel] = params["coefficient"] * sat
    baseline_mean = float(_baseline_series(weeks).mean())
    total_contribution = baseline_mean + sum(c.mean() for c in contributions.values())
    for channel, contrib in contributions.items():
        rows.append({
            "channel": channel,
            "true_mean_weekly_contribution": round(float(contrib.mean()), 2),
            "true_contribution_share": round(float(contrib.mean() / total_contribution), 4),
        })
    rows.append({
        "channel": "baseline (trend + seasonality + access event)",
        "true_mean_weekly_contribution": round(baseline_mean, 2),
        "true_contribution_share": round(float(baseline_mean / total_contribution), 4),
    })
    return pd.DataFrame(rows)


GEO_HOLDOUT_SEED = RNG_SEED + 777
GEO_HOLDOUT_N_GEOS = 18            # independent geographies in the synthetic lift test
GEO_HOLDOUT_INPUT_LEVEL = 2.0      # baseline weekly calls/rep in the under-called treated geos
GEO_HOLDOUT_DELTA_CALLS = 2.5      # calls/rep added on top of that baseline in the treated geos
GEO_HOLDOUT_MEASUREMENT_NOISE = 0.22  # relative SD of one geo's noisy incremental read


def true_field_incremental_response(input_level: float, delta_input: float) -> float:
    """Ground-truth incremental field response between input_level +/- delta_input/2.

    Uses the same steady-state segment convention as `_geo_prior_penalty()` in
    model.py: a 20-week constant-exposure window at the low and high input
    level, run through field's true adstock and Hill curve. Used only to
    synthesize the geo-holdout calibration read below, never imported by the
    fitting code.
    """
    params = _TRUE_PARAMS["field"]
    lo = np.full(20, max(input_level - delta_input / 2, 0.0))
    hi = np.full(20, input_level + delta_input / 2)
    resp_lo = _hill_transform(_adstock_transform(lo, params["adstock_decay"]), params["hill_ec50"], params["hill_slope"]).mean()
    resp_hi = _hill_transform(_adstock_transform(hi, params["adstock_decay"]), params["hill_ec50"], params["hill_slope"]).mean()
    return float(params["coefficient"] * (resp_hi - resp_lo))


def generate_field_geo_holdout(
    input_level: float = GEO_HOLDOUT_INPUT_LEVEL,
    delta_input: float = GEO_HOLDOUT_DELTA_CALLS,
    n_geos: int = GEO_HOLDOUT_N_GEOS,
    measurement_noise: float = GEO_HOLDOUT_MEASUREMENT_NOISE,
    seed: int = GEO_HOLDOUT_SEED,
) -> dict[str, float]:
    """Synthesize a field-only geo-holdout lift test as calibration evidence.

    A brand team picks a sample of under-called geographies, averaging
    `input_level` calls per rep per week, and raises call frequency by
    `delta_input` there while holding the rest as control, reading the
    incremental NRx response once both arms reach steady state. Deliberately
    testing from a below-typical baseline (rather than the national panel
    mean) is what makes the read informative about field's response shape,
    not just its local slope at the level the brand already runs at. This is
    a synthetic stand-in for that experiment, clearly labeled as such: it
    measures field's own true response segment plus geo-level measurement
    noise, so the resulting standard error is realistic rather than zero.
    The reported mean and SD are the only things the calibration step in
    model.py is allowed to see; the true field parameters that generated
    them are not passed through.
    """
    rng = np.random.default_rng(seed)
    true_incremental = true_field_incremental_response(input_level, delta_input)
    geo_reads = rng.normal(true_incremental, abs(true_incremental) * measurement_noise, n_geos)
    mean_incremental_nrx = float(geo_reads.mean())
    sd_incremental_nrx = float(geo_reads.std(ddof=1) / np.sqrt(n_geos))
    return {
        "channel": "field",
        "n_geos": n_geos,
        "input_level": round(float(input_level), 3),
        "delta_input": round(float(delta_input), 3),
        "mean_incremental_nrx": round(mean_incremental_nrx, 4),
        "sd_incremental_nrx": round(max(sd_incremental_nrx, 1e-6), 4),
    }
