"""Shared constants for the Chapter 14 forecasting lifecycle.

Reused verbatim from earlier chapters (named constants, not derived scale
conversions): WASHOUT_DAYS from the patient-journey line-of-therapy rule and
ACCESS_PROBABILITY from the market-sizing access weights.

The Bass ceiling and the rest of the lifecycle parameters are this chapter's
own ground-truth assumptions, in the same order of magnitude as the 2,798
Roventra line-1 (new-to-therapy) entries already published in the
patient-journey chapter and the sample-scale market-sizing funnel, following
the same convention the marketing-mix-modeling chapter uses for its own
ground-truth adstock and saturation parameters. They are not forced into an
exact arithmetic identity with the market-sizing funnel, because that funnel
is NHANES-population-calibrated (millions) while this chapter forecasts at
the synthetic cohort's own raw sample scale (hundreds to low thousands),
which keeps every printed number concrete.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    """Load a module by explicit file path without touching sys.path.

    Several chapters' scripts directories contain a generically named
    module (every chapter has its own run_analysis.py, for example), so
    adding those directories to sys.path directly causes module-name
    collisions in whichever process imports more than one chapter's
    scripts. Loading by explicit path avoids that entirely.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lot = _load_module("ch05_lot", ROOT / "ch05_journey" / "scripts" / "lot.py")
_estimands = _load_module("ch04_estimands", ROOT / "ch04_market" / "scripts" / "estimands.py")
WASHOUT_DAYS = _lot.WASHOUT_DAYS
LAUNCH_PRODUCT = _estimands.LAUNCH_PRODUCT

SEED = 20260704

# Timeline. AS_OF_DATE reuses the omnichannel/NBA chapters' established
# "current" analysis date so the short observed history in this chapter
# lines up with the Roventra narrative the reader already knows. The full
# lifecycle extends years beyond AS_OF_DATE as the chapter's omniscient
# ground truth; only the slice up to AS_OF_DATE is "observed" for backtests.
LAUNCH_DATE = pd.Timestamp("2024-03-04")
AS_OF_DATE = pd.Timestamp("2025-02-28")
GENERIC_ENTRY_DATE = pd.Timestamp("2030-03-04")
SERIES_END_DATE = pd.Timestamp("2032-03-01")

# Access-quality weights, reused verbatim from the market-sizing chapter.
ACCESS_PROBABILITY = {
    "Covered": 0.90,
    "Covered with Step Edit": 0.75,
    "Covered with PA": 0.65,
    "Non-covered": 0.10,
}

# Formulary-access step: the payer mix improves on this date (a coverage win
# lands), raising the blended access probability used in the covariate.
ACCESS_IMPROVEMENT_DATE = pd.Timestamp("2024-09-16")
ACCESS_MIX_BEFORE = {
    "Covered": 0.30,
    "Covered with Step Edit": 0.20,
    "Covered with PA": 0.25,
    "Non-covered": 0.25,
}
ACCESS_MIX_AFTER = {
    "Covered": 0.55,
    "Covered with Step Edit": 0.20,
    "Covered with PA": 0.20,
    "Non-covered": 0.05,
}

# Promotional flight: a paid-media and field push over a defined window.
PROMO_FLIGHT_START = pd.Timestamp("2024-10-07")
PROMO_FLIGHT_END = pd.Timestamp("2024-12-30")
PROMO_LIFT = 0.18

# Bass diffusion ground truth for cumulative new-to-therapy starts from the
# depleting prevalent pool (see BACKGROUND_INCIDENCE_WEEKLY below for the
# separate, non-depleting incident-patient term that sustains the plateau).
# m is deliberately larger than the 2,798 line-1 entries already observed by
# the Chapter 3 mature snapshot, since that snapshot falls inside the launch
# ramp (adoption not yet complete). p and q are chosen so cumulative NBRx
# (Bass wave plus background) at AS_OF_DATE lands in the same order of
# magnitude as 2,798 (about 3,100 with the background term included), with
# time-to-peak around 16 months and the Bass wave essentially exhausted by
# month 60, followed by a plateau sustained by background incidence until
# generic entry.
BASS_P = 0.010
BASS_Q = 0.16
BASS_CEILING_M = 8_200

# Background incident-patient rate: newly diagnosed patients arrive every
# week independent of the one-time prevalent pool the Bass wave depletes.
# Scaled by the same adoption-awareness fraction F(t) so it ramps in with
# launch rather than starting at full rate on day one, then sustains a
# plateau once F(t) approaches 1, instead of the stock decaying back toward
# zero as the prevalent pool's Bass wave exhausts itself.
BACKGROUND_INCIDENCE_WEEKLY = 44.0

# Persistence: the fraction of starters still on therapy after t months,
# a Weibull survival curve fit directly to the patient-journey chapter's
# Kaplan-Meier line-1 persistence estimate (73.0% at day 60, 60.6% at day
# 90, 113-day median), by nonlinear least squares on those 3 points. The
# noisy day-180 point (only 50 of 3,415 patients still at risk in that
# chapter) is deliberately excluded from the fit. Unlike the earlier
# hand-set version, this shape has an increasing hazard rate (shape > 1):
# patients who make it past the first few months become more likely to
# drop off, not less, which is what the Kaplan-Meier data actually shows.
PERSISTENCE_SHAPE = 1.2583
PERSISTENCE_SCALE_MONTHS = 5.0286

# Refill rate: average fills per on-therapy patient-month, converting
# on-therapy stock into TRx.
REFILLS_PER_PATIENT_MONTH = 1.05

# Weekly seasonality: January insurance-reset spike and end-of-year holiday
# dip, applied as multipliers on top of the structural new-start rate.
JANUARY_RESET_LIFT = 0.22
HOLIDAY_DIP_WEEKS = {51, 52}
HOLIDAY_DIP_FACTOR = 0.55

NOISE_STD_FRACTION = 0.06

# Loss-of-exclusivity erosion: residual brand share retained in the long run
# after generic entry, and the half-life of the post-entry decline. Re-tuned
# alongside the persistence re-fit above: the shorter, increasing-hazard
# persistence curve clears patients out of the on-therapy stock much faster,
# so the same 12%/10-week setting no longer sustains a visible residual
# floor within the chapter's 2-year post-entry window. The half-life also
# has to be short enough that the true deterministic floor is fully reached
# by week 78, not just underway: at 30%/6 weeks, the 78-week fit recovers a
# residual fraction of about 10.3%, matching the true generated floor
# (about 10.2% of the pre-entry level) directly, rather than only detecting
# a nonzero residual that still understates it.
LOE_RESIDUAL_SHARE = 0.30
LOE_HALF_LIFE_WEEKS = 6.0

# Geographic hierarchy: national total decomposed into regions and, within
# each region, territories. Shares are fixed population weights.
REGION_SHARES = {
    "Northeast": 0.28,
    "Midwest": 0.24,
    "South": 0.31,
    "West": 0.17,
}
TERRITORIES_PER_REGION = 3

# Pre-launch patient-based funnel: the business-case assumptions available
# before any real Roventra uptake data exists. These are fresh, chapter-
# owned assumptions (not re-derived from the market-sizing funnel, which is
# NHANES-population-calibrated and answers a different question); they
# reuse the blended access probability computed from ACCESS_MIX_AFTER and
# ACCESS_PROBABILITY above. The resulting ceiling is deliberately allowed to
# land below the Bass-fitted ceiling from real early data (roughly 4,700 vs
# roughly 8,200): the gap is the chapter's teaching point, not an error to
# paper over.
TOTAL_ADDRESSABLE_PATIENTS = 20_000
PEAK_BRAND_SHARE_ASSUMPTION = 0.30

# Monte Carlo ranges for the pre-launch uncertainty section. Each range is
# (low, high) for a uniform draw; base_case is the single-point value used
# for the deterministic funnel and the tornado's center line.
FUNNEL_ASSUMPTION_RANGES = {
    "addressable_patients": {"low": 16_000, "high": 24_000, "base": TOTAL_ADDRESSABLE_PATIENTS},
    "brand_share": {"low": 0.20, "high": 0.40, "base": PEAK_BRAND_SHARE_ASSUMPTION},
    "access_adjustment": {"low": 0.65, "high": 0.85, "base": 0.78},
    "persistence_scale_months": {
        "low": 10.0,
        "high": 18.0,
        "base": PERSISTENCE_SCALE_MONTHS,
    },
}
MONTE_CARLO_DRAWS = 10_000
MONTE_CARLO_SEED = SEED

# Analog launches: fictional comparable historical launches with their own
# Bass shape, used for the analog-based forecast before any Roventra data
# exists. Selection later picks whichever analog's early normalized shape
# best matches the brand's own early normalized shape.
ANALOG_LAUNCHES = {
    "Comparable A (fast KOL-driven uptake)": {"p": 0.030, "q": 0.35},
    "Comparable B (slower primary-care-driven uptake)": {"p": 0.005, "q": 0.08},
}

# Demand-to-supply translation.
SERVICE_LEVEL_Z = 1.28  # approximately a 90% service level
SUPPLY_LEAD_TIME_WEEKS = 4.0

# Backtesting.
SEASONAL_PERIOD_WEEKS = 52
BACKTEST_HOLDOUT_WEEKS = 8
BACKTEST_FOLDS = 4
INTERVAL_LEVEL = 0.80

# Foundation models.
CHRONOS_MODEL_ID = "amazon/chronos-t5-small"
TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

# Loss-of-exclusivity analog erosion curves: fictional comparable molecules'
# post-generic-entry decline, used for an early erosion projection before
# enough of the brand's own post-entry tail exists to fit directly.
ANALOG_EROSIONS = {
    "Comparable erosion A (fast generic substitution)": {
        "residual_fraction": 0.08,
        "half_life_weeks": 6.0,
    },
    "Comparable erosion B (slower substitution, branded loyalty)": {
        "residual_fraction": 0.20,
        "half_life_weeks": 16.0,
    },
}

# TFT training across the territory panel.
TFT_INPUT_SIZE_WEEKS = 24
TFT_MAX_STEPS = 300
TFT_HIDDEN_SIZE = 32

__all__ = [
    "ROOT",
    "SEED",
    "LAUNCH_PRODUCT",
    "LAUNCH_DATE",
    "AS_OF_DATE",
    "GENERIC_ENTRY_DATE",
    "SERIES_END_DATE",
    "WASHOUT_DAYS",
    "ACCESS_PROBABILITY",
    "ACCESS_IMPROVEMENT_DATE",
    "ACCESS_MIX_BEFORE",
    "ACCESS_MIX_AFTER",
    "PROMO_FLIGHT_START",
    "PROMO_FLIGHT_END",
    "PROMO_LIFT",
    "BASS_P",
    "BASS_Q",
    "BASS_CEILING_M",
    "BACKGROUND_INCIDENCE_WEEKLY",
    "TOTAL_ADDRESSABLE_PATIENTS",
    "PEAK_BRAND_SHARE_ASSUMPTION",
    "FUNNEL_ASSUMPTION_RANGES",
    "MONTE_CARLO_DRAWS",
    "MONTE_CARLO_SEED",
    "ANALOG_LAUNCHES",
    "PERSISTENCE_SHAPE",
    "PERSISTENCE_SCALE_MONTHS",
    "REFILLS_PER_PATIENT_MONTH",
    "JANUARY_RESET_LIFT",
    "HOLIDAY_DIP_WEEKS",
    "HOLIDAY_DIP_FACTOR",
    "NOISE_STD_FRACTION",
    "LOE_RESIDUAL_SHARE",
    "LOE_HALF_LIFE_WEEKS",
    "REGION_SHARES",
    "TERRITORIES_PER_REGION",
    "SEASONAL_PERIOD_WEEKS",
    "BACKTEST_HOLDOUT_WEEKS",
    "BACKTEST_FOLDS",
    "INTERVAL_LEVEL",
    "CHRONOS_MODEL_ID",
    "TIMESFM_MODEL_ID",
    "TFT_INPUT_SIZE_WEEKS",
    "TFT_MAX_STEPS",
    "TFT_HIDDEN_SIZE",
    "ANALOG_EROSIONS",
    "SERVICE_LEVEL_Z",
    "SUPPLY_LEAD_TIME_WEEKS",
]
