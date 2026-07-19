"""Named constants for the resource-allocation and optimization chapter.

Every scenario assumption, cost figure, calibration target, and random seed
lives here as a documented module constant, per the book's convention of one
source of truth for constants shared by the generator, the response fit, the
allocation methods, the figures, and the tests.

The synthetic laboratory keeps hidden truth and analyst-visible planning
inputs physically apart. Latent response parameters are drawn once from a
truth seed and written only to an audit artifact. Everything an optimizer is
allowed to read comes from the observed field history and the response fit,
each built from its own seed so a reader can regenerate the planning inputs
from observed data alone.
"""

from __future__ import annotations

# ── Random seeds (one per generation stage) ────────────────────────────────
# Section 15.6 requires separate seeds so latent truth, observed outcomes,
# bootstrap sampling, frontier generation, and the learning simulation cannot
# accidentally share randomness. All five are recorded in the manifest.
SEED_LATENT_TRUTH = 20260718
SEED_OBSERVED = 515
SEED_BOOTSTRAP = 909
SEED_FRONTIER = 3131
SEED_LEARNING = 4242

# ── Territory structure ────────────────────────────────────────────────────
# 4 regions, 3 territories each, 25 accounts each: 12 territories, 300
# accounts. The "{region prefix}-T{n}" naming matches the forecasting
# chapter's territory hierarchy (MI-T1 .. WE-T3) without importing its
# generator; only the naming convention is shared.
REGIONS = ["Midwest", "Northeast", "South", "West"]
TERRITORIES_PER_REGION = 3
ACCOUNTS_PER_TERRITORY = 25

REPS_PER_TERRITORY_RANGE = (2, 5)          # low, high (exclusive) for integers
CALLS_PER_REP_PER_WEEK_RANGE = (6, 10)     # low, high (exclusive) for integers
WEEKS_PER_QUARTER = 13

# ── Observed field history ─────────────────────────────────────────────────
# Number of past quarters of account-period records the analyst can fit. Each
# account is observed at several call intensities so a shared response curve
# is identifiable from the panel without fitting one curve per account.
N_HISTORY_PERIODS = 6

# ── Access states ──────────────────────────────────────────────────────────
# Open / Restricted / Closed is the formulary-access vocabulary used for
# account rows in the AI-decision chapter. The multipliers are call-response
# multipliers: the share of an account's incremental call response that
# survives its formulary state. They are a distinct system from the
# market-sizing chapter's access-quality weights (0.90 / 0.75 / 0.65 / 0.10),
# which describe population reachability, not per-call response.
ACCESS_STATES = ["Open", "Restricted", "Closed"]
ACCESS_STATE_SHARE = {"Open": 0.55, "Restricted": 0.35, "Closed": 0.10}
RESPONSE_MULTIPLIER = {"Open": 1.0, "Restricted": 0.65, "Closed": 0.20}

# Share of accounts on a compliance or medical hold: protected rows that must
# receive zero promotional calls whatever the optimizer would otherwise do.
PROTECTED_SHARE = 0.04

# ── Response segments (latent truth) ───────────────────────────────────────
# Every account belongs to one of four response segments. A segment carries a
# shared latent response curve: an incremental scale (the share of account
# opportunity a fully saturated call plan can convert), a half-saturation
# call count (ec50), and a Hill shape. Accounts are assigned to segments by
# opportunity band so the roster mixes large low-response "anchor" accounts
# with smaller high-response "adopter" accounts. These true parameters are
# written only to the audit artifact.
SEGMENTS = {
    "Anchor":  {"scale": 0.16, "ec50": 10.0, "shape": 1.05, "baseline_share": (0.72, 0.90)},
    "Maintain": {"scale": 0.32, "ec50": 8.0, "shape": 1.15, "baseline_share": (0.45, 0.70)},
    "Growth":  {"scale": 0.48, "ec50": 6.0, "shape": 1.25, "baseline_share": (0.25, 0.45)},
    "Adopter": {"scale": 0.58, "ec50": 4.5, "shape": 1.40, "baseline_share": (0.10, 0.30)},
}
SEGMENT_ORDER = ["Anchor", "Maintain", "Growth", "Adopter"]

# Per-account latent heterogeneity: a lognormal multiplier on the segment
# curve's ceiling, hidden from the analyst and averaged out by the fit.
ACCOUNT_HETEROGENEITY_SD = 0.20

# Observation noise on the account-period incremental NRx record.
OBSERVATION_NOISE_SD = 0.25

MAX_CALLS_PER_ACCOUNT = 16
MIN_COVERAGE_CALLS = 1  # minimum quarterly calls for eligible Open/Restricted accounts

# ── Bootstrap response uncertainty ─────────────────────────────────────────
# Territory block bootstrap: resample territories with replacement, refit the
# segment response model inside every replicate, convert each fit into
# account-call step gains. N_BOOTSTRAP_DRAWS draws feed the sample-average
# (SAA), CVaR, regret, and frontier work.
N_BOOTSTRAP_DRAWS = 200

# Scenario count used inside the CVaR and epsilon-frontier solves. The full
# draw set scores every fixed plan; a stratified subset keeps the national
# scenario-slack MILP small enough for repeated frontier solves.
N_CVAR_SCENARIOS = 40
CVAR_ALPHA = 0.90  # tail confidence: CVaR averages the worst 10% of outcomes
CVAR_PROTECTION_FRACTION = 0.75
MAX_CVAR_SHORTFALL_SHARE = 0.10

# ── Business value assumptions ─────────────────────────────────────────────
# Stated planning assumptions, not measured constants, used to translate
# incremental NRx into dollars. NRX_VALUE_DOLLARS is net contribution per
# incremental quarterly NRx: price net of gross-to-net plus the refills a new
# prescription anchors over the months a patient persists, at the scale of
# this chapter's synthetic per-account NRx units. The sensitivity band is
# carried through the final package rather than hidden inside one factor.
NRX_VALUE_DOLLARS = 1_700.0
NRX_VALUE_DOLLARS_LOW = 1_300.0
NRX_VALUE_DOLLARS_HIGH = 2_100.0
LOADED_COST_PER_REP_QUARTER = 45_000.0

# ── Call-plan business rules ───────────────────────────────────────────────
CHURN_CAP_SHARE = 0.20              # max account-call changes, share of current national calls
CALL_MOVEMENT_CAP_STEP = 0.05      # tested relaxation increment for the trade-off table

# ── Epsilon-constraint frontier grid ───────────────────────────────────────
# Plan change is the number of account-level call additions plus removals,
# expressed as a share of current total calls. Every movement setting faces
# the same weak-quarter thresholds.
PLAN_CHANGE_CAP_GRID = [0.05, 0.10, 0.20, 0.35]
FRONTIER_CVAR_FLOOR_NRX = [1_000.0, 1_050.0]
NEAR_OPTIMAL_VALUE_LOSS_BUDGETS_DOLLARS = [10_000.0, 25_000.0, 75_000.0]
RELEASE_VALUE_LOSS_BUDGET_DOLLARS = 25_000.0
RELEASE_MIN_CVAR_NRX = 1_045.0

# ── Commit, reserve, learn ─────────────────────────────────────────────────
# One flexible reserve of field calls can be committed now or held for a
# measurement read on the highest-uncertainty segment. The study observes a
# noisy incremental-response signal and updates the planning weight on the
# bootstrap draws.
RESERVE_CALL_SHARE = 0.10          # share of total capacity held as a flexible reserve
STUDY_COST_DOLLARS = 1_500.0       # loaded cost of reading the segment engagement-outcome signal
STUDY_FOREGONE_NRX = 0.3           # incremental NRx given up by holding the reserve one quarter
STUDY_SIGNAL_NOISE_SD = 0.25       # measurement noise on the study read of segment response
N_LEARNING_TRIALS = 600            # Monte Carlo trials for value of sample information

# Planning dates make the release and refresh fields concrete without tying the
# synthetic case to a real brand calendar.
PLANNING_QUARTER = "2026 Q3"
MEASUREMENT_READ_DATE = "2026-09-15"
PLAN_REFRESH_DATE = "2026-10-01"

CHANNEL_LIST = ["field", "email", "digital", "paid_media"]
