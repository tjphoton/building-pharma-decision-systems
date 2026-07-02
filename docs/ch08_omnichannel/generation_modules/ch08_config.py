"""Scenario constants for the Chapter 8 omnichannel case."""

from __future__ import annotations

import pandas as pd


SEED = 20260622
GENERATOR_VERSION = "3.0.0"
ANALYSIS_DATE = pd.Timestamp("2025-02-28")
DATA_START = pd.Timestamp("2024-01-01")
DATA_END = pd.Timestamp("2025-03-31")
CYCLE_START = pd.Timestamp("2025-03-03")
CYCLE_END = pd.Timestamp("2025-03-30")
REFRESH_DATE = pd.Timestamp("2025-03-31")

LOOKBACK_DAYS = 90
OUTCOME_DAYS = 28
PRESSURE_LOW_MAX = 2
PRESSURE_MODERATE_MAX = 4
HIGH_PRESSURE_MIN = PRESSURE_MODERATE_MAX + 1
SNAPSHOT_START = pd.Timestamp("2024-06-30")
SNAPSHOT_END = pd.Timestamp("2025-02-28")
SNAPSHOT_FREQUENCY = "ME"
RECENCY_DECAY_DAYS = 30
FIELD_CAPACITY_PER_TERRITORY = 2
PREDICTED_RESPONSE_FLOOR = 0.24
TRACE_OVERRIDE_RESPONSE_PROBABILITY = 0.5
MODEL_VERSION = "ch08-logit-v1.0"
POLICY_VERSION = "ch08-channel-policy-v1.1"
OPT_OUT_START = pd.Timestamp("2024-11-15")
OPT_OUT_END = pd.Timestamp("2025-01-15")
MONTHLY_LAUNCH_RAMP = {
    "2024-01": -0.05,
    "2024-02": -0.03,
    "2024-03": -0.02,
    "2024-04": 0.00,
    "2024-05": 0.02,
    "2024-06": 0.04,
    "2024-07": 0.06,
    "2024-08": 0.08,
    "2024-09": 0.10,
    "2024-10": 0.12,
    "2024-11": 0.14,
    "2024-12": 0.16,
    "2025-01": 0.24,
    "2025-02": 0.26,
    "2025-03": 0.28,
}

# --- Per-event response model (generator) ------------------------------------
# The probability that a delivered event earns a meaningful response is a logistic
# function of stable HCP traits, topic match, same-channel response memory, and
# recent contact fatigue. These weights are named here so the chapter prose can
# quote them and tests can assert recovery. Raising the trait weights relative to
# the noise floor is what lets the temporal response model rank later snapshots.
RESPONSE_BASE_LOGIT = -3.25
RESPONSE_AFFINITY_WEIGHT = 3.65
RESPONSE_EVIDENCE_WEIGHT = 1.75
RESPONSE_ACCESS_WEIGHT = 1.10
RESPONSE_TOPIC_WEIGHT = 0.85
RESPONSE_MEMORY_WEIGHT = 0.80
RESPONSE_FATIGUE_WEIGHT = 0.42

# --- Planted live-program treatment effect (answer key) -----------------------
# A live-program meaningful response (peer, speaker, or conference) in the prior
# LIVE_PROGRAM_EFFECT_WINDOW days raises the log-odds of a later meaningful
# response by a uniform amount. On the probability scale this uniform log-odds
# lift produces a heterogeneous effect: relationships already near the top of the
# response curve barely move, while mid-range "persuadable" relationships move the
# most. That is the effect the uplift section recovers, and it is why ranking by
# uplift differs from ranking by predicted response. A small evidence-need gain
# keeps the documented moderator without overturning the persuadable pattern.
LIVE_PROGRAM_EFFECT_WINDOW = 180
LIVE_PROGRAM_UPLIFT_BASE_LOGIT = 0.85
LIVE_PROGRAM_UPLIFT_EVIDENCE_GAIN = 0.0

# --- Planted field-then-digital sequence effect (answer key) ------------------
# A field meaningful response in the prior FIELD_THEN_DIGITAL_WINDOW days raises
# the response log-odds of a later Email or Web touch, scaled by evidence need.
# This is the recoverable order effect the sequence section reports.
FIELD_THEN_DIGITAL_WINDOW = 45
FIELD_THEN_DIGITAL_BASE_LOGIT = 0.30
FIELD_THEN_DIGITAL_EVIDENCE_GAIN = 1.25

CHANNEL_METADATA = {
    "Field": {
        "prefix": "field",
        "source_system": "CRM",
        "event_type": "Field interaction",
    },
    "Email": {
        "prefix": "email",
        "source_system": "Email platform",
        "event_type": "Approved email",
    },
    "Web": {
        "prefix": "web",
        "source_system": "Web analytics",
        "event_type": "Authenticated content visit",
    },
    "Peer program": {
        "prefix": "peer",
        "source_system": "Peer program platform",
        "event_type": "Peer program invitation",
    },
    "Speaker program": {
        "prefix": "speaker",
        "source_system": "Speaker program platform",
        "event_type": "Speaker program",
    },
    "Paid media": {
        "prefix": "paid",
        "source_system": "Media platform",
        "event_type": "Paid media exposure",
    },
    "Conference": {
        "prefix": "conference",
        "source_system": "Event platform",
        "event_type": "Conference engagement",
    },
    "Direct mail": {
        "prefix": "mail",
        "source_system": "Fulfillment platform",
        "event_type": "Direct mail",
    },
    "Phone": {
        "prefix": "phone",
        "source_system": "Call center",
        "event_type": "Phone outreach",
    },
    "Account support": {
        "prefix": "account",
        "source_system": "Access operations",
        "event_type": "Account support",
    },
}

# --- Illustrative per-touch channel cost (USD) -------------------------------
# Deterministic planning constants, not a planted random effect. They encode the
# real ordering an omnichannel team works with: a field call or a live program
# costs two to three orders of magnitude more than a delivered email or web
# visit. The economics section divides channel cost by modeled incremental
# response so a high-credit, high-cost channel can still be a poor place to add
# budget. Values are illustrative and only the ordering carries the lesson.
CHANNEL_UNIT_COST = {
    "Field": 225.0,
    "Phone": 28.0,
    "Email": 0.25,
    "Web": 0.12,
    "Peer program": 340.0,
    "Speaker program": 1150.0,
    "Paid media": 1.40,
    "Conference": 760.0,
    "Direct mail": 2.60,
    "Account support": 130.0,
}

# --- Scenario value of one incremental meaningful response (USD) --------------
# The same planning constant the next-best-action engine uses: one incremental
# meaningful prescriber response carries an estimated net TRx value of $4,000.
# This is a documented scenario assumption, not a measured quantity. The value
# bridge multiplies modeled incremental response per touch by this constant so
# channel cost and channel return land in the same dollar unit.
RESPONSE_VALUE = 4_000.0

CHANNELS = tuple(CHANNEL_METADATA)
TOPICS = (
    "Clinical evidence",
    "Patient identification",
    "Coverage support",
    "Dosing and administration",
)

CHANNEL_BASE_WEIGHTS = {
    "Field": 0.24,
    "Email": 0.26,
    "Web": 0.12,
    "Peer program": 0.08,
    "Speaker program": 0.06,
    "Paid media": 0.07,
    "Conference": 0.04,
    "Direct mail": 0.04,
    "Phone": 0.04,
    "Account support": 0.05,
}

TRACE_OVERRIDES = {
    "9000000280": [
        ("2025-01-15", "Field", "Clinical evidence", "Positive"),
        ("2025-01-24", "Email", "Clinical evidence", "Open only"),
        ("2025-02-10", "Web", "Patient identification", "Qualified action"),
    ],
    "9000000389": [
        ("2025-01-08", "Email", "Clinical evidence", "Clicked"),
        ("2025-01-23", "Web", "Clinical evidence", "Qualified action"),
        ("2025-02-12", "Field", "Clinical evidence", "Follow-up"),
    ],
    "9000000430": [
        ("2025-01-06", "Field", "Coverage support", "Follow-up"),
        ("2025-01-20", "Email", "Coverage support", "Open only"),
        ("2025-02-03", "Field", "Coverage support", "Neutral"),
        ("2025-02-14", "Email", "Coverage support", "No response"),
        ("2025-02-24", "Field", "Coverage support", "No reach"),
    ],
    "9000000469": [
        ("2024-11-18", "Email", "Dosing and administration", "Clicked"),
        ("2024-12-05", "Web", "Dosing and administration", "Qualified action"),
    ],
}
