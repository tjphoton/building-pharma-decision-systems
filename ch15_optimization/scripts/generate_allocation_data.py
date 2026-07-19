"""Synthetic laboratory for the resource-allocation and optimization chapter.

The design goal is strict truth isolation. Latent response parameters are
drawn once from a truth seed and written only to `account_response_truth.csv`,
an audit artifact. The analyst-visible world is an observed account-period
field history and a per-account planning table derived from it. No planning
table carries a column whose name begins with ``true_``.

Physical artifacts (Section 15.5):

- ``observed_field_history``  synthetic account-period records for fitting.
- ``account_planning_inputs`` account rules, opportunity, current calls,
  segment assignment; the only account table an optimizer reads.
- ``account_response_truth``  hidden curve parameters and latent response,
  read exclusively by audit functions after a plan is fixed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from allocation_config import (
    ACCESS_STATE_SHARE,
    ACCOUNT_HETEROGENEITY_SD,
    ACCOUNTS_PER_TERRITORY,
    CALLS_PER_REP_PER_WEEK_RANGE,
    MAX_CALLS_PER_ACCOUNT,
    MIN_COVERAGE_CALLS,
    N_HISTORY_PERIODS,
    OBSERVATION_NOISE_SD,
    PROTECTED_SHARE,
    REGIONS,
    REPS_PER_TERRITORY_RANGE,
    RESPONSE_MULTIPLIER,
    SEED_BOOTSTRAP,
    SEED_FRONTIER,
    SEED_LEARNING,
    SEED_LATENT_TRUTH,
    SEED_OBSERVED,
    SEGMENT_ORDER,
    SEGMENTS,
    TERRITORIES_PER_REGION,
    WEEKS_PER_QUARTER,
)

MEASUREMENT_SOURCES = ["specialty_pharmacy_status", "prescription_panel"]


# ── Shared response mechanics ──────────────────────────────────────────────

def centered_lognormal(rng: np.random.Generator, sigma: float, size) -> np.ndarray:
    """Multiplicative noise with an arithmetic mean of 1, so estimates stay unbiased.

    A plain ``rng.lognormal(mean=0, sigma=sigma)`` draw averages ``exp(sigma**2/2)``,
    above 1 for any positive sigma. Setting the log-space mean to ``-sigma**2/2``
    cancels that shift.
    """
    return rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=size)


def hill_fraction(calls: np.ndarray, ec50: float, shape: float) -> np.ndarray:
    """Saturating Hill fraction in [0, 1): calls**shape / (ec50**shape + calls**shape)."""
    calls = np.asarray(calls, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(calls <= 0, 0.0, calls**shape / (ec50**shape + calls**shape))
    return frac


def account_ceiling(opportunity: np.ndarray, access_multiplier: np.ndarray, scale: float, hetero: np.ndarray) -> np.ndarray:
    """Incremental NRx ceiling: the response as calls grow without bound."""
    return scale * opportunity * access_multiplier * hetero


# ── Stage 1: latent truth ──────────────────────────────────────────────────

def generate_latent_response_truth(seed: int = SEED_LATENT_TRUTH) -> dict:
    """Draw the hidden roster: opportunity, segment, access, and true curve parameters.

    Returns a dict with the territory table, the per-account truth table, and
    the segment-level true parameters. Only this stage touches the truth seed.
    """
    rng = np.random.default_rng(seed)

    territory_rows = []
    for region in REGIONS:
        prefix = region[:2].upper()
        for t_index in range(1, TERRITORIES_PER_REGION + 1):
            n_reps = int(rng.integers(*REPS_PER_TERRITORY_RANGE))
            calls_per_rep_per_week = int(rng.integers(*CALLS_PER_REP_PER_WEEK_RANGE))
            territory_rows.append({
                "territory_id": f"{prefix}-T{t_index}",
                "region": region,
                "n_reps": n_reps,
                "calls_per_rep_per_week": calls_per_rep_per_week,
                "quarterly_call_capacity": n_reps * calls_per_rep_per_week * WEEKS_PER_QUARTER,
            })
    territories = pd.DataFrame(territory_rows)
    # Hidden territory response heterogeneity: some territories convert calls
    # better than the segment average. The analyst fits pooled segment curves,
    # so this factor is absorbed into fit residual and surfaces as wider,
    # partly territory-specific bootstrap uncertainty when territories are
    # resampled. It is part of the truth and is never written to the
    # analyst-visible territory table.
    territory_effect = dict(zip(
        territories["territory_id"], centered_lognormal(rng, 0.22, len(territories))
    ))

    rows = []
    counter = 0
    for t_row in territories.itertuples():
        for _ in range(ACCOUNTS_PER_TERRITORY):
            counter += 1
            account_id = f"A{counter:04d}"
            opportunity = float(rng.gamma(2.2, 10.0)) + 4.0
            access_state = str(rng.choice(list(ACCESS_STATE_SHARE), p=list(ACCESS_STATE_SHARE.values())))
            protected = bool(rng.random() < PROTECTED_SHARE)
            rows.append({
                "account_id": account_id,
                "territory_id": t_row.territory_id,
                "region": t_row.region,
                "opportunity_nrx": opportunity,
                "access_state": access_state,
                "protected": protected,
            })
    accounts = pd.DataFrame(rows)

    # Assign segments by opportunity band, with noise, so large accounts skew
    # toward slow-responding "anchor" segments and small accounts toward fast
    # "adopter" segments, but the mapping is not a clean cut.
    rank = accounts["opportunity_nrx"].rank(pct=True).to_numpy()
    jitter = rng.normal(0, 0.12, len(accounts))
    score = np.clip(rank + jitter, 0, 1)
    # High opportunity -> Anchor, low opportunity -> Adopter.
    bins = np.array([0.0, 0.30, 0.55, 0.80, 1.01])
    seg_index = np.digitize(1.0 - score, bins) - 1
    seg_index = np.clip(seg_index, 0, len(SEGMENT_ORDER) - 1)
    accounts["segment"] = [SEGMENT_ORDER[i] for i in seg_index]

    accounts["access_multiplier"] = accounts["access_state"].map(RESPONSE_MULTIPLIER)
    hetero = centered_lognormal(rng, ACCOUNT_HETEROGENEITY_SD, len(accounts))
    accounts["true_hetero"] = hetero

    baseline_share = np.empty(len(accounts))
    true_scale = np.empty(len(accounts))
    true_ec50 = np.empty(len(accounts))
    true_shape = np.empty(len(accounts))
    for seg, params in SEGMENTS.items():
        mask = (accounts["segment"] == seg).to_numpy()
        lo, hi = params["baseline_share"]
        baseline_share[mask] = rng.uniform(lo, hi, mask.sum())
        true_scale[mask] = params["scale"]
        true_ec50[mask] = params["ec50"]
        true_shape[mask] = params["shape"]

    accounts["baseline_nrx"] = (accounts["opportunity_nrx"].to_numpy() * baseline_share).round(3)
    accounts["true_scale"] = true_scale
    accounts["true_ec50"] = true_ec50
    accounts["true_shape"] = true_shape
    t_effect = accounts["territory_id"].map(territory_effect).to_numpy()
    accounts["true_ceiling"] = account_ceiling(
        accounts["opportunity_nrx"].to_numpy(),
        accounts["access_multiplier"].to_numpy(),
        1.0,  # scale applied per-row below
        hetero,
    ) * true_scale * t_effect
    accounts["true_ceiling"] = accounts["true_ceiling"].round(4)

    segment_truth = {
        seg: {"scale": p["scale"], "ec50": p["ec50"], "shape": p["shape"]}
        for seg, p in SEGMENTS.items()
    }
    return {"territories": territories, "accounts": accounts, "segment_truth": segment_truth}


def true_incremental_nrx(truth: pd.DataFrame, calls: np.ndarray) -> np.ndarray:
    """Ground-truth incremental NRx for a call vector. Audit and test use only."""
    calls = np.asarray(calls, dtype=float)
    frac = hill_fraction(calls, truth["true_ec50"].to_numpy(), truth["true_shape"].to_numpy())
    return truth["true_ceiling"].to_numpy() * frac


# ── Stage 2: observed field history ────────────────────────────────────────

def _opportunity_weighted_incumbent(truth: dict, blocked: np.ndarray, fill_share: float = 0.85) -> np.ndarray:
    """Incumbent plan: fill ~85% of each territory's capacity, weighted by opportunity."""
    accounts = truth["accounts"]
    capacity = truth["territories"].set_index("territory_id")["quarterly_call_capacity"]
    opportunity = accounts["opportunity_nrx"].to_numpy()
    calls = np.zeros(len(accounts), dtype=int)
    for t_id, group in accounts.groupby("territory_id"):
        pos = group.index.to_numpy()
        elig = ~blocked[pos]
        if not elig.any():
            continue
        target = fill_share * float(capacity[t_id])
        weight = opportunity[pos] * elig
        share = weight / weight.sum()
        raw = np.clip(np.round(share * target), 0, MAX_CALLS_PER_ACCOUNT)
        raw[~elig] = 0
        # Every eligible account keeps at least the minimum coverage call.
        raw[elig] = np.clip(raw[elig], MIN_COVERAGE_CALLS, MAX_CALLS_PER_ACCOUNT)
        calls[pos] = raw.astype(int)
    return calls


def generate_observed_field_history(truth: dict, seed: int = SEED_OBSERVED) -> pd.DataFrame:
    """Build the account-period panel an analyst can fit, drawn from a fresh seed.

    Each account is observed at several call intensities across
    ``N_HISTORY_PERIODS`` quarters. The observed incremental NRx is the latent
    response at that period's calls, blurred by multiplicative observation
    noise; baseline and opportunity are recorded directly.
    """
    rng = np.random.default_rng(seed)
    accounts = truth["accounts"]
    n = len(accounts)

    account_mean_calls = rng.uniform(2.0, 12.0, n)
    windows = [f"2024Q{q}" if q <= 4 else f"2025Q{q - 4}" for q in range(1, N_HISTORY_PERIODS + 1)]

    ceiling = truth["accounts"]["true_ceiling"].to_numpy()
    ec50 = truth["accounts"]["true_ec50"].to_numpy()
    shape = truth["accounts"]["true_shape"].to_numpy()
    baseline = accounts["baseline_nrx"].to_numpy()
    protected = accounts["protected"].to_numpy()
    closed = (accounts["access_state"] == "Closed").to_numpy()
    blocked = protected | closed

    # The current incumbent plan (the latest observed quarter) fills most of
    # each territory's capacity but is allocated by account opportunity, not by
    # incremental response: reps over-serve large, mostly-loyal accounts. This
    # is the misallocation the chapter's reallocation optimizer improves on.
    incumbent = _opportunity_weighted_incumbent(truth, blocked)

    frames = []
    for period_id, window in enumerate(windows, start=1):
        if period_id == N_HISTORY_PERIODS:
            calls = incumbent.copy()
        else:
            raw = account_mean_calls + rng.normal(0, 3.0, n)
            calls = np.clip(np.round(raw), 0, MAX_CALLS_PER_ACCOUNT).astype(int)
            calls[blocked] = np.clip(calls[blocked] - 3, 0, 2)

        frac = hill_fraction(calls, ec50, shape)
        latent_incremental = ceiling * frac
        noise = centered_lognormal(rng, OBSERVATION_NOISE_SD, n)
        observed_incremental = latent_incremental * noise + rng.normal(0, 0.4, n)
        observed_nrx = np.clip(baseline + observed_incremental, 0.0, None)

        frames.append(pd.DataFrame({
            "account_id": accounts["account_id"].to_numpy(),
            "territory_id": accounts["territory_id"].to_numpy(),
            "period_id": period_id,
            "segment": accounts["segment"].to_numpy(),
            "access_state": accounts["access_state"].to_numpy(),
            "eligible_flag": (~blocked),
            "protected_flag": protected,
            "baseline_nrx": baseline.round(3),
            "opportunity_nrx": accounts["opportunity_nrx"].to_numpy().round(3),
            "calls": calls,
            "channel_exposure": rng.poisson(6.0, n),
            "observed_nrx": observed_nrx.round(3),
            "measurement_source": rng.choice(MEASUREMENT_SOURCES, n),
            "measurement_window": window,
        }))
    history = pd.concat(frames, ignore_index=True)
    return history


# ── Stage 3: planning inputs (analyst-visible only) ────────────────────────

def build_account_planning_inputs(
    history: pd.DataFrame, territories: pd.DataFrame
) -> pd.DataFrame:
    """Per-account planning table derived from observed history. No ``true_`` columns.

    ``current_calls`` is the account's most recent observed quarter. Access,
    opportunity, baseline, and segment are read straight from the panel. Call
    bounds encode the access and compliance rules every optimizer must honour.
    """
    latest = history["period_id"].max()
    visible = history.loc[history["period_id"] == latest, [
        "account_id", "territory_id", "segment", "access_state",
        "protected_flag", "opportunity_nrx", "baseline_nrx", "calls",
    ]].copy()
    planning = visible.rename(columns={
        "protected_flag": "protected",
        "calls": "current_calls",
    }).merge(territories[["territory_id", "region"]], on="territory_id", how="left")

    blocked = planning["protected"] | (planning["access_state"] == "Closed")
    planning["eligible_flag"] = ~blocked
    planning["min_calls"] = np.where(blocked, 0, MIN_COVERAGE_CALLS)
    planning["max_calls"] = np.where(blocked, 0, MAX_CALLS_PER_ACCOUNT)
    # A blocked account carried no promotional plan going forward.
    planning.loc[blocked, "current_calls"] = 0
    planning["current_calls"] = planning["current_calls"].fillna(0).astype(int)
    planning["opportunity_nrx"] = planning["opportunity_nrx"].round(3)
    planning["response_multiplier"] = planning["access_state"].map(RESPONSE_MULTIPLIER)

    assert not any(c.startswith("true_") for c in planning.columns), "planning inputs leaked hidden truth"
    return planning[[
        "account_id", "territory_id", "region", "segment", "access_state",
        "response_multiplier", "eligible_flag", "protected", "opportunity_nrx",
        "baseline_nrx", "current_calls", "min_calls", "max_calls",
    ]]


def account_response_truth_table(truth: dict) -> pd.DataFrame:
    """Hidden curve parameters and latent ceiling, for the audit phase only."""
    accounts = truth["accounts"]
    return accounts[[
        "account_id", "territory_id", "segment", "access_multiplier",
        "opportunity_nrx", "true_scale", "true_ec50", "true_shape",
        "true_hetero", "true_ceiling",
    ]].copy()


# ── Manifest ───────────────────────────────────────────────────────────────

def build_generation_manifest(
    history: pd.DataFrame,
    planning: pd.DataFrame,
    seed_truth: int = SEED_LATENT_TRUTH,
    seed_observed: int = SEED_OBSERVED,
) -> dict:
    payload = pd.util.hash_pandas_object(planning, index=False).values.tobytes()
    return {
        "seed_latent_truth": seed_truth,
        "seed_observed": seed_observed,
        "seed_bootstrap": SEED_BOOTSTRAP,
        "seed_frontier": SEED_FRONTIER,
        "seed_learning": SEED_LEARNING,
        "n_territories": int(planning["territory_id"].nunique()),
        "n_accounts": len(planning),
        "n_history_rows": len(history),
        "n_history_periods": int(history["period_id"].nunique()),
        "n_protected": int(planning["protected"].sum()),
        "n_eligible": int(planning["eligible_flag"].sum()),
        "planning_table_hash": hashlib.sha256(payload).hexdigest()[:16],
    }


def run_generation(seed_truth: int = SEED_LATENT_TRUTH, seed_observed: int = SEED_OBSERVED) -> dict:
    """Full generation: truth, observed history, planning inputs, and manifest."""
    truth = generate_latent_response_truth(seed_truth)
    history = generate_observed_field_history(truth, seed_observed)
    planning = build_account_planning_inputs(history, truth["territories"])
    audit_truth = account_response_truth_table(truth)
    return {
        "territories": truth["territories"],
        "planning": planning,
        "observed_history": history,
        "truth": audit_truth,
        "segment_truth": truth["segment_truth"],
        "manifest": build_generation_manifest(history, planning, seed_truth, seed_observed),
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "assets" / "generated_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_generation()
    result["territories"].to_csv(out_dir / "territories.csv", index=False)
    result["observed_history"].to_csv(out_dir / "observed_field_history.csv", index=False)
    result["planning"].to_csv(out_dir / "account_planning_inputs.csv", index=False)
    result["truth"].to_csv(out_dir / "account_response_truth.csv", index=False)
    print(pd.Series(result["manifest"]).to_string())
    print(f"Wrote generation artifacts to {out_dir}")
