"""Budget optimisation for the marketing-mix-modeling chapter.

Solves for the spend allocation that maximises expected NRx under a total budget
constraint using scipy SLSQP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import spend_to_exposure
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


# ── Channel-decision gating ────────────────────────────────────────────────────

DIRECTIONAL_BAND = 0.20  # causal-anchored channel whose own MMM fit is still directional
WEAK_DIRECTIONAL_BAND = 0.10  # mmm-only directional: neither a causal anchor nor a clean MMM fit
DECISION_READY_INCREASE_CAP = 0.30  # mmm-only decision-ready: clean MMM fit, no causal anchor yet


def _guardrail_lookup(
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    """Normalize a guardrail table into a channel-keyed lookup."""
    if guardrails is None:
        return {}
    if isinstance(guardrails, pd.DataFrame):
        return guardrails.set_index("channel").to_dict(orient="index")
    return guardrails


def channel_bounds(
    current_spends: np.ndarray,
    channels: list[str],
    budget: float,
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> list[tuple[float, float]]:
    """Per-channel optimizer bounds, gated by `measurement_decision_record.csv`.

    A "decision-ready" channel gets the full (0, budget) range. A "directional"
    or "not usable" channel is not trusted for an unconstrained reallocation:
    directional channels may move by at most `DIRECTIONAL_BAND` off their
    current spend, and not-usable channels are frozen at current spend. If no
    decision_status is supplied, every channel gets the unconstrained range
    (the pre-gate behaviour).
    """
    guardrail_by_channel = _guardrail_lookup(guardrails)
    bounds = []
    for i, ch in enumerate(channels):
        current = float(current_spends[i])
        if ch in guardrail_by_channel:
            rule = guardrail_by_channel[ch]
            permission = str(rule.get("move_permission", "full-range"))
            max_move_pct = float(rule.get("max_move_pct", DIRECTIONAL_BAND) or 0.0)
            if permission == "full-range":
                bounds.append((0.0, budget))
            elif permission == "increase-capped":
                bounds.append((0.0, current * (1 + max_move_pct)))
            elif permission == "bounded":
                bounds.append((max(current * (1 - max_move_pct), 0.0), current * (1 + max_move_pct)))
            else:
                bounds.append((current, current))
            continue

        status = "decision-ready" if decision_status is None else decision_status.get(ch, "decision-ready")
        if status == "decision-ready":
            bounds.append((0.0, budget))
        elif status == "directional":
            bounds.append((max(current * (1 - DIRECTIONAL_BAND), 0.0), current * (1 + DIRECTIONAL_BAND)))
        else:
            bounds.append((current, current))
    return bounds


def build_constrained_channel_note(
    decision_status: dict[str, str],
    channels: list[str],
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """One row per channel explaining what the optimizer was allowed to do with it."""
    guardrail_by_channel = _guardrail_lookup(guardrails)
    rows = []
    for ch in channels:
        status = decision_status.get(ch, "decision-ready")
        if ch in guardrail_by_channel:
            rule = guardrail_by_channel[ch]
            permission = str(rule.get("move_permission", "full-range"))
            max_move_pct = float(rule.get("max_move_pct", DIRECTIONAL_BAND) or 0.0)
            if permission == "full-range":
                constraint = "none: full (0, budget) range"
            elif permission == "increase-capped":
                constraint = f"increase capped at +{max_move_pct:.0%}; optimizer may still cut to zero"
            elif permission == "bounded":
                constraint = f"bounded to +/-{max_move_pct:.0%} of current spend"
            else:
                constraint = "frozen at current spend"
        elif status == "decision-ready":
            constraint = "none: full (0, budget) range"
        elif status == "directional":
            constraint = f"bounded to +/-{DIRECTIONAL_BAND:.0%} of current spend"
        else:
            constraint = "frozen at current spend"
        rows.append({
            "channel": ch,
            "decision_status": status,
            "optimizer_constraint": constraint,
        })
    return pd.DataFrame(rows)


# ── Optimisation functions ────────────────────────────────────────────────────

def _nrx_for_draw(
    weekly_spends: np.ndarray,
    row: pd.Series,
    channels: list[str],
) -> float:
    """Expected steady-state weekly NRx for one posterior draw and one spend allocation."""
    nrx = float(row["baseline0"])
    for i, ch in enumerate(channels):
        exposure_val = spend_to_exposure(ch, float(weekly_spends[i]))
        spend_arr = np.array([exposure_val] * 20)  # 20-week steady state
        decay = float(row[f"{ch}_decay"])
        ec50 = float(row[f"{ch}_ec50"])
        slope = float(row[f"{ch}_slope"])
        coef = float(row[f"{ch}_coef"])
        ads = _adstock(spend_arr, decay)
        nrx += coef * _hill(ads, ec50, slope).mean()
    return nrx


def _expected_nrx(
    weekly_spends: np.ndarray,
    draws: pd.DataFrame,
    channels: list[str],
    n_draws: int = 200,
) -> float:
    """Average expected weekly NRx across posterior draws for given spend allocation."""
    total = 0.0
    for j in range(n_draws):
        row = draws.iloc[j]
        total += _nrx_for_draw(weekly_spends, row, channels)
    return total / n_draws


def optimal_allocation_at_budget(
    current_spends: np.ndarray,
    budget: float,
    draws: pd.DataFrame,
    channels: list[str],
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> np.ndarray:
    """SLSQP allocation that maximises posterior-mean expected NRx at one fixed budget.

    `decision_status`, when given, gates each channel's bounds through
    `channel_bounds()` so a directional or not-usable channel cannot be moved
    into on the strength of an estimate that hasn't cleared the health gate.
    """
    current_budget = float(current_spends.sum())

    def neg_nrx(x: np.ndarray) -> float:
        if any(xi < 0 for xi in x):
            return 1e9
        return -_expected_nrx(x, draws, channels, n_draws=100)

    bounds = channel_bounds(current_spends, channels, budget, decision_status, guardrails)
    constraints = {"type": "eq", "fun": lambda x: x.sum() - budget}
    result = minimize(
        neg_nrx,
        current_spends * budget / current_budget,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-4},
    )
    return result.x


def _optimal_allocation_for_draw(
    current_spends: np.ndarray,
    budget: float,
    row: pd.Series,
    channels: list[str],
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> np.ndarray:
    """SLSQP optimum for one posterior draw."""
    current_budget = float(current_spends.sum())

    def neg_nrx(x: np.ndarray) -> float:
        if any(xi < 0 for xi in x):
            return 1e9
        return -_nrx_for_draw(x, row, channels)

    bounds = channel_bounds(current_spends, channels, budget, decision_status, guardrails)
    constraints = {"type": "eq", "fun": lambda x: x.sum() - budget}
    result = minimize(
        neg_nrx,
        current_spends * budget / current_budget,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 150, "ftol": 1e-4},
    )
    return result.x


def optimal_allocation_by_draw(
    current_spends: np.ndarray,
    budget: float,
    draws: pd.DataFrame,
    channels: list[str],
    max_draws: int = 80,
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize the allocation on a thinned posterior sample and summarize it."""
    sample = draws.iloc[: min(max_draws, len(draws))]
    rows = []
    for draw_number, (_, row) in enumerate(sample.iterrows()):
        opt = _optimal_allocation_for_draw(current_spends, budget, row, channels, decision_status, guardrails)
        for i, ch in enumerate(channels):
            rows.append({
                "draw_number": draw_number,
                "channel": ch,
                "optimized_weekly_spend": round(float(opt[i]), 2),
            })
    per_draw = pd.DataFrame(rows)

    summary_rows = []
    for ch in channels:
        vals = per_draw.loc[per_draw["channel"] == ch, "optimized_weekly_spend"].to_numpy(dtype=float)
        summary_rows.append({
            "channel": ch,
            "optimized_weekly_spend_mean": round(float(vals.mean()), 1),
            "optimized_weekly_spend_median": round(float(np.median(vals)), 1),
            "optimized_weekly_spend_p10": round(float(np.percentile(vals, 10)), 1),
            "optimized_weekly_spend_p90": round(float(np.percentile(vals, 90)), 1),
        })
    return pd.DataFrame(summary_rows), per_draw


def budget_optimisation(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    budgets: list[float] | None = None,
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Optimise channel spend to maximise expected NRx under total budget constraints.

    Solves for each total weekly budget level and compares to the current allocation.
    """
    channels = CHANNELS
    current_spends = np.array([df[f"spend_{ch}"].mean() for ch in channels])
    current_budget = float(current_spends.sum())

    if budgets is None:
        budgets = [current_budget * f for f in [0.75, 0.90, 1.0, 1.10, 1.25]]

    rows = []
    # Current allocation baseline
    current_nrx = _expected_nrx(current_spends, draws, channels)
    rows.append({
        "scenario": "Current allocation",
        "total_budget": round(current_budget, 0),
        **{f"spend_{ch}": round(float(s), 0) for ch, s in zip(channels, current_spends)},
        "expected_weekly_nrx": round(current_nrx, 1),
        "nrx_vs_current": 0.0,
    })

    for budget in budgets:
        if abs(budget - current_budget) < 1e-6:
            continue
        opt_spends = optimal_allocation_at_budget(current_spends, budget, draws, channels, decision_status, guardrails)
        opt_nrx = _expected_nrx(opt_spends, draws, channels, n_draws=200)
        rows.append({
            "scenario": f"Optimised (budget={budget:,.0f})",
            "total_budget": round(budget, 0),
            **{f"spend_{ch}": round(float(s), 0) for ch, s in zip(channels, opt_spends)},
            "expected_weekly_nrx": round(opt_nrx, 1),
            "nrx_vs_current": round(opt_nrx - current_nrx, 1),
        })

    return pd.DataFrame(rows)


# ── Decision-probability evaluation ────────────────────────────────────────────

def evaluate_reallocation(
    current_spends: np.ndarray,
    candidate_spends: np.ndarray,
    draws: pd.DataFrame,
    channels: list[str],
) -> dict:
    """Evaluate one fixed candidate allocation against the current allocation.

    Pairs draw-by-draw across the full posterior (not averaged first): for every
    posterior draw, computes expected weekly NRx under `current_spends` and under
    `candidate_spends` using that draw's own parameters, then differences within
    the draw before summarising across draws.
    """
    n = len(draws)
    current_nrx = np.empty(n)
    candidate_nrx = np.empty(n)
    for j in range(n):
        row = draws.iloc[j]
        current_nrx[j] = _nrx_for_draw(current_spends, row, channels)
        candidate_nrx[j] = _nrx_for_draw(candidate_spends, row, channels)
    gains = candidate_nrx - current_nrx

    return {
        "mean_nrx_current": round(float(current_nrx.mean()), 2),
        "mean_nrx_candidate": round(float(candidate_nrx.mean()), 2),
        "mean_nrx_gain": round(float(gains.mean()), 2),
        "p10_nrx_gain": round(float(np.percentile(gains, 10)), 2),
        "p90_nrx_gain": round(float(np.percentile(gains, 90)), 2),
        "win_rate": round(float(np.mean(gains > 0)), 4),
    }


# ── Flat-maximum sweep ─────────────────────────────────────────────────────────

def _rescale_within_bounds(
    candidate: np.ndarray,
    bounds: list[tuple[float, float]],
    budget: float,
    max_iter: int = 6,
) -> np.ndarray:
    """Clip `candidate` to per-channel `bounds`, then redistribute the resulting
    budget shortfall or surplus only across channels that still have slack,
    iterating a few times since redistribution can push another channel onto
    its own bound. Keeps every point on the sweep inside the guardrails
    instead of just proportionally rescaling, which can push a channel back
    outside a bound it was just clipped to.
    """
    candidate = candidate.astype(float).copy()
    for _ in range(max_iter):
        candidate = np.array([np.clip(c, lo, hi) for c, (lo, hi) in zip(candidate, bounds, strict=True)])
        deficit = budget - candidate.sum()
        if abs(deficit) < 1e-6:
            break
        if deficit > 0:
            room = np.array([hi - c for c, (lo, hi) in zip(candidate, bounds, strict=True)])
        else:
            room = np.array([c - lo for c, (lo, hi) in zip(candidate, bounds, strict=True)])
        room = np.clip(room, 0, None)
        if room.sum() <= 1e-9:
            break
        candidate = candidate + deficit * (room / room.sum())
    return candidate


def flat_maximum_sweep(
    opt_spends: np.ndarray,
    draws: pd.DataFrame,
    channels: list[str],
    budget: float,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
    n_points: int = 9,
) -> pd.DataFrame:
    """Sweep allocations around the guardrail-constrained optimum to show how
    flat the NRx surface is within the range the current evidence permits.

    Blends `opt_spends` toward (and past) a uniform equal-weight allocation
    using blend weights in [-0.3, 0.3], clips each channel to its own
    guardrail bound at every step (the same bounds the optimizer itself
    obeys), and reports posterior-mean expected NRx at each blend weight. A
    near-flat `pct_of_max` across the sweep means the exact allocation
    inside the evidence-permitted range matters less than avoiding a badly
    saturated channel. `signed_reallocated_pct_of_budget` restates each
    point's distance from the recommendation in dollar terms (percent of
    total budget shifted between channels), positive toward the equal
    split and negative toward a more field-concentrated mix, which reads
    more directly than the raw blend weight.
    """
    uniform_alloc = np.full(len(channels), budget / len(channels))
    blend_weights = np.linspace(-0.3, 0.3, n_points)
    # Read each channel's already-computed floor/ceiling straight from the
    # guardrail record (anchored to the true current spend) rather than
    # re-deriving a bound from opt_spends, which would center the band on
    # the optimum instead of on current spend.
    guardrail_by_channel = _guardrail_lookup(guardrails)
    bounds = [
        (
            (float(guardrail_by_channel[ch]["spend_floor"]), float(guardrail_by_channel[ch]["spend_ceiling"]))
            if ch in guardrail_by_channel
            else (0.0, budget)
        )
        for ch in channels
    ]

    recommended = _rescale_within_bounds(np.clip(opt_spends, 0, None), bounds, budget)

    rows = []
    for w in blend_weights:
        candidate = opt_spends + w * (uniform_alloc - opt_spends)
        candidate = np.clip(candidate, 0, None)
        candidate = _rescale_within_bounds(candidate, bounds, budget)
        expected_nrx = _expected_nrx(candidate, draws, channels, n_draws=200)
        # Dollars moved off the recommended allocation, as a percent of total
        # budget: half the sum of absolute per-channel differences, since
        # every dollar shifted off one channel lands on another and would
        # otherwise be counted twice.
        reallocated_pct = float(np.abs(candidate - recommended).sum()) / (2 * budget) * 100
        signed_reallocated_pct = reallocated_pct if w >= 0 else -reallocated_pct
        row = {
            "blend_weight": round(float(w), 4),
            "signed_reallocated_pct_of_budget": round(signed_reallocated_pct, 2),
            "expected_weekly_nrx": round(expected_nrx, 2),
        }
        row.update({f"spend_{ch}": round(float(c), 1) for ch, c in zip(channels, candidate, strict=True)})
        rows.append(row)

    result = pd.DataFrame(rows)
    result["pct_of_max"] = round(
        result["expected_weekly_nrx"] / result["expected_weekly_nrx"].max() * 100, 2
    )
    return result


# ── Budget recommendation file writer ───────────────────────────────────────────

def build_mmm_budget_recommendation(
    current_spends: np.ndarray,
    optimized_spends: np.ndarray,
    optimized_summary: pd.DataFrame,
    marginal_roi: pd.DataFrame,
    decision_status: dict[str, str] | None = None,
    guardrails: pd.DataFrame | dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build the mmm_budget_recommendation.csv table for the resource-allocation chapter.

    Compares the current allocation against the optimised allocation at the same
    total budget, and flags each channel's headroom relative to saturation based
    on how far the optimiser wants to move its spend. When `decision_status` is
    given, also reports the decision status and optimizer constraint applied,
    so a directional or not-usable channel's small or zero move reads as a
    gate decision, not a saturation finding.
    """
    channels = CHANNELS
    roi_by_channel = marginal_roi.set_index("channel")["marginal_roi_mean"]
    optimized_summary = optimized_summary.set_index("channel")
    constraint_note = build_constrained_channel_note(decision_status or {}, channels, guardrails).set_index("channel")

    rows = []
    for i, ch in enumerate(channels):
        current_spend = float(current_spends[i])
        optimized_spend = float(optimized_summary.loc[ch, "optimized_weekly_spend_median"])
        posterior_mean_optimum = float(optimized_spends[i])
        pct_change = (optimized_spend - current_spend) / current_spend * 100 if current_spend else 0.0

        if optimized_spend > current_spend * 1.05:
            headroom_flag = "below_saturation"
        elif optimized_spend < current_spend * 0.95:
            headroom_flag = "above_saturation"
        else:
            headroom_flag = "near_saturation"

        rows.append({
            "channel": ch,
            "current_weekly_spend": round(current_spend, 1),
            "optimized_weekly_spend_at_current_budget": round(optimized_spend, 1),
            "optimized_weekly_spend_p10": round(float(optimized_summary.loc[ch, "optimized_weekly_spend_p10"]), 1),
            "optimized_weekly_spend_p90": round(float(optimized_summary.loc[ch, "optimized_weekly_spend_p90"]), 1),
            "posterior_mean_optimum_weekly_spend": round(posterior_mean_optimum, 1),
            "current_input_value": round(float(spend_to_exposure(ch, current_spend)), 2),
            "optimized_input_value_median": round(float(spend_to_exposure(ch, optimized_spend)), 2),
            "input_unit": "calls" if ch == "field" else "dollars",
            "pct_change": round(pct_change, 2),
            "marginal_roi_mean": round(float(roi_by_channel[ch]), 4),
            "headroom_flag": headroom_flag,
            "decision_status": constraint_note.loc[ch, "decision_status"],
            "optimizer_constraint": constraint_note.loc[ch, "optimizer_constraint"],
        })

    return pd.DataFrame(rows)


def build_unified_budget_recommendation(
    mmm_recommendation: pd.DataFrame,
    evidence_record: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> pd.DataFrame:
    """Join evidence and guardrail status to the MMM budget recommendation."""
    evidence_record_by_channel = evidence_record.set_index("channel")
    guardrail_by_channel = guardrails.set_index("channel")
    unified = mmm_recommendation.copy()
    unified["evidence_tier"] = unified["channel"].map(evidence_record_by_channel["evidence_tier"])
    unified["causal_anchor"] = unified["evidence_tier"] == "causal-anchored"
    unified["attribution_support"] = unified["channel"].map(
        lambda ch: evidence_record_by_channel.loc[ch, "attribution_signal"] != "not available in current attribution pull"
    )
    unified["experiment_coverage"] = unified["channel"].map(
        lambda ch: evidence_record_by_channel.loc[ch, "experiment_signal"] != "not available in current experiment pull"
    )
    unified["comparability_status"] = unified["channel"].map(evidence_record_by_channel["comparability_status"])
    unified["allowed_budget_move"] = unified["channel"].map(guardrail_by_channel["allowed_budget_move"])
    unified["move_permission"] = unified["channel"].map(guardrail_by_channel["move_permission"])
    unified["max_move_pct"] = unified["channel"].map(guardrail_by_channel["max_move_pct"])
    unified["spend_floor"] = unified["channel"].map(guardrail_by_channel["spend_floor"])
    unified["spend_ceiling"] = unified["channel"].map(guardrail_by_channel["spend_ceiling"])
    unified["new_anchor_required"] = unified["channel"].map(guardrail_by_channel["new_anchor_required"])
    unified["refresh_required"] = unified["channel"].map(guardrail_by_channel["refresh_required"])
    unified["next_measurement_action_type"] = unified["channel"].map(guardrail_by_channel["next_measurement_action_type"])
    unified["anchor_staleness_status"] = unified["channel"].map(guardrail_by_channel["anchor_staleness_status"])
    unified["guardrail_reason"] = unified["channel"].map(guardrail_by_channel["guardrail_reason"])
    unified["next_test_required"] = unified["channel"].map(guardrail_by_channel["new_anchor_required"])
    return unified
