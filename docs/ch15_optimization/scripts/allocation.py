"""Allocation methods for the resource-allocation and optimization chapter.

Every discrete method here runs on a call-step gain matrix ``g[i, k]``: the
incremental NRx of the k-th call to account ``i``. Point-estimate gains come
from the fitted response curve; the uncertainty draws come from the territory
block bootstrap. Because the objective is linear in the step variables, the
sample-average of scenario objectives equals the objective built from mean
step gains, so expected-value optimization keeps the deterministic MILP
dimensions.

Planning functions read the account planning table, the point gains, and the
gain draws. They never read hidden truth. Audit functions take a frozen plan
and score it against the truth table; they are called only after a plan is
fixed, in the audit phase of ``run_analysis``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import lil_matrix

from allocation_config import (
    CHURN_CAP_SHARE,
    CVAR_ALPHA,
    CVAR_PROTECTION_FRACTION,
    FRONTIER_CVAR_FLOOR_NRX,
    LOADED_COST_PER_REP_QUARTER,
    NEAR_OPTIMAL_VALUE_LOSS_BUDGETS_DOLLARS,
    NRX_VALUE_DOLLARS,
    NRX_VALUE_DOLLARS_HIGH,
    NRX_VALUE_DOLLARS_LOW,
    N_CVAR_SCENARIOS,
    PLAN_CHANGE_CAP_GRID,
    PLAN_REFRESH_DATE,
    PLANNING_QUARTER,
    RELEASE_MIN_CVAR_NRX,
    RELEASE_VALUE_LOSS_BUDGET_DOLLARS,
    SEED_FRONTIER,
    MEASUREMENT_READ_DATE,
    WEEKS_PER_QUARTER,
)
from generate_allocation_data import hill_fraction, true_incremental_nrx


# ── Plan representation ─────────────────────────────────────────────────────
# A plan is a NumPy vector of integer calls aligned to the planning table's
# account order. Helpers convert to and from an account-id Series for reporting.

def plan_series(planning: pd.DataFrame, calls: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(calls, dtype=float), index=planning["account_id"].to_numpy())


def series_to_plan(planning: pd.DataFrame, series: pd.Series) -> np.ndarray:
    aligned = series.copy()
    aligned.index = aligned.index.astype(object)
    return aligned.reindex(planning["account_id"].astype(object)).fillna(0.0).to_numpy()


def value_from_gains(gains: np.ndarray, calls: np.ndarray) -> float:
    """Total planning NRx of a plan, summing each account's first ``c`` step gains."""
    calls = np.clip(np.asarray(calls, dtype=int), 0, gains.shape[1])
    cumulative = gains.cumsum(axis=1)
    per_account = np.where(calls > 0, cumulative[np.arange(len(calls)), np.clip(calls - 1, 0, gains.shape[1] - 1)], 0.0)
    return float(per_account.sum())


def account_value_from_gains(gains: np.ndarray, calls: np.ndarray) -> np.ndarray:
    calls = np.clip(np.asarray(calls, dtype=int), 0, gains.shape[1])
    cumulative = gains.cumsum(axis=1)
    return np.where(calls > 0, cumulative[np.arange(len(calls)), np.clip(calls - 1, 0, gains.shape[1] - 1)], 0.0)


def continuous_account_response(planning: pd.DataFrame, segment_params: dict, calls: np.ndarray) -> np.ndarray:
    """Smooth fitted incremental response for a continuous call vector (SLSQP reference)."""
    scaling = (planning["opportunity_nrx"] * planning["response_multiplier"]).to_numpy()
    segments = planning["segment"].to_numpy()
    out = np.zeros(len(planning))
    for seg, (scale, ec50, shape) in segment_params.items():
        mask = segments == seg
        if mask.any():
            out[mask] = scaling[mask] * scale * hill_fraction(calls[mask], ec50, shape)
    return out


# ── 15.2 Marginal value: toy case ───────────────────────────────────────────

_TOY_A = {"ceiling": 34.0, "ec50": 2.0, "shape": 1.15}
_TOY_B = {"ceiling": 20.0, "ec50": 7.0, "shape": 1.35}


def _toy_response(calls: float, ceiling: float, ec50: float, shape: float) -> float:
    return float(ceiling * hill_fraction(np.array([calls]), ec50, shape)[0])


def toy_split_table(total_calls: int = 10) -> pd.DataFrame:
    """Every feasible split of ``total_calls`` between two toy accounts."""
    rows = []
    for calls_a in range(total_calls + 1):
        calls_b = total_calls - calls_a
        nrx_a = _toy_response(calls_a, **_TOY_A)
        nrx_b = _toy_response(calls_b, **_TOY_B)
        rows.append({
            "calls_to_A": calls_a,
            "calls_to_B": calls_b,
            "nrx_A": round(nrx_a, 2),
            "nrx_B": round(nrx_b, 2),
            "total_nrx": round(nrx_a + nrx_b, 2),
        })
    table = pd.DataFrame(rows)
    table["is_best_split"] = table["total_nrx"] == table["total_nrx"].max()
    return table


def toy_greedy_path(total_calls: int = 10) -> pd.DataFrame:
    """Step-by-step greedy assignment on the same two-account toy case."""
    calls_a = calls_b = 0
    rows = []
    for step in range(1, total_calls + 1):
        marg_a = _toy_response(calls_a + 1, **_TOY_A) - _toy_response(calls_a, **_TOY_A)
        marg_b = _toy_response(calls_b + 1, **_TOY_B) - _toy_response(calls_b, **_TOY_B)
        assign_to = "A" if marg_a >= marg_b else "B"
        if assign_to == "A":
            calls_a += 1
            selected_gain = marg_a
        else:
            calls_b += 1
            selected_gain = marg_b
        rows.append({
            "step": step,
            "marginal_if_A": round(marg_a, 2),
            "marginal_if_B": round(marg_b, 2),
            "assigned_to": assign_to,
            "calls_to_A": calls_a,
            "calls_to_B": calls_b,
            "call_number": step,
            "placed_at": f"Account {assign_to}",
            "nrx_added": round(selected_gain, 2),
            "running_nrx": round(
                _toy_response(calls_a, **_TOY_A) + _toy_response(calls_b, **_TOY_B), 2
            ),
            "allocation_after_call": f"A={calls_a}, B={calls_b}",
        })
    return pd.DataFrame(rows)


def account_marginal_return_table(
    planning: pd.DataFrame, point_gains: np.ndarray, territory_id: str = "MI-T1", n: int = 12
) -> pd.DataFrame:
    """Marginal NRx of the next call for one territory's accounts, ranked."""
    idx = {a: i for i, a in enumerate(planning["account_id"])}
    sample = planning[planning["territory_id"] == territory_id].head(n).copy()
    next_gain = []
    at_current = []
    for row in sample.itertuples():
        i = idx[row.account_id]
        c = int(row.current_calls)
        at_current.append(account_value_from_gains(point_gains[i:i + 1], np.array([c]))[0])
        next_gain.append(point_gains[i, c] if c < point_gains.shape[1] else 0.0)
    sample["response_at_current_calls"] = np.round(at_current, 2)
    sample["marginal_nrx_next_call"] = np.round(next_gain, 3)
    sample["planning_action"] = np.where(sample["eligible_flag"], "Compete for next call", "Blocked, zero calls")
    return sample[[
        "account_id", "segment", "access_state", "current_calls",
        "response_at_current_calls", "marginal_nrx_next_call", "planning_action",
    ]].sort_values("marginal_nrx_next_call", ascending=False).reset_index(drop=True)


# ── Channel budget (read from the unified measurement bridge) ────────────────

def _channel_grids(mmm_dir: Path) -> dict[str, pd.DataFrame]:
    curves = pd.read_csv(mmm_dir / "response_curves.csv")
    return {ch: g.sort_values("weekly_spend").reset_index(drop=True) for ch, g in curves.groupby("channel")}


def _grid_value(grid: pd.DataFrame, spend: float) -> float:
    return float(np.interp(spend, grid["weekly_spend"], grid["mean_nrx_contribution"]))


def channel_response_summary(mmm_dir: Path) -> pd.DataFrame:
    """Current spend, response, and marginal ROI by channel, from the fitted MMM curves."""
    marginal = pd.read_csv(mmm_dir / "marginal_roi.csv")
    saturation = pd.read_csv(mmm_dir / "saturation_points.csv")
    grids = _channel_grids(mmm_dir)
    marginal = marginal.copy()
    marginal["current_weekly_nrx"] = [
        _grid_value(grids[ch], spend) for ch, spend in zip(marginal["channel"], marginal["current_weekly_spend"])
    ]
    summary = marginal.merge(saturation[["channel", "at_or_above_saturation"]], on="channel")
    return summary[[
        "channel", "current_weekly_spend", "current_weekly_nrx",
        "marginal_roi_mean", "marginal_roi_p10", "marginal_roi_p90", "at_or_above_saturation",
    ]].sort_values("marginal_roi_mean", ascending=False).reset_index(drop=True)


def channel_budget_move(mmm_dir: Path) -> pd.DataFrame:
    """Bounded channel spend move, quarterly value, and evidence permissions.

    Reads the unified budget recommendation directly. The optimizer that
    equalized marginal return across channels already ran upstream; this table
    reports its move against the current spend and applies the evidence
    guardrail that caps how far each channel is allowed to move.
    """
    bridge = pd.read_csv(mmm_dir / "unified_budget_recommendation.csv")
    recommendation_col = "optimized_weekly_spend_at_current_budget"
    residual = float(bridge["current_weekly_spend"].sum() - bridge[recommendation_col].sum())
    if abs(residual) > 1e-9:
        reconcile_idx = bridge[recommendation_col].idxmax()
        bridge.loc[reconcile_idx, recommendation_col] += residual
    grids = _channel_grids(mmm_dir)
    rows = []
    for row in bridge.itertuples():
        current_spend = float(row.current_weekly_spend)
        recommended_spend = float(getattr(row, recommendation_col))
        current_nrx = _grid_value(grids[row.channel], current_spend)
        recommended_nrx = _grid_value(grids[row.channel], recommended_spend)
        weekly_nrx_change = recommended_nrx - current_nrx
        quarterly_nrx_change = weekly_nrx_change * WEEKS_PER_QUARTER
        rows.append({
            "channel": row.channel,
            "current_weekly_spend": round(current_spend, 1),
            "recommended_weekly_spend": round(recommended_spend, 1),
            "weekly_spend_change": round(recommended_spend - current_spend, 1),
            "quarterly_nrx_change": round(quarterly_nrx_change, 1),
            "quarterly_value_change": round(quarterly_nrx_change * NRX_VALUE_DOLLARS, 0),
            "evidence_tier": row.evidence_tier,
            "allowed_budget_move": row.allowed_budget_move,
            "new_anchor_required": row.new_anchor_required,
            "next_measurement_action_type": row.next_measurement_action_type,
        })
    result = pd.DataFrame(rows).sort_values("weekly_spend_change", ascending=False).reset_index(drop=True)
    if not np.isclose(result["recommended_weekly_spend"].sum(), result["current_weekly_spend"].sum()):
        raise RuntimeError("rounded channel recommendation does not conserve the weekly budget")
    reallocation = pd.read_csv(mmm_dir / "reallocation_decision.csv").iloc[0]
    result.attrs["portfolio_mean_quarterly_nrx_change"] = float(reallocation["mean_nrx_gain"]) * WEEKS_PER_QUARTER
    result.attrs["portfolio_p10_quarterly_nrx_change"] = float(reallocation["p10_nrx_gain"]) * WEEKS_PER_QUARTER
    return result


def channel_movement_cap_tradeoff(mmm_dir: Path, cap_step: float = 0.05) -> dict:
    """Re-solve the channel mix after adding 5 percentage points to evidence caps."""
    bridge = pd.read_csv(mmm_dir / "unified_budget_recommendation.csv")
    grids = _channel_grids(mmm_dir)
    channels = bridge["channel"].tolist()
    current = bridge["current_weekly_spend"].to_numpy(dtype=float)
    baseline = bridge["optimized_weekly_spend_at_current_budget"].to_numpy(dtype=float)
    total_budget = float(baseline.sum())
    bounds = []
    for row in bridge.itertuples():
        wider = float(row.max_move_pct) + cap_step
        if row.move_permission == "increase-capped":
            lower = 0.0
            upper = row.current_weekly_spend * (1.0 + wider)
        else:
            lower = row.current_weekly_spend * (1.0 - wider)
            upper = row.current_weekly_spend * (1.0 + wider)
        grid = grids[row.channel]
        bounds.append((max(lower, float(grid["weekly_spend"].min())), min(upper, float(grid["weekly_spend"].max()))))

    def total_response(spend: np.ndarray) -> float:
        return float(sum(_grid_value(grids[ch], value) for ch, value in zip(channels, spend)))

    result = minimize(
        lambda x: -total_response(x),
        np.clip(baseline, [b[0] for b in bounds], [b[1] for b in bounds]),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda x: x.sum() - total_budget},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"channel cap re-solve failed: {result.message}")
    baseline_response = total_response(baseline)
    wider_response = total_response(result.x)
    return {
        "tested_change": f"Add {int(cap_step * 100)} percentage points to channel movement bounds",
        "expected_nrx_change": (wider_response - baseline_response) * WEEKS_PER_QUARTER,
        "current_weekly_spend": baseline,
        "resolved_weekly_spend": result.x,
        "budget_residual": float(result.x.sum() - total_budget),
    }


def _national_step_columns(planning: pd.DataFrame) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Enumerate national call-step columns and group them by account."""
    var_index: list[tuple[int, int]] = []
    by_account: list[list[int]] = [[] for _ in range(len(planning))]
    for i, max_calls in enumerate(planning["max_calls"].to_numpy(dtype=int)):
        for k in range(1, max_calls + 1):
            by_account[i].append(len(var_index))
            var_index.append((i, k))
    return var_index, by_account


def _solve_national_step_model(
    planning: pd.DataFrame,
    objective_gains: np.ndarray,
    territories: pd.DataFrame,
    *,
    integral: bool = True,
    enforce_min_coverage: bool = True,
    movement_cap_calls: float | None = None,
    scenario_gains: np.ndarray | None = None,
    cvar_floor: float | None = None,
    maximize_cvar: bool = False,
    require_full_territory_capacity: bool = False,
    alpha: float = CVAR_ALPHA,
) -> dict:
    """Solve one national step model with global movement and optional CVaR.

    Territory capacity remains a hard local rule. Total account-call change is
    one national epsilon measured against the current national call plan.
    When scenarios are supplied, the Rockafellar-Uryasev variables model the
    lower-tail CVaR of national plan value, including response shocks shared
    across territories.
    """
    var_index, by_account = _national_step_columns(planning)
    n_y = len(var_index)
    n_accounts = len(planning)
    n_scenarios = 0 if scenario_gains is None else int(scenario_gains.shape[0])
    has_risk = n_scenarios > 0
    n_base = n_y + n_accounts
    eta_col = n_base if has_risk else None
    z0 = n_base + 1 if has_risk else None
    n_total = n_base + (1 + n_scenarios if has_risk else 0)

    lb = np.zeros(n_total)
    ub = np.full(n_total, np.inf)
    ub[:n_y] = 1.0
    integrality = np.zeros(n_total)
    if integral:
        integrality[:n_y] = 1
    if has_risk:
        lb[eta_col] = -np.inf

    if enforce_min_coverage:
        for i, min_calls in enumerate(planning["min_calls"].to_numpy(dtype=int)):
            for col in by_account[i][:min_calls]:
                lb[col] = 1.0

    objective_flat = np.array([objective_gains[i, k - 1] for i, k in var_index])
    c = np.zeros(n_total)
    if maximize_cvar:
        if not has_risk:
            raise ValueError("scenario_gains are required when maximize_cvar=True")
        tail_weight = 1.0 / ((1.0 - alpha) * n_scenarios)
        c[eta_col] = -1.0
        c[z0:] = tail_weight
    else:
        c[:n_y] = -objective_flat

    constraints: list[LinearConstraint] = []
    territory_ids = territories["territory_id"].tolist()
    territory_row = {t: r for r, t in enumerate(territory_ids)}
    a_capacity = lil_matrix((len(territory_ids), n_total))
    for col, (i, _) in enumerate(var_index):
        a_capacity[territory_row[planning["territory_id"].iloc[i]], col] = 1.0
    capacity = territories.set_index("territory_id").loc[territory_ids, "quarterly_call_capacity"].to_numpy(dtype=float)
    capacity_lb = capacity if require_full_territory_capacity else -np.inf
    constraints.append(LinearConstraint(a_capacity.tocsr(), capacity_lb, capacity))

    n_order = sum(max(0, len(cols) - 1) for cols in by_account)
    if n_order:
        a_order = lil_matrix((n_order, n_total))
        row = 0
        for cols in by_account:
            for earlier, later in zip(cols, cols[1:]):
                a_order[row, later] = 1.0
                a_order[row, earlier] = -1.0
                row += 1
        constraints.append(LinearConstraint(a_order.tocsr(), -np.inf, 0.0))

    current = planning["current_calls"].to_numpy(dtype=float)
    a_deviation = lil_matrix((2 * n_accounts, n_total))
    deviation_ub = np.empty(2 * n_accounts)
    for i, cols in enumerate(by_account):
        dev_col = n_y + i
        for col in cols:
            a_deviation[2 * i, col] = 1.0
            a_deviation[2 * i + 1, col] = -1.0
        a_deviation[2 * i, dev_col] = -1.0
        a_deviation[2 * i + 1, dev_col] = -1.0
        deviation_ub[2 * i] = current[i]
        deviation_ub[2 * i + 1] = -current[i]
    constraints.append(LinearConstraint(a_deviation.tocsr(), -np.inf, deviation_ub))

    if movement_cap_calls is not None:
        a_movement = lil_matrix((1, n_total))
        a_movement[0, n_y:n_base] = 1.0
        constraints.append(LinearConstraint(a_movement.tocsr(), -np.inf, float(movement_cap_calls)))

    if has_risk:
        tail_weight = 1.0 / ((1.0 - alpha) * n_scenarios)
        a_tail = lil_matrix((n_scenarios, n_total))
        for s in range(n_scenarios):
            for col, (i, k) in enumerate(var_index):
                a_tail[s, col] = -scenario_gains[s, i, k - 1]
            a_tail[s, eta_col] = 1.0
            a_tail[s, z0 + s] = -1.0
        constraints.append(LinearConstraint(a_tail.tocsr(), -np.inf, 0.0))
        if cvar_floor is not None:
            a_floor = lil_matrix((1, n_total))
            a_floor[0, eta_col] = -1.0
            a_floor[0, z0:] = tail_weight
            constraints.append(LinearConstraint(a_floor.tocsr(), -np.inf, -float(cvar_floor)))

    n_constraint_rows = int(sum(constraint.A.shape[0] for constraint in constraints))
    dimensions = {
        "step_variables": n_y,
        "binary_variables": n_y if integral else 0,
        "continuous_variables": n_total - (n_y if integral else 0),
        "constraint_rows": n_constraint_rows,
        "scenario_count": n_scenarios,
    }

    result = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"mip_rel_gap": 1e-8},
    )
    if not result.success:
        return {
            "success": False,
            "status": int(result.status),
            "message": str(result.message),
            "calls": current.copy(),
            "step_values": np.zeros(n_y),
            "objective_nrx": float("nan"),
            "plan_change_calls": 0.0,
            "cvar_model": float("nan"),
            "mip_gap": float("nan"),
            "dimensions": dimensions,
        }

    step_values = result.x[:n_y]
    calls = np.array([step_values[cols].sum() if cols else 0.0 for cols in by_account])
    expected_value = float(objective_flat @ step_values)
    cvar_model = float("nan")
    if has_risk:
        tail_weight = 1.0 / ((1.0 - alpha) * n_scenarios)
        cvar_model = float(result.x[eta_col] - tail_weight * result.x[z0:].sum())
    return {
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "calls": np.round(calls) if integral else calls,
        "step_values": step_values,
        "objective_nrx": expected_value,
        "plan_change_calls": float(result.x[n_y:n_base].sum()),
        "cvar_model": cvar_model,
        "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
        "dimensions": dimensions,
    }


# ── Field call plans ────────────────────────────────────────────────────────

def _territory_arrays(planning: pd.DataFrame, territories: pd.DataFrame, gains: np.ndarray):
    """Iterate territories, yielding the per-territory slice of planning and gains."""
    idx = {a: i for i, a in enumerate(planning["account_id"])}
    capacity_by_t = territories.set_index("territory_id")["quarterly_call_capacity"]
    for t_id, group in planning.groupby("territory_id", sort=False):
        positions = [idx[a] for a in group["account_id"]]
        yield t_id, group, np.array(positions), float(capacity_by_t[t_id])


def greedy_call_plan(planning: pd.DataFrame, gains: np.ndarray, territories: pd.DataFrame) -> np.ndarray:
    """Marginal-gain greedy allocation per territory, seeded from minimum coverage."""
    calls = np.zeros(len(planning))
    for _, group, positions, capacity in _territory_arrays(planning, territories, gains):
        g = gains[positions]
        cap = group["max_calls"].to_numpy()
        min_calls = group["min_calls"].to_numpy()
        local = min_calls.astype(float).copy()
        used = local.sum()
        while used < capacity - 1e-9:
            room = local < cap
            step_idx = np.clip(local.astype(int), 0, g.shape[1] - 1)
            marginal = np.where(room, g[np.arange(len(g)), step_idx], -np.inf)
            best = int(np.argmax(marginal))
            if marginal[best] <= 1e-12:
                break
            local[best] += 1
            used += 1
        calls[positions] = local
    return calls


def slsqp_call_plan(
    planning: pd.DataFrame,
    segment_params: dict,
    territories: pd.DataFrame,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Continuous relaxation with SLSQP: the smooth planning-limit reference."""
    calls = np.zeros(len(planning))
    idx = {a: i for i, a in enumerate(planning["account_id"])}
    capacity_by_t = territories.set_index("territory_id")["quarterly_call_capacity"]
    diagnostics = []
    for t_id, group in planning.groupby("territory_id", sort=False):
        positions = np.array([idx[a] for a in group["account_id"]])
        capacity = float(capacity_by_t[t_id])
        lower = group["min_calls"].to_numpy().astype(float)
        upper = group["max_calls"].to_numpy().astype(float)
        sub = group.reset_index(drop=True)

        def neg_value(x: np.ndarray) -> float:
            return -float(np.sum(continuous_account_response(sub, segment_params, x)))

        x0 = np.clip(lower + (capacity - lower.sum()) / max(len(group), 1), lower, upper)
        result = minimize(
            neg_value, x0, method="SLSQP", bounds=list(zip(lower, upper)),
            constraints={"type": "ineq", "fun": lambda x: capacity - x.sum()},
            options={"ftol": 1e-7, "maxiter": 400},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP failed in {t_id}: {result.message}")
        calls[positions] = np.clip(result.x, lower, upper)
        multipliers = np.atleast_1d(getattr(result, "multipliers", np.array([np.nan])))
        diagnostics.append({
            "territory_id": t_id,
            "success": bool(result.success),
            "continuous_nrx": round(float(-result.fun), 3),
            "capacity_residual_calls": round(float(capacity - result.x.sum()), 6),
            "capacity_multiplier_nrx_per_call": round(float(multipliers[0]), 6),
            "fractional_accounts": int((np.abs(result.x - np.round(result.x)) > 1e-6).sum()),
        })
    if return_diagnostics:
        return calls, pd.DataFrame(diagnostics)
    return calls


def lp_relaxation_call_plan(
    planning: pd.DataFrame,
    gains: np.ndarray,
    territories: pd.DataFrame,
    movement_cap_share: float | None = CHURN_CAP_SHARE,
    movement_cap_calls: float | None = None,
) -> dict:
    """LP relaxation of the rule-complete step MILP: drop integrality, one continuous solve.

    The relaxation carries the same capacity, coverage, and movement-cap
    constraints as the release MILP. Where the movement cap binds, the last
    unit of movement budget splits across accounts and the step variables come
    back fractional, which is why the plan cannot be released as written.
    """
    movement_cap = movement_cap_calls
    if movement_cap is None and movement_cap_share is not None:
        movement_cap = movement_cap_share * float(planning["current_calls"].sum())
    solved = _solve_national_step_model(
        planning,
        gains,
        territories,
        integral=False,
        movement_cap_calls=movement_cap,
    )
    x = solved["step_values"]
    return {
        "calls": solved["calls"],
        "fractional_step_variables": int(np.sum((x > 1e-6) & (x < 1 - 1e-6))),
        "objective_nrx": solved["objective_nrx"],
        "success": solved["success"],
        "dimensions": solved["dimensions"],
        "movement_cap_calls": movement_cap,
    }


def milp_call_plan(
    planning: pd.DataFrame,
    gains: np.ndarray,
    territories: pd.DataFrame,
    enforce_min_coverage: bool = True,
    movement_cap_share: float | None = CHURN_CAP_SHARE,
) -> tuple[np.ndarray, dict]:
    """Integer, rule-complete call plan built on a step-gain matrix."""
    movement_cap = None
    if movement_cap_share is not None:
        movement_cap = movement_cap_share * float(planning["current_calls"].sum())
    solved = _solve_national_step_model(
        planning,
        gains,
        territories,
        integral=True,
        enforce_min_coverage=enforce_min_coverage,
        movement_cap_calls=movement_cap,
    )
    diagnostics = {
        t_id: {
            "feasible": solved["success"],
            "relaxed_rules": [],
            "optimality_gap": solved["mip_gap"],
            "n_accounts": int((planning["territory_id"] == t_id).sum()),
        }
        for t_id in territories["territory_id"]
    }
    diagnostics["national"] = {
        "feasible": solved["success"],
        "relaxed_rules": [],
        "optimality_gap": solved["mip_gap"],
        "movement_cap_calls": movement_cap,
        "plan_change_calls": solved["plan_change_calls"],
        **solved["dimensions"],
    }
    return solved["calls"], diagnostics


def expected_gain_milp_plan(
    planning: pd.DataFrame,
    gain_draws: np.ndarray,
    territories: pd.DataFrame,
    movement_cap_share: float | None = CHURN_CAP_SHARE,
) -> tuple[np.ndarray, dict]:
    """Sample-average (SAA) plan: MILP on the mean step gains across draws."""
    mean_gains = gain_draws.mean(axis=0)
    return milp_call_plan(planning, mean_gains, territories, True, movement_cap_share)


def cvar_milp_plan(
    planning: pd.DataFrame,
    gain_draws: np.ndarray,
    territories: pd.DataFrame,
    alpha: float = CVAR_ALPHA,
    n_scenarios: int = N_CVAR_SCENARIOS,
    movement_cap_share: float | None = CHURN_CAP_SHARE,
    seed: int = SEED_FRONTIER,
) -> tuple[np.ndarray, dict]:
    """Maximize expected value subject to an explicit lower-tail CVaR floor."""
    current = planning["current_calls"].to_numpy(dtype=float)
    scenarios = _stratified_scenario_sample(
        gain_draws, current, n_scenarios, seed
    )
    mean_gains = gain_draws.mean(axis=0)
    movement_cap = None
    if movement_cap_share is not None:
        movement_cap = movement_cap_share * float(planning["current_calls"].sum())

    expected = _solve_national_step_model(
        planning, mean_gains, territories, movement_cap_calls=movement_cap
    )
    risk_anchor = _solve_national_step_model(
        planning,
        mean_gains,
        territories,
        movement_cap_calls=movement_cap,
        scenario_gains=scenarios,
        maximize_cvar=True,
        alpha=alpha,
    )
    expected_cvar = cvar_of_values(scenario_plan_values(scenarios, expected["calls"]), alpha)
    maximum_cvar = cvar_of_values(scenario_plan_values(scenarios, risk_anchor["calls"]), alpha)
    floor = expected_cvar + CVAR_PROTECTION_FRACTION * max(0.0, maximum_cvar - expected_cvar)
    protected = _solve_national_step_model(
        planning,
        mean_gains,
        territories,
        movement_cap_calls=movement_cap,
        scenario_gains=scenarios,
        cvar_floor=floor,
        alpha=alpha,
    )
    diagnostics = {
        "national": {
            "feasible": protected["success"],
            "scenario_count": len(scenarios),
            "expected_plan_cvar": expected_cvar,
            "maximum_cvar": maximum_cvar,
            "cvar_floor": floor,
            "optimality_gap": protected["mip_gap"],
        }
    }
    return protected["calls"], diagnostics


def _stratified_scenario_sample(
    gain_draws: np.ndarray,
    reference_calls: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    """Keep evenly spaced response draws after ordering by reference-plan value."""
    n_draws = gain_draws.shape[0]
    if n_scenarios >= n_draws:
        return gain_draws
    rng = np.random.default_rng(seed)
    score = scenario_plan_values(gain_draws, reference_calls)
    order = np.argsort(score + rng.normal(0.0, 1e-12, size=n_draws))
    positions = np.floor((np.arange(n_scenarios) + 0.5) * n_draws / n_scenarios).astype(int)
    return gain_draws[order[positions]]


# ── Scenario scoring ────────────────────────────────────────────────────────

def scenario_plan_values(gain_draws: np.ndarray, calls: np.ndarray) -> np.ndarray:
    """Plan value under every uncertainty draw."""
    calls = np.clip(np.asarray(calls, dtype=int), 0, gain_draws.shape[2])
    cumulative = gain_draws.cumsum(axis=2)
    n = len(calls)
    per = np.where(
        calls[None, :] > 0,
        cumulative[:, np.arange(n), np.clip(calls - 1, 0, gain_draws.shape[2] - 1)],
        0.0,
    )
    return per.sum(axis=1)


def cvar_of_values(values: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Lower-tail CVaR: mean of the worst ``1-alpha`` fraction of scenario values."""
    values = np.sort(np.asarray(values))
    k = max(1, int(np.ceil((1.0 - alpha) * len(values))))
    return float(values[:k].mean())


def plan_risk_summary(
    gain_draws: np.ndarray,
    calls: np.ndarray,
    alpha: float = CVAR_ALPHA,
    benchmark_calls: np.ndarray | None = None,
) -> dict:
    values = scenario_plan_values(gain_draws, calls)
    summary = {
        "expected_nrx": float(values.mean()),
        "p10_nrx": float(np.quantile(values, 0.10)),
        "cvar_nrx": cvar_of_values(values, alpha),
        "std_nrx": float(values.std()),
    }
    if benchmark_calls is not None:
        benchmark = scenario_plan_values(gain_draws, benchmark_calls)
        summary["prob_under_benchmark"] = float((values < benchmark).mean())
    return summary


def incremental_plan_risk_summary(
    gain_draws: np.ndarray,
    calls: np.ndarray,
    benchmark_calls: np.ndarray,
    alpha: float = CVAR_ALPHA,
) -> dict:
    """Expected and lower-tail incremental value against a fixed benchmark plan."""
    incremental = (
        scenario_plan_values(gain_draws, calls)
        - scenario_plan_values(gain_draws, benchmark_calls)
    )
    return {
        "expected_incremental_nrx": float(incremental.mean()),
        "p10_incremental_nrx": float(np.quantile(incremental, 0.10)),
        "cvar_incremental_nrx": cvar_of_values(incremental, alpha),
        "probability_below_zero": float((incremental < 0).mean()),
    }


def plan_change_calls(planning: pd.DataFrame, calls: np.ndarray) -> float:
    """Total account-level call additions plus removals."""
    return plan_change_summary(planning, calls)["total_account_call_changes"]


def plan_change_summary(planning: pd.DataFrame, calls: np.ndarray) -> dict[str, float]:
    """Decompose plan change into additions, removals, and reassignments."""
    current = planning["current_calls"].to_numpy().astype(float)
    delta = np.asarray(calls, dtype=float) - current
    added = float(np.clip(delta, 0, None).sum())
    removed = float(np.clip(-delta, 0, None).sum())
    return {
        "calls_added": added,
        "calls_removed": removed,
        "net_calls_added": added - removed,
        "calls_reassigned": min(added, removed),
        "total_account_call_changes": added + removed,
    }


def coverage_min_by_territory(planning: pd.DataFrame, calls: np.ndarray) -> float:
    """Smallest eligible-account coverage share across territories."""
    df = planning.copy()
    df["calls"] = np.asarray(calls, dtype=float)
    eligible = df[df["eligible_flag"]]
    by_t = eligible.groupby("territory_id").apply(
        lambda g: float((g["calls"] >= 1).mean()), include_groups=False
    )
    return float(by_t.min()) if len(by_t) else 0.0


def coverage_summary(planning: pd.DataFrame, calls: np.ndarray) -> dict:
    """Minimum, maximum, and spread of eligible-account territory coverage."""
    df = planning.copy()
    df["calls"] = np.asarray(calls, dtype=float)
    eligible = df[df["eligible_flag"]]
    by_t = eligible.groupby("territory_id").apply(
        lambda g: float((g["calls"] >= g["min_calls"]).mean()), include_groups=False
    )
    if by_t.empty:
        return {"min": 0.0, "max": 0.0, "spread": 0.0}
    return {
        "min": float(by_t.min()),
        "max": float(by_t.max()),
        "spread": float(by_t.max() - by_t.min()),
    }


# ── Executable-optimization scorecard ───────────────────────────────────────

def call_plan_scorecard(
    planning: pd.DataFrame,
    point_gains: np.ndarray,
    territories: pd.DataFrame,
    plans: dict[str, np.ndarray],
    runtimes: dict[str, float] | None = None,
    predicted_values: dict[str, float] | None = None,
    operational_integrality: dict[str, bool] | None = None,
    movement_cap_share: float = CHURN_CAP_SHARE,
) -> pd.DataFrame:
    """Score call-plan methods on predicted value, feasibility, and release role.

    A plan is a release candidate only if it is integer and clears every hard
    rule: territory capacity, zero calls to blocked accounts, minimum coverage
    for eligible accounts, and the national movement cap.
    """
    runtimes = runtimes or {}
    predicted_values = predicted_values or {}
    operational_integrality = operational_integrality or {}
    current = planning["current_calls"].to_numpy().astype(float)
    eligible = planning["eligible_flag"].to_numpy()
    capacity_by_t = territories.set_index("territory_id")["quarterly_call_capacity"]
    territory = planning["territory_id"].to_numpy()
    min_calls = planning["min_calls"].to_numpy()

    rows = []
    for method, calls in plans.items():
        calls = np.asarray(calls, dtype=float)
        predicted = predicted_values.get(method, value_from_gains(point_gains, calls))
        is_integer = operational_integrality.get(
            method, bool(np.allclose(calls, np.round(calls)))
        )
        changed = int((np.round(calls, 2) != np.round(current, 2)).sum())

        used = pd.Series(calls).groupby(territory).sum()
        util = float((used / capacity_by_t.reindex(used.index).to_numpy()).max())
        capacity_ok = util <= 1.005
        coverage_ok = bool((calls[eligible] >= min_calls[eligible] - 1e-6).all())
        blocked_ok = bool((calls[~eligible] <= 1e-6).all())
        movement = float(np.abs(calls - current).sum())
        movement_budget = float(current.sum()) * movement_cap_share
        movement_ok = movement <= movement_budget + 1e-6
        rules_complete = capacity_ok and coverage_ok and blocked_ok and movement_ok

        if "current" in method.lower():
            role = "incumbent baseline"
        elif not is_integer:
            role = "continuous reference"
        elif rules_complete:
            role = "release candidate"
        else:
            role = "planning benchmark"

        rows.append({
            "method": method,
            "integer_calls": is_integer,
            "total_calls": round(float(calls.sum()), 1),
            "predicted_nrx": round(predicted, 1),
            "accounts_changed": changed,
            "max_capacity_use_pct": round(util * 100, 1),
            "capacity_ok": capacity_ok,
            "coverage_ok": coverage_ok,
            "blocked_ok": blocked_ok,
            "movement_ok": movement_ok,
            "rules_complete": rules_complete,
            "decision_role": role,
            "runtime_seconds": round(runtimes.get(method, float("nan")), 3),
        })
    return pd.DataFrame(rows).sort_values("predicted_nrx", ascending=False).reset_index(drop=True)


def milp_worst_optimality_gap(diagnostics: dict) -> float:
    gaps = [d.get("optimality_gap", 0.0) for d in diagnostics.values() if d.get("feasible")]
    gaps = [g for g in gaps if g == g]
    return float(max(gaps)) if gaps else 0.0


# ── Headcount (priced on planning value, never on hidden truth) ──────────────

def headcount_business_case(
    planning: pd.DataFrame, point_gains: np.ndarray, territories: pd.DataFrame
) -> pd.DataFrame:
    """Net value of one added rep-quarter of capacity by territory, on planning gains.

    Every territory is tested with the same capacity bump (one median rep's
    quarterly capacity). The marginal value is the planning-NRx gain the field
    MILP earns from that capacity, valued at the stated NRx assumption, minus
    the loaded rep cost. No true-curve response enters this decision.
    """
    typical = float(territories["calls_per_rep_per_week"].median() * WEEKS_PER_QUARTER)
    # Added headcount brings new call capacity. Keep the original movement
    # allowance for reshuffling existing calls, then add the rep's calls on top.
    movement_cap = CHURN_CAP_SHARE * float(planning["current_calls"].sum())
    base = _solve_national_step_model(
        planning, point_gains, territories, movement_cap_calls=movement_cap
    )
    base_value = base["objective_nrx"]
    rows = []
    for t_id in territories["territory_id"]:
        bumped = territories.copy()
        bumped.loc[bumped["territory_id"] == t_id, "quarterly_call_capacity"] += typical
        solved = _solve_national_step_model(
            planning, point_gains, bumped, movement_cap_calls=movement_cap + typical
        )
        bumped_value = solved["objective_nrx"]
        t_row = territories.loc[territories["territory_id"] == t_id].iloc[0]
        marginal_nrx = float(bumped_value - base_value)
        marginal_value = marginal_nrx * NRX_VALUE_DOLLARS
        rows.append({
            "territory_id": t_row["territory_id"],
            "region": t_row["region"],
            "current_reps": int(t_row["n_reps"]),
            "marginal_nrx_per_added_rep": round(marginal_nrx, 2),
            "marginal_value_dollars": round(marginal_value, 0),
            "loaded_cost_per_rep_quarter": LOADED_COST_PER_REP_QUARTER,
            "net_value_per_added_rep": round(marginal_value - LOADED_COST_PER_REP_QUARTER, 0),
            "recommendation": "Add a rep" if marginal_value > LOADED_COST_PER_REP_QUARTER else "Hold",
        })
    return pd.DataFrame(rows).sort_values("net_value_per_added_rep", ascending=False).reset_index(drop=True)


# ── Constraint prices ───────────────────────────────────────────────────────

def continuous_constraint_prices(
    slsqp_diagnostics: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Local value of relaxing a continuous bound by one unit.

    The capacity price is the median SLSQP KKT multiplier across territories
    whose smooth capacity constraint binds. The budget-dollar price reads the
    highest channel marginal ROI upstream.
    """
    binding = slsqp_diagnostics[
        slsqp_diagnostics["capacity_residual_calls"].abs() <= 1e-4
    ]
    capacity_price_nrx = float(binding["capacity_multiplier_nrx_per_call"].median())
    best_channel = channel_summary.iloc[0]
    quarterly_budget_nrx = float(best_channel["marginal_roi_mean"]) * WEEKS_PER_QUARTER
    rows = [
        {
            "constraint": "One call of quarterly capacity",
            "unit": "1 call",
            "local_value_nrx": round(capacity_price_nrx, 3),
            "local_value_dollars": round(capacity_price_nrx * NRX_VALUE_DOLLARS, 0),
            "basis": "Median SLSQP KKT multiplier across binding territories",
        },
        {
            "constraint": f"One $1K/week budget increase ({best_channel['channel']})",
            "unit": "$1K per week for 13 weeks",
            "local_value_nrx": round(quarterly_budget_nrx, 3),
            "local_value_dollars": round(quarterly_budget_nrx * NRX_VALUE_DOLLARS, 0),
            "basis": f"Quarterly marginal response of {best_channel['channel']} from the fitted MMM curve",
        },
    ]
    return pd.DataFrame(rows)


def discrete_constraint_tradeoffs(
    planning: pd.DataFrame,
    gain_draws: np.ndarray,
    territories: pd.DataFrame,
    movement_cap_share: float = CHURN_CAP_SHARE,
    movement_cap_step: float = 0.05,
    capacity_territory: str | None = None,
) -> pd.DataFrame:
    """Discrete opportunity cost of field rules, from controlled MILP re-solves.

    Each row states one exact change with every other input held fixed, then
    reports the expected-value, downside (CVaR), and plan-change effect of the
    re-solved plan against the release plan.
    """
    mean_gains = gain_draws.mean(axis=0)
    typical = float(territories["calls_per_rep_per_week"].median() * WEEKS_PER_QUARTER)

    def summarize(calls):
        risk = plan_risk_summary(gain_draws, calls)
        return risk["expected_nrx"], risk["cvar_nrx"], plan_change_calls(planning, calls)

    base_calls, _ = milp_call_plan(planning, mean_gains, territories, True, movement_cap_share)
    base_e, base_c, base_change = summarize(base_calls)

    scenarios = []
    # Territory capacity: use the best territory from the separate headcount
    # business case when supplied, keeping the tested change consistent.
    tight_t = capacity_territory or str(territories.iloc[0]["territory_id"])
    bumped = territories.copy()
    bumped.loc[bumped["territory_id"] == tight_t, "quarterly_call_capacity"] += typical
    cap_calls, _ = milp_call_plan(planning, mean_gains, bumped, True, movement_cap_share)
    scenarios.append(("Territory capacity", f"+1 rep-quarter in {tight_t}", cap_calls))

    # Movement cap: add 5 percentage points.
    wider_calls, _ = milp_call_plan(planning, mean_gains, territories, True, movement_cap_share + movement_cap_step)
    scenarios.append(("Account-call change cap", f"+{int(movement_cap_step * 100)} percentage points", wider_calls))

    # Minimum coverage removed.
    no_cov_calls, _ = milp_call_plan(planning, mean_gains, territories, False, movement_cap_share)
    scenarios.append(("Minimum coverage", "Remove one-call floor", no_cov_calls))

    rows = [{
        "rule": "Release plan (all rules)",
        "tested_change": "none",
        "expected_nrx": round(base_e, 1),
        "expected_nrx_change": 0.0,
        "cvar_nrx_change": 0.0,
        "plan_change_calls": round(base_change, 0),
        "plan_change_effect_calls": 0.0,
    }]
    for rule, change, calls in scenarios:
        e, c, ch = summarize(calls)
        rows.append({
            "rule": rule,
            "tested_change": change,
            "expected_nrx": round(e, 1),
            "expected_nrx_change": round(e - base_e, 1),
            "cvar_nrx_change": round(c - base_c, 1),
            "plan_change_calls": round(ch, 0),
            "plan_change_effect_calls": round(ch - base_change, 0),
        })
    return pd.DataFrame(rows)


# ── Epsilon-constraint frontier ─────────────────────────────────────────────

def epsilon_frontier(
    planning: pd.DataFrame,
    gain_draws: np.ndarray,
    territories: pd.DataFrame,
    plan_change_caps: list[float] = PLAN_CHANGE_CAP_GRID,
    cvar_floor_nrx: list[float] = FRONTIER_CVAR_FLOOR_NRX,
    alpha: float = CVAR_ALPHA,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate plans under shared change limits and weak-quarter floors."""
    current = planning["current_calls"].to_numpy().astype(float)
    total_current_calls = max(current.sum(), 1.0)
    mean_gains = gain_draws.mean(axis=0)
    scenarios = _stratified_scenario_sample(
        gain_draws, current, N_CVAR_SCENARIOS, SEED_FRONTIER
    )

    plans: dict[str, np.ndarray] = {"current": current}
    metadata: dict[str, dict] = {
        "current": {
            "movement_epsilon_pct": 0.0,
            "max_plan_change_calls": 0.0,
            "cvar_floor_nrx": np.nan,
        }
    }
    solve_grid = []
    seen = {tuple(np.round(current).astype(int)): "current"}
    for cap in plan_change_caps:
        movement_limit = cap * total_current_calls
        expected = _solve_national_step_model(
            planning, mean_gains, territories, movement_cap_calls=movement_limit
        )
        if not expected["success"]:
            solve_grid.append({
                "movement_epsilon_pct": cap * 100,
                "max_plan_change_calls": movement_limit,
                "cvar_floor_nrx": np.nan,
                "solve_status": "infeasible",
                "candidate_plan": "",
            })
            continue
        expected_cvar = cvar_of_values(scenario_plan_values(scenarios, expected["calls"]), alpha)
        requests = [(None, expected)]
        for floor in cvar_floor_nrx:
            if expected_cvar >= floor:
                solved = expected
            else:
                solved = _solve_national_step_model(
                    planning,
                    mean_gains,
                    territories,
                    movement_cap_calls=movement_limit,
                    scenario_gains=scenarios,
                    cvar_floor=floor,
                    alpha=alpha,
                )
            requests.append((floor, solved))

        for floor, solved in requests:
            suffix = "expected" if floor is None else f"cvar_{int(floor)}"
            requested_plan_id = f"change_{int(cap * 100):02d}_{suffix}"
            status = "optimal" if solved["success"] else "infeasible"
            if not solved["success"]:
                solve_grid.append({
                    "movement_epsilon_pct": cap * 100,
                    "max_plan_change_calls": movement_limit,
                    "cvar_floor_nrx": floor,
                    "solve_status": status,
                    "candidate_plan": "",
                })
                continue
            key = tuple(np.round(solved["calls"]).astype(int))
            if key in seen:
                plan_id = seen[key]
            else:
                plan_id = requested_plan_id
                seen[key] = plan_id
                plans[plan_id] = solved["calls"]
                metadata[plan_id] = {
                    "movement_epsilon_pct": cap * 100,
                    "max_plan_change_calls": movement_limit,
                    "cvar_floor_nrx": floor,
                }
            solve_grid.append({
                "movement_epsilon_pct": cap * 100,
                "max_plan_change_calls": movement_limit,
                "cvar_floor_nrx": floor,
                "solve_status": status,
                "candidate_plan": plan_id,
            })

    rows = []
    current_risk = plan_risk_summary(gain_draws, current, alpha, current)
    for plan_id, calls in plans.items():
        risk = plan_risk_summary(gain_draws, calls, alpha, current)
        change = plan_change_summary(planning, calls)
        coverage = coverage_summary(planning, calls)
        rows.append({
            "plan_id": plan_id,
            "committee_label": "Current" if plan_id == "current" else "",
            "expected_nrx": risk["expected_nrx"],
            "expected_gain_vs_current_nrx": risk["expected_nrx"] - current_risk["expected_nrx"],
            "cvar_nrx": risk["cvar_nrx"],
            "cvar_gain_vs_current_nrx": risk["cvar_nrx"] - current_risk["cvar_nrx"],
            "cvar_shortfall_nrx": risk["expected_nrx"] - risk["cvar_nrx"],
            "p10_nrx": risk["p10_nrx"],
            "prob_under_current": risk["prob_under_benchmark"],
            "plan_change_calls": change["total_account_call_changes"],
            "plan_change_pct": change["total_account_call_changes"] / total_current_calls * 100,
            "calls_added": change["calls_added"],
            "calls_removed": change["calls_removed"],
            "net_calls_added": change["net_calls_added"],
            "calls_reassigned": change["calls_reassigned"],
            "min_territory_coverage": coverage["min"],
            "coverage_spread": coverage["spread"],
            "accounts_changed": int((np.round(calls) != np.round(current)).sum()),
            **metadata[plan_id],
        })
    frontier = pd.DataFrame(rows)
    frontier.attrs["solve_grid"] = pd.DataFrame(solve_grid)
    return frontier, plans


def filter_nondominated_plans(frontier: pd.DataFrame) -> pd.DataFrame:
    """Flag Pareto-nondominated plans: higher expected value, lower plan change, higher CVaR."""
    frontier = frontier.copy()
    dominated = np.zeros(len(frontier), dtype=bool)
    e = frontier["expected_nrx"].to_numpy()
    ch = frontier["plan_change_calls"].to_numpy()
    cv = frontier["cvar_nrx"].to_numpy()
    for i in range(len(frontier)):
        better = (e >= e[i]) & (ch <= ch[i]) & (cv >= cv[i])
        strictly = (e > e[i] + 1e-6) | (ch < ch[i] - 1e-6) | (cv > cv[i] + 1e-6)
        if np.any(better & strictly):
            dominated[i] = True
    frontier["nondominated"] = ~dominated
    return frontier


def label_frontier_choices(frontier: pd.DataFrame, selected_plan_id: str) -> pd.DataFrame:
    """Attach direct labels to the current, selected, and maximum-value plans."""
    out = frontier.copy()
    out["committee_label"] = ""
    current = out["plan_id"] == "current"
    out.loc[current, "committee_label"] = "Current"
    out.loc[out["plan_id"] == selected_plan_id, "committee_label"] = "Stable"

    noncurrent = out[~current]
    if not noncurrent.empty:
        best_value_id = noncurrent.loc[noncurrent["expected_nrx"].idxmax(), "plan_id"]
        if best_value_id != selected_plan_id:
            out.loc[out["plan_id"] == best_value_id, "committee_label"] = "Maximum value"
    return out


def near_optimal_plan_set(
    frontier: pd.DataFrame,
    value_loss_budgets_dollars: list[float] = NEAR_OPTIMAL_VALUE_LOSS_BUDGETS_DOLLARS,
    nrx_value_dollars: float = NRX_VALUE_DOLLARS,
    min_cvar_nrx: float = RELEASE_MIN_CVAR_NRX,
) -> pd.DataFrame:
    """Compare plan sets under explicit modeled-value loss budgets."""
    best = frontier["expected_nrx"].max()
    rows = []
    for budget in value_loss_budgets_dollars:
        loss_limit_nrx = budget / nrx_value_dollars
        threshold = best - loss_limit_nrx
        members = frontier[frontier["expected_nrx"] >= threshold]
        lowest_change = members.loc[members["plan_change_calls"].idxmin()]
        rows.append({
            "value_loss_budget_dollars": budget,
            "value_loss_limit_nrx": loss_limit_nrx,
            "value_threshold_nrx": threshold,
            "n_plans_in_set": int(len(members)),
            "min_plan_change_calls": float(members["plan_change_calls"].min()),
            "max_cvar_in_set": float(members["cvar_nrx"].max()),
            "lowest_change_plan_id": lowest_change["plan_id"],
            "lowest_change_cvar_nrx": float(lowest_change["cvar_nrx"]),
            "lowest_change_clears_risk_floor": bool(lowest_change["cvar_nrx"] >= min_cvar_nrx),
        })
    return pd.DataFrame(rows)


def select_stable_plan(
    frontier: pd.DataFrame,
    value_loss_budget_dollars: float = RELEASE_VALUE_LOSS_BUDGET_DOLLARS,
    min_cvar_nrx: float = RELEASE_MIN_CVAR_NRX,
    min_coverage: float = 1.0,
    value_band: float | None = None,
) -> dict:
    """Pick the lowest-change plan inside the approved value and risk limits."""
    best = frontier["expected_nrx"].max()
    current = frontier.loc[frontier["plan_id"] == "current", "expected_nrx"].iloc[0]
    loss_limit_nrx = value_loss_budget_dollars / NRX_VALUE_DOLLARS
    threshold = best - loss_limit_nrx
    if value_band is not None:
        threshold = (1 - value_band) * best
        loss_limit_nrx = best - threshold
        value_loss_budget_dollars = loss_limit_nrx * NRX_VALUE_DOLLARS
    eligible = frontier[
        (frontier["expected_nrx"] >= threshold)
        & (frontier["min_territory_coverage"] >= min_coverage - 1e-9)
        & (frontier["cvar_nrx"] >= min_cvar_nrx - 1e-9)
        & (frontier["plan_id"] != "current")
    ]
    if eligible.empty:
        raise ValueError("No frontier plan meets the approved value, coverage, and downside rules")
    chosen = eligible.loc[eligible["plan_change_calls"].idxmin()]
    value_loss_nrx = best - chosen["expected_nrx"]
    available_uplift_nrx = best - current
    chosen_uplift_nrx = chosen["expected_nrx"] - current
    return {
        "selected_plan_id": chosen["plan_id"],
        "expected_nrx": float(chosen["expected_nrx"]),
        "value_vs_best_pct": round((chosen["expected_nrx"] / best - 1) * 100, 2),
        "value_loss_nrx": float(value_loss_nrx),
        "value_loss_dollars": float(value_loss_nrx * NRX_VALUE_DOLLARS),
        "value_loss_budget_dollars": float(value_loss_budget_dollars),
        "available_optimization_uplift_nrx": float(available_uplift_nrx),
        "selected_optimization_uplift_nrx": float(chosen_uplift_nrx),
        "uplift_captured_pct": float(chosen_uplift_nrx / available_uplift_nrx * 100),
        "cvar_nrx": float(chosen["cvar_nrx"]),
        "plan_change_calls": float(chosen["plan_change_calls"]),
        "plan_change_pct": float(chosen["plan_change_pct"]),
        "movement_epsilon_pct": float(chosen["movement_epsilon_pct"]),
        "calls_added": float(chosen["calls_added"]),
        "calls_removed": float(chosen["calls_removed"]),
        "net_calls_added": float(chosen["net_calls_added"]),
        "calls_reassigned": float(chosen["calls_reassigned"]),
        "min_territory_coverage": float(chosen["min_territory_coverage"]),
        "min_cvar_nrx": float(min_cvar_nrx),
        "rule": (
            f"Lowest change within ${value_loss_budget_dollars:,.0f} of best modeled value, "
            f"full coverage, and CVaR of at least {min_cvar_nrx:,.0f} NRx"
        ),
    }


# ── Commit, reserve, learn ──────────────────────────────────────────────────

def two_stage_reserve_policy(
    planning: pd.DataFrame,
    gain_draws: np.ndarray,
    territories: pd.DataFrame,
    reference_calls: np.ndarray,
    movement_cap_calls: float,
    reserve_call_share: float,
    study_cost: float,
    study_foregone_nrx: float,
    signal_noise_sd: float,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Compare commit, wait, study, and perfect-information reserve policies.

    The risk-neutral reference and one segment-favorable alternative use the
    same calls by territory and the same national movement limit. Calls common
    to both plans are committed. Their disputed placements form the reserve. A
    noisy study signal updates the bootstrap weights and chooses one complete
    recourse plan. Every option preserves capacity, account caps, access
    exclusions, coverage, and the approved movement epsilon.
    """
    rng = np.random.default_rng(seed)
    n_draws = gain_draws.shape[0]
    mean_gains = gain_draws.mean(axis=0)
    reference = np.round(np.asarray(reference_calls)).astype(int)
    reference_by_t = pd.Series(reference).groupby(planning["territory_id"].to_numpy()).sum()
    fixed_territories = territories.copy()
    fixed_territories["quarterly_call_capacity"] = fixed_territories["territory_id"].map(reference_by_t)
    max_reserve = int(round(reserve_call_share * reference.sum()))

    candidates = []
    segments = planning["segment"].to_numpy()
    reference_values = scenario_plan_values(gain_draws, reference)
    for segment in sorted(planning.loc[planning["eligible_flag"], "segment"].unique()):
        segment_mask = segments == segment
        segment_signal = gain_draws[:, segment_mask, :].sum(axis=(1, 2))
        high_draws = gain_draws[segment_signal >= np.quantile(segment_signal, 0.75)]
        conditional_gains = high_draws.mean(axis=0)
        solved = _solve_national_step_model(
            planning,
            conditional_gains,
            fixed_territories,
            movement_cap_calls=movement_cap_calls,
            require_full_territory_capacity=True,
        )
        if not solved["success"]:
            continue
        alternative = np.round(solved["calls"]).astype(int)
        committed_candidate = np.minimum(reference, alternative)
        reserve_reference = int((reference - committed_candidate).sum())
        reserve_alternative = int((alternative - committed_candidate).sum())
        if reserve_reference == 0 or reserve_reference != reserve_alternative or reserve_reference > max_reserve:
            continue
        alternative_values = scenario_plan_values(gain_draws, alternative)
        perfect_information_gain = float(
            np.maximum(reference_values, alternative_values).mean()
            - max(reference_values.mean(), alternative_values.mean())
        )
        candidates.append((perfect_information_gain, segment, alternative, committed_candidate))

    if not candidates:
        raise RuntimeError("no feasible reserve alternative fits the approved reserve cap")
    _, target_segment, target_plan, committed_calls = max(candidates, key=lambda item: item[0])
    fallback_plan = reference
    reserve = int((fallback_plan - committed_calls).sum())
    reserve_by_t = pd.Series(fallback_plan - committed_calls).groupby(
        planning["territory_id"].to_numpy()
    ).sum()
    base_values = scenario_plan_values(gain_draws, committed_calls)
    target_payoff = scenario_plan_values(gain_draws, target_plan) - base_values
    fallback_payoff = reference_values - base_values
    payoff_difference = target_payoff - fallback_payoff

    study_cost_nrx = study_cost / NRX_VALUE_DOLLARS
    band = signal_noise_sd * payoff_difference.std() + 1e-9
    commit = np.empty(n_trials)
    wait = np.empty(n_trials)
    learn = np.empty(n_trials)
    perfect = np.empty(n_trials)
    signals = np.empty(n_trials)
    posterior_differences = np.empty(n_trials)
    chosen_target = np.zeros(n_trials, dtype=bool)
    realized_target_values = np.empty(n_trials)
    realized_stable_values = np.empty(n_trials)
    commit_target = float(target_payoff.mean()) >= float(fallback_payoff.mean())
    for t in range(n_trials):
        d = int(rng.integers(n_draws))
        realized_target = target_payoff[d]
        realized_fallback = fallback_payoff[d]
        realized_target_values[t] = realized_target
        realized_stable_values[t] = realized_fallback
        fixed_realized = realized_target if commit_target else realized_fallback
        commit[t] = fixed_realized
        wait[t] = fixed_realized - study_foregone_nrx
        signal = payoff_difference[d] + rng.normal(0, band)
        signals[t] = signal
        weights = np.exp(-0.5 * ((payoff_difference - signal) / band) ** 2)
        posterior_difference = float(np.sum(weights * payoff_difference) / weights.sum())
        posterior_differences[t] = posterior_difference
        chosen_target[t] = posterior_difference >= 0
        learn[t] = (
            realized_target if chosen_target[t] else realized_fallback
        ) - study_foregone_nrx
        perfect[t] = max(realized_target, realized_fallback)

    commit_now = float(commit.mean())
    wait_value = float(wait.mean())
    learn_gross = float(learn.mean())
    perfect_val = float(perfect.mean())
    net_learning_value = (learn_gross - study_cost_nrx) - commit_now

    rows = [
        {"policy": "Commit best reserve option now", "expected_nrx": round(commit_now, 3),
         "study_cost_nrx": 0.0, "net_nrx": round(commit_now, 3)},
        {"policy": "Hold reserve, no measurement", "expected_nrx": round(wait_value, 3),
         "study_cost_nrx": 0.0, "net_nrx": round(wait_value, 3)},
        {"policy": "Hold reserve, run study, reallocate", "expected_nrx": round(learn_gross, 3),
         "study_cost_nrx": round(study_cost_nrx, 3), "net_nrx": round(learn_gross - study_cost_nrx, 3)},
        {"policy": "Perfect information (upper bound)", "expected_nrx": round(perfect_val, 3),
         "study_cost_nrx": 0.0, "net_nrx": round(perfect_val, 3)},
    ]
    out = pd.DataFrame(rows)
    out.attrs["target_segment"] = target_segment
    out.attrs["reserve_calls"] = reserve
    out.attrs["committed_calls"] = int(np.round(committed_calls).sum())
    out.attrs["available_calls"] = int(np.round(committed_calls).sum()) + reserve
    out.attrs["net_learning_value_nrx"] = round(net_learning_value, 3)
    out.attrs["net_learning_value_dollars"] = round(net_learning_value * NRX_VALUE_DOLLARS, 0)
    out.attrs["study_design_assumptions"] = pd.DataFrame([{
        "target_segment": target_segment,
        "fallback_population": "risk-neutral Stable plan",
        "committed_calls": int(np.round(committed_calls).sum()),
        "reserve_calls": reserve,
        "available_flexible_calls": int(np.round(committed_calls).sum()) + reserve,
        "signal_noise_sd_nrx": round(band, 3),
        "study_cost_dollars": study_cost,
        "foregone_nrx": study_foregone_nrx,
        "trials": n_trials,
    }])
    out.attrs["study_result_distribution"] = pd.DataFrame({
        "trial": np.arange(1, n_trials + 1),
        "study_signal_nrx_difference": signals.round(4),
        "posterior_mean_nrx_difference": posterior_differences.round(4),
        "selected_recourse": np.where(chosen_target, f"{target_segment}-favorable plan", "Stable plan"),
        "realized_target_nrx": realized_target_values.round(6),
        "realized_stable_nrx": realized_stable_values.round(6),
    })
    out.attrs["recourse_allocation_table"] = pd.DataFrame([
        {
            "recourse_option": f"{target_segment}-favorable plan",
            "reserve_calls": int(np.round(target_plan - committed_calls).sum()),
            "accounts_receiving_reserve": int((target_plan > committed_calls).sum()),
            "expected_reserve_nrx": round(float(target_payoff.mean()), 2),
        },
        {
            "recourse_option": "Stable plan",
            "reserve_calls": int(np.round(fallback_plan - committed_calls).sum()),
            "accounts_receiving_reserve": int((fallback_plan > committed_calls).sum()),
            "expected_reserve_nrx": round(float(fallback_payoff.mean()), 2),
        },
    ])
    out.attrs["posterior_update_summary"] = pd.DataFrame([{
        "probability_target_selected": round(float(chosen_target.mean()), 3),
        "signal_p10": round(float(np.quantile(signals, 0.10)), 3),
        "signal_p50": round(float(np.quantile(signals, 0.50)), 3),
        "signal_p90": round(float(np.quantile(signals, 0.90)), 3),
    }])
    committed_by_t = pd.Series(committed_calls).groupby(
        planning["territory_id"].to_numpy()
    ).sum()
    out.attrs["commitment_reserve_by_territory"] = pd.DataFrame({
        "territory_id": reference_by_t.index,
        "committed_calls": committed_by_t.reindex(reference_by_t.index).to_numpy(dtype=int),
        "reserve_calls": reserve_by_t.reindex(reference_by_t.index).to_numpy(dtype=int),
        "available_calls": reference_by_t.to_numpy(dtype=int),
    })
    out.attrs["_committed_calls"] = committed_calls
    out.attrs["_target_plan"] = target_plan
    out.attrs["_fallback_plan"] = fallback_plan
    out.attrs["_commit_plan"] = target_plan if commit_target else fallback_plan
    out.attrs["_reference_calls"] = reference
    return out


def value_of_sample_information(reserve_comparison: pd.DataFrame) -> pd.DataFrame:
    """Compact net-value-of-sample-information summary from the policy comparison."""
    net_nrx = reserve_comparison.attrs.get("net_learning_value_nrx", 0.0)
    return pd.DataFrame([{
        "target_segment": reserve_comparison.attrs.get("target_segment", ""),
        "reserve_calls": reserve_comparison.attrs.get("reserve_calls", 0),
        "net_learning_value_nrx": net_nrx,
        "net_learning_value_dollars": reserve_comparison.attrs.get("net_learning_value_dollars", 0.0),
        "decision": "Run study" if net_nrx > 0 else "Commit now",
    }])


def saa_parameter_vs_outcome_demo(
    planning: pd.DataFrame,
    mean_parameter_gains: np.ndarray,
    gain_draws: np.ndarray,
) -> pd.DataFrame:
    """Show a next-call ranking flip between mean-parameter and mean-outcome gains.

    ``mean_parameter_gains`` is the step-gain matrix built from the average
    curve parameters across draws; the mean-outcome matrix averages the
    per-draw step gains. Because the step gain is nonlinear in ec50 and shape,
    the two differ (Jensen's inequality), and for two accounts competing for the
    same call the two rules disagree on which account earns it.
    """
    current = planning["current_calls"].to_numpy().astype(int)
    cap = planning["max_calls"].to_numpy()
    eligible = planning["eligible_flag"].to_numpy() & (current < cap)
    idx = np.arange(len(planning))[eligible]
    step = np.clip(current, 0, mean_parameter_gains.shape[1] - 1)
    param_next = mean_parameter_gains[np.arange(len(planning)), step]
    outcome_next = gain_draws.mean(axis=0)[np.arange(len(planning)), step]

    # Find two accounts whose order reverses between the two rules, preferring
    # a same-territory, same-segment pair so the manuscript example is a real
    # within-territory capacity trade-off.
    best_pair, best_score = None, 0.0
    for require_same_segment in [True, False]:
        for require_same_territory in [True, False]:
            for a in idx:
                for b in idx:
                    if a >= b:
                        continue
                    same_territory = planning["territory_id"].iloc[a] == planning["territory_id"].iloc[b]
                    same_segment = planning["segment"].iloc[a] == planning["segment"].iloc[b]
                    if require_same_territory and not same_territory:
                        continue
                    if require_same_segment and not same_segment:
                        continue
                    reverses = (param_next[a] > param_next[b]) != (outcome_next[a] > outcome_next[b])
                    if reverses:
                        score = min(param_next[a], param_next[b]) + min(outcome_next[a], outcome_next[b])
                        if score > best_score:
                            best_pair, best_score = (a, b), score
            if best_pair is not None:
                break
        if best_pair is not None:
            break
    pair = best_pair or (idx[0], idx[1])

    rows = []
    for i in pair:
        rows.append({
            "account_id": planning["account_id"].iloc[i],
            "territory_id": planning["territory_id"].iloc[i],
            "segment": planning["segment"].iloc[i],
            "current_calls": int(current[i]),
            "mean_parameter_next_gain": round(float(param_next[i]), 3),
            "mean_outcome_next_gain": round(float(outcome_next[i]), 3),
        })
    demo = pd.DataFrame(rows)
    demo.attrs["ranking_flips"] = best_pair is not None
    return demo


# ── Audit phase (reads hidden truth; called only after plans are fixed) ──────

def plan_delivery_audit(
    planning: pd.DataFrame,
    truth: pd.DataFrame,
    gain_draws: np.ndarray,
    fixed_plans: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Score frozen plans: planning promise versus hidden-truth delivery."""
    truth = truth.set_index("account_id").reindex(planning["account_id"]).reset_index()
    current = planning["current_calls"].to_numpy().astype(float)
    current_delivered = float(true_incremental_nrx(truth, current).sum())
    mean_gains = gain_draws.mean(axis=0)

    rows = []
    for method, calls in fixed_plans.items():
        promised = value_from_gains(mean_gains, calls)
        delivered = float(true_incremental_nrx(truth, calls).sum())
        rows.append({
            "plan": method,
            "promised_nrx": round(promised, 1),
            "delivered_nrx": round(delivered, 1),
            "promise_gap": round(promised - delivered, 1),
            "delivered_vs_current": round(delivered - current_delivered, 1),
        })
    return pd.DataFrame(rows)


def repeated_lab_selection_bias(
    n_labs: int = 500,
    n_candidate_steps: int = 40,
    n_selected_steps: int = 8,
    observations_per_step: int = 4,
    seed: int = 700,
) -> pd.DataFrame:
    """Expose selection on noise with repeated observed call-step estimates.

    Each lab creates exchangeable candidate call steps with latent incremental
    response, generates several noisy observed reads per step, and freezes the
    top estimated steps. Hidden response is used only after those selections are
    fixed. Subtracting the average estimation error from the selected-step error
    isolates winner selection from broad calibration error.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lab in range(n_labs):
        latent_gain = rng.lognormal(mean=np.log(0.35), sigma=0.25, size=n_candidate_steps)
        observed = latent_gain[:, None] + rng.normal(
            0.0, 0.24, size=(n_candidate_steps, observations_per_step)
        )
        estimated_gain = observed.mean(axis=1)
        selected = np.argsort(estimated_gain)[-n_selected_steps:]
        all_error = estimated_gain - latent_gain
        selected_error = all_error[selected]
        promised = float(estimated_gain[selected].sum())
        delivered = float(latent_gain[selected].sum())
        rows.append({
            "lab": lab + 1,
            "promised_nrx": round(promised, 3),
            "delivered_nrx": round(delivered, 3),
            "promise_gap": round(promised - delivered, 3),
            "average_estimation_error": round(float(all_error.mean()), 3),
            "selected_estimation_error": round(float(selected_error.mean()), 3),
            "selection_effect": round(float(selected_error.mean() - all_error.mean()), 3),
        })
    audit = pd.DataFrame(rows)
    audit.attrs["mean_promise_gap"] = round(float(audit["promise_gap"].mean()), 3)
    audit.attrs["mean_selection_effect"] = round(float(audit["selection_effect"].mean()), 3)
    return audit


def roventra_selection_noise_audit(
    planning: pd.DataFrame,
    truth: pd.DataFrame,
    gain_draws: np.ndarray,
    selected_calls: np.ndarray,
) -> pd.DataFrame:
    """Compare fitted-vs-true gain error for all possible calls and selected calls."""
    truth = truth.set_index("account_id").reindex(planning["account_id"]).reset_index()
    max_calls = gain_draws.shape[2]
    levels = np.vstack([
        true_incremental_nrx(truth, np.full(len(planning), c))
        for c in range(max_calls + 1)
    ]).T
    true_gains = np.diff(levels, axis=1)
    estimated_gains = gain_draws.mean(axis=0)
    error = estimated_gains - true_gains

    steps = np.arange(1, max_calls + 1)[None, :]
    feasible = steps <= planning["max_calls"].to_numpy()[:, None]
    selected = steps <= np.round(np.asarray(selected_calls)).astype(int)[:, None]
    rows = []
    for label, mask in [
        ("All feasible calls", feasible),
        ("Selected planned calls", selected),
    ]:
        rows.append({
            "call_set": label,
            "possible_calls": int(mask.sum()),
            "mean_estimation_error_nrx": round(float(error[mask].mean()), 3),
            "promised_nrx": round(float(estimated_gains[mask].sum()), 1),
            "hidden_truth_nrx": round(float(true_gains[mask].sum()), 1),
        })
    out = pd.DataFrame(rows)
    effect = (
        out.loc[out["call_set"] == "Selected planned calls", "mean_estimation_error_nrx"].iloc[0]
        - out.loc[out["call_set"] == "All feasible calls", "mean_estimation_error_nrx"].iloc[0]
    )
    out.attrs["selection_effect_nrx_per_call"] = round(float(effect), 3)
    return out


# ── Final packages ──────────────────────────────────────────────────────────

def account_release_package(
    planning: pd.DataFrame,
    committed_calls: np.ndarray,
    target_recourse_calls: np.ndarray,
    fallback_recourse_calls: np.ndarray,
) -> pd.DataFrame:
    """One row per account with committed calls and both reserve contingencies."""
    committed = np.round(np.asarray(committed_calls)).astype(int)
    target = np.round(np.asarray(target_recourse_calls)).astype(int)
    fallback = np.round(np.asarray(fallback_recourse_calls)).astype(int)
    package = planning[[
        "account_id", "territory_id", "segment", "access_state", "protected",
        "eligible_flag", "current_calls",
    ]].copy()
    package["committed_calls"] = committed
    package["target_recourse_calls"] = target
    package["fallback_recourse_calls"] = fallback
    package["max_released_calls"] = np.maximum(target, fallback)
    package["priority_rank"] = (
        package.groupby("territory_id")["max_released_calls"].rank(ascending=False, method="first").astype(int)
    )
    contingent = target != fallback
    package["commit_status"] = np.where(contingent, "Reserve decision pending", "Commit now")

    def reason(row) -> str:
        if row["protected"]:
            return "Compliance hold, zero calls"
        if row["access_state"] == "Closed":
            return "Closed access, zero calls"
        if row["max_released_calls"] == 0:
            return "Below capacity threshold this cycle"
        if row["segment"] == "Anchor":
            return "High opportunity, low incremental response"
        return "Incremental-response priority"

    package["reason_code"] = package.apply(reason, axis=1)
    package["measurement_hook"] = "engagement_outcome_log"
    package["refresh_date"] = PLAN_REFRESH_DATE
    return package[[
        "account_id", "territory_id", "access_state", "current_calls", "committed_calls",
        "target_recourse_calls", "fallback_recourse_calls", "priority_rank", "reason_code",
        "commit_status", "measurement_hook", "refresh_date",
    ]].sort_values(["territory_id", "priority_rank"]).reset_index(drop=True)


def quarterly_resource_package(
    channel_move: pd.DataFrame,
    selection: dict,
    gain_draws: np.ndarray,
    headcount: pd.DataFrame,
    vosi: pd.DataFrame,
    reserve_comparison: pd.DataFrame,
    reference_calls: np.ndarray,
    planning: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-resource release scorecard built entirely from planning information."""
    current = planning["current_calls"].to_numpy().astype(float)
    reference = np.round(np.asarray(reference_calls))
    candidate_commitment = np.round(np.asarray(reserve_comparison.attrs["_committed_calls"]))
    commit_plan = np.round(np.asarray(reserve_comparison.attrs["_commit_plan"]))
    study_runs = vosi.iloc[0]["decision"] == "Run study"
    committed = candidate_commitment if study_runs else commit_plan
    reported_reserve = int(vosi.iloc[0]["reserve_calls"]) if study_runs else 0
    accounts_changed = int((committed != current).sum())
    field_risk = incremental_plan_risk_summary(gain_draws, commit_plan, current)
    best_rep = headcount.iloc[0]
    vosi_row = vosi.iloc[0]

    channel_nrx = float(channel_move["quarterly_nrx_change"].sum())
    channel_value = channel_nrx * NRX_VALUE_DOLLARS
    channel_p10 = float(channel_move.attrs.get("portfolio_p10_quarterly_nrx_change", channel_nrx))
    increases = channel_move.loc[channel_move["weekly_spend_change"] > 0, "channel"].tolist()
    decreases = channel_move.loc[channel_move["weekly_spend_change"] < 0, "channel"].tolist()

    def value_range(incremental_nrx: float) -> str:
        values = sorted([
            incremental_nrx * NRX_VALUE_DOLLARS_LOW,
            incremental_nrx * NRX_VALUE_DOLLARS_HIGH,
        ])
        return f"${values[0]:,.0f} to ${values[1]:,.0f}"

    rows = [
        {
            "decision": "Channel budget",
            "resource_unit": "weekly dollars by channel",
            "current_commitment": f"${channel_move['current_weekly_spend'].sum():,.0f} per week",
            "recommended_commitment": f"${channel_move['recommended_weekly_spend'].sum():,.0f} per week",
            "recommended_action": f"Increase {', '.join(increases)}; trim {', '.join(decreases)}",
            "expected_incremental_nrx": round(channel_nrx, 1),
            "expected_quarterly_value_dollars": round(channel_value, 0),
            "expected_value_range": value_range(channel_nrx),
            "reserve": "$0 in this release",
            "plan_change_required": f"${channel_move['weekly_spend_change'].abs().sum():,.0f} weekly gross movement",
            "downside_metric": "10th-percentile shortfall",
            "downside_nrx": round(channel_nrx - channel_p10, 1),
            "cvar_shortfall_nrx": np.nan,
            "coverage_status": "Evidence permissions applied to every channel",
            "constraint_cost": "Bounded by channel evidence guardrails",
            "measurement_action": "New anchors for email/digital/paid; refresh field",
            "release_status": "Release bounded move",
            "pacing": f"Weekly through {PLANNING_QUARTER}",
            "refresh_date": PLAN_REFRESH_DATE,
        },
        {
            "decision": "Field call plan",
            "resource_unit": "whole calls by account",
            "current_commitment": f"{int(current.sum())} calls",
            "recommended_commitment": f"{int(committed.sum())} calls now",
            "recommended_action": f"Release Stable plan: {accounts_changed} account changes",
            "expected_incremental_nrx": round(field_risk["expected_incremental_nrx"], 1),
            "expected_quarterly_value_dollars": round(
                field_risk["expected_incremental_nrx"] * NRX_VALUE_DOLLARS, 0
            ),
            "expected_value_range": value_range(field_risk["expected_incremental_nrx"]),
            "reserve": f"{reported_reserve} calls",
            "plan_change_required": f"{int(selection['plan_change_calls'])} account-call changes",
            "downside_metric": "90% CVaR shortfall",
            "downside_nrx": round(
                field_risk["expected_incremental_nrx"] - field_risk["cvar_incremental_nrx"], 1
            ),
            "cvar_shortfall_nrx": round(
                field_risk["expected_incremental_nrx"] - field_risk["cvar_incremental_nrx"], 1
            ),
            "coverage_status": f"{selection['min_territory_coverage']:.0%} minimum territory coverage",
            "constraint_cost": "Coverage and account-call change cap priced in trade-off table",
            "measurement_action": (
                "Engagement outcome study" if study_runs else "Routine engagement outcome log"
            ),
            "release_status": (
                "Release commitment; hold reserve" if study_runs else "Release full call plan"
            ),
            "pacing": (
                f"Commit now; reserve after {MEASUREMENT_READ_DATE}"
                if study_runs else f"Release all calls in {PLANNING_QUARTER}"
            ),
            "refresh_date": PLAN_REFRESH_DATE,
        },
        {
            "decision": "Headcount",
            "resource_unit": "representative-quarter",
            "current_commitment": f"{int(headcount['current_reps'].sum())} representatives",
            "recommended_commitment": "No added representative-quarter",
            "recommended_action": f"{best_rep['recommendation']} in {best_rep['territory_id']}",
            "expected_incremental_nrx": 0.0,
            "expected_quarterly_value_dollars": 0.0,
            "expected_value_range": "$0 for the hold decision",
            "reserve": "0 representative-quarters",
            "plan_change_required": "No roster change",
            "downside_metric": "Not applicable",
            "downside_nrx": np.nan,
            "cvar_shortfall_nrx": np.nan,
            "coverage_status": "Current capacity retained",
            "constraint_cost": (
                f"Best addition nets ${best_rep['net_value_per_added_rep']:,.0f} after "
                f"${best_rep['loaded_cost_per_rep_quarter']:,.0f} loaded cost"
            ),
            "measurement_action": "Capacity delivery and incremental NRx",
            "release_status": "Hold headcount",
            "pacing": f"No change in {PLANNING_QUARTER}",
            "refresh_date": PLAN_REFRESH_DATE,
        },
        {
            "decision": "Learning action",
            "resource_unit": "one response study",
            "current_commitment": "No study",
            "recommended_commitment": (
                f"Study read by {MEASUREMENT_READ_DATE}" if study_runs else "No study this quarter"
            ),
            "recommended_action": (
                f"Run {vosi_row['target_segment']} segment study"
                if study_runs else f"Defer {vosi_row['target_segment']} segment study"
            ),
            "expected_incremental_nrx": float(vosi_row["net_learning_value_nrx"]),
            "expected_quarterly_value_dollars": float(vosi_row["net_learning_value_dollars"]),
            "expected_value_range": value_range(float(vosi_row["net_learning_value_nrx"])),
            "reserve": f"{reported_reserve} calls",
            "plan_change_required": "One study and one reserve release decision",
            "downside_metric": "Perfect-information upper bound",
            "downside_nrx": round(
                float(reserve_comparison.loc[
                    reserve_comparison["policy"].str.startswith("Perfect"), "expected_nrx"
                ].iloc[0])
                - float(reserve_comparison.loc[
                    reserve_comparison["policy"].str.startswith("Commit"), "expected_nrx"
                ].iloc[0]), 2
            ),
            "cvar_shortfall_nrx": np.nan,
            "coverage_status": "Reserve options preserve field rules",
            "constraint_cost": "Study cost and one quarter of foregone reserve value",
            "measurement_action": (
                f"{vosi_row['target_segment']} segment response study"
                if study_runs else "Continue routine outcome logging"
            ),
            "release_status": "Run study" if study_runs else "Defer study",
            "pacing": (
                f"Read by {MEASUREMENT_READ_DATE}" if study_runs else f"Reassess {PLAN_REFRESH_DATE}"
            ),
            "refresh_date": PLAN_REFRESH_DATE,
        },
    ]
    return pd.DataFrame(rows)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
