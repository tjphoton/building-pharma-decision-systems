"""Two-phase orchestration for the resource-allocation and optimization chapter.

The planning phase reads observed data, the fitted response uncertainty, the
business rules, and the upstream channel-budget record. It builds every plan
and makes the final release selection. Only then does the audit phase run: it
takes the frozen plan call vectors and the hidden truth table and scores
promise against delivery. Audit results cannot change the released plan, since
selection has already happened.

``run_analysis()`` returns the single results dictionary the manuscript, the
figures, and the notebooks all read from the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import allocation as al  # noqa: E402
from allocation_config import (  # noqa: E402
    CHURN_CAP_SHARE,
    CVAR_ALPHA,
    NRX_VALUE_DOLLARS,
    NRX_VALUE_DOLLARS_HIGH,
    NRX_VALUE_DOLLARS_LOW,
    N_LEARNING_TRIALS,
    RESERVE_CALL_SHARE,
    SEED_LEARNING,
    SEGMENT_ORDER,
    STUDY_COST_DOLLARS,
    STUDY_FOREGONE_NRX,
    STUDY_SIGNAL_NOISE_SD,
)
from generate_allocation_data import run_generation  # noqa: E402
from response_uncertainty import (  # noqa: E402
    block_bootstrap_response_draws,
    fit_segment_response,
    params_to_step_gains,
    point_estimate_step_gains,
    response_draws_to_step_gains,
    summarize_step_gain_draws,
    validate_draw_quality,
)


def resource_decision_map() -> pd.DataFrame:
    """Which record each method reads, the decision it supports, and its boundary."""
    rows = [
        ("Unified budget recommendation", "Channel response and evidence permissions",
         "Bounded channel budget move", "Does not assign account calls"),
        ("Forecast scenario", "Approved quarterly demand and value assumptions",
         "Resource envelope and value translation", "Does not estimate channel incrementality"),
        ("Fitted field response", "Observed account-period calls and NRx",
         "Expected marginal call gains", "Does not override access or compliance rules"),
        ("Step-gain MILP", "Marginal gains, capacity, and business rules",
         "Integer account call plan", "Does not validate response estimates"),
        ("CVaR model", "Scenario plan value across draws",
         "Downside-protected call plan", "Does not set the accepted risk limit"),
        ("Epsilon frontier", "Feasible plans from repeated solves",
         "Committee choice among value, risk, change", "Does not relax hard compliance rules"),
        ("Learning simulation", "Proposed study, reserve, and outcome model",
         "Commit, reserve, and test recommendation", "Does not reveal hidden truth in planning"),
    ]
    return pd.DataFrame(rows, columns=["input_or_method", "record_read", "decision_supported", "decision_boundary"])


def methods_field_guide() -> pd.DataFrame:
    """Compact method map for problems outside the main field-call build."""
    rows = [
        ("Marginal analysis", "Separable allocation", "Full", "Needs incremental response"),
        ("Greedy", "Declining gains and simple capacity", "Full", "Thresholds break exactness"),
        ("SLSQP", "Smooth continuous response", "Full", "Can be local and fractional"),
        ("LP", "Piecewise-linear continuous plan", "Full", "May return fractional calls"),
        ("MILP", "Integer calls and fixed choices", "Full", "Discrete sensitivity needs re-solves"),
        ("SAA", "Expected value across draws", "Full", "Finite-draw error remains"),
        ("CVaR", "Lower-tail protection", "Full", "Risk limit needs approval"),
        ("Minimax regret", "Worst scenario decision loss", "Extension", "Can be overly cautious"),
        ("Robust optimization", "Controlled uncertainty set", "Extension", "Set width controls conservatism"),
        ("Epsilon constraint", "Multi-objective frontier", "Full", "Limits need business units"),
        ("Two-stage model", "Commitment followed by recourse", "Full", "Needs a valid information update"),
        ("Genetic or local search", "Territory geometry", "Exercise", "No global optimality certificate"),
        ("Reinforcement learning", "Repeated next-action policy", "Outside scope", "Needs online evaluation"),
    ]
    return pd.DataFrame(rows, columns=["method", "problem_fit", "build_depth", "boundary"])


def _observed_history_summary(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.groupby("segment", as_index=False)
        .agg(account_periods=("account_id", "size"), accounts=("account_id", "nunique"),
             call_min=("calls", "min"), call_max=("calls", "max"),
             mean_observed_nrx=("observed_nrx", "mean"))
        .round({"mean_observed_nrx": 2})
    )


def _planning_data_boundary(planning: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"artifact": "observed_field_history", "rows": len(history),
         "true_columns": sum(c.startswith("true_") for c in history), "allowed_use": "fit and diagnose"},
        {"artifact": "account_planning_inputs", "rows": len(planning),
         "true_columns": sum(c.startswith("true_") for c in planning), "allowed_use": "all planning models"},
        {"artifact": "account_response_truth", "rows": len(planning),
         "true_columns": "audit only", "allowed_use": "score frozen plans"},
    ])


def _bootstrap_diagnostics(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(SEGMENT_ORDER):
        rows.append({
            "segment": seg,
            "scale_p10": round(float(np.quantile(draws[:, i, 0], 0.10)), 4),
            "scale_p90": round(float(np.quantile(draws[:, i, 0], 0.90)), 4),
            "ec50_p10": round(float(np.quantile(draws[:, i, 1], 0.10)), 2),
            "ec50_p90": round(float(np.quantile(draws[:, i, 1], 0.90)), 2),
            "shape_p10": round(float(np.quantile(draws[:, i, 2], 0.10)), 3),
            "shape_p90": round(float(np.quantile(draws[:, i, 2], 0.90)), 3),
        })
    return pd.DataFrame(rows)


def _plan_table(planning: pd.DataFrame, calls: np.ndarray, name: str) -> pd.DataFrame:
    out = planning[["account_id", "territory_id", "segment", "current_calls"]].copy()
    out[name] = np.round(np.asarray(calls)).astype(int)
    return out


def _plan_uncertainty_scorecard(
    planning: pd.DataFrame, gain_draws: np.ndarray, plans: dict[str, np.ndarray], alpha: float
) -> pd.DataFrame:
    current = planning["current_calls"].to_numpy(dtype=float)
    rows = []
    for method, calls in plans.items():
        risk = al.plan_risk_summary(gain_draws, calls, alpha, current)
        rows.append({
            "plan": method,
            "expected_nrx": round(risk["expected_nrx"], 1),
            "p10_nrx": round(risk["p10_nrx"], 1),
            "cvar_nrx": round(risk["cvar_nrx"], 1),
            "cvar_shortfall_nrx": round(risk["expected_nrx"] - risk["cvar_nrx"], 1),
            "std_nrx": round(risk["std_nrx"], 1),
            "prob_under_current": round(risk["prob_under_benchmark"], 3),
            "plan_change_calls": round(al.plan_change_calls(planning, calls), 0),
            "min_territory_coverage": round(al.coverage_min_by_territory(planning, calls), 3),
            "accounts_changed": int((np.round(calls) != planning["current_calls"].to_numpy()).sum()),
        })
    return pd.DataFrame(rows)


def _constraint_tradeoff_table(
    field_tradeoffs: pd.DataFrame, channel_tradeoff: dict, vosi: pd.DataFrame,
    reserve_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the Section 18.3 trade-off table across resources."""
    rows = []
    for row in field_tradeoffs.itertuples():
        if row.rule == "Release plan (all rules)":
            continue
        rows.append({
            "rule_or_resource": row.rule,
            "current_level": {
                "Territory capacity": "Current territory roster",
                "Account-call change cap": "20% of current national calls",
                "Minimum coverage": "1 call per eligible account",
            }.get(row.rule, ""),
            "tested_change": row.tested_change,
            "expected_nrx_change": row.expected_nrx_change,
            "cvar_nrx_change": row.cvar_nrx_change,
            "plan_change_effect_calls": row.plan_change_effect_calls,
            "interpretation": {
                "Territory capacity": "Value of added capacity",
                "Account-call change cap": "Cost of implementation stability",
                "Minimum coverage": "Cost of the service condition",
            }.get(row.rule, ""),
        })
    rows.append({
        "rule_or_resource": "Channel movement cap",
        "current_level": "Channel-specific evidence bounds",
        "tested_change": channel_tradeoff["tested_change"],
        "expected_nrx_change": round(channel_tradeoff["expected_nrx_change"], 1),
        "cvar_nrx_change": np.nan,
        "plan_change_effect_calls": np.nan,
        "interpretation": "Value blocked by current channel evidence",
    })
    vosi_row = vosi.iloc[0]
    rows.append({
        "rule_or_resource": "Flexible reserve",
        "current_level": "Commit all flexible calls now",
        "tested_change": f"Hold {int(vosi_row['reserve_calls'])} calls and learn",
        "expected_nrx_change": float(vosi_row["net_learning_value_nrx"]),
        "cvar_nrx_change": np.nan,
        "plan_change_effect_calls": np.nan,
        "interpretation": "Value of waiting for a measurement read",
    })
    return pd.DataFrame(rows)


def _plan_change_components(
    planning: pd.DataFrame,
    selected_calls: np.ndarray,
    channel_move: pd.DataFrame,
    reserve_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Concrete components behind the selected plan's implementation change."""
    current = planning["current_calls"].to_numpy(dtype=float)
    selected = np.round(np.asarray(selected_calls))
    changed = selected != current
    return pd.DataFrame([
        {"component": "Accounts with changed calls", "value": int(changed.sum()), "unit": "accounts"},
        {"component": "Absolute call movement", "value": int(np.abs(selected - current).sum()), "unit": "calls"},
        {"component": "Territories affected", "value": int(planning.loc[changed, "territory_id"].nunique()), "unit": "territories"},
        {"component": "Gross channel budget movement", "value": round(float(channel_move["weekly_spend_change"].abs().sum()), 1), "unit": "weekly dollars"},
        {"component": "Flexible call reserve", "value": int(reserve_comparison.attrs["reserve_calls"]), "unit": "calls"},
    ])


def value_sensitivity_band() -> pd.DataFrame:
    return pd.DataFrame([
        {"assumption": "value_per_incremental_nrx_low", "dollars": NRX_VALUE_DOLLARS_LOW},
        {"assumption": "value_per_incremental_nrx_base", "dollars": NRX_VALUE_DOLLARS},
        {"assumption": "value_per_incremental_nrx_high", "dollars": NRX_VALUE_DOLLARS_HIGH},
    ])


def run_analysis(repo_root: Path | None = None) -> dict:
    if repo_root is None:
        repo_root = _REPO_ROOT
    mmm_dir = repo_root / "ch13_mmm" / "assets" / "generated_outputs"

    # ── Planning phase ──────────────────────────────────────────────────────
    generated = run_generation()
    planning = generated["planning"]
    territories = generated["territories"]
    history = generated["observed_history"]

    fit = fit_segment_response(history)
    segment_params = {r.segment: (r.scale, r.ec50, r.shape) for r in fit.itertuples()}
    point_gains = point_estimate_step_gains(planning, fit)
    param_draws = block_bootstrap_response_draws(history)
    gain_draws = response_draws_to_step_gains(planning, param_draws)
    step_gain_summary = summarize_step_gain_draws(planning, gain_draws)
    draw_quality = validate_draw_quality(planning, fit, param_draws, gain_draws)
    bootstrap_diag = _bootstrap_diagnostics(param_draws)

    decisions = resource_decision_map()
    method_guide = methods_field_guide()
    history_summary = _observed_history_summary(history)
    data_boundary = _planning_data_boundary(planning, history)
    channel_summary = al.channel_response_summary(mmm_dir)
    channel_move = al.channel_budget_move(mmm_dir)

    marginal_table = al.account_marginal_return_table(planning, point_gains)
    toy_split = al.toy_split_table(total_calls=12)
    toy_greedy = al.toy_greedy_path(total_calls=12)

    current = planning["current_calls"].to_numpy().astype(float)
    greedy, t_greedy = al.timed(al.greedy_call_plan, planning, point_gains, territories)
    slsqp_result, t_slsqp = al.timed(
        al.slsqp_call_plan, planning, segment_params, territories, True
    )
    slsqp, slsqp_diag = slsqp_result
    # A half-call movement budget exposes the fractional LP boundary. The
    # release MILP retains the approved 20% integer-call limit.
    lp_diagnostic_cap = CHURN_CAP_SHARE * float(current.sum()) + 0.5
    lp = al.lp_relaxation_call_plan(
        planning, point_gains, territories, movement_cap_share=None,
        movement_cap_calls=lp_diagnostic_cap,
    )
    milp_point, milp_diag = al.milp_call_plan(planning, point_gains, territories)
    expected_calls, _ = al.expected_gain_milp_plan(planning, gain_draws, territories)
    cvar_calls, _ = al.cvar_milp_plan(planning, gain_draws, territories)

    runtimes = {
        "Current plan (incumbent)": 0.0,
        "Greedy": t_greedy,
        "SLSQP (smooth reference)": t_slsqp,
        "LP relaxation": float("nan"),
        "MILP (point estimate)": float("nan"),
        "Expected-value MILP": float("nan"),
    }
    exec_plans = {
        "Current plan (incumbent)": current,
        "Greedy": greedy,
        "SLSQP (smooth reference)": slsqp,
        "LP relaxation": lp["calls"],
        "MILP (point estimate)": milp_point,
        "Expected-value MILP": expected_calls,
    }
    predicted_values = {
        "SLSQP (smooth reference)": float(al.continuous_account_response(planning, segment_params, slsqp).sum()),
        "LP relaxation": float(lp["objective_nrx"]),
    }
    call_scorecard = al.call_plan_scorecard(
        planning,
        point_gains,
        territories,
        exec_plans,
        runtimes,
        predicted_values,
        {"LP relaxation": lp["fractional_step_variables"] == 0},
    )
    milp_dimensions = pd.DataFrame([
        {"model": "LP diagnostic", **lp["dimensions"]},
        {"model": "Release MILP", **{
            key: milp_diag["national"][key]
            for key in ["step_variables", "binary_variables", "continuous_variables",
                        "constraint_rows", "scenario_count"]
        }},
    ])
    milp_feasibility = pd.DataFrame([
        {"territory_id": t_id, "feasible": row["feasible"],
         "optimality_gap": row["optimality_gap"], "n_accounts": row["n_accounts"]}
        for t_id, row in milp_diag.items() if t_id != "national"
    ])

    mean_params = {seg: tuple(param_draws[:, i].mean(axis=0)) for i, seg in enumerate(SEGMENT_ORDER)}
    mean_parameter_gains = params_to_step_gains(planning, mean_params)
    saa_demo = al.saa_parameter_vs_outcome_demo(planning, mean_parameter_gains, gain_draws)

    uncertainty_plans = {
        "Point-estimate MILP": milp_point,
        "Expected-value MILP": expected_calls,
        "CVaR MILP": cvar_calls,
    }
    uncertainty_scorecard = _plan_uncertainty_scorecard(planning, gain_draws, uncertainty_plans, CVAR_ALPHA)

    headcount = al.headcount_business_case(planning, point_gains, territories)
    continuous_prices = al.continuous_constraint_prices(slsqp_diag, channel_summary)
    field_tradeoffs = al.discrete_constraint_tradeoffs(
        planning, gain_draws, territories,
        capacity_territory=str(headcount.iloc[0]["territory_id"]),
    )
    channel_tradeoff = al.channel_movement_cap_tradeoff(mmm_dir)

    frontier_candidates, frontier_plans = al.epsilon_frontier(planning, gain_draws, territories)
    frontier_grid = frontier_candidates.attrs["solve_grid"]
    frontier_candidates = al.filter_nondominated_plans(frontier_candidates)
    frontier = frontier_candidates[frontier_candidates["nondominated"]].copy().reset_index(drop=True)
    near_optimal = al.near_optimal_plan_set(frontier)
    selection = al.select_stable_plan(frontier)
    frontier = al.label_frontier_choices(frontier, selection["selected_plan_id"])
    selection["committee_label"] = "Stable"
    reference_calls = frontier_plans[selection["selected_plan_id"]]

    reserve_comparison = al.two_stage_reserve_policy(
        planning, gain_draws, territories, reference_calls,
        selection["movement_epsilon_pct"] / 100 * float(current.sum()),
        RESERVE_CALL_SHARE, STUDY_COST_DOLLARS,
        STUDY_FOREGONE_NRX, STUDY_SIGNAL_NOISE_SD, N_LEARNING_TRIALS, SEED_LEARNING,
    )
    vosi = al.value_of_sample_information(reserve_comparison)
    tradeoff_table = _constraint_tradeoff_table(
        field_tradeoffs, channel_tradeoff, vosi, reserve_comparison
    )
    change_components = _plan_change_components(
        planning, reference_calls, channel_move, reserve_comparison
    )

    # The release package is complete before the audit receives hidden truth.
    study_runs = vosi.iloc[0]["decision"] == "Run study"
    if study_runs:
        released_commitment = reserve_comparison.attrs["_committed_calls"]
        released_target = reserve_comparison.attrs["_target_plan"]
        released_fallback = reserve_comparison.attrs["_fallback_plan"]
    else:
        released_commitment = reserve_comparison.attrs["_commit_plan"]
        released_target = released_commitment
        released_fallback = released_commitment
    release_package = al.account_release_package(
        planning, released_commitment, released_target, released_fallback
    )
    quarterly_package = al.quarterly_resource_package(
        channel_move, selection, gain_draws, headcount, vosi, reserve_comparison,
        reference_calls, planning,
    )

    # ── Audit phase (frozen plans + hidden truth) ───────────────────────────
    truth = generated["truth"]
    frozen_plans = {
        "Current plan (incumbent)": current,
        "Point-estimate MILP": milp_point,
        "Expected-value MILP": expected_calls,
        "CVaR MILP": cvar_calls,
        "Selected Stable reference": reference_calls,
        "Committed before reserve read": reserve_comparison.attrs["_committed_calls"],
    }
    frozen_plan_registry = pd.DataFrame({
        "frozen_plan_id": list(frozen_plans),
        "frozen_before_audit": True,
    })
    delivery_audit = al.plan_delivery_audit(planning, truth, gain_draws, frozen_plans)
    repeated_lab = al.repeated_lab_selection_bias()
    roventra_selection_noise = al.roventra_selection_noise_audit(
        planning, truth, gain_draws, expected_calls
    )

    return {
        "territories": territories,
        "planning": planning,
        "account_planning_inputs": planning,
        "observed_field_history": history,
        "account_response_truth": truth,
        "resource_decision_map": decisions,
        "methods_field_guide": method_guide,
        "observed_history_summary": history_summary,
        "planning_data_boundary": data_boundary,
        "fitted_response_summary": fit,
        "bootstrap_diagnostics": bootstrap_diag,
        "draw_quality_checks": draw_quality,
        "call_step_gain_summary": step_gain_summary,
        "channel_response_summary": channel_summary,
        "channel_budget_move": channel_move,
        "account_marginal_return_table": marginal_table,
        "toy_split_table": toy_split,
        "toy_greedy_path": toy_greedy,
        "call_plan_scorecard": call_scorecard,
        "slsqp_diagnostics": slsqp_diag,
        "lp_relaxation_plan": _plan_table(planning, lp["calls"], "lp_calls"),
        "lp_fractional_step_variables": lp["fractional_step_variables"],
        "lp_movement_cap_calls": lp["movement_cap_calls"],
        "milp_model_dimensions": milp_dimensions,
        "milp_territory_feasibility": milp_feasibility,
        "saa_parameter_demo": saa_demo,
        "expected_value_milp_plan": _plan_table(planning, expected_calls, "expected_calls"),
        "cvar_milp_plan": _plan_table(planning, cvar_calls, "cvar_calls"),
        "plan_uncertainty_scorecard": uncertainty_scorecard,
        "continuous_constraint_prices": continuous_prices,
        "constraint_tradeoff_table": tradeoff_table,
        "headcount_business_case": headcount,
        "frontier_solutions": frontier,
        "frontier_grid": frontier_grid,
        "frontier_candidates": frontier_candidates,
        "near_optimal_plans": near_optimal,
        "selected_plan": pd.DataFrame([selection]),
        "plan_change_components": change_components,
        "reserve_learning_comparison": reserve_comparison,
        "study_design_assumptions": reserve_comparison.attrs["study_design_assumptions"],
        "study_result_distribution": reserve_comparison.attrs["study_result_distribution"],
        "posterior_update_summary": reserve_comparison.attrs["posterior_update_summary"],
        "recourse_allocation_table": reserve_comparison.attrs["recourse_allocation_table"],
        "commitment_reserve_by_territory": reserve_comparison.attrs["commitment_reserve_by_territory"],
        "value_of_sample_information": vosi,
        "plan_delivery_audit": delivery_audit,
        "frozen_plan_registry": frozen_plan_registry,
        "repeated_lab_selection_bias": repeated_lab,
        "roventra_selection_noise_audit": roventra_selection_noise,
        "account_release_package": release_package,
        "quarterly_resource_package": quarterly_package,
        "value_sensitivity_band": value_sensitivity_band(),
        "manifest": generated["manifest"],
        "_gain_draws": gain_draws,
        "_point_gains": point_gains,
        "_param_draws": param_draws,
        "_segment_params": segment_params,
        "_frontier_plans": frontier_plans,
    }


_NPZ_KEYS = {
    "planning", "_gain_draws", "_point_gains", "_param_draws",
    "_segment_params", "_frontier_plans",
}


def write_outputs(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.json", "*.npz"):
        for stale in output_dir.glob(pattern):
            stale.unlink()
    for name, value in results.items():
        if name in _NPZ_KEYS or name == "manifest":
            continue
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{name}.csv", index=False)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            pd.DataFrame([{name: value}]).to_csv(output_dir / f"{name}.csv", index=False)
    np.savez_compressed(output_dir / "call_step_gain_draws.npz", gain_draws=results["_gain_draws"])
    import json
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(results["manifest"], indent=2) + "\n"
    )


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "assets" / "generated_outputs"
    results = run_analysis()
    write_outputs(results, out_dir)
    print(pd.Series(results["manifest"]).to_string())
    print(f"Wrote resource-allocation chapter outputs to {out_dir}")
