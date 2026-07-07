"""Build and execute the Chapter 14 companion notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "ch14_forecasting"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


SETUP_BASE = """
# ruff: noqa: E402
from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
CHAPTER_DIR = ROOT / "ch14_forecasting"
SCRIPT_DIR = CHAPTER_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_analysis import run_analysis  # noqa: E402
"""


SETUP_EXERCISES = SETUP_BASE + """
from forecasting import (
    accuracy_scorecard,
    fit_bass,
    fit_erosion,
    fit_ets,
    fit_sarima,
    naive_forecast,
    rolling_origin_backtest,
    seasonal_naive,
)  # noqa: E402

results = run_analysis()
national = results["national_series"]
observed = results["observed_series"]
print(f"Observed weeks: {len(observed)}")
print(f"Full lifecycle weeks: {len(national)}")
"""


def figure(index: int, name: str, caption: str, ext: str = "svg"):
    text = f"Figure 14.{index}. {caption}"
    return md(f"![{text}](assets/figures/figure_14_{index}_{name}.{ext})\n\n*{text}*")


def listing(caption: str):
    return md(f"**Listing**: {caption}")


def walkthrough():
    cells = [
        md(
            """
# Chapter 14: Forecasting from Launch to Loss of Exclusivity

This notebook executes the Chapter 14 forecasting chain: pre-launch business case, in-market scorecard, demand-supply reconciliation, loss of exclusivity, and consensus scenarios.
"""
        ),
        md("## 14.1 Forecasting Decisions"),
        listing("Load the lifecycle series"),
        code(
            """
from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "ch14_forecasting").exists():
    ROOT = ROOT.parent
SCRIPT_DIR = ROOT / "ch14_forecasting" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_analysis import run_analysis

results = run_analysis()

national = results["national_series"]
observed = results["observed_series"]
print(observed[["week_start", "nbrx", "trx"]].tail().round(1))
"""
        ),
        figure(
            1,
            "lifecycle_series",
            "Roventra lifecycle series with launch, observed window, peak, and generic entry marked. Synthetic data.",
        ),
        md("## 14.2 Sizing the Launch"),
        listing("The pre-launch funnel ceiling"),
        code(
            """
from forecasting import patient_based_forecast

business_case = patient_based_forecast()
print(pd.Series(business_case).to_string())
"""
        ),
        listing("Reconstructing on-therapy stock from observed NBRx"),
        code(
            """
from forecasting import persistence_to_trx

reconstructed = persistence_to_trx(observed["nbrx"].to_numpy())
reconstructed["cumulative_nbrx"] = reconstructed["nbrx"].cumsum()
reconstructed = reconstructed[["nbrx", "cumulative_nbrx", "on_therapy_stock"]]
print(reconstructed.tail().round(1))
"""
        ),
        figure(
            2,
            "funnel_timeline",
            "Cumulative new starts and on-therapy stock over the observed window.",
        ),
        figure(
            3,
            "bass_toy_shapes",
            "Prescriber innovation, imitation, and blended adoption shapes.",
        ),
        figure(
            4,
            "analog_shapes",
            "Comparable A and Comparable B normalized adoption shapes over 60 months, before selection.",
        ),
        figure(
            5,
            "analog_selection_zoom",
            "Roventra's normalized uptake overlaid against both analog curves at 26 weeks (left) and 52 weeks (right).",
        ),
        listing("Fitting Bass diffusion to the observed launch data"),
        code(
            """
from forecasting import fit_bass

weeks_since_launch = (observed["week_start"] - national["week_start"].iloc[0]).dt.days / 7.0
months_since_launch = (weeks_since_launch * 12.0 / 52.0).to_numpy()
cumulative_starts = observed["nbrx"].cumsum().to_numpy()
bass_fit = fit_bass(months_since_launch, cumulative_starts)
print(pd.Series(bass_fit).round(3).to_string())
"""
        ),
        figure(
            6,
            "bass_fit",
            "Fitted Bass adoption curve against Roventra's observed cumulative NBRx, projected to month 20.",
        ),
        md("## 14.3 In-Market Demand"),
        figure(
            7,
            "opening_window",
            "52 weeks of observed prescribing: 44 weeks to train on, the last 8, shaded, held out and scored against.",
        ),
        figure(
            8,
            "baseline_naive",
            "The naive forecast against the held-out weeks: a flat line against a still-climbing launch.",
        ),
        figure(
            9,
            "backtest_schematic",
            "Four backtest folds, each training on the gray span and testing on the darker span that immediately follows it.",
        ),
        figure(
            10,
            "ets_forecast",
            "The ETS forecast tracks the held-out weeks closely, level and trend alone.",
        ),
        figure(
            11,
            "sarima_forecast",
            "SARIMA against the held-out weeks: a reasonable track that drifts slightly high late in the window.",
        ),
        listing("Prophet with and without the access and promotion covariates"),
        code(
            """
holdout = 8
ds_y = observed.rename(columns={"week_start": "ds", "trx": "y"})
covariate_columns = ["access_multiplier", "promo_multiplier"]
train = ds_y.iloc[:-holdout].reset_index(drop=True)
future_covariates = ds_y.iloc[-holdout:][covariate_columns].reset_index(drop=True)

from forecasting import fit_prophet

without_covariates = fit_prophet(train, horizon=holdout)
with_covariates = fit_prophet(
    train, horizon=holdout, covariate_columns=covariate_columns, future_covariates=future_covariates
)
print(without_covariates[:3].round(1))
print(with_covariates[:3].round(1))
"""
        ),
        figure(
            12,
            "prophet_forecast",
            "Prophet with and without the planned access and promotion covariates, against the held-out weeks.",
        ),
        figure(
            13,
            "gbt_forecast",
            "Gradient-boosted trees against the held-out weeks: a nearly flat line against a still-climbing launch.",
        ),
        listing("Training the TFT across the territory panel"),
        code(
            """
holdout = 8
tft_forecast = results["holdout_forecasts"]["tft"].to_numpy()
print(tft_forecast.round(1))
"""
        ),
        figure(
            14,
            "tft_forecast",
            "The Temporal Fusion Transformer overshoots the held-out weeks and keeps climbing.",
        ),
        listing("Zero-shot forecasts from Chronos and TimesFM"),
        code(
            """
chronos_result = results["chronos_forecast"]
timesfm_result = results["holdout_forecasts"]["timesfm"].to_numpy()
print(chronos_result["median"].round(1).to_numpy())
print(timesfm_result.round(1))
"""
        ),
        figure(
            15,
            "foundation_forecast",
            "Chronos and TimesFM against the held-out weeks, with no training on Roventra data at all.",
        ),
        listing("The full accuracy scorecard"),
        code(
            """
scorecard = results["in_market_scorecard"]
print(scorecard.round({"mae": 1, "wmape": 3, "mape": 3, "mase": 2}))
"""
        ),
        figure(
            16,
            "foundation_vs_classical",
            "Every method against the held-out weeks, on one chart, colored to match its earlier dedicated figure.",
        ),
        listing("A calibrated interval around the ETS forecast"),
        code(
            """
from forecasting import conformal_interval, empirical_coverage

classical_backtest = results["in_market_backtest"]
holdout_forecasts = results["holdout_forecasts"]
actual_holdout = results["holdout_actual"]["actual"].to_numpy()
calibration_backtest = classical_backtest.loc[classical_backtest["method"] == "naive"]
interval = conformal_interval(calibration_backtest, holdout_forecasts["ets"], alpha=0.20)
coverage = empirical_coverage(interval, actual_holdout)
print(interval.round(1))
print(f"empirical coverage: {coverage:.0%}")
"""
        ),
        figure(
            17,
            "calibrated_fan_chart",
            "The calibrated 80% interval around the ETS point forecast, with the actual holdout overlaid.",
        ),
        figure(
            18,
            "production_refit",
            "The ETS production forecast against the weeks that actually followed: a straight-line extrapolation against a launch approaching its peak.",
        ),
        md("## 14.4 Demand-Supply Planning"),
        listing("Independent base forecasts by level"),
        code(
            """
hierarchy_base_forecast = results["hierarchy_base_forecast"]
print(hierarchy_base_forecast.iloc[0].round(1).to_string())
"""
        ),
        listing("All 3 reconciliation methods against the unreconciled base forecasts"),
        code(
            """
from forecasting import reconcile

hierarchy_base = {
    column: hierarchy_base_forecast[column].to_numpy()
    for column in hierarchy_base_forecast.columns
}
territory_order = [column for column in hierarchy_base if column != "National"]
territory_shares = results["territory_historical_shares"]
bottom_up = reconcile(hierarchy_base, territory_order, method="bottom_up")
top_down = reconcile(hierarchy_base, territory_order, method="top_down", historical_shares=territory_shares)
ols = reconcile(hierarchy_base, territory_order, method="ols")

comparison = pd.DataFrame({
    "Unreconciled": [hierarchy_base_forecast["National"].iloc[0], sum(hierarchy_base[r][0] for r in territory_order)],
    "Bottom-up": [bottom_up.loc["National", "h1"], bottom_up.loc[territory_order, "h1"].sum()],
    "Top-down": [top_down.loc["National", "h1"], top_down.loc[territory_order, "h1"].sum()],
    "OLS": [ols.loc["National", "h1"], ols.loc[territory_order, "h1"].sum()],
}, index=["National", "Territory sum"])
print(comparison.round(1))
"""
        ),
        figure(
            19,
            "forecast_hierarchy",
            "Only reconciliation makes the national forecast match the sum of 12 territories; unreconciled leaves a visible gap, every reconciled method closes it, and each closes it at a different shared total.",
        ),
        listing("Safety stock and the order signal, national level"),
        code(
            """
demand_to_supply = results["demand_to_supply"]
print(demand_to_supply.iloc[0].round(1).to_string())
"""
        ),
        listing("Safety stock and the order signal, by territory"),
        code(
            """
demand_to_supply_by_territory = results["demand_to_supply_by_territory"]
print(demand_to_supply_by_territory.round(1))
"""
        ),
        md("## 14.5 Loss of Exclusivity"),
        figure(
            20,
            "erosion_schematic",
            "A single half-life decay curve, reading fast in the early weeks and slow near its residual floor, with no break in the curve itself.",
        ),
        listing("Compare 2 analog erosion shapes from the same pre-entry level"),
        code(
            """
analog_erosion = results["analog_erosion_comparison"]
comparison = analog_erosion.loc[analog_erosion["weeks_since_entry"].isin([12.0, 52.0])].copy()
comparison["weeks_since_entry"] = comparison["weeks_since_entry"].astype(int)
comparison = comparison.rename(columns={
    "Comparable erosion A (fast generic substitution)": "Comparable A",
    "Comparable erosion B (slower substitution, branded loyalty)": "Comparable B",
})
print(comparison.round(1).to_string(index=False))
"""
        ),
        figure(
            21,
            "analog_erosion_curves",
            "Comparable erosion A and Comparable erosion B projected from the same pre-entry level.",
        ),
        listing("Fitting the decline on 20 weeks of post-entry data, then on 78"),
        code(
            """
from forecasting import fit_erosion

erosion_tail = results["erosion_tail"]
weeks_since_entry = erosion_tail["weeks_since_entry"].to_numpy()
trx_tail = erosion_tail["trx"].to_numpy()
early_fit = fit_erosion(weeks_since_entry[:20], trx_tail[:20])
mature_fit = fit_erosion(weeks_since_entry[:78], trx_tail[:78])
fit_comparison = pd.DataFrame(
    {
        "residual_fraction": [early_fit["residual_fraction"], mature_fit["residual_fraction"]],
        "half_life_weeks": [early_fit["half_life_weeks"], mature_fit["half_life_weeks"]],
    },
    index=["fit on 20 weeks", "fit on 78 weeks"],
).round({"residual_fraction": 4, "half_life_weeks": 1})
print(fit_comparison.to_string())
"""
        ),
        figure(
            22,
            "erosion_curve",
            "Actual post-entry TRx, the analog erosion band, and the Chronos zero-shot cross-check, with the right panel zooming the tail.",
        ),
        md("## 14.6 Consensus and Scenario Forecast"),
        listing("An accuracy-weighted consensus"),
        code(
            """
from forecasting import ensemble_consensus

consensus_base = {
    "patient_based": [results["patient_based_forecast"].iloc[0]["ceiling"] / 12.0] * 8,
    "ets": results["production_forecast"]["forecast"].to_numpy(),
    "chronos": results["chronos_production_forecast"]["forecast"].to_numpy(),
}
consensus = ensemble_consensus(consensus_base, scorecard)
consensus_table = pd.DataFrame({"horizon_step": range(1, 9), "consensus": consensus})
print(consensus_table.round({"consensus": 1}).to_string(index=False))
"""
        ),
        figure(
            23,
            "consensus_vs_actual",
            "The consensus, blended from patient-based, ETS, and Chronos, sits inside ETS's own 80% interval and tracks the actual continuation more closely than ETS alone.",
        ),
        listing("The reconciliation waterfall"),
        code(
            """
consensus_waterfall = results["consensus_waterfall"].round({"running_total": 1})
print(consensus_waterfall.to_string(formatters={"adjustment_pct": "{:.3f}".format}))
"""
        ),
        figure(
            24,
            "consensus_waterfall",
            "From the analytics consensus to the committed number: each adjustment shown as both a volume delta and the percentage that produced it, against a reference line at the starting total.",
        ),
        listing("Low, Base, and High scenarios"),
        code(
            """
from forecasting import scenario_forecast

print(scenario_forecast())
"""
        ),
        figure(
            25,
            "scenario_fan",
            "Low, Base, and High launch scenarios form a fan from explicit driver assumptions.",
        ),
        md(
            """
## 14.7 A Forecasting Method Selection Field Guide

Gather these inputs before reusing any method in this chapter on a live brand:

- target series and unit, such as NBRx, TRx, units, or dollars
- forecast horizon and cadence
- known future events and covariates
- hierarchy levels that must reconcile
- enough backtest folds to score without leakage
- quantified business adjustments, traceable back to a specific driver

Table 14.2 (methods used in this chapter) and Table 14.3 (common pharma-forecasting methods this chapter did not need) in the manuscript turn this into a practical selection rule.

Interval recap:
- pre-launch business case: a Monte Carlo band over the funnel's own assumption ranges
- in-market demand: a conformal interval calibrated from real backtest residuals
- loss of exclusivity: an analog band, then a Chronos cross-check
- supply planning: a safety-stock buffer sized to each territory's own volatility
- consensus: driver-based low, base, and high scenarios, not a statistical band
"""
        ),
        md(
            """
## 14.8 Revising the Forecast

A forecast earns its keep by being corrected as real data arrives, not by being right the first time: re-fit as prescribing accumulates, backtest on a fixed schedule so a materially better challenger method replaces the incumbent, reconcile a hierarchy every cycle rather than once, and keep every business adjustment on the consensus named and traceable. Real pharma demand-forecasting functions run this whole loop, monthly or quarterly, inside the S&OP (sales and operations planning) cycle that brings commercial, finance, and supply planning to the same table. See the manuscript's 14.8 for the full stage-by-stage discussion.
"""
        ),
        md(
            """
## 14.9 Summary

This chapter built a forecast for every stage of the Roventra lifecycle, from a pre-launch business case with no data at all to a post-generic decline years in the future. The transferable lesson is the thinking, not any specific number this fictional brand produced: trust an assumption only until data exists to check it, let a method earn its place through an honest backtest, don't trust a decline curve before its tail shows itself, reconcile a hierarchy because independent good models won't agree by default, and keep every business adjustment named and traceable.
"""
        ),
    ]
    return notebook(cells)


def exercise_solutions():
    cells = [
        md(
            """
# Chapter 14: Forecasting Exercise Solutions

These worked answers use the same Chapter 14 data and functions as the manuscript.
"""
        ),
        code(SETUP_EXERCISES),
        md("## Exercise 1"),
        code(
            """
weeks_since_launch = (observed["week_start"] - national["week_start"].iloc[0]).dt.days / 7.0
months_since_launch = (weeks_since_launch * 12.0 / 52.0).to_numpy()
cumulative_starts = observed["nbrx"].cumsum().to_numpy()
fit_26 = fit_bass(months_since_launch[:26], cumulative_starts[:26])
fit_52 = fit_bass(months_since_launch, cumulative_starts)
comparison = pd.DataFrame([fit_26, fit_52], index=["26 weeks", "52 weeks"])
print(comparison[["m", "time_to_peak_months"]].round(1))
"""
        ),
        md("The 26-week fit is less stable because it sees only the early launch ramp. A finance team should treat it as a directional read, not a ceiling commitment."),
        md("## Exercise 2"),
        code(
            """
nbrx_series = observed.set_index("week_start")["nbrx"]
methods = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive,
    "ets": fit_ets,
    "sarima": fit_sarima,
}
nbrx_backtest = rolling_origin_backtest(nbrx_series, methods)
nbrx_scorecard = accuracy_scorecard(nbrx_backtest)
print(nbrx_scorecard)
"""
        ),
        md("NBRx is a flow of new starts, while TRx is a stock-and-flow measure that accumulates persistent patients. A method can rank differently when the target is noisier and less cumulative."),
        md("## Exercise 3"),
        code(
            """
erosion_tail = results["erosion_tail"]
weeks_since_entry = erosion_tail["weeks_since_entry"].to_numpy()
trx_tail = erosion_tail["trx"].to_numpy()
mature = fit_erosion(weeks_since_entry[:78], trx_tail[:78])["residual_fraction"]
rows = []
for window in range(20, 79):
    estimate = fit_erosion(weeks_since_entry[:window], trx_tail[:window])["residual_fraction"]
    within_10_pct = abs(estimate - mature) <= 0.10 * mature
    rows.append({"window": window, "residual_fraction": estimate, "within_10_pct": within_10_pct})
windows = pd.DataFrame(rows)
print(windows.loc[windows["within_10_pct"]].head(1).round(3).to_string(index=False))
"""
        ),
        md("The practical judgment is to avoid committing the residual tail until enough post-entry data shows the curve flattening. Before that point, use analogs and a cross-check, and label the estimate provisional."),
    ]
    return notebook(cells)


def execute_and_write(notebook_node: nbf.NotebookNode, path: Path) -> None:
    client = NotebookClient(notebook_node, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    for cell in executed.cells:
        if "outputs" not in cell:
            continue
        cell.outputs = [
            output
            for output in cell.outputs
            if not (
                output.get("output_type") == "display_data"
                and "application/vnd.jupyter.widget-view+json" in output.get("data", {})
            )
        ]
    nbf.write(executed, path)
    print(f"Wrote and executed {path.relative_to(ROOT)}")


def main() -> None:
    execute_and_write(walkthrough(), CHAPTER / "ch14_walkthrough.ipynb")
    execute_and_write(exercise_solutions(), CHAPTER / "ch14_exercise_solutions.ipynb")


if __name__ == "__main__":
    main()
