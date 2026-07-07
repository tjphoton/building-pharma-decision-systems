"""Run the complete Chapter 14 forecasting analysis across all five use cases."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import forecasting as fc  # noqa: E402
from forecast_config import (  # noqa: E402
    AS_OF_DATE,
    GENERIC_ENTRY_DATE,
    LAUNCH_DATE,
)
from generate_forecast_data import run_generation  # noqa: E402


def run_analysis(seed: int | None = None) -> dict[str, object]:
    """Return the full Chapter 14 package: one result set per use case."""
    warnings.filterwarnings("ignore")
    lifecycle = run_generation(seed=seed) if seed is not None else run_generation()
    national = lifecycle["national_series"]
    observed = lifecycle["observed_series"]
    region_series = lifecycle["region_series"]
    territory_series = lifecycle["territory_series"]

    results: dict[str, object] = {
        "national_series": national,
        "observed_series": observed,
        "region_series": region_series,
        "territory_series": territory_series,
    }

    # 14.2 Pre-launch demand ------------------------------------------------
    business_case = fc.patient_based_forecast()
    results["patient_based_forecast"] = pd.DataFrame([business_case])

    weeks_since_launch = (observed["week_start"] - LAUNCH_DATE).dt.days / 7.0
    months_since_launch = (weeks_since_launch * 12.0 / 52.0).to_numpy()
    cumulative_starts = observed["nbrx"].cumsum().to_numpy()
    bass_fit = fc.fit_bass(months_since_launch, cumulative_starts)
    results["bass_fit"] = pd.DataFrame([bass_fit])

    weeks_per_month = 52.0 / 12.0
    early_fraction = observed["nbrx"].cumsum() / business_case["ceiling"]
    months_grid = np.arange(0, 96, dtype=float)
    analog = fc.analog_forecast(
        months_grid, business_case["ceiling"], early_actual_fraction=early_fraction, periods_per_month=weeks_per_month
    )
    results["analog_forecast"] = pd.DataFrame(
        {
            "month": analog["months"],
            "cumulative": analog["cumulative"],
            "monthly_new_starts": analog["monthly_new_starts"],
        }
    )
    results["analog_choice"] = pd.DataFrame([{"analog_name": analog["analog_name"], **analog["analog_params"]}])
    analog_months = np.arange(1, 61, dtype=float)
    analog_shape_rows = {"month": analog_months}
    for name, params in fc.ANALOG_LAUNCHES.items():
        analog_shape_rows[name] = fc.bass_cumulative_fraction(analog_months, params["p"], params["q"])
    results["analog_launch_shapes"] = pd.DataFrame(analog_shape_rows)

    def _analog_errors(n_weeks: int) -> pd.DataFrame:
        weeks = np.arange(1, n_weeks + 1, dtype=float)
        months = weeks / weeks_per_month
        actual = early_fraction.iloc[:n_weeks].to_numpy()
        rows = []
        for name, params in fc.ANALOG_LAUNCHES.items():
            projected = fc.bass_cumulative_fraction(months, params["p"], params["q"])
            rows.append({"analog_name": name, "mse": float(np.mean((projected - actual) ** 2))})
        return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)

    results["analog_selection_errors_early"] = _analog_errors(26)
    results["analog_selection_errors"] = _analog_errors(52)

    selection_weeks = np.arange(1, 53, dtype=float)
    selection_months = selection_weeks / weeks_per_month
    actual_selection = early_fraction.iloc[:52].to_numpy()
    analog_selection_rows = {
        "week": selection_weeks,
        "month": selection_months,
        "Roventra actual": actual_selection,
    }
    for name, params in fc.ANALOG_LAUNCHES.items():
        analog_selection_rows[name] = fc.bass_cumulative_fraction(selection_months, params["p"], params["q"])
    results["analog_selection_overlay"] = pd.DataFrame(analog_selection_rows)

    persistence_check = fc.persistence_to_trx(observed["nbrx"].to_numpy())
    persistence_check["actual_stock"] = observed["on_therapy_stock"].to_numpy()
    persistence_check["actual_trx"] = observed["trx"].to_numpy()
    results["persistence_check"] = persistence_check

    monte_carlo = fc.monte_carlo_funnel()
    results["monte_carlo_summary"] = monte_carlo.describe()
    results["assumption_tornado"] = fc.assumption_tornado()

    # 14.3 In-market demand ---------------------------------------------------
    trx_series = observed.set_index("week_start")["trx"]
    classical_methods = {
        "naive": fc.naive_forecast,
        "seasonal_naive": fc.seasonal_naive,
        "ets": fc.fit_ets,
        "sarima": fc.fit_sarima,
    }
    classical_backtest = fc.rolling_origin_backtest(trx_series, classical_methods)

    ds_y = observed.rename(columns={"week_start": "ds", "trx": "y"})
    covariate_columns = ["access_multiplier", "promo_multiplier"]
    holdout = 8
    prophet_train = ds_y.iloc[:-holdout].reset_index(drop=True)
    prophet_future_cov = ds_y.iloc[-holdout:][covariate_columns].reset_index(drop=True)
    prophet_forecast = fc.fit_prophet(
        prophet_train, horizon=holdout, covariate_columns=covariate_columns, future_covariates=prophet_future_cov
    )
    prophet_no_covariates = fc.fit_prophet(prophet_train, horizon=holdout)
    gbt_forecast = fc.fit_gbt(
        prophet_train, horizon=holdout, covariate_columns=covariate_columns, future_covariates=prophet_future_cov
    )

    territory_train = territory_series.loc[territory_series["week_start"] <= observed["week_start"].iloc[-1 - holdout]]
    panel = territory_train.rename(columns={"territory": "unique_id", "week_start": "ds", "trx": "y"})[
        ["unique_id", "ds", "y"]
    ]
    tft_forecast = fc.fit_tft(panel, horizon=holdout)

    context = trx_series.to_numpy()[:-holdout]
    chronos_result = fc.chronos_forecast(context, horizon=holdout)
    timesfm_forecast = fc.timesfm_forecast(context, horizon=holdout)

    actual_holdout = trx_series.to_numpy()[-holdout:]
    holdout_forecasts = {
        "naive": fc.naive_forecast(trx_series.iloc[:-holdout], holdout),
        "seasonal_naive": fc.seasonal_naive(trx_series.iloc[:-holdout], holdout),
        "ets": fc.fit_ets(trx_series.iloc[:-holdout], holdout),
        "sarima": fc.fit_sarima(trx_series.iloc[:-holdout], holdout),
        "prophet": prophet_forecast,
        "gbt": gbt_forecast,
        "tft": tft_forecast,
        "chronos": chronos_result["median"].to_numpy(),
        "timesfm": timesfm_forecast,
    }
    holdout_backtest_rows = []
    for method_name, forecast in holdout_forecasts.items():
        for step, (actual_value, predicted_value) in enumerate(zip(actual_holdout, forecast, strict=True), start=1):
            holdout_backtest_rows.append(
                {"fold": 0, "method": method_name, "horizon_step": step, "actual": actual_value, "predicted": predicted_value}
            )
    full_backtest = pd.concat([classical_backtest, pd.DataFrame(holdout_backtest_rows)], ignore_index=True)
    scorecard = fc.accuracy_scorecard(full_backtest)

    results["in_market_backtest"] = full_backtest
    results["in_market_scorecard"] = scorecard
    results["prophet_no_covariates_holdout"] = prophet_no_covariates
    results["chronos_forecast"] = chronos_result
    results["holdout_forecasts"] = pd.DataFrame(holdout_forecasts, index=pd.RangeIndex(1, holdout + 1, name="horizon_step"))
    results["holdout_actual"] = pd.DataFrame({"horizon_step": range(1, holdout + 1), "actual": actual_holdout})

    # The backtest and holdout scoring above exist to pick a method honestly:
    # every one of those forecasts is fit on weeks 1-44 and scored against
    # weeks 45-52, which have already happened. The number operations actually
    # needs is different: the winning method refit on all 52 weeks, forecasting
    # the genuinely unknown weeks 53-60. Reusing the 44-week holdout forecast
    # for that purpose would silently pass off a scoring artifact as a
    # production number.
    production_dispatch: dict[str, Callable[[], np.ndarray]] = {
        "naive": lambda: fc.naive_forecast(trx_series, holdout),
        "seasonal_naive": lambda: fc.seasonal_naive(trx_series, holdout),
        "ets": lambda: fc.fit_ets(trx_series, holdout),
        "sarima": lambda: fc.fit_sarima(trx_series, holdout),
        "chronos": lambda: fc.chronos_forecast(trx_series.to_numpy(), horizon=holdout)["median"].to_numpy(),
        "timesfm": lambda: fc.timesfm_forecast(trx_series.to_numpy(), horizon=holdout),
    }
    winning_method = scorecard.sort_values("mase").iloc[0]["method"]
    if winning_method in production_dispatch:
        production_forecast = production_dispatch[winning_method]()
    else:
        # Prophet, GBT, and TFT all need future-covariate or panel inputs this
        # exercise never defines past the holdout window, so a full-history
        # refit isn't available for them; ETS is the highest-ranked method
        # with a plain (series, horizon) signature to fall back to.
        winning_method = "ets"
        production_forecast = production_dispatch["ets"]()
    results["production_method"] = winning_method
    results["production_forecast"] = pd.DataFrame(
        {"horizon_step": range(1, holdout + 1), "forecast": production_forecast}
    )

    calibration_backtest = classical_backtest.loc[classical_backtest["method"] == "naive"]
    interval = fc.conformal_interval(calibration_backtest, holdout_forecasts["ets"], alpha=0.20)
    coverage = fc.empirical_coverage(interval, actual_holdout)
    results["conformal_interval"] = interval
    results["conformal_coverage"] = pd.DataFrame([{"nominal": 0.80, "empirical": coverage}])

    # Same calibration set, same half-width, applied to the production forecast
    # (weeks 53-60) instead of the holdout-fit one (weeks 45-52): the interval
    # is only as trustworthy as the assumption that error behavior stays the
    # same going forward, which Figure 14.18 puts to a direct test.
    results["production_interval"] = fc.conformal_interval(calibration_backtest, production_forecast, alpha=0.20)

    # 14.4 Loss of exclusivity ------------------------------------------------
    post_entry = national.loc[national["week_start"] >= GENERIC_ENTRY_DATE].reset_index(drop=True)
    weeks_since_entry = np.arange(len(post_entry), dtype=float)
    trx_tail = post_entry["trx"].to_numpy()

    early_fit = fc.fit_erosion(weeks_since_entry[:20], trx_tail[:20])
    mature_fit = fc.fit_erosion(weeks_since_entry[:78], trx_tail[:78])
    results["erosion_fits"] = pd.DataFrame(
        [
            {"fit_window_weeks": 20, **{k: v for k, v in early_fit.items() if k != "fitted"}},
            {"fit_window_weeks": 78, **{k: v for k, v in mature_fit.items() if k != "fitted"}},
        ]
    )
    erosion_weeks = np.arange(0, 105, dtype=float)
    analog_erosion_rows = {"weeks_since_entry": erosion_weeks}
    for name in fc.ANALOG_EROSIONS:
        analog_curve = fc.analog_erosion_forecast(erosion_weeks, pre_entry_reference=trx_tail[0], analog_name=name)
        analog_erosion_rows[name] = analog_curve["projected"]
    results["analog_erosion_comparison"] = pd.DataFrame(analog_erosion_rows)
    first_analog_name = next(iter(fc.ANALOG_EROSIONS))
    results["analog_erosion"] = pd.DataFrame(
        {
            "weeks_since_entry": erosion_weeks,
            "projected": results["analog_erosion_comparison"][first_analog_name].to_numpy(),
        }
    )
    chronos_erosion_check = fc.chronos_forecast(trx_tail[:78], horizon=8)
    results["chronos_erosion_cross_check"] = chronos_erosion_check
    results["erosion_tail"] = pd.DataFrame({"weeks_since_entry": weeks_since_entry, "trx": trx_tail})

    # 14.5 Demand-supply planning ----------------------------------------------
    observed_national_trx = observed.set_index("week_start")["trx"]
    observed_territories = territory_series.loc[territory_series["week_start"] <= AS_OF_DATE]
    territories_by_name = {
        name: group.set_index("week_start")["trx"] for name, group in observed_territories.groupby("territory")
    }
    # Every level here uses the same validated method (ETS), fit independently
    # on its own series. That isolates the actual hierarchical-forecasting
    # problem: independently fit forecasts do not sum correctly across levels
    # even when every level's model is good, not merely when one level is
    # stuck with a worse method than another. 12 territories, not 4 regions,
    # because more independently fit bottom-level series compound more
    # estimation error: the same exercise with 4 regions understates how bad
    # the coherence problem gets as a hierarchy gets more granular, which is
    # exactly the direction real supply planning needs (territory, not
    # region, is where inventory actually gets held). Reconciliation below is
    # the mechanism built to force them back into one coherent set of numbers.
    hierarchy_base = {"National": production_forecast}
    for territory_name, territory_series_data in territories_by_name.items():
        hierarchy_base[territory_name] = fc.fit_ets(territory_series_data, horizon=8)
    territory_order = list(territories_by_name.keys())
    results["hierarchy_base_forecast"] = pd.DataFrame(
        hierarchy_base, index=pd.RangeIndex(1, 9, name="horizon_step")
    )
    territory_totals = {name: float(series.sum()) for name, series in territories_by_name.items()}
    total_of_totals = sum(territory_totals.values())
    territory_shares = {name: value / total_of_totals for name, value in territory_totals.items()}
    results["territory_historical_shares"] = territory_shares
    results["reconciled_bottom_up"] = fc.reconcile(hierarchy_base, territory_order, method="bottom_up")
    results["reconciled_top_down"] = fc.reconcile(
        hierarchy_base, territory_order, method="top_down", historical_shares=territory_shares
    )
    results["reconciled_ols"] = fc.reconcile(hierarchy_base, territory_order, method="ols")

    reconciled_national = results["reconciled_ols"].loc["National"].to_numpy()
    demand_std = np.full(len(reconciled_national), float(np.std(observed_national_trx.diff().dropna())))
    results["demand_to_supply"] = fc.demand_to_supply(reconciled_national, demand_std)

    # Inventory is held at the territory warehouse, not nationally, so the
    # order signal that actually gets acted on has to be computed once per
    # territory, each against its own reconciled demand and its own demand
    # volatility, not once against the national total. The national number
    # above is kept only as the illustration of the formula's mechanics.
    territory_supply_rows = []
    for territory_name in territory_order:
        territory_demand = float(results["reconciled_ols"].loc[territory_name, "h1"])
        territory_std = float(np.std(territories_by_name[territory_name].diff().dropna()))
        territory_signal = fc.demand_to_supply(np.array([territory_demand]), np.array([territory_std])).iloc[0]
        territory_supply_rows.append({"territory": territory_name, **territory_signal.to_dict()})
    results["demand_to_supply_by_territory"] = pd.DataFrame(territory_supply_rows).set_index("territory")

    # 14.6 Consensus and scenario forecast --------------------------------------
    # Every contribution here is a full-history production forecast for weeks
    # 53-60, not a 44-week holdout forecast used to score a method in 14.3.9:
    # the consensus is a number for weeks that have not happened yet.
    chronos_production_forecast = fc.chronos_forecast(trx_series.to_numpy(), horizon=holdout)["median"].to_numpy()
    results["chronos_production_forecast"] = pd.DataFrame(
        {"horizon_step": range(1, holdout + 1), "forecast": chronos_production_forecast}
    )
    consensus_base = {
        "patient_based": np.full(8, business_case["ceiling"] / 12.0),
        winning_method: production_forecast,
        "chronos": chronos_production_forecast,
    }
    consensus = fc.ensemble_consensus(consensus_base, scorecard)
    results["ensemble_consensus"] = pd.DataFrame({"horizon_step": range(1, 9), "consensus": consensus})

    adjustments = {
        "Access assumption tightened": -0.08,
        "Launch-support level confirmed": 0.05,
        "Competitor entry risk": -0.03,
    }
    results["consensus_waterfall"] = fc.consensus_reconcile(float(consensus.sum()), adjustments)

    results["scenario_forecast"] = fc.scenario_forecast()

    return results


def write_outputs(results: Mapping[str, object], output_dir: Path) -> None:
    """Write every DataFrame result to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    results = run_analysis()
    output_dir = ROOT / "ch14_forecasting" / "assets" / "generated_outputs"
    write_outputs(results, output_dir)
    print(f"Wrote {sum(isinstance(v, pd.DataFrame) for v in results.values())} tables to {output_dir}")


if __name__ == "__main__":
    main()
