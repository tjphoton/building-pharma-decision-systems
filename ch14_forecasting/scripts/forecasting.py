"""Forecasting methods for Chapter 14, organized by use case.

Section 14.2, Pre-Launch Demand: `patient_based_forecast`, `persistence_to_trx`,
`analog_forecast`, `fit_bass`, `monte_carlo_funnel`.

`bass_cumulative_fraction` and `persistence_survival` are the canonical
structural-model functions: `generate_forecast_data.py` imports them
directly to build the chapter's ground truth, and the functions below apply
the same math to the observed data a reader actually has, so the generator
and the taught methods never drift apart.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Do not import torch at module level. LightGBM and PyTorch each bundle
# their own OpenMP runtime; on this platform, importing torch before
# lightgbm loads and fits a model crashes the process with SIGSEGV the
# instant LightGBM initializes its runtime. Importing lightgbm first and
# torch afterward (the order every function below already uses) avoids the
# conflict. Every function that touches torch (fit_tft, chronos_forecast,
# timesfm_forecast) imports it locally, right where it is used.

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from forecast_config import (  # noqa: E402
    ANALOG_EROSIONS,
    ANALOG_LAUNCHES,
    BACKTEST_FOLDS,
    BACKTEST_HOLDOUT_WEEKS,
    CHRONOS_MODEL_ID,
    FUNNEL_ASSUMPTION_RANGES,
    MONTE_CARLO_DRAWS,
    MONTE_CARLO_SEED,
    PEAK_BRAND_SHARE_ASSUMPTION,
    PERSISTENCE_SCALE_MONTHS,
    PERSISTENCE_SHAPE,
    REFILLS_PER_PATIENT_MONTH,
    SEASONAL_PERIOD_WEEKS,
    SERVICE_LEVEL_Z,
    SUPPLY_LEAD_TIME_WEEKS,
    TFT_HIDDEN_SIZE,
    TFT_INPUT_SIZE_WEEKS,
    TFT_MAX_STEPS,
    TIMESFM_MODEL_ID,
    TOTAL_ADDRESSABLE_PATIENTS,
)

WEEKS_PER_MONTH = 52.0 / 12.0


# ---------------------------------------------------------------------------
# Shared structural functions (also used by generate_forecast_data.py)
# ---------------------------------------------------------------------------


def bass_cumulative_fraction(months_since_launch: np.ndarray, p: float, q: float) -> np.ndarray:
    """Fraction of the eventual ceiling adopted by each time point.

    Standard Bass (1969) closed-form cumulative adoption fraction. `p` is
    the innovation coefficient (adoption driven by outside information, such
    as approval news or medical-conference coverage), `q` the imitation
    coefficient (adoption driven by peer prescribers already using the
    drug), and time is measured in months since launch.
    """
    t = np.clip(months_since_launch, a_min=0.0, a_max=None)
    exp_term = np.exp(-(p + q) * t)
    return (1.0 - exp_term) / (1.0 + (q / p) * exp_term)


def persistence_survival(weeks_on_therapy: np.ndarray) -> np.ndarray:
    """Fraction of a starting cohort still on therapy after k weeks.

    Weibull survival with shape below 1, giving the decreasing hazard rate
    (steep early dropout, flattening tail) that a real Kaplan-Meier
    persistence curve shows, consistent in shape with the line-1 persistence
    curve in the patient-journey chapter without re-deriving it from
    patient-level claims.
    """
    scale_weeks = PERSISTENCE_SCALE_MONTHS * WEEKS_PER_MONTH
    return np.exp(-((weeks_on_therapy / scale_weeks) ** PERSISTENCE_SHAPE))


# ---------------------------------------------------------------------------
# 14.2.1 The patient-based funnel forecast (top-down)
# ---------------------------------------------------------------------------


def patient_based_forecast(
    addressable_patients: float = TOTAL_ADDRESSABLE_PATIENTS,
    access_adjustment: float = 0.78,
    brand_share: float = PEAK_BRAND_SHARE_ASSUMPTION,
) -> dict[str, float]:
    """The pre-launch business-case ceiling: addressable patients times access times peak share.

    Returns the ceiling (peak treated patients under these assumptions) and
    the three factors that produced it, so a reader can trace exactly which
    assumption to challenge.
    """
    ceiling = addressable_patients * access_adjustment * brand_share
    return {
        "addressable_patients": addressable_patients,
        "access_adjustment": access_adjustment,
        "brand_share": brand_share,
        "ceiling": ceiling,
    }


# ---------------------------------------------------------------------------
# 14.2.2 From new starts to volume: the persistence conversion
# ---------------------------------------------------------------------------


def persistence_to_trx(
    new_starts: np.ndarray, refills_per_patient_month: float = REFILLS_PER_PATIENT_MONTH
) -> pd.DataFrame:
    """Convert a weekly new-start series into on-therapy stock and TRx.

    On-therapy stock in week t is the convolution of every prior week's new
    starts with the persistence survival curve: a start in week s is still
    on therapy in week t with probability `persistence_survival(t - s)`.
    TRx follows from patient-months on therapy times the refill rate.
    """
    new_starts = np.asarray(new_starts, dtype=float)
    n = len(new_starts)
    kernel = persistence_survival(np.arange(n, dtype=float))
    stock = np.convolve(new_starts, kernel)[:n]
    trx = stock * refills_per_patient_month * (1.0 / WEEKS_PER_MONTH)
    return pd.DataFrame({"nbrx": new_starts, "on_therapy_stock": stock, "trx": trx})


# ---------------------------------------------------------------------------
# 14.2.3 Analog-based forecasting
# ---------------------------------------------------------------------------


def analog_forecast(
    months_grid: np.ndarray,
    ceiling: float,
    early_actual_fraction: pd.Series | None = None,
    periods_per_month: float = 1.0,
) -> dict[str, object]:
    """Borrow a comparable launch's normalized shape and scale it to the ceiling.

    Without any Roventra data, an analyst has to pick one comparable launch
    from the analog library. With a few months of early actual data
    available, the analog whose own early normalized shape (fraction of its
    ceiling reached by the same number of months) best matches the current
    brand's early trajectory is selected automatically; otherwise the first
    analog in the library is used as the default. `early_actual_fraction` is
    assumed to hold one observation per month unless `periods_per_month` says
    otherwise, for example 52 / 12 for weekly data, so that early-period
    elapsed time is converted to months before evaluating the Bass curve.
    """
    best_name = next(iter(ANALOG_LAUNCHES))
    best_error = np.inf
    if early_actual_fraction is not None and len(early_actual_fraction) > 0:
        early_months = np.arange(1, len(early_actual_fraction) + 1, dtype=float) / periods_per_month
        for name, params in ANALOG_LAUNCHES.items():
            analog_early = bass_cumulative_fraction(early_months, params["p"], params["q"])
            error = float(np.mean((analog_early - early_actual_fraction.to_numpy()) ** 2))
            if error < best_error:
                best_error = error
                best_name = name

    chosen = ANALOG_LAUNCHES[best_name]
    projected_fraction = bass_cumulative_fraction(months_grid, chosen["p"], chosen["q"])
    cumulative = projected_fraction * ceiling
    monthly_new_starts = np.diff(cumulative, prepend=0.0)
    return {
        "analog_name": best_name,
        "analog_params": chosen,
        "months": months_grid,
        "cumulative": cumulative,
        "monthly_new_starts": monthly_new_starts,
    }


# ---------------------------------------------------------------------------
# 14.2.4 The launch uptake curve: Bass diffusion
# ---------------------------------------------------------------------------


def fit_bass(months_since_launch: np.ndarray, cumulative_starts: np.ndarray) -> dict[str, float]:
    """Fit Bass p, q, m to observed cumulative new-to-therapy starts.

    Uses nonlinear least squares on the closed-form cumulative curve. The
    ceiling m is fit alongside p and q rather than fixed, since a
    pre-launch business case's ceiling assumption is exactly what the fit
    is meant to check.
    """

    def model(t, p, q, m):
        return bass_cumulative_fraction(t, p, q) * m

    max_observed = float(np.max(cumulative_starts))
    initial_guess = (0.02, 0.30, max(max_observed * 2.0, 1.0))
    bounds = ([1e-4, 1e-4, max_observed], [1.0, 2.0, max_observed * 20.0])
    (p_fit, q_fit, m_fit), _ = curve_fit(
        model, months_since_launch, cumulative_starts, p0=initial_guess, bounds=bounds, maxfev=20_000
    )

    peak_month = float(np.log(q_fit / p_fit) / (p_fit + q_fit))
    peak_monthly_rate = float(m_fit * (p_fit + q_fit) ** 2 / (4.0 * q_fit))
    return {
        "p": float(p_fit),
        "q": float(q_fit),
        "m": float(m_fit),
        "time_to_peak_months": peak_month,
        "peak_monthly_new_starts": peak_monthly_rate,
    }


# ---------------------------------------------------------------------------
# 14.2.5 Pre-launch uncertainty: Monte Carlo on the assumptions
# ---------------------------------------------------------------------------


def monte_carlo_funnel(
    n_draws: int = MONTE_CARLO_DRAWS, seed: int = MONTE_CARLO_SEED
) -> pd.DataFrame:
    """Draw each funnel assumption from its stated range and compute the resulting ceiling.

    Returns one row per draw with every sampled assumption and the ceiling
    it implies, so the reader can both read the distribution and attribute
    variance to a specific assumption.
    """
    rng = np.random.default_rng(seed)
    draws = {}
    for name, spec in FUNNEL_ASSUMPTION_RANGES.items():
        draws[name] = rng.uniform(spec["low"], spec["high"], size=n_draws)
    ceiling = draws["addressable_patients"] * draws["access_adjustment"] * draws["brand_share"]
    result = pd.DataFrame(draws)
    result["ceiling"] = ceiling
    return result


def assumption_tornado() -> pd.DataFrame:
    """Single-factor sensitivity: vary each assumption across its range, others at base.

    Ranks assumptions by how much the ceiling moves when that assumption
    alone spans its stated low-to-high range, which is the driver a
    business case should spend the most diligence on.
    """
    base = {name: spec["base"] for name, spec in FUNNEL_ASSUMPTION_RANGES.items()}
    rows = []
    for name, spec in FUNNEL_ASSUMPTION_RANGES.items():
        if name == "persistence_scale_months":
            continue
        low_case = dict(base)
        high_case = dict(base)
        low_case[name] = spec["low"]
        high_case[name] = spec["high"]
        ceiling_low = (
            low_case["addressable_patients"] * low_case["access_adjustment"] * low_case["brand_share"]
        )
        ceiling_high = (
            high_case["addressable_patients"]
            * high_case["access_adjustment"]
            * high_case["brand_share"]
        )
        rows.append(
            {
                "assumption": name,
                "ceiling_low": ceiling_low,
                "ceiling_high": ceiling_high,
                "range_width": abs(ceiling_high - ceiling_low),
            }
        )
    result = pd.DataFrame(rows).sort_values("range_width", ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 14.3.1 Baselines to beat: naive and seasonal-naive
# ---------------------------------------------------------------------------


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step of the horizon."""
    return np.full(horizon, float(train.iloc[-1]))


def seasonal_naive(
    train: pd.Series, horizon: int, season_length: int = SEASONAL_PERIOD_WEEKS
) -> np.ndarray:
    """Repeat the value from one season ago; fall back to the plain naive forecast if there is no full season of history yet."""
    if len(train) < season_length:
        return naive_forecast(train, horizon)
    last_season = train.iloc[-season_length:].to_numpy()
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


# ---------------------------------------------------------------------------
# 14.3.2 How we score: rolling-origin backtest and error metrics
# ---------------------------------------------------------------------------


def rolling_origin_backtest(
    series: pd.Series,
    methods: dict[str, Callable[[pd.Series, int], np.ndarray]],
    horizon: int = BACKTEST_HOLDOUT_WEEKS,
    n_folds: int = BACKTEST_FOLDS,
    min_train_size: int = 12,
) -> pd.DataFrame:
    """Walk-forward backtest: every fold's training window strictly precedes its test window.

    Each entry in `methods` is a callable `(train_series, horizon) ->
    forecast array of length horizon`. All methods are scored on the same
    folds so the resulting scorecard compares them fairly. Returns a
    long-format table: one row per fold, method, and horizon step.
    """
    n = len(series)
    rows: list[dict[str, object]] = []
    for fold in range(n_folds):
        test_end = n - fold * horizon
        test_start = test_end - horizon
        train_end = test_start
        if train_end < min_train_size or test_start < 0:
            continue
        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]
        for name, method in methods.items():
            predicted = np.asarray(method(train, horizon), dtype=float)
            for step, (actual_value, predicted_value) in enumerate(
                zip(test.to_numpy(), predicted, strict=True), start=1
            ):
                rows.append(
                    {
                        "fold": fold,
                        "method": name,
                        "horizon_step": step,
                        "actual": float(actual_value),
                        "predicted": float(predicted_value),
                    }
                )
    return pd.DataFrame(rows)


def accuracy_scorecard(backtest: pd.DataFrame, baseline_method: str = "seasonal_naive") -> pd.DataFrame:
    """Summarize a backtest into MAE, WMAPE, MAPE, and MASE by method.

    MASE divides each method's mean absolute error by the baseline
    method's mean absolute error on the same folds, so a MASE below 1.0
    means the method beats the baseline.
    """
    baseline_rows = backtest.loc[backtest["method"] == baseline_method]
    baseline_mae = (baseline_rows["actual"] - baseline_rows["predicted"]).abs().mean()

    rows = []
    for method, group in backtest.groupby("method"):
        abs_error = (group["actual"] - group["predicted"]).abs()
        mae = abs_error.mean()
        wmape = abs_error.sum() / group["actual"].abs().sum()
        mape = (abs_error / group["actual"].replace(0, np.nan)).mean()
        mase = mae / baseline_mae if baseline_mae > 0 else np.nan
        rows.append({"method": method, "mae": mae, "wmape": wmape, "mape": mape, "mase": mase})
    return pd.DataFrame(rows).sort_values("mase").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 14.3.3 Exponential smoothing (ETS / Holt-Winters)
# ---------------------------------------------------------------------------


def fit_ets(train: pd.Series, horizon: int, season_length: int = SEASONAL_PERIOD_WEEKS) -> np.ndarray:
    """Fit Holt-Winters exponential smoothing and forecast the horizon.

    Falls back to a non-seasonal fit (trend only) when the training window
    does not cover two full seasons, since a 52-week seasonal component is
    not identifiable from less than about two years of weekly history; this
    is exactly the short-history limitation the chapter's cold-start
    argument is built on.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    has_full_seasons = len(train) >= 2 * season_length
    model = ExponentialSmoothing(
        train.to_numpy(),
        trend="add",
        seasonal="add" if has_full_seasons else None,
        seasonal_periods=season_length if has_full_seasons else None,
        initialization_method="estimated",
    )
    fitted = model.fit()
    return np.asarray(fitted.forecast(horizon))


# ---------------------------------------------------------------------------
# 14.3.4 ARIMA / SARIMA
# ---------------------------------------------------------------------------


def fit_sarima(train: pd.Series, horizon: int, season_length: int = SEASONAL_PERIOD_WEEKS) -> np.ndarray:
    """Fit a SARIMA model as the statistical benchmark, with the same short-history fallback as `fit_ets`."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    has_full_seasons = len(train) >= 2 * season_length
    seasonal_order = (1, 1, 0, season_length) if has_full_seasons else (0, 0, 0, 0)
    model = SARIMAX(
        train.to_numpy(),
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(horizon))


# ---------------------------------------------------------------------------
# 14.3.5 Events and covariates with Prophet
# ---------------------------------------------------------------------------


def _check_future_covariates(
    covariate_columns: list[str] | None, future_covariates: pd.DataFrame | None, horizon: int
) -> None:
    """Enforce the future-covariate rule: only planned values may condition a forecast.

    At forecast time only the planned promo calendar and the contracted
    access state are known, never realized actuals. This does not (and
    cannot, from a shape check alone) verify that the caller supplied
    planned rather than realized values; it enforces the structural half of
    the rule, that every requested covariate has exactly `horizon` future
    values supplied explicitly, so a forecast can never silently condition
    on the training data's own future rows.
    """
    if not covariate_columns:
        return
    if future_covariates is None:
        raise ValueError(
            "covariate_columns were requested but future_covariates is None; "
            "the future-covariate rule requires explicit planned values for "
            "every forecast horizon step, never realized actuals."
        )
    if len(future_covariates) != horizon:
        raise ValueError(
            f"future_covariates has {len(future_covariates)} rows, expected exactly "
            f"horizon={horizon}."
        )
    missing = set(covariate_columns) - set(future_covariates.columns)
    if missing:
        raise ValueError(f"future_covariates is missing columns: {sorted(missing)}")


def fit_prophet(
    train: pd.DataFrame,
    horizon: int,
    covariate_columns: list[str] | None = None,
    future_covariates: pd.DataFrame | None = None,
) -> np.ndarray:
    """Fit Prophet's additive decomposition with launch events, holidays, and planned covariates.

    `train` must have `ds` (date) and `y` (target) columns, plus any
    columns named in `covariate_columns`. `future_covariates` must supply
    exactly `horizon` rows of planned values for those same columns (the
    January insurance-reset and holiday effects are Prophet's built-in
    seasonality and need no explicit column).
    """
    from prophet import Prophet

    _check_future_covariates(covariate_columns, future_covariates, horizon)
    covariate_columns = covariate_columns or []

    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    for column in covariate_columns:
        model.add_regressor(column)
    model.fit(train[["ds", "y", *covariate_columns]])

    future = model.make_future_dataframe(periods=horizon, freq="W-MON", include_history=False)
    for column in covariate_columns:
        future[column] = future_covariates[column].to_numpy()  # type: ignore[index]

    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()


# ---------------------------------------------------------------------------
# 14.3.6 Covariate-driven ML: gradient-boosted trees
# ---------------------------------------------------------------------------


def _lag_feature_matrix(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a lag-feature design matrix and target vector from a 1-D series."""
    rows = []
    targets = []
    for t in range(n_lags, len(y)):
        rows.append(y[t - n_lags : t])
        targets.append(y[t])
    return np.array(rows), np.array(targets)


def fit_gbt(
    train: pd.DataFrame,
    horizon: int,
    covariate_columns: list[str] | None = None,
    future_covariates: pd.DataFrame | None = None,
    n_lags: int = 8,
) -> np.ndarray:
    """Fit a gradient-boosted-tree regressor on lag, calendar, and covariate features.

    `train` must have `ds`, `y`, and any `covariate_columns`.
    `future_covariates` must supply exactly `horizon` planned values for
    those columns (same future-covariate rule as `fit_prophet`). Forecasts
    recursively: each step's prediction becomes the next step's most recent
    lag, since future lags of the target are not observable.
    """
    import lightgbm as lgb

    _check_future_covariates(covariate_columns, future_covariates, horizon)
    covariate_columns = covariate_columns or []

    y = train["y"].to_numpy()
    week_of_year = train["ds"].dt.isocalendar().week.to_numpy()
    month = train["ds"].dt.month.to_numpy()

    lag_x, lag_y = _lag_feature_matrix(y, n_lags)
    calendar_x = np.column_stack([week_of_year[n_lags:], month[n_lags:]])
    covariate_x = train[covariate_columns].to_numpy()[n_lags:] if covariate_columns else np.empty((len(lag_y), 0))
    feature_x = np.column_stack([lag_x, calendar_x, covariate_x])

    model = lgb.LGBMRegressor(n_estimators=200, max_depth=3, min_child_samples=5, verbose=-1)
    model.fit(feature_x, lag_y)

    history = list(y[-n_lags:])
    last_date = train["ds"].iloc[-1]
    predictions = []
    for step in range(horizon):
        future_date = last_date + pd.Timedelta(weeks=step + 1)
        lag_features = np.array(history[-n_lags:])
        calendar_features = np.array([future_date.isocalendar().week, future_date.month])
        if covariate_columns:
            covariate_features = future_covariates.iloc[step][covariate_columns].to_numpy(dtype=float)  # type: ignore[union-attr]
        else:
            covariate_features = np.empty(0)
        row = np.concatenate([lag_features, calendar_features, covariate_features]).reshape(1, -1)
        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        history.append(prediction)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# 14.3.7 A supervised deep model: Temporal Fusion Transformer
# ---------------------------------------------------------------------------


def fit_tft(panel: pd.DataFrame, horizon: int) -> np.ndarray:
    """Train a Temporal Fusion Transformer across the territory panel and aggregate to a national forecast.

    `panel` must have `unique_id`, `ds`, and `y` columns (one row per
    territory-week). A single national series has too few weeks to train a
    deep model; pooling the 12 territory series gives the model enough
    total observations to fit, which is how deep models get workable
    sample sizes in practice. The per-territory forecasts are summed to a
    national total so this method can sit on the same scorecard as the
    national-level classical and foundation-model forecasts.

    Training is forced onto the CPU, single-threaded, with no dataloader
    worker processes. This chapter's development surfaced two real
    conflicts running Prophet (cmdstan), LightGBM, and PyTorch in one
    process on this platform: importing torch before LightGBM fits a model
    crashes the process outright (LightGBM and PyTorch each bundle a
    conflicting OpenMP runtime, so torch must always be imported after
    LightGBM has already run, which every function in this module already
    does), and even with that ordering fixed, PyTorch Lightning's default
    multi-threaded, multi-worker training hung indefinitely once cmdstan
    and LightGBM had already claimed threads. `torch.set_num_threads(1)`
    and `dataloader_kwargs={"num_workers": 0}` avoid the second conflict.
    Apple Silicon's MPS GPU backend also hung the one time two PyTorch
    processes tried to use it at once, which the CPU-only setting avoids.
    None of this costs meaningful time: the model is tiny and training
    finishes in a few seconds either way.
    """
    import torch

    torch.backends.mps.is_available = lambda: False
    torch.set_num_threads(1)

    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT

    input_size = min(TFT_INPUT_SIZE_WEEKS, panel.groupby("unique_id").size().min() - 1)
    model = TFT(
        h=horizon,
        input_size=max(input_size, 4),
        hidden_size=TFT_HIDDEN_SIZE,
        n_head=1,
        max_steps=TFT_MAX_STEPS,
        val_check_steps=TFT_MAX_STEPS + 1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        random_seed=1,
        dataloader_kwargs={"num_workers": 0},
    )
    nf = NeuralForecast(models=[model], freq="W-MON")
    nf.fit(df=panel)
    predictions = nf.predict()
    national = predictions.groupby("ds")["TFT"].sum().sort_index()
    return national.to_numpy()


# ---------------------------------------------------------------------------
# 14.3.8 Zero-shot foundation models: Chronos and TimesFM
# ---------------------------------------------------------------------------

_CHRONOS_PIPELINE_CACHE: dict[str, object] = {}
_TIMESFM_MODEL_CACHE: dict[str, object] = {}


def chronos_forecast(
    context: np.ndarray,
    horizon: int,
    model_id: str = CHRONOS_MODEL_ID,
    quantile_levels: tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Zero-shot probabilistic forecast with Chronos.

    Chronos scales the context series, tokenizes it into a fixed
    vocabulary, and forecasts the next tokens with a pretrained T5 language
    model, so no training happens here at all: `context` is only ever the
    history available up to the forecast origin. Returns a DataFrame with
    the low, median, and high quantiles requested.
    """
    import torch
    from chronos import BaseChronosPipeline

    if model_id not in _CHRONOS_PIPELINE_CACHE:
        _CHRONOS_PIPELINE_CACHE[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32
        )
    pipeline = _CHRONOS_PIPELINE_CACHE[model_id]

    context_tensor = torch.tensor(np.asarray(context, dtype=float), dtype=torch.float32)
    quantiles, mean = pipeline.predict_quantiles(  # type: ignore[attr-defined]
        context_tensor, prediction_length=horizon, quantile_levels=list(quantile_levels)
    )
    low, median, high = (quantiles[0, :, i].numpy() for i in range(3))
    return pd.DataFrame({"low": low, "median": median, "high": high, "mean": mean[0].numpy()})


def timesfm_forecast(
    context: np.ndarray, horizon: int, model_id: str = TIMESFM_MODEL_ID
) -> np.ndarray:
    """Zero-shot deterministic forecast with TimesFM, the patched-transformer contrast to Chronos.

    TimesFM operates on patches of the series directly (a numeric
    transformer, not a language-token model) and returns a single point
    forecast per horizon step rather than Chronos's sampled distribution.
    """
    import timesfm

    if model_id not in _TIMESFM_MODEL_CACHE:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=max(64, len(context)),
                max_horizon=max(horizon, 16),
                normalize_inputs=True,
                use_continuous_quantile_head=False,
            )
        )
        _TIMESFM_MODEL_CACHE[model_id] = model
    model = _TIMESFM_MODEL_CACHE[model_id]

    point_forecast, _ = model.forecast(horizon=horizon, inputs=[np.asarray(context, dtype=float)])  # type: ignore[attr-defined]
    return point_forecast[0]


# ---------------------------------------------------------------------------
# 14.3.9 The scorecard and honest intervals: conformal prediction
# ---------------------------------------------------------------------------


def conformal_interval(
    calibration_backtest: pd.DataFrame, point_forecast: np.ndarray, alpha: float = 0.20
) -> pd.DataFrame:
    """Calibrate a prediction interval from backtest residuals and apply it to a new forecast.

    Split conformal prediction: take the (1 - alpha) quantile of absolute
    residuals from a calibration backtest, and use that as a constant
    half-width around a new point forecast. Coverage is checked separately
    against held-out actuals, since the interval is only as honest as the
    calibration set it was built from.
    """
    residuals = (calibration_backtest["actual"] - calibration_backtest["predicted"]).abs()
    half_width = float(residuals.quantile(1.0 - alpha))
    return pd.DataFrame(
        {
            "point_forecast": point_forecast,
            "lower": point_forecast - half_width,
            "upper": point_forecast + half_width,
            "half_width": half_width,
        }
    )


def empirical_coverage(interval: pd.DataFrame, actual: np.ndarray) -> float:
    """Fraction of actuals that fall within [lower, upper]."""
    within = (actual >= interval["lower"].to_numpy()) & (actual <= interval["upper"].to_numpy())
    return float(np.mean(within))


# ---------------------------------------------------------------------------
# 14.4.2 Analog-based erosion curves
# ---------------------------------------------------------------------------


def _erosion_curve(
    weeks_since_entry: np.ndarray,
    pre_entry_reference: float,
    residual_fraction: float,
    half_life_weeks: float,
) -> np.ndarray:
    residual = residual_fraction * pre_entry_reference
    return residual + (pre_entry_reference - residual) * np.exp(
        -np.log(2) * weeks_since_entry / half_life_weeks
    )


def analog_erosion_forecast(
    weeks_since_entry: np.ndarray, pre_entry_reference: float, analog_name: str | None = None
) -> dict[str, object]:
    """Project the post-entry decline using a comparable molecule's erosion shape.

    Used before enough of the brand's own post-entry tail exists to fit a
    curve directly; defaults to the first entry in the analog library.
    """
    if analog_name is None:
        analog_name = next(iter(ANALOG_EROSIONS))
    params = ANALOG_EROSIONS[analog_name]
    projected = _erosion_curve(
        weeks_since_entry, pre_entry_reference, params["residual_fraction"], params["half_life_weeks"]
    )
    return {"analog_name": analog_name, "params": params, "projected": projected}


# ---------------------------------------------------------------------------
# 14.4.3 Parametric decline fit and Chronos cross-check
# ---------------------------------------------------------------------------


def fit_erosion(weeks_since_entry: np.ndarray, post_entry_values: np.ndarray) -> dict[str, float]:
    """Fit a half-life decay curve to the observed post-entry tail.

    Fits the residual fraction of the pre-entry reference level and the
    decay half-life by nonlinear least squares, using the first observed
    post-entry value as the pre-entry reference point.
    """
    pre_entry_reference = float(post_entry_values[0])

    def model(t, residual_fraction, half_life_weeks):
        return _erosion_curve(t, pre_entry_reference, residual_fraction, half_life_weeks)

    (residual_fraction, half_life_weeks), _ = curve_fit(
        model,
        weeks_since_entry,
        post_entry_values,
        p0=(0.15, 10.0),
        bounds=([0.001, 1.0], [0.99, 200.0]),
        maxfev=20_000,
    )
    fitted = model(weeks_since_entry, residual_fraction, half_life_weeks)
    return {
        "residual_fraction": float(residual_fraction),
        "half_life_weeks": float(half_life_weeks),
        "pre_entry_reference": pre_entry_reference,
        "fitted": fitted,
    }


# ---------------------------------------------------------------------------
# 14.5.1 Hierarchical forecasting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 14.5.2 Reconciliation
# ---------------------------------------------------------------------------


def _stack_base_forecasts(base_forecasts: dict[str, np.ndarray], region_order: list[str]) -> np.ndarray:
    return np.vstack([base_forecasts["National"], *[base_forecasts[name] for name in region_order]])


def reconcile(
    base_forecasts: dict[str, np.ndarray],
    region_order: list[str],
    method: str = "ols",
    historical_shares: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reconcile independent base forecasts into a coherent set that sums correctly.

    `method` is one of:
    - "bottom_up": trust the region forecasts; redefine the national total
      as their sum.
    - "top_down": trust the national forecast; split it across regions by
      `historical_shares`.
    - "ols": the MinT family's OLS-trace-minimization special case (equal
      error variance assumed across levels), which blends every level's
      own base forecast rather than fully trusting either the top or the
      bottom.

    Returns a DataFrame with one row per level (`National` first, then each
    region in `region_order`) and one column per horizon step.
    """
    stacked = _stack_base_forecasts(base_forecasts, region_order)
    horizon = stacked.shape[1]
    n_regions = len(region_order)

    if method == "bottom_up":
        region_rows = stacked[1:]
        national_row = region_rows.sum(axis=0, keepdims=True)
        reconciled = np.vstack([national_row, region_rows])
    elif method == "top_down":
        if historical_shares is None:
            raise ValueError("top_down reconciliation requires historical_shares")
        shares = np.array([historical_shares[name] for name in region_order]).reshape(-1, 1)
        national_row = stacked[0:1]
        region_rows = shares * national_row
        reconciled = np.vstack([national_row, region_rows])
    elif method == "ols":
        summing_matrix = np.vstack([np.ones((1, n_regions)), np.eye(n_regions)])
        projection = np.linalg.inv(summing_matrix.T @ summing_matrix) @ summing_matrix.T
        reconciled_bottom = projection @ stacked
        reconciled = summing_matrix @ reconciled_bottom
    else:
        raise ValueError(f"unknown reconciliation method: {method}")

    index = ["National", *region_order]
    return pd.DataFrame(reconciled, index=index, columns=[f"h{step + 1}" for step in range(horizon)])


# ---------------------------------------------------------------------------
# 14.5.3 From demand to supply signal
# ---------------------------------------------------------------------------


def demand_to_supply(
    reconciled_demand: np.ndarray,
    demand_std: np.ndarray,
    lead_time_weeks: float = SUPPLY_LEAD_TIME_WEEKS,
    service_level_z: float = SERVICE_LEVEL_Z,
) -> pd.DataFrame:
    """Translate reconciled dispensed demand (TRx) into a safety-stock band and an order signal.

    `reconciled_demand` and `demand_std` are dispensed-demand quantities
    (TRx), not ex-factory shipments; wholesaler inventory buffers the two,
    so forecasting the ex-factory shipment series directly from its own
    history amplifies every swing in the underlying dispensed demand (the
    bullwhip effect). Safety stock uses the standard service-level formula:
    z times the demand standard deviation scaled by the square root of the
    replenishment lead time.
    """
    safety_stock = service_level_z * demand_std * np.sqrt(lead_time_weeks)
    order_signal = reconciled_demand * lead_time_weeks + safety_stock
    return pd.DataFrame(
        {
            "reconciled_demand": reconciled_demand,
            "safety_stock": safety_stock,
            "order_signal": order_signal,
        }
    )


# ---------------------------------------------------------------------------
# 14.6.1 Ensemble the methods
# ---------------------------------------------------------------------------


def ensemble_consensus(
    base_forecasts: dict[str, np.ndarray], scorecard: pd.DataFrame | None = None
) -> np.ndarray:
    """Combine every method's forecast into one consensus, weighted by backtest accuracy.

    Weights are proportional to inverse MASE from `scorecard` (a more
    accurate method gets more say); without a scorecard, every method is
    weighted equally.
    """
    names = list(base_forecasts.keys())
    stacked = np.vstack([base_forecasts[name] for name in names])
    if scorecard is not None:
        mase_by_name = scorecard.set_index("method")["mase"]
        inverse_mase = np.array([1.0 / max(mase_by_name.get(name, 1.0), 1e-6) for name in names])
        weights = inverse_mase / inverse_mase.sum()
    else:
        weights = np.full(len(names), 1.0 / len(names))
    return weights @ stacked


# ---------------------------------------------------------------------------
# 14.6.2 Analytics-finance-commercial reconciliation
# ---------------------------------------------------------------------------


def consensus_reconcile(
    statistical_consensus_total: float, adjustments: dict[str, float]
) -> pd.DataFrame:
    """Apply named, quantified finance and commercial adjustments to the analytics consensus.

    `adjustments` maps a named driver (for example "Access assumption
    tightened", "Launch-support level confirmed", "Competitor entry risk")
    to a fractional delta applied in sequence, ending at the committed
    number. Every step is visible, so the gap between the analytics
    consensus and the committed number is always attributable to a named
    cause.
    """
    running_total = float(statistical_consensus_total)
    rows = [{"step": "Analytics consensus", "adjustment_pct": 0.0, "running_total": running_total}]
    for name, adjustment_pct in adjustments.items():
        running_total *= 1.0 + adjustment_pct
        rows.append({"step": name, "adjustment_pct": adjustment_pct, "running_total": running_total})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 14.6.3 Scenarios
# ---------------------------------------------------------------------------


def scenario_forecast(assumption_ranges: dict[str, dict[str, float]] = FUNNEL_ASSUMPTION_RANGES) -> pd.DataFrame:
    """Build base, low, and high scenarios from explicit driver assumptions, not from the interval alone.

    Each scenario states the addressable-patient, access, and brand-share
    assumption it depends on, so a reader can trace exactly which driver
    would have to move for the high case to happen.
    """
    low = patient_based_forecast(
        addressable_patients=assumption_ranges["addressable_patients"]["low"],
        access_adjustment=assumption_ranges["access_adjustment"]["low"],
        brand_share=assumption_ranges["brand_share"]["low"],
    )
    base = patient_based_forecast(
        addressable_patients=assumption_ranges["addressable_patients"]["base"],
        access_adjustment=assumption_ranges["access_adjustment"]["base"],
        brand_share=assumption_ranges["brand_share"]["base"],
    )
    high = patient_based_forecast(
        addressable_patients=assumption_ranges["addressable_patients"]["high"],
        access_adjustment=assumption_ranges["access_adjustment"]["high"],
        brand_share=assumption_ranges["brand_share"]["high"],
    )
    return pd.DataFrame(
        [{"scenario": "Low", **low}, {"scenario": "Base", **base}, {"scenario": "High", **high}]
    )
