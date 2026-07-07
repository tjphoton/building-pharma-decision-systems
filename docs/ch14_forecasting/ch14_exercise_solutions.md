# Chapter 14: Forecasting Exercise Solutions

These worked answers use the same Chapter 14 data and functions as the manuscript.



```python
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

```

    12:30:25 - cmdstanpy - INFO - Chain [1] start processing


    12:30:25 - cmdstanpy - INFO - Chain [1] done processing


    12:30:25 - cmdstanpy - INFO - Chain [1] start processing


    12:30:25 - cmdstanpy - INFO - Chain [1] done processing


    Seed set to 1


    GPU available: False, used: False


    TPU available: False, using: 0 TPU cores


    
      | Name                    | Type                     | Params | Mode 
    -----------------------------------------------------------------------------
    0 | loss                    | MAE                      | 0      | train
    1 | padder_train            | ConstantPad1d            | 0      | train
    2 | scaler                  | TemporalNorm             | 0      | train
    3 | embedding               | TFTEmbedding             | 128    | train
    4 | temporal_encoder        | TemporalCovariateEncoder | 39.6 K | train
    5 | temporal_fusion_decoder | TemporalFusionDecoder    | 18.0 K | train
    6 | output_adapter          | Linear                   | 33     | train
    -----------------------------------------------------------------------------
    57.8 K    Trainable params
    0         Non-trainable params
    57.8 K    Total params
    0.231     Total estimated model params size (MB)
    88        Modules in train mode
    0         Modules in eval mode


    `Trainer.fit` stopped: `max_steps=300` reached.


    GPU available: False, used: False


    TPU available: False, using: 0 TPU cores


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.


    Observed weeks: 52
    Full lifecycle weeks: 418


## Exercise 1



```python
weeks_since_launch = (observed["week_start"] - national["week_start"].iloc[0]).dt.days / 7.0
months_since_launch = (weeks_since_launch * 12.0 / 52.0).to_numpy()
cumulative_starts = observed["nbrx"].cumsum().to_numpy()
fit_26 = fit_bass(months_since_launch[:26], cumulative_starts[:26])
fit_52 = fit_bass(months_since_launch, cumulative_starts)
comparison = pd.DataFrame([fit_26, fit_52], index=["26 weeks", "52 weeks"])
print(comparison[["m", "time_to_peak_months"]].round(1))

```

                   m  time_to_peak_months
    26 weeks  3858.1                 10.1
    52 weeks  8470.4                 13.6


The 26-week fit is less stable because it sees only the early launch ramp. A finance team should treat it as a directional read, not a ceiling commitment.


## Exercise 2



```python
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

```

               method        mae     wmape      mape      mase
    0           naive  15.748579  0.195880  0.214511  1.000000
    1  seasonal_naive  15.748579  0.195880  0.214511  1.000000
    2          sarima  16.058999  0.199741  0.219651  1.019711
    3             ets  17.967512  0.223479  0.225823  1.140897


NBRx is a flow of new starts, while TRx is a stock-and-flow measure that accumulates persistent patients. A method can rank differently when the target is noisier and less cumulative.


## Exercise 3



```python
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

```

     window  residual_fraction  within_10_pct
         40              0.093           True


The practical judgment is to avoid committing the residual tail until enough post-entry data shows the curve flattening. Before that point, use analogs and a cross-check, and label the estimate provisional.

