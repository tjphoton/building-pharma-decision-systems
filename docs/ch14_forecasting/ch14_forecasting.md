# Chapter 14: Forecasting from Launch to Loss of Exclusivity

Roventra's launch team is working with Finance to put concrete numbers on a timeline before it will commit budget: how fast the brand ramps, how much it sells at peak, what next quarter looks like, how many units manufacturing should build, and what the business looks like once patent protection ends.

A new brand like Roventra, only a few months past approval, cannot support the seasonal forecasting models a mature product with years of history would use. Worse, the pre-launch assumptions no longer match what the first months of real prescribing show. Forecasting is the discipline that turns those mismatched assumptions into a number grounded in real prescribing, corrected again every time more data arrives, because the business cannot wait for the picture to become clear before it commits.

In this chapter, you will build forecasts for every stage of the Roventra lifecycle:

- size the pre-launch business case and fit the launch uptake curve once real data exists
- forecast in-market demand with classical statistical models, gradient-boosted trees, a deep model trained across territories, and pretrained time-series foundation models
- reconcile a demand forecast down to a region-and-territory supply signal
- fit and project the post-patent decline once generic competition enters
- combine every method into one governed number with base, low, and high scenarios

Open [`ch14_walkthrough.ipynb`](ch14_walkthrough.ipynb), or run the blocks below from the repository root.

> **Note:** Roventra's launch trajectory, generic-entry date, and territory structure are fictional and generated with known parameters, so every forecast in this chapter can be checked against the process that actually produced the data.

## 14.1 Forecasting Decisions

### 14.1.1 Forecasting Decisions

Each pharma forecasting decision runs on its own time horizon and its own methods. Table 14.1 lists five use cases this chapter builds.

*Table 14.1. Forecasting use cases by decision, horizon, and method.*

| Use case | Decision it serves | Horizon | Methods |
| --- | --- | --- | --- |
| Sizing the launch | Size the pre-launch demand ceiling and the launch ramp | Years, monthly | Patient-based funnel, persistence conversion, analogs, Bass diffusion, Monte Carlo |
| In-market demand | Field, supply, and quarterly finance planning | Weeks to a year | Baselines, ETS, SARIMA, Prophet, gradient-boosted trees, TFT, Chronos, TimesFM, conformal intervals |
| Demand-supply planning | Convert demand into a coherent regional forecast and a national order signal | Rolling operational | Hierarchical forecasting, reconciliation, safety stock |
| Loss of exclusivity | Plan the post-generic decline and its P&L | Years after generic entry | Analog-based erosion curves, a parametric decline fit, a foundation-model cross-check |
| Consensus and scenario | Commit one governed number across functions | Spans every horizon above | An accuracy-weighted ensemble, a named adjustment waterfall, and driver-based scenarios |

### 14.1.2 The Roventra Series

This chapter builds all five on one Roventra series, moving from launch planning to in-market demand, supply planning, lifecycle risk, and final consensus. `generate_forecast_data.py` produces the series, and `run_analysis.py` builds the result tables used by the listings.

![Figure 14.1. Roventra lifecycle series with launch, observed window, peak, and generic entry marked. Synthetic data.](assets/figures/figure_14_1_lifecycle_series.svg)

*Figure 14.1. Roventra lifecycle series with launch, observed window, peak, and generic entry marked. Synthetic data.*

**Listing 14.1**: Load the lifecycle series

```python
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
```

```
   week_start   nbrx    trx
47 2025-01-27  125.0  346.0
48 2025-02-03  101.1  380.0
49 2025-02-10   98.8  365.1
50 2025-02-17  112.9  413.1
51 2025-02-24  102.7  456.2
```

The generator in `generate_forecast_data.py` builds one weekly national demand series from launch through a simulated generic entry and a 2-year tail, using known Bass diffusion, persistence, and erosion parameters. `run_generation()` returns that full multi-year series as `national_series`, the chapter's ground truth, and a second table, `observed_series`, holding the same national numbers cut off at the current analysis date. The two hold the same national demand line at different lengths: `national_series` carries the complete, multi-year lifecycle; `observed_series` stops at the roughly 1 year of history a launch team actually has on hand right now. Every forecast through the in-market and demand-supply sections is built from `observed_series` alone. The loss-of-exclusivity section later reaches past that window: once the simulated generic entry has occurred, it reads the post-entry weeks directly from `national_series` to model a decline that happens years after `observed_series` ends. `national_series` exists to check each method's forecast against the process that generated it.


## 14.2 Sizing the Launch

### 14.2.1 The Patient-Based Funnel Forecast

Before a single prescription is written, the launch needs a demand ceiling built entirely from assumptions, since real prescribing data does not exist yet to build it from instead. The patient-based funnel starts from the addressable patient population, applies the access probability a payer mix implies, and applies an assumed peak brand share against named competitors, to produce a ceiling: the most patients the brand could plausibly have on therapy at once. It's a snapshot count, the peak number of patients simultaneously on therapy in a given week, reached once the launch ramp fully plays out.

`patient_based_forecast()` in `forecasting.py` multiplies these three factors and returns the ceiling. Listing 14.2 runs it with the launch team's stated assumptions.

**Listing 14.2**: The pre-launch funnel ceiling

```python
from forecasting import patient_based_forecast

business_case = patient_based_forecast()
print(pd.Series(business_case).to_string())
```

```
addressable_patients    20000.00
access_adjustment           0.78
brand_share                 0.30
ceiling                  4680.00
```

20,000 addressable patients, a 78% blended access probability under the mature payer mix, and a 30% peak brand share against Nexoral and Vexpro combine to a ceiling of 4,680 patients on therapy at once. Every factor is a stated assumption, not a measurement, and every one of those assumptions could land anywhere in a defensible range.

This ceiling is a fresh, chapter-specific scenario. 20,000 addressable patients is a top-down planning number the launch team is asserting before any real prescribing exists, the kind of judgment call a commercial team makes when an epidemiological funnel like the market-sizing chapter's is not yet the tool at hand. It is deliberately not derived from that funnel: the market-sizing chapter anchored Roventra's disease to an external prevalence source and moved from 29.2 million diagnosed patients to 24.9 million age-eligible, 8.1 million untreated, and a 4.2 million access-adjusted reachable population, converting to 1.0 million expected starts at a 25% conversion assumption. Those numbers size the national epidemiological opportunity from measured claims; 20,000 sizes a launch team's own planning assumption from judgment. In real pre-launch planning, before any real prescribing exists, this is exactly how many business cases get built: a top-down number the launch team assumes (informed by market research, analog launches, or leadership judgment), not a bottom-up count from claims. What does carry over is the access logic: the 78% blended access probability applies the same payer-tier weights the market-sizing funnel used, Covered 0.90, Covered with Step Edit 0.75, Covered with PA 0.65, and Non-covered 0.10, to this launch's own payer mix.

A ceiling this size supports a real business. It counts patients on therapy at one time: patients on Roventra refill roughly monthly for as long as they persist, across the drug's full 6 years of exclusivity, and new patients keep entering as others drop off, since a steady background incidence rate keeps adding newly diagnosed patients. The generated lifecycle carries 312,269 total prescriptions over those 6 years, averaging just under 1,000 a week and peaking near 1,500. At even a modest net price for a drug competing against 2 named rivals, that is tens of millions of dollars in cumulative revenue, a real niche brand, not a blockbuster.

`monte_carlo_funnel()` in `forecasting.py` draws the addressable population, access adjustment, and peak brand share each from a stated range and reruns the funnel calculation thousands of times to produce a distribution of ceilings. `assumption_tornado()` runs a single-factor sensitivity on top of that: it varies one assumption at a time across its own range, holding the others at their base case, and ranks the assumptions by how much the ceiling moves. The base case (4,680) sits close to the median of the simulated distribution (4,419), and the middle half of simulated outcomes spans 3,677 to 5,244.

### 14.2.2 From New Starts to Volume: the Persistence Conversion

A funnel ceiling counts patients, not prescriptions. Converting new-to-therapy starts (NBRx) into total prescriptions (TRx) requires one more step, a persistence curve. On-therapy stock is the number of patients still active in a given week, carried forward from every prior week's cohort net of its own dropout. 

Consider 3 weeks of new starts: 100 patients in week 1, 80 in week 2, 60 in week 3, and a persistence curve that keeps 90% of any cohort on therapy 1 week later and 75% 2 weeks later. Week 3's on-therapy stock is not just that week's 60 new starts: it also carries 90% of week 2's 80 starts (72 patients) and 75% of week 1's 100 starts (75 patients), for a stock of 207 patients from 240 cumulative starts. Every week layers in a new cohort while every earlier cohort keeps contributing its own, shrinking share.

`persistence_to_trx()` in `forecasting.py` convolves a weekly new-start series with a Weibull persistence-survival curve to produce that on-therapy stock, then applies a refill rate to convert it into TRx. `PERSISTENCE_SHAPE` and `PERSISTENCE_SCALE_MONTHS` in `forecast_config.py` are fit directly to the patient-journey chapter's Kaplan-Meier line-1 persistence curve, by nonlinear least squares on its day-60, day-90, and day-113 points, excluding its noisy day-180 point (only 50 of 3,415 patients still at risk in that chapter). The fitted curve has an increasing hazard rate: patients who make it past the first few months on therapy become more likely to stop.

The fit reproduces the measured curve closely: 73.4% still on the initial regimen at day 60 against a measured 73.0%, 59.8% at day 90 against 60.6%, and a median time to departure of 114 days against a measured 113. This is the patient-journey chapter's retention curve carried forward into the national demand series.

**Listing 14.3**: Reconstructing on-therapy stock from observed NBRx

```python
from forecasting import persistence_to_trx

reconstructed = persistence_to_trx(observed["nbrx"].to_numpy())
reconstructed["cumulative_nbrx"] = reconstructed["nbrx"].cumsum()
reconstructed = reconstructed[["nbrx", "cumulative_nbrx", "on_therapy_stock"]]
print(reconstructed.tail().round(1))
```

```
     nbrx  cumulative_nbrx  on_therapy_stock
47  125.0           2683.8            1541.1
48  101.1           2784.9            1576.4
49   98.8           2883.7            1607.1
50  112.9           2996.6            1650.1
51  102.7           3099.3            1680.9
```

By the 52nd week, the on-therapy stock (1,681 patients) is running well ahead of that same week's new starts (103), because the convolution accumulates every prior week's cohort net of its own dropout. Figure 14.2 plots cumulative new starts (3,099 by week 52) and on-therapy stock over the observed window.

![Figure 14.2. Cumulative new starts and on-therapy stock over the observed window.](assets/figures/figure_14_2_funnel_timeline.svg)

*Figure 14.2. Cumulative new starts and on-therapy stock over the observed window.*

Both curves are still well short of the 4,680-patient business-case ceiling at the end of the observed window. The next two sections test whether that ceiling is the right one to climb toward.

### 14.2.3 The Launch Uptake Curve: Bass Diffusion

A launch does not ramp linearly. Word of a new option spreads two ways: some prescribers act on outside information such as approval coverage or a conference presentation, and others act because a peer prescriber is already using the drug. The Bass diffusion model ([Bass, 1969](https://doi.org/10.1287/mnsc.15.5.215)) captures both effects in one S-shaped adoption curve: slow at first, a steep acceleration through peer imitation, then a flattening ceiling as the addressable population runs out.

Toy case: two prescribers, two adoption mechanisms. Consider a small population of eligible prescribers. If innovation alone drove adoption, a constant fraction of the *remaining* non-adopters would start each month regardless of how many prescribers already use the drug. If imitation alone drove it, the adoption rate would depend on how many prescribers already use the drug, which is zero at the very start and cannot get going without an initial nudge. Real adoption blends both: the innovation coefficient `p` seeds the first prescribers, and the imitation coefficient `q` accelerates adoption once enough peers have started.

Figure 14.3 shows how the innovation and imitation coefficients each produce a different adoption shape.

![Figure 14.3. Prescriber innovation, imitation, and blended adoption shapes.](assets/figures/figure_14_3_bass_toy_shapes.svg)

*Figure 14.3. Prescriber innovation, imitation, and blended adoption shapes.*

The closed-form cumulative adoption fraction is:

$$
F(t)=\frac{1-e^{(-(p+q)t}}
{1+\frac{q}{p}e^{(-(p+q)t}}
$$

$t$ is time since launch in months, $p$ and $q$ are the innovation and imitation coefficients, and multiplying $F(t)$ by the ceiling $m$ gives cumulative adopters at time $t$. `bass_cumulative_fraction()` in `forecasting.py` implements this formula. The analog library in the next section, and Roventra's own fit in the section after that, both use this same formula, just with different $p$ and $q$.

### 14.2.4 Analog-Based Forecasting

Before any Roventra prescribing data exists at all, a forecaster has to borrow. Analog-based forecasting projects a new launch's trajectory from a comparable molecule that has already gone through its own launch, whether that molecule is a direct competitor, an earlier product from the same company, or simply a drug that reached a similar prescriber base, and uses that molecule's observed adoption shape as a stand-in for a brand with no history of its own yet. It is because the funnel's ceiling is a single number with no timeline attached to it: an analog gives that ceiling a shape over time, fast or slow, specialist-driven or primary-care-driven, before Roventra has generated enough of its own data to fit a shape directly. Real pharma teams usually pull these shapes from syndicated industry prescribing data (like IQVIA) precisely so the comparison is not limited to the company's own past launches. As real prescribing arrives, the same tool checks which analog the brand actually resembles; the next section replaces the analog altogether once there is enough real history to fit the brand's own curve.

`analog_forecast()` in `forecasting.py` holds a small library of two comparable historical launches, `Comparable A (fast KOL-driven uptake)` and `Comparable B (slower primary-care-driven uptake)`, each with its own Bass diffusion shape using the p and q coefficients from the previous section. Before any brand data exists, the function defaults to the first analog in the library, Comparable A here, an arbitrary tie-break with no data to break it, not a claim that fast adoption is the more likely case. Once early uptake exists, it converts elapsed weeks to months and compares Roventra's normalized cumulative starts against each analog's own normalized shape at the same elapsed time, keeping the lower-error match.

| Analog | p (innovation) | q (imitation) | What it represents |
| --- | --- | --- | --- |
| Comparable A (fast KOL-driven uptake) | 0.030 | 0.35 | A concentrated specialist market where a small set of influential prescribers drive rapid peer adoption; both outside-information and peer-driven uptake are strong. |
| Comparable B (slower primary-care-driven uptake) | 0.005 | 0.08 | A broad primary-care market with many prescribers and weaker peer influence; adoption builds gradually through ordinary promotion and awareness. |

Figure 14.4 compares those 2 shapes before any selection happens. Comparable A rises fast and saturates early. Comparable B builds much more slowly.

![Figure 14.4. Comparable A and Comparable B normalized adoption shapes over 60 months, before selection.](assets/figures/figure_14_4_analog_shapes.svg)

*Figure 14.4. Comparable A and Comparable B normalized adoption shapes over 60 months, before selection.*

| Week | Month | Roventra actual | Comparable A projected | Comparable B projected |
| --- | --- | --- | --- | --- |
| 1 | 0.2 | 0.000 | 0.007 | 0.001 |
| 9 | 2.1 | 0.038 | 0.087 | 0.011 |
| 17 | 3.9 | 0.091 | 0.214 | 0.023 |
| 26 | 6.0 | 0.165 | 0.409 | 0.038 |
| 35 | 8.1 | 0.300 | 0.618 | 0.055 |
| 43 | 9.9 | 0.448 | 0.770 | 0.072 |
| 52 | 12.0 | 0.662 | 0.882 | 0.094 |

![Figure 14.5. Roventra's normalized uptake overlaid against both analog curves at 26 weeks (left) and 52 weeks (right).](assets/figures/figure_14_5_analog_selection_zoom.svg)

*Figure 14.5. Roventra's normalized uptake overlaid against both analog curves at 26 weeks (left) and 52 weeks (right).*

Against the 4,680-patient business-case ceiling, Roventra's first 26 weeks of real uptake look closer to the slower comparable B launch than the fast one: Comparable B's mean squared error over that window is 0.004, against 0.016 for Comparable A.

Comparable B's early lead does not hold. By week 52, Roventra has reached 66.2% of the pre-launch ceiling, far past what Comparable B's slow shape ever predicted and now tracking closer, though still not exactly, to Comparable A's faster climb. Over the full 52 weeks, Comparable A's mean squared error is 0.051 against Comparable B's 0.066: A now wins, reversing the 26 week read.

Figure 14.5 puts each checkpoint on its own panel, both drawn to the same 0–80-week, 0–1 scale so the two are directly comparable. The left panel, Roventra's first 26 weeks, sits almost on top of Comparable B's slow climb. The right panel, the same curve extended to 52 weeks, has pulled up and away from B toward Comparable A's steeper path.

That reversal is not a clean read of Roventra's own shape, because it depends on the 4,680-patient ceiling used to convert raw counts into the "share of ceiling" the comparison runs on. Re-run the same 52-week comparison against 8,470, the ceiling the Bass fit in the next section recovers directly from this same data, and the reversal disappears entirely: Comparable B wins at both 26 and 52 weeks, with no crossover. A ceiling assumed too low inflates how far along its own S-curve Roventra appears to be, which mechanically favors whichever analog rises steepest early; a larger assumed ceiling does the opposite. Over a window that has not yet shown the curve bending toward its own inflection point, a steep climb toward a low ceiling and a gentle climb toward a high one produce nearly identical early trajectories, so normalized-fraction matching cannot separate shape from scale until the ceiling itself is pinned down. `analog_forecast()` picks whichever analog matches the data it has been given, but which analog that is can flip from a scale assumption alone, not just from six more months of data arriving. The trustworthy conclusion at week 52 is not "A now wins" but "the analog read no longer means much, because it rests on a ceiling the data is starting to contradict", which is exactly why the next section stops assuming the ceiling and fits it instead.

### 14.2.5 Fitting the Launch Curve to Roventra's Data

With 52 weeks of real prescribing now in hand, Roventra no longer needs to borrow anyone else's shape. `fit_bass()` fits the same formula from two sections ago directly to Roventra's own cumulative new-start series by nonlinear least squares, recovering the brand's own $p$, $q$, and ceiling instead of assuming them from an analog.

**Listing 14.4**: Fitting Bass diffusion to the observed launch data

```python
from forecasting import fit_bass

weeks_since_launch = (observed["week_start"] - national["week_start"].iloc[0]).dt.days / 7.0
months_since_launch = (weeks_since_launch * 12.0 / 52.0).to_numpy()
cumulative_starts = observed["nbrx"].cumsum().to_numpy()
bass_fit = fit_bass(months_since_launch, cumulative_starts)
print(pd.Series(bass_fit).round(3).to_string())
```

```
p                             0.008
q                             0.244
m                          8470.418
time_to_peak_months          13.557
peak_monthly_new_starts     550.468
```

Figure 14.6 plots the fit against Roventra's observed cumulative NBRx, its first 12 months of real history, and projects the curve out to month 20.

![Figure 14.6. Fitted Bass adoption curve against Roventra's observed cumulative NBRx, projected to month 20.](assets/figures/figure_14_6_bass_fit.svg)

*Figure 14.6. Fitted Bass adoption curve against Roventra's observed cumulative NBRx, projected to month 20.*

This figure supports the budget action: set budget and staffing against the fitted ramp and peak month.

q (0.244) is about 30 times p (0.008). p measures how many prescribers start from outside information alone, with no peers to copy yet; q measures how much each new start pulls in the next one. A q this far above p means peer prescribing, not outside promotion, is doing almost all the work once the first few prescribers commit. The fitted ceiling, 8,470 patients, is not close to the 4,680-patient pre-launch funnel ceiling, and sits above even the high end of the funnel's own simulated uncertainty range (5,244): real early uptake implies a ceiling 81% higher than the pre-launch plan assumed. That 81% gap means the budget, staffing, and supply plans built from the 4,680-patient business case should be rebuilt from the fitted ceiling. Once real uptake data exists, the fitted ceiling supersedes the funnel's assumption-only estimate, and 8,470 becomes the new planning number until the next re-fit. That next re-fit is necessary, because a Bass ceiling estimated from only a few months of data can move substantially as more weeks accumulate, so this number is the best current read, not a permanent commitment, and gets revisited on the same cadence as the rest of the forecast.

## 14.3 In-Market Demand

Section 14.2 produced a peak-patient ceiling and a Bass curve for the multi-year shape of the ramp toward it. Operations now needs a different number, at a different timescale: expected TRx over the next 8 weeks, to plan field coverage, supply, and the quarterly forecast finance reports. The observed series has 52 weeks of history, just one calendar year and not a full two, which turns out affects almost every method in this section.

Figure 14.7 sets up the problem: 44 weeks train every method in this section, and the last 8, shaded, are held out and scored against, never seen during fitting. Every figure in the rest of 14.3 reuses this same chart, swapping in one method's forecast over that shaded window, so the reader watches each method's line move closer to (or further from) the real held-out actuals as the section goes.

![Figure 14.7. 52 weeks of observed prescribing: 44 weeks to train on, the last 8, shaded, held out and scored against.](assets/figures/figure_14_7_opening_window.svg)

*Figure 14.7. 52 weeks of observed prescribing: 44 weeks to train on, the last 8, shaded, held out and scored against.*

### 14.3.1 Baselines to Beat: Naive and Seasonal-Naive

Every forecasting method in this chapter has to clear a bar: it has to beat the two simplest possible forecasts. `naive_forecast()` in `forecasting.py` repeats the last observed value for every step of the horizon. `seasonal_naive()` repeats the value from one full season ago, and falls back to the plain naive forecast when the training window does not yet cover a full season, which is exactly the case here: 52 weeks is one season, not two, so `seasonal_naive` has no prior season to reach back into and collapses to `naive_forecast` for the entire chapter.

Figure 14.8 puts that flat repeated-last-value line against the same held-out weeks. Roventra keeps climbing through the shaded window; naive does not move at all, which is exactly the gap every method after this one is trying to close.

![Figure 14.8. The naive forecast against the held-out weeks: a flat line against a still-climbing launch.](assets/figures/figure_14_8_baseline_naive.svg)

*Figure 14.8. The naive forecast against the held-out weeks: a flat line against a still-climbing launch.*

### 14.3.2 How We Score: Rolling-Origin Backtest and Error Metrics

A single train-test split tests one moment in time. A method could win by luck on that one split and lose on every other week of the year. What a forecaster actually needs is a method's error measured repeatedly, at several different points in the series, always under the rule that it never sees the future it is being asked to predict.

Rolling-origin backtesting does this by moving the forecast origin backward through the series in steps. At each origin, the method trains on everything strictly before that point and is scored on the fixed-length window immediately after it. Moving the origin back one horizon length at a time produces several independent folds from a single series, each one a small rehearsal of the real forecasting task.

`rolling_origin_backtest()` in `forecasting.py` implements this directly: given a series, a horizon, and a number of folds, it walks the test window backward fold by fold, and for each fold, trains every candidate method only on the data strictly before that fold's test window begins.

![Figure 14.9. Four backtest folds, each training on the gray span and testing on the darker span that immediately follows it.](assets/figures/figure_14_9_backtest_schematic.svg)

*Figure 14.9. Four backtest folds, each training on the gray span and testing on the darker span that immediately follows it.*

A backtest produces a pile of errors, one per method per fold per horizon step. Turning that pile into one metric a business can compare across methods, and that metric here is MASE, the mean absolute scaled error.

Consider a toy series of 6 actual values, 100, 105, 98, 110, 102, 108, and two candidate forecasts for the last 3 of them: a naive method that just repeats 110 (the last training value) three times, and a method that predicts 104, 106, 109. The naive method's errors are |102-110|=8, |108-110|=2, and whatever the sixth point would add; the candidate method's errors are |102-104|=2, |108-106|=2. Averaging each gives a mean absolute error (MAE) for each method. MASE takes the candidate's MAE and divides it by the seasonal-naive baseline's MAE computed on the exact same folds:

$$
\mathrm{MASE}=
\frac{\sum_t|y_t-\hat y_t|}
{\sum_t|y_t-\tilde y_t|}
$$

$y_t$ is the actual value at each held-out point, $\hat y_t$ is the method's forecast, and $\tilde y_t$ is the seasonal-naive forecast for the same point, computed across the same folds.

A MASE of 0.5 means the method's average error is half the seasonal-naive baseline's; a MASE of 1.5 means the method is 50% worse than just repeating last year. The scale cancels out of the ratio, which is what makes MASE comparable across series measured in wildly different units, TRx counts here, dollars or units elsewhere, without any further adjustment.

`accuracy_scorecard()` in `forecasting.py` computes MASE this way alongside mean absolute error, weighted mean absolute percentage error, and mean absolute percentage error, for every method in a backtest at once. A MASE below 1.0 means the method beats the baseline; a MASE above 1.0 means a forecaster would have done better repeating last year's number. The scorecard shows that several methods miss that bar.

### 14.3.3 Smoothed Level and Trend: Exponential Smoothing (ETS)

The naive forecast repeats last week's value exactly. That throws away everything the series showed in every earlier week except the very last one. A method that instead lets every past observation vote on the forecast, with recent observations voting more heavily than old ones, should do better whenever the series has a smooth underlying level rather than pure noise.

Exponential smoothing does exactly that. The forecast for next period is the current smoothed level, and the level itself updates every period as a weighted blend of the newest observation and the previous smoothed level:

$$
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}
$$

$\ell_t$ is the smoothed level after observing period $t$, $y_t$ is the actual observation at period $t$, and $\alpha$, between 0 and 1, controls how much weight the newest point gets. An $\alpha$ near 1 makes the smoothed level track the raw series closely, almost like the naive forecast; an $\alpha$ near 0 makes it barely move, averaging over a long history. The name "exponential" comes from writing this update out recursively: the weight on an observation $k$ periods in the past shrinks by a factor of $(1-\alpha)$ for every additional period back, an exponentially decaying weighted average of the whole history rather than a hard cutoff.

Holt's extension adds a second smoothed state for trend, tracking both the current series level and its rate of change. Holt-Winters adds a third state for a repeating seasonal pattern on top of both. `fit_ets()` in `forecasting.py` fits this three-state model with a trend component always included, but it falls back to a non-seasonal fit whenever the training window covers fewer than 2 full seasons: a 52-week seasonal component needs roughly 2 years of weekly data before the model can distinguish "this week is always higher" from "the series happened to be rising this year." The 52-week Roventra window never clears that bar. The scorecard still ranks ETS first, because a well-fit level and trend captures most of what this launch-stage series does: a rising level, with no seasonal component to lose.

Figure 14.10 shows why: a level-plus-trend line tracks the held-out weeks closely enough that the gap left open by naive in Figure 14.8 is mostly closed.

![Figure 14.10. The ETS forecast tracks the held-out weeks closely, level and trend alone.](assets/figures/figure_14_10_ets_forecast.svg)

*Figure 14.10. The ETS forecast tracks the held-out weeks closely, level and trend alone.*

### 14.3.4 Autoregression on Differenced History: ARIMA and SARIMA

Exponential smoothing describes a series through smoothed states: a level, a trend. ARIMA, short for AutoRegressive Integrated Moving Average, describes the same kind of series a different way: through its own recent history and its own recent forecast errors, after a preprocessing step that strips out any trend first.

A series is stationary when its mean and variance do not drift over time. Roventra's rising TRx clearly drifts, so ARIMA's first step, the "I" for integrated, replaces the raw series with its week-over-week change, $d_t = y_t - y_{t-1}$, and differences again if a trend still shows up in $d_t$ itself. A model that differences $d$ times is described as having order $d$.

The "AR" for autoregressive term then predicts each differenced value from a fixed number of its own recent lagged values, the way a linear regression predicts today's change in TRx from yesterday's change and the day before's:

$$
d_t=\phi_1 d_{t-1}+\phi_2 d_{t-2}+\epsilon_t
$$

$d_t$ is the differenced series at time $t$, $d_{t-1}$ and $d_{t-2}$ are its 2 most recent values, $\phi_1$ and $\phi_2$ are fitted coefficients, and $\epsilon_t$ is the leftover error the AR term does not explain. This example looks back 2 lags; an ARIMA model that looks back $p$ AR lags, differences $d$ times, and uses $q$ MA terms (next) is written ARIMA($p$, $d$, $q$).

The "MA" for moving average term adds a second correction on top of the AR term, built from the model's own past forecast errors rather than the series' own past values: if last week's forecast came in 10 units low, the MA term nudges this week's forecast up to partly correct for that same kind of miss happening again. AR corrects from where the series has been; MA corrects from how wrong the model just was.

SARIMA is seasonal ARIMA: it adds a second copy of that same AR-I-MA structure at a lag equal to the season length instead of 1 week. A seasonal AR term looks back 52 weeks instead of 1, a seasonal difference removes a repeating yearly pattern instead of a short-term trend, and a seasonal MA term corrects from last year's same-week error instead of last week's.

`fit_sarima()` in `forecasting.py` fits a SARIMA model, with the same short-history fallback to a non-seasonal order that `fit_ets` uses.

Figure 14.11 SARIMA's forecast tracks the held-out weeks reasonably well.

![Figure 14.11. SARIMA against the held-out weeks: a reasonable track that drifts slightly high late in the window.](assets/figures/figure_14_11_sarima_forecast.svg)

*Figure 14.11. SARIMA against the held-out weeks: a reasonable track that drifts slightly high late in the window.*

### 14.3.5 Additive Decomposition with Prophet

The naive, ETS, and SARIMA forecasts see only the target series. A forecaster who knows a formulary win landed on a known date, or that a promotional flight is scheduled for specific weeks, has information those methods cannot use, because they accept only the history of the target itself.

Prophet is an open-source forecasting library Meta released in 2017 and still maintains, built so that an analyst without a statistics background could get a reasonable forecast out of a messy business time series and hand it named events, a holiday, a launch, a marketing push, rather than hand-tune an ARIMA order. Its approach is to decompose the series additively into named, interpretable pieces and fit them together:

$$
y(t)=g(t)+s(t)+\sum_k r_k(t)+\epsilon_t
$$

$g(t)$ is the trend, $s(t)$ is the seasonal component, $r_k(t)$ is the $k$-th named regressor (the promotional flight, the access state) at time $t$, and $\epsilon_t$ is the leftover error the additive pieces do not explain. A covariate, in this decomposition, is one of those $r_k(t)$ terms: a separate, named input series, known or planned in advance, whose value at each time $t$ gets its own fitted coefficient and adds straight into the forecast alongside the trend and seasonality.

The trend term is a piecewise-linear curve that can bend at a small number of automatically detected changepoints, rather than a single straight line for the whole history. The seasonality term is a Fourier sum, a handful of sine and cosine waves at the yearly and weekly frequencies, fit as coefficients. Every named regressor, added with `add_regressor`, contributes its own coefficient times its value at time `t`. Because every piece is additive, a forecaster can read off how much of the forecast came from the trend, how much from the calendar, and how much from each named driver.

`fit_prophet()` in `forecasting.py` fits this decomposition and, when covariate columns are named, adds each one as a regressor. Under the hood it uses `Prophet` from the `prophet` package with `weekly_seasonality=False` and `daily_seasonality=False`, then calls `add_regressor()` for each planned driver before fitting on the `ds`, `y`, and covariate columns. `model.predict(future)` returns a DataFrame with one row per horizon step and a column for every additive piece: `yhat` (the point forecast), `yhat_lower`, `yhat_upper`, `trend`, `weekly`, `yearly`, and one column per named regressor. At forecast time, only the *planned* value of a covariate is knowable: the promotional calendar and the contracted access state are business decisions made in advance, not measurements of the future, so the regressor's value for a future week has to come from the plan, not from the actual outcome, since that outcome has not happened yet and cannot be known.

**Listing 14.5**: Prophet with and without the access and promotion covariates

```python
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
```

```
[339.4 350.8 362.1]
[308.7 319.2 329.7]
```

Telling Prophet about the planned access and promotion state shifts the first three weeks of forecast down by about 30 to 32 units, and it is tempting to read that as a straightforward improvement: more information in, a lower and presumably better-calibrated number out. It is not the regressors themselves doing that: access contributes a steady +7.6 across the held-out weeks and promo a steady -7.8, and those two nearly cancel. What actually moves is the trend term underneath. Without the covariates, Prophet has no way to explain the bump in weeks 31 to 43, when the promotional flight actually ran, other than by bending its own trend line steeper right around then, and that artificially steep trend is what carries forward into the holdout. With the covariates, Prophet correctly attributes that bump to the promotional regressor instead, leaving a trend shallower. The trouble is that the shallower trend is only more correct about the past: Roventra's launch is still genuinely accelerating through this window on its own Bass-diffusion curve, a force Prophet has no term for at all, and the artificially steep trend the no-covariates version leaned on happened to track that real acceleration reasonably well, for the wrong reason. More information does not fail here because it is wrong; it fails because correcting one real effect strips out an accidental proxy for a second, unmodeled one, and the forecast loses more than it gains.

Figure 14.12 puts both variants against the held-out weeks at once. The no-covariates line and the with-covariates line stay a consistent gap apart across the whole window, the same shift Listing 14.5 measures in the first three weeks.

![Figure 14.12. Prophet with and without the planned access and promotion covariates, against the held-out weeks.](assets/figures/figure_14_12_prophet_forecast.svg)

*Figure 14.12. Prophet with and without the planned access and promotion covariates, against the held-out weeks.*

### 14.3.6 Lag-Feature Machine Learning: Gradient-Boosted Trees

Prophet decides in advance what shape the trend and seasonality can take. A tabular machine-learning model makes no such assumption: it learns whatever relationship the training rows show between a set of input features and the target, however nonlinear, at the cost of needing enough rows to learn it from.

The features have to be engineered by hand, since a tree model has no built-in concept of time. `fit_gbt()` in `forecasting.py` builds a lag-and-calendar feature table: for every training week, 8 columns holding TRx from 1 to 8 weeks earlier, plus the week-of-year and month, and, when covariate columns are supplied, the same planned access and promotion values Prophet uses. Each row becomes one training example: given last week's value, the week before that, and so on, predict this week's value. A gradient-boosted forest of decision trees then learns to split on whichever lag or calendar feature best separates high-TRx weeks from low-TRx weeks, adding trees one at a time to correct the previous trees' remaining error.

Forecasting more than one step ahead with a lag-based model has a structural wrinkle a Prophet-style decomposition does not: the model needs last week's actual value as a feature, but for anything past the first forecast step, last week's actual value has not happened yet. `fit_gbt` handles this recursively, feeding each step's own prediction back in as the next step's most recent lag, so an early error compounds forward through the rest of the horizon instead of staying contained to one step. `fit_gbt()` uses `LGBMRegressor` from the `lightgbm` package with 200 trees, a maximum depth of 3, and a minimum child-sample count of 5. Unlike Prophet's single batched call, `model.predict(row)` here takes one row of lag, calendar, and covariate features and returns one float; `fit_gbt` calls it 8 times, once per horizon step, building the forecast up one scalar at a time as each prediction feeds the next step's lag.

With only about 36 usable training rows once 8 lags are removed from a 44-week training window, the model has too little data to learn much beyond the series average, and the forecast comes out nearly flat across the horizon. That is what happens when a method that typically wants hundreds of rows has to work with 3 dozen. The scorecard confirms it: gradient-boosted trees finish last.

Figure 14.13 shows that flatness: the forecast barely moves off the last training value while the actual series keeps climbing away from it.

![Figure 14.13. Gradient-boosted trees against the held-out weeks: a nearly flat line against a still-climbing launch.](assets/figures/figure_14_13_gbt_forecast.svg)

*Figure 14.13. Gradient-boosted trees against the held-out weeks: a nearly flat line against a still-climbing launch.*

### 14.3.7 A Supervised Deep Model: Temporal Fusion Transformer

Gradient-boosted trees learn from rows of hand-engineered lag features. The Temporal Fusion Transformer (TFT) is a deep-learning architecture Google Research published in 2019, built specifically for multi-horizon time series forecasting. It borrows the attention mechanism that also powers language models like GPT, but applies that mechanism to past time steps and input variables instead of words: at every forecast step, the network learns which past weeks and which input variables deserve the most weight, rather than a human deciding in advance that lag 1 through lag 8 are the features that matter. Attention itself is a general-purpose way to learn a weighted combination of a set of inputs; nothing about it is specific to text, which is exactly what lets the same mechanism work on time steps here.

That flexibility costs data. A transformer with attention layers and learned variable weights has far more parameters to fit than a handful of Bass or Holt-Winters coefficients, and a single national series of 52 weeks is nowhere near enough to fit them reliably. `fit_tft()` in `forecasting.py` works around that by training across the territory panel: 12 territories, each with the same 44-to-52 weeks of history, pool into roughly 500 to 600 training rows total. The model learns one shared set of weights across all 12 series, which is how deep models reach a workable sample size in commercial forecasting practice, borrowing statistical strength across many related but individually short series rather than demanding one long one. The per-territory forecasts are then summed back to a national total for the shared scorecard.

**Listing 14.6**: Training the TFT across the territory panel

```python
holdout = 8
tft_forecast = results["holdout_forecasts"]["tft"].to_numpy()
print(tft_forecast.round(1))
```

```
[359.9 387.1 412.5 433.9 456.5 484.3 510.5 535.4]
```

Pooling territories gets the model trained, but it does not make the model right: the TFT forecast keeps climbing across the horizon well past where the actual series goes, and it finishes behind even the naive baseline. A deep model trained on 500 short, noisy series is not automatically better than a well-specified classical model fit to one clean series; here it is worse.

Figure 14.14 shows the overshoot directly: the TFT line pulls away from the actual series almost immediately and never comes back.

![Figure 14.14. The Temporal Fusion Transformer overshoots the held-out weeks and keeps climbing.](assets/figures/figure_14_14_tft_forecast.svg)

*Figure 14.14. The Temporal Fusion Transformer overshoots the held-out weeks and keeps climbing.*

`neuralforecast`, the open-source Nixtla package this code depends on, is a library of published forecasting architectures behind one shared interface. `neuralforecast.models.TFT` is Nixtla's ready-to-train implementation of it, sitting alongside its implementations of other published architectures like N-BEATS and DeepAR. `fit_tft()` trains that `TFT` class inside a `NeuralForecast` wrapper, with horizon `h=8`, an input window capped by `TFT_INPUT_SIZE_WEEKS`, hidden size `TFT_HIDDEN_SIZE`, and `max_steps=TFT_MAX_STEPS`. `nf.fit(df=panel)` trains on a table with `unique_id`, `ds`, and `y` columns, one row per territory-week, the schema every `neuralforecast` model shares; `nf.predict()` returns one row per territory per horizon step with a column named after the model, `TFT`, which `fit_tft` sums across territories to get the national total shown above.

### 14.3.8 Zero-Shot Foundation Models: Chronos and TimesFM

Every method so far needs history to fit: ETS and SARIMA need 2 full seasons they do not have, Prophet and gradient-boosted trees need enough rows to learn a pattern, and the TFT needs a whole territory panel just to reach a workable sample size. A brand 8 to 12 weeks past launch has none of that, and waiting a year to get it is not a real option when finance needs next quarter's number now. Time-series foundation models exist for exactly this cold-start problem.

They follow the same recipe that worked first for language and then for images: pretrain one large model, once, on a broad, heterogeneous pile of series, public datasets spanning finance, web traffic, weather, and energy demand, plus large amounts of synthetic data, so it has already seen enough different trends, seasonal shapes, and shocks to forecast a brand-new series directly from the context handed to it at inference time, no fine-tuning and no training step of its own required. Amazon released Chronos and Google released TimesFM, both in 2024.

Chronos treats forecasting as a language-modeling problem, the same problem a large language model solves for text. A language model does not see words directly; it sees a sequence of tokens from a fixed vocabulary and learns to predict which token comes next. Chronos applies that idea to numbers: it scales the context window (so absolute price or volume level does not confuse a model trained on many different series in many different units), quantizes the scaled values into a few thousand discrete bins, the model's vocabulary, and predicts the next bin the same way a language model predicts the next word, using a pretrained T5 encoder-decoder architecture. Sampling many possible continuations from that trained model, the way a language model can generate many different plausible sentences from the same prompt, produces not one forecast but a distribution of them. Because the underlying object is a distribution over tokens rather than a single number, Chronos's output is naturally probabilistic with no separate calibration step: `chronos_forecast()` in `forecasting.py` calls `predict_quantiles()` and returns a low, median, and high quantile for every horizon step directly from the sampled continuations.

TimesFM takes the opposite route, closer to how a vision transformer handles an image than how a language model handles text. Instead of tokenizing individual values, it slices the context window into contiguous patches of several time steps each, embeds each patch as a single vector, and lets a decoder-only transformer attend across those patch embeddings to predict the next patch. There is no vocabulary and no sampling: at its core, the model predicts the next patch directly rather than a distribution over discrete tokens. TimesFM also has an optional quantile head that can return an uncertainty band.

**Listing 14.7**: Zero-shot forecasts from Chronos and TimesFM

```python
chronos_result = results["chronos_forecast"]
timesfm_result = results["holdout_forecasts"]["timesfm"].to_numpy()
print(chronos_result["median"].round(1).to_numpy())
print(timesfm_result.round(1))
```

```
[341.6 355.1 356.1 374.4 383.5 390.7 395.1 404.2]
[344.3 354.7 368.9 378.9 375.4 389.2 401.8 405.6]
```

The same family includes TimeGPT (API-only, excluded here for offline reproducibility), Lag-Llama, and Moirai; the selection criterion is license, local runnability, and probabilistic output, which Chronos satisfies most directly. 

`chronos_forecast()` wraps `BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)` from the `chronos-forecasting` package, which downloads and caches the pretrained weights the first time it runs, then calls `pipeline.predict_quantiles(context_tensor, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])` on the context window; the call returns a quantiles tensor and a mean tensor, which `chronos_forecast` unpacks into low, median, high, and mean columns. 

`timesfm_forecast()` loads `TimesFM_2p5_200M_torch.from_pretrained()` from the `timesfm` package, compiles it once with a `timesfm.ForecastConfig` that caps the context and horizon length, then calls `model.forecast(horizon=horizon, inputs=[context_array])`, which returns a point-forecast array and a quantile array;

Figure 14.15 puts both zero-shot forecasts on the same held-out window: neither model has seen a single Roventra data point in training, yet both track the actual climb closely.

![Figure 14.15. Chronos and TimesFM against the held-out weeks, with no training on Roventra data at all.](assets/figures/figure_14_15_foundation_forecast.svg)

*Figure 14.15. Chronos and TimesFM against the held-out weeks, with no training on Roventra data at all.*

> **Note:** Running this chapter's analysis prints "sending unauthenticated requests to the HF Hub." Both `chronos_forecast()` and `timesfm_forecast()` call `from_pretrained()`, which downloads the pretrained weights from Hugging Face Hub the first time each model runs; the warning is Hugging Face's standard notice that no `HF_TOKEN` is set, not an error. Setting an `HF_TOKEN` environment variable raises the download rate limit but is not required for either model to run.

### 14.3.9 The Scorecard and Calibrated Intervals

The real question is which of these 9 methods a forecaster should actually use. Running every method through the same `rolling_origin_backtest()` and `accuracy_scorecard()` answers it.

**Listing 14.8**: The full accuracy scorecard

```python
scorecard = results["in_market_scorecard"]
print(scorecard.round({"mae": 1, "wmape": 3, "mape": 3, "mase": 2}))
```

```
           method    mae  wmape   mape  mase
0             ets   17.4  0.064  0.070  0.36
1         chronos   18.3  0.049  0.047  0.37
2         timesfm   19.8  0.053  0.052  0.40
3          sarima   27.1  0.100  0.104  0.55
4         prophet   29.0  0.078  0.074  0.59
5           naive   48.9  0.180  0.186  1.00
6  seasonal_naive   48.9  0.180  0.186  1.00
7             tft   73.0  0.195  0.194  1.49
8             gbt  100.4  0.268  0.261  2.05
```

Figure 14.16 puts every method from this section on the same chart.

![Figure 14.16. Every method against the held-out weeks, on one chart, colored to match its earlier dedicated figure.](assets/figures/figure_14_16_foundation_vs_classical.svg)

*Figure 14.16. Every method against the held-out weeks, on one chart.*

Neither foundation model was trained on a single Roventra data point, yet both track the actual holdout (340 to 456) closely enough to land second and third in the scorecard, behind only ETS: Chronos finishes with a MASE of 0.37, TimesFM with 0.40.

Chronos's output is a distribution over tokens to begin with, `predict_quantiles()` returns a low, median, and high value for free. TimesFM can return quantile head as well; ETS, SARIMA, Prophet, GBT, and the TFT hand back a single point forecast, so "next week is 500" is the entire answer; a planner cannot size a safety margin against that, or later tell whether a miss was actually surprising or well within normal range.

Split conformal prediction builds an interval for any of those point-forecast methods, using only the method's own track record and no assumption about the shape of its errors. Hold out a calibration set the model never trained on, measure the absolute error on every calibration point, and take a chosen percentile of those errors, say the 80th, as a fixed half-width. Adding and subtracting that half-width from a new point forecast produces an interval that, by construction, covered roughly 80% of the calibration set's actual values; if the future behaves like the recent past that calibrated it, the interval should cover about 80% of future outcomes too.

The calibration set here is `classical_backtest`, the same rolling-origin folds from 14.3.2, not the 8-week holdout the scorecard used to rank methods. Each of those folds already has a known actual and a known prediction from earlier in the series, so the absolute errors are measurable today, on data that already happened. Assume this method will be about as wrong on the next 8 weeks as it has been on the last several 8-week windows it was already tested against, and size the band from that history.

`conformal_interval()` in `forecasting.py` implements this: it takes the residuals from a calibration backtest, finds the requested percentile of their absolute value, and applies that as a constant half-width around a new point forecast. `empirical_coverage()` then checks the promise against reality, counting how often the actual holdout value fell inside the resulting band.

**Listing 14.9**: A calibrated interval around the ETS forecast

```python
from forecasting import conformal_interval, empirical_coverage

classical_backtest = results["in_market_backtest"]
holdout_forecasts = results["holdout_forecasts"]
actual_holdout = results["holdout_actual"]["actual"].to_numpy()
calibration_backtest = classical_backtest.loc[classical_backtest["method"] == "naive"]
interval = conformal_interval(calibration_backtest, holdout_forecasts["ets"], alpha=0.20)
coverage = empirical_coverage(interval, actual_holdout)
print(interval.round(1))
print(f"empirical coverage: {coverage:.0%}")
```

```
              point_forecast  lower  upper  half_width
horizon_step
1                      340.8  259.5  422.2        81.4
2                      351.8  270.5  433.2        81.4
3                      362.8  281.5  444.2        81.4
4                      373.9  292.5  455.2        81.4
5                      384.9  303.5  466.2        81.4
6                      395.9  314.5  477.3        81.4
7                      406.9  325.6  488.3        81.4
8                      417.9  336.6  499.3        81.4
empirical coverage: 100%
```

Each row is the ETS point forecast for that week, plus a lower and upper bound 81.4 TRx below and above it, the fixed half-width every row shares. Empirical coverage is a check on that band after the fact: of the 8 actual holdout values, all 8 landed inside their row's lower-to-upper range, a 100% hit rate against the 80% the interval was built to promise.

![Figure 14.17. The calibrated 80% interval around the ETS point forecast, with the actual holdout overlaid.](assets/figures/figure_14_17_calibrated_fan_chart.svg)

*Figure 14.17. The calibrated 80% interval around the ETS point forecast, with the actual holdout overlaid.*

Landing at 100% instead of 80% is not a defect: with only 3 calibration folds feeding the residual quantile, the interval came out wide, and a wide interval is the safer failure mode than an overconfident one. More calibration history, once it exists, is the fix; a business decision made today should use the wide interval rather than pretend the narrower one.

ETS wins outright: a single well-specified trend model, cheap to fit and requiring no external dependency, beats every alternative here. What foundation models earn is second and third place with zero training, ahead of SARIMA, Prophet, and, notably, both the deep model and gradient-boosted trees, which is exactly the argument for reaching for them first when history is this short and a classical fit has not yet been tried or tuned.

Winning the backtest is not the same as being the number operations uses next week. Every ETS forecast in this section, including the one behind the calibrated interval above, came from a model fit on weeks 1 to 44 and scored against weeks 45 to 52. The production step refits ETS one more time, now on the complete 52 weeks, and forecasts weeks 53 to 60, the ones nobody has observed yet: 443 to 539 TRx. This refit, not the holdout-fit forecast used to pick the winner, is the number that carries forward into the demand-supply and consensus sections that follow.

Figure 14.18 checks that refit against what actually happened next. This is synthetic data, so the truth for weeks 53 to 60 is available to compare against here, something a real forecasting team would obviously not have at decision time. The production forecast climbs in a straight line, 443 to 539 TRx, exactly what a level-and-trend model does when extrapolated. The actual weeks 53 to 60 instead oscillate between 412 and 472, essentially flat, consistent with Roventra approaching the peak the Bass fit already placed around month 14, week 59. Averaged over these 8 weeks, the production forecast misses by 46.8 TRx, worse than the 17.4 MAE that made ETS the scorecard's winner in the first place. The scorecard measured how well ETS extrapolates a trend that is still accelerating; it said nothing about what happens once that trend runs out, and this is the window where it runs out.

The same conformal band from Listing 14.9, the identical 82-unit half-width from the identical calibration set, is drawn around this production forecast too. It covers 7 of the 8 actual weeks, close to its 80% promise, but misses on week 60, the very last one, where the actual value sits just below the band's lower edge. That is the calibration assumption failing exactly where it was weakest: the half-width is fixed from folds where ETS was still tracking an accelerating launch, and a constant band never widens just because the series has started bending away from trend, at precisely the point where the point forecast has already drifted the furthest from it.

![Figure 14.18. The ETS production forecast against the weeks that actually followed: a straight-line extrapolation against a launch approaching its peak.](assets/figures/figure_14_18_production_refit.svg)

*Figure 14.18. The ETS production forecast against the weeks that actually followed: a straight-line extrapolation against a launch approaching its peak.*

This figure supports the monitoring action: re-check a committed forecast against real data as it arrives, rather than treating a backtest win as a permanent guarantee.

## 14.4 Demand-Supply Planning

Inventory sits in specific warehouses serving specific regions and territories, each with its own lead time, wholesaler relationships, and demand volatility. A single national number cannot tell a supply team where to place stock or how large a buffer each location needs. Undersupply one region and a pharmacy runs out mid-refill: a patient's therapy lapses, and a prescriber remembers the outage the next time a formulary conversation comes up. Oversupply it and working capital sits idle in inventory that, for many specialty products, has a real shelf life and an expiry date closing in on it. Both failures are expensive, and both are avoidable with the same fix: split the national forecast down to the region and territory level, reconcile the pieces back into one coherent set of numbers, and add a stated safety-stock buffer sized to each level's own demand volatility. That translation, forecast to reconciled regional signal to order quantity, is what this section builds.

### 14.4.1 Independent Forecasts Do Not Add Up

A national total should match its bottom-level pieces added together. Forecast every level on its own, though, and each forecast only ever sees its own slice of history. Nobody is coordinating, so nothing forces the pieces to add up.

ETS, the winning method from 14.3, is fit independently on every series: the national base forecast is the production refit, and each of Roventra's 12 territories gets its own ETS fit on its own territory-level history. The mismatch shows up anyway, because forecasting a hierarchy level by level, with no communication between the levels.

**Listing 14.10**: Independent base forecasts by level

```python
hierarchy_base_forecast = results["hierarchy_base_forecast"]
print(hierarchy_base_forecast.iloc[0].round(1).to_string())
```

```
National    443.2
MI-T1        93.9
MI-T2         2.7
MI-T3        10.0
NO-T1        46.5
NO-T2        67.2
NO-T3         3.7
SO-T1        37.6
SO-T2        34.3
SO-T3        62.4
WE-T1         7.2
WE-T2        58.0
WE-T3         7.3
```

The 12 territory forecasts sum to 431.0, not the 443.2 the national ETS refit produced on its own, a 12.2-unit, 2.8% gap. Nothing is wrong with any number individually: each was fit from its own history with no knowledge of the others. The gap is the coherence problem hierarchical reconciliation exists to solve.

### 14.4.2 Reconciliation

Reconciliation forces the independent forecasts back into agreement, and there is more than one defensible way to do it. Consider a miniature version of the problem: a national forecast of 100 and two regional forecasts, 45 and 65, that sum to 110, not 100. Someone has to give.

`bottom_up` reconciliation trusts the regions completely and simply redefines the national total as whatever the regions sum to: 110 in the toy case, both regions left untouched at 45 and 65. It is the right choice when the regional data is richer or more current than the national aggregate, since regional teams often see local access wins or field disruptions before they show up cleanly in a national roll-up.

`top_down` reconciliation makes the opposite bet: it trusts the national forecast and splits it across regions by a stated historical share. If the regions have historically split 50/50, the toy national number stays 100 and is handed back down as 50 and 50, regardless of what each region's own model just said (45 and 65). This is the right choice when the national series is more stable or better-modeled (more data, less noise) than any single region on its own, at the cost of ignoring whatever a region's own model might have picked up that the national number cannot see.

The third option treats neither level as automatically correct: not fully trusting the regions like bottom-up, not fully trusting the national number like top-down, but pulling both toward each other by just enough to make them agree, in a least-squares sense. Turning that idea into arithmetic needs a precise way to state which rows of the hierarchy must equal the sum of which other rows, and a summing matrix `S` is exactly that statement. For the toy hierarchy, `S` maps the 2 bottom-level regions to all 3 rows (the national total and the 2 regions):

$$
S=
\begin{bmatrix}
1 & 1\\
1 & 0\\
0 & 1
\end{bmatrix}
$$

The first row says "national equals region 1 plus region 2"; the other two rows say each region equals itself. Given the vector of independent base forecasts $\hat y$ (100, 45, 65 in the toy case), the reconciled bottom-level values are $(S^\top S)^{-1}S^\top\hat y$, and multiplying back by $S$ gives the reconciled full set: national 103.33, Region 1 down to 41.67, Region 2 down to 61.67, coherent by construction and as close as possible, in squared-error terms, to every level's own original forecast. Both regions move by the identical 3.33 units despite starting 20 apart (45 versus 65): unweighted OLS spreads the correction evenly across the bottom-level series here, it does not scale each region's share of the adjustment to how far apart the regions' own readings were from each other. This is the ordinary-least-squares member of the MinT (minimum trace) family of reconciliation methods: the version that assumes every level's forecast error has the same variance, rather than the fuller MinT approach, which weights each level by its own estimated error covariance. The real hierarchy has 12 bottom-level territories instead of 2 toy regions, so `S` there is a 13-row, 12-column matrix, but the idea is identical: one row per level, a 1 wherever that level's total includes a given territory. `reconcile()` in `forecasting.py` implements all 3: `bottom_up`, `top_down` (which needs a stated historical share), and `ols`.

**Listing 14.11**: All 3 reconciliation methods against the unreconciled base forecasts

```python
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
```

```
               Unreconciled  Bottom-up  Top-down    OLS
National              443.2      431.0     443.2  442.2
Territory sum         431.0      431.0     443.2  442.2
```

Only the unreconciled column disagrees with itself, national and territory-sum 12.2 units (2.8%) apart. Every reconciled method fixes that by construction, national exactly equal to the territory sum in each of its 3 columns, but the 3 columns still land on 3 different shared totals: bottom-up (431.0) discards the national ETS read almost entirely in favor of 12 noisier territory fits, several estimated from a small, choppy slice of demand; top-down goes to the other extreme, keeping the national number exactly and reallocating it by historical share regardless of what each territory's own model currently says. OLS (442.2) sits close to the national number without discarding the territory-level signal, which reflects a real asymmetry: a national series aggregates more volume and is the more stable of the two levels to fit, so leaning toward its reading, without ignoring the territory-level one, is the more defensible default.

![Figure 14.19. Only reconciliation makes the national forecast match the sum of 12 territories; unreconciled leaves a visible gap, every reconciled method closes it, and each closes it at a different shared total.](assets/figures/figure_14_19_forecast_hierarchy.svg)

*Figure 14.19. Only reconciliation makes the national forecast match the sum of 12 territories; unreconciled leaves a visible gap, every reconciled method closes it, and each closes it at a different shared total.*

This figure supports the supply-planning action: choose a reconciliation rule before territory-level orders are released from an incoherent hierarchy.

### 14.4.3 From Demand to Supply Signal

The reconciled national number is dispensed demand, TRx: prescriptions actually filled for patients. Ex-factory shipments are a different number entirely, what the manufacturer ships out to wholesalers and distributors, and the two are not interchangeable because wholesaler inventory sits between them and acts as a buffer. A given week's shipment volume reflects not only how many patients filled a prescription that week, but also whether distributors happened to be building up or drawing down their own stock at the same time. Forecast from shipment history directly, instead of from the underlying TRx signal, and that buffering effect rides along uninvited: a small dip in prescribing can look like a much larger dip in orders once distributors respond by trimming their own inventory, the bullwhip effect. Forecasting TRx and translating it into an order signal, the path this section takes, keeps the real demand signal separate from the distribution noise sitting on top of it.

`demand_to_supply()` in `forecasting.py` converts reconciled demand and its uncertainty into an order quantity with the standard service-level safety-stock formula:

$$
OS=(D\cdot L)+\bigl(z\cdot\sigma_d\cdot\sqrt{L}\bigr)
$$

- $D$: reconciled weekly demand
- $L$: replenishment lead time, weeks between placing an order and receiving it, 4 here; $D\cdot L$ is lead-time demand, what gets consumed while waiting for the next shipment
- $z$: service-level multiplier, standard deviations of buffer held for a chosen probability of not stocking out; 1.28 in this chapter, roughly a 90% service level, about 9 orders in 10 will not run short before the next shipment arrives
- $\sigma_d$: standard deviation of week-to-week demand changes, so safety stock scales up with both lead time and how volatile demand has actually been
- $SS$: safety stock, the second term in parentheses; $z\cdot\sigma_d\cdot\sqrt{L}$
- $OS$: the order signal, lead-time demand plus safety stock; what actually gets ordered, not the raw demand number a naive read of the forecast would suggest

This is the standard safety-stock rule from classical inventory theory (the reorder-point model taught throughout operations-management curricula and the APICS/CPIM body of knowledge, and the default "statistical safety stock" calculation in ERP systems like SAP and Oracle). The intuition behind it: if demand each week is roughly independent, variances add across weeks, but standard deviation is the square root of variance, so lead-time demand's volatility grows with $\sqrt{L}$, not $L$. A 4-week lead time needs only 2 times ($\sqrt{4}$) the 1-week buffer, not 4 times it, because a run of unusually high weeks and a run of unusually low weeks partially cancel out over a longer wait. Treating lead time as scaling risk linearly, the intuitive but wrong assumption, overstates the buffer needed and ties up working capital for no service-level benefit.

**Listing 14.12**: Safety stock and the order signal, national level

```python
demand_to_supply = results["demand_to_supply"]
print(demand_to_supply.iloc[0].round(1).to_string())
```

```
reconciled_demand     442.2
safety_stock           38.7
order_signal         1807.7
```

The national number is useful for illustrating the formula, but it is not what gets ordered. Inventory sits in a specific territory's warehouse, not in a national pool, and each territory has its own reconciled demand and its own week-to-week volatility. The same formula has to run once per territory, against that territory's own numbers, not once against the national total.

**Listing 14.13**: Safety stock and the order signal, by territory

```python
demand_to_supply_by_territory = results["demand_to_supply_by_territory"]
print(demand_to_supply_by_territory.round(1))
```

```
           reconciled_demand  safety_stock  order_signal
territory
MI-T1                   94.8          17.8         397.1
MI-T2                    3.7           0.5          15.2
MI-T3                   11.0           1.6          45.5
NO-T1                   47.4           8.9         198.6
NO-T2                   68.2          12.1         284.8
NO-T3                    4.6           0.8          19.3
SO-T1                   38.6           8.9         163.2
SO-T2                   35.3           8.1         149.2
SO-T3                   63.3          12.8         266.2
WE-T1                    8.2           1.3          33.9
WE-T2                   58.9          10.8         246.5
WE-T3                    8.2           1.2          34.0
```

These 12 order signals sum to 1,853.7, about 46 units more than the single national order signal of 1,807.7. That gap is not an error, it is the cost of splitting the buffer: pooling demand into 1 national number partly cancels out each territory's independent ups and downs before any safety stock gets sized, while sizing safety stock separately per territory cannot capture that cancellation, so the 12 buffers add up to more total inventory than 1 pooled buffer would need. Real supply chains hold both truths at once: the pooled number is the right way to think about total exposure, and the per-territory table is the actual set of order quantities a planner sends to 12 different warehouses.

This is not the reconciliation problem from 14.4.2, and reconciling the two order signals would be a mistake. Reconciliation there works because national demand is definitionally the sum of territory demand, 1 true quantity independently estimated at 2 levels, so forcing agreement corrects a real estimation gap. Safety stock does not add the same way: standard deviations shrink under pooling, so the pooled and per-territory order signals are not 2 estimates of 1 number, they are correct answers to 2 different questions, buffer for a single central warehouse versus buffer for 12 independent ones. Which answer to use is a network-design decision, not a forecasting one.


## 14.5 Loss of Exclusivity

Loss of exclusivity, LOE, is the date a brand's protection from direct copies ends and generic (or, for biologics, biosimilar) versions become legally free to enter the market. Nothing about demand changes on that date, prescribers still write for the same condition and patients still need the same molecule, but a chemically identical, far cheaper alternative becomes substitutable at the pharmacy counter, and under most states' substitution laws that swap happens automatically, without a new prescription or the original brand being consulted at all. The launch curve in 14.2 and the in-market models in 14.3 all describe a brand accumulating and holding volume, and LOE is the point where that logic reverses.

A drug's underlying patents run 20 years from filing, but filing happens early in development, often before clinical trials start, so much of that term is already spent by the time the drug launches; Hatch-Waxman patent term restoration can add back some of the review time lost to the FDA process, capped so total patent protection cannot exceed 14 years from approval. Layered on top are separate regulatory exclusivities that run from approval, not from filing, and do not depend on having a patent at all: 5 years for a genuinely new chemical entity, 7 years for an orphan-designated drug, 12 years for a biologic under the BPCIA. Litigation adds another layer of uncertainty on the exact date, since Paragraph IV patent challenges from generic manufacturers are frequently settled with a negotiated entry date years before the underlying patent would otherwise expire. Net effect: real small-molecule brands typically hold something like 10 to 14 years of protected sales after launch, not a fixed number, and Roventra's fictional 6 years compresses that range for a teaching example.

The genuine forecasting problem is: how much volume survives it, and how fast the rest goes. Underestimate the erosion and the business overbuilds: sales reps stay deployed against a shrinking market, manufacturing keeps producing volume nobody will buy, and R&D reinvestment that should have been redirected toward the next pipeline asset gets delayed a year too long. Overestimate the erosion and the business gives up early: field force gets pulled and price gets discounted before the real residual demand, the patients and prescribers who stay on brand for years after generic entry, ever shows up to justify that retreat. Because the date itself is known so far in advance, this is exactly the kind of decision finance and commercial leadership expect a real number for, which is what makes the shape of the decline.

### 14.5.1 The Erosion Problem

Bass diffusion, the model behind the launch curve, describes prescribers converting one at a time to an option they did not have before, spreading through a population by awareness and peer influence and approaching a ceiling from below. Generic entry drives TRx down for a specific, mechanical reason that has nothing to do with prescribing behavior at all: once a bioequivalent, far cheaper generic is approved, state substitution laws direct the dispensing pharmacist to fill the prescription with it instead of the brand by default, a swap that happens at the pharmacy counter, prescription by prescription, without the prescriber being consulted on that specific fill. The price-sensitive, administratively simple share converts almost immediately; a residual share, patients on a specific formulation, prescribers with a clinical reason to stay on brand, payers with a contractual carve-out, converts far more slowly, and persists for years. One smooth curve carries both: it declines fastest in the first weeks, when the gap between brand and floor is largest, and keeps slowing as it closes in on that floor.

![Figure 14.20. A single half-life decay curve, reading fast in the early weeks and slow near its residual floor.](assets/figures/figure_14_20_erosion_schematic.svg)

*Figure 14.20. A single half-life decay curve, reading fast in the early weeks and slow near its residual floor, with no break in the curve itself.*

### 14.5.2 Analog-Based Erosion Curves

Before a brand has any post-entry history of its own, the same borrowing logic used for launch analogs applies: find comparable molecules that have already been through generic entry, and use their shape as a starting projection. Both analogs below share the same half-life decay form 14.5.3 derives and fits directly from data, with $P$ pinned to Roventra's own pre-entry level and $R$ and $h$ taken from each analog's own known history instead of estimated. `analog_erosion_forecast()` in `forecasting.py` holds 2 fictional comparable erosion shapes:

| Analog | Residual fraction ($R\div P$) | Half-life $h$ (weeks) | What it represents |
| --- | --- | --- | --- |
| Comparable erosion A (fast generic substitution) | 8% | 6 | A commodity small molecule with easy generic substitution and little reason for patients or prescribers to stay on brand; most volume converts within a few half-lives. |
| Comparable erosion B (slower substitution, branded loyalty) | 20% | 16 | A brand with substitution friction, patients tied to a specific formulation, or a payer carve-out, that keeps a much larger share on brand for much longer. |

Comparable A closes half its remaining distance to the floor every 6 weeks, Comparable B every 16 weeks, so A collapses toward its (lower) floor much faster.

Projected forward from Roventra's own pre-entry level of 251 TRx, the first observed value in the same post-entry window plotted in Figure 14.1, the 2 shapes diverge fast: by week 12, A has already fallen to 78 while B is still at 169; by week 52, A has flattened near its 20-TRx floor while B is still descending toward 50.

**Listing 14.14**: Compare 2 analog erosion shapes from the same pre-entry level

```python
analog_erosion = results["analog_erosion_comparison"]
comparison = analog_erosion.loc[analog_erosion["weeks_since_entry"].isin([12.0, 52.0])].copy()
comparison["weeks_since_entry"] = comparison["weeks_since_entry"].astype(int)
comparison = comparison.rename(columns={
    "Comparable erosion A (fast generic substitution)": "Comparable A",
    "Comparable erosion B (slower substitution, branded loyalty)": "Comparable B",
})
print(comparison.round(1).to_string(index=False))
```

```
 weeks_since_entry  Comparable A  Comparable B
                12          77.7         169.4
                52          20.6          71.2
```

Figure 14.21 shows the full projected path from week 0 to week 104. The fast-substitution analog is already near its floor within the first year. The slower analog is still carrying a visible tail far later.

![Figure 14.21. Comparable erosion A and Comparable erosion B projected from the same pre-entry level.](assets/figures/figure_14_21_analog_erosion_curves.svg)

*Figure 14.21. Comparable erosion A and Comparable erosion B projected from the same pre-entry level.*

### 14.5.3 Parametric Decline Fit and Chronos Cross-Check

Once a few months of the brand's own post-entry data exist, an analog is no longer the best available evidence; the brand's own tail is. The decline itself follows the same shape a radioactive decay or a drug's own plasma concentration follows: a value moving from a starting level toward a floor, always closing the same fraction of the remaining distance in each fixed unit of time. That is a half-life:

Consider a toy brand at 1,000 TRx the week before generic entry, a 10% residual fraction (a 100-TRx floor), and a 10-week half-life. Each 10 weeks closes half the remaining distance to that floor: at week 10, TRx = 100 + 900 x 0.5 = 550; at week 20, another half-life closes half of what's left, TRx = 100 + 900 x 0.25 = 325; by week 40, 4 half-lives in, TRx = 100 + 900 x 0.0625 ≈ 156, closing in on the 100 floor.

$$
\mathrm{TRx}(t)=R+(P-R)\times 0.5^{t/h}
$$

$t$ is weeks since generic entry, $P$ is the brand's TRx level right before entry (`pre_entry`), $R$ is the floor the decline approaches asymptotically (`residual`), and $h$ is how many weeks it takes to close half the remaining distance to that floor (`half_life`). `fit_erosion()` in `forecasting.py` fits `residual` and `half_life` to the brand's own observed post-entry tail by nonlinear least squares, using the first observed post-entry value as `pre_entry`.

**Listing 14.15**: Fitting the decline on 20 weeks of post-entry data, then on 78

```python
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
```

```
                 residual_fraction  half_life_weeks
fit on 20 weeks              0.0010              10.7
fit on 78 weeks              0.1028               9.0
```

Fitting on only the first 20 weeks after generic entry, which is exactly the steepest part of the drop, projects a residual fraction of essentially zero: the fit reads the steep early decline and extrapolates it all the way to nothing. Fitting on 78 weeks, enough to see the decline actually flatten, recovers a residual fraction of about 10%. The 20-week fit is a different, wrong answer, produced by a real and common mistake: extrapolating a decay curve before the tail has had time to show itself. The analog-based projection makes the same error even more visibly, since it never sees the brand's own tail at all.

![Figure 14.22. Actual post-entry TRx, the analog erosion band, and the Chronos zero-shot cross-check, with the right panel zooming the tail.](assets/figures/figure_14_22_erosion_curve.svg)

*Figure 14.22. Actual post-entry TRx, the analog erosion band, and the Chronos zero-shot cross-check, with the right panel zooming the tail.*

The Chronos cross-check in Figure 14.22 runs the same zero-shot model from the foundation-model section on the post-entry tail, with no fitting and no residual-fraction assumption at all. Chronos, not ETS, SARIMA, Prophet, GBT, or TFT, because those 5 would have to be trained on this same 78-week tail to say anything. Zero-shot forecasting sidesteps that by construction: Chronos was pretrained on a large corpus of real-world series, including plenty that decay toward a floor, and never sees Roventra's own numbers as training data, only as context at inference time, a genuinely different source of signal than a curve fit to this brand's own history, which is what makes agreement between the 2 worth something. Chronos does not carry the parametric model's specific assumption, a single fixed floor reached at a single half-life, so an independent method landing near the same continuation is corroborating evidence; a mismatch would be the signal that the parametric shape does not hold for this particular brand even 78 weeks in.

This figure supports the LOE-planning action: avoid committing a decline curve until the brand tail and an independent cross-check tell the same story. Once they agree, `residual_fraction` and `half_life_weeks` are not just 2 diagnostic numbers, they are the entire decline curve: fed back into the formula above, they project TRx, and with it revenue, at any week past the 78 observed here. That projection is what finance models the post-LOE P&L against, and what supply planning uses to decide how far to wind down production and territory-level safety stock (14.4): the residual fraction alone tells the business roughly what fraction of peak revenue this brand keeps forever, and the half-life tells it how many years that transition actually takes, the 2 numbers a board needs to decide when to stop investing in the brand and how much standing infrastructure, sales force, manufacturing capacity, to keep committed to it after generic entry.

## 14.6 Consensus and Scenario Forecast

The company cannot commit 5 different forecasts to its board. It commits one, and getting there is a governance problem as much as a modeling one: the scorecard in 14.3.9 already named a best-performing method, but "won last quarter's backtest" is not the same claim as "right for next quarter," and no single method's number carries management's own market read or survives contact with whatever finance and the field believe the backtest missed. This section builds that 1 number in 3 steps: blend several methods by their own demonstrated accuracy instead of betting everything on 1 backtest winner, let commercial and finance adjust that blend through a change that gets named and tested rather than an unlogged override, and wrap the committed number in a low, base, and high range tied to explicit business drivers instead of an unlabeled statistical band.

### 14.6.1 Ensemble the Methods

The scorecard crowned one winner, ETS, but a single method's forecast still carries that method's own particular blind spots: ETS has no way to see a scheduled promotional flight, and it has never been checked against a launch pattern the way the Bass fit and the foundation models were. A committed number built from several methods that each get things wrong in different ways is generally steadier than betting everything on the single method that happened to win one backtest.

The question is how much weight each method should get in that blend. Weighting every method equally treats a method with 3 times the error of another as equally trustworthy, which is not a reasonable use of the scorecard the chapter already built. `ensemble_consensus()` in `forecasting.py` instead weights each method by the inverse of its own MASE from the accuracy scorecard: a method with MASE 0.3 gets roughly twice the weight of a method with MASE 0.6, and a method that could not beat the naive baseline at all still contributes, just with very little say. The weights are then normalized to sum to 1, so the result is a proper weighted average of the individual forecasts, not merely their sum.

**Listing 14.16**: An accuracy-weighted consensus

```python
from forecasting import ensemble_consensus

consensus_base = {
    "patient_based": [results["patient_based_forecast"].iloc[0]["ceiling"] / 12.0] * 8,
    "ets": results["production_forecast"]["forecast"].to_numpy(),
    "chronos": results["chronos_production_forecast"]["forecast"].to_numpy(),
}
consensus = ensemble_consensus(consensus_base, scorecard)
consensus_table = pd.DataFrame({"horizon_step": range(1, 9), "consensus": consensus})
print(consensus_table.round({"consensus": 1}).to_string(index=False))
```

```
 horizon_step  consensus
            1      448.7
            2      456.4
            3      468.8
            4      479.3
            5      488.0
            6      499.8
            7      507.8
            8      519.6
```

Every contribution here has to be a full-history production forecast for weeks 53-60, the genuinely unknown weeks, not a 44-week holdout forecast fit only to earn its spot on the 14.3.9 scorecard: ETS uses the same production refit Figure 14.18 plots, and Chronos gets its own full-52-week refit for the same reason, run through `chronos_forecast()` again exactly as ETS was, rather than reusing the backtest forecast that scored it for weeks 45-52, weeks that have already happened and describe a different, shorter-history model.

![Figure 14.23. The consensus, blended from patient-based, ETS, and Chronos, sits inside ETS's own 80% interval and tracks the actual continuation more closely than ETS alone.](assets/figures/figure_14_23_consensus_vs_actual.svg)

*Figure 14.23. The consensus, blended from patient-based, ETS, and Chronos, sits inside ETS's own 80% interval and tracks the actual continuation more closely than ETS alone.*

Weeks 53-60 are exactly the window Figure 14.18 already plotted: the 8 weeks right after the 52-week training history, genuinely unknown at the time either forecast was made, with real outcomes available here only because this is a book, not a live launch. The consensus carries no interval of its own, since it is a weighted average of point forecasts, not a fitted model with its own residuals to calibrate a band from; the conformal machinery from 14.3.9 was built for a single model's own error history; a blend across methods has no single such history to draw from without inventing 1. What Figure 14.23 shows instead is the ETS 80% interval, still valid since the consensus is 43% ETS by weight, as a reference band: the consensus point forecast sits inside it at every 1 of the 8 weeks. Against the actual continuation, the consensus's mean absolute error is 39.2 TRx, better than ETS alone at 46.8, because ETS's straight-line extrapolation keeps climbing through a launch that is starting to level off near its peak, and Chronos's and the flat patient-based ceiling's smaller contributions pull the blend down toward that leveling-off, closer to what actually happened. That is not a guarantee the blend always helps; it is 1 forecast's proof that combining several models with different, imperfect views of the same series moved the number closer to reality this time.

### 14.6.2 Analytics-Finance-Commercial Reconciliation

The consensus from 14.6.1 is a statistical number: it only knows what the historical series and models can see in it. Finance and commercial routinely know things that history cannot show, a formulary rule about to tighten, a launch-support budget just confirmed, a competitor's filing date leaking through the grapevine, and the consensus has no way to incorporate any of it on its own. Letting them adjust the number is therefore correct in principle; however an override that just changes the final number to whatever feels right is indistinguishable, after the fact, from a guess. `consensus_reconcile()` in `forecasting.py` fixes that by forcing every adjustment to be a named driver and a stated percentage applied in sequence, so the board sees a traceable path: this specific business reason moved the number by exactly this much.

Three illustrative adjustment percentages represent what a real forecasting cycle produces, each with a plausible-sounding size a real commercial and finance team would propose. The *reasons* ("access tightened," "launch support confirmed," "competitor risk") are exactly the kind of forward-looking information a statistical model genuinely cannot see and a business genuinely can, so overriding the consensus for reasons like these is legitimate practice. The *sizes* of the overrides, though, -8%, +5%, -3% here, are business judgment calls.

**Listing 14.17**: The reconciliation waterfall

```python
consensus_waterfall = results["consensus_waterfall"].round({"running_total": 1})
print(consensus_waterfall.to_string(formatters={"adjustment_pct": "{:.3f}".format}))
```

```
                             step adjustment_pct  running_total
0             Analytics consensus          0.000         3868.6
1     Access assumption tightened         -0.080         3559.1
2  Launch-support level confirmed          0.050         3737.0
3           Competitor entry risk         -0.030         3624.9
```

![Figure 14.24. From the analytics consensus to the committed number: each adjustment shown as both a volume delta and the percentage that produced it, against a reference line at the starting total.](assets/figures/figure_14_24_consensus_waterfall.svg)

*Figure 14.24. From the analytics consensus to the committed number: each adjustment shown as both a volume delta and the percentage that produced it, against a reference line at the starting total.*

 The dashed reference line at the starting total in the figure makes the net move visible, and each bar states both the volume it moved and the percentage that drove it, so a board member can trace back to 3 specific, sized business calls. Three adjustments move the total volume from 3,868.6 down to 3,624.9, a net 6.3% cut.

### 14.6.3 Scenarios

A committed number, even one built from an accuracy-weighted blend and reconciled through an adjustment process, is still a single number, and a board approving a multi-year launch budget is not only asking what to expect, it is asking what could go wrong and what would have to go right. 

Manufacturing capacity, sales-force headcount, and marketing spend all get committed against that number months in advance: underbuild against an upside outcome and the brand runs out of supply mid-launch; overbuild against a downside one and capital sits idle in unsold inventory and an oversized field force. 

A statistical interval, like the conformal band 14.3.9 built around next quarter's demand, answers how uncertain a point forecast is. Scenario planning  says which specific, actionable business driver assumption would move the outcome. `scenario_forecast()` in `forecasting.py` reuses the same peak-patient funnel from 14.2 and the same assumption ranges from its Monte Carlo section, re-run at a Low, Base, and High value for each driver rather than from a statistical interval, so a board member asking "what would the High case require" gets a specific answer, a stated peak brand share, not a number pulled from a wider confidence band.

**Listing 14.18**: Low, Base, and High scenarios

```python
from forecasting import scenario_forecast

print(scenario_forecast())
```

```
  scenario  addressable_patients  access_adjustment  brand_share  ceiling
0      Low                 16000               0.65          0.2   2080.0
1     Base                 20000               0.78          0.3   4680.0
2     High                 24000               0.85          0.4   8160.0
```

![Figure 14.25. Low, Base, and High launch scenarios form a fan from explicit driver assumptions.](assets/figures/figure_14_25_scenario_fan.svg)

*Figure 14.25. Low, Base, and High launch scenarios form a fan from explicit driver assumptions.*

The High case, 8,160 patients on a 40% peak brand share, lands almost exactly on the 8,470-patient ceiling the Bass fit recovered from real uptake data. That closes the business-case loop: the pre-launch business case treated 8,160 as an optimistic upside; the first year of real prescribing says it may be closer to the realistic case.

## 14.7 A Forecasting Method Selection Field Guide

None of the forecasting method in this chapter is universal applicable to your business use case. A different brand, in a different market, with a different history length or a different competitive dynamic, will not automatically call for the same method just because it won here. A Temporal Fusion Transformer and gradient-boosted trees, two methods with real strengths elsewhere, both lost to a naive baseline in 14.3.9, run at a history length neither was built for. Before you reuse any method in this chapter on a live brand, gather the few inputs that decide whether the method is even eligible in the first place.

- target series and unit, such as NBRx, TRx, units, or dollars
- forecast horizon and cadence
- known future events and covariates
- hierarchy levels that must reconcile
- enough backtest folds to score without leakage
- quantified business adjustments, traceable back to a specific driver

The scorecard earlier in this chapter ranks 9 methods on one backtest, 4 folds of one brand's history, and a ranking like that should never be the only basis for picking a method on a live brand. History length, whether future covariates are known in advance, and how many related series exist all matter more than which model happened to win this one comparison. Table 14.2 turns every method this chapter actually used into a single practical selection rule, in the order they appeared: pre-launch, in-market, supply, loss of exclusivity, then the final consensus.

*Table 14.2. Practical forecasting method-selection rule, methods used in this chapter.*

| Situation | Start with | Use if | Avoid if |
| --- | --- | --- | --- |
| No brand data yet, a business case built on judgment alone | Patient-based funnel, Monte Carlo over its own assumption ranges | before launch, when the ceiling has to come from stated assumptions, not data | real prescribing already exists to check or fit against directly |
| Early prescribing exists, but not enough to fit a curve of its own | Analog-based launch curve, replaced by a fitted Bass diffusion curve once enough data accumulates | a comparable historical launch exists to borrow a shape from | the assumed ceiling is still unverified; the analog read itself can flip depending on that ceiling (14.2.4) |
| Less than 1 year of weekly history | ETS and Chronos | trend dominates and seasonality is not mature | the business needs driver attribution |
| A flat or low-signal series with nothing else worth modeling | Naive or seasonal-naive | you need a floor every other method has to beat | used as the production forecast itself |
| 2 or more full seasonal cycles of history | SARIMA | the series is seasonal and stabilizes after differencing | history does not yet span 2 full seasons |
| Planned access or promotion changes | Prophet | future covariates are known from the plan | future covariates are guessed from actuals |
| Rich lagged history and many rows | GBT | nonlinear lag and calendar effects matter | the training set is tiny |
| Many related short territory series | TFT | the panel is broad and stable | there are only a few noisy series |
| A zero-shot cross-check sharing no assumptions with a fitted model | Chronos or TimesFM | you want an independent read, or have no training data at all | you need explicit driver attribution, not just a point forecast |
| Hierarchical supply planning | OLS reconciliation, then a safety-stock buffer per level | national and regional forecasts both carry signal | one level is clearly trusted more |
| Post-generic-entry decline | Parametric half-life fit, cross-checked against an analog and a zero-shot foundation model | enough post-entry data exists to see the curve flatten | extrapolating from only the first few weeks after entry |
| One committed number built from several valid methods | Accuracy-weighted ensemble, then a named adjustment waterfall | each method has different, largely uncorrelated blind spots | a single backtest winner is trusted blindly, never checked against a blend |
| A multi-year range a board can act on | Driver-based Low, Base, High scenarios | the board needs to know which specific assumption to hold accountable | a statistical interval stands in for a driver-based range instead |

A handful of other methods come up often enough in pharma forecasting to be worth knowing.

*Table 14.3. Common pharma-forecasting methods not used in this chapter.*

| Situation | Start with | Use if | Avoid if |
| --- | --- | --- | --- |
| Many zero-demand periods, low-volume specialty or orphan-drug orders | Croston's method, or another intermittent-demand model | most periods show 0 units and ETS or ARIMA badly overfit the zeros | volume is smooth and rarely, if ever, hits 0 |
| Multiple brands that cannibalize or lift each other | Vector autoregression (VAR), or a shared-driver panel model | growth in one series measurably steals from or lifts another | the series are genuinely independent of each other |
| Complex, overlapping seasonal cycles, for example daily pharmacy fill data with weekly, monthly, and holiday effects stacked together | TBATS | seasonality is multi-period and too complex for a single seasonal term | the series is short or has 1 simple seasonal pattern |
| Hundreds of related series across brands, countries, or indications | DeepAR, N-BEATS, or another supervised deep model built for large panels | there is enough panel breadth and compute to justify training one from scratch | a single brand with limited history, the exact case where TFT lost to ETS in 14.3.9 |
| A genuinely first-in-class launch with no workable analog and no data | Structured expert elicitation, for example a Delphi panel | nothing in the portfolio or the market resembles this launch closely enough to borrow from | a workable analog or real data already exists; judgment alone is the weakest source available once either does |

Every number in the forecast needs an uncertainty range attached to it, and the mechanism differs by stage:

- pre-launch business case: a Monte Carlo band over the funnel's own assumption ranges, since no residual exists yet to calibrate from
- in-market demand: a conformal interval calibrated from real backtest residuals
- loss of exclusivity: an analog band before enough post-entry data exists, then a Chronos cross-check once it does
- supply planning: a safety-stock buffer sized to each territory's own volatility
- consensus: driver-based low, base, and high scenarios, not a statistical band, so a board member knows which assumption each case depends on

## 14.8 Revising the Forecast

A forecast is wrong more often than it is right, in the narrow sense of matching the eventual actual to the decimal. The question every pharma forecasting function has to answer is not how to get the number right the first time; no method guarantees that under real market uncertainty, and this chapter's own numbers prove it repeatedly. The question is how to find out the number is wrong quickly enough to correct it before that error becomes a committed decision. A forecast that misses by 20% and gets caught and corrected within a planning cycle costs a planning cycle. The same miss, left uncorrected, compounds through a quarter of manufacturing orders, a year of field deployment, and a multi-year P&L, and by the time it surfaces the fix costs a warehouse of unsold inventory or a sales force sized for a launch that never happened. Every stage of this chapter built in a specific mechanism for catching that gap.

The business case for pre-launch is a chain of stated assumptions with no data yet; the funnel is the right tool before data exists. Once the first months of real prescribing come in, they do not adjust the funnel, they replace it outright. Roventra's launch team built a 4,680-patient ceiling from assumption alone; 6 months of real uptake, read through a fitted Bass diffusion curve, corrected that ceiling to 8,470.

Once real weekly prescribing exists, correction stops being a one-time event and becomes a cadence. The rolling-origin backtest this chapter used to pick ETS over eight competing methods is not a one-time verdict; it is a standing test that has to be rerun as more history accumulates. A seasonal method like SARIMA cannot even fit a seasonal term until 2 full years of history exist, so a brand whose ETS forecast wins comfortably in its first year may face a genuine seasonal challenger by its third. The discipline is re-running the same scorecard on a fixed schedule and letting a materially better challenger replace the incumbent the same way ETS replaced the naive baseline here.

Supply planning carries its own version of the same problem. Reconciling a national number against 12 independently fit territory forecasts is not a one-time fix either, since every new week of territory-level data moves that territory's own fit, and a hierarchy that agreed last cycle can drift back out of agreement by the next one. Reconciliation has to run on the same cadence as the forecasts feeding it.

Loss of exclusivity flips the correction problem around: here the risk is not catching a wrong number too late, it is correcting it too early. Fitting Roventra's own post-entry decline on only the first 20 weeks extrapolated a residual fraction of essentially 0, when 78 weeks of the same brand's own tail later confirmed a real, durable floor near 10% of pre-entry volume. Lesson: wait for the tail to show itself before committing to a residual fraction, and cross-check whatever fit exists against an independent read.

The consensus stage is where business judgment gets to correct the statistical read. Forecasting a launch with no commercial or finance input at all would be wrong. The discipline is making every override traceable.

None of these corrections happen once. Real pharma demand-forecasting functions run this entire loop, re-fitting, backtesting, reconciling, adjusting, on a fixed monthly or quarterly cadence inside the S&OP (sales and operations planning) cycle, the standing meeting that brings commercial, finance, and supply planning to the same table. Forecast accuracy and bias are tracked as a running KPI at that table, and a miss that crosses an agreed materiality threshold triggers a formal re-baseline. The mechanics in this chapter, run once here on Roventra from launch through generic entry, are the same mechanics a live brand's S&OP cycle runs for years: the business case gets corrected because market data proved it wrong, the point forecast gets corrected because a new backtest fold proved a challenger method better, the hierarchy gets corrected because two independently fit numbers stopped agreeing, and the committed number gets corrected because a business assumption changed, never because someone simply felt the old number looked off.

## 14.9 Summary

Finance opened this chapter needing five numbers: the ramp, the peak, next quarter, the units to manufacture, and the post-cliff decline. This chapter built an answer for each one.

You can now build a pre-launch business case and check it against real early data with a fitted diffusion curve, run a disciplined backtest across classical, machine-learning, deep-learning, and zero-shot foundation-model methods and let the scorecard, not intuition, pick the winner, fit and cross-check a post-exclusivity decline curve without extrapolating too early, reconcile a demand forecast down to a coherent regional and territory supply signal, and combine every method into one governed number with a named adjustment process and defensible scenarios.

Key takeaways:

- A pre-launch funnel ceiling is a chain of assumptions, and Monte Carlo simulation over that chain reveals which assumption is worth the most diligence, not just how uncertain the final number is.
- Short history is the practical constraint that governs which methods work at all: seasonal models need 2 full seasons they may not have, and zero-shot foundation models exist precisely to work without them.
- A backtest scorecard should include every candidate method on the same folds, because the answer for which method wins is an empirical question, not a foregone conclusion favoring the newest technique.
- An erosion or decline curve fit on too short a post-entry window will extrapolate the steep part of the drop and miss the eventual plateau; wait for the tail to show itself, or cross-check with an independent method.
- Reconciliation exists because independently forecasting a hierarchy's levels produces genuinely incoherent numbers, and the choice between bottom-up, top-down, and a blended method is a real decision, not a formality.
- A commercial adjustment to a consensus forecast should be named, quantified, and applied in a traceable sequence, not folded into the number as an unexplained overlay, so the gap between the statistical read and the committed number always has a stated business reason behind it.

## 14.10 Exercises

1. From "The Launch Uptake Curve: Bass Diffusion," refit the Bass curve using only the first 26 weeks of observed data. Compare the recovered ceiling and time-to-peak to the values in Listing 14.4. How much does the estimate move, and what does that say about how early is too early to trust a Bass fit? Use the [walkthrough notebook](ch14_walkthrough.ipynb) for the shared setup, and compare your work to [Exercise 1 in the solutions notebook](ch14_exercise_solutions.ipynb).

2. From "The Scorecard and Calibrated Intervals," rebuild the accuracy scorecard using NBRx as the target (`observed["nbrx"]`). Does the same method win, and if the ranking changes, what about NBRx's shape, compared with TRx in Listing 14.1, would explain the difference? Reuse the [walkthrough notebook](ch14_walkthrough.ipynb) backtest setup, then compare your result to [Exercise 2 in the solutions notebook](ch14_exercise_solutions.ipynb).

3. From "Parametric Decline Fit and Chronos Cross-Check," use `fit_erosion()` to find the smallest fit window, in weeks, that gets within 10% of the mature 78-week residual fraction. In a real post-launch program with a fixed reporting cadence, what would you tell a finance team about how soon after generic entry they can trust an erosion projection? Start from the [walkthrough notebook](ch14_walkthrough.ipynb) erosion setup, and check [Exercise 3 in the solutions notebook](ch14_exercise_solutions.ipynb) after you finish.

Open the [walkthrough notebook](ch14_walkthrough.ipynb) to run these blocks in order, and the [exercise solutions notebook](ch14_exercise_solutions.ipynb) for one worked approach to each. The reconciled, safety-stock-adjusted demand this chapter produced is exactly the input the resource-allocation optimizer needs next: a forecast with an interval is a planning input, and the next decision is how to spend against it.
