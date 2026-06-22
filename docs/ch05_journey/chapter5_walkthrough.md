# Chapter 5 Walkthrough: Treatment Patterns and Patient Journeys

This notebook replays the chapter as one executable story. Each step runs the same code printed in [`ch05_patient_journey.md`](ch05_patient_journey.md), so the outputs here match the chapter exactly. Run it from the repository root after generating the Chapter 3 data package.

The dashboard reports 3,193 Roventra line-1 entries without a washout. The VP asks how many are newly observed starts. By the end of this notebook, the 180-day washout reduces that count to 2,798 and every rule between the 2 numbers is explicit.

## 1. Build the cohorts

The journey and line cohort requires a qualifying launch-condition diagnosis, 180-day lookback, 90-day follow-up, and a mature study end. The initiation analysis uses the same lookback but does not require 90 future days at entry.


```python
import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "ch05_journey").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "ch05_journey/scripts"))
from episode_construction import build_newly_observed_cohort, load_chapter3_data

tables = load_chapter3_data(ROOT / "ch03_data/output_data/generated_data")
cohort, attrition = build_newly_observed_cohort(
    tables, minimum_lookback_days=180, minimum_followup_days=90
)
print(attrition)
```

                            stage  patients                                                       rule
    Patients in source population     20000                     One row in the patient reference table
    Observed qualifying diagnosis      8213 At least one encounter with ICD prefix E11.9|E11.65|E11.40
              Sufficient lookback      6562                     At least 180 covered days before index
                  Analysis cohort      5637      Lookback plus at least 90 observable days after index


The 5,637 patients form the journey and line-of-therapy cohort. The 90-day follow-up requirement gives early treatment patterns enough time to appear.


```python
import subprocess
result = subprocess.run(
    ["uv", "run", "python", "ch05_journey/scripts/run_analysis.py"],
    capture_output=True, text=True,
)
print(result.stdout)
```

    


## 3. A transaction is not a therapy

Resolve status before sequencing: completed fills are treatment evidence, while rejections and reversals are access signals.


```python
from episode_construction import prepare_pharmacy_events

basket = tables["products"]["product_name"].tolist()
paid, nonpaid = prepare_pharmacy_events(tables["pharmacy_claims"], cohort, basket)
events = pd.concat([paid, nonpaid], ignore_index=True)
print("post-index basket transactions by status:")
print(events.transaction_type.value_counts())

paid_ids = set(paid.patient_id)
any_ids = set(events.patient_id)
print(f"\npatients with a treatment-basket fill: {len(paid_ids):>6,}")
print(f"patients with any basket transaction:  {len(any_ids):>6,}")
print(f"access signals without treatment:     {len(any_ids - paid_ids):>6,}")
```

    post-index basket transactions by status:
    transaction_type
    PAID        8749
    PENDED      1364
    REVERSED      69
    
    patients with a treatment-basket fill:  3,928
    patients with any basket transaction:   3,980
    access signals without treatment:         52


Transaction status separates access attempts from completed treatment. The treatment-fill and access-signal counts show the size of that distinction.

## 4. Lines of therapy: the worked patients

PAT00839 passes the washout, starts Nexoral, then switches to Vexpro after the Nexoral supply ends.


```python
out = ROOT / "ch05_journey/assets/generated_outputs"
pharmacy = tables["pharmacy_claims"]
basket = tables["products"]["product_name"].tolist()
mine = pharmacy[
    pharmacy.patient_id.eq("PAT00839") & pharmacy.product_name.isin(basket)
].sort_values("date_of_service")
print("PAT00839, diagnosis index 2024-01-26:")
print(mine[["date_of_service", "product_name", "days_supply",
            "transaction_type"]])

lines = pd.read_csv(f"{out}/lines.csv")
cols = ["line_number", "regimen", "line_start", "line_end",
        "fill_count", "entry_reason", "end_reason", "line_days"]
print("\nlines of therapy:")
print(lines.loc[lines.patient_id.eq("PAT00839"), cols])
```

    PAT00839, diagnosis index 2024-01-26:
    date_of_service product_name  days_supply transaction_type
         2024-06-20      Nexoral           30             PAID
         2024-07-22       Vexpro           30           PENDED
         2024-07-24       Vexpro           30             PAID
         2024-08-21       Vexpro           30             PAID
    
    lines of therapy:
     line_number regimen line_start   line_end  fill_count    entry_reason   end_reason  line_days
               1 Nexoral 2024-06-20 2024-07-19           1 Initial therapy       Switch         30
               2  Vexpro 2024-07-24 2024-09-19           2          Switch Discontinued         58


PAT03874 demonstrates the addition rule: Nexoral arrives while Vexpro still has supply, after the regimen window closes, so the line advances to the combination.


```python
import pandas as pd

out = ROOT / "ch05_journey/assets/generated_outputs"
lines = pd.read_csv(f"{out}/lines.csv")
cols = ["line_number", "regimen", "line_start", "line_end",
        "fill_count", "entry_reason", "end_reason", "line_days"]
print(lines.loc[lines.patient_id.eq("PAT03874"), cols])
```

     line_number          regimen line_start   line_end  fill_count    entry_reason end_reason  line_days
               1           Vexpro 2024-07-06 2024-09-03           1 Initial therapy   Addition         60
               2 Nexoral + Vexpro 2024-08-29 2025-01-01           2        Addition   Censored        126


## 5. The cohort's lines, and Thursday's answer

Line depth, entries, ends, line-1 regimens, and the washout comparison that resolves the opening scene.


```python
import pandas as pd

out = ROOT / "ch05_journey/assets/generated_outputs"
lines = pd.read_csv(f"{out}/lines.csv")

print("patients by deepest line reached:",
      lines.groupby("patient_id").line_number.max().value_counts().sort_index().to_dict())
print("how lines are entered:", lines.entry_reason.value_counts().to_dict())
print("how lines end:        ", lines.end_reason.value_counts().to_dict())

print("\nline-1 regimens:")
print(pd.read_csv(f"{out}/lot_line1_summary.csv"))

print("\nRoventra line entries, with and without the washout rule:")
base = pd.read_csv(f"{out}/lot_entry_shares.csv").assign(rule="180-day washout")
naive = pd.read_csv(f"{out}/lot_entry_shares_naive.csv").assign(rule="no washout")
print(pd.concat([naive, base])[["rule", "position", "line_entries", "share"]]
      )
```

    patients by deepest line reached: {1: 3387, 2: 28}
    how lines are entered: {'Initial therapy': 3415, 'Switch': 24, 'Addition': 4}
    how lines end:         {'Censored': 1938, 'Discontinued': 1477, 'Switch': 24, 'Addition': 4}
    
    line-1 regimens:
             regimen  patients  median_line_days  discontinued_share
            Roventra      2798              59.0               0.434
              Vexpro       309              67.0               0.443
             Nexoral       303              66.0               0.356
    Nexoral + Vexpro         5              58.0               0.600
    
    Roventra line entries, with and without the washout rule:
               rule position  line_entries  share
         no washout   Line 1          3193    1.0
    180-day washout   Line 1          2798    1.0


Without the washout, Roventra has 3,193 line-1 entries. With it, the count is 2,798. The 395 records in between are continuing users that the no-washout view recounted as new starts.


## 6. Shake the rulers

One rule varies per row; the others hold at base.


```python
import pandas as pd

out = ROOT / "ch05_journey/assets/generated_outputs"
grid = pd.read_csv(f"{out}/lot_sensitivity.csv")
view = grid[["varied", "washout_days", "regimen_window_days", "allowable_gap_days",
             "new_to_therapy_patients", "combination_line1_share",
             "line1_discontinued_share", "roventra_line1_entry_share"]]
print(view)
```

            varied  washout_days  regimen_window_days  allowable_gap_days  new_to_therapy_patients  combination_line1_share  line1_discontinued_share  roventra_line1_entry_share
           washout             0                   30                  60                     3928                    0.001                     0.474                         1.0
           washout            90                   30                  60                     3444                    0.001                     0.426                         1.0
           washout           180                   30                  60                     3415                    0.001                     0.428                         1.0
    regimen window           180                   14                  60                     3415                    0.000                     0.428                         1.0
    regimen window           180                   45                  60                     3415                    0.004                     0.430                         1.0
     allowable gap           180                   30                  30                     3415                    0.001                     0.543                         1.0
     allowable gap           180                   30                  90                     3415                    0.001                     0.323                         1.0


The allowable gap moves the discontinuation result because it changes when a refill gap becomes an event. The Roventra entry share is less sensitive in this package. The commercial answer uses the 3,415 new-to-therapy patients and 2,798 Roventra first-line regimens as the corrected uptake baseline. Only 28 patients reach line 2, which supports rule validation but is too sparse for reliable later-line commercial comparisons.

## 7. Time to treatment, with censoring

The business questions determine the clock. Diagnosis to treatment start supports demand timing. Biomarker order to test result isolates testing delay. Prescription to PA approval isolates payer review. The available chapter data support the overall diagnosis-to-treatment-start clock.

Start with all 5 patients untreated. In this table, **at risk** means a patient is still being observed, has not started treatment yet, and could still start on that day. The **untreated risk set** is the group of patients who meet that condition right before the day's event. Patients A, B, and C start treatment on days 19, 31, and 59. Patients D and E stay in the untreated risk set until day 90, when follow-up ends and we censor them.



```python
import pandas as pd
from survival import km_curve, km_median

toy = pd.DataFrame({"day": [19, 31, 59, 90, 90],
                    "treated": [True, True, True, False, False]})
observed = toy.loc[toy.treated, "day"]
toy_curve = km_curve(toy.day, toy.treated)
print(f"treated-only mean: {observed.mean():.1f} days")
print(f"treated-only median: {observed.median():.0f} days")
print(f"Kaplan-Meier median: {km_median(toy_curve):.0f} days")
print(toy_curve[["day", "at_risk", "events", "censored", "survival"]]
      .round(3))
```

    treated-only mean: 36.3 days
    treated-only median: 31 days
    Kaplan-Meier median: 59 days
     day  at_risk  events  censored  survival
       0        5       0         0       1.0
      19        5       1         0       0.8
      31        4       1         0       0.6
      59        3       1         0       0.4
      90        2       0         2       0.4


The median is day 59, the first treatment day when the estimated untreated probability reaches 0.50 or lower. The censored patients enlarge each earlier denominator and leave on day 90 without creating an event.

Now apply the method to the chapter cohort. Listing 5.1 found 6,562 patients with sufficient lookback. Among them, 169 have diagnosis dates after the December 31, 2024 study cutoff and provide no post-diagnosis time within the study. Removing them leaves 6,393 patients for treatment-initiation analysis. We do not require 90 days of follow-up because Kaplan-Meier uses each patient's available observation time.


```python
import pandas as pd

out = ROOT / "ch05_journey/assets/generated_outputs"
journeys = pd.read_csv(f"{out}/initiation_journeys.csv")
treated = journeys[journeys.initiated_treatment]
print(f"end-of-study count: {len(treated):,} of {len(journeys):,} started; "
      f"{1 - journeys.initiated_treatment.mean():.1%} without observed start")
print(f"naive median days to treatment, treated only: {treated.days_to_treatment.median():.0f}")

curve = pd.read_csv(f"{out}/initiation_curve.csv")
km_median = curve.loc[curve.cumulative_initiation.ge(0.5), "day"].iloc[0]
print(f"Kaplan-Meier median time to treatment: {km_median:.0f} days")
for day in (90, 180, 270):
    row = curve.loc[curve.day.le(day)].iloc[-1]
    print(f"KM cumulative initiation by day {day}: {row.cumulative_initiation:.1%} "
          f"(95% CI {row.cumulative_initiation_lower_95:.1%} to "
          f"{row.cumulative_initiation_upper_95:.1%}; {int(row.at_risk):,} at risk)")
```

    end-of-study count: 4,110 of 6,393 started; 35.7% without observed start
    naive median days to treatment, treated only: 104
    Kaplan-Meier median time to treatment: 168 days
    KM cumulative initiation by day 90: 30.5% (95% CI 29.4% to 31.7%; 3,957 at risk)
    KM cumulative initiation by day 180: 52.9% (95% CI 51.6% to 54.3%; 2,142 at risk)
    KM cumulative initiation by day 270: 74.3% (95% CI 72.9% to 75.6%; 722 at risk)


The initiation curve provides measured planning rates at specific horizons. A forecast team can multiply a separate forecast of comparable diagnoses by the cumulative initiation estimate for day 90, 180, or 270. The diagnosis forecast and the initiation curve remain separate because they come from different evidence. Stage dates are still needed to locate testing, treatment-decision, and access delays. HCP and payer comparisons require adequate sample sizes and adjustment for case mix.

### Death before treatment

Replace Patient B's day-31 treatment with death. Death prevents a later start, so it is a competing event. The chapter's synthetic cohort does not contain a death-before-treatment record, so this section stays a teaching example. In a real study, death usually comes from linked EHR data, a mortality registry, or another source that records the death date. Standard pharmacy claims alone usually do not carry that field. Aalen-Johansen assigns probability separately to treatment, death, and remaining untreated and alive.


```python
from survival import aalen_johansen_curve

days = pd.Series([19, 31, 59, 90, 90])
outcomes = pd.Series(["Treated", "Died", "Treated", "Censored", "Censored"])
aj = aalen_johansen_curve(days, outcomes)
cols = ["day", "at_risk", "event_free", "cumulative_interest",
        "cumulative_competing"]
print(aj[cols].round(3))
```

     day  at_risk  event_free  cumulative_interest  cumulative_competing
       0        5         1.0                  0.0                   0.0
      19        5         0.8                  0.2                   0.0
      31        4         0.6                  0.2                   0.2
      59        3         0.4                  0.4                   0.2
      90        2         0.4                  0.4                   0.2


By day 90, the competing-risk estimate assigns 40% to treatment, 20% to death before treatment, and 40% to remaining untreated and alive. The treatment median is not reached. The final commercial artifact should carry the chosen clock, cohort, observation window, censoring and competing-event rules, confidence intervals, numbers at risk, and data cutoff.

## 8. Persistence and adherence after treatment starts

Persistence measures time from the first treatment fill until departure from the initial regimen. Adherence, also called compliance, measures the share of observed days with qualifying supply available. PDC counts unique covered days, while MPR counts all dispensed days supply. The product basket determines whether coverage follows the starting product or any treatment for the condition.

The commercial review needs the day-90 persistence estimate, the PDC distribution, the effect of product scope, and payer comparisons with uncertainty.


```python
import pandas as pd
from statistics import NormalDist

out = ROOT / "ch05_journey/assets/generated_outputs"
persistence = pd.read_csv(f"{out}/line1_persistence.csv")
for day in (60, 90, 113, 180):
    row = persistence.loc[persistence.day.le(day)].iloc[-1]
    print(f"day {day}: {row.survival:.1%} persistent; "
          f"{int(row.at_risk):,} at risk")

product = pd.read_csv(f"{out}/adherence_index_product.csv")
basket = pd.read_csv(f"{out}/adherence_market_basket.csv")
print(f"index-product PDC: mean {product.pdc.mean():.3f}, "
      f"median {product.pdc.median():.3f}, "
      f"{product.adherent_pdc.mean():.1%} at or above 0.80")
print(f"higher basket PDC than index-product PDC: "
      f"{basket.pdc.gt(product.pdc).sum():,} patients")

payer = pd.read_csv(f"{out}/adherence_by_payer.csv").query("payer_id != 'All'")
z = NormalDist().inv_cdf(0.975)
n, rate = payer.treated_patients, payer.adherent_pdc_rate
denominator = 1 + z**2 / n
center = (rate + z**2 / (2 * n)) / denominator
half_width = z * (rate * (1 - rate) / n + z**2 / (4 * n**2))**0.5 / denominator
payer = payer.assign(lower_95=center - half_width, upper_95=center + half_width)
print(payer[["payer_id", "adherent_pdc_rate", "lower_95", "upper_95"]]
      .round(4))
```

    day 60: 73.0% persistent; 1,776 at risk
    day 90: 60.6% persistent; 1,128 at risk
    day 113: 49.9% persistent; 701 at risk
    day 180: 19.2% persistent; 50 at risk
    index-product PDC: mean 0.445, median 0.395, 15.6% at or above 0.80
    higher basket PDC than index-product PDC: 36 patients
    payer_id  adherent_pdc_rate  lower_95  upper_95
      PAY001             0.1661    0.1290    0.2113
      PAY002             0.1837    0.1457    0.2289
      PAY003             0.1677    0.1315    0.2115
      PAY004             0.1318    0.1003    0.1713
      PAY005             0.1400    0.1075    0.1803
      PAY006             0.1637    0.1280    0.2070
      PAY007             0.1450    0.1111    0.1870
      PAY008             0.1522    0.1177    0.1946


At day 90, 60.6% remain on the initial regimen. Among patients with at least 90 observable days, 15.6% have index-product PDC at or above 0.80, and product switching changes PDC for 36 patients. The payer intervals overlap substantially, so the raw ranking provides weak evidence for a payer-specific adherence difference.

## 9. The post-index hub pathway

Only referrals between diagnosis index and follow-up end belong to this journey. The funnel combines conversion counts with median time from referral.


```python
import pandas as pd

out = ROOT / "ch05_journey/assets/generated_outputs"
print(pd.read_csv(f"{out}/sp_funnel.csv"))
print()
outcomes = pd.read_csv(f"{out}/sp_abandonment_outcomes.csv")
pivot = outcomes.pivot_table(index="discontinue_reason", columns="outcome",
                             values="patients", fill_value=0).astype(int)
print(pivot)
```

                     stage  patients  share_of_referrals  median_days_from_referral
         Referral received      2597               1.000                        0.0
    Authorization approved      1958               0.754                        5.0
                   Shipped      1836               0.707                       10.0
                 Abandoned       722               0.278                        4.0
    
    outcome             Later Roventra fill  No further treatment-basket fill
    discontinue_reason                                                       
    Cost                                236                                 7
    Coverage                            198                                 3
    Documentation                       218                                 5
    Lost follow-up                       26                                 0
    Patient decision                     29                                 0


## Exercises

1. **Audit a false new start** by tracing 1 patient removed by the 180-day washout.
2. **Choose the adherence product scope** by inspecting patients whose basket PDC exceeds index-product PDC.
3. **Stress-test the data cutoff** by moving `study_end` to 2025-01-31 and measuring the maturity artifact.

Worked answers with discussion: [`exercise_solutions.ipynb`](exercise_solutions.ipynb).
