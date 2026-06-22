# Chapter 5 Exercise Solutions

Worked answers for the three exercises in [`ch05_patient_journey.md`](ch05_patient_journey.md). The judgment call at the end of each answer matters more than the pandas.

## Setup

Rebuild the chapter cohort and fills once.


```python
import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "ch05_journey").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "ch05_journey/scripts"))

import pandas as pd

from episode_construction import build_newly_observed_cohort, load_chapter3_data
from lot import construct_lines_of_therapy

tables = load_chapter3_data(ROOT / "ch03_data/output_data/generated_data")
cohort, _ = build_newly_observed_cohort(tables, minimum_lookback_days=180, minimum_followup_days=90)
basket = tables["products"]["product_name"].tolist()
pharmacy = tables["pharmacy_claims"]
paid_all = pharmacy[
    pharmacy.transaction_type.eq("PAID")
    & pharmacy.product_name.isin(basket)
    & pharmacy.patient_id.isin(cohort.patient_id)
].copy()
lines_base, initiators = construct_lines_of_therapy(paid_all, cohort)
print(f"cohort {len(cohort):,} | new to therapy {initiators.new_to_therapy.sum():,} | lines {len(lines_base):,}")
```

    cohort 5,637 | new to therapy 3,415 | lines 3,443


## Exercise 1: Audit a false new start

Find 1 patient counted as a new start with no washout and excluded by the 180-day washout. Then inspect the completed fills that caused the reclassification.


```python
_, naive = construct_lines_of_therapy(paid_all, cohort, washout_days=0)
_, washed = construct_lines_of_therapy(paid_all, cohort, washout_days=180)
print(f"new starts: no washout {naive.new_to_therapy.sum():,}; 180-day washout {washed.new_to_therapy.sum():,}")
false_ids = washed.loc[~washed.new_to_therapy, "patient_id"]
patient_id = false_ids.sort_values().iloc[0]
therapy_index = washed.set_index("patient_id").loc[patient_id, "therapy_index"]
prior = paid_all.loc[
    paid_all.patient_id.eq(patient_id)
    & paid_all.date_of_service.lt(therapy_index)
    & paid_all.date_of_service.ge(therapy_index - pd.Timedelta(days=180)),
    ["date_of_service", "product_name", "days_supply"],
]
print(f"{patient_id}: therapy index {therapy_index.date()}")
print(prior)
```

    new starts: no washout 3,928; 180-day washout 3,415
    PAT00002: therapy index 2024-08-15
    date_of_service product_name  days_supply
         2024-07-21     Roventra           28


**Judgment.** The pre-index fill shows that this patient was already receiving treatment before the first post-diagnosis fill. The 0-day rule relabels a continuing refill as a launch start. In real data, the methods note should state the washout length, the source used to observe prior fills, and the number of apparent starts removed by the rule.

## Exercise 2: Choose the adherence product scope

Compare index-product and market-basket PDC, then inspect the patient with the largest increase under the broader basket.


```python
out = ROOT / "ch05_journey/assets/generated_outputs"
index_pdc = pd.read_csv(out / "adherence_index_product.csv")
basket_pdc = pd.read_csv(out / "adherence_market_basket.csv")
comparison = index_pdc[["patient_id", "pdc"]].merge(
    basket_pdc[["patient_id", "pdc"]], on="patient_id", suffixes=("_index", "_basket")
)
comparison["gain"] = comparison.pdc_basket - comparison.pdc_index
changed = comparison.loc[comparison.gain.gt(0)].sort_values("gain", ascending=False)
patient_id = changed.iloc[0].patient_id
print(f"patients with higher basket PDC: {len(changed):,}")
print(changed.head(5))
print("\nCompleted products for the largest increase:")
print(paid_all.loc[paid_all.patient_id.eq(patient_id), ["date_of_service", "product_name"]]
      .sort_values("date_of_service"))
```

    patients with higher basket PDC: 36
    patient_id  pdc_index  pdc_basket   gain
      PAT18206     0.2174      0.8696 0.6522
      PAT03874     0.3352      0.9665 0.6313
      PAT15311     0.2027      0.8108 0.6081
      PAT18715     0.2778      0.8611 0.5833
      PAT10932     0.2745      0.8235 0.5490
    
    Completed products for the largest increase:
    date_of_service product_name
         2024-08-16       Vexpro
         2024-09-20      Nexoral
         2024-10-17      Nexoral
         2024-11-16      Nexoral


**Judgment.** Index-product PDC belongs in a brand continuity report because it stops crediting coverage when the patient moves to another product. Market-basket PDC belongs in a condition-treatment continuity report because it follows qualifying treatment across product changes. The report must name the basket and reconcile changed patients against the line table.

## Exercise 3: The immature tail

Move `study_end` to 2025-01-31, the raw edge of the data, and compare cohort size, line-1 discontinuation, and censoring with the mature cutoff.


```python
def cutoff_summary(study_end):
    cohort_x, _ = build_newly_observed_cohort(
        tables, minimum_lookback_days=180, minimum_followup_days=90, study_end=study_end
    )
    paid_x = pharmacy.loc[
        pharmacy.transaction_type.eq("PAID")
        & pharmacy.product_name.isin(basket)
        & pharmacy.patient_id.isin(cohort_x.patient_id)
    ]
    lines_x, _ = construct_lines_of_therapy(paid_x, cohort_x)
    l1 = lines_x.loc[lines_x.line_number.eq(1)]
    return len(cohort_x), l1.end_reason.eq("Discontinued").mean(), l1.end_reason.eq("Censored").mean()

for label, cutoff in [("mature", "2024-12-31"), ("raw edge", "2025-01-31")]:
    patients, discontinued, censored = cutoff_summary(pd.Timestamp(cutoff))
    print(f"{label}: cohort {patients:,} | L1 discontinued {discontinued:.1%} | L1 censored {censored:.1%}")
```

    mature: cohort 5,637 | L1 discontinued 42.8% | L1 censored 56.3%


    raw edge: cohort 5,932 | L1 discontinued 52.0% | L1 censored 47.1%


**Judgment.** The raw-edge cutoff admits newer records whose claims have had less time to mature. It also gives the 60-day discontinuation rule more calendar time to fire. The methods note should fix a mature study end, report the claims-lag rule used to choose it, and compare journey KPIs only when their maturity windows match.
