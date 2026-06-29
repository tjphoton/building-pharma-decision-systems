# Chapter 4 Walkthrough: Market Sizing and Patient Populations

This notebook follows the revised Chapter 4 as one executable story. It uses the Chapter 3
synthetic claims package, keeps the chapter's visual figures in the Markdown cells, and
prints the key outputs immediately after the code that creates them.

All patient records are synthetic. The prevalence and population anchors are public
reference values used for teaching, and the launch product Roventra is fictional.


```python
import os
from pathlib import Path
import pandas as pd

# Resolve the repository root so relative data paths work from any launch directory.
repo = Path.cwd()
while not (repo / "ch03_data").exists() and repo != repo.parent:
    repo = repo.parent
os.chdir(repo)

DATA = "ch03_data/output_data/generated_data"
LAUNCH_CODES = ["E11.9", "E11.65", "E11.40"]   # the launch condition in ICD-10
PRODUCT = "Roventra"

patients = pd.read_csv(f"{DATA}/reference/patients.csv")
enroll = pd.read_csv(f"{DATA}/reference/patient_enrollments.csv")
patients = patients.merge(
    enroll[["patient_id", "payer_id"]].drop_duplicates("patient_id"),
    on="patient_id", how="left",
)

DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]
mc = pd.read_csv(f"{DATA}/claims_medical/medical_claims_mature.csv")
dx_mask = mc[DX_COLS].isin(LAUNCH_CODES).any(axis=1)
coded = mc[dx_mask]
paid_dx_count = coded.groupby("patient_id").size()

rx = pd.read_csv(f"{DATA}/claims_pharmacy/pharmacy_claims.csv",
                 dtype={"ndc": str, "ndc_prescribed": str})
ndc = pd.read_csv(f"{DATA}/reference/ndc_codes.csv", dtype={"ndc": str})
rx["drug_name"] = rx.ndc_prescribed.map(ndc.set_index("ndc").drug_name)
roventra = rx[rx.drug_name.eq(PRODUCT)].copy()
roventra["net"] = roventra.transaction_type.map({"PAID": 1, "REVERSED": -1}).fillna(0)
net_fills = roventra.groupby("patient_id").net.sum()

p = patients.copy()
p["true_condition"] = p["true_launch_condition"].fillna(False).astype(bool)
p["coded"] = p.patient_id.isin(coded.patient_id)
p["diagnosed"] = p.patient_id.map(paid_dx_count).fillna(0).ge(1)
p["diagnosed_2plus"] = p.patient_id.map(paid_dx_count).fillna(0).ge(2)
p["age_eligible"] = p.age_band.isin(["35-49", "50-64", "65+"])
p["treated"] = p.patient_id.map(net_fills).fillna(0).gt(0)
p["untreated"] = ~p.treated
print("patient rows:", len(p))
```

    patient rows: 20000


## 4.1 One Disease, One Medicine, Four Market Sizes


```python
sizes = pd.DataFrame([
    ("True condition (answer key)", p.true_condition.sum()),
    ("Launch diagnosis coded",      p.coded.sum()),
    ("Age-eligible diagnosed",      (p.diagnosed & p.age_eligible).sum()),
    ("Untreated opportunity",       (p.diagnosed & p.age_eligible & p.untreated).sum()),
], columns=["market_size", "patients"])
print(sizes.to_string(index=False))
```

                    market_size  patients
    True condition (answer key)      9308
         Launch diagnosis coded      8213
         Age-eligible diagnosed      6998
          Untreated opportunity      2278



```python
eligible = p.diagnosed & p.age_eligible
print(f"age-eligible diagnosed: {eligible.sum()}")
print(f"treated within cohort:  {(eligible & p.treated).sum()}")
print(f"untreated opportunity:  {(eligible & p.untreated).sum()}")
```

    age-eligible diagnosed: 6998
    treated within cohort:  4720
    untreated opportunity:  2278


![Four rows of dots show the share of the true-condition population retained after diagnosis coding, age eligibility, and untreated status.](assets/figures/figure_4_1_market_sizes.svg)

*Figure 4.1. Each row starts from the same 9,308 true-condition patients. Colored dots show the share retained as the market question becomes more specific. Synthetic data.*

## 4.2 What Claims Can See and Miss


```python
gates = pd.DataFrame([
    ("True condition (answer key)",               p.true_condition.sum()),
    ("... coded with a launch diagnosis",         (p.true_condition & p.coded).sum()),
    ("False positives (other condition, flagged)",(~p.true_condition & p.coded).sum()),
], columns=["gate", "patients"])
print(gates.to_string(index=False))

true_n = int(p.true_condition.sum())
print(f"\ncoded share: {100*(p.true_condition & p.coded).sum()/true_n:.1f}%")
```

                                          gate  patients
                   True condition (answer key)      9308
             ... coded with a launch diagnosis      8058
    False positives (other condition, flagged)       155
    
    coded share: 86.6%



```python
invisible = p[p.true_condition & ~p.diagnosed & ~p.treated].sort_values("patient_id")
pid = invisible.iloc[0].patient_id
print("first invisible true patient:", pid)
print()
print("Patient table")
print(p.loc[p.patient_id.eq(pid)].to_string(index=False))
print()
print("Medical claims table")
print(mc.loc[mc.patient_id.eq(pid), ["claim_date"] + DX_COLS[:3]].to_string(index=False))
print()
print("Rx table")
rx_view = rx.loc[rx.patient_id.eq(pid), ["date_of_service", "drug_name", "transaction_type", "reject_code"]].copy()
rx_view["reject_code"] = rx_view["reject_code"].apply(
    lambda x: "" if pd.isna(x) else f"{int(x)}" if float(x).is_integer() else f"{x}"
)
print(rx_view.to_string(index=False))
```

    first invisible true patient: PAT00046
    
    Patient table
    patient_id state    region age_band sex  true_launch_condition payer_id  true_condition  coded  diagnosed  diagnosed_2plus  age_eligible  treated  untreated
      PAT00046    NY Northeast    18-34   F                   True   PAY002            True  False      False            False         False    False       True
    
    Medical claims table
    claim_date diagnosis_1 diagnosis_2 diagnosis_3
    2024-06-28       F41.9     J45.909       E78.5
    
    Rx table
    date_of_service drug_name transaction_type reject_code
         2024-06-09    Vexpro           PENDED          70
         2024-06-13    Vexpro             PAID            
         2024-07-06    Vexpro             PAID            


## 4.3 How Much Evidence to Require? One Diagnosis or Two?


```python
def diagnostics(flag, truth):
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    return dict(TP=tp, FP=fp, FN=fn,
               sensitivity_pct=round(100 * tp / (tp + fn), 1),
               precision_pct=round(100 * tp / (tp + fp), 1),
               )

diag = pd.DataFrame([
    {"rule": "1+ paid diagnosis",  **diagnostics(p.diagnosed,       p.true_condition)},
    {"rule": "2+ paid diagnoses",  **diagnostics(p.diagnosed_2plus, p.true_condition)},
])
print(diag.to_string(index=False))
print(f"\nstrict rule correctly rejected {int((p.diagnosed & ~p.true_condition).sum())} false positive patients.")
print(f"at the cost of missed {int((p.diagnosed & p.true_condition).sum() - (p.diagnosed_2plus & p.true_condition).sum())} true patients.")
```

                 rule   TP  FP   FN  sensitivity_pct  precision_pct
    1+ paid diagnosis 8058 155 1250             86.6           98.1
    2+ paid diagnoses 6008   0 3300             64.5          100.0
    
    strict rule correctly rejected 155 false positive patients.
    at the cost of missed 2050 true patients.


![Two dot panels compare true positives, false negatives, false positives, and true negatives for one-claim and two-claim diagnosis rules, with sensitivity and PPV shown below.](assets/figures/figure_4_2_phenotype_tradeoff.svg)

*Figure 4.2. Requiring at least two diagnoses removes the 155 false positives but misses 2,050 true patients. Synthetic data.*

## 4.4 Anchor the Denominator and Build the Opportunity Funnel


```python
US_ADULTS = 258_554_106     # 2024 Census resident population age 20+
PREVALENCE = 0.113          # NCHS Data Brief 516: diagnosed diabetes, adults 20+

panel_share = p.diagnosed.mean()
print(f"panel diagnosis share:      {panel_share:.1%}  ({p.diagnosed.sum()} of {len(p)})")
print(f"diagnosed diabetes rate:    {PREVALENCE:.1%}")
print(f"US adults age 20+:          {US_ADULTS:,.0f}")
print(f"external prevalence anchor:  {US_ADULTS:,.0f} * {PREVALENCE:.1%} = {PREVALENCE * US_ADULTS:,.0f}")
```

    panel diagnosis share:      41.1%  (8213 of 20000)
    diagnosed diabetes rate:    11.3%
    US adults age 20+:          258,554,106
    external prevalence anchor:  258,554,106 * 11.3% = 29,216,614



```python
access = pd.read_csv(f"{DATA}/market_access/market_access_rules.csv",
                     parse_dates=["effective_start", "effective_end"])
ANALYSIS_DATE = pd.Timestamp("2024-12-31")
rules = access[access.product_name.eq(PRODUCT)
               & access.effective_start.le(ANALYSIS_DATE)
               & access.effective_end.ge(ANALYSIS_DATE)]
preview = (access[access.effective_start.le(ANALYSIS_DATE)
                  & access.effective_end.ge(ANALYSIS_DATE)]
           [["payer_id", "region", "product_name", "coverage_status"]]
           .drop_duplicates()
           .sort_values(["coverage_status", "product_name", "payer_id", "region"])
           .drop_duplicates("coverage_status")
           [["payer_id", "region", "product_name", "coverage_status"]])
print(preview.to_string(index=False))
```

    payer_id  region product_name        coverage_status
      PAY002 Midwest       Vexpro                Covered
      PAY003 Midwest      Nexoral        Covered with PA
      PAY002 Midwest      Nexoral Covered with Step Edit
      PAY001 Midwest      Nexoral            Non-covered



```python
ACCESS_PROBABILITY = {
    "Covered": 0.90,
    "Covered with Step Edit": 0.75,
    "Covered with PA": 0.65,
    "Non-covered": 0.10,
}
```


```python
p = p.merge(rules[["payer_id", "region", "coverage_status"]],
            on=["payer_id", "region"], how="left")
p["access_probability"] = p.coverage_status.map(ACCESS_PROBABILITY).fillna(0.0)

elig = p.diagnosed & p.age_eligible
untr = elig & p.untreated
diagnosed_pop  = PREVALENCE * US_ADULTS
age_eligible_rate = elig.sum() / p.diagnosed.sum()
untreated_rate = untr.sum() / elig.sum()
reachable_rate = p.loc[untr, "access_probability"].mean()
age_pop        = diagnosed_pop * age_eligible_rate
untreated_pop  = age_pop * untreated_rate
reachable      = untreated_pop * reachable_rate
expected_starts = reachable * 0.25

funnel = pd.DataFrame([
    ("Diagnosed population",        diagnosed_pop),
    ("Age eligible",               age_pop),
    ("Untreated opportunity",      untreated_pop),
    ("Reachable (access-adjusted)", reachable),
    ("Expected starts (25% assumed)", expected_starts),
], columns=["stage", "population"])
funnel["population"] = funnel["population"].map(lambda v: f"{v:,.0f}")
print(funnel.to_string(index=False))
```

                            stage population
             Diagnosed population 29,216,614
                     Age eligible 24,894,419
            Untreated opportunity  8,103,671
      Reachable (access-adjusted)  4,188,972
    Expected starts (25% assumed)  1,047,243


![A horizontal funnel declines from 29.2 million diagnosed patients to 1.0 million expected starts, with a marked boundary where weighted counts become expected values.](assets/figures/figure_4_3_market_funnel.svg)

*Figure 4.3. The first three stages are calculated counts from the panel and national anchors. The last two are based on access and conversion assumptions. Synthetic analysis with public anchors.*

## 4.5 Estimate the Unobserved Population


```python
import math

def chapman(source_a, source_b):
    n1, n2 = len(source_a), len(source_b)
    m = len(source_a & source_b)
    n_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    z = 1.96
    sigma = ((1 / (m + 0.5))
             + (1 / (n2 - m + 0.5))
             + (1 / (n1 - m + 0.5))
             + ((m + 0.5) / ((n1 - m + 0.5) * (n2 - m + 0.5)))) ** 0.5
    base = n2 + n1 - m - 0.5
    scale = ((n2 - m + 0.5) * (n1 - m + 0.5)) / (m + 0.5)
    ci_low = base + scale * math.exp(-z * sigma)
    ci_high = base + scale * math.exp(z * sigma)
    return n1, n2, m, n_hat, ci_low, ci_high

paid_dx = set(coded.patient_id)
paid_rx = rx[rx.transaction_type.eq("PAID")]
source_b = set(paid_rx[paid_rx.drug_name.eq(PRODUCT)].patient_id)
n1, n2, m, n_hat, ci_low, ci_high = chapman(paid_dx, source_b)
print(f"n1={n1} n2={n2} m={m}")
print(f"Chapman={n_hat:,.0f}  CI=[{int(ci_low):,}, {int(ci_high):,}]")
```

    n1=8213 n2=6401 m=5541
    Chapman=9,488  CI=[9,438, 9,543]


![A large estimated-population circle contains overlapping paid-diagnosis and paid-Roventra sources, their 5,541-patient overlap, and an estimated 415 patients missed by both.](assets/figures/figure_4_4_capture_recapture.svg)

*Figure 4.4. The overlap between two sources estimates how many patients both miss. Synthetic data.*

## 4.6 Structural Under-Observation


```python
SDOH_OUT = "ch04_market/assets/generated_outputs"
market = pd.read_csv(f"{SDOH_OUT}/sdoh_market_summary.csv")
print(market.to_string(index=False))
```

     sdoh_barrier_quintile  areas  true_condition_patients  observed_diagnosed  estimated_unobserved  observed_share_pct
                         1      4                      166                 129                    37                77.7
                         2      4                      187                 134                    53                71.7
                         3      4                      216                 131                    85                60.6
                         4      4                      184                  99                    85                53.8
                         5      4                      202                  97                   105                48.0


![Observed diagnosis share and estimated unobserved patients by SDOH barrier quintile.](assets/figures/figure_4_5_sdoh_under_observation.svg)

*Figure 4.5. Observed diagnosis share falls and estimated unobserved patients rise as SDOH barrier quintile increases. Synthetic area-level data.*

## 4.7 Patient Finding: From Count to List


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

lab = pd.read_csv(f"{DATA}/claims_lab/lab_results.csv")
a1c = lab[lab.test_name.eq("Hemoglobin A1c")]
diabetes_rx_patients = rx[
    rx.diagnosis_code.str.startswith("E11", na=False)
    & rx.transaction_type.eq("PAID")
].patient_id

f = p[["patient_id", "age_band", "region", "sex", "diagnosed", "true_condition"]].copy()
f["n_class_fills"] = f.patient_id.map(
    paid_rx[paid_rx.drug_name.isin([PRODUCT, "Nexoral", "Vexpro"])].groupby("patient_id").size()
).fillna(0)
f["max_a1c"] = f.patient_id.map(a1c.groupby("patient_id")["result"].max()).fillna(0)
f["has_elevated_a1c"] = (f.max_a1c >= 6.5).astype(int)
f["diabetes_rx_proxy"] = f.patient_id.isin(diabetes_rx_patients).astype(int)

X = pd.get_dummies(f.drop(columns=["patient_id", "diagnosed", "true_condition"]),
                   columns=["age_band", "region", "sex"])
Xtr, Xte, ytr, yte = train_test_split(X, f.diagnosed.astype(int),
                                      test_size=0.3, random_state=20260613, stratify=f.diagnosed)
clf = GradientBoostingClassifier(random_state=20260613).fit(Xtr, ytr)
f["score"] = clf.predict_proba(X)[:, 1]
print("held-out AUC (predicting the confirmed condition):",
      round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3))
```

    held-out AUC (predicting the confirmed condition): 0.93



```python
undx = f[f.diagnosed.eq(0)].sort_values("score", ascending=False)
base = undx.true_condition.mean()
print(f"undiagnosed patients: {len(undx):,}")
print(f"truly positive among them: {int(undx.true_condition.sum()):,} ({100*base:.1f}%)")
for frac in (0.10, 0.20):
    top = undx.head(int(len(undx) * frac))
    print(f"top {int(frac*100):>2}%: {int(top.true_condition.sum()):,} true "
          f"({100*top.true_condition.mean():.1f}%), lift {top.true_condition.mean()/base:.1f}x")
print("PAT00046 percentile among undiagnosed:",
      round(100 * (undx.score < undx.set_index('patient_id').loc['PAT00046', 'score']).mean(), 1))
```

    undiagnosed patients: 11,787
    truly positive among them: 1,250 (10.6%)
    top 10%: 1,177 true (99.9%), lift 9.4x
    top 20%: 1,209 true (51.3%), lift 4.8x
    PAT00046 percentile among undiagnosed: 94.4



```python
SDOH_OUT = "ch04_market/assets/generated_outputs"
summary = pd.read_csv(f"{SDOH_OUT}/sdoh_model_summary.csv").set_index("model")
top_decile = pd.read_csv(f"{SDOH_OUT}/sdoh_model_top_decile.csv")

for label in ["Claims only", "Claims + SDOH"]:
    row = summary.loc[label]
    print(f"{label}: AUC {row['auc']:.3f}, "
          f"true selected {int(row['true_patients_selected'])}, "
          f"Q4-Q5 true {int(row['high_barrier_true_patients'])} "
          f"({row['high_barrier_selected_share_pct']:.1f}% of top decile)")
print("True undiagnosed patients in top decile, by SDOH quintile:")
print(top_decile[["sdoh_barrier_quintile",
                   "claims_only_true", "claims_sdoh_true"]].to_string(index=False))
```

    Claims only: AUC 0.772, true selected 104, Q4-Q5 true 36 (34.3% of top decile)
    Claims + SDOH: AUC 0.805, true selected 110, Q4-Q5 true 64 (63.5% of top decile)
    True undiagnosed patients in top decile, by SDOH quintile:
     sdoh_barrier_quintile  claims_only_true  claims_sdoh_true
                         1                21                10
                         2                25                16
                         3                22                20
                         4                17                26
                         5                19                38


![Claims plus SDOH selects more true undiagnosed patients from high-barrier quintiles than the claims-only model.](assets/figures/figure_4_6_sdoh_model_lift.svg)

*Figure 4.6. SDOH features redirect the fixed-size patient-finding action list toward high-barrier areas. Synthetic area-level data.*

## 4.8 From a Scored List to a Commercial Action


```python
hcp_targets = pd.read_csv(f"{DATA}/reference/hcp_targets.csv")
provider_events = pd.concat([
    mc[["patient_id", "rendering_npi"]].rename(columns={"rendering_npi": "npi"}),
    rx[["patient_id", "prescriber_npi"]].rename(columns={"prescriber_npi": "npi"}),
], ignore_index=True).dropna()
top_decile = undx.head(int(len(undx) * 0.10))[["patient_id", "score"]]
hcp_output = (provider_events.merge(top_decile, on="patient_id")
              .merge(hcp_targets, on="npi", how="inner")
              .groupby(["npi", "account_id", "territory", "state", "specialty_1"])
              .agg(high_score_patients=("patient_id", "nunique"),
                   mean_score=("score", "mean"))
              .reset_index()
              .sort_values(["high_score_patients", "mean_score", "npi"],
                           ascending=[False, False, True])
              .head(10))
hcp_output["mean_score"] = hcp_output.mean_score.map(lambda v: f"{v:.3f}")
print(hcp_output[["npi", "specialty_1", "territory", "state",
                  "high_score_patients", "mean_score", "account_id"]].to_string(index=False))
```

           npi   specialty_1 territory state  high_score_patients mean_score account_id
    9000000249      Oncology       T04    FL                    9      0.879     ACC147
    9000000617  Primary Care       T02    PA                    8      0.867     ACC169
    9000000026 Endocrinology       T03    WA                    8      0.855     ACC226
    9000000506      Oncology       T05    IL                    8      0.852     ACC172
    9000000640  Primary Care       T06    FL                    6      0.879     ACC005
    9000000160    Cardiology       T08    TX                    6      0.876     ACC207
    9000000616      Oncology       T07    CA                    6      0.867     ACC046
    9000000665      Oncology       T06    NJ                    6      0.850     ACC085
    9000000469 Endocrinology       T02    IL                    6      0.846     ACC121
    9000000520  Primary Care       T07    FL                    5      0.889     ACC110



```python
audience_seed = undx.head(10)[[
    "patient_id", "score", "age_band", "region", "sex",
    "n_class_fills", "max_a1c", "diabetes_rx_proxy",
]].copy()
audience_seed["score"] = audience_seed.score.map(lambda v: f"{v:.3f}")
audience_seed["max_a1c"] = audience_seed.max_a1c.map(lambda v: f"{v:.1f}")
print(audience_seed.to_string(index=False))
```

    patient_id score age_band    region sex  n_class_fills max_a1c  diabetes_rx_proxy
      PAT19069 0.980    50-64 Northeast   F            3.0    11.6                  1
      PAT06732 0.960      65+ Northeast   M            1.0    11.2                  1
      PAT13709 0.944    50-64     South   F            5.0    11.4                  1
      PAT11493 0.937    35-49     South   F            2.0    10.5                  1
      PAT14943 0.926    18-34   Midwest   M            1.0    11.2                  1
      PAT12161 0.924    35-49   Midwest   F            5.0    10.5                  1
      PAT05955 0.924    50-64     South   F            0.0    10.4                  1
      PAT19399 0.922    18-34     South   F            2.0     8.0                  1
      PAT04869 0.921    18-34     South   F            3.0    10.1                  1
      PAT06198 0.919    35-49     South   F            1.0    11.4                  1


## 4.9 The Market-Sizing Question is Answered

| Stage | Estimate | Main input | Lever |
| --- | ---: | --- | --- |
| Diagnosed population | 29,216,614 | External prevalence and Census | Diagnosis programs |
| Age eligible | 24,894,419 | Age rule | Label and indication |
| Untreated opportunity | 8,103,671 | Net Roventra treatment state | Competitive share |
| Reachable opportunity | 4,188,972 | Assumed access probabilities | Payer access |
| Expected starts | 1,047,243 | Assumed 25% conversion scenario | Conversion and field |

PAT00046 sits outside the coded and treated rows. The diagnosis-based filter misses her, capture-recapture counts patients like her, and the patient-finding model ranks her for review.

## 4.10 Summary

The chapter starts with 2 market size numbers and makes the definitions explicit. The external anchor gives the national diagnosed population. The panel supplies age eligibility, untreated status, access probabilities, and the ranked patient-finding list. The reusable artifact is the bridge from denominator to action, with each assumption visible.
