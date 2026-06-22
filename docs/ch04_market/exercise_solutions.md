# Chapter 4 Exercise Solutions

Try the exercises in `ch04_market_sizing.md` first. Each solution includes the
calculation and the judgment that belongs in a methods note. Run the walkthrough setup
block below first.


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


## Exercise 1: Pick an index date


```python
mc_dates = pd.read_csv(f"{DATA}/claims_medical/medical_claims_mature.csv", parse_dates=["claim_date"])
dx_mask_dates = mc_dates[DX_COLS].isin(LAUNCH_CODES).any(axis=1)
paid_launch = mc_dates[dx_mask_dates]
first_dx = paid_launch.groupby("patient_id").claim_date.min()
p["incident_h2"] = p.patient_id.map(first_dx).ge("2024-07-01")

for label, mask in [("prevalent (diagnosed in 2024)", p.diagnosed),
                    ("incident (first dx in H2 2024)", p.diagnosed & p.incident_h2)]:
    eligible = mask & p.age_eligible
    untreated = eligible & p.untreated
    print(f"{label:<32} diagnosed={int(mask.sum()):>5}  "
          f"age-eligible={int(eligible.sum()):>5}  untreated(panel)={int(untreated.sum()):>5}")
```

    prevalent (diagnosed in 2024)    diagnosed= 8213  age-eligible= 6998  untreated(panel)= 2278
    incident (first dx in H2 2024)   diagnosed= 2275  age-eligible= 1927  untreated(panel)=  648


**Methods note:** The two cohorts answer different commercial questions. The prevalent
cohort sizes the total treatable pool that supports the steady-state forecast. The incident
cohort sizes the flow of new patients per period, which is the right denominator for a
launch's near-term starts. They need different external anchors, prevalence versus
incidence, so do not calibrate the incident cohort to the prevalence number from
Section 4.4. Choosing the index window is the same decision that defines a line of therapy
in Chapter 5.

## Exercise 2: Change the access date


```python
ACCESS_PROBABILITY = {
    "Covered": 0.90,
    "Covered with Step Edit": 0.75,
    "Covered with PA": 0.65,
    "Non-covered": 0.10,
}
access = pd.read_csv(f"{DATA}/market_access/market_access_rules.csv",
                     parse_dates=["effective_start", "effective_end"])
ANALYSIS_DATE = pd.Timestamp("2024-12-31")
rules = access[access.product_name.eq(PRODUCT)
               & access.effective_start.le(ANALYSIS_DATE)
               & access.effective_end.ge(ANALYSIS_DATE)]
p2 = p.merge(rules[["payer_id", "region", "coverage_status"]], on=["payer_id", "region"], how="left")
p2["access_probability"] = p2.coverage_status.map(ACCESS_PROBABILITY).fillna(0.0)

untr = p2.diagnosed & p2.age_eligible & ~p2.treated
diagnosed_pop = 0.113 * 258_554_106
age_eligible_rate = (p2.diagnosed & p2.age_eligible).sum() / p2.diagnosed.sum()
untreated_rate = untr.sum() / (p2.diagnosed & p2.age_eligible).sum()
untreated_pop = diagnosed_pop * age_eligible_rate * untreated_rate
before = untreated_pop * p2.loc[untr, "access_probability"].mean()
# Midyear policy change: PAY001 moves to non-covered (0.10).
after_prob = p2.access_probability.where(~p2.payer_id.eq("PAY001"), 0.10)
after = untreated_pop * after_prob.loc[untr].mean()
print(f"reachable before: {before:,.0f}")
print(f"reachable after:  {after:,.0f}")
print(f"change:           {after - before:,.0f}")
```

    reachable before: 4,188,972
    reachable after:  3,529,970
    change:           -659,001


**Methods note:** The disease, eligibility, and treatment estimand is unchanged. Only the
access component is time-indexed, so moving the assessment date changes the value of the
reachable estimand and the rule the estimator applies, not the underlying population
definition. This is the as-of-date access logic Chapter 7 develops.

## Exercise 3: Break patient finding on purpose


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Rebuild the feature frame, but add a leaking feature: the count of paid launch
# diagnosis codes. That is exactly the signal the phenotype label is built from.
paid_launch = coded
non_launch = mc[~dx_mask]
f = p[["patient_id", "age_band", "region", "sex", "diagnosed", "true_condition"]].copy()
f["n_medical_claims"] = f.patient_id.map(non_launch.groupby("patient_id").size()).fillna(0)
f["n_distinct_dx"] = f.patient_id.map(
    non_launch[DX_COLS].stack().groupby(level=0).nunique()
).fillna(0)
f["leak_launch_dx"] = f.patient_id.map(paid_launch.groupby("patient_id").size()).fillna(0)  # leakage

X = pd.get_dummies(f.drop(columns=["patient_id", "diagnosed", "true_condition"]),
                   columns=["age_band", "region", "sex"])
Xtr, Xte, ytr, yte = train_test_split(X, f.diagnosed.astype(int), test_size=0.3,
                                      random_state=20260613, stratify=f.diagnosed)
clf = GradientBoostingClassifier(random_state=20260613).fit(Xtr, ytr)
print("held-out AUC with leakage:", round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3))

f["score"] = clf.predict_proba(X)[:, 1]
undx = f[f.diagnosed.eq(0)].sort_values("score", ascending=False)
base = undx.true_condition.mean()
top = undx.head(int(len(undx) * 0.10))
print(f"undiagnosed base rate: {base:.1%}")
print(f"top 10% true rate with leakage: {top.true_condition.mean():.1%}")
```

    held-out AUC with leakage: 1.0
    undiagnosed base rate: 10.6%
    top 10% true rate with leakage: 10.6%


**Methods note:** The leaking feature is the paid launch-diagnosis count, which is exactly
what defines the label. Held-out AUC looks excellent, but every undiagnosed patient has a
leak value of zero by construction, so the model cannot separate them: the top decile
collapses toward the base rate and finds no one new. A model that scores well in validation
can be useless in production when a feature encodes the label. This is the bias-analysis
thinking Chapter 11 develops.
