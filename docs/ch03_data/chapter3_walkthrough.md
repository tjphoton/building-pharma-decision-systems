# Chapter 3 Walkthrough: Synthetic Lab for Real Pharma Questions

This notebook executes every code listing from Chapter 3 in order. Run cells from top to bottom. The first cell generates the synthetic data package; all subsequent outputs are reproducible against the default package (seed `20260609`, 20,000 patients).


```python
import os
import subprocess
from pathlib import Path

# Resolve repository root so the script path works from any launch directory
repo = Path.cwd()
while not (repo / "ch03_data").exists() and repo != repo.parent:
    repo = repo.parent
os.chdir(repo)

print(subprocess.run(
    ["uv", "run", "python", "ch03_data/scripts/generate_all_synthetic_data.py"],
    check=True,
))
```

    Generating data for 20,000 patients, 250 accounts, 666 providers...
      Medical encounters: 49,882 mature / 44,567 early snapshot
      Service lines: 62,285
      Pharmacy claims: 46,055
      Lab results: 55,815
      Patients with Roventra treatment timeline: 6,401
      Formulary change events: 13
      Specialty pharmacy hub referrals: 6,096
      CRM interactions: 3,359
      Territory alignment records: 8
      Digital events: 1,095
      Open Payments records: 783
      CMS Part D records: 1,057
    Done. Data written to /Users/qiu/Projects/hands-on-pharma-decision-science/ch03_data/output_data/generated_data
    CompletedProcess(args=['uv', 'run', 'python', 'ch03_data/scripts/generate_all_synthetic_data.py'], returncode=0)



```python
from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()
if ROOT.name == "ch03_data":
    ROOT = ROOT.parent
DATA    = ROOT / "ch03_data" / "output_data" / "generated_data"
SCRIPTS = ROOT / "ch03_data" / "scripts"

assert DATA.exists(), f"Data directory not found: {DATA}"
print("Data directory:", DATA)
```

    Data directory: /Users/qiu/Projects/hands-on-pharma-decision-science/ch03_data/output_data/generated_data


## 3.3 One Patient through Multiple Data Sources

PAT02034 is a female patient aged 65 or older, insured under commercial payer PAY002 in New Jersey, whose prescribing endocrinologist is HCP0280 at account ACC089. The next sections trace her treatment path through each data source.


![Figure 3.1. PAT02034's treatment path across all four data sources from first encounter through final A1C measurement.](assets/figures/figure-3-1-patient-journey.svg)


**Reference tables: patients, providers, and accounts**

Five reference files establish the universe of patients, providers, accounts, payers, and products before any claims are read.


```python
patients = pd.read_csv(DATA / "reference" / "patients.csv")
print(patients.loc[patients["patient_id"].eq("PAT02034")])
```

         patient_id state     region age_band sex  true_launch_condition
    2033   PAT02034    NJ  Northeast      65+   F                   True



```python
providers = pd.read_csv(DATA / "reference" / "providers.csv")
targets   = pd.read_csv(DATA / "reference" / "hcp_targets.csv")
mc        = pd.read_csv(DATA / "claims_medical" / "medical_claims_mature.csv")

# PAT02034's prescriber comes from claims. Get the NPI that appears on her encounters.
hcp_npi = mc.loc[mc["patient_id"].eq("PAT02034"), "rendering_npi"].mode()[0]

print("Prescriber in vendor clinical directory (providers.csv):")
print(providers.loc[providers["npi"].eq(hcp_npi)])

print("\nPrescriber in internal target list (hcp_targets.csv):")
print(targets.loc[targets["npi"].eq(hcp_npi)])
```

    Prescriber in vendor clinical directory (providers.csv):
                npi    specialty_1 specialty_2 provider_state provider_type  \
    279  9000000280  Endocrinology         NaN             NJ    Individual   
    
        credential  primary_facility_npi  
    279         MD                   NaN  
    
    Prescriber in internal target list (hcp_targets.csv):
                npi account_id territory state     region    specialty_1
    115  9000000280     ACC089       T02    NJ  Northeast  Endocrinology


**Key pattern:** medical claims carry `rendering_npi`. Join that NPI to `providers.csv` for specialty and credential. Join it to `hcp_targets.csv` for territory and `account_id`. Both joins are direct NPI lookups, no intermediate key translation needed. `hcp_targets.csv` is an internal company file covering ~42% of the prescriber universe; providers outside that list appear in claims but are not actively managed by the field team.


```python
# Apply the NPI join pattern across all medical claims
mc = pd.read_csv(DATA / "claims_medical" / "medical_claims_mature.csv")

with_specialty = mc.merge(
    providers[["npi", "specialty_1", "credential"]],
    left_on="rendering_npi", right_on="npi", how="left",
)
with_commercial = with_specialty.merge(
    targets[["npi", "territory", "account_id"]],
    left_on="rendering_npi", right_on="npi", how="left",
)
in_universe_pct = 100 * with_commercial["territory"].notna().mean()
print(f"Mature encounters:                    {len(mc):,}")
print(f"With a matching target-list record:   {in_universe_pct:.1f}%")
```

    Mature encounters:                    49,882
    With a matching target-list record:   42.6%



```python
# Coverage windows: one row per patient-payer-period
enroll = pd.read_csv(
    DATA / "reference" / "patient_enrollments.csv",
    parse_dates=["eligibility_start_date", "eligibility_end_date"],
)
print(enroll.loc[enroll["patient_id"].eq("PAT02034")])
```

         patient_id eligibility_start_date eligibility_end_date payer_id  \
    2033   PAT02034             2023-03-01           2025-02-20   PAY002   
    
          payer_type  has_medical_coverage  has_pharmacy_coverage product_type  
    2033  Commercial                  True                   True          PPO  


**What to notice:** PAT02034 has one continuous coverage period under PAY002. In the full population some patients have multiple rows when they changed plans mid-year. Coverage windows for cohort construction come from this table, not from a single pair of dates on the patient record.

### 3.3.1 Medical claims

Each row in the claims header table is one encounter. Up to ten diagnosis columns (`diagnosis_1` through `diagnosis_10`) carry the ICD-10 codes attached to that visit.


```python
print("Columns:", mc.columns.tolist())
print(f"Rows (mature):  {len(mc):,}")
```

    Columns: ['encounter_id', 'patient_id', 'claim_type', 'claim_date', 'admitting_diagnosis', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4', 'diagnosis_5', 'diagnosis_6', 'diagnosis_7', 'diagnosis_8', 'diagnosis_9', 'diagnosis_10', 'icd_procedure_1', 'icd_procedure_2', 'icd_procedure_3', 'patient_gender', 'patient_state', 'coverage_type', 'rendering_npi', 'attending_npi', 'referring_npi', 'facility_npi', 'payer_id']
    Rows (mature):  49,882



```python
# Listing 3.1: Wide diagnosis filter with .any(axis=1)
dx_cols  = ["admitting_diagnosis"] + [f"diagnosis_{i}" for i in range(1, 11)]
t2d_mask = mc[dx_cols].apply(
    lambda col: col.astype(str).str.startswith("E11") & col.notna()
).any(axis=1)

show = ["encounter_id", "claim_date", "claim_type",
        "rendering_npi", "admitting_diagnosis", "diagnosis_1", "diagnosis_2", "diagnosis_3"]
print((
    mc.loc[mc["patient_id"].eq("PAT02034") & t2d_mask, show]
    .sort_values("claim_date")
    .fillna("")
    .reset_index(drop=True)
))
```

      encounter_id  claim_date     claim_type  rendering_npi admitting_diagnosis  \
    0   ENC0005011  2024-02-17   Professional     9000000280                       
    1   ENC0005013  2024-06-03   Professional     9000000280                       
    2   ENC0005010  2024-08-14  Institutional     9000000280                       
    3   ENC0005012  2024-10-23   Professional     9000000280                       
    
      diagnosis_1 diagnosis_2 diagnosis_3  
    0       E11.9                          
    1       E11.9     J45.909              
    2       E11.9       N18.9              
    3       E11.9     J45.909              



```python
# Service lines: procedure codes and charges joined on encounter_id
sl = pd.read_csv(DATA / "claims_medical" / "service_lines.csv")
print((
    sl.loc[sl["patient_id"].eq("PAT02034"),
           ["encounter_id", "line_number", "service_from",
            "procedure_code", "place_of_service", "line_charge"]]
    .sort_values(["encounter_id", "line_number"])
    .reset_index(drop=True)
))
```

      encounter_id  line_number service_from procedure_code  place_of_service  \
    0   ENC0005010            1   2024-08-14          99214                22   
    1   ENC0005011            1   2024-02-17          96413                11   
    2   ENC0005012            1   2024-10-23          99213                11   
    3   ENC0005012            2   2024-10-23          99215                11   
    4   ENC0005013            1   2024-06-03          99215                11   
    
       line_charge  
    0       491.56  
    1       192.11  
    2      1027.50  
    3       363.92  
    4       521.17  


![Figure 3.2. One encounter in the medical claims header expands to multiple service lines.](assets/figures/figure-3-2-claims-grain.svg)


**What to notice:** CPT 96413 is a chemotherapy administration infusion code. CPT codes 99213, 99214, and 99215 are evaluation and management codes tiered by complexity. Place of service 11 is an office setting; place 22 is an outpatient hospital. Population counts use the header grain. Procedure-level analysis joins `service_lines.csv` to the header on `encounter_id`.

### 3.3.2 Pharmacy claims

`transaction_type` takes three values: PAID (approved and cleared), PENDED (held for review), and REVERSED (a prior payment voided). Each prescription may generate multiple rows. Group on `(prescriber_npi, ndc_prescribed, fill_number)` to identify each fill attempt.


```python
rx  = pd.read_csv(
    DATA / "claims_pharmacy" / "pharmacy_claims.csv",
    dtype={"ndc": str, "ndc_prescribed": str, "reject_code": str},
)
ref = pd.read_csv(DATA / "reference" / "ndc_codes.csv", dtype={"ndc": str})
ndc_map = ref.set_index("ndc")["drug_name"]

# ndc_prescribed carries the prescribed drug code; use it for stable product attribution
rx["drug_name"] = rx["ndc_prescribed"].map(ndc_map)

pat = rx.loc[rx["patient_id"].eq("PAT02034") & rx["drug_name"].eq("Roventra")].copy()
cols = ["claim_id", "date_of_service", "transaction_type",
        "fill_number", "reject_code", "patient_pay"]
print(pat.sort_values("date_of_service")[cols].fillna("").reset_index(drop=True))
```

          claim_id date_of_service transaction_type  fill_number reject_code  \
    0  RXCL0004665      2024-07-02           PENDED            0          70   
    1  RXCL0004666      2024-07-09             PAID            0               
    2  RXCL0004667      2024-08-09             PAID            1               
    3  RXCL0004668      2024-08-10         REVERSED            1               
    4  RXCL0004669      2024-08-15             PAID            1               
    5  RXCL0004670      2024-09-09             PAID            2               
    6  RXCL0004671      2024-10-09             PAID            3               
    
       patient_pay  
    0         0.00  
    1        45.06  
    2        64.88  
    3       -64.88  
    4        64.88  
    5        45.13  
    6        59.90  



```python
# Listing 3.2: Derive completed fills
chains = (
    pat.sort_values(["date_of_service", "claim_id"])
    .groupby(["prescriber_npi", "ndc_prescribed", "fill_number"], sort=False)
    .agg(
        first_date=("date_of_service", "first"),
        final_type=("transaction_type", "last"),
        net_patient_pay=("patient_pay", "sum"),
    )
    .reset_index()
)
chains["completed_fill"] = (
    chains["final_type"].eq("PAID") & chains["net_patient_pay"].ge(0)
)

print(
    chains[["fill_number", "first_date", "final_type",
             "net_patient_pay", "completed_fill"]]
)
print(f"Completed fills: {int(chains.completed_fill.sum())}")
```

       fill_number  first_date final_type  net_patient_pay  completed_fill
    0            0  2024-07-02       PAID            45.06            True
    1            1  2024-08-09       PAID            64.88            True
    2            2  2024-09-09       PAID            45.13            True
    3            3  2024-10-09       PAID            59.90            True
    Completed fills: 4


![Figure 3.3. PAT02034's seven Roventra transactions grouped into four completed fills.](assets/figures/figure-3-3-operational-claims-to-analytical-records.svg)


**What to notice:** fill 0 generates two rows (PENDED then PAID), and fill 1 generates three rows (PAID, REVERSED, PAID). Both resolve to a positive net payment, so both count as completed. Seven raw transactions, four groups, four completed fills. Counting rows gives seven. Counting PAID rows gives five. Only grouping by fill attempt gives the correct four.

### 3.3.3 Lab results

Lab data is delivered by the same longitudinal claims vendor, sourced from reference lab partnerships (Quest, LabCorp). LOINC codes are the standard test identifier. This is not an EHR extract.


```python
lab = pd.read_csv(
    DATA / "claims_lab" / "lab_results.csv",
    parse_dates=["service_date"],
)
print("Columns:", lab.columns.tolist())
print(f"Rows: {len(lab):,}")
```

    Columns: ['lab_id', 'patient_id', 'service_date', 'loinc_code', 'test_name', 'result', 'result_unit', 'ref_low', 'ref_high', 'abnormal_flag', 'ordering_npi', 'diagnosis_1']
    Rows: 55,815



```python
cols = ["lab_id", "service_date", "loinc_code", "test_name",
        "result", "result_unit", "ref_low", "ref_high", "abnormal_flag"]
print((
    lab.loc[lab["patient_id"].eq("PAT02034"), cols]
    .sort_values("service_date").fillna("")
    .reset_index(drop=True)
))
```

           lab_id service_date loinc_code        test_name  result result_unit  \
    0  LAB0005734   2024-06-13     4548-4   Hemoglobin A1c    10.6     percent   
    1  LAB0005735   2024-06-22     4548-4   Hemoglobin A1c     8.9     percent   
    2  LAB0005736   2024-12-30     2089-1  LDL Cholesterol   170.0       mg/dL   
    3  LAB0005737   2025-01-01     4548-4   Hemoglobin A1c     8.4     percent   
    
       ref_low  ref_high abnormal_flag  
    0      4.0       5.6             H  
    1      4.0       5.6             H  
    2      0.0     100.0             H  
    3      4.0       5.6             H  


**What to notice:** PAT02034's A1C of 10.6 percent on June 13 is far above the 6.5 percent ADA diagnostic threshold. The SP referral opens the same day. By the January 2025 test, the A1C has fallen to 8.4 percent.

Two analytical details:
1. Apply a numeric threshold on `result` rather than relying on `abnormal_flag`. The flag reflects each instrument's reference range, which may differ across labs.
2. Patients with no lab record in the data remain eligible for the analysis cohort. The absence of a test result in the data does not mean the condition is absent.

### 3.3.4 Formulary and access

The current formulary status table captures the latest plan rules. For retrospective analysis, each claim must be matched to the rule active on its service date.


```python
fs = pd.read_csv(DATA / "formulary" / "formulary_status.csv")
fh = pd.read_csv(
    DATA / "formulary" / "formulary_history.csv",
    parse_dates=["effective_date"],
)

# PAT02034's payer comes from patient_enrollments, patients.csv holds demographics only
payer = enroll.loc[enroll["patient_id"].eq("PAT02034"), "payer_id"].iloc[0]
print(fs.loc[
    fs["plan_id"].eq(payer) & fs["product_name"].eq("Roventra"),
    ["plan_id", "tier", "prior_authorization", "step_therapy",
     "quantity_limit", "specialty_pharmacy"],
])
```

      plan_id       tier prior_authorization step_therapy quantity_limit  \
    3  PAY002  Specialty                 Yes          Yes            Yes   
    
      specialty_pharmacy  
    3                Yes  



```python
# Listing 3.3: Join claims to the payer rule active on each service date
# PAY005 added a Roventra quantity limit on July 1, 2024

pay005_hist = (
    fh.loc[fh["plan_id"].eq("PAY005") & fh["product_name"].eq("Roventra")]
    .sort_values("effective_date")
)
print("PAY005 Roventra restriction history:")
print(
    pay005_hist[
        ["effective_date", "prior_tier", "new_tier",
         "prior_prior_authorization", "new_prior_authorization",
         "prior_step_therapy", "new_step_therapy",
         "prior_quantity_limit", "new_quantity_limit"]
    ]
)

first = pay005_hist.iloc[0]
states = pd.DataFrame(
    [{"effective_date": pd.Timestamp("2024-01-01"),
      "tier": first["prior_tier"],
      "prior_authorization": first["prior_prior_authorization"],
      "step_therapy": first["prior_step_therapy"],
      "quantity_limit": first["prior_quantity_limit"]}]
    + [
        {"effective_date": r["effective_date"],
         "tier": r["new_tier"],
         "prior_authorization": r["new_prior_authorization"],
         "step_therapy": r["new_step_therapy"],
         "quantity_limit": r["new_quantity_limit"]}
        for _, r in pay005_hist.iterrows()
    ]
).sort_values("effective_date")

claims = pd.DataFrame({
    "claim_date": pd.to_datetime(["2024-05-01", "2024-09-01"])
})

assigned = pd.merge_asof(
    claims.sort_values("claim_date"),
    states,
    left_on="claim_date",
    right_on="effective_date",
    direction="backward",
)
print(assigned)

```

    PAY005 Roventra restriction history:
      effective_date prior_tier   new_tier prior_prior_authorization  \
    6     2024-01-01  Specialty  Specialty                       Yes   
    7     2024-07-01  Specialty  Specialty                       Yes   
    
      new_prior_authorization prior_step_therapy new_step_therapy  \
    6                     Yes                 No               No   
    7                     Yes                 No               No   
    
      prior_quantity_limit new_quantity_limit  
    6                   No                 No  
    7                   No                Yes  
      claim_date effective_date       tier prior_authorization step_therapy  \
    0 2024-05-01     2024-01-01  Specialty                 Yes           No   
    1 2024-09-01     2024-07-01  Specialty                 Yes           No   
    
      quantity_limit  
    0             No  
    1            Yes  


**Key lesson:** reading only the current status table assigns the current rules to every historical PAY005 claim, overstating the restriction burden in the first half. Join each claim to the formulary record whose `effective_date` is at or before the service date.


![Figure 3.4. PAY005 Roventra access rules changed on July 1; claim events are circles and the policy change is a diamond.](assets/figures/figure-3-4-effective-dated-access.svg)


## 3.4 Pre-Analysis Data Checks

Two systematic checks run against the full population: snapshot completeness and NDC mapping gaps. Run `data_quality.py` to execute all checks and write the detailed audit files.



```python
import sys
sys.path.insert(0, str(SCRIPTS))
from data_quality import run_data_quality

_, summary = run_data_quality(DATA)
print(summary)
```

                                   metric   value unit
    0     Missing NDC prescribed mappings    0.00    %
    1         Eligible observation window   52.04    %
    2  Median early-snapshot completeness  100.00    %


### 3.4.1 Claim maturity: the false December decline

Medical claim receipt delay is long-tailed and right-skewed. Most claims arrive quickly, but a meaningful minority arrive weeks later. Compare the January 5 early snapshot with the mature file, then label months that have not cleared a 90% completeness cutoff.



```python
# Listing 3.4: Compare the early snapshot with the mature claim file
providers = pd.read_csv(DATA / "reference" / "providers.csv")
endo_npis = providers.loc[providers["specialty_1"].eq("Endocrinology"), "npi"]

dx_cols = ["admitting_diagnosis"] + [f"diagnosis_{i}" for i in range(1, 11)]

def t2d_endo_by_month(df: pd.DataFrame) -> pd.Series:
    # astype(str) handles all-NaN float64 columns (diagnosis_5 through diagnosis_10)
    t2d_mask = df[dx_cols].apply(
        lambda col: col.astype(str).str.startswith("E11") & col.notna()
    ).any(axis=1)
    return (
        df.loc[t2d_mask & df["rendering_npi"].isin(endo_npis)]
        .assign(month=df["claim_date"].str[:7])["month"]
        .value_counts().sort_index()
    )

early  = pd.read_csv(DATA / "claims_medical" / "medical_claims.csv")
mature = pd.read_csv(DATA / "claims_medical" / "medical_claims_mature.csv")

view = pd.DataFrame({
    "snapshot_jan05": t2d_endo_by_month(early),
    "mature":         t2d_endo_by_month(mature),
}).dropna()
view["completeness_pct"] = (100 * view["snapshot_jan05"] / view["mature"]).round(1)
view["cleared_90pct"] = view["completeness_pct"].ge(90)
print(view.loc["2024-01":"2024-12"])
```

             snapshot_jan05  mature  completeness_pct  cleared_90pct
    month                                                           
    2024-01             236     236             100.0           True
    2024-02             253     253             100.0           True
    2024-03             217     217             100.0           True
    2024-04             242     242             100.0           True
    2024-05             205     205             100.0           True
    2024-06             223     223             100.0           True
    2024-07             225     225             100.0           True
    2024-08             207     207             100.0           True
    2024-09             224     224             100.0           True
    2024-10             227     229              99.1           True
    2024-11             199     213              93.4           True
    2024-12             178     260              68.5          False


**What to notice:** December is 68.5% of its mature count at the January 5 snapshot. January through November 2024 clear the 90% cutoff. December has not cleared, so it should be excluded from this trend view. January 2025 is omitted because the snapshot date is January 5, so the service month has barely materialized.

Two details in the query matter:
1. The T2D filter checks all ten diagnosis columns with `.any(axis=1)`. A filter on `diagnosis_1` alone misses encounters where E11.x appears only in a secondary position.
2. Provider specialty comes from a join to `providers.csv` on NPI (`specialty_1` column), not from a column in the claims file.


![Figure 3.5. Claim receipt lag is right-skewed: about 92% of claims arrive within 30 days of service.](assets/figures/figure-3-5-claim-receipt-lag.svg)


![Figure 3.6. Monthly endocrinology T2D visit counts showing snapshot completeness by month.](assets/figures/figure-3-6-claim-maturity-cutoff.svg)


**Snapshot completeness audit**

The T2D endocrinology count above shows the teaching case. The audit file covers all months and labels the months below the selected cutoff.


```python
dq_dir = DATA.parent / "analysis_results" / "data_quality"
if (dq_dir / "dq_snapshot_comparison.csv").exists():
    snap = pd.read_csv(dq_dir / "dq_snapshot_comparison.csv")
    THRESHOLD = 90.0
    print(f"Months below {THRESHOLD}% completeness:")
    print(snap.loc[snap["completeness_pct"] < THRESHOLD])
else:
    print("Run: uv run python ch03_data/scripts/data_quality.py")
```

    Run: uv run python ch03_data/scripts/data_quality.py


### 3.4.2 NDC mapping gaps

Join `ndc_prescribed` for product attribution. Join `ndc` separately to surface pack-size variants the reference does not yet recognize.


```python
# Listing 3.5: NDC gap analysis
rx  = pd.read_csv(
    DATA / "claims_pharmacy" / "pharmacy_claims.csv",
    dtype={"ndc": str, "ndc_prescribed": str},
)
ref = pd.read_csv(DATA / "reference" / "ndc_codes.csv", dtype={"ndc": str})
ndc_map = ref.set_index("ndc")["drug_name"]

paid = rx.loc[rx["transaction_type"].eq("PAID")].copy()
paid["drug_prescribed"] = paid["ndc_prescribed"].map(ndc_map)
paid["drug_dispensed"]  = paid["ndc"].map(ndc_map)

print("Join on ndc_prescribed (prescribed code) -- all fills attributed:")
print(paid.groupby("drug_prescribed").size().sort_values(ascending=False))

gaps = paid.loc[paid["drug_dispensed"].isna() & paid["drug_prescribed"].notna()]
print(f"\nDispensed NDC absent from reference: {len(gaps):,} of {len(paid):,} "
      f"paid fills ({100*len(gaps)/len(paid):.1f}%)")
```

    Join on ndc_prescribed (prescribed code) -- all fills attributed:
    drug_prescribed
    Roventra          16637
    Supportive Med     7632
    Nexoral            7574
    Vexpro             7398
    dtype: int64
    
    Dispensed NDC absent from reference: 1,692 of 39,241 paid fills (4.3%)


**What to notice:** all fills are attributed correctly via `ndc_prescribed`. The dispensed-NDC gap shows pack-size variants (suffix variants) the reference does not recognize. These are valid dispensings. The reference table needs updating for those codes. Until then, use `ndc_prescribed` for brand share analysis.

**Three conclusions:**

1. Always include patients without a lab result in the cohort denominator. Restricting to tested patients changes the prevalence estimate and overstates the share with elevated A1C.
2. Use a numeric threshold on `result`, not `abnormal_flag`, for clinical classification. The flag reflects the instrument's reference range, which varies by lab.
3. A1C is the strongest discriminator for undiagnosed T2D in the patient-finding model built in the market sizing chapter.

---
*The market sizing chapter uses the patients, claims, and lab results built here to measure market size and locate the undiagnosed opportunity.*
