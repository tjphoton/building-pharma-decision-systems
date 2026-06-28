# Chapter 7: Exercise Solutions

Each solution ends with the judgment that should accompany the calculation in real data.



```python
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "ch07_competitive").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ch07_competitive.scripts.run_analysis import run_analysis  # noqa: E402

results = run_analysis(ROOT)
print("Loaded Chapter 7 evidence package.")

```

    Loaded Chapter 7 evidence package.


## Exercise 1: Rebuild covered lives



```python
sample = results["policy_landscape"].query(
    "product_name == 'Roventra'"
).head(4)
total = sample.enrolled_lives.sum()
covered = sample.loc[sample.workable_coverage, "enrolled_lives"].sum()
unrestricted = sample.loc[sample.unrestricted, "enrolled_lives"].sum()
quality = (sample.enrolled_lives * sample.access_quality_weight).sum() / total
print(f"plan coverage: {sample.workable_coverage.mean():.1%}")
print(f"covered lives: {covered:,} of {total:,} ({covered/total:.1%})")
print(f"unrestricted lives: {unrestricted:,} of {total:,} ({unrestricted/total:.1%})")
print(f"access-quality score: {quality:.3f}")

```

    plan coverage: 100.0%
    covered lives: 1,576,000 of 1,576,000 (100.0%)
    unrestricted lives: 0 of 1,576,000 (0.0%)
    access-quality score: 0.750


**Methods note:** Contracting review should lead with covered and restricted lives. The access-quality score remains a scenario-weighted supplement.


## Exercise 2: Trace an attempt



```python
attempts = results["prescription_attempts"]
patient = attempts.loc[attempts.had_pend, "patient_id"].iloc[0]
trace = attempts.loc[attempts.patient_id.eq(patient), [
    "patient_id", "fill_number", "first_submission_date",
    "last_transaction_date", "transaction_rows", "had_pend", "final_outcome",
]]
print(trace.reset_index(drop=True))

```

      patient_id  fill_number first_submission_date last_transaction_date  \
    0   PAT00001            0            2024-10-14            2024-10-18   
    1   PAT00001            1            2024-11-22            2024-11-22   
    2   PAT00001            2            2024-12-25            2024-12-25   
    3   PAT00001            3            2025-01-21            2025-01-21   
    
       transaction_rows  had_pend final_outcome  
    0                 2      True     Completed  
    1                 1     False     Completed  
    2                 1     False     Completed  
    3                 1     False     Completed  


**Methods note:** Count the grouped attempt once. Counting every transaction row overstates access friction.


## Exercise 3: Change the operating benchmark



```python
from ch07_competitive.scripts.decomposition import payer_region_decisions  # noqa: E402

baseline = results["payer_region_decisions"].action.value_counts()
alternative = payer_region_decisions(
    results["competitive_start_evidence"], results["policy_landscape"],
    results["access_friction"], brand="Roventra", benchmark=0.80,
    min_treated=30, posterior_threshold=0.80,
    restricted_lives_threshold=0.25, friction_threshold=0.15,
    rule_version="exercise-80pct", analysis_date="2024-12-31",
).action.value_counts()
print("82% benchmark")
print(baseline)
print("\n80% benchmark")
print(alternative)

```

    82% benchmark
    action
    Access review      19
    Sustain            10
    Adoption review     2
    Dual workstream     1
    Name: count, dtype: int64
    
    80% benchmark
    action
    Access review      19
    Sustain            12
    Dual workstream     1
    Name: count, dtype: int64


**Methods note:** The source evidence stays fixed. The action changes because the operating benchmark changed, so the output must retain the rule version.

