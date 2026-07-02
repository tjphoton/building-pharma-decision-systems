# Next Best Action: Exercise Solutions

These solutions use the same generated results as the walkthrough and end with the production judgment an analyst should make.



```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ch09_nba.scripts.next_best_action import run_analysis  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
results = run_analysis(ROOT)
print(f"Recommendations: {len(results['recommendations'])}")
print(f"Candidates after content expansion: {len(results['candidate_audit'])}")

```

    Recommendations: 158
    Candidates after content expansion: 1896


## 1. Content asset that passes but cannot release



```python
trace = results["hcp0280_content_trace"]
blocked_after_content = trace.loc[
    trace["content_gate_reason"].eq("Passed") & trace["eligible"].eq(False),
    ["candidate_action", "content_id", "content_gate_reason", "eligible"],
]
print(blocked_after_content.to_string(index=False))

```

      candidate_action            content_id content_gate_reason  eligible
    Field conversation    CNT_FIELD_GUIDE_01              Passed     False
    Program invitation CNT_PROGRAM_INVITE_01              Passed     False


Judgment: passing the content gate is necessary but not sufficient. Field conversation still needs the priority tier, and program invitation still needs the live-program signal.


## 2. Stable program rows across 3 rankings



```python
program = results["reward_candidates"].loc[
    results["reward_candidates"]["candidate_action"].eq("Program invitation")
].copy()
top_response = set(program.nsmallest(10, "rank_by_response")["npi"])
top_uplift = set(program.nsmallest(10, "rank_by_uplift")["npi"])
top_value = set(program.nsmallest(10, "rank_by_value")["npi"])
stable = sorted(top_response & top_uplift & top_value)
print(f"stable across all 3 top-10 lists: {len(stable)}")
print(stable)

```

    stable across all 3 top-10 lists: 3
    ['9000000174', '9000000232', '9000000239']


Judgment: when capacity is capped, value ranking is the right release rule because it accounts for incremental effect, cost, and fatigue risk.


## 3. Fields needed before replay is trusted



```python
controls = results["model_risk_controls"]
print(controls.to_string(index=False))

```

               control                          failure_it_catches                              release_requirement
     Policy versioning               Unknown rule set in execution policy_version and rule_set_version on every row
         Content audit             Expired or wrong-audience asset            approved content ID with active dates
    Propensity logging                     Unusable offline replay  logged action probability when exploration runs
        Overlap review Candidate policy outside historical support   match rate, ESS, and max weight before rollout
    Execution feedback       Recommendations ignored or overridden         status and override reason after release


Judgment: the replay needs decision-time content eligibility, consent status, eligible action set, logged action probability, outcome window, execution status, and override reason. Missing any of those fields weakens the evidence.

