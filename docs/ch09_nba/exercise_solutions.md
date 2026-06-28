# Next Best Action: Exercise Solutions

Each solution ends with the judgment that belongs in a real-data review.



```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ch09_nba.scripts.next_best_action import run_analysis  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
results = run_analysis(ROOT)
print(f"Recommendations: {len(results['recommendations'])}")
print(f"Candidates: {len(results['action_candidates'])}")

```

    Recommendations: 158
    Candidates: 1106


## 1. Reverse field and email precedence



```python
import ch09_nba.scripts.next_best_action as nba

original = dict(nba.PRECEDENCE)
nba.PRECEDENCE["Field conversation"] = original["Approved email"]
nba.PRECEDENCE["Approved email"] = original["Field conversation"]
swapped_candidates = nba.generate_candidates(results["state"])
swapped = nba.select_recommendations(results["state"], swapped_candidates)
nba.PRECEDENCE.update(original)

merged = results["recommendations"][["npi", "recommended_action"]].merge(
    swapped[["npi", "recommended_action"]],
    on="npi", suffixes=("_base", "_swapped"),
)
changed = merged.recommended_action_base.ne(merged.recommended_action_swapped)
print(f"relationships that change action: {int(changed.sum())}")
print(merged.loc[changed].head().to_string(index=False))

```

    relationships that change action: 25
           npi recommended_action_base recommended_action_swapped
    9000000136      Field conversation             Approved email
    9000000086      Field conversation             Approved email
    9000000389      Field conversation             Approved email
    9000000273      Field conversation             Approved email
    9000000026      Program invitation             Approved email


Judgment: the original ordering puts field conversation ahead of email because the field team carries priority-account execution. Swapping it spends scarce field capacity only after email, which a field leader should authorize explicitly, not a default.


## 2. Rank a tier by uplift



```python
candidates = results["action_candidates"]
field_eligible = candidates.loc[
    candidates.eligible & candidates.candidate_action.eq("Field conversation")
].drop_duplicates("npi").copy()
by_response = set(
    field_eligible.sort_values("predicted_response", ascending=False).head(3).npi
)
by_uplift = set(
    field_eligible.sort_values("estimated_uplift", ascending=False).head(3).npi
)
print(f"field-eligible relationships: {len(field_eligible)}")
print(f"top-3 by response: {sorted(by_response)}")
print(f"top-3 by uplift:   {sorted(by_uplift)}")
print(f"in response top-3 only: {sorted(by_response - by_uplift)}")

```

    field-eligible relationships: 4
    top-3 by response: ['9000000086', '9000000136', '9000000389']
    top-3 by uplift:   ['9000000086', '9000000273', '9000000389']
    in response top-3 only: ['9000000136']


Judgment: a relationship the response ranking calls and the uplift ranking does not is a likely responder regardless of the call. Spending a scarce field slot there buys little incremental change. Rank scarce field slots by uplift and confirm with a holdout.


## 3. Design the precedence test



```python
print(results["experiment_design"].to_string(index=False))

```

                            parameter    value
               Baseline response rate    0.588
            Minimum detectable effect    0.050
                                Power    0.800
                      Two-sided alpha    0.050
       Required relationships per arm 1488.000
    Eligible relationships this cycle  112.000
            Cycles to reach both arms   27.000


Solution: randomize at the HCP-account relationship, because the action is delivered to a relationship. The control arm runs the current precedence; the treatment arm runs the digital-first variant. The primary outcome is meaningful response within the 14-day recommendation window, and the design needs about 1,488 relationships per arm, which means pooling across roughly 27 cycles or several geographies.

Judgment: before trusting the doubly-robust off-policy estimate, require one real-world source the logs do not contain, such as a small live holdout that confirms the logging propensities were recorded correctly.

