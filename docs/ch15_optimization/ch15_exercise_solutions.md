# Chapter 15 Exercise Solutions: Resource Optimization Under Uncertainty

Worked answers to the 3 exercises. Each solution reuses the released analysis and changes one rule at a time.



```python
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "ch15_optimization").exists():
    ROOT = ROOT.parent
SCRIPT_DIR = ROOT / "ch15_optimization" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_analysis import run_analysis

results = run_analysis()
planning = results["planning"]
territories = results["territories"]
gain_draws = results["_gain_draws"]
print(f"territories: {len(territories)}, accounts: {len(planning)}")

```

    territories: 12, accounts: 300


## Exercise 1

Raise the movement cap from 20% to 35% and re-solve the field MILP. Report the predicted NRx recovered, the account changes, and the new plan change.



```python
import numpy as np
from allocation import expected_gain_milp_plan, value_from_gains, plan_change_calls

base, _ = expected_gain_milp_plan(planning, gain_draws, territories, movement_cap_share=0.20)
wider, _ = expected_gain_milp_plan(planning, gain_draws, territories, movement_cap_share=0.35)
current = planning["current_calls"].to_numpy()
mean_gains = gain_draws.mean(axis=0)

recovered = value_from_gains(mean_gains, wider) - value_from_gains(mean_gains, base)
base_changes = int((np.round(base) != current).sum())
wider_changes = int((np.round(wider) != current).sum())
print(f"predicted NRx recovered by the wider cap: {recovered:.1f}")
print(f"accounts changed: {base_changes} at 20% -> {wider_changes} at 35%")
print(f"plan change: {plan_change_calls(planning, base):.0f} -> {plan_change_calls(planning, wider):.0f} calls")

```

    predicted NRx recovered by the wider cap: 13.8
    accounts changed: 149 at 20% -> 196 at 35%
    plan change: 534 -> 889 calls


The wider cap recovers modeled response but disrupts more of the field. Approval depends on whether district teams can absorb the added account movement in one quarter.


## Exercise 2

Change the selection rule to the lowest-change plan within a 0.5% value band instead of 3%. Report which plan it picks and how much more the field has to move.



```python
from allocation import select_stable_plan

frontier = results["frontier_solutions"]
band_3 = select_stable_plan(frontier, value_band=0.03)
band_half = select_stable_plan(frontier, value_band=0.005)
for band, sel in [("3.0%", band_3), ("0.5%", band_half)]:
    print(f"{band} band -> {sel['selected_plan_id']}: "
          f"{sel['plan_change_pct']:.1f}% change, {sel['value_vs_best_pct']:.2f}% vs best")

```

    3.0% band -> change_20_expected: 20.0% change, -1.19% vs best
    0.5% band -> change_35_expected: 33.3% change, 0.00% vs best


The tighter 0.5% band forces a higher-change plan because a lower-change plan no longer stays within 0.5% of the best expected value. The tighter value guarantee requires more field movement.


## Exercise 3

Reduce the study cost to $50, then recompute the value of sample information. Does the decision change?



```python
from allocation import two_stage_reserve_policy, value_of_sample_information
from allocation_config import (
    RESERVE_CALL_SHARE, STUDY_SIGNAL_NOISE_SD, STUDY_FOREGONE_NRX,
    N_LEARNING_TRIALS, SEED_LEARNING,
)

selected = results["selected_plan"].iloc[0]
reference = results["_frontier_plans"][selected["selected_plan_id"]]
comparison = two_stage_reserve_policy(
    planning, gain_draws, territories, reference,
    movement_cap_calls=selected["movement_epsilon_pct"] / 100 * planning["current_calls"].sum(),
    reserve_call_share=RESERVE_CALL_SHARE, study_cost=50.0, study_foregone_nrx=STUDY_FOREGONE_NRX,
    signal_noise_sd=STUDY_SIGNAL_NOISE_SD, n_trials=N_LEARNING_TRIALS, seed=SEED_LEARNING,
)
vosi = value_of_sample_information(comparison)
print(comparison.to_string(index=False))
print()
print(vosi.to_string(index=False))

```

                                 policy  expected_nrx  study_cost_nrx  net_nrx
         Commit best reserve option now         4.624           0.000    4.624
           Hold reserve, no measurement         4.324           0.000    4.324
    Hold reserve, run study, reallocate         4.719           0.029    4.690
      Perfect information (upper bound)         5.032           0.000    5.032
    
    target_segment  reserve_calls  net_learning_value_nrx  net_learning_value_dollars  decision
           Adopter             38                   0.066                       112.0 Run study


The cheaper study can cross the decision threshold. The comparison shows that measurement value depends on the decision it can change, the quality of its signal, the delay cost, and the study cost.

