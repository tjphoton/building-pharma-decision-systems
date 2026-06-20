"""Build the Chapter 6 walkthrough and exercise-solution notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CHAPTER = Path(__file__).resolve().parents[1]


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells: list[nbf.NotebookNode], prefix: str) -> nbf.NotebookNode:
    for index, cell in enumerate(cells):
        cell["id"] = f"{prefix}-{index:02d}"
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )


def walkthrough() -> nbf.NotebookNode:
    cells = [
        md(
            """
# Chapter 6 Walkthrough: HCP and Account Targeting

This notebook builds the chapter artifact from the Chapter 3 synthetic source tables and the Chapter 5 journey output. The analysis date is December 31, 2024.
"""
        ),
        code(
            """
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
while not (ROOT / "ch06_hcp").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Run this notebook inside the repository.")
    ROOT = ROOT.parent

SCRIPT_DIR = ROOT / "ch06_hcp" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_analysis import load_inputs, write_outputs
from targeting import (
    apply_account_policy,
    build_account_features,
    build_call_plan,
    build_decile_summary,
    build_gate_summary,
    build_hcp_actions,
    build_hcp_features,
    build_territory_summary,
    compare_naive_and_gated,
)
"""
        ),
        md("## 1. Load the source tables"),
        code(
            """
inputs = load_inputs(ROOT)
inventory = pd.DataFrame(
    {"table": inputs.keys(), "rows": [len(frame) for frame in inputs.values()]}
)
display(inventory)
"""
        ),
        md(
            """
The patient journey is the analysis cohort. Medical claims provide the diagnosis-index rendering HCP. The field roster links that HCP to an account, while CRM provides recency and current permission.
"""
        ),
        md("## 2. Build HCP evidence"),
        code(
            """
hcp_features, patient_hcp = build_hcp_features(inputs)
coverage = patient_hcp.patient_id.nunique() / inputs["journeys"].patient_id.nunique()
print(f"Target-roster patients: {patient_hcp.patient_id.nunique():,}")
print(f"Target-roster coverage: {coverage:.1%}")
print(f"HCPs: {hcp_features.npi.nunique():,}")

display(
    hcp_features.nlargest(10, "cohort_patients")[[
        "npi", "account_name", "cohort_patients", "opportunity_patients",
        "recent_contacts", "consent_status"
    ]]
)
"""
        ),
        md(
            """
The target roster covers 43.0% of the journey cohort. The denominator stays visible because the table describes the field roster, not the full provider market.
"""
        ),
        md("## 3. Diagnose volume concentration"),
        code(
            """
hcp_deciles, decile_summary = build_decile_summary(hcp_features)
display(
    decile_summary[[
        "volume_decile", "hcps", "cohort_patients", "opportunity_patients",
        "permitted_hcps", "contactable_share"
    ]]
)
"""
        ),
        md(
            """
Decile 10 carries the largest patient opportunity, while its contactable share is 64.3%. Volume and permission remain separate decision inputs.
"""
        ),
        md("## 4. Apply account gates"),
        code(
            """
account_features = build_account_features(hcp_features, inputs["accounts"])
account_targets = apply_account_policy(account_features)
gate_summary = build_gate_summary(account_targets)

display(gate_summary)
display(
    account_targets[[
        "account_name", "cohort_patients", "opportunity_patients",
        "roventra_share", "contactable_hcps", "access_signal_patients",
        "account_action", "action_reason"
    ]].head(20)
)
"""
        ),
        md(
            """
The gates leave 94 field-eligible accounts. The action table preserves the access-review and hold queues instead of deleting them.
"""
        ),
        md("## 5. Resolve HCP actions and calls"),
        code(
            """
hcp_targets = build_hcp_actions(hcp_deciles, account_targets)
call_plan = build_call_plan(account_targets, hcp_targets)
comparison = compare_naive_and_gated(hcp_targets, call_plan)

display(hcp_targets.hcp_action.value_counts().rename_axis("action").to_frame("hcps"))
display(comparison)
display(call_plan.head(20))
"""
        ),
        md(
            """
The near-term plan contains 69 permitted HCPs and 82 contacts. The equal-sized comparison shows the tradeoff against a top-30 volume list.
"""
        ),
        md("## 6. Audit territory allocation"),
        code(
            """
territory_summary = build_territory_summary(account_targets, call_plan)
display(territory_summary)
"""
        ),
        md(
            """
T03 has 12.1% of field-eligible opportunity and 6.1% of planned calls. This gap belongs in the field-leadership review before release.
"""
        ),
        md("## 7. Validate and write the artifact"),
        code(
            """
assert not account_targets.duplicated("account_id").any()
assert hcp_targets.loc[hcp_targets.hcp_action.eq("Prioritize"), "contact_permitted"].all()
assert call_plan.recommended_calls.gt(0).all()
assert set(call_plan.hcp_action) <= {"Prioritize", "Maintain"}
assert account_targets.loc[account_targets.account_action.eq("Access review"), "field_eligible"].eq(False).all()

results = {
    "patient_hcp": patient_hcp,
    "hcp_features": hcp_features,
    "hcp_targets": hcp_targets,
    "hcp_deciles": hcp_deciles,
    "decile_summary": decile_summary,
    "account_features": account_features,
    "account_targets": account_targets,
    "gate_summary": gate_summary,
    "call_plan": call_plan,
    "territory_summary": territory_summary,
    "plan_comparison": comparison,
}
write_outputs(results, ROOT / "ch06_hcp" / "assets" / "generated_outputs")
print("Validation passed and outputs were written.")
"""
        ),
        md(
            """
## Conclusion

The transparent policy converts patient evidence into account actions, HCP actions, and capacity-aware activity. The artifact keeps held, monitored, and access-review accounts visible for the owners who need them.
"""
        ),
    ]
    return notebook(cells, "ch6w")


def solutions() -> nbf.NotebookNode:
    cells = [
        md(
            """
# Chapter 6 Exercise Solutions

Each solution uses the default synthetic data and ends with a judgment to document for real data.
"""
        ),
        code(
            """
from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "ch06_hcp").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "ch06_hcp" / "scripts"))

from run_analysis import load_inputs
from targeting import (
    apply_account_policy, build_account_features, build_call_plan,
    build_decile_summary, build_hcp_actions, build_hcp_features,
    build_territory_summary, compare_naive_and_gated,
)

inputs = load_inputs(ROOT)
hcp_features, _ = build_hcp_features(inputs)
hcp_deciles, _ = build_decile_summary(hcp_features)
account_features = build_account_features(hcp_features, inputs["accounts"])
"""
        ),
        md("## Exercise 1: Change the evidence floor"),
        code(
            """
rows = []
policies = {}
for minimum in [5, 10, 15]:
    policy = apply_account_policy(account_features, min_account_patients=minimum)
    policies[minimum] = policy.set_index("account_id")["account_action"]
    counts = policy.account_action.value_counts()
    rows.append({"minimum": minimum, **counts.to_dict()})

print(pd.DataFrame(rows).fillna(0).to_string(index=False))
changed = policies[5].ne(policies[15])
print()
print("Example changed account:", changed[changed].index[0])
"""
        ),
        md(
            """
**Methods note:** The threshold controls how much sparse account evidence enters the plan. In real data, document the minimum together with count stability, data completeness, and privacy rules.
"""
        ),
        md("## Exercise 2: Audit the volume trap"),
        code(
            """
account_targets = apply_account_policy(account_features)
hcp_targets = build_hcp_actions(hcp_deciles, account_targets)
call_plan = build_call_plan(account_targets, hcp_targets)

print(compare_naive_and_gated(hcp_targets, call_plan).to_string(index=False))
"""
        ),
        md(
            """
**Methods note:** The gated list is executable under current consent and has less recent saturation. Real deployment still needs approved frequency policy, local review, and a defined measurement plan.
"""
        ),
        md("## Exercise 3: Rebalance one territory"),
        code(
            """
territory = build_territory_summary(account_targets, call_plan)
total_calls = call_plan.recommended_calls.sum()
t03 = territory.set_index("territory").loc["T03"]
minimum_calls = int((t03.opportunity_share - 0.02) * total_calls + 0.999)
move = minimum_calls - int(t03.recommended_calls)
donor = territory.sort_values("allocation_gap", ascending=False).iloc[0].territory

print(f"Move {move} calls from {donor} to T03.")
print(f"T03 call share after move: {(t03.recommended_calls + move) / total_calls:.1%}")
print(f"T03 opportunity share:     {t03.opportunity_share:.1%}")
"""
        ),
        md(
            """
**Methods note:** The territory arithmetic identifies the size of the gap. A production reallocation must select permitted HCPs, respect account and representative capacity, and record which lower-priority activity is displaced.
"""
        ),
    ]
    return notebook(cells, "ch6e")


def main() -> None:
    nbf.write(walkthrough(), CHAPTER / "chapter6_walkthrough.ipynb")
    nbf.write(solutions(), CHAPTER / "exercise_solutions.ipynb")
    print(f"Wrote Chapter 6 notebooks to {CHAPTER}")


if __name__ == "__main__":
    main()
