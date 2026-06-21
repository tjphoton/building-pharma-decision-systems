"""Build the 2 Chapter 7 companion notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CHAPTER_DIR = Path(__file__).resolve().parents[1]


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def _metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }


def walkthrough() -> nbf.NotebookNode:
    """Return the executable chapter walkthrough."""

    cells = [
        md(
            """
# Chapter 7: Competitive Intelligence and Market Access

This notebook executes the chapter as one decision chain. It uses fictional products, patients, payers, accounts, and planted synthetic events.
"""
        ),
        code(
            """
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "ch07_competitive").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ch07_competitive.scripts.run_analysis import run_analysis  # noqa: E402

results = run_analysis(ROOT)
print("Loaded Chapter 7 evidence package.")
"""
        ),
        md(
            "## 7.1 Opening evidence\n\nThe corrected treatment cohort stays aligned with Chapter 5."
        ),
        code(
            """
headline = results["headline"].iloc[0]
print(f"New-to-therapy patients: {int(headline.new_to_therapy_patients):,}")
print(f"Roventra new starts: {int(headline.roventra_new_starts):,}")
print(
    f"Materially restricted lives: {int(headline.restricted_lives):,} "
    f"of {int(headline.total_lives):,} "
    f"({headline.restricted_lives / headline.total_lives:.1%})"
)
print(f"Payer-region access flags: {int(headline.payer_region_access_flags)} of 32")
print(f"Payer-region adoption flags: {int(headline.payer_region_adoption_flags)} of 32")
"""
        ),
        md(
            "The 3,415-patient cohort and 2,798 Roventra starts match the Chapter 5 washout-corrected line table."
        ),
        md("## 7.2 Effective-dated access and covered lives"),
        code(
            """
summary = results["covered_lives_summary"].query("payer_type == 'All'").copy()
summary["plan_coverage_rate"] = summary.plan_coverage_rate.map(lambda v: f"{v:.1%}")
summary["covered_lives_rate"] = summary.covered_lives_rate.map(lambda v: f"{v:.1%}")
summary["unrestricted_lives_rate"] = summary.unrestricted_lives_rate.map(
    lambda v: f"{v:.1%}"
)
summary["access_quality_score"] = summary.access_quality_score.map(lambda v: f"{v:.3f}")
print(summary.to_string(index=False))
print()
restriction_lives = results["restriction_lives"].copy()
restriction_lives["lives_share"] = restriction_lives.lives_share.map(
    lambda v: f"{v:.1%}"
)
print(restriction_lives.to_string(index=False))
"""
        ),
        md(
            "Plan coverage and covered lives use different denominators. Every covered cell retains at least 1 utilization-management restriction in this synthetic case."
        ),
        md("## 7.3 Corrected competitive starts"),
        code(
            """
print(results["source_of_business"].to_string(index=False))
print()
line1 = results["corrected_line1"]
mix = (
    line1.groupby("first_regimen").patient_id.nunique()
    .sort_values(ascending=False)
    .rename("patients")
    .reset_index()
)
mix["share"] = (mix.patients / len(line1)).map(lambda v: f"{v:.1%}")
print(mix.to_string(index=False))
"""
        ),
        md(
            "The source-of-business table keeps 24 switches and 4 additions in separate categories."
        ),
        md("## 7.4 Prescription attempts"),
        code(
            """
trace = results["pat02034_attempt_trace"]
columns = [
    "fill_number", "first_submission_date", "last_transaction_date",
    "transaction_rows", "had_pend", "had_reversal",
    "final_outcome", "days_to_paid",
]
print(trace[columns].to_string(index=False))
"""
        ),
        md("Seven transaction rows collapse into 4 completed attempts for PAT02034."),
        md("## 7.5 Access and adoption decisions"),
        code(
            """
decisions = results["payer_region_decisions"]
selected = (
    decisions.set_index(["payer_id", "region"])
    .loc[[("PAY002", "Northeast"), ("PAY004", "Midwest"), ("PAY005", "South")]]
    .reset_index()
)
selected["brand_share"] = selected.brand_share.map(lambda v: f"{v:.1%}")
selected["probability_below_benchmark"] = selected.probability_below_benchmark.map(
    lambda v: f"{v:.1%}"
)
print(selected[[
    "payer_id", "region", "access_state", "treated_patients",
    "brand_starts", "brand_share", "probability_below_benchmark", "action",
]].to_string(index=False))
print()
print(decisions.action.value_counts().to_string())
"""
        ),
        md(
            "The 3 selected cells demonstrate access work, adoption review, and a dual workstream."
        ),
        md("## 7.6 Controlled formulary-event measurement"),
        code(
            """
event = results["formulary_event_effect"].iloc[0]
print(f"Immediate level effect: {event.immediate_effect:+.1%}")
print(f"Slope change per week: {event.slope_change_per_week:+.2%}")
print(
    f"Week {int(event.effect_week)} effect: {event.effect_at_week:+.1%} "
    f"(95% CI {event.effect_at_week_lower_95:+.1%} "
    f"to {event.effect_at_week_upper_95:+.1%})"
)
diagnostic = results["synthetic_control_diagnostics"].iloc[0]
print(f"Pre-period RMSPE: {diagnostic.pre_rmspe:.3f}")
print(f"Post-period mean gap: {diagnostic.post_mean_gap:+.1%}")
"""
        ),
        md(
            "The controlled ITS recovers the planted level and slope improvement. The synthetic control uses independent donor series."
        ),
        md("## 7.7 Account actions"),
        code(
            """
accounts = results["account_access_adoption_actions"]
print(accounts.action.value_counts().to_string())
print()
print(
    accounts.set_index("account_id")
    .loc[["ACC155", "ACC005", "ACC121"], [
        "attributed_patients", "treated_patients", "brand_starts",
        "restricted_patients", "action", "reason_code",
    ]]
    .to_string()
)
"""
        ),
        md(
            "The account table reuses the Chapter 6 patient-HCP-account mapping and preserves mixed payer evidence."
        ),
        md("## 7.8 Monitoring and evidence sufficiency"),
        code(
            """
alerts = results["changepoint_alerts"].head(3).copy()
alerts["standardized_cusum"] = alerts.standardized_cusum.round(3)
print(alerts.to_string(index=False))
print()
print(results["switch_evidence"][[
    "first_regimen", "patients", "switch_events",
    "median_time_to_switch", "comparison_status",
]].to_string(index=False))
"""
        ),
        md(
            "CUSUM supplies a review date. The switch table records that comparative medians are not reached."
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"ch07-walk-{index:02d}"
    return nbf.v4.new_notebook(cells=cells, metadata=_metadata())


def exercise_solutions() -> nbf.NotebookNode:
    """Return the worked exercise notebook."""

    cells = [
        md(
            """
# Chapter 7: Exercise Solutions

Each solution ends with the judgment that should accompany the calculation in real data.
"""
        ),
        code(
            """
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "ch07_competitive").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ch07_competitive.scripts.run_analysis import run_analysis  # noqa: E402

results = run_analysis(ROOT)
print("Loaded Chapter 7 evidence package.")
"""
        ),
        md("## Exercise 1: Rebuild covered lives"),
        code(
            """
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
"""
        ),
        md(
            "**Methods note:** Contracting review should lead with covered and restricted lives. The access-quality score remains a scenario-weighted supplement."
        ),
        md("## Exercise 2: Trace an attempt"),
        code(
            """
attempts = results["prescription_attempts"]
patient = attempts.loc[attempts.had_pend, "patient_id"].iloc[0]
trace = attempts.loc[attempts.patient_id.eq(patient), [
    "patient_id", "fill_number", "first_submission_date",
    "last_transaction_date", "transaction_rows", "had_pend", "final_outcome",
]]
print(trace.to_string(index=False))
"""
        ),
        md(
            "**Methods note:** Count the grouped attempt once. Counting every transaction row overstates access friction."
        ),
        md("## Exercise 3: Change the operating benchmark"),
        code(
            """
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
print(baseline.to_string())
print("\\n80% benchmark")
print(alternative.to_string())
"""
        ),
        md(
            "**Methods note:** The source evidence stays fixed. The action changes because the operating benchmark changed, so the output must retain the rule version."
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"ch07-ex-{index:02d}"
    return nbf.v4.new_notebook(cells=cells, metadata=_metadata())


def main() -> None:
    targets = {
        CHAPTER_DIR / "chapter7_walkthrough.ipynb": walkthrough(),
        CHAPTER_DIR / "exercise_solutions.ipynb": exercise_solutions(),
    }
    for path, notebook in targets.items():
        nbf.write(notebook, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
