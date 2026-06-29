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

import pandas as pd

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
            "## 7.1 Opening evidence\n\nThe corrected cohort comes straight from the patient-journey line table, so competitive share starts from the same population."
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
        md("## 7.2 Effective-dated access and covered lives"),
        code(
            """
history = results["access_history"].query(
    "payer_id == 'PAY005' and region == 'South' and product_name == 'Roventra'"
)
cols = ["effective_start", "effective_end", "coverage_status", "step_edit"]
print(history[cols].to_string(index=False))

analysis_date = pd.Timestamp("2024-12-31")
active = history.query("effective_start <= @analysis_date <= effective_end")
print(f"\\nIn force on {analysis_date.date()}: {active.iloc[0].coverage_status}")
"""
        ),
        md(
            "PAY005 South covered Roventra in January, added a step edit in July, and dropped it to non-covered in October. Each cell contributes the record in force on 2024-12-31."
        ),
        code(
            """
summary = results["covered_lives_summary"].query("payer_type == 'All'").iloc[0]
print(f"Plan-region records:          {int(summary.plans)}")
print(f"Records covering Roventra:    {int(summary.covered_plans)} ({summary.plan_coverage_rate:.1%})")
print(f"Enrolled lives:               {int(summary.total_lives):,}")
print(f"Lives with workable coverage: {int(summary.covered_lives):,} ({summary.covered_lives_rate:.1%})")
print(f"Lives with no restriction:    {int(summary.unrestricted_lives):,} ({summary.unrestricted_lives_rate:.1%})")
print(f"Access-quality score:         {summary.access_quality_score:.3f}")
print()
restriction_lives = results["restriction_lives"].copy()
restriction_lives["lives_share"] = restriction_lives.lives_share.map(
    lambda v: f"{v:.1%}"
)
print(restriction_lives.to_string(index=False))
print()
print(results["relative_position"].position.value_counts().to_string())
"""
        ),
        md(
            "Non-coverage and step therapy are the two states a patient cannot clear alone, and a competitor holds the better formulary position in 20 of 32 cells."
        ),
        md("## 7.3 TRx, NRx, and NBRx by brand"),
        code(
            """
import pandas as pd

attempts = results["prescription_attempts"]
completed = attempts.query("final_outcome == 'Completed'")
therapy_brands = ["Roventra", "Vexpro", "Nexoral"]
therapy = completed.query("product_name in @therapy_brands")
nbrx_reg = results["corrected_line1"].groupby("first_regimen").patient_id.nunique()
sob = results["source_of_business"]
all_nbrx = int(sob.loc[sob.source_of_business == "New to therapy", "patients"].iloc[0])
combo_nbrx = int(nbrx_reg.get("Nexoral + Vexpro", 0))

def fmt_brand(prod):
    sub = completed.query(f"product_name == '{prod}'")
    trx = len(sub)
    nrx = len(sub.query("fill_number == 0"))
    nbrx = int(nbrx_reg.get(prod, 0))
    return [f"{trx:,}", f"{nrx:,}", f"{nbrx:,}", f"{nbrx / all_nbrx:.1%}"]

tbl = pd.DataFrame(
    {
        "All brands":     [f"{len(therapy):,}", f"{len(therapy.query('fill_number == 0')):,}", f"{all_nbrx:,}", "100%"],
        "Roventra":       fmt_brand("Roventra"),
        "Vexpro":         fmt_brand("Vexpro"),
        "Nexoral":        fmt_brand("Nexoral"),
        "Nexoral+Vexpro": ["", "", f"{combo_nbrx:,}", f"{combo_nbrx / all_nbrx:.1%}"],
    },
    index=["TRx", "NRx", "NBRx", "Share"],
)
print(tbl.to_string())
"""
        ),
        md(
            "Roventra holds 81.9% (2,798 of 3,415) of new-to-therapy NBRx starts. The Share row is the right base for competitive comparison."
        ),
        md("## 7.4 Access and adoption decisions"),
        code(
            """
from scipy.stats import beta  # noqa: E402

prior_mean = 0.8193
prior_strength = 40
alpha0 = prior_mean * prior_strength
beta0 = (1 - prior_mean) * prior_strength
for name, brand_starts, competitor_starts in [("Small cell", 7, 2), ("Large cell", 88, 30)]:
    treated = brand_starts + competitor_starts
    raw = brand_starts / treated
    pooled = (brand_starts + alpha0) / (treated + alpha0 + beta0)
    prob_below = beta.cdf(0.82, brand_starts + alpha0, competitor_starts + beta0)
    print(f"{name}: raw {raw:.1%}, pooled {pooled:.1%}, P(<82%) {prob_below:.1%}")
"""
        ),
        code(
            """
decisions = results["payer_region_decisions"]
sel = decisions.set_index(["payer_id", "region"]).loc[
    [("PAY002", "Northeast"), ("PAY004", "Midwest"), ("PAY005", "South")]
]
view = pd.DataFrame(
    {
        "access_state": sel.access_state,
        "treated_patients": sel.treated_patients.astype(int),
        "brand_share": sel.brand_share.map(lambda v: f"{v:.1%}"),
        "share_95ci": sel.apply(
            lambda x: f"{x.share_lower_95:.0%}-{x.share_upper_95:.0%}", axis=1
        ),
        "prob_below_82": sel.probability_below_benchmark.map(lambda v: f"{v:.0%}"),
        "access_flag": sel.access_flag,
        "adoption_flag": sel.adoption_flag,
        "action": sel.action,
    }
)
view.index = [f"{p} {r}" for p, r in view.index]
print(view.T.to_string())
print()
print(decisions.action.value_counts().to_string())
"""
        ),
        md(
            "Partial pooling holds the small cell back. The access and adoption flags route each cell independently: similar shares reach access work, adoption review, and a dual workstream."
        ),
        md("## 7.5 Controlled formulary-event measurement"),
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
            "The controlled time series separates the PAY004 lift from the market trend, and the synthetic control lands in the same place."
        ),
        md("## 7.6 Monitoring and evidence sufficiency"),
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
            "CUSUM opens the PAY004 increase episode at week 20. Later threshold crossings are persistence evidence for the same episode."
        ),
        md(
            """
![Two stacked panels show PAY004 weekly Roventra share and the positive standardized CUSUM trace. The formulary event is marked at week 17, the alarm threshold is marked at 4 standard deviations, the green point marks the episode-opening alarm, and hollow gray points mark later persistence crossings.](assets/figures/figure_7_5_cusum_detection.svg)

*Figure 7.5. PAY004 share rises after the week 17 formulary event. The positive CUSUM opens an increase episode at week 20. Later threshold crossings show persistence of the same episode, not separate events. Synthetic data.*
"""
        ),
        md(
            """
![Conceptual survival curves contrasting a cohort with sufficient switch events (green, median reached at a marked week) against a sparse cohort (blue, curve stays above 50% through week 52).](assets/figures/figure_7_6_switch_support.svg)

*Figure 7.6. When the survival curve stays above 50% through all of follow-up, the median time to switch is not reached. Reporting a number here would invent precision the data does not hold. Track event accumulation across quarterly refreshes before publishing a comparative median. Synthetic data.*
"""
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
        CHAPTER_DIR / "ch07_walkthrough.ipynb": walkthrough(),
        CHAPTER_DIR / "ch07_exercise_solutions.ipynb": exercise_solutions(),
    }
    for path, notebook in targets.items():
        nbf.write(notebook, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
