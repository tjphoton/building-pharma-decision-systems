# Chapter 7: Competitive Intelligence and Market Access

This notebook executes the chapter as one decision chain. It uses fictional products, patients, payers, accounts, and planted synthetic events.



```python
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

```

    Loaded Chapter 7 evidence package.


## 7.1 Opening evidence

The corrected cohort comes straight from the patient-journey line table, so competitive share starts from the same population.



```python
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

```

    New-to-therapy patients: 3,415
    Roventra new starts: 2,798
    Materially restricted lives: 6,740,000 of 10,926,000 (61.7%)
    Payer-region access flags: 20 of 32
    Payer-region adoption flags: 3 of 32


## 7.2 Effective-dated access and covered lives



```python
history = results["access_history"].query(
    "payer_id == 'PAY005' and region == 'South' and product_name == 'Roventra'"
)
cols = ["effective_start", "effective_end", "coverage_status", "step_edit"]
print(history[cols].reset_index(drop=True))

analysis_date = pd.Timestamp("2024-12-31")
active = history.query("effective_start <= @analysis_date <= effective_end")
print(f"\nIn force on {analysis_date.date()}: {active.iloc[0].coverage_status}")

```

      effective_start effective_end coverage_status step_edit
    0      2024-01-01    2024-06-30         Covered        No
    1      2024-07-01    2024-09-30         Covered       Yes
    2      2024-10-01    2025-12-31     Non-covered        No
    
    In force on 2024-12-31: Non-covered


PAY005 South covered Roventra in January, added a step edit in July, and dropped it to non-covered in October. Each cell contributes the record in force on 2024-12-31.



```python
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
print(restriction_lives)
print()
print(results["relative_position"].position.value_counts())

```

    Plan-region records:          32
    Records covering Roventra:    24 (75.0%)
    Enrolled lives:               10,926,000
    Lives with workable coverage: 8,314,000 (76.1%)
    Lives with no restriction:    0 (0.0%)
    Access-quality score:         0.533
    
              access_state  payer_region_cells  enrolled_lives lives_share
    0  Prior authorization                  12         4186000       38.3%
    1            Step edit                  12         4128000       37.8%
    2          Non-covered                   8         2612000       23.9%
    
    position
    Competitor favored    20
    Parity                 8
    Brand favored          4
    Name: count, dtype: int64


Non-coverage and step therapy are the two states a patient cannot clear alone, and a competitor holds the better formulary position in 20 of 32 cells.


## 7.3 TRx, NRx, and NBRx by brand


![A timeline for one patient shows three rows. TRx marks every fill including refills. NRx marks the first fill of each episode, including restarts after a gap. NBRx marks only the patient's very first fill ever.](assets/figures/figure_7_1_prescription_types.svg)

*Figure 7.1. TRx grows with every refill. A restart after a treatment gap adds one NRx but no NBRx. NBRx is capped at one per patient per drug. Synthetic data.*



```python
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
print(tbl)

```

          All brands Roventra Vexpro Nexoral Nexoral+Vexpro
    TRx       30,552   16,636  6,884   7,032               
    NRx       13,867    6,401  3,684   3,782               
    NBRx       3,415    2,798    309     303              5
    Share       100%    81.9%   9.0%    8.9%           0.1%


Roventra holds 81.9% (2,798 of 3,415) of new-to-therapy NBRx starts. The Share row is the right base for competitive comparison.


## 7.4 Access and adoption decisions



```python
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

```

    Small cell: raw 77.8%, pooled 81.2%, P(<82%) 52.9%
    Large cell: raw 74.6%, pooled 76.4%, P(<82%) 95.7%


![A scatter plot of 32 payer-region cells shows raw brand share on the x-axis and pooled posterior share on the y-axis. Points for small cells cluster near the diagonal of the national prior; points for large cells cluster near the no-pooling diagonal.](assets/figures/figure_7_2_partial_pooling.svg)

*Figure 7.2. Each point is one payer-region cell. Small cells (light color) are pulled far toward the 81.9% national prior. Large cells (dark color) stay near the diagonal because local evidence outweighs the prior. Synthetic data.*



```python
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
print(view.T)
print()
print(decisions.action.value_counts())

```

                     PAY002 Northeast       PAY004 Midwest     PAY005 South
    access_state          Non-covered  Prior authorization      Non-covered
    treated_patients              100                  118              129
    brand_share                 85.0%                77.1%            75.2%
    share_95ci                77%-91%              69%-84%          67%-82%
    prob_below_82                 24%                  87%              95%
    access_flag                  True                False             True
    adoption_flag               False                 True             True
    action              Access review      Adoption review  Dual workstream
    
    action
    Access review      19
    Sustain            10
    Adoption review     2
    Dual workstream     1
    Name: count, dtype: int64


Partial pooling holds the small cell back. The access and adoption flags route each cell independently: similar shares reach access work, adoption review, and a dual workstream.


![Five aligned panels show, for 32 payer-region cells, Roventra share with its uncertainty interval, adoption flag (filled orange square for True), restricted lives, access flag (filled red square for True), and the assigned action labeled with a shape and color legend keyed to Sustain, Adoption review, Access review, Dual workstream, and Monitor.](assets/figures/figure_7_3_payer_region_matrix.svg)

*Figure 7.3. The adoption flag and access flag are set independently from share uncertainty and restricted lives; the action column combines them. Wide share intervals mark the small cells the partial-pooling rule holds back. Synthetic data.*


## 7.5 Controlled formulary-event measurement



```python
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

```

    Immediate level effect: +7.4%
    Slope change per week: +0.24%
    Week 28 effect: +10.0% (95% CI +6.1% to +14.0%)
    Pre-period RMSPE: 0.038
    Post-period mean gap: +7.5%


The controlled time series separates the PAY004 lift from the market trend, and the synthetic control lands in the same place.


![Two stacked panels show PAY004 observed share, the ITS model fit, and the counterfactual through the year, and the lower panel shows observed week-by-week gaps as dots with the smooth model estimate as a green line.](assets/figures/figure_7_4_formulary_event.svg)

*Figure 7.4. The counterfactual (dashed gray) follows the slightly downward class trend the donors carry; PAY004's observed share rises above that baseline after week 17. The lower panel dots show the observed gap each week; the green line is the model's linear estimate, reaching +10.0 points by week 28. Synthetic data.*


![Single panel showing PAY004 observed share as a solid blue line and the synthetic counterfactual as a dashed gray line over 52 weeks, with the event at week 17 marked in red and the post-event gap shaded green.](assets/figures/figure_7_5_synthetic_control.svg)

*Figure 7.5. The synthetic counterfactual (dashed gray) follows the weighted-donor blend through the pre-event weeks with RMSPE 0.038. After week 17, PAY004's observed share (blue) separates from the counterfactual; the green-shaded gap averages +7.5 points across the post-event period. PAY006 receives zero weight because it tracks PAY004 less well than PAY003 and PAY008 in the pre-event period. Synthetic data.*

