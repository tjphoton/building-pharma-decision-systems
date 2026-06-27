# Omnichannel Analytics: Exercise Solutions

Each solution ends with the judgment that belongs in a real-data review.



```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ch08_omnichannel" / "generation_modules"))
sys.path.insert(0, str(ROOT / "ch08_omnichannel" / "scripts"))

from ch08_omnichannel.generation_modules.synthetic import generate  # noqa: E402
from ch08_omnichannel.scripts.run_analysis import run_analysis  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
generate(ROOT, ROOT / "ch08_omnichannel" / "data" / "generated")
results = run_analysis(ROOT)
print(f"Ledger events: {len(results['event_ledger']):,}")
print(f"HCP-account snapshots: {len(results['snapshot_panel']):,}")
print(f"Planning HCP-account rows: {len(results['channel_plan']):,}")

```

    Ledger events: 3,707
    HCP-account snapshots: 1,422
    Planning HCP-account rows: 158


## 1. Count email opens as meaningful responses



```python
from ch08_omnichannel.scripts.features import build_snapshot_panel
from ch08_omnichannel.scripts.modeling import fit_response_model
from ch08_omnichannel.scripts.run_analysis import load_inputs

ledger = results["event_ledger"].copy()
ledger.loc[
    ledger.channel.eq("Email") & ledger.response_type.eq("Opened"),
    "meaningful_response",
] = True
inputs = load_inputs(ROOT)
panel = build_snapshot_panel(
    ledger, inputs["hcp_features"], inputs["hcp_segments"],
    inputs["engagement_signals"], inputs["account_targets"],
    inputs["account_actions"],
)
loose = fit_response_model(panel)
print(loose["model_metrics"].query("split == 'test'"))

```

      split  snapshots  responses  response_rate   roc_auc  average_precision  brier_score  base_rate_brier
    2  test        316        158            0.5  0.716712           0.712998     0.226143         0.268231


Judgment: the larger label includes machine-open noise. A higher event count weakens the response definition.


## 2. Rank by uplift instead of response



```python
from sklearn.linear_model import LogisticRegression

panel = results["snapshot_panel"].copy()
panel["treated"] = panel.live_program_attendance_180.gt(0).astype(int)
cov = [
    "review_opportunity", "evidence_need_score", "access_resource_score",
    "digital_response_rate", "field_response_rate",
    "total_pressure_30", "total_pressure_90", "shrunken_response_rate_90",
]
treated = panel[panel.treated.eq(1)]
control = panel[panel.treated.eq(0)]
model_t = LogisticRegression(C=0.3, max_iter=1000).fit(treated[cov], treated.future_response)
model_c = LogisticRegression(C=0.3, max_iter=1000).fit(control[cov], control.future_response)
panel["uplift"] = (
    model_t.predict_proba(panel[cov])[:, 1] - model_c.predict_proba(panel[cov])[:, 1]
)
analysis = panel[panel.snapshot_date.eq(panel.snapshot_date.max())].merge(
    results["scored_snapshots"][
        ["npi", "account_id", "snapshot_date", "predicted_response"]
    ],
    on=["npi", "account_id", "snapshot_date"],
)
top_response = set(analysis.sort_values("predicted_response", ascending=False).head(16).npi)
top_uplift = set(analysis.sort_values("uplift", ascending=False).head(16).npi)
print(f"shared between the two top-16 lists: {len(top_response & top_uplift)}")
print(f"only in the response ranking: {len(top_response - top_uplift)}")
print(f"only in the uplift ranking: {len(top_uplift - top_response)}")

```

    shared between the two top-16 lists: 0
    only in the response ranking: 16
    only in the uplift ranking: 16


Judgment: the two rankings disagree because predicted response and uplift answer separate questions. An HCP-account row with routine response momentum is a weak use of a scarce invitation. Rank scarce programs by estimated uplift, then confirm with a holdout.


## 3. Create a registered holdout assignment



```python
eligible = results["channel_plan"].loc[
    results["channel_plan"].recommended_action.isin(
        ["Email follow-up", "Peer-program invitation", "Speaker-program invitation"]
    ),
    ["npi", "account_id", "recommended_action"],
].copy()
eligible["assignment"] = (
    eligible.sample(frac=1, random_state=20260622)
    .reset_index()
    .index.map(lambda i: "Action" if i % 2 == 0 else "Holdout")
)
eligible["outcome"] = "Meaningful response in 28 days"
print(eligible)

```

                npi account_id          recommended_action assignment                         outcome
    35   9000000033     ACC044             Email follow-up     Action  Meaningful response in 28 days
    36   9000000648     ACC199             Email follow-up    Holdout  Meaningful response in 28 days
    37   9000000567     ACC030             Email follow-up     Action  Meaningful response in 28 days
    38   9000000389     ACC155             Email follow-up    Holdout  Meaningful response in 28 days
    101  9000000505     ACC176     Peer-program invitation     Action  Meaningful response in 28 days
    102  9000000296     ACC113     Peer-program invitation    Holdout  Meaningful response in 28 days
    103  9000000621     ACC069     Peer-program invitation     Action  Meaningful response in 28 days
    104  9000000650     ACC210     Peer-program invitation    Holdout  Meaningful response in 28 days
    105  9000000026     ACC226  Speaker-program invitation     Action  Meaningful response in 28 days
    106  9000000239     ACC009  Speaker-program invitation    Holdout  Meaningful response in 28 days
    107  9000000157     ACC247  Speaker-program invitation     Action  Meaningful response in 28 days
    108  9000000128     ACC160  Speaker-program invitation    Holdout  Meaningful response in 28 days
    109  9000000522     ACC099  Speaker-program invitation     Action  Meaningful response in 28 days
    110  9000000170     ACC132  Speaker-program invitation    Holdout  Meaningful response in 28 days
    111  9000000372     ACC174  Speaker-program invitation     Action  Meaningful response in 28 days


Judgment: register eligibility, assignment, the 28-day meaningful-response outcome, and post-assignment exclusions before execution.

