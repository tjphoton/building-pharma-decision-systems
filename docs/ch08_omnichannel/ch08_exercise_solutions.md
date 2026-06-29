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

pd.set_option("display.width", 88)
pd.set_option("display.max_columns", None)
generate(ROOT, ROOT / "ch08_omnichannel" / "data" / "generated")
results = run_analysis(ROOT)
print(f"Ledger events: {len(results['event_ledger']):,}")
print(f"HCP-account snapshots: {len(results['snapshot_panel']):,}")
print(f"Planning HCP-account rows: {len(results['channel_plan']):,}")

```

    Ledger events: 3,650
    HCP-account snapshots: 1,422
    Planning HCP-account rows: 158


## 1. Audit the snapshot boundary



```python
ledger = results["event_ledger"].copy()
panel = results["snapshot_panel"].copy()
row = panel.loc[
    panel.npi.eq("9000000280")
    & panel.snapshot_date.eq("2025-02-28")
].iloc[0]
events = ledger.loc[ledger.npi.eq("9000000280")].copy()
history = events.loc[
    events.event_date.between(row.snapshot_date - pd.Timedelta(days=89), row.snapshot_date)
]
future = events.loc[
    events.event_date.gt(row.snapshot_date)
    & events.event_date.le(row.outcome_end)
]
audit = pd.DataFrame([
    ("snapshot_date", row.snapshot_date.date()),
    ("outcome_end", row.outcome_end.date()),
    ("prior_90_day_events", len(history)),
    ("snapshot_total_pressure_90", int(row.total_pressure_90)),
    ("future_events", len(future)),
    ("snapshot_future_response", int(row.future_response)),
])
print(audit.to_string(index=False, header=False))

```

                 snapshot_date 2025-02-28
                   outcome_end 2025-03-28
           prior_90_day_events          3
    snapshot_total_pressure_90          3
                 future_events          0
      snapshot_future_response          0


Judgment: the feature count and outcome count come from different sides of the snapshot date. A response model that reads future events as features would leak the answer into the score.


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


Judgment: the two rankings disagree because predicted response and uplift answer separate questions. An HCP-account row with routine response momentum is a weak use of a scarce invitation. Rank scarce programs by estimated uplift, then flag the change for experimental confirmation.


## 3. Stress-test channel economics



```python
econ = results["channel_economics"].copy()
stress = econ.loc[econ.channel.isin(["Email", "Paid media"])].copy()
stress.loc[stress.channel.eq("Email"), "unit_cost"] *= 2
stress.loc[stress.channel.eq("Paid media"), "incremental_per_touch"] *= 0.5
stress["cost_per_incremental_response"] = (
    stress.unit_cost / stress.incremental_per_touch
)
stress["unit_cost"] = stress.unit_cost.map(lambda x: f"${x:,.2f}")
stress["incremental_per_touch"] = stress.incremental_per_touch.map(
    lambda x: f"{x * 100:+.1f} pp"
)
stress["cost_per_incremental_response"] = (
    stress.cost_per_incremental_response.map(lambda x: f"${x:,.0f}")
)
print(stress[[
    "channel", "unit_cost", "incremental_per_touch",
    "cost_per_incremental_response",
]].to_string(index=False))

```

       channel unit_cost incremental_per_touch cost_per_incremental_response
         Email     $0.50               +1.4 pp                           $35
    Paid media     $1.40               +0.5 pp                          $285


Judgment: email remains cheaper per incremental response after its cost doubles, while paid media becomes less attractive when its lift assumption weakens. Cost, lift, and credit must be read together.

