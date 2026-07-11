# Chapter 13 Exercise Solutions: Marketing Mix Modeling and Unified Measurement

Worked solutions for the three exercises at the end of the chapter.



```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "ch13_mmm").exists():
            return candidate
    raise FileNotFoundError("Run inside the repository.")


REPO_ROOT = find_repo_root(Path.cwd().resolve())
CHAPTER_DIR = REPO_ROOT / "ch13_mmm"
SCRIPT_DIR = CHAPTER_DIR / "scripts"
OUTPUT_DIR = CHAPTER_DIR / "assets" / "generated_outputs"
sys.path.insert(0, str(SCRIPT_DIR))
pd.set_option("display.width", 160)

```


```python
from run_analysis import (
    build_measurement_guardrails,
    build_next_measurement_agenda,
)

evidence_record = pd.read_csv(OUTPUT_DIR / "channel_evidence_record.csv")
decision_record = pd.read_csv(OUTPUT_DIR / "measurement_decision_record.csv")
guardrails = pd.read_csv(OUTPUT_DIR / "measurement_guardrails.csv")
agenda = pd.read_csv(OUTPUT_DIR / "next_measurement_agenda.csv")
weekly = pd.read_csv(OUTPUT_DIR / "synthetic_weekly_data.csv")
current_spends = np.array([weekly[f"spend_{ch}"].mean() for ch in ["field", "email", "digital", "paid_media"]])

```

## Exercise 1

Add a hypothetical paid-media holdout result: set paid media's `causal_signal`, `evidence_tier`, and `comparability_status` as if the holdout had just read out with a plausible incremental-NRx result, then rebuild the guardrails and the agenda from that updated evidence record. Report how paid media's evidence tier, allowed move, and agenda rank change.



```python
hypothetical_evidence_record = evidence_record.copy()
paid_row = hypothetical_evidence_record["channel"] == "paid_media"
hypothetical_evidence_record.loc[paid_row, "causal_signal"] = "6.10 incremental weekly NRx across 15 geos (hypothetical paid-media holdout, response-curve segment)"
hypothetical_evidence_record.loc[paid_row, "evidence_tier"] = "causal-anchored"
hypothetical_evidence_record.loc[paid_row, "comparability_status"] = "fully comparable calibration or budget guardrail available"
hypothetical_evidence_record.loc[paid_row, "next_measurement_action"] = "refresh the calibration periodically (repeat the geo-holdout or experiment); monitor for drift"

hypothetical_guardrails = build_measurement_guardrails(hypothetical_evidence_record, decision_record, current_spends)
hypothetical_agenda = build_next_measurement_agenda(hypothetical_evidence_record, decision_record, current_spends, hypothetical_guardrails)

print(hypothetical_evidence_record.loc[paid_row, ["channel", "evidence_tier", "comparability_status"]].to_string(index=False))
print(hypothetical_guardrails.loc[hypothetical_guardrails["channel"] == "paid_media", ["channel", "allowed_budget_move", "new_anchor_required", "refresh_required"]].to_string(index=False))
print(hypothetical_agenda.loc[hypothetical_agenda["channel"] == "paid_media", ["channel", "priority_rank", "priority_score"]].to_string(index=False))

```

       channel   evidence_tier                                       comparability_status
    paid_media causal-anchored fully comparable calibration or budget guardrail available
       channel                                                              allowed_budget_move  new_anchor_required  refresh_required
    paid_media bounded to +/-20%; causal anchor exists but MMM diagnostics still limit movement                False              True
       channel  priority_rank  priority_score
    paid_media              4            39.4


Paid media moves from `mmm-only directional` to `causal-anchored`. Its allowed move becomes bounded to +/-20% because the MMM gate is still directional (paid media's worst R-hat is above the 1.20 line), but the reason changes: now the cap reflects model diagnostics rather than missing causal evidence. Its agenda rank falls because the channel no longer needs a first anchor, and digital keeps the top rank it already held.


## Exercise 2

Give digital stronger attribution support without adding an experiment. Explain why the budget guardrail should still stay bounded.



```python
hypothetical_evidence_record = evidence_record.copy()
digital_row = hypothetical_evidence_record["channel"] == "digital"
hypothetical_evidence_record.loc[digital_row, "attribution_signal"] = "18.4% of conversion credit (stronger authenticated web proxy)"
hypothetical_evidence_record.loc[digital_row, "comparability_status"] = "partial cross-check available; compare metric, scope, and population before using it against MMM"
hypothetical_guardrails = build_measurement_guardrails(hypothetical_evidence_record, decision_record, current_spends)

print(hypothetical_evidence_record.loc[digital_row, ["channel", "attribution_signal", "comparability_status", "evidence_tier"]].to_string(index=False))
print(hypothetical_guardrails.loc[hypothetical_guardrails["channel"] == "digital", ["channel", "move_permission", "allowed_budget_move", "guardrail_reason"]].to_string(index=False))

```

    channel                                            attribution_signal                                                                             comparability_status        evidence_tier
    digital 18.4% of conversion credit (stronger authenticated web proxy) partial cross-check available; compare metric, scope, and population before using it against MMM mmm-only directional
    channel move_permission                                             allowed_budget_move                                          guardrail_reason
    digital         bounded bounded to +/-10%; no causal anchor and MMM remains directional MMM remains directional and no causal anchor is available


Attribution alone remains a scope check, not a causal anchor. The scope is partial, the metric is conversion credit rather than incremental NRx, and the population is authenticated web paths rather than the full weekly spend series. That stronger cross-check still leaves digital in the mmm-only directional tier, bounded to +/-10%, the tightest band, because neither a causal anchor nor a clean model-health read exists for the channel.


## Exercise 3

Using `next_measurement_agenda.csv` and `unified_budget_recommendation.csv`, write two to three sentences on what you would want to see before allowing any channel's budget to move by more than its current guardrail.



```python
agenda_full = pd.read_csv(OUTPUT_DIR / "next_measurement_agenda.csv")
unified_full = pd.read_csv(OUTPUT_DIR / "unified_budget_recommendation.csv")
print(agenda_full.to_string(index=False))
print(unified_full[["channel", "decision_status", "evidence_tier", "allowed_budget_move"]].to_string(index=False))

```

       channel           evidence_tier  gap_severity                                                                                                 recommended_next_test          target_metric  spend_share_pct  confound_score  error_score  decision_unlock_score  priority_score  priority_rank
       digital    mmm-only directional          70.0                     run a targeted holdout or geo experiment to sharpen decomposition before requesting a wider bound incremental weekly NRx             14.9            18.0          6.7                   20.0           123.7              1
    paid_media    mmm-only directional          70.0                     run a targeted holdout or geo experiment to sharpen decomposition before requesting a wider bound incremental weekly NRx             19.3            10.0          2.8                   20.0           114.4              2
         email mmm-only decision-ready          40.0 add a low-cost incrementality or holdout test to anchor this channel before scaling meaningfully beyond current spend incremental weekly NRx              3.3             1.5          5.2                   25.0            73.7              3
         field         causal-anchored          10.0                        refresh the calibration periodically (repeat the geo-holdout or experiment); monitor for drift incremental weekly NRx             62.4             2.5          4.1                    5.0            59.1              4
       channel decision_status           evidence_tier                                                              allowed_budget_move
         field     directional         causal-anchored bounded to +/-20%; causal anchor exists but MMM diagnostics still limit movement
         email  decision-ready mmm-only decision-ready         increase capped at +30%; causal anchor required before sustained scaling
       digital     directional    mmm-only directional                  bounded to +/-10%; no causal anchor and MMM remains directional
    paid_media     directional    mmm-only directional                  bounded to +/-10%; no causal anchor and MMM remains directional


**Answer.** Every channel bounded today is bounded because it is either `directional` on the MMM gate, `mmm-only` on the evidence record, or both; before asking for a wider move, I would want that channel to clear the model-health gate on its own series (a well-identified R-hat, error, and coefficient of variation) or to pick up a causal anchor, a geo-holdout or randomized read, sized for that channel's own spend level and time window. A favorable point estimate from the current calibrated fit is not, by itself, a reason to widen the band; the next-measurement agenda's top-ranked channel is where that anchor should come from first.

