# Next Best Action

Build a governed release layer that turns the omnichannel HCP-account state into one executable recommendation. The walkthrough follows the same objects as the manuscript: state, candidates, content gates, contract, expiration, constrained value ranking, logged policy data, replay diagnostics, test design, and execution feedback.



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


## 9.1 Build The NBA Recommendation Engine


### Object model



```python
print(results["nba_object_model"].to_string(index=False))

```

           object               record_level                                       job                           example_fields
            state           HCP-account-date     Current facts available to the engine permission, access state, response score
        candidate HCP-account-action-content        Every action the policy considered        action, channel, content ID, gate
         contract    released recommendation Executable row sent to downstream systems  reason, timing, measurement, expiration
       policy log        historical decision          Evidence for learning and replay      logged action, probability, outcome
    execution log    released recommendation    Last-mile adoption and override record   status, override reason, feedback time


### Load the state



```python
row = results["state"].loc[results["state"].npi.eq("9000000280")].iloc[0]
for field in [
    "npi", "account_id", "territory", "account_action",
    "competitive_action", "contact_permission_status", "pressure_band",
    "total_pressure_30", "predicted_response", "digital_signal",
    "field_signal", "live_program_signal", "priority_flag", "context_bucket",
]:
    print(f"{field}: {row[field]}")

```

    npi: 9000000280
    account_id: ACC089
    territory: T02
    account_action: Monitor
    competitive_action: Defend and learn
    contact_permission_status: Allowed
    pressure_band: Low
    total_pressure_30: 1
    predicted_response: 0.4962414873343488
    digital_signal: True
    field_signal: True
    live_program_signal: False
    priority_flag: False
    context_bucket: Digital-responsive


### Build the candidate menu



```python
trace = results["hcp0280_rejected_alternatives"][[
    "candidate_action", "content_id", "policy_precedence",
    "candidate_status", "binding_gate"
]].copy()
print(trace.to_string(index=False))

```

               candidate_action               content_id  policy_precedence              candidate_status    binding_gate
                      No action                                           1                    Ineligible  not_suppressed
               Access follow-up                                          10                    Ineligible no_access_route
             Field conversation     CNT_FIELD_EXPIRED_02                 20                    Ineligible    not_priority
             Field conversation       CNT_FIELD_GUIDE_01                 20                    Ineligible    not_priority
             Program invitation    CNT_PROGRAM_INVITE_01                 25                    Ineligible  program_signal
             Program invitation CNT_PROGRAM_WRONG_AUD_02                 25                    Ineligible  program_signal
                 Approved email      CNT_EMAIL_ACCESS_01                 30                      Selected          passed
                 Approved email       CNT_EMAIL_DRAFT_03                 30                    Ineligible         content
                 Approved email     CNT_EMAIL_EXPIRED_02                 30                    Ineligible         content
    Continue responsive content       CNT_WEB_ACCOUNT_02                 40                    Ineligible         content
    Continue responsive content          CNT_WEB_RESP_01                 40 Eligible but lower precedence          passed
                        Monitor                                          80 Eligible but lower precedence          passed


![Figure 9.1. HCP0280 candidate trace across action, binding gate, content asset, and final status. Synthetic data.](assets/figures/figure_9_1_governed_engine.svg)

*Figure 9.1. HCP0280 candidate trace across action, binding gate, content asset, and final status. Synthetic data.*



```python
content = results["hcp0280_content_trace"][[
    "candidate_action", "content_id", "mlr_status", "audience",
    "approved_channel", "content_gate_reason", "eligible"
]].copy()
print(content.to_string(index=False))

```

               candidate_action               content_id mlr_status audience approved_channel  content_gate_reason  eligible
                 Approved email      CNT_EMAIL_ACCESS_01   Approved      HCP            Email               Passed      True
                 Approved email       CNT_EMAIL_DRAFT_03      Draft      HCP            Email Content not approved     False
                 Approved email     CNT_EMAIL_EXPIRED_02   Approved      HCP            Email      Content expired     False
    Continue responsive content       CNT_WEB_ACCOUNT_02   Approved  Account              Web    Audience mismatch     False
    Continue responsive content          CNT_WEB_RESP_01   Approved      HCP              Web               Passed      True
             Field conversation     CNT_FIELD_EXPIRED_02   Approved      HCP            Field      Content expired     False
             Field conversation       CNT_FIELD_GUIDE_01   Approved      HCP            Field               Passed     False
             Program invitation    CNT_PROGRAM_INVITE_01   Approved      HCP          Program               Passed     False
             Program invitation CNT_PROGRAM_WRONG_AUD_02   Approved  Account          Program    Audience mismatch     False


### Apply the gates



```python
gate = results["gate_summary_by_gate"].head(8).copy()
print(gate.to_string(index=False))

```

       binding_gate  blocked_candidates  affected_hcp_account_rows     example_action
         suppressed                 506                         46   Access follow-up
       access_route                 350                         35 Field conversation
            content                 167                         57     Approved email
       not_priority                 138                         69 Field conversation
     not_suppressed                 112                        112          No action
    no_email_signal                  81                         27     Approved email
    no_access_route                  77                         77   Access follow-up
     program_signal                  72                         36 Program invitation


![Figure 9.2. Largest binding gates across the expanded candidate table. Synthetic data.](assets/figures/figure_9_2_gate_summary.svg)

*Figure 9.2. Largest binding gates across the expanded candidate table. Synthetic data.*


### Write the recommendation contract



```python
print(results["recommendation_contract_dictionary"].to_string(index=False))

```

                  field             question_answered                                why_it_matters
     recommended_action      What should happen next?                   Execution role and workload
    recommended_channel       Where should it happen?        CRM, email, program, or access routing
             content_id Which approved asset is used? MLR, indication, audience, and expiry control
            reason_code Why was this action selected?                          User trust and audit
       measurement_hook  What must be observed later?                   Learning and accountability
             expires_on        When is the row stale?              Prevents outdated action release
         policy_version   Which rule set produced it?                Rollback and model-risk review



```python
row = results["recommendation_contract"].iloc[0]
fields = [
    "recommendation_id", "recommended_action", "recommended_channel",
    "content_id", "content_family", "reason_code", "measurement_hook",
    "policy_version", "rule_set_version", "model_version",
    "expected_incremental_value", "expires_on", "review_required",
]
for field in fields:
    value = row[field]
    if field == "expected_incremental_value":
        value = f"{value:.2f}"
    print(f"{field}: {value}")

```

    recommendation_id: NBA00071
    recommended_action: Approved email
    recommended_channel: Email
    content_id: CNT_EMAIL_ACCESS_01
    content_family: Access support
    reason_code: Available email frequency with a qualifying signal
    measurement_hook: Delivery; click; qualified follow-up
    policy_version: nba_policy_2025_02_v2
    rule_set_version: nba_rules_2025_02_v2
    model_version: omni_response_2025_02_v1
    expected_incremental_value: 707.07
    expires_on: 2025-03-14 00:00:00
    review_required: False


### Set the expiration



```python
print(results["expiration_policy"].to_string(index=False))
print()
print(results["expiration_analysis"].to_string(index=False))

```

               candidate_action  default_ttl_days  stale_when                              refresh_trigger
                      No action                14 TTL reached           Permission or access state changes
               Access follow-up                 7 TTL reached         Access-state change or resolved case
             Field conversation                21 TTL reached          Completed call or territory refresh
             Program invitation                10 TTL reached Seat date, attendance, or new access barrier
                 Approved email                14 TTL reached   Content expiry, opt-out, or new engagement
    Continue responsive content                14 TTL reached       Content expiry or new digital response
                        Monitor                30 TTL reached    Material evidence change or cycle refresh
    
                          metric  value
      Median days between events 12.000
        Mean days between events 17.300
    Share of gaps within 14 days  0.573
    Share of gaps within 30 days  0.828


![Figure 9.3. Evidence refresh curve: cumulative share of inter-event gaps by days elapsed. Synthetic data.](assets/figures/figure_9_3_expiration_policy.svg)

*Figure 9.3. Evidence refresh curve: cumulative share of inter-event gaps by days elapsed. Synthetic data.*


## 9.2 Improve The Baseline Engine


### Rank resource-constrained actions



```python
value = results["value_components_trace"].copy()
for column in [
    "predicted_response", "p_no_action", "p_action",
    "estimated_uplift_action", "fatigue_risk",
]:
    value[column] = value[column].round(3)
value["expected_incremental_value"] = value["expected_incremental_value"].round(1)
print(value.to_string(index=False))

```

             example        npi account_id  predicted_response  p_no_action  p_action  estimated_uplift_action  unit_cost  fatigue_risk  expected_incremental_value
    Highest response 9000000128     ACC160               0.844        0.764     0.954                    0.190      340.0          0.01                       418.0
       Highest value 9000000389     ACC155               0.593        0.483     0.748                    0.265      340.0          0.01                       718.0



```python
reward = results["reward_overlap"].copy()
allocation = results["constrained_allocation_summary"].copy()
allocation[[
    "mean_predicted_response", "mean_estimated_uplift",
    "expected_incremental_value",
]] = allocation[[
    "mean_predicted_response", "mean_estimated_uplift",
    "expected_incremental_value",
]].round(3)
print(reward.to_string(index=False))
print()
print(allocation.to_string(index=False))

```

                                   metric  value
    Promotional-eligible HCP-account rows 79.000
               Spearman response vs value  0.085
           Top-20 shared by both rankings  4.000
          Top-20 only in response ranking  7.000
    
    allocation_rule  released_slots  mean_predicted_response  mean_estimated_uplift  expected_incremental_value  shared_rows   candidate_action
    response_ranked              10                    0.797                  0.195                      4380.0            3 Program invitation
       value_ranked              10                    0.718                  0.223                      5500.0            3 Program invitation


### Explore safely



```python
history = results["logged_policy_history"][[
    "snapshot_id", "context_bucket", "eligible_actions",
    "base_policy_action", "logged_action",
    "logged_probability", "exploration_flag",
]].head(5).copy()
history["logged_probability"] = history["logged_probability"].round(3)
print(history.to_string(index=False))

```

    snapshot_id     context_bucket                                                         eligible_actions base_policy_action    logged_action  logged_probability  exploration_flag
      SNAP00001    Program-history Program invitation; Approved email; Continue responsive content; Monitor     Approved email   Approved email               0.925             False
      SNAP00002 Digital-responsive                                                         Access follow-up   Access follow-up Access follow-up               1.000             False
      SNAP00003   Field-responsive                                                                No action          No action        No action               1.000             False
      SNAP00004    Program-history Program invitation; Approved email; Continue responsive content; Monitor     Approved email   Approved email               0.925             False
      SNAP00005 Digital-responsive                                                         Access follow-up   Access follow-up Access follow-up               1.000             False



```python
exploration = results["thompson_exploration"][[
    "logged_action", "snapshots", "successes", "failures",
    "posterior_mean", "posterior_sd", "explore_share",
]].copy()
exploration[[
    "posterior_mean", "posterior_sd", "explore_share",
]] = exploration[[
    "posterior_mean", "posterior_sd", "explore_share",
]].round(3)
print(exploration.to_string(index=False))
print()
decision = results["thompson_decision_log"].copy()
decision["logged_probability"] = decision["logged_probability"].round(3)
print(decision.to_string(index=False))

```

                  logged_action  snapshots  successes  failures  posterior_mean  posterior_sd  explore_share
                        Monitor          7          5         2           0.667         0.149          0.484
               Access follow-up         90         58        32           0.641         0.050          0.212
    Continue responsive content         23         15         8           0.640         0.094          0.281
                 Approved email        117         67        50           0.571         0.045          0.023
                      No action         79         30        49           0.383         0.054          0.000
    
           npi account_id     context_bucket                                        eligible_arms base_policy_action selected_arm  logged_probability  exploration_flag        policy_version
    9000000280     ACC089 Digital-responsive Approved email; Continue responsive content; Monitor     Approved email      Monitor               0.484              True nba_policy_2025_02_v2


![Figure 9.4. Left: posterior mean and one SD per action. Right: share of Thompson draws each arm wins. Synthetic data.](assets/figures/figure_9_4_thompson.svg)

*Figure 9.4. Left: posterior mean and one SD per action. Right: share of Thompson draws each arm wins. Synthetic data.*


## 9.3 Evaluate A New Policy Offline


### 9.3.1 Replay The Candidate Policy



```python
trace = results["ope_replay_trace"].copy()
for column in [
    "logged_probability", "inverse_weight",
    "model_candidate_response", "dr_contribution",
]:
    trace[column] = trace[column].round(3)
print(trace.to_string(index=False))

```

    snapshot_id     context_bucket               logged_action candidate_action  logged_probability  future_response  matched  inverse_weight  model_candidate_response  dr_contribution
      SNAP00001    Program-history              Approved email   Approved email               0.925                1     True           1.081                     0.584            1.034
      SNAP00002 Digital-responsive            Access follow-up Access follow-up               1.000                1     True           1.000                     0.646            1.000
      SNAP00003   Field-responsive                   No action        No action               1.000                1     True           1.000                     0.374            1.000
      SNAP00043 Digital-responsive Continue responsive content   Approved email               0.033                0    False           0.000                     0.846            0.846
      SNAP00114 Digital-responsive Continue responsive content   Approved email               0.033                0    False           0.000                     0.545            0.545



```python
ope = results["off_policy_evaluation"].copy()
ope[[
    "estimated_response_rate", "match_rate",
    "effective_sample_size", "max_weight",
]] = ope[[
    "estimated_response_rate", "match_rate",
    "effective_sample_size", "max_weight",
]].round(3)
print(ope[["policy", "estimator", "estimated_response_rate"]].to_string(index=False))
print()
diag = ope.loc[ope["policy"].eq("digital_first")].iloc[0]
print("digital_first overlap diagnostics:")
print(f"  matched_snapshots:     {diag['matched_snapshots']}")
print(f"  match_rate:            {diag['match_rate']}")
print(f"  effective_sample_size: {diag['effective_sample_size']}")
print(f"  max_weight:            {diag['max_weight']}")
print(f"  overlap_warning:       {diag['overlap_warning']}")

```

           policy      estimator  estimated_response_rate
    logged_policy on_policy_mean                    0.573
    digital_first            ips                    0.575
    digital_first          snips                    0.573
    digital_first  direct_method                    0.574
    digital_first  doubly_robust                    0.573
    
    digital_first overlap diagnostics:
      matched_snapshots:     1392
      match_rate:            0.979
      effective_sample_size: 1390.472
      max_weight:            1.081
      overlap_warning:       Overlap acceptable


### 9.3.2 Design The Live Test



```python
design = results["experiment_design"].copy()
for _, r in design.iterrows():
    if r["parameter"] == "Guardrail outcomes":
        items = r["value"].split("; ")
        print(f"{'Guardrail outcomes':>36} {items[0]}")
        for item in items[1:]:
            print(f"{'':>37}{item}")
    else:
        print(f"{r['parameter']:>36} {r['value']}")

```

                      Randomization unit HCP-account row
                          Control policy Current precedence
                        Candidate policy Digital-first precedence
                         Primary outcome Meaningful response
                 Measurement window days 14
                      Guardrail outcomes Opt-out
                                         stale row
                                         field burden
                                         access delay
                  Baseline response rate 0.654
               Minimum detectable effect 0.05
                                   Power 0.8
                         Two-sided alpha 0.05
       Required HCP-account rows per arm 1367
    Eligible HCP-account rows this cycle 112
                           Cycles needed 25


## 9.4 More NBA Decisions


### 9.4.1 Operate The Last Mile



```python
feedback = results["execution_feedback_summary"].copy()
feedback["share"] = feedback["share"].round(3)
print(feedback.to_string(index=False))
print()
print(results["override_reason_summary"].to_string(index=False))

```

            execution_status  recommendations  share
                    Executed              117  0.741
                     Expired               17  0.108
         Viewed not executed               13  0.082
                  Overridden                9  0.057
    Suppressed after release                2  0.013
    
          override_reason  recommendations     example_action
    Access issue resolved                5   Access follow-up
          HCP unavailable                3 Program invitation
         Content mismatch                1     Approved email

