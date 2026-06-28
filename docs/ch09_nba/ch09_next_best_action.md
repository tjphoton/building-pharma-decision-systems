# Chapter 9: Next Best Action

The omnichannel work released a channel plan: one cycle action per HCP-account relationship, chosen after permission, access, pressure, and capacity checks. The next decision is the single dated action for each HCP-account row. On February 28, 2025, HCP0280 at account ACC089 could be called, emailed, invited to a peer or speaker program, routed to access follow-up, or left alone. Three teams could act on that relationship in the same week, or a compliant action could go unmade.

Next best action commits to exactly one action per relationship on a date, records the reason, records the alternatives it rejected, sets an expiration, and stays auditable. We build that engine from the channel-plan state. The engine generates a candidate menu, applies eligibility as a hard gate, resolves the choice by policy precedence, ranks inside the gates with predicted response and incremental uplift, and attaches a recommendation contract. It then adds the modern layer: a contextual bandit that explores where the policy is least certain, an off-policy estimate of an alternative policy before any live test, and the experiment that would settle the question. Open [`chapter9_walkthrough.ipynb`](chapter9_walkthrough.ipynb), or run the blocks below from the repository root.

> **Note:** All products, HCPs, accounts, and events are fictional and synthetic. The state, scores, and rewards are read from the omnichannel outputs, whose response rates are compressed high for teaching visibility. The decision architecture is what transfers, not the rate level.

## 9.1 The Candidate Set

The engine reads the omnichannel channel-plan state, the snapshot scores, and the event ledger, then builds a candidate menu for every relationship.

**Listing 9.1**: Load the engine package

```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

from ch09_nba.scripts.next_best_action import run_analysis

results = run_analysis(ROOT)
print(f"Recommendations: {len(results['recommendations'])}")
print(f"Candidates: {len(results['action_candidates'])}")
```

```text
Recommendations: 158
Candidates: 1106
```

Each of the 158 relationships generates a 7-action menu, so the candidate table holds 1,106 rows. The menu always includes no action. Inaction is a legitimate recommendation, and it is the only compliant one when permission or policy suppresses contact.

A selected action alone cannot show whether another action was considered, found ineligible, or simply ranked lower. Candidate generation makes the whole decision set inspectable. HCP0280 shows the menu.

**Listing 9.2**: Inspect one relationship's candidate menu

```python
candidates = results["action_candidates"]
trace = candidates.loc[candidates.npi.eq("9000000280")]
print(trace[[
    "candidate_action", "eligible", "policy_precedence", "reason_code"
]].to_string(index=False))
```

```text
           candidate_action  eligible  policy_precedence                                                  reason_code
                  No action      True                 90                  No higher-precedence eligible action passed
           Access follow-up     False                 10                   Account evidence points to access friction
         Field conversation     False                 20          Priority relationship with permitted field capacity
         Program invitation     False                 25   Prior live-program attendance supports a repeat invitation
             Approved email      True                 30  Available email frequency with a priority or digital signal
Continue responsive content      True                 40 Meaningful digital response without a higher-priority action
                    Monitor      True                 80       Eligible relationship without a stronger action signal
```

Four of HCP0280's seven candidates are eligible. The account is in a monitor state, not a priority account, so field conversation and program invitation fail their gates. The relationship has no recent access friction, so access follow-up fails. Approved email, responsive content, monitor, and no action remain.

![Figure 9.1. HCP0280 starts with 7 candidates; the gate removes access, field, and program actions, then precedence selects approved email and records the rejected alternatives. Synthetic data.](assets/figures/figure_9_1_decision_engine.svg)

*Figure 9.1. HCP0280 starts with 7 candidates; the gate removes access, field, and program actions, then precedence selects approved email and records the rejected alternatives. Synthetic data.*

## 9.2 Eligibility as a Hard Gate

Let $\mathcal{A}_i$ be the candidate actions for relationship $i$. The eligible set keeps only actions that pass every required gate:

$$
\mathcal{A}^{E}_i = \{a \in \mathcal{A}_i : g_k(i,a)=1 \text{ for every gate } k\}.
$$

Permission, access routing, recent pressure, and account priority define the gates. The gate is a hard constraint. An ineligible action never returns through a high score, which is the most common failure mode in systems built around one unified score.

| Failure | Result |
| --- | --- |
| Contact not permitted, or account on hold | Only no action remains |
| Account routed to access work | Remove promotional candidates, keep access follow-up |
| 5 or more events in the prior 30 days | Remove promotional candidates |
| Account not a priority | Remove field conversation |
| No prior live-program attendance | Remove program invitation |
| Email frequency at the cap, no digital or priority signal | Remove approved email |

The gates produce the eligibility flags in Listing 9.2. They run before any model score, and the policy applies the same gates in Section 9.3.

```python
reasons = [
    "Suppressed", "Access route first", "Not priority",
    "No live-program signal", "Passed",
]
gate_summary = results["gate_summary"].set_index("ineligibility_reason")
print(gate_summary.loc[reasons].reset_index().to_string(index=False))
```

```text
  ineligibility_reason  blocked_candidates
            Suppressed                 276
    Access route first                 175
          Not priority                  69
No live-program signal                  33
                Passed                 400
```

Suppression blocks the largest number of candidate actions because a relationship without permission or under an account hold can only release no action. Access routing blocks promotional candidates until the access issue is handled. Four hundred candidates pass their action-specific gates and move to precedence.

## 9.3 Policy Precedence

Among the eligible actions, the engine selects the one with the lowest precedence number. Suppression selects no action first. Access follow-up precedes promotion, because an unresolved barrier should be cleared before a promotional touch. Field conversation precedes program invitation and email for a priority account. Monitoring precedes an unsupported contact. The selected action is

$$
a_i^* = \arg\min_{a \in \mathcal{A}^{E}_i} \pi(a),
$$

where $\pi(a)$ is the declared precedence. A model may rank relationships inside a tier, but it never changes $\pi$.

**Listing 9.3**: Count recommendations by action

```python
summary = results["recommendation_summary"].copy()
summary["mean_predicted_response"] = summary.mean_predicted_response.round(3)
print(summary.to_string(index=False))
```

```text
         recommended_action  recommendations  review_required  mean_predicted_response
                  No action               46                0                    0.505
         Program invitation               37                0                    0.675
           Access follow-up               35               35                    0.678
             Approved email               16                0                    0.532
                    Monitor               15                0                    0.444
Continue responsive content                5                0                    0.656
         Field conversation                4                4                    0.645
```

The distribution reflects the gates and the synthetic state, not an optimization target. Permission and policy suppress 46 relationships into no action. Access friction routes 35 to access follow-up, each flagged for review. Prior live-program attendance makes program invitation the most common promotional action, 37 relationships. Only 4 priority accounts with permitted field capacity reach field conversation. HCP0280 takes the lowest-precedence eligible action on its menu, approved email at precedence 30.

## 9.4 Reward Design: Response and Uplift

The response model and uplift model have already been fitted in the channel analysis. This section does not refit either model. It uses their scores as fixed inputs to a different decision: which reward should control a scarce next-best-action slot after the gates and precedence rules have run.

That distinction matters. A response model answers, "Which HCP-account row is most likely to show a meaningful response in the next window?" An uplift model answers, "Which HCP-account row is most likely to change because we take this action?" A recommendation engine needs both, but it must assign each score to the right job.

The simplest intuition is a two-row program-invitation choice:

| Relationship | Probability of response with no invitation | Probability of response with invitation | Expected uplift | Reward if the goal is response | Reward if the goal is incremental change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sure thing | 82% | 86% | 4 points | 86% | 4 points |
| Persuadable | 43% | 58% | 15 points | 58% | 15 points |

The sure thing has the higher response probability. The persuadable row has the higher expected change. If the action is nearly free, response can be a useful rank because the team wants near-term engagement. If the action is scarce, expensive, or field-heavy, uplift is the better reward because the team wants incremental movement.

For relationship \(i\) and candidate action \(a\), the two planning rewards are:

$$
R^{\text{response}}_{i,a} = \hat{p}_{i,a},
\qquad
R^{\text{uplift}}_{i,a} =
\hat{p}_{i,a} - \hat{p}_{i,0}.
$$

Here \(\hat{p}_{i,a}\) is the estimated response probability if action \(a\) is taken, and \(\hat{p}_{i,0}\) is the estimated response probability under the control or routine-follow-up state. The next-best-action engine uses these rewards only after an action is eligible. A high reward never overrides permission, access routing, pressure, or capacity.

The practical rule used here is compact: use response to rank routine, low-cost follow-up inside a precedence tier; use uplift to review scarce promotional slots such as program invitations and field-heavy actions. The table below checks whether those two reward choices point to the same relationships.

**Listing 9.4**: Compare the response ranking with the uplift ranking

```python
overlap = results["reward_overlap"].copy()
print(overlap.to_string(index=False))
```

```text
                            metric  value
Promotional-eligible relationships  57.00
       Spearman response vs uplift  -0.64
    Top-20 shared by both rankings   0.00
   Top-20 only in response ranking  20.00
```

Across the 57 relationships eligible for a promotional action, the response score and the uplift score have a Spearman correlation of -0.64. The two rankings are strongly inverted: the relationships that respond most are the ones an action moves least. Rank the relationships by predicted response and by uplift, and the two top-20 lists share no names at all. All 20 of the highest responders sit outside the 20 highest-uplift relationships.

**Listing 9.5**: Trace where the rankings disagree

```python
reward = results["reward_candidates"].copy()
print(reward[[
    "npi", "candidate_action", "predicted_response",
    "estimated_uplift", "rank_by_response", "rank_by_uplift"
]].head(6).round(3).to_string(index=False))
```

```text
       npi   candidate_action  predicted_response  estimated_uplift  rank_by_response  rank_by_uplift
9000000026 Program invitation               0.909             0.050                 1              27
9000000239 Program invitation               0.905             0.005                 2              46
9000000505 Program invitation               0.873            -0.031                 3              56
9000000157 Program invitation               0.871             0.018                 4              39
9000000033 Program invitation               0.856             0.062                 5              25
9000000648     Approved email               0.847             0.019                 6              38
```

The six highest responders all sit deep in the uplift ranking, between 25th and 56th. HCP0505 ranks third by response but has a slightly negative estimated uplift: it is a near-certain responder that a program invitation would not move and might mildly fatigue. These are exactly the relationships a response ranking would call first and a budget-conscious plan should not. Ranking a scarce program by uplift sends it to the relationships it changes, not to the sure things.

![Figure 9.3. Promotional-eligible relationships split into high-response sure things and higher-uplift persuadable rows; HCP0505 shows why response alone can waste a scarce program slot. Synthetic data.](assets/figures/figure_9_3_reward_design.svg)

*Figure 9.3. Promotional-eligible relationships split into high-response sure things and higher-uplift persuadable rows; HCP0505 shows why response alone can waste a scarce program slot. Synthetic data.*

The engine keeps precedence as the primary order and uses these scores only inside a tier. The response score is the within-tier capacity rank. The uplift score is the signal a field manager should consult before spending a scarce program slot on a relationship that would convert anyway.

## 9.5 The Recommendation Contract

A selected action becomes a recommendation when it carries the context that lets a field team trust it, a compliance team audit it, and a data science team test it later. The contract attaches that context to every row.

**Listing 9.6**: Read the recommendation contract for one relationship

```python
recommendations = results["recommendations"]
row = recommendations.loc[recommendations.npi.eq("9000000280")].iloc[0]
for field in [
    "recommendation_id", "account_id", "recommended_action",
    "recommended_channel", "reason_code", "expected_result",
    "measurement_hook", "recommendation_date", "expires_on",
    "review_required",
]:
    print(f"{field}: {row[field]}")
```

```text
recommendation_id: NBA00129
account_id: ACC089
recommended_action: Approved email
recommended_channel: Email
reason_code: Available email frequency with a priority or digital signal
expected_result: Deliver approved content and earn a click
measurement_hook: Delivery and click
recommendation_date: 2025-02-28 00:00:00
expires_on: 2025-03-14 00:00:00
review_required: False
```

The reason code names the rule that selected the action, not a clinical intent or a sales claim. The expected result is operational: deliver approved content and earn a click. The measurement hook names what execution must record. The expiration freezes the evidence to a 14-day window. This row is the chapter's reusable artifact.

![Figure 9.2. The engine reduces 1,106 candidates to 400 eligible candidates and 158 selected actions, with most released rows going to no action, program invitation, and access follow-up. Synthetic data.](assets/figures/figure_9_2_recommendation_funnel.svg)

*Figure 9.2. The engine reduces 1,106 candidates to 400 eligible candidates and 158 selected actions, with most released rows going to no action, program invitation, and access follow-up. Synthetic data.*

## 9.6 Rejected-Alternative Audit

When a field manager asks why HCP0280 received an email rather than a field call, the audit answers. It labels every candidate as selected, ineligible, or eligible but lower precedence.

**Listing 9.7**: Summarize the candidate audit

```python
print(results["audit_summary"].to_string(index=False))
```

```text
             candidate_status  candidates
                   Ineligible         706
Eligible but lower precedence         242
                     Selected         158
```

**Listing 9.8**: Audit one relationship's rejected alternatives

```python
audit = results["candidate_audit"]
trace = audit.loc[audit.npi.eq("9000000280")]
print(trace[[
    "candidate_action", "candidate_status", "policy_precedence"
]].to_string(index=False))
```

```text
           candidate_action              candidate_status  policy_precedence
                  No action Eligible but lower precedence                 90
           Access follow-up                    Ineligible                 10
         Field conversation                    Ineligible                 20
         Program invitation                    Ineligible                 25
             Approved email                      Selected                 30
Continue responsive content Eligible but lower precedence                 40
                    Monitor Eligible but lower precedence                 80
```

HCP0280 received email because field conversation and program invitation were ineligible at the gate, and email was the highest-precedence action that passed. Responsive content and monitor were eligible but ranked lower. The audit makes that chain explicit for every relationship.

![Figure 9.4. Each candidate action is split into selected, eligible but lower precedence, or ineligible status, making rejected alternatives visible by action type. Synthetic data.](assets/figures/figure_9_4_candidate_audit.svg)

*Figure 9.4. Each candidate action is split into selected, eligible but lower precedence, or ineligible status, making rejected alternatives visible by action type. Synthetic data.*

## 9.7 Lifecycle and Expiration

A recommendation is a dated decision. New contact, consent, access, or treatment evidence can change eligibility, so a recommendation that sits unexecuted for too long can act on stale evidence. The 14-day expiration should follow from how fast the evidence actually refreshes, not from convention.

**Listing 9.9**: Measure the evidence refresh rate

```python
print(results["expiration_analysis"].to_string(index=False))
```

```text
                      metric  value
  Median days between events 11.000
    Mean days between events 17.100
Share of gaps within 14 days  0.586
Share of gaps within 30 days  0.838
```

The median gap between consecutive events for the same relationship is 11 days, inside the 14-day window. Nearly 60 percent of all gaps fall within 14 days, so most recommendations will face new evidence at the next contact and should be recomputed by then. A 30-day window would let a third of relationships accumulate a new event before the recommendation refreshed. The engine recomputes after the window and preserves the prior record, so execution, rejection, and later outcome stay auditable.

![Figure 9.5. The cumulative refresh curve shows that 59% of relationship event gaps close within 14 days and 84% close within 30 days. Synthetic data.](assets/figures/figure_9_5_expiration.svg)

*Figure 9.5. The cumulative refresh curve shows that 59% of relationship event gaps close within 14 days and 84% close within 30 days. Synthetic data.*

## 9.8 Exploration with a Contextual Bandit

The precedence engine exploits current evidence. It always selects the highest-precedence eligible action. That is the right default for a settled policy where field consistency and compliance matter. Learning the policy requires some governed exploration, because a fixed policy rarely tries an action outside its current preference.

For HCP0280, the context bucket is `Digital-responsive`: the relationship has a recent qualified digital action and remains eligible for email. A contextual bandit estimates action value inside a context like that one. The engine chooses an action, observes the later outcome, and updates its belief about how well each action works for similar HCP-account rows. The exploration question is how often to try a less-favored eligible action to learn.

The intuition is a row of slot machines whose payout rates are unknown. Pulling the arm that has paid best so far exploits what you know. Pulling an uncertain arm explores. Thompson sampling resolves the tradeoff with a simple rule. Represent each action's success rate as a Beta distribution, wide when evidence is thin and narrow when evidence is thick. To choose, draw one random number from each action's distribution and take the highest draw. An action with a high estimated rate is usually chosen. An action whose distribution is wide still wins some draws, so exploration concentrates exactly where uncertainty is greatest.

The Beta distribution updates by counting. After $s$ successes and $f$ failures, the action's distribution is $\text{Beta}(s+1, f+1)$, with mean $(s+1)/(s+f+2)$. More data tightens it around the true rate.

**Listing 9.10**: Seed the digital-responsive arms from full history

```python
exploration = results["thompson_exploration"].copy()
print(exploration[[
    "context_bucket", "logged_action", "snapshots", "posterior_mean",
    "posterior_sd", "explore_share"
]].to_string(index=False))
```

```text
    context_bucket      logged_action  snapshots  posterior_mean  posterior_sd  explore_share
Digital-responsive     Approved email        141           0.615         0.041          0.804
Digital-responsive Field conversation        101           0.563         0.049          0.196
Digital-responsive          No action         89           0.407         0.051          0.000
```

The `explore_share` is the fraction of 2,000 Thompson draws in which each action would be chosen. With the full history seeding the digital-responsive arms, approved email wins 80 percent of draws. Field conversation still wins about 20 percent because its posterior overlaps with email. No action has a lower posterior mean and rarely wins.

Exploration only appears when the evidence is thin. Seeding the same arms from the first 2 months changes the picture.

**Listing 9.11**: Seed the digital-responsive arms from a cold start

```python
cold = results["thompson_cold_start"].copy()
print(cold[[
    "context_bucket", "logged_action", "snapshots", "posterior_mean",
    "posterior_sd", "explore_share"
]].to_string(index=False))
```

```text
    context_bucket      logged_action  snapshots  posterior_mean  posterior_sd  explore_share
Digital-responsive          No action         24           0.615         0.094          0.520
Digital-responsive     Approved email         38           0.575         0.077          0.289
Digital-responsive Field conversation         17           0.526         0.112          0.192
```

With only 2 months of evidence, the digital-responsive arms have wide, overlapping distributions. No action wins 52 percent of draws, approved email wins 29 percent, and field conversation wins 19 percent. As evidence accumulates, approved email separates from the other arms and exploration fades. The context matters: a program-history relationship would have a different arm set and a different posterior.

![Figure 9.6. For digital-responsive relationships, cold-start action posteriors overlap and Thompson sampling explores; with full history, approved email separates from the other arms. Synthetic data.](assets/figures/figure_9_6_thompson.svg)

*Figure 9.6. For digital-responsive relationships, cold-start action posteriors overlap and Thompson sampling explores; with full history, approved email separates from the other arms. Synthetic data.*

A production system that adds a bandit treats the exploration arm as a governed randomized experiment, with prespecified outcome windows, suppression of ineligible actions, and human review of the explored population. The bandit ranks inside fixed gates.

## 9.9 Off-Policy Evaluation of an Alternative Policy

Suppose leadership proposes a digital-first variant: for a priority relationship with a high response score, put approved email ahead of field conversation. Testing it live is expensive and slow. Off-policy evaluation estimates the variant's value from the logged history of the current policy, before any live test.

The logged history records, for each past snapshot, the action the base policy took and whether a meaningful response followed. Inverse-propensity scoring reweights each logged reward by the inverse of the probability that the logging policy chose that action, counting only snapshots where the logged action matches what the candidate policy would have chosen:

$$
\hat{V}_{\text{IPS}}(\pi) = \frac{1}{N}\sum_i \frac{\mathbb{1}\{a_i = \pi(x_i)\}\, y_i}{p_i}.
$$

When some actions were logged rarely, their inverse weights are large, and a few matched rows dominate the estimate. Self-normalized IPS divides by the sum of weights rather than $N$, which removes that scale distortion. The doubly-robust estimator adds a fitted reward model and stays accurate if either the weights or the reward model is right.

**Listing 9.12**: Evaluate the digital-first variant four ways

```python
policy = results["off_policy_evaluation"].copy()
policy["estimated_response_rate"] = policy.estimated_response_rate.map(
    lambda x: f"{x:.1%}"
)
policy["effective_sample_size"] = policy.effective_sample_size.round(1)
print(policy.to_string(index=False))
```

```text
       policy      estimator estimated_response_rate  matched_snapshots  effective_sample_size
logged_policy on_policy_mean                   56.9%               1422                 1422.0
digital_first            ips                   54.8%               1288                 1287.1
digital_first          snips                   55.6%               1288                 1287.1
digital_first  doubly_robust                   57.1%               1288                 1287.1
```

The digital-first variant differs from the logged policy on 134 of 1,422 snapshots, so overlap is high. The effective sample size stays close to the 1,288 matched snapshots because the logging propensities are stable. IPS, self-normalized IPS, and the doubly-robust estimator all place the variant near the logged policy, from 54.8% to 57.1% against the logged 56.9%. That spread is too small to justify a policy switch. The off-policy result screens the variant as plausible, then sends the question to a live randomized test.

## 9.10 The Experiment That Would Settle It

Off-policy evaluation narrows the field. A randomized test settles it. The cleanest design assigns each eligible relationship to the current precedence or the digital-first variant, holds every eligibility gate identical across arms, and measures meaningful response within the recommendation window.

**Listing 9.13**: Size the precedence experiment

```python
print(results["experiment_design"].to_string(index=False))
```

```text
                        parameter    value
           Baseline response rate    0.588
        Minimum detectable effect    0.050
                            Power    0.800
                  Two-sided alpha    0.050
   Required relationships per arm 1488.000
Eligible relationships this cycle  112.000
        Cycles to reach both arms   27.000
```

Detecting a 5-point absolute change from a 59 percent baseline, at 80 percent power and a two-sided 5 percent level, needs about 1,488 relationships per arm. This cycle has 112 eligible relationships. The eligible population is far smaller than a well-powered test requires, so a recommendation-level experiment must pool across roughly 27 planning cycles or across geographies. That sample-size reality, not the statistics, is the binding constraint on recommendation-level testing in commercial settings. The recommendation log built here becomes the randomization register for that test.

## 9.11 The Frontier

The engine spans the practical spectrum. At the basic end, a deterministic rule engine with hard gates, precedence, reason codes, and an audit trail is what a compliance team will accept and a field team will trust. At the modern end, response and uplift scores rank inside the gates, a contextual bandit explores where the policy is least certain, and off-policy evaluation screens a new policy before deployment.

Beyond this chapter, the same architecture extends in two directions. When actions chain over time, so that today's email changes the value of next month's field call, the decision becomes a sequence and offline reinforcement learning, such as fitted Q-evaluation, estimates the value of a multi-step policy from logged trajectories. When many specialized models propose actions, an agentic orchestration layer arbitrates among them. Both raise the same governance question this chapter answered for the single-step case: the optimizer proposes inside the gates, and permission, access, pressure, capacity, and approved-content rules run first. The current guidance is to start with the rule engine and bandit, and graduate to constrained reinforcement learning only when the actions genuinely chain and the governance around exploration is in place.

## 9.12 From State to Governed Recommendation

The opening problem was that a channel plan leaves the single action undecided and the responsibility unassigned. The engine resolved it. It generated a 7-action menu for each of 158 relationships, removed ineligible actions at the gate, and selected one action per relationship by precedence: 46 to no action, 37 to a program invitation, 35 to access follow-up, and a small set to field and email. It ranked scarce promotional slots by uplift rather than response, so a near-certain responder like HCP0505 does not consume a program slot. It attached a reason, an expected result, a measurement hook, and a 14-day expiration justified by an 11-day median refresh, and it recorded every rejected alternative. For HCP0280 the governed recommendation is approved email, with the field and program actions shown ineligible and the reason on the row.

## 9.13 Summary

Next best action turns a dated state into one auditable action per relationship.

- Generate the full candidate menu, including no action, before selecting.
- Treat eligibility as a hard gate that no score can override.
- Resolve the choice by declared precedence, and rank only inside a tier.
- Rank scarce actions by uplift, not predicted response, to avoid spending on sure things.
- Attach a reason code, expected result, measurement hook, expiration, and review flag to every row.
- Record every rejected alternative as selected, ineligible, or lower precedence.
- Justify the expiration window from the measured evidence refresh rate.
- Explore with a bandit only where the action posteriors overlap, and keep exploration inside the gates.
- Screen an alternative policy off-policy before a live test, and compare IPS, self-normalized IPS, doubly-robust estimates, overlap, and effective sample size.
- Size the confirmatory experiment, and expect to pool cycles to reach power.

> **What you have learned from this chapter:** You can now commit to exactly one dated action per relationship by generating the full candidate menu, applying eligibility as a hard gate, resolving the choice by policy precedence, and ranking inside a tier with response and uplift. You know to record the action, reason, expected result, measurement hook, expiration, and rejected alternatives on the same row, to let a contextual bandit explore only where the policy is least certain, and to keep every learning component proposing inside fixed gates that permission, access, pressure, and capacity define.

## 9.14 Exercises

1. **Reverse field and email precedence.** Swap the precedence of field conversation and approved email, rebuild the recommendations, and report how many relationships change action. State which business function the original ordering represents and what the swap would cost. (Section 9.3.)
2. **Rank a tier by uplift.** Within the field-eligible relationships, select the field slots by estimated uplift rather than predicted response, and compare the two selected sets. Name one relationship that the response ranking would call and the uplift ranking would not, and say why. (Section 9.4.)
3. **Design the precedence test.** You have run the digital-first variant off-policy and the doubly-robust estimate is close to the logged policy. Specify the randomized test you would register to decide it: the randomization unit, the control arm, the primary outcome, the measurement window, and the number of cycles you would expect to need. End with the one real-world evidence source you would require before trusting the off-policy estimate. (Sections 9.9 and 9.10.)

Worked solutions are in [`exercise_solutions.ipynb`](exercise_solutions.ipynb). Each solution ends with the judgment an analyst should record for real data.

The experiments chapter takes the recommendation log built here as its randomization register and measures incremental effect.
