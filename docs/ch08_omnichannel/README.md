# Chapter 8: Omnichannel Analytics: Planning and Attribution

Welcome to Chapter 8! This chapter unifies ten engagement channels into a trusted event ledger and builds the analytics to plan next actions and measure channel impact.

## What You'll Learn

In this chapter, you'll discover:
- **Event ledger construction** - Standardize ten communication channels into a single trustworthy record
- **Response modeling** - Build predictive models from historical channel engagement and outcomes
- **Channel attribution** - Identify which past channels are associated with HCP response
- **Uplift analysis** - Test whether associated channels actually caused response, not just correlation
- **Channel economics** - Convert causal estimates into cost per incremental response
- **Engagement sequencing** - Design effective channel combinations and sequences
- **Governed channel plans** - Build transparent, auditable engagement recommendations
- **Measurement contracts** - Create metrics that feed the next-best-action engine

## Read the Full Chapter

👉 **[Start reading Chapter 8: Omnichannel Analytics](ch08_omnichannel_analytics.md)**

Also available:
- 📓 **[Walkthrough Notebook](chapter8_walkthrough.ipynb)** - Interactive Python notebook with step-by-step code examples
- 🧪 **[Exercise Solutions](exercise_solutions.ipynb)** - Solutions to chapter exercises

This chapter teaches you to move from "Which channels work?" to "Which channel combinations work best for this HCP, at this time, under this access state, and at what cost per incremental outcome?"

## Chapter Sections

**8.1 Teaching Data**
Generate and understand the synthetic engagement records across ten channels from January 2024 through March 2025. Load the complete analysis package with event ledger, HCP-account snapshots, and channel plans.

**8.2 Prepare the Modeling Data**
Build the event ledger that unifies ten source systems into one trustworthy record. Standardize channel-specific fields, define meaningful response metrics, and prepare snapshot features for modeling.

**8.3 Response Modeling**
Train predictive models on historical channel engagement to forecast HCP response probability. Handle sparse response rates and build model evaluation frameworks.

**8.4 Channel Attribution**
Identify which past channels are associated with HCP response. Implement attribution methods and analyze channel contributions to outcomes.

**8.5 Uplift Analysis**
Test whether associated channels actually caused response, not just correlation. Use causal inference methods to estimate true channel treatment effects.

**8.6 Off-Policy Evaluation**
Evaluate channel strategies without running them live. Use off-policy evaluation estimators to test channel plans before deployment.

**8.7 Channel Economics**
Convert causal estimates into cost per incremental response. Calculate channel ROI and compare engagement strategies.

**8.8 Governed Channel Planning**
Build transparent, auditable engagement recommendations. Create channel plans with decision rules and governance structures.

**8.9 Measurement Contracts**
Design metrics that feed the next-best-action engine. Define measurement frameworks for closed-loop learning.

## Engagement Channels Covered

The chapter works with ten channels unified in a single ledger:
1. **Field** - Face-to-face HCP interactions
2. **Email** - Approved branded communications
3. **Web** - Authenticated HCP portal engagement
4. **Peer programs** - Small peer-to-peer education sessions
5. **Speaker programs** - Formal speaker-led events
6. **Paid media** - Digital advertising
7. **Conferences** - Congress and event engagement
8. **Direct mail** - Postal communications
9. **Phone** - Outbound support calls
10. **Account support** - Access and fulfillment assistance
