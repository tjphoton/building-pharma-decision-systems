# Building Decision Systems: A Hands-On Playbook for Pharmaceutical Commercial Decision Science

**A practical guide to pharmaceutical commercial analytics, from raw data to defensible action.**

Pharmaceutical companies have no shortage of data. The harder problem is deciding what to do with it. A forecast may size a market, but a launch team still has to decide where to focus. A targeting model may rank physicians, but a field leader still has to decide which accounts deserve attention and why. A campaign report may show a lift in prescriptions, but a brand team still has to judge whether the campaign caused the change.

No single book covers the full path: from commercial data infrastructure through patient journeys, targeting, competitive intelligence, omnichannel measurement, causal inference, and AI-supported decision engines. This is the book I wish I had before I learned these topics the hard way.

It follows the complete fictional launch of **Roventra**, a once-daily oral medicine for a chronic condition, from FDA approval to field execution.

---

## Book Structure

The book moves from foundations to execution. It starts with the launch context and data layer, then moves through market and customer understanding, engagement, measurement, and decision engines. Every method, including the advanced ones, is demonstrated with working Python and verified output.

| Part | Focus | Decision capability |
| --- | --- | --- |
| Part 1 | Foundations and data | Frame the launch problem and build the data base |
| Part 2 | Market and customer understanding | Find patient opportunity and explain treatment behavior |
| Part 3 | Engagement | Choose channels and define the next action |
| Part 4 | Measurement and causal inference | Estimate impact and compare performance |
| Part 5 | Decision engines | Forecast, allocate resources, and govern recommendations |

## Table of Contents

### Part 1: Foundations and Data

<div><a href="ch01_intro/ch01_introduction/">Chapter 1. A Medicine, a Market, and the Decisions Between Them</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#11-the-roventra-world">1.1 The Roventra World</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#12-from-a-finding-to-a-decision">1.2 From a Finding to a Decision</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#13-the-decision-record">1.3 The Decision Record</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#14-five-kinds-of-analytical-work">1.4 Five Kinds of Analytical Work</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#15-three-questions-for-the-book">1.5 Three Questions for the Book</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#16-business-close">1.6 Business Close</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#17-summary">1.7 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#18-exercises">1.8 Exercises</a></div>
<div style="margin-left: 1.5em;"><a href="ch01_intro/ch01_introduction/#19-exercise-solutions">1.9 Exercise Solutions</a></div>
<div><a href="ch02_ecosystem/ch02_ecosystem/">Chapter 2. The Commercialization Operating System</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#21-prescription-to-treatment-path">2.1 Prescription to Treatment Path</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#22-the-launch-organization-around-the-path">2.2 The Launch Organization Around the Path</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#23-what-each-event-leaves-behind">2.3 What Each Event Leaves Behind</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#24-summary">2.4 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#25-exercises">2.5 Exercises</a></div>
<div style="margin-left: 1.5em;"><a href="ch02_ecosystem/ch02_ecosystem/#26-exercise-solutions">2.6 Exercise Solutions</a></div>

<div><a href="ch03_data/ch03_data/">Chapter 3. A Synthetic Lab for Real Pharma Questions</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#31-the-pharmaceutical-data-sources">3.1 The Pharmaceutical Data Sources</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#32-synthetic-data-design-and-generation">3.2 Synthetic Data: Design and Generation</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#33-one-patient-through-multiple-data-sources">3.3 One Patient through Multiple Data Sources</a></div>
<div style="margin-left: 3em;"><a href="ch03_data/ch03_data/#331-medical-claims">3.3.1 Medical claims</a></div>
<div style="margin-left: 3em;"><a href="ch03_data/ch03_data/#332-pharmacy-claims">3.3.2 Pharmacy claims</a></div>
<div style="margin-left: 3em;"><a href="ch03_data/ch03_data/#333-lab-results">3.3.3 Lab results</a></div>
<div style="margin-left: 3em;"><a href="ch03_data/ch03_data/#334-formulary-and-access">3.3.4 Formulary and access</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#34-pre-analysis-data-checks">3.4 Pre-Analysis Data Checks</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#35-summary">3.5 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch03_data/ch03_data/#36-exercises">3.6 Exercises</a></div>

### Part 2: Market and Customer Understanding

<div><a href="ch04_market/ch04_market_sizing/">Chapter 4. Market Sizing and Patient Populations</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#41-one-disease-one-medicine-four-market-sizes">4.1 One Disease, One Medicine, Four Market Sizes</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#42-what-claims-can-see-and-miss">4.2 What Claims Can See and Miss</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#43-one-diagnosis-or-two">4.3 One Diagnosis or Two?</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#44-national-prevalence-anchor-and-opportunity-funnel">4.4 National Prevalence Anchor and Opportunity Funnel</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#45-the-unobserved-population">4.5 The Unobserved Population</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#46-sdoh-and-under-observation">4.6 SDOH and Under-Observation</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#47-patient-finding-from-count-to-list">4.7 Patient Finding: From Count to List</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#48-from-a-scored-list-to-a-commercial-action">4.8 From a Scored List to a Commercial Action</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#49-market-sizing-bridge">4.9 Market-Sizing Bridge</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#410-summary">4.10 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch04_market/ch04_market_sizing/#411-exercises">4.11 Exercises</a></div>
<div><a href="ch05_journey/ch05_patient_journey/">Chapter 5. Building the Patient Journey</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#51-define-the-journey">5.1 Define the Journey</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#52-lines-of-therapy-sequence-rules-and-patterns">5.2 Lines of Therapy: Sequence Rules and Patterns</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#521-the-washout-rule">5.2.1 The washout rule</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#522-the-rule-set">5.2.2 The rule set</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#523-switch-example">5.2.3 Switch example</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#524-addition-example">5.2.4 Addition example</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#525-cohort-treatment-pattern">5.2.5 Cohort treatment pattern</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#526-treatment-sequence-in-commercial-reporting">5.2.6 Treatment Sequence in Commercial Reporting</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#53-time-to-treatment-when-patients-start-therapy">5.3 Time to Treatment: When Patients Start Therapy</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#531-why-naive-average-fails">5.3.1 Why naive average fails</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#532-kaplan-meier-fundamentals">5.3.2 Kaplan-Meier Fundamentals</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#533-kaplan-meier-estimation-in-the-cohort">5.3.3 Kaplan-Meier Estimation in the Cohort</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#534-competing-risk-the-aalen-johansen-method">5.3.4 Competing Risk: The Aalen-Johansen Method</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#535-initiation-curves-in-commercial-planning">5.3.5 Initiation Curves in Commercial Planning</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#54-staying-on-therapy-persistence-and-adherence">5.4 Staying on Therapy: Persistence and Adherence</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#541-persistence-time-until-departure-from-the-initial-regimen">5.4.1 Persistence: time until departure from the initial regimen</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#542-adherence-coverage-during-the-observed-window">5.4.2 Adherence: coverage during the observed window</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#543-product-scope-in-adherence-measurement">5.4.3 Product Scope in Adherence Measurement</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#544-payer-adherence-rates-with-confidence-intervals">5.4.4 Payer Adherence Rates with Confidence Intervals</a></div>
<div style="margin-left: 3em;"><a href="ch05_journey/ch05_patient_journey/#545-persistence-and-adherence-in-commercial-strategy">5.4.5 Persistence and Adherence in Commercial Strategy</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#55-sdoh-and-refill-gaps">5.5 SDOH and Refill Gaps</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#56-modern-extensions-to-rule-based-patient-journeys">5.6 Modern Extensions to Rule-Based Patient Journeys</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#57-summary">5.7 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch05_journey/ch05_patient_journey/#58-exercises">5.8 Exercises</a></div>
<div><a href="ch06_hcp/ch06_hcp_targeting/">Chapter 6. HCP Targeting</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#61-generate-supplemental-datasets">6.1 Generate Supplemental Datasets</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#62-assign-patients-to-hcps">6.2 Assign Patients to HCPs</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#63-build-the-hcp-evidence-table">6.3 Build the HCP Evidence Table</a></div>
<div style="margin-left: 3em;"><a href="ch06_hcp/ch06_hcp_targeting/#631-opportunity-adoption-and-permission">6.3.1 Opportunity, Adoption, and Permission</a></div>
<div style="margin-left: 3em;"><a href="ch06_hcp/ch06_hcp_targeting/#632-opportunity-concentration">6.3.2 Opportunity Concentration</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#64-map-referral-pathways">6.4 Map Referral Pathways</a></div>
<div style="margin-left: 3em;"><a href="ch06_hcp/ch06_hcp_targeting/#641-build-the-referral-graph">6.4.1 Build the Referral Graph</a></div>
<div style="margin-left: 3em;"><a href="ch06_hcp/ch06_hcp_targeting/#642-referral-flows">6.4.2 Referral Flows</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#65-build-kol-scientific-profiles">6.5 Build KOL Scientific Profiles</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#66-segment-hcp-engagement-patterns">6.6 Segment HCP Engagement Patterns</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#67-build-the-4-week-call-plan">6.7 Build the 4-Week Call Plan</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#68-summary">6.8 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch06_hcp/ch06_hcp_targeting/#69-exercises">6.9 Exercises</a></div>
<div><a href="ch07_competitive/ch07_competitive_intelligence_market_access/">Chapter 7. Competitive Intelligence and Market Access</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#71-generate-teaching-datasets">7.1 Generate Teaching Datasets</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#72-build-effective-dated-access">7.2 Build Effective-Dated Access</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#73-measure-prescriptions-nbrx-nrx-and-trx">7.3 Measure Prescriptions: NBRx, NRx, and TRx</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#74-separate-access-from-adoption">7.4 Separate Access from Adoption</a></div>
<div style="margin-left: 3em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#741-partial-pooling">7.4.1 Partial Pooling</a></div>
<div style="margin-left: 3em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#742-payer-region-actions">7.4.2 Payer-Region Actions</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#75-measure-the-formulary-event">7.5 Measure the Formulary Event</a></div>
<div style="margin-left: 3em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#751-fit-the-controlled-interrupted-time-series">7.5.1 Fit the Controlled Interrupted Time Series</a></div>
<div style="margin-left: 3em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#752-check-with-synthetic-control">7.5.2 Check with Synthetic Control</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#76-summary">7.6 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch07_competitive/ch07_competitive_intelligence_market_access/#77-exercises">7.7 Exercises</a></div>

### Part 3: Engagement

<div><a href="ch08_omnichannel/ch08_omnichannel_analytics/">Chapter 8. Omnichannel Analytics</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#81-the-event-ledger">8.1 The Event Ledger</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#811-generate-the-engagement-data">8.1.1 Generate the Engagement Data</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#812-standardize-the-ten-channels">8.1.2 Standardize the Ten Channels</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#82-prepare-the-modeling-data">8.2 Prepare the Modeling Data</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#821-past-state-and-later-outcome">8.2.1 Past State and Later Outcome</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#822-sparse-response-signals">8.2.2 Sparse Response Signals</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#83-the-response-model">8.3 The Response Model</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#831-regularized-logistic-regression">8.3.1 Regularized Logistic Regression</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#832-channel-order-effects">8.3.2 Channel Order Effects</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#84-separate-credit-from-impact">8.4 Separate Credit from Impact</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#841-reach-and-saturation">8.4.1 Reach and Saturation</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#842-attribution-credit">8.4.2 Attribution Credit</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#843-incrementality-who-responds-because-of-us">8.4.3 Incrementality: Who Responds Because of Us</a></div>
<div style="margin-left: 3em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#844-credit-lift-and-cost">8.4.4 Credit, Lift, and Cost</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#85-the-channel-plan">8.5 The Channel Plan</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#86-summary">8.6 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch08_omnichannel/ch08_omnichannel_analytics/#87-exercises">8.7 Exercises</a></div>
<div><a href="ch09_nba/ch09_next_best_action/">Chapter 9. Next Best Action</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#91-build-the-nba-recommendation-engine">9.1 Build The NBA Recommendation Engine</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#911-load-the-state">9.1.1 Load The State</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#912-build-and-gate-the-candidate-menu">9.1.2 Build and Gate the Candidate Menu</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#913-understand-the-content-approval-layer">9.1.3 Understand the Content Approval Layer</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#914-read-the-gate-distribution">9.1.4 Read the Gate Distribution</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#915-set-the-expiration">9.1.5 Set The Expiration</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#92-improve-the-baseline-engine">9.2 Improve The Baseline Engine</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#921-rank-resource-constrained-actions">9.2.1 Rank Resource-Constrained Actions</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#922-explore-safely">9.2.2 Explore Safely</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#93-evaluate-a-new-policy-offline">9.3 Evaluate A New Policy Offline</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#931-replay-the-candidate-policy">9.3.1 Replay The Candidate Policy</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#932-design-the-live-test">9.3.2 Design The Live Test</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#94-more-nba-decisions">9.4 More NBA Decisions</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#941-operate-the-last-mile">9.4.1 Operate The Last Mile</a></div>
<div style="margin-left: 3em;"><a href="ch09_nba/ch09_next_best_action/#942-extend-the-engine-with-modern-techniques">9.4.2 Extend The Engine With Modern Techniques</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#95-summary">9.5 Summary</a></div>
<div style="margin-left: 1.5em;"><a href="ch09_nba/ch09_next_best_action/#96-exercises">9.6 Exercises</a></div>
### Part 4: Measurement and Causal Inference

<div style="margin-left: 1.5em;">Chapter 10. Randomized Experiments and Incrementality</div>
<div style="margin-left: 1.5em;">Chapter 11. Natural Experiments and Quasi-Experimental Designs</div>
<div style="margin-left: 1.5em;">Chapter 12. Observational Causal Inference and Real-World Evidence</div>
<div><a href="ch13_mmm/ch13_mmm_unified_measurement/">Chapter 13. MMM and Unified Measurement</a></div>

### Part 5: Decision Engines

<div><a href="ch14_forecasting/ch14_forecasting/">Chapter 14. Forecasting from Launch to Loss of Exclusivity</a></div>
<div style="margin-left: 1.5em;">Chapter 15. Resource Allocation and Optimization</div>
<div style="margin-left: 1.5em;">Chapter 16. AI Decision Intelligence</div>
<div style="margin-left: 1.5em;">Chapter 17. Capstone Case Studies</div>

---

## What You Will Build

- A linked synthetic pharmaceutical dataset that mirrors claims, EHR, payer, CRM, and specialty pharmacy data
- Patient population funnels from true prevalence to eligible, treated patients
- Lines of therapy with explicit washout, switch, addition, and persistence rules
- HCP targeting with transparent scoring, action filters, and field capacity allocation
- Competitive intelligence on formulary position, payer access barriers, and corrected treatment share
- Omnichannel engagement plans, next-best-action recommendations, and incrementality tests
- Causal inference models and marketing mix models for unified measurement
- A resource allocation optimizer and a governed AI decision engine

---

## Who This Is For

**Commercial analysts and data scientists** in pharmaceutical companies who need to move from reporting to decision support. The book focuses on the judgment layer: when a method is good enough, when it is not, and how to present a finding as a recommendation with a clear owner and measure of success.

**Analytics engineers and BI teams** building commercial data platforms. The data chapter covers claim receipt lag, drug code mapping gaps, and data quality rules that protect downstream analysis.

**Brand teams and commercial leaders** who review analytical recommendations and want to understand what the data can and cannot tell them about market opportunity, customer behavior, and campaign impact.

**Students and independent consultants** entering pharmaceutical commercial analytics. The book assumes no prior pharma experience and defines terms at first use.

### Prerequisites

- Python 3.11 or later
- Familiarity with pandas and basic data manipulation
- Basic understanding of statistical concepts (means, distributions, regression)

No pharmaceutical industry background is required. No prior exposure to pharma data sources, commercial roles, or marketing experience is assumed.

---

## Getting Started

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/tjphoton/building-pharma-decision-systems.git
cd building-pharma-decision-systems

# Install dependencies
uv sync

# Launch Jupyter to run chapter notebooks
uv run jupyter lab
```

Hands-on analytical chapters from Chapter 3 onward contain two executed notebooks:

- `chNN_walkthrough.ipynb`: the chapter as one executable story
- `chNN_exercise_solutions.ipynb`: worked answers with analyst judgment notes

Run the notebooks in order within each analytical chapter. Data generation is handled inside each walkthrough; no external data download is required. Chapters 1 and 2 are prose foundations and do not require companion notebooks.


---

## The Roventra World

All examples run against a single fictional launch. Using a consistent case across 15 chapters means the patient found in the data chapter reappears in the targeting chapter, the payer that denied coverage in the competitive chapter feeds the access analysis in the resource allocation chapter, and the HCP ranked first in targeting becomes the unit of measurement in the incrementality chapter.

| Entity | ID | Role |
| --- | --- | --- |
| Roventra | `90000-1001-11` | The launch product: once-daily oral medicine |
| Nexoral | `90000-1002-11` | Established oral competitor |
| Vexpro | `90000-1003-11` | Established weekly injectable competitor |
| Patient | `PAT02034` | The canonical patient traced across all data sources |
| HCP | `HCP0280` | The prescriber: a targeting priority and measurement unit |
| Account | `ACC089` | The clinic where field prioritization and call planning occur |
| Payer | `PAY002` | The organization whose formulary decisions shape patient access |

---

## About the Author

Xinjie Qiu has spent more than a decade leading data organizations at Havas Health, Real Chemistry and Horizon Next, building marketing data science, advanced analytics, pharmaceutical and healthcare commercial analytics capabilities, and working with brands including Pfizer, Sanofi, Novartis, Amgen, UnitedHealthcare, Google on new product launch, HCP targeting, patient journey analysis, marketing causal inference, and AI-supported decision systems. His background combines a PhD in Physics from the University of Minnesota with oncology research at Memorial Sloan Kettering Cancer Center.

Connect on [LinkedIn](https://www.linkedin.com/in/xinjieqiu).

---

*Fictional products, patients, HCPs, accounts, payers, and clinical events are used throughout. No real patient data appears in this repository.*
