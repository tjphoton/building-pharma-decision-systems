# Building Decision Systems: A Hands-On Playbook for Pharmaceutical Commercial Decision Science

**A practical guide to pharmaceutical commercial analytics, from raw data to defensible action.**

Pharmaceutical companies have no shortage of data. The harder problem is deciding what to do with it. A forecast may size a market, but a launch team still has to decide where to focus. A targeting model may rank physicians, but a field leader still has to decide which accounts deserve attention and why. A campaign report may show a lift in prescriptions, but a brand team still has to judge whether the campaign caused the change.

No single book covers the full path: from commercial data infrastructure through patient journeys, targeting, competitive intelligence, omnichannel measurement, causal inference, and AI-supported decision engines. This is the book I wish I had before I learned these topics the hard way.

It follows the complete fictional launch of **Roventra**, a once-daily oral medicine for a chronic condition, from FDA approval to field execution.

---

## What You Will Build

- A linked synthetic pharmaceutical dataset that mirrors real claims, EHR, payer, CRM, and specialty pharmacy structures
- Patient population funnels from true prevalence down to eligible, treated patients
- Lines of therapy with explicit washout, switch, addition, and persistence rules
- HCP and account targeting with transparent scoring, actionability filters, and field capacity allocation
- Competitive intelligence: formulary position, payer access barriers, and corrected treatment share
- Omnichannel engagement plans, next-best-action recommendations, and incrementality tests
- Causal inference models and marketing mix models for unified measurement
- A resource allocation optimizer and a governed AI decision engine

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

- `chapterN_walkthrough.ipynb`: the chapter as one executable story
- `exercise_solutions.ipynb`: worked answers with analyst judgment notes

Run the notebooks in order within each analytical chapter. Data generation is handled inside each walkthrough; no external data download is required. Chapters 1 and 2 are prose foundations and do not require companion notebooks.

---

## Book Structure

| Part | Focus | Decision capability |
| --- | --- | --- |
| Part 1 | Foundations | Frame a commercial problem and judge whether the data fit the decision |
| Part 2 | Market and customer understanding | Find patient opportunity, explain treatment behavior, and identify actionable customers |
| Part 3 | Engagement and measurement | Select channels, estimate incremental impact, and compare performance |
| Part 4 | Decision engines | Allocate resources and build governed recommendation systems |

The first half covers foundational methods: market sizing, patient journeys, HCP and account targeting, and competitive intelligence. The second half adds omnichannel measurement, real-world evidence, causal inference, machine learning for treatment-effect estimation, and fine-tuned language models for AI decision support. Every method, including the advanced ones, is demonstrated with working Python and verified output.

### Chapters

| Chapter | Title | Status | Walkthrough | Exercises |
| --- | --- | --- | --- | --- |
| 1 | [A Medicine, a Market, and the Decisions Between Them](ch01_intro/ch01_introduction.md) | Draft completed | | |
| 2 | [The Commercialization Operating System](ch02_ecosystem/ch02_ecosystem.md) | Draft completed | | |
| 3 | [A Synthetic Lab for Real Pharma Questions](ch03_data/ch03_data.md) | Draft completed | [Walkthrough](ch03_data/chapter3_walkthrough.md) | [Exercises](ch03_data/exercise_solutions.md) |
| 4 | [Market Sizing and Patient Populations](ch04_market/ch04_market_sizing.md) | Draft completed | [Walkthrough](ch04_market/chapter4_walkthrough.md) | [Exercises](ch04_market/exercise_solutions.md) |
| 5 | [Building the Patient Journey](ch05_journey/ch05_patient_journey.md) | Draft completed | [Walkthrough](ch05_journey/chapter5_walkthrough.md) | [Exercises](ch05_journey/exercise_solutions.md) |
| 6 | [HCP and Account Targeting](ch06_hcp/ch06_hcp_account_targeting.md) | Draft completed | [Walkthrough](ch06_hcp/chapter6_walkthrough.md) | [Exercises](ch06_hcp/exercise_solutions.md) |
| 7 | [Competitive Intelligence and Market Access](ch07_competitive/ch07_competitive_intelligence_market_access.md) | Draft completed | [Walkthrough](ch07_competitive/chapter7_walkthrough.md) | [Exercises](ch07_competitive/exercise_solutions.md) |
| 8 | [Omnichannel Analytics: Planning and Attribution](ch08_omnichannel/ch08_omnichannel_analytics) | Draft completed | [Walkthrough](ch08_omnichannel/chapter8_walkthrough) | [Exercises](ch08_omnichannel/exercise_solutions) |
| 9 | Next Best Action | In progress | | |
| 10 | Experimental Design and Incrementality | In progress | | |
| 11 | Real-World Evidence and Causal Inference | In progress | | |
| 12 | MMM and Unified Measurement | In progress | | |
| 13 | Resource Allocation and Optimization | In progress | | |
| 14 | AI Decision Intelligence | In progress | | |
| 15 | Capstone Case Studies | In progress | | |


---

*Fictional products, patients, HCPs, accounts, payers, and clinical events are used throughout. No real patient data appears in this repository.*
