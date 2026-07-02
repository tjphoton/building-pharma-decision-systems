# Building Decision Systems: A Hands-On Playbook for Pharmaceutical Commercial Decision Science

**A practical guide to pharmaceutical commercial analytics, from raw data to defensible action.**

Book link: https://tjphoton.github.io/building-pharma-decision-systems/

Pharmaceutical companies have no shortage of data. The harder problem is deciding what to do with it. A forecast may size a market, but a launch team still has to decide where to focus. A targeting model may rank physicians, but a field leader still has to decide which accounts deserve attention and why. A campaign report may show a lift in prescriptions, but a brand team still has to judge whether the campaign caused the change.

No single book covers the full path: from commercial data infrastructure through patient journeys, targeting, competitive intelligence, omnichannel measurement, causal inference, and AI-supported decision engines. This is the book I wish I had before I learned these topics the hard way.

It follows the complete fictional launch of **Roventra**, a once-daily oral medicine for a chronic condition, from FDA approval to field execution.

---

## What You Will Build

- A linked synthetic pharmaceutical dataset that mirrors real claims, EHR, payer, CRM, and specialty pharmacy structures
- Patient population funnels from true prevalence down to eligible, treated patients
- Lines of therapy with explicit washout, switch, addition, and persistence rules
- HCP targeting with transparent scoring, actionability filters, and field capacity allocation
- Competitive intelligence: formulary position, payer access barriers, and corrected treatment share
- Omnichannel engagement plans, next-best-action recommendations, and incrementality tests
- Causal inference models and marketing mix models for unified measurement
- A resource allocation optimizer and a governed AI decision engine

---

## Who This Is For

**Commercial analysts and data scientists** in pharmaceutical companies who need to move from reporting to decision support. The book teaches the judgment layer: when a method is adequate, when it is not, and how to present a finding as a recommendation with a clear owner and measure of success.

**Analytics engineers and BI teams** building commercial data platforms. The chapter on data infrastructure covers claim receipt lag, drug code mapping gaps, and data quality rules that protect downstream analysis.

**Brand teams and commercial leaders** who review analytical recommendations and want to understand what the data can and cannot tell you about market opportunity, customer behavior, and campaign impact.

**Students and independent consultants** entering pharmaceutical commercial analytics. The book assumes no prior pharma experience and defines terms at first use.

### Prerequisites

- Python 3.11 or later
- Familiarity with pandas and basic data manipulation
- Basic understanding of statistical concepts (means, distributions, regression)

No pharmaceutical industry background is required. No prior exposure to pharmaceutical data sources, commercial roles, or marketing experience is assumed.

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

## Book Structure

| Part | Focus | Decision capability |
| --- | --- | --- |
| Part 1 | Foundations | Frame a commercial problem and judge whether the data fit the decision |
| Part 2 | Market and customer understanding | Find patient opportunity, explain treatment behavior, and identify actionable customers |
| Part 3 | Engagement and measurement | Select channels, estimate incremental impact, and compare performance |
| Part 4 | Decision engines | Allocate resources and build governed recommendation systems |

The first half covers foundational methods: market sizing, patient journeys, HCP targeting, and competitive intelligence. The second half adds omnichannel measurement, real-world evidence, causal inference, machine learning for treatment-effect estimation, and fine-tuned language models for AI decision support. Every method, including the advanced ones, is demonstrated with working Python and verified output.

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
