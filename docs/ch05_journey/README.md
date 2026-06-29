# Chapter 5: Building the Patient Journey: Treatment Patterns, Lines of Therapy, and Persistence

The chapter manuscript is [`ch05_patient_journey.md`](ch05_patient_journey.md). It turns the
Chapter 3 synthetic package into a journey analysis: a newly observed diagnosis
cohort, a transaction ledger, treatment episodes, lines of therapy built from
nine explicit sequencing rules (index, washout, regimen window, allowable gap,
addition, switch, restart, discontinuation, censoring), initiation and
persistence curves, adherence metrics, and a sensitivity grid that shows which
findings the rules create.

## Current Scenario Inputs

The launch product is `Roventra`, with Nexoral and Vexpro as competitors. Chapter 5 reads the regenerated Chapter 3 package directly, derives product names from `ndc_prescribed`, and builds the launch-condition cohort from the mature medical claims diagnosis columns.

Two executed companion notebooks sit next to the manuscript:

- [`ch05_walkthrough.ipynb`](ch05_walkthrough.ipynb): the chapter as one executable story.
- [`ch05_exercise_solutions.ipynb`](ch05_exercise_solutions.ipynb): worked answers with discussion.

Regenerate everything from the repository root:

```bash
uv run python ch05_journey/scripts/run_analysis.py
uv run python ch05_journey/scripts/build_figures.py
```

See [`scripts/README.md`](scripts/README.md) for the module layout and the
scenario constants behind the sequencing rules.
