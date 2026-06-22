# Chapter 6: HCP and Account Targeting: From Opportunity to Field Action

Welcome to Chapter 6! This chapter turns patient opportunity data into field-actionable targeting decisions.

## What You'll Learn

In this chapter, you'll discover:
- **HCP and account evidence** - Join patient journeys, CRM records, and field roster data to build targeting features
- **Account actions** - Apply rules to determine which accounts need increase priority, maintain, monitor, access review, or contact hold
- **HCP filtering and ordering** - Build a ranked HCP list within each account with transparency and traceable rules
- **Permission and capacity gates** - Filter accounts by CRM consent status and field team capacity
- **Call planning** - Allocate field visits in proportion to actionable opportunity
- **Targeting transparency** - Enable stakeholders to trace every recommendation back to its underlying evidence

## Read the Full Chapter

👉 **[Start reading Chapter 6: HCP and Account Targeting](ch06_hcp_account_targeting.md)**

Also available:
- 📓 **[Walkthrough Notebook](chapter6_walkthrough.ipynb)** - Interactive Python notebook with step-by-step code examples
- 🧪 **[Exercise Solutions](exercise_solutions.ipynb)** - Solutions to chapter exercises

This chapter teaches you to move from "Which HCPs should we target?" to "Here's the evidence, here's the rule, and here's the bounded call plan."

## Running the Analysis

Generate outputs from the repository root:

```bash
uv run python ch06_hcp/scripts/run_analysis.py
MPLCONFIGDIR=/tmp/matplotlib uv run python ch06_hcp/scripts/build_figures.py
uv run python ch06_hcp/scripts/build_notebooks.py
uv run python ch06_hcp/scripts/verify_chapter_blocks.py
```
