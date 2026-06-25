"""Entry point for the Chapter 5 treatment pattern and patient journey analysis."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

from adherence import adherence_summary, compute_adherence_metrics  # noqa: E402
from episode_construction import (  # noqa: E402
    build_newly_observed_cohort,
    construct_treatment_episodes,
    load_chapter3_data,
    prepare_pharmacy_events,
    summarize_patient_journeys,
)
from lot import (  # noqa: E402
    construct_lines_of_therapy,
    lot_entry_shares,
    lot_sensitivity,
)
from survival import (  # noqa: E402
    km_estimate_at,
    km_median,
    line_persistence_curve,
    treatment_initiation_curve,
)

# Cohort design constants. The 180-day diagnostic lookback exists so the
# 180-day therapy washout in lot.py is observable for every cohort patient.
MINIMUM_LOOKBACK_DAYS = 180
MINIMUM_FOLLOWUP_DAYS = 90
PERMISSIBLE_GAP_DAYS = 30


def run_analysis(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Execute the Chapter 5 analysis and return its core artifacts."""

    tables = load_chapter3_data(data_dir)
    cohort, attrition = build_newly_observed_cohort(
        tables,
        minimum_lookback_days=MINIMUM_LOOKBACK_DAYS,
        minimum_followup_days=MINIMUM_FOLLOWUP_DAYS,
    )
    initiation_cohort, initiation_attrition = build_newly_observed_cohort(
        tables,
        minimum_lookback_days=MINIMUM_LOOKBACK_DAYS,
        minimum_followup_days=0,
    )
    analysis_cohorts = pd.DataFrame(
        [
            {
                "analysis": "Treatment initiation",
                "patients": initiation_cohort["patient_id"].nunique(),
                "required_lookback_days": MINIMUM_LOOKBACK_DAYS,
                "required_followup_days": 0,
                "purpose": "Time from diagnosis index to first treatment",
            },
            {
                "analysis": "Journey and line construction",
                "patients": cohort["patient_id"].nunique(),
                "required_lookback_days": MINIMUM_LOOKBACK_DAYS,
                "required_followup_days": MINIMUM_FOLLOWUP_DAYS,
                "purpose": "Post-index treatment patterns with stable follow-up",
            },
        ]
    )

    # All patients who have a qualifying index date (no lookback/followup filter).
    # Used by build_figures.py to select 3 example patients for Figure 5.1.
    all_indexed, _ = build_newly_observed_cohort(
        tables, minimum_lookback_days=0, minimum_followup_days=0
    )
    all_indexed_export = all_indexed[
        [
            "patient_id",
            "coverage_start",
            "coverage_end",
            "index_date",
            "followup_end",
            "lookback_days",
            "followup_days",
        ]
    ].copy()

    # Lookback sensitivity for Figure 5.1: hold 90-day follow-up fixed, vary lookback.
    lookback_rows = []
    for lb in [0, 30, 60, 90, 120, 150, 180, 270, 365]:
        coh, _ = build_newly_observed_cohort(
            tables, minimum_lookback_days=lb, minimum_followup_days=90
        )
        lookback_rows.append({"lookback_days": lb, "patients": len(coh)})
    lookback_sensitivity = pd.DataFrame(lookback_rows)

    basket = tables["products"]["product_name"].tolist()
    paid_events, nonpaid_events = prepare_pharmacy_events(
        tables["pharmacy_claims"], cohort, basket
    )
    initiation_paid, initiation_nonpaid = prepare_pharmacy_events(
        tables["pharmacy_claims"],
        initiation_cohort,
        basket,
    )

    episodes = construct_treatment_episodes(
        paid_events, cohort, permissible_gap_days=PERMISSIBLE_GAP_DAYS
    )
    journeys = summarize_patient_journeys(cohort, episodes, nonpaid_events)
    initiation_episodes = construct_treatment_episodes(
        initiation_paid,
        initiation_cohort,
        permissible_gap_days=PERMISSIBLE_GAP_DAYS,
    )
    initiation_journeys = summarize_patient_journeys(
        initiation_cohort,
        initiation_episodes,
        initiation_nonpaid,
    )
    initiation = treatment_initiation_curve(initiation_journeys)

    # Lines of therapy need every paid basket fill, including pre-index
    # fills, so the washout rule can see prior exposure.
    pharmacy = tables["pharmacy_claims"]
    paid_all = pharmacy.loc[
        pharmacy["transaction_type"].eq("PAID")
        & pharmacy["product_name"].isin(basket)
        & pharmacy["patient_id"].isin(cohort["patient_id"])
    ].copy()
    lines, initiators = construct_lines_of_therapy(paid_all, cohort)
    lines_naive, _ = construct_lines_of_therapy(paid_all, cohort, washout_days=0)

    line1 = lines.loc[lines["line_number"].eq(1)]
    lot_summary = (
        line1.groupby("regimen", as_index=False)
        .agg(
            patients=("patient_id", "nunique"),
            median_line_days=("line_days", "median"),
            discontinued_share=("end_reason", lambda s: round(s.eq("Discontinued").mean(), 3)),
        )
        .sort_values("patients", ascending=False)
        .reset_index(drop=True)
    )

    persistence = line_persistence_curve(lines)
    sensitivity = lot_sensitivity(paid_all, cohort)

    index_product_by_patient = (
        journeys.loc[journeys["initiated_treatment"], ["patient_id", "first_product"]]
        .dropna()
        .set_index("patient_id")["first_product"]
        .to_dict()
    )
    adherence_index_product = compute_adherence_metrics(
        paid_events,
        cohort,
        observation_days=365,
        product_by_patient=index_product_by_patient,
        scope_label="Index product",
    )
    adherence_market_basket = compute_adherence_metrics(
        paid_events,
        cohort,
        observation_days=365,
        scope_label="Any market product",
    )
    adherence_by_payer = adherence_summary(
        adherence_index_product,
        cohort,
        by_columns=["payer_id"],
    )

    return {
        "cohort": cohort,
        "attrition": attrition,
        "analysis_cohorts": analysis_cohorts,
        "initiation_cohort": initiation_cohort,
        "initiation_attrition": initiation_attrition,
        "paid_events": paid_events,
        "nonpaid_events": nonpaid_events,
        "episodes": episodes,
        "journeys": journeys,
        "initiation_journeys": initiation_journeys,
        "initiation_curve": initiation,
        "initiators": initiators,
        "lines": lines,
        "lines_naive": lines_naive,
        "lot_line1_summary": lot_summary,
        "lot_entry_shares": lot_entry_shares(lines),
        "lot_entry_shares_naive": lot_entry_shares(lines_naive),
        "lot_sensitivity": sensitivity,
        "line1_persistence": persistence,
        "adherence": adherence_index_product,
        "adherence_index_product": adherence_index_product,
        "adherence_market_basket": adherence_market_basket,
        "adherence_by_payer": adherence_by_payer,
        "all_indexed": all_indexed_export,
        "lookback_sensitivity": lookback_sensitivity,
    }


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write reusable Chapter 5 outputs to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_tables = [
        "attrition",
        "analysis_cohorts",
        "initiation_attrition",
        "initiation_journeys",
        "journeys",
        "episodes",
        "initiation_curve",
        "initiators",
        "lines",
        "lot_line1_summary",
        "lot_entry_shares",
        "lot_entry_shares_naive",
        "lot_sensitivity",
        "line1_persistence",
        "adherence",
        "adherence_index_product",
        "adherence_market_basket",
        "adherence_by_payer",
        "all_indexed",
        "lookback_sensitivity",
    ]
    for name in csv_tables:
        if name in results and not results[name].empty:
            results[name].to_csv(output_dir / f"{name}.csv", index=False)


def print_summary(results: Mapping[str, pd.DataFrame]) -> None:
    """Print the headline numbers in the format quoted by the chapter."""

    attrition = results["attrition"]
    print("Cohort attrition:")
    print(attrition.to_string(index=False))

    initiators = results["initiators"]
    journeys = results["journeys"]
    initiation_journeys = results["initiation_journeys"]
    print(
        f"\nInitiation cohort patients: {len(initiation_journeys):,}"
        " (180-day lookback, no required future 90-day window)"
    )
    print(
        f"Journey cohort treated patients: {int(journeys['initiated_treatment'].sum()):,}"
        f" of {len(journeys):,}"
        f" | new to therapy after washout: {int(initiators['new_to_therapy'].sum()):,}"
        f" | prevalent users excluded: {int((~initiators['new_to_therapy']).sum()):,}"
    )

    lines = results["lines"]
    print(
        f"Lines of therapy: {len(lines):,} lines across"
        f" {lines['patient_id'].nunique():,} patients"
        f" | line distribution: "
        + ", ".join(
            f"L{int(line)}={count:,}"
            for line, count in lines["line_number"].value_counts().sort_index().items()
        )
    )

    entries = results["lot_entry_shares"]
    naive = results["lot_entry_shares_naive"]
    line1_true = int(entries.loc[entries["position"].eq("Line 1"), "line_entries"].sum())
    line1_naive = int(naive.loc[naive["position"].eq("Line 1"), "line_entries"].sum())
    print(
        f"Roventra line-1 entries: {line1_naive:,} without washout,"
        f" {line1_true:,} with the 180-day washout"
        f" ({line1_naive - line1_true:,} prevalent continuations relabeled)"
    )

    persistence = results["line1_persistence"]
    print(
        f"Line-1 persistence: median {km_median(persistence):.0f} days"
        f" | still on line 1 at 180 days: {km_estimate_at(persistence, 180):.1%}"
    )


if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    source_dir = chapter_dir.parent / "ch03_data" / "output_data" / "generated_data"
    output_dir = chapter_dir / "assets" / "generated_outputs"
    analysis = run_analysis(source_dir)
    write_outputs(analysis, output_dir)
    print_summary(analysis)
    print(f"\nWrote Chapter 5 outputs to {output_dir}")
