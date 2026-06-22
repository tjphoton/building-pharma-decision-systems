"""Geographic aggregation and visualization functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from calibration import calibrate_diagnosed_weights, diagnosed_population_targets


def account_opportunity(
    patients: pd.DataFrame,
    accounts: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    conversion_rate: float = 0.25,
) -> pd.DataFrame:
    """Aggregate the calibrated opportunity to the account action grain."""

    eligible = (
        patients[phenotype_column]
        & patients["age_eligible"]
        & patients["untreated_opportunity"]
    )
    patient_rows = patients.loc[eligible].copy()
    patient_rows["reachable_opportunity"] = (
        patient_rows["population_weight"] * patient_rows["access_probability"]
    )

    summary = (
        patient_rows.groupby("account_id", as_index=False)
        .agg(
            observed_patients=("patient_id", "nunique"),
            weighted_untreated_population=("population_weight", "sum"),
            reachable_opportunity=("reachable_opportunity", "sum"),
            mean_access_probability=("access_probability", "mean"),
        )
        .merge(
            accounts[
                [
                    "account_id",
                    "account_name",
                    "account_type",
                    "region",
                    "territory",
                    "capacity",
                ]
            ],
            on="account_id",
            how="left",
            validate="one_to_one",
        )
    )
    summary["expected_starts"] = summary["reachable_opportunity"] * conversion_rate
    return summary.sort_values(
        ["expected_starts", "reachable_opportunity"],
        ascending=False,
    ).reset_index(drop=True)


def account_rank_stability(
    patients: pd.DataFrame,
    accounts: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    n_boot: int = 200,
    n_top: int = 5,
    seed: int = 20260610,
) -> pd.DataFrame:
    """Measure how stable the top account ranks are under bootstrap resampling.

    For each replicate, resample patients within region, recalibrate the
    diagnosed weights, recompute each account's reachable opportunity, and
    record the rank of the accounts that lead the point-estimate table.
    """

    targets = diagnosed_population_targets()
    point = account_opportunity(patients, accounts, phenotype_column)
    top_ids = point["account_id"].head(n_top).tolist()

    rng = np.random.default_rng(seed)
    regions = {
        region: group.reset_index(drop=True)
        for region, group in patients.groupby("region", sort=True)
    }
    ranks: dict[str, list[int]] = {account_id: [] for account_id in top_ids}

    for _ in range(n_boot):
        sampled_groups = []
        for group in regions.values():
            positions = rng.integers(0, len(group), size=len(group))
            sampled_groups.append(group.iloc[positions].copy())
        sample = pd.concat(sampled_groups, ignore_index=True)
        sample = calibrate_diagnosed_weights(sample, targets, phenotype_column)
        eligible = (
            sample[phenotype_column]
            & sample["age_eligible"]
            & sample["untreated_opportunity"]
        )
        rows = sample.loc[eligible]
        reachable = (
            (rows["population_weight"] * rows["access_probability"])
            .groupby(rows["account_id"])
            .sum()
            .rank(ascending=False, method="min")
        )
        for account_id in top_ids:
            ranks[account_id].append(int(reachable.get(account_id, len(reachable) + 1)))

    summary = pd.DataFrame(
        [
            {
                "account_id": account_id,
                "point_rank": rank_index + 1,
                "median_bootstrap_rank": float(np.median(ranks[account_id])),
                "rank_p5": float(np.percentile(ranks[account_id], 5)),
                "rank_p95": float(np.percentile(ranks[account_id], 95)),
                "share_of_replicates_in_top5": round(
                    float(np.mean(np.array(ranks[account_id]) <= n_top)), 3
                ),
            }
            for rank_index, account_id in enumerate(top_ids)
        ]
    )
    return summary.merge(
        accounts[["account_id", "account_name", "region"]], on="account_id", how="left"
    )


def state_opportunity(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    conversion_rate: float = 0.25,
) -> pd.DataFrame:
    """Aggregate the calibrated opportunity to the patient's actual state."""

    eligible = (
        patients[phenotype_column]
        & patients["age_eligible"]
        & patients["untreated_opportunity"]
    )
    rows = patients.loc[eligible].copy()
    rows["reachable_opportunity"] = (
        rows["population_weight"] * rows["access_probability"]
    )
    summary = rows.groupby("state", as_index=False).agg(
        observed_patients=("patient_id", "nunique"),
        reachable_opportunity=("reachable_opportunity", "sum"),
    )
    summary["expected_starts"] = summary["reachable_opportunity"] * conversion_rate
    return summary.sort_values("reachable_opportunity", ascending=False).reset_index(
        drop=True
    )


def opportunity_choropleth(
    state_opportunity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save an interactive choropleth map of state-level reachable opportunity.

    Expects the output of state_opportunity(), which aggregates on each
    patient's actual state rather than distributing regional totals.
    """
    state_df = state_opportunity_df
    if state_df.empty:
        return
    fig = px.choropleth(
        state_df,
        locations="state",
        locationmode="USA-states",
        color="reachable_opportunity",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"reachable_opportunity": "Reachable opportunity"},
        title="Reachable Opportunity by State",
        hover_data={"expected_starts": True, "observed_patients": True},
    )
    fig.update_layout(height=450)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    # The static map for the chapter is produced by build_figures.py as
    # figure_4_10_state_opportunity_map; this function writes only the
    # interactive companion HTML.
