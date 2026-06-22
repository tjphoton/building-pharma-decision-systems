"""Population weighting and NHANES-anchored calibration functions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# NCHS Data Brief 516 reports crude diagnosed-diabetes prevalence of 11.3%
# among U.S. adults age 20 and older in August 2021-August 2023.
NATIONAL_DIAGNOSED_PREVALENCE = 0.113

# U.S. Census 2024 resident population age 20 and older, calculated from
# NC-EST2025-AGESEX-RES single-year age estimates.
US_ADULT_20_PLUS_POPULATION = 258_554_106

# Census regional populations are used only to allocate the national target.
# They do not imply region-specific prevalence.
US_REGION_POPULATION = {
    "Northeast": 57_609_148,
    "South": 126_018_935,
    "Midwest": 68_985_454,
    "West": 78_588_572,
}


def diagnosed_population_targets() -> dict[str, int]:
    """Allocate the national diagnosed target across regions by population share."""

    national_target = round(
        NATIONAL_DIAGNOSED_PREVALENCE * US_ADULT_20_PLUS_POPULATION
    )
    region_total = sum(US_REGION_POPULATION.values())
    targets = {
        region: round(national_target * population / region_total)
        for region, population in US_REGION_POPULATION.items()
    }
    rounding_difference = national_target - sum(targets.values())
    targets["South"] += rounding_difference
    return targets


def calibrate_diagnosed_weights(
    patients: pd.DataFrame,
    targets: Mapping[str, int | float],
    phenotype_column: str = "base_phenotype",
) -> pd.DataFrame:
    """Weight phenotype-positive patients so each region sums to its diagnosed target.

    Patients outside the phenotype receive weight 0: they do not represent
    anyone in the diagnosed universe the funnel estimates.
    """

    result = patients.copy()
    mask = result[phenotype_column].astype(bool)
    observed = result.loc[mask].groupby("region").size()
    missing_regions = set(targets) - set(observed.index)
    if missing_regions:
        raise ValueError(
            f"No phenotype-positive sample records for regions: {sorted(missing_regions)}"
        )
    region_weights = {
        region: float(targets[region]) / float(observed.loc[region])
        for region in targets
    }
    result["population_weight"] = 0.0
    result.loc[mask, "population_weight"] = result.loc[mask, "region"].map(region_weights)
    if result.loc[mask, "population_weight"].isna().any():
        unmatched = sorted(
            result.loc[mask & result["population_weight"].isna(), "region"].unique()
        )
        raise ValueError(f"No target supplied for regions: {unmatched}")
    return result


def nhanes_calibration(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate diagnosed patients to the national prevalence anchor.

    Returns the calibrated patient table and a bridge table that traces each
    regional weight back to the national prevalence and regional allocation.
    """

    targets = diagnosed_population_targets()
    calibrated = calibrate_diagnosed_weights(patients, targets, phenotype_column)
    mask = patients[phenotype_column].astype(bool)
    observed = patients.loc[mask].groupby("region").size()

    bridge = pd.DataFrame(
        [
            {
                "region": region,
                "national_diagnosed_prevalence": NATIONAL_DIAGNOSED_PREVALENCE,
                "us_adult_20_plus_population": US_ADULT_20_PLUS_POPULATION,
                "regional_allocation_share": (
                    US_REGION_POPULATION[region] / sum(US_REGION_POPULATION.values())
                ),
                "target_diagnosed_population": targets[region],
                "observed_diagnosed_patients": int(observed.loc[region]),
                "population_weight": round(targets[region] / observed.loc[region], 1),
            }
            for region in targets
        ]
    )
    return calibrated, bridge


def bootstrap_access_opportunity(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    n_boot: int = 1_000,
    seed: int = 20260610,
) -> np.ndarray:
    """Bootstrap within region and recalculate the calibrated access opportunity.

    Each replicate resamples patients with replacement inside each region,
    recalibrates the diagnosed weights to the same NHANES-anchored targets, and
    recomputes the access-adjusted reachable opportunity. The percentile spread
    of the replicates reflects sampling uncertainty only.
    """

    targets = diagnosed_population_targets()
    rng = np.random.default_rng(seed)
    regions = {
        region: group.reset_index(drop=True)
        for region, group in patients.groupby("region", sort=True)
    }
    estimates = np.empty(n_boot)

    for boot_index in range(n_boot):
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
        estimates[boot_index] = (
            sample.loc[eligible, "population_weight"]
            * sample.loc[eligible, "access_probability"]
        ).sum()
    return estimates
