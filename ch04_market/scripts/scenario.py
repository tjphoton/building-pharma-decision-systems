"""Monte Carlo scenario analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scenario_grid(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    access_multipliers: tuple[float, ...] = (0.85, 1.00, 1.15),
    conversion_rates: tuple[float, ...] = (0.15, 0.25, 0.35),
) -> pd.DataFrame:
    """Evaluate access and conversion assumptions together.

    Phenotype choice is handled separately as a structural comparison with
    recalibrated weights (see run_analysis), because switching phenotype
    without recalibrating to the same external anchor conflates scale and
    composition.
    """

    rows: list[dict] = []
    eligible = (
        patients[phenotype_column]
        & patients["age_eligible"]
        & patients["untreated_opportunity"]
    )
    base_access = (
        patients.loc[eligible, "population_weight"]
        * patients.loc[eligible, "access_probability"]
    ).sum()
    for access_multiplier in access_multipliers:
        reachable = base_access * access_multiplier
        for conversion_rate in conversion_rates:
            rows.append(
                {
                    "access_multiplier": access_multiplier,
                    "conversion_rate": conversion_rate,
                    "reachable_opportunity": reachable,
                    "expected_starts": reachable * conversion_rate,
                }
            )
    return pd.DataFrame(rows)


def correlated_mc_scenario(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    n_draws: int = 2_000,
    seed: int = 20260610,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlated Monte Carlo draws for uncertainty over (sensitivity, access, conversion).

    Uses a Cholesky decomposition to impose positive correlations between the
    three parameters (high phenotype sensitivity -> higher access -> higher conversion).
    Returns a DataFrame of draws with joint uncertainty statistics.
    """
    rng = np.random.default_rng(seed)
    # Correlation matrix for (phenotype_sensitivity, access_probability, conversion_rate)
    corr = np.array([[1.0, 0.6, 0.4],
                     [0.6, 1.0, 0.5],
                     [0.4, 0.5, 1.0]])
    L = np.linalg.cholesky(corr)

    # Parameter marginal distributions (mean, sd, lo, hi)
    params = {
        "phenotype_sensitivity": (0.80, 0.08, 0.55, 1.0),
        "access_multiplier": (1.0, 0.10, 0.70, 1.30),
        "conversion_rate": (0.25, 0.06, 0.10, 0.45),
    }
    means = np.array([v[0] for v in params.values()])
    sds = np.array([v[1] for v in params.values()])
    los = np.array([v[2] for v in params.values()])
    his = np.array([v[3] for v in params.values()])

    # Draw correlated standard normals, then transform to marginals
    z = rng.standard_normal((n_draws, 3))
    z_corr = z @ L.T
    draws = np.clip(z_corr * sds + means, los, his)

    # Base opportunity
    eligible = patients[phenotype_column] & patients["age_eligible"] & patients["untreated_opportunity"]
    base_access_opportunity = float(
        (patients.loc[eligible, "population_weight"] * patients.loc[eligible, "access_probability"]).sum()
    )

    expected_starts = base_access_opportunity * draws[:, 0] * draws[:, 1] * draws[:, 2]

    result = pd.DataFrame({
        "phenotype_sensitivity": draws[:, 0].round(4),
        "access_multiplier": draws[:, 1].round(4),
        "conversion_rate": draws[:, 2].round(4),
        "expected_starts": expected_starts.round(2),
    })
    summary = pd.DataFrame([{
        "statistic": "Mean",
        "expected_starts": round(float(expected_starts.mean()), 1),
    }, {
        "statistic": "Median",
        "expected_starts": round(float(np.median(expected_starts)), 1),
    }, {
        "statistic": "P10",
        "expected_starts": round(float(np.percentile(expected_starts, 10)), 1),
    }, {
        "statistic": "P90",
        "expected_starts": round(float(np.percentile(expected_starts, 90)), 1),
    }, {
        "statistic": "SD",
        "expected_starts": round(float(expected_starts.std()), 1),
    }])
    return result, summary
