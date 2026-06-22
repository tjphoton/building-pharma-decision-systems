"""Claims maturity adjustment and capture-recapture estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def claims_maturity_adjustment(
    patients: pd.DataFrame,
    lag_months: int = 4,
    phenotype_column: str = "base_phenotype",
) -> pd.DataFrame:
    """Demonstrate how claims maturity lag affects the untreated opportunity count.

    Simulates what the phenotype count would look like at lag_months before the
    analysis date (i.e., how many patients are 'missing' from the visible cohort
    because their claims haven't been processed yet). Assumes a linear claim arrival
    rate over the first 6 months post-service.
    """
    eligible_full = (
        patients[phenotype_column]
        & patients["age_eligible"]
        & patients["untreated_opportunity"]
    )
    full_count = int(eligible_full.sum())
    full_pop = float(patients.loc[eligible_full, "population_weight"].sum())

    # Lag adjustment: fraction of claims that arrive within lag_months
    # Using a simple linear ramp: 100% by month 6, proportional before
    max_lag_months = 6
    arrived_fraction = min(lag_months / max_lag_months, 1.0)

    lag_count = int(full_count * arrived_fraction)
    lag_pop = full_pop * arrived_fraction
    missing_count = full_count - lag_count
    missing_pop = full_pop - lag_pop

    return pd.DataFrame([{
        "scenario": "Full matured claims (no lag)",
        "lag_months": 0,
        "eligible_patient_count": full_count,
        "calibrated_population": round(full_pop, 0),
        "pct_of_full": 1.0,
    }, {
        "scenario": f"Claims at {lag_months}-month maturity",
        "lag_months": lag_months,
        "eligible_patient_count": lag_count,
        "calibrated_population": round(lag_pop, 0),
        "pct_of_full": round(arrived_fraction, 4),
    }, {
        "scenario": "Missing (claims not yet received)",
        "lag_months": lag_months,
        "eligible_patient_count": missing_count,
        "calibrated_population": round(missing_pop, 0),
        "pct_of_full": round(1.0 - arrived_fraction, 4),
    }])


def capture_recapture(
    source_a_count: int,
    source_b_count: int,
    overlap_count: int,
) -> dict[str, float]:
    """Chapman estimator for the true population size from two independent sources.

    N_hat = ((n1+1)*(n2+1) / (m+1)) - 1
    where n1 = source A captures, n2 = source B captures, m = overlap.
    Returns the bias-corrected Chapman estimate and a 95% confidence interval.
    """
    n1, n2, m = source_a_count, source_b_count, overlap_count
    n_hat = ((n1 + 1) * (n2 + 1)) / (m + 1) - 1
    variance = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)) / ((m + 1) ** 2 * (m + 2))
    se = float(np.sqrt(variance)) if variance > 0 else 0.0
    z = 1.96
    return {
        "source_a_count": n1,
        "source_b_count": n2,
        "overlap_count": m,
        "chapman_estimate": round(n_hat, 1),
        "standard_error": round(se, 1),
        "ci_95_low": round(n_hat - z * se, 1),
        "ci_95_high": round(n_hat + z * se, 1),
        "naive_union_estimate": n1 + n2 - m,
        "capture_efficiency_a": round(n1 / n_hat, 4) if n_hat > 0 else np.nan,
        "capture_efficiency_b": round(n2 / n_hat, 4) if n_hat > 0 else np.nan,
    }
