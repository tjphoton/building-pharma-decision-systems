"""Estimand-building functions: patient funnel from raw tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

# Analysis date: all phenotype, eligibility, and access assessments are as of this date.
ANALYSIS_DATE = pd.Timestamp("2024-12-31")

# ICD-10 codes the Chapter 3 generator assigns to the launch condition.
LAUNCH_CONDITION_CODES = ["E11.9", "E11.65", "E11.40"]

LAUNCH_PRODUCT = "Roventra"

ACCESS_PROBABILITY = {
    "Covered": 0.90,
    "Covered with PA": 0.65,
    "Non-covered": 0.10,
}

# Wide diagnosis column names in medical claims
DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]


def load_chapter3_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the Chapter 3 tables used by the market-sizing example."""
    return {
        "patients": pd.read_csv(data_dir / "reference" / "patients.csv"),
        "patient_enrollments": pd.read_csv(data_dir / "reference" / "patient_enrollments.csv"),
        "accounts": pd.read_csv(data_dir / "reference" / "accounts.csv"),
        "providers": pd.read_csv(data_dir / "reference" / "providers.csv"),
        "hcp_targets": pd.read_csv(data_dir / "reference" / "hcp_targets.csv"),
        "ndc_codes": pd.read_csv(data_dir / "reference" / "ndc_codes.csv", dtype={"ndc": str}),
        # Use the mature snapshot for all downstream analysis
        "medical_claims": pd.read_csv(data_dir / "claims_medical" / "medical_claims_mature.csv"),
        "pharmacy_claims": pd.read_csv(
            data_dir / "claims_pharmacy" / "pharmacy_claims.csv",
            dtype={"ndc": str, "ndc_prescribed": str},
        ),
        "lab_results": pd.read_csv(
            data_dir / "claims_lab" / "lab_results.csv",
            parse_dates=["service_date"],
        ),
        "access": pd.read_csv(data_dir / "market_access" / "market_access_rules.csv"),
    }


def drug_name_from_ndc(
    pharmacy: pd.DataFrame, ndc_codes: pd.DataFrame
) -> pd.Series:
    """Derive the drug name by joining on ndc_prescribed.

    Joining on ndc_prescribed (what the prescriber wrote) gives the most stable
    product attribution. About 5% of rows have a dispensed NDC variant that the
    reference does not map; those resolve to NaN but the prescribed code still maps.
    """
    ndc_map = ndc_codes.set_index("ndc")["drug_name"]
    return pharmacy["ndc_prescribed"].map(ndc_map)


def _has_launch_dx(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask: True if any diagnosis column contains a launch-condition code."""
    return df[DX_COLS].isin(LAUNCH_CONDITION_CODES).any(axis=1)


def _patient_account_map(tables: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Map each patient to the account for the provider seen most often."""

    hcp_to_account = tables["hcp_targets"].set_index("npi")["account_id"]
    encounters = tables["medical_claims"][["patient_id", "rendering_npi"]].rename(
        columns={"rendering_npi": "npi"}
    )
    fills = tables["pharmacy_claims"][["patient_id", "prescriber_npi"]].rename(
        columns={"prescriber_npi": "npi"}
    )
    provider_events = pd.concat([encounters, fills], ignore_index=True).dropna()
    provider_events["account_id"] = provider_events["npi"].map(hcp_to_account)
    provider_events = provider_events.dropna(subset=["account_id"])
    if provider_events.empty:
        return pd.Series(dtype="object")
    counts = (
        provider_events.groupby(["patient_id", "account_id"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["patient_id", "n_events", "account_id"], ascending=[True, False, True])
    )
    return counts.drop_duplicates("patient_id").set_index("patient_id")["account_id"]


def build_patient_analysis(
    tables: Mapping[str, pd.DataFrame],
    answer_key: pd.DataFrame | None = None,
    product_name: str = LAUNCH_PRODUCT,
) -> pd.DataFrame:
    """Create one patient-level record with phenotype, treatment, and access flags."""
    patients = tables["patients"].copy()
    # payer_id lives in patient_enrollments. Join it before access-rule merge.
    enroll_payer = (
        tables["patient_enrollments"][["patient_id", "payer_id"]]
        .drop_duplicates("patient_id")
    )
    patients = patients.merge(enroll_payer, on="patient_id", how="left")
    medical = tables["medical_claims"].copy()
    pharmacy = tables["pharmacy_claims"].copy()
    access = tables["access"].copy()

    # All rows in medical_claims_mature.csv are completed encounters (no status filter needed).
    launch_coded = _has_launch_dx(medical)
    launch_diagnoses_any = medical.loc[launch_coded]
    diagnosis_counts = launch_diagnoses_any.groupby("patient_id").size()

    pharmacy["drug_name"] = drug_name_from_ndc(pharmacy, tables["ndc_codes"])
    product_transactions = pharmacy.loc[pharmacy["drug_name"].eq(product_name)].copy()
    product_transactions["net_fill_change"] = product_transactions["transaction_type"].map(
        {"PAID": 1, "REVERSED": -1}
    ).fillna(0)
    net_product_fills = product_transactions.groupby("patient_id")["net_fill_change"].sum()

    if answer_key is not None:
        truth_map = answer_key.set_index("patient_id")["true_launch_condition"]
        patients["reference_condition"] = (
            patients["patient_id"].map(truth_map).fillna(False).astype(bool)
        )
    else:
        patients["reference_condition"] = (
            patients["true_launch_condition"].fillna(False).astype(bool)
        )
    patients["launch_diagnosis_coded"] = patients["patient_id"].isin(
        launch_diagnoses_any["patient_id"]
    )
    patients["diagnosis_claim_count"] = (
        patients["patient_id"].map(diagnosis_counts).fillna(0).astype(int)
    )
    patients["base_phenotype"] = patients["diagnosis_claim_count"].ge(1)
    patients["strict_phenotype"] = patients["diagnosis_claim_count"].ge(2)
    patients["age_eligible"] = patients["age_band"].isin(["35-49", "50-64", "65+"])
    patients["net_product_fills"] = (
        patients["patient_id"].map(net_product_fills).fillna(0).astype(int)
    )
    patients["current_product_user"] = patients["net_product_fills"].gt(0)
    patients["untreated_opportunity"] = ~patients["current_product_user"]

    access["effective_start"] = pd.to_datetime(access["effective_start"])
    access["effective_end"] = pd.to_datetime(access["effective_end"])
    access_on_date = access.loc[
        (access["product_name"].eq(product_name))
        & (access["effective_start"].le(ANALYSIS_DATE))
        & (access["effective_end"].ge(ANALYSIS_DATE))
    ]
    product_access = access_on_date[
        ["payer_id", "region", "coverage_status", "prior_authorization", "step_edit", "specialty_pharmacy_required"]
    ]
    patients = patients.merge(
        product_access,
        on=["payer_id", "region"],
        how="left",
        validate="many_to_one",
    )
    patients["access_probability"] = (
        patients["coverage_status"].map(ACCESS_PROBABILITY).fillna(0.0)
    )
    patients["account_id"] = patients["patient_id"].map(_patient_account_map(tables))
    return patients


def phenotype_diagnostics(
    patients: pd.DataFrame,
    phenotype_column: str,
) -> dict[str, float | int]:
    predicted = patients[phenotype_column].astype(bool)
    truth = patients["reference_condition"].astype(bool)
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return {
        "phenotype": phenotype_column,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": ppv,
    }


def panel_market_sizes(patients: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("True condition", "answer key", patients["reference_condition"]),
        ("Launch diagnosis coded", "any encounter", patients["launch_diagnosis_coded"]),
        ("Paid-claims phenotype", "D_i = 1", patients["base_phenotype"]),
        ("Age-eligible diagnosed", "D_i E_i = 1", patients["base_phenotype"] & patients["age_eligible"]),
        (
            "Untreated opportunity",
            "D_i E_i U_i = 1",
            patients["base_phenotype"] & patients["age_eligible"] & patients["untreated_opportunity"],
        ),
    ]
    return pd.DataFrame(
        [{"stage": stage, "rule": rule, "panel_count": int(mask.sum())} for stage, rule, mask in stages]
    )


def funnel_estimates(
    patients: pd.DataFrame,
    phenotype_column: str = "base_phenotype",
    conversion_rate: float = 0.25,
) -> pd.DataFrame:
    stages = [
        ("Diagnosed population (calibrated)", patients[phenotype_column]),
        ("Age-eligible (35 or older)", patients["age_eligible"]),
        ("Untreated opportunity", patients["untreated_opportunity"]),
    ]
    mask = pd.Series(True, index=patients.index)
    rows: list[dict] = []
    for stage, criterion in stages:
        mask &= criterion.astype(bool)
        rows.append(
            {
                "stage": stage,
                "sample_count": int(mask.sum()),
                "population_estimate": float(patients.loc[mask, "population_weight"].sum()),
                "estimate_type": "count",
            }
        )
    access_adjusted = float(
        (patients.loc[mask, "population_weight"] * patients.loc[mask, "access_probability"]).sum()
    )
    rows.append({"stage": "Access-adjusted reachable opportunity", "sample_count": None, "population_estimate": access_adjusted, "estimate_type": "expected count"})
    rows.append({"stage": "Expected starts at selected conversion", "sample_count": None, "population_estimate": access_adjusted * conversion_rate, "estimate_type": "expected count"})
    return pd.DataFrame(rows)
