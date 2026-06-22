"""Entry point: assembles and runs the full Chapter 4 market-sizing workflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from calibration import (  # noqa: E402
    bootstrap_access_opportunity,
    nhanes_calibration,
)
from estimands import (  # noqa: E402
    DX_COLS,
    LAUNCH_CONDITION_CODES,
    LAUNCH_PRODUCT,
    build_patient_analysis,
    drug_name_from_ndc,
    funnel_estimates,
    load_chapter3_data,
    panel_market_sizes,
    phenotype_diagnostics,
)
from geography import (  # noqa: E402
    account_opportunity,
    account_rank_stability,
    opportunity_choropleth,
    state_opportunity,
)
from maturity import capture_recapture, claims_maturity_adjustment  # noqa: E402
from patient_finding import patient_finding_lift  # noqa: E402
from scenario import scenario_grid  # noqa: E402

# Bootstrap replicates for the sampling-uncertainty interval.
N_BOOT = 1_000


# Drugs that treat the launch condition in the synthetic market. Roventra is the
# launch product and is condition-specific. Nexoral and Vexpro are established
# competitors that are also dispensed for other conditions, so a pharmacy source
# built from the whole class captures patients outside the launch condition.
LAUNCH_CLASS_DRUGS = ["Roventra", "Nexoral", "Vexpro"]


def condition_capture_recapture(tables: dict) -> pd.DataFrame:
    """Two-source Chapman estimates: a contaminated pairing and a clean one.

    Both sources pair medical claims (Source A: a paid launch-condition diagnosis)
    against pharmacy claims (Source B). The difference is which drugs define the
    pharmacy source.

    Contaminated: any paid fill for the drug class (Roventra, Nexoral, Vexpro).
    Two of those drugs are also dispensed for other conditions, so Source B no
    longer identifies the same population and the estimate overshoots.

    Condition-specific: only paid Roventra fills. Roventra is specific to the
    launch condition, so both sources identify the same target population.
    """

    medical = tables["medical_claims"]
    pharmacy = tables["pharmacy_claims"].copy()

    paid_dx = medical.loc[
        medical[DX_COLS].isin(LAUNCH_CONDITION_CODES).any(axis=1)
    ]
    source_a = set(paid_dx["patient_id"])

    pharmacy["drug_name"] = drug_name_from_ndc(pharmacy, tables["ndc_codes"])
    paid_rx = pharmacy.loc[pharmacy["transaction_type"].eq("PAID")]

    pairings = {
        "Contaminated: medical dx x paid drug class": set(
            paid_rx.loc[paid_rx["drug_name"].isin(LAUNCH_CLASS_DRUGS), "patient_id"]
        ),
        f"Condition-specific: medical dx x paid {LAUNCH_PRODUCT}": set(
            paid_rx.loc[paid_rx["drug_name"].eq(LAUNCH_PRODUCT), "patient_id"]
        ),
    }
    rows = []
    for pairing, source_b in pairings.items():
        result = capture_recapture(
            source_a_count=len(source_a),
            source_b_count=len(source_b),
            overlap_count=len(source_a & source_b),
        )
        rows.append({"pairing": pairing, **result})
    return pd.DataFrame(rows)


def structural_phenotype_check(patients: pd.DataFrame) -> pd.DataFrame:
    """Compare base and strict phenotypes after recalibrating each to the anchor."""

    rows = []
    for phenotype in ("base_phenotype", "strict_phenotype"):
        calibrated, _ = nhanes_calibration(patients, phenotype_column=phenotype)
        eligible = (
            calibrated[phenotype]
            & calibrated["age_eligible"]
            & calibrated["untreated_opportunity"]
        )
        rows.append(
            {
                "phenotype": phenotype,
                "diagnosed_sample": int(calibrated[phenotype].sum()),
                "untreated_sample": int(eligible.sum()),
                "untreated_population": round(
                    float(calibrated.loc[eligible, "population_weight"].sum()), 0
                ),
                "reachable_opportunity": round(
                    float(
                        (
                            calibrated.loc[eligible, "population_weight"]
                            * calibrated.loc[eligible, "access_probability"]
                        ).sum()
                    ),
                    0,
                ),
            }
        )
    return pd.DataFrame(rows)


def diagnosis_visibility(patients: pd.DataFrame) -> pd.DataFrame:
    """Trace the synthetic condition from truth to coded and paid evidence."""

    truth = patients["reference_condition"]
    coded = patients["launch_diagnosis_coded"]
    paid = patients["base_phenotype"]
    return pd.DataFrame(
        [
            {
                "measure": "True condition patients",
                "patient_count": int(truth.sum()),
            },
            {
                "measure": "True patients with any launch diagnosis code",
                "patient_count": int((truth & coded).sum()),
            },
            {
                "measure": "Other-condition patients with any launch diagnosis code",
                "patient_count": int((~truth & coded).sum()),
            },
            {
                "measure": "True positives in paid-claims phenotype",
                "patient_count": int((truth & paid).sum()),
            },
            {
                "measure": "False positives in paid-claims phenotype",
                "patient_count": int((~truth & paid).sum()),
            },
        ]
    )


def treatment_intersection(patients: pd.DataFrame) -> pd.DataFrame:
    """Show why treated counts must use the same funnel denominator."""

    stages = [
        ("All panel patients", pd.Series(True, index=patients.index)),
        ("True condition", patients["reference_condition"]),
        ("Paid-claims phenotype", patients["base_phenotype"]),
        (
            "Age-eligible paid-claims phenotype",
            patients["base_phenotype"] & patients["age_eligible"],
        ),
    ]
    return pd.DataFrame(
        [
            {
                "population": label,
                "population_count": int(mask.sum()),
                "current_product_users": int(
                    (mask & patients["current_product_user"]).sum()
                ),
                "untreated_patients": int(
                    (mask & patients["untreated_opportunity"]).sum()
                ),
            }
            for label, mask in stages
        ]
    )


def run_analysis(repo_root: Path) -> dict:
    """Execute the full Chapter 4 market-sizing workflow."""

    data_dir = repo_root / "ch03_data" / "output_data" / "generated_data"
    tables = load_chapter3_data(data_dir)
    patients = build_patient_analysis(tables)

    patients_cal, nhanes_bridge = nhanes_calibration(patients)

    panel_sizes = panel_market_sizes(patients)
    diagnostics = pd.DataFrame(
        [
            phenotype_diagnostics(patients, "base_phenotype"),
            phenotype_diagnostics(patients, "strict_phenotype"),
        ]
    )
    funnel = funnel_estimates(patients_cal)
    acct_opp = account_opportunity(patients_cal, tables["accounts"])
    state_opp = state_opportunity(patients_cal)

    maturity = claims_maturity_adjustment(patients_cal)
    cr_table = condition_capture_recapture(tables)
    scenarios = scenario_grid(patients_cal)
    structural = structural_phenotype_check(patients)
    rank_stability = account_rank_stability(patients_cal, tables["accounts"])

    boot = bootstrap_access_opportunity(patients_cal, n_boot=N_BOOT)
    reachable_point = float(
        funnel.loc[
            funnel["stage"].eq("Access-adjusted reachable opportunity"),
            "population_estimate",
        ].iloc[0]
    )
    uncertainty = pd.DataFrame(
        [
            {
                "estimate": "Access-adjusted reachable opportunity",
                "point_estimate": round(reachable_point, 0),
                "bootstrap_replicates": N_BOOT,
                "ci_95_low": round(float(np.percentile(boot, 2.5)), 0),
                "ci_95_high": round(float(np.percentile(boot, 97.5)), 0),
            }
        ]
    )

    finding = patient_finding_lift(tables, patients)

    # Uncertainty layers expressed on one metric (reachable opportunity) for the tornado.
    base_access = scenarios["reachable_opportunity"].iloc[0] / 0.85
    strict_reachable = float(
        structural.loc[structural["phenotype"].eq("strict_phenotype"), "reachable_opportunity"].iloc[0]
    )
    layers = pd.DataFrame(
        [
            {
                "layer": "Parameter (access)",
                "low": round(base_access * 0.85, 0),
                "high": round(base_access * 1.15, 0),
            },
            {
                "layer": "Sampling (bootstrap)",
                "low": round(float(np.percentile(boot, 2.5)), 0),
                "high": round(float(np.percentile(boot, 97.5)), 0),
            },
            {
                "layer": "Structural (phenotype)",
                "low": round(min(strict_reachable, reachable_point), 0),
                "high": round(max(strict_reachable, reachable_point), 0),
            },
        ]
    )
    layers["point"] = round(reachable_point, 0)
    layers["width"] = layers["high"] - layers["low"]

    return {
        "panel_market_sizes": panel_sizes,
        "diagnosis_visibility": diagnosis_visibility(patients),
        "phenotype_diagnostics": diagnostics,
        "treatment_intersection": treatment_intersection(patients),
        "nhanes_bridge": nhanes_bridge,
        "funnel": funnel,
        "account_opportunity": acct_opp,
        "state_opportunity": state_opp,
        "claims_maturity": maturity,
        "capture_recapture": cr_table,
        "scenario_grid": scenarios,
        "structural_phenotype": structural,
        "account_rank_stability": rank_stability,
        "uncertainty": uncertainty,
        "uncertainty_layers": layers,
        "patient_finding": finding,
    }


def write_outputs(results: dict, output_dir: Path) -> None:
    """Write Chapter 4 outputs to CSV and HTML."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_names = [
        "panel_market_sizes",
        "diagnosis_visibility",
        "phenotype_diagnostics",
        "treatment_intersection",
        "nhanes_bridge",
        "funnel",
        "account_opportunity",
        "state_opportunity",
        "claims_maturity",
        "capture_recapture",
        "scenario_grid",
        "structural_phenotype",
        "account_rank_stability",
        "uncertainty",
        "uncertainty_layers",
        "patient_finding",
    ]
    for name in csv_names:
        results[name].to_csv(output_dir / f"{name}.csv", index=False)
    opportunity_choropleth(
        results["state_opportunity"],
        output_dir / "opportunity_choropleth.html",
    )


def print_summary(results: dict) -> None:
    """Print the headline tables in the format quoted by the chapter."""

    print("Sequential panel counts:")
    print(results["panel_market_sizes"].to_string(index=False))

    funnel = results["funnel"].copy()
    funnel["population_estimate"] = funnel["population_estimate"].map(
        lambda value: f"{value:,.0f}"
    )
    print("\nPatient opportunity funnel (nationally anchored):")
    print(funnel.to_string(index=False))

    uncertainty = results["uncertainty"].iloc[0]
    print(
        f"\nBootstrap 95% interval for reachable opportunity "
        f"({int(uncertainty['bootstrap_replicates'])} replicates): "
        f"{uncertainty['ci_95_low']:,.0f} to {uncertainty['ci_95_high']:,.0f}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    outputs = run_analysis(root)
    destination = Path(__file__).resolve().parents[1] / "assets" / "generated_outputs"
    write_outputs(outputs, destination)
    print_summary(outputs)
    print(f"\nWrote Chapter 4 outputs to {destination}")
