"""Run data-quality checks against the Chapter 3 synthetic data package."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_modules.entities import validate_manifest_contract  # noqa: E402


def load_chapter3_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the generated tables needed by the quality checks."""
    return {
        "patients": pd.read_csv(data_dir / "reference" / "patients.csv"),
        "patient_enrollments": pd.read_csv(
            data_dir / "reference" / "patient_enrollments.csv",
            parse_dates=["eligibility_start_date", "eligibility_end_date"],
        ),
        "providers": pd.read_csv(data_dir / "reference" / "providers.csv"),
        "hcp_targets": pd.read_csv(data_dir / "reference" / "hcp_targets.csv"),
        "pharmacy_claims": pd.read_csv(
            data_dir / "claims_pharmacy" / "pharmacy_claims.csv",
            dtype={"ndc": str, "ndc_prescribed": str},
            parse_dates=["date_of_service"],
        ),
        "medical_claims": pd.read_csv(
            data_dir / "claims_medical" / "medical_claims.csv",
            parse_dates=["claim_date"],
        ),
        "medical_claims_mature": pd.read_csv(
            data_dir / "claims_medical" / "medical_claims_mature.csv",
            parse_dates=["claim_date"],
        ),
        "specialty_pharmacy": pd.read_csv(
            data_dir / "specialty_pharmacy" / "sp_events.csv",
            parse_dates=["referral_date", "status_date", "ship_date"],
        ),
        "ndc_codes": pd.read_csv(
            data_dir / "reference" / "ndc_codes.csv",
            dtype={"ndc": str},
        ),
    }


def snapshot_comparison_audit(
    early: pd.DataFrame, mature: pd.DataFrame
) -> pd.DataFrame:
    """Compare early vs mature medical claim snapshot counts by month."""
    dx_cols = [f"diagnosis_{i}" for i in range(1, 11)]

    def _t2d_by_month(df: pd.DataFrame) -> pd.Series:
        t2d_mask = df[dx_cols].apply(
            lambda col: col.astype(str).str.startswith("E11") & col.notna()
        ).any(axis=1)
        return (
            df.loc[t2d_mask]
            .assign(month=df["claim_date"].dt.to_period("M").astype(str))
            ["month"].value_counts()
            .sort_index()
        )

    early_counts = _t2d_by_month(early).rename("early_snapshot")
    mature_counts = _t2d_by_month(mature).rename("mature_snapshot")
    view = pd.DataFrame({"early_snapshot": early_counts, "mature_snapshot": mature_counts}).fillna(0).astype(int)
    view["completeness_pct"] = (100 * view["early_snapshot"] / view["mature_snapshot"].replace(0, 1)).round(1)
    view.index.name = "service_month"
    return view.reset_index()


def product_mapping_audit(
    pharmacy_claims: pd.DataFrame,
    ndc_codes: pd.DataFrame,
) -> pd.DataFrame:
    """Audit NDC presence and resolved product names for both prescribed and dispensed codes."""
    known = set(ndc_codes["ndc"].dropna().astype(str))

    # Check prescribed code (ndc_prescribed): stable product attribution
    prescribed_known = pharmacy_claims["ndc_prescribed"].astype(str).isin(known)
    # Check dispensed code (ndc): may differ from prescribed for pack-size variants
    dispensed_known = pharmacy_claims["ndc"].astype(str).isin(known)

    status = pd.Series("Both NDC codes in reference", index=pharmacy_claims.index)
    status.loc[~prescribed_known] = "Prescribed NDC absent from reference"
    status.loc[prescribed_known & ~dispensed_known] = "Dispensed NDC absent - pack-size variant"

    result = status.value_counts().rename_axis("mapping_status").reset_index(name="claim_count")
    result["percent"] = (100 * result["claim_count"] / max(len(pharmacy_claims), 1)).round(2)
    result["analysis_action"] = result["mapping_status"].map(
        {
            "Both NDC codes in reference": "Join on ndc_prescribed for stable product name",
            "Prescribed NDC absent from reference": "Update reference; flag as unmapped before product analysis",
            "Dispensed NDC absent - pack-size variant": "Join on ndc_prescribed; note dispensed code needs reference refresh",
        }
    )
    return result


def coverage_observation_audit(
    patient_enrollments: pd.DataFrame,
    index_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
    lookback_days: int = 180,
    followup_end: pd.Timestamp = pd.Timestamp("2024-12-31"),
) -> pd.DataFrame:
    """Check whether each patient has the required observation window via enrollment table."""
    required_start = index_date - pd.Timedelta(days=lookback_days)
    coverage = (
        patient_enrollments
        .groupby("patient_id")
        .agg(
            coverage_start=("eligibility_start_date", "min"),
            coverage_end=("eligibility_end_date", "max"),
        )
        .reset_index()
    )
    enough_lookback = coverage["coverage_start"] <= required_start
    enough_followup = coverage["coverage_end"] >= followup_end
    status = pd.Series("Eligible observation window", index=coverage.index)
    status.loc[~enough_lookback & enough_followup] = "Insufficient lookback"
    status.loc[enough_lookback & ~enough_followup] = "Insufficient follow-up"
    status.loc[~enough_lookback & ~enough_followup] = "Insufficient lookback and follow-up"
    result = status.value_counts().rename_axis("coverage_status").reset_index(name="patient_count")
    result["percent"] = (100 * result["patient_count"] / max(len(coverage), 1)).round(2)
    result["analysis_action"] = result["coverage_status"].map(
        {
            "Eligible observation window": "Include in analyses requiring this window",
            "Insufficient lookback": "Exclude or use a shorter prespecified lookback",
            "Insufficient follow-up": "Exclude from outcomes requiring complete follow-up",
            "Insufficient lookback and follow-up": "Exclude from fixed-window analysis",
        }
    )
    return result


def temporal_integrity_audit(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Check that clinical events fall within each patient's coverage period."""
    enrollments = tables["patient_enrollments"]
    coverage = (
        enrollments
        .groupby("patient_id")
        .agg(
            coverage_start=("eligibility_start_date", "min"),
            coverage_end=("eligibility_end_date", "max"),
        )
        .reset_index()
    )
    checks: list[dict] = []
    date_specs = [
        ("Medical claim service date", tables["medical_claims_mature"], "claim_date"),
        ("Pharmacy transaction date", tables["pharmacy_claims"], "date_of_service"),
        ("Specialty-pharmacy referral date", tables["specialty_pharmacy"], "referral_date"),
        ("Specialty-pharmacy authorization date", tables["specialty_pharmacy"], "status_date"),
    ]
    for check_name, frame, date_field in date_specs:
        if date_field not in frame.columns:
            continue
        dated = frame[["patient_id", date_field]].merge(coverage, on="patient_id", how="left")
        valid = dated[date_field].notna()
        violations = valid & (
            (dated[date_field] < dated["coverage_start"])
            | (dated[date_field] > dated["coverage_end"])
        )
        checks.append(
            {
                "check": check_name,
                "violation_count": int(violations.sum()),
                "expected": "0 outside patient coverage",
            }
        )

    sp = tables["specialty_pharmacy"]
    if "status_date" in sp.columns and "referral_date" in sp.columns:
        auth_before_referral = sp["status_date"] < sp["referral_date"]
        checks.append(
            {
                "check": "Specialty authorization is on or after referral",
                "violation_count": int(auth_before_referral.sum()),
                "expected": "0 reversed sequences",
            }
        )
    if "ship_date" in sp.columns and "status_date" in sp.columns:
        ship_before_auth = sp["ship_date"].notna() & (sp["ship_date"] < sp["status_date"])
        checks.append(
            {
                "check": "Specialty shipment is on or after authorization",
                "violation_count": int(ship_before_auth.sum()),
                "expected": "0 reversed sequences",
            }
        )
    return pd.DataFrame(checks)


def build_quality_summary(
    tables: dict[str, pd.DataFrame],
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create the headline quality measures for the chapter figure."""
    pharmacy = tables["pharmacy_claims"]

    known_ndcs = set(tables["ndc_codes"]["ndc"].dropna().astype(str))
    unmapped_rate = 100 * (~pharmacy["ndc_prescribed"].astype(str).isin(known_ndcs)).mean()

    eligible_row = results["dq_coverage_observation"].query(
        "coverage_status == 'Eligible observation window'"
    )
    eligible_rate = float(eligible_row["percent"].iloc[0]) if not eligible_row.empty else 0.0

    snapshot_df = results.get("dq_snapshot_comparison", pd.DataFrame())
    if not snapshot_df.empty and "completeness_pct" in snapshot_df.columns:
        median_completeness = round(float(snapshot_df["completeness_pct"].median()), 1)
    else:
        median_completeness = float("nan")

    return pd.DataFrame(
        [
            {"metric": "Missing NDC prescribed mappings", "value": round(unmapped_rate, 2), "unit": "%"},
            {"metric": "Eligible observation window", "value": round(eligible_rate, 2), "unit": "%"},
            {"metric": "Median early-snapshot completeness", "value": median_completeness, "unit": "%"},
        ]
    )


def run_data_quality(data_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run all quality checks and return detailed and headline results."""
    validate_manifest_contract(data_dir)
    tables = load_chapter3_tables(data_dir)
    results = {
        "dq_snapshot_comparison": snapshot_comparison_audit(
            tables["medical_claims"], tables["medical_claims_mature"]
        ),
        "dq_product_mapping": product_mapping_audit(tables["pharmacy_claims"], tables["ndc_codes"]),
        "dq_coverage_observation": coverage_observation_audit(tables["patient_enrollments"]),
        "dq_temporal_integrity": temporal_integrity_audit(tables),
    }
    return results, build_quality_summary(tables, results)


if __name__ == "__main__":
    output_root = Path(__file__).resolve().parents[1] / "output_data"
    data_dir = output_root / "generated_data"
    output_dir = output_root / "analysis_results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    results, summary = run_data_quality(data_dir)
    for name, frame in {**results, "dq_summary": summary}.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")
