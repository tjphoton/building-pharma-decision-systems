"""Run the Chapter 6 HCP and account targeting analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from targeting import (  # noqa: E402
    ANALYSIS_DATE,
    MIN_ACCESS_SIGNAL_PATIENTS,
    MIN_ACCOUNT_PATIENTS,
    MIN_OPPORTUNITY_PATIENTS,
    RECENT_DAYS,
    SATURATION_CONTACTS_PER_HCP,
    apply_account_policy,
    build_account_features,
    build_call_plan,
    build_decile_summary,
    build_gate_summary,
    build_hcp_actions,
    build_hcp_features,
    build_territory_summary,
    compare_naive_and_gated,
)


def input_paths(repo_root: Path) -> dict[str, Path]:
    """Return the authoritative source paths for Chapter 6."""
    data = repo_root / "ch03_data" / "output_data" / "generated_data"
    journey = repo_root / "ch05_journey" / "assets" / "generated_outputs"
    return {
        "journeys": journey / "initiation_journeys.csv",
        "medical_claims": data / "claims_medical" / "medical_claims_mature.csv",
        "hcp_roster": data / "reference" / "hcp_targets.csv",
        "providers": data / "reference" / "providers.csv",
        "accounts": data / "reference" / "accounts.csv",
        "crm": data / "crm_veeva" / "crm_interactions.csv",
        "chapter3_manifest": data / "manifest.json",
    }


def load_inputs(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Load Chapter 3 source tables and the Chapter 5 journey artifact."""

    paths = input_paths(repo_root)
    return {
        "journeys": pd.read_csv(
            paths["journeys"],
            parse_dates=["index_date", "followup_end", "first_treatment_date"],
        ),
        "medical_claims": pd.read_csv(
            paths["medical_claims"],
            parse_dates=["claim_date"],
        ),
        "hcp_roster": pd.read_csv(paths["hcp_roster"]),
        "providers": pd.read_csv(paths["providers"]),
        "accounts": pd.read_csv(paths["accounts"]),
        "crm": pd.read_csv(
            paths["crm"],
            parse_dates=["interaction_date"],
        ),
    }


def run_analysis(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Return the full Chapter 6 targeting evidence package."""

    inputs = load_inputs(repo_root)
    hcp_features, patient_hcp = build_hcp_features(inputs)
    hcp_deciles, decile_summary = build_decile_summary(hcp_features)
    account_features = build_account_features(hcp_features, inputs["accounts"])
    account_targets = apply_account_policy(account_features)
    hcp_targets = build_hcp_actions(hcp_deciles, account_targets)
    call_plan = build_call_plan(account_targets, hcp_targets)
    return {
        "patient_hcp": patient_hcp,
        "hcp_features": hcp_features,
        "hcp_targets": hcp_targets,
        "hcp_deciles": hcp_deciles,
        "decile_summary": decile_summary,
        "account_features": account_features,
        "account_targets": account_targets,
        "gate_summary": build_gate_summary(account_targets),
        "call_plan": call_plan,
        "territory_summary": build_territory_summary(account_targets, call_plan),
        "plan_comparison": compare_naive_and_gated(hcp_targets, call_plan),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    repo_root: Path | None = None,
) -> None:
    """Write reusable CSV artifacts and their provenance manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_contracts = {}
    for name, frame in results.items():
        exported = frame.copy()
        exported.insert(0, "analysis_date", ANALYSIS_DATE.date().isoformat())
        path = output_dir / f"{name}.csv"
        exported.to_csv(path, index=False)
        output_contracts[path.name] = {
            "rows": len(exported),
            "columns": list(exported.columns),
            "file_sha256": _sha256(path),
        }

    if repo_root is None:
        repo_root = output_dir.parents[2]
    paths = input_paths(repo_root)
    chapter3_manifest = json.loads(paths["chapter3_manifest"].read_text())
    journey_dates = pd.read_csv(
        paths["journeys"], usecols=["index_date", "followup_end"]
    )
    medical_dates = pd.read_csv(paths["medical_claims"], usecols=["claim_date"])
    crm_dates = pd.read_csv(paths["crm"], usecols=["interaction_date"])
    manifest = {
        "chapter": 6,
        "artifact": "HCP and account targeting evidence package",
        "analysis_date": ANALYSIS_DATE.date().isoformat(),
        "upstream_synthetic_seed": chapter3_manifest["run_config"]["seed"],
        "source_files": {
            name: {
                "path": str(path.relative_to(repo_root)),
                "file_sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "date_ranges": {
            "journey_index_date": [
                journey_dates["index_date"].min(),
                journey_dates["index_date"].max(),
            ],
            "journey_followup_end": [
                journey_dates["followup_end"].min(),
                journey_dates["followup_end"].max(),
            ],
            "medical_claim_date": [
                medical_dates["claim_date"].min(),
                medical_dates["claim_date"].max(),
            ],
            "crm_interaction_date": [
                crm_dates["interaction_date"].min(),
                crm_dates["interaction_date"].max(),
            ],
        },
        "analysis_filters": {
            "journey_followup_through": ANALYSIS_DATE.date().isoformat(),
            "crm_interactions_through": ANALYSIS_DATE.date().isoformat(),
            "attribution_event": "Rendering NPI on diagnosis index encounter",
            "provider_scope": "Chapter 3 HCP target roster",
        },
        "scenario_assumptions": {
            "recent_days": RECENT_DAYS,
            "minimum_account_patients": MIN_ACCOUNT_PATIENTS,
            "minimum_opportunity_patients": MIN_OPPORTUNITY_PATIENTS,
            "minimum_access_signal_patients": MIN_ACCESS_SIGNAL_PATIENTS,
            "saturation_contacts_per_hcp": SATURATION_CONTACTS_PER_HCP,
        },
        "outputs": output_contracts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(results: Mapping[str, pd.DataFrame]) -> None:
    """Print the headline results quoted in the chapter."""

    patient_hcp = results["patient_hcp"]
    accounts = results["account_targets"]
    calls = results["call_plan"]
    print(f"Target-roster patients: {patient_hcp['patient_id'].nunique():,}")
    print(f"Target HCPs with attributed patients: {results['hcp_features']['npi'].nunique():,}")
    print(f"Accounts with attributed patients: {accounts['account_id'].nunique():,}")
    print("\nAccount actions:")
    print(accounts["account_action"].value_counts().to_string())
    print(f"\nPlanned HCPs: {calls['npi'].nunique():,}")
    print(f"Recommended calls: {calls['recommended_calls'].sum():,}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "ch06_hcp" / "assets" / "generated_outputs"
    analysis = run_analysis(root)
    write_outputs(analysis, output)
    print_summary(analysis)
    print(f"\nWrote Chapter 6 outputs to {output.relative_to(root)}")
