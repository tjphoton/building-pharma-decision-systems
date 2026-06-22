"""Run the Chapter 6 HCP and account targeting analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from kol import (  # noqa: E402
    build_kol_profiles,
    build_kol_validation,
    build_transparency_review,
)
from referral_network import (  # noqa: E402
    build_account_referral_context,
    build_referral_graph,
    build_referral_stability,
    prepare_referral_episodes,
    referral_centrality,
)
from segmentation import (  # noqa: E402
    build_policy_baseline,
    compare_with_policy_baseline,
    evaluate_cluster_counts,
    fit_hcp_segments,
    prepare_segmentation_features,
    select_cluster_count,
)
from targeting import (  # noqa: E402
    ADOPTION_THRESHOLD,
    ANALYSIS_DATE,
    MIN_ACCESS_SIGNAL_PATIENTS,
    MIN_ACCOUNT_PATIENTS,
    MIN_OPPORTUNITY_PATIENTS,
    MIN_TREATED_PATIENTS,
    RECENT_DAYS,
    SATURATION_CONTACTS_PER_HCP,
    apply_account_policy,
    attribute_index_hcp,
    build_account_features,
    build_call_plan,
    build_coverage_funnel,
    build_coverage_summary,
    build_decile_summary,
    build_gate_summary,
    build_hcp_actions,
    build_hcp_features,
    build_override_template,
    build_policy_sensitivity,
    build_target_universe,
    build_territory_summary,
    compare_attribution_rules,
    compare_naive_and_gated,
)


def input_paths(repo_root: Path) -> dict[str, Path]:
    """Return every authoritative source path used by Chapter 6."""

    upstream = repo_root / "ch03_data" / "output_data" / "generated_data"
    journey = repo_root / "ch05_journey" / "assets" / "generated_outputs"
    generated = repo_root / "ch06_hcp" / "data" / "generated"
    return {
        "journeys": journey / "initiation_journeys.csv",
        "medical_claims": upstream / "claims_medical" / "medical_claims_mature.csv",
        "providers": upstream / "reference" / "providers.csv",
        "hcp_roster": upstream / "reference" / "hcp_targets.csv",
        "accounts": upstream / "reference" / "accounts.csv",
        "crm": upstream / "crm_veeva" / "crm_interactions.csv",
        "chapter3_manifest": upstream / "manifest.json",
        "chapter6_data_manifest": generated / "manifest.json",
        "affiliations": generated / "hcp_account_affiliations.csv",
        "permissions": generated / "contact_permissions.csv",
        "attribution_events": generated / "attribution_events.csv",
        "current_treatment_state": generated / "current_treatment_state.csv",
        "referral_episodes": generated / "referral_episodes.csv",
        "scientific_profiles": generated / "scientific_profiles.csv",
        "scientific_evidence": generated / "scientific_evidence.csv",
        "scientific_collaborations": generated / "scientific_collaborations.csv",
        "medical_reviews": generated / "medical_reviews.csv",
        "engagement_signals": generated / "engagement_signals.csv",
        "transparency": generated / "transparency_review.csv",
        "territory_capacity": generated / "territory_capacity.csv",
    }


def _require_inputs(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Chapter 6 inputs are missing. Run "
            "`uv run python ch06_hcp/scripts/generate_ch06_data.py` first:\n"
            f"{joined}"
        )


def load_inputs(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Load read-only upstream sources and Chapter 6 supplemental tables."""

    paths = input_paths(repo_root)
    _require_inputs(paths)
    return {
        "journeys": pd.read_csv(
            paths["journeys"],
            parse_dates=["index_date", "followup_end", "first_treatment_date"],
        ),
        "medical_claims": pd.read_csv(
            paths["medical_claims"],
            dtype={"rendering_npi": str, "referring_npi": str},
            parse_dates=["claim_date"],
        ),
        "providers": pd.read_csv(paths["providers"], dtype={"npi": str}),
        "hcp_roster": pd.read_csv(paths["hcp_roster"], dtype={"npi": str}),
        "accounts": pd.read_csv(paths["accounts"]),
        "crm": pd.read_csv(
            paths["crm"], dtype={"hcp_npi": str}, parse_dates=["interaction_date"]
        ),
        "hcp_account_affiliations": pd.read_csv(
            paths["affiliations"],
            dtype={"npi": str},
            parse_dates=["effective_start", "effective_end"],
        ),
        "contact_permissions": pd.read_csv(
            paths["permissions"],
            dtype={"npi": str},
            parse_dates=["effective_start", "effective_end"],
        ),
        "attribution_events": pd.read_csv(
            paths["attribution_events"],
            dtype={"npi": str},
            parse_dates=["event_date"],
        ),
        "current_treatment_state": pd.read_csv(
            paths["current_treatment_state"],
            parse_dates=["state_as_of"],
        ),
        "referral_episodes": pd.read_csv(
            paths["referral_episodes"],
            dtype={"source_npi": str, "destination_npi": str},
            parse_dates=["source_date", "destination_date"],
        ),
        "scientific_profiles": pd.read_csv(
            paths["scientific_profiles"], dtype={"npi": str}
        ),
        "scientific_evidence": pd.read_csv(
            paths["scientific_evidence"],
            dtype={"npi": str},
            parse_dates=["event_date"],
        ),
        "scientific_collaborations": pd.read_csv(
            paths["scientific_collaborations"],
            dtype={"source_npi": str, "destination_npi": str},
            parse_dates=["collaboration_date"],
        ),
        "medical_reviews": pd.read_csv(
            paths["medical_reviews"], dtype={"npi": str}, parse_dates=["review_date"]
        ),
        "engagement_signals": pd.read_csv(
            paths["engagement_signals"], dtype={"npi": str}
        ),
        "transparency": pd.read_csv(paths["transparency"], dtype={"npi": str}),
        "territory_capacity": pd.read_csv(
            paths["territory_capacity"], parse_dates=["cycle_start", "cycle_end"]
        ),
    }


def run_analysis(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Return the complete Chapter 6 evidence and decision package."""

    inputs = load_inputs(repo_root)

    target_universe = build_target_universe(inputs)
    attribution, attribution_summary = compare_attribution_rules(
        inputs["journeys"], inputs["attribution_events"]
    )
    hcp_features, patient_hcp = build_hcp_features(inputs)
    hcp_deciles, decile_summary = build_decile_summary(hcp_features)
    coverage_summary = build_coverage_summary(inputs["journeys"], patient_hcp)

    referral_episodes = prepare_referral_episodes(inputs["referral_episodes"])
    referral_graph, referral_edges = build_referral_graph(referral_episodes)
    referral_metrics = referral_centrality(
        referral_graph, inputs["hcp_account_affiliations"]
    )
    referral_stability = build_referral_stability(
        inputs["referral_episodes"], inputs["hcp_account_affiliations"]
    )
    account_referral_context = build_account_referral_context(
        referral_metrics, referral_stability
    )

    kol_profiles, kol_domain_evidence = build_kol_profiles(
        inputs["scientific_evidence"],
        inputs["scientific_collaborations"],
        inputs["scientific_profiles"],
    )
    kol_review_detail, kol_validation = build_kol_validation(
        kol_profiles, inputs["medical_reviews"]
    )
    kol_transparency = build_transparency_review(
        kol_profiles, inputs["transparency"]
    )

    segment_features, segment_matrix, _ = prepare_segmentation_features(
        hcp_features, inputs["engagement_signals"]
    )
    cluster_evaluation = evaluate_cluster_counts(segment_matrix)
    cluster_evaluation["selected"] = cluster_evaluation["k"].eq(
        select_cluster_count(cluster_evaluation)
    )
    _, hcp_segments, segment_profiles = fit_hcp_segments(
        segment_features, segment_matrix, cluster_evaluation
    )
    policy_segments = build_policy_baseline(segment_features)
    segment_policy_comparison = compare_with_policy_baseline(
        hcp_segments, policy_segments
    )
    hcp_segments = hcp_segments.rename(
        columns={
            "cluster_id": "segment_id",
            "centroid_distance": "distance_to_centroid",
            "model_version": "segment_model_version",
        }
    )

    account_features = build_account_features(hcp_features, inputs["accounts"])
    account_targets = apply_account_policy(account_features).merge(
        account_referral_context,
        on="account_id",
        how="left",
        validate="one_to_one",
    )
    account_targets["pathway_action"] = account_targets["pathway_action"].fillna(
        "No observed pathway"
    )
    account_targets["pathway_reason"] = account_targets["pathway_reason"].fillna(
        "No qualifying disease-specific referral edge"
    )
    policy_sensitivity = build_policy_sensitivity(account_features)
    hcp_targets = build_hcp_actions(hcp_deciles, account_targets, hcp_segments)
    call_plan = build_call_plan(
        account_targets, hcp_targets, inputs["territory_capacity"]
    )
    territory_summary = build_territory_summary(
        account_targets, call_plan, inputs["territory_capacity"]
    )
    index_attribution = attribute_index_hcp(
        inputs["journeys"], inputs["attribution_events"]
    )

    return {
        "target_universe": target_universe,
        "attribution_comparison": attribution,
        "attribution_summary": attribution_summary,
        "patient_hcp": patient_hcp,
        "hcp_features": hcp_features,
        "hcp_deciles": hcp_deciles,
        "decile_summary": decile_summary,
        "coverage_summary": coverage_summary,
        "referral_episodes": referral_episodes,
        "referral_edges": referral_edges,
        "referral_metrics": referral_metrics,
        "referral_stability": referral_stability,
        "account_referral_context": account_referral_context,
        "kol_profiles": kol_profiles,
        "kol_domain_evidence": kol_domain_evidence,
        "kol_review_detail": kol_review_detail,
        "kol_validation": kol_validation,
        "kol_transparency_review": kol_transparency,
        "cluster_evaluation": cluster_evaluation,
        "hcp_segments": hcp_segments,
        "segment_profiles": segment_profiles,
        "segment_policy_comparison": segment_policy_comparison,
        "account_features": account_features,
        "account_targets": account_targets,
        "policy_sensitivity": policy_sensitivity,
        "gate_summary": build_gate_summary(account_targets),
        "hcp_targets": hcp_targets,
        "call_plan": call_plan,
        "override_template": build_override_template(call_plan),
        "coverage_funnel": build_coverage_funnel(
            inputs["journeys"],
            index_attribution,
            patient_hcp,
            hcp_targets,
            call_plan,
        ),
        "territory_summary": territory_summary,
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
    """Write reusable artifacts and a complete provenance manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in [*output_dir.glob("*.csv"), output_dir / "manifest.json"]:
        stale_path.unlink(missing_ok=True)
    output_contracts = {}
    for name, frame in results.items():
        exported = frame.copy()
        if "analysis_date" in exported.columns:
            exported["analysis_date"] = ANALYSIS_DATE.date().isoformat()
            ordered = ["analysis_date", *[c for c in exported if c != "analysis_date"]]
            exported = exported[ordered]
        else:
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
    chapter6_data_manifest = json.loads(paths["chapter6_data_manifest"].read_text())
    manifest = {
        "chapter": 6,
        "artifact": "HCP and account decision package",
        "analysis_date": ANALYSIS_DATE.date().isoformat(),
        "rule_version": "ch06-targeting-v2",
        "upstream_synthetic_seed": chapter3_manifest["run_config"]["seed"],
        "chapter6_seed": chapter6_data_manifest["seed"],
        "source_files": {
            name: {
                "path": str(path.relative_to(repo_root)),
                "file_sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "analysis_filters": {
            "data_cutoff": ANALYSIS_DATE.date().isoformat(),
            "primary_attribution": "Rendering HCP on diagnosis index date",
            "attribution_sensitivity": [
                "Plurality relevant HCP within 180 days",
                "Latest relevant HCP through cutoff",
            ],
            "commercial_specialties": ["Endocrinology", "Primary Care"],
            "referral_condition": "Type 2 diabetes",
            "referral_transition_days": 60,
            "segmentation_population": (
                "Field-permitted HCPs with at least 5 cohort and 3 treated patients"
            ),
        },
        "scenario_assumptions": {
            "recent_days": RECENT_DAYS,
            "minimum_account_patients": MIN_ACCOUNT_PATIENTS,
            "minimum_treated_patients": MIN_TREATED_PATIENTS,
            "minimum_review_opportunity": MIN_OPPORTUNITY_PATIENTS,
            "minimum_access_signal_patients": MIN_ACCESS_SIGNAL_PATIENTS,
            "adoption_threshold": ADOPTION_THRESHOLD,
            "saturation_contacts_per_hcp": SATURATION_CONTACTS_PER_HCP,
        },
        "decision_boundaries": {
            "referral": "Pathway and account context only",
            "kol": "Medical-affairs scientific role review only",
            "open_payments": "Transparency review only",
            "kmeans": "Post-gate engagement pattern only",
            "commercial_policy": "Account and HCP action eligibility",
        },
        "outputs": output_contracts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def print_summary(results: Mapping[str, pd.DataFrame]) -> None:
    """Print the headline results used by the chapter."""

    accounts = results["account_targets"]
    calls = results["call_plan"]
    selected_k = int(
        results["cluster_evaluation"].loc[
            results["cluster_evaluation"]["selected"], "k"
        ].iloc[0]
    )
    print(f"Eligible-roster patients: {results['patient_hcp']['patient_id'].nunique():,}")
    print(f"Eligible HCPs with attributed patients: {results['hcp_features']['npi'].nunique():,}")
    print(f"Eligible accounts: {accounts['account_id'].nunique():,}")
    print(f"Qualifying referral episodes: {len(results['referral_episodes']):,}")
    print(f"KOL role candidates: {int(results['kol_profiles']['kol_candidate'].sum()):,}")
    print(f"Selected k-means clusters: {selected_k}")
    print("\nAccount actions:")
    print(accounts["account_action"].value_counts().to_string())
    print(f"\nPlanned HCPs: {calls['npi'].nunique():,}")
    print(f"Recommended calls: {calls['recommended_calls'].sum():,}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "ch06_hcp" / "assets" / "generated_outputs"
    analysis = run_analysis(root)
    write_outputs(analysis, output, root)
    print_summary(analysis)
    print(f"\nWrote Chapter 6 outputs to {output.relative_to(root)}")
