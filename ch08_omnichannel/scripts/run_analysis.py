"""Run the complete Chapter 8 omnichannel analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = ROOT / "ch08_omnichannel" / "generation_modules"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GENERATION_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from ch08_omnichannel.generation_modules.ch08_config import (  # noqa: E402
    ANALYSIS_DATE,
    CYCLE_END,
    CYCLE_START,
    FIELD_CAPACITY_PER_TERRITORY,
    LOOKBACK_DAYS,
    MODEL_VERSION,
    OUTCOME_DAYS,
    POLICY_VERSION,
    REFRESH_DATE,
    SEED,
)
from ch08_omnichannel.scripts.event_ledger import (  # noqa: E402
    build_event_ledger,
    channel_delivery_summary,
    email_quality_summary,
)
from ch08_omnichannel.scripts.economics import (  # noqa: E402
    channel_affinity_trace,
    channel_economics,
)
from ch08_omnichannel.scripts.features import (  # noqa: E402
    build_snapshot_panel,
    pressure_response_summary,
    response_shrinkage_summary,
    saturation_summary,
)
from ch08_omnichannel.scripts.modeling import fit_response_model  # noqa: E402
from ch08_omnichannel.scripts.modern_methods import (  # noqa: E402
    field_then_digital_contrast,
    off_policy_evaluation,
    off_policy_support,
    sequence_feature_effects,
    sequence_feature_model,
    uplift_diagnostics,
    uplift_ranking_comparison,
    uplift_response_contrast,
    uplift_scatter_data,
    uplift_segment_summary,
)
from ch08_omnichannel.scripts.policy import (  # noqa: E402
    build_channel_plan,
    plan_summary,
)
from ch08_omnichannel.scripts.sequences import (  # noqa: E402
    attribution_comparison,
    markov_attribution,
    observed_sequences,
    sequence_pattern_examples,
    sequence_pattern_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "events": (
            repo_root
            / "ch08_omnichannel"
            / "data"
            / "generated"
            / "engagement_events.csv"
        ),
        "truth": (
            repo_root
            / "ch08_omnichannel"
            / "data"
            / "generated"
            / "engagement_truth.csv"
        ),
        "generation_manifest": (
            repo_root / "ch08_omnichannel" / "data" / "generated" / "manifest.json"
        ),
        "hcp_features": (
            repo_root / "ch06_hcp" / "assets" / "generated_outputs" / "hcp_features.csv"
        ),
        "hcp_segments": (
            repo_root / "ch06_hcp" / "assets" / "generated_outputs" / "hcp_segments.csv"
        ),
        "engagement_signals": (
            repo_root / "ch06_hcp" / "data" / "generated" / "engagement_signals.csv"
        ),
        "account_targets": (
            repo_root
            / "ch06_hcp"
            / "assets"
            / "generated_outputs"
            / "account_targets.csv"
        ),
        "account_actions": (
            repo_root
            / "ch07_competitive"
            / "assets"
            / "generated_outputs"
            / "account_access_adoption_actions.csv"
        ),
    }


def load_inputs(repo_root: Path) -> dict[str, pd.DataFrame]:
    paths = input_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Chapter 8 inputs are missing. Run the Chapter 8 generator first:\n"
            + "\n".join(missing)
        )
    return {
        "events": pd.read_csv(
            paths["events"], dtype={"npi": str}, parse_dates=["event_date"]
        ),
        "truth": pd.read_csv(paths["truth"]),
        "hcp_features": pd.read_csv(
            paths["hcp_features"], dtype={"npi": str}
        ),
        "hcp_segments": pd.read_csv(
            paths["hcp_segments"], dtype={"npi": str}
        ),
        "engagement_signals": pd.read_csv(
            paths["engagement_signals"], dtype={"npi": str}
        ),
        "account_targets": pd.read_csv(paths["account_targets"]),
        "account_actions": pd.read_csv(paths["account_actions"]),
    }


def run_analysis(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Return the complete Chapter 8 evidence and action package."""

    inputs = load_inputs(repo_root)
    ledger = build_event_ledger(inputs["events"])
    channel_summary = channel_delivery_summary(ledger)
    email_summary = email_quality_summary(ledger, inputs["truth"])
    panel = build_snapshot_panel(
        ledger,
        inputs["hcp_features"],
        inputs["hcp_segments"],
        inputs["engagement_signals"],
        inputs["account_targets"],
        inputs["account_actions"],
        LOOKBACK_DAYS,
        OUTCOME_DAYS,
    )
    pressure = pressure_response_summary(panel)
    saturation = saturation_summary(panel)
    model_results = fit_response_model(panel)
    sequence_detail, sequence_summary, transitions = observed_sequences(
        ledger,
        model_results["scored_snapshots"],
        ANALYSIS_DATE,
        LOOKBACK_DAYS,
    )
    attribution = attribution_comparison(ledger)
    markov = markov_attribution(ledger)
    plan = build_channel_plan(
        model_results["scored_snapshots"],
        ANALYSIS_DATE,
        CYCLE_START,
        CYCLE_END,
        REFRESH_DATE,
        POLICY_VERSION,
    )
    capacity_value = capacity_value_summary(plan)
    economics = channel_economics(model_results["scored_snapshots"], markov)
    affinity_trace = channel_affinity_trace(
        model_results["scored_snapshots"],
        plan,
        ANALYSIS_DATE,
        ["9000000522", "9000000567", "9000000406"],
    )
    return {
        "event_ledger": ledger,
        "channel_summary": channel_summary,
        "email_quality": email_summary,
        "snapshot_panel": panel,
        "pressure_response": pressure,
        "saturation": saturation,
        "channel_economics": economics,
        "channel_affinity": affinity_trace,
        **model_results,
        "response_shrinkage": response_shrinkage_summary(
            model_results["scored_snapshots"]
        ),
        "sequence_detail": sequence_detail,
        "sequence_summary": sequence_summary,
        "sequence_patterns": sequence_pattern_summary(sequence_detail),
        "sequence_pattern_examples": sequence_pattern_examples(sequence_detail),
        "transition_summary": transitions,
        "attribution": attribution,
        "markov_attribution": markov,
        "uplift_segment_summary": uplift_segment_summary(
            model_results["scored_snapshots"]
        ),
        "uplift_response_contrast": uplift_response_contrast(
            model_results["scored_snapshots"]
        ),
        "uplift_diagnostics": uplift_diagnostics(
            model_results["scored_snapshots"]
        ),
        "uplift_scatter_data": uplift_scatter_data(
            model_results["scored_snapshots"]
        ),
        "uplift_ranking_comparison": uplift_ranking_comparison(
            model_results["scored_snapshots"]
        ),
        "policy_evaluation": off_policy_evaluation(
            model_results["scored_snapshots"]
        ),
        "policy_support": off_policy_support(
            model_results["scored_snapshots"]
        ),
        "field_then_digital_contrast": field_then_digital_contrast(
            model_results["scored_snapshots"]
        ),
        "sequence_model_comparison": sequence_feature_model(
            model_results["scored_snapshots"]
        ),
        "sequence_feature_effects": sequence_feature_effects(
            model_results["scored_snapshots"]
        ),
        "channel_plan": plan,
        "plan_summary": plan_summary(plan),
        "capacity_value": capacity_value,
    }


def capacity_value_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Compare model-ranked capacity selection with a deterministic random baseline."""

    eligible = plan.loc[plan["promotion_eligible"]].copy()
    selected = eligible.loc[eligible["capacity_selected"]]
    random_baseline = (
        eligible.assign(
            random_rank=eligible.groupby("territory").cumcount() + 1
        )
        .loc[lambda frame: frame["random_rank"].le(FIELD_CAPACITY_PER_TERRITORY)]
        .copy()
    )
    rows = [
        {
            "selection_rule": "model_ranked",
            "relationships": len(selected),
            "expected_responses": selected["predicted_response"].sum(),
            "mean_predicted_response": selected["predicted_response"].mean(),
        },
        {
            "selection_rule": "territory_order_baseline",
            "relationships": len(random_baseline),
            "expected_responses": random_baseline["predicted_response"].sum(),
            "mean_predicted_response": random_baseline[
                "predicted_response"
            ].mean(),
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    repo_root: Path,
) -> None:
    """Write analysis outputs with provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for name, frame in results.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[path.name] = {
            "rows": len(frame),
            "sha256": _sha256(path),
        }
    manifest = {
        "analysis_date": ANALYSIS_DATE.date().isoformat(),
        "cycle_start": CYCLE_START.date().isoformat(),
        "cycle_end": CYCLE_END.date().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "outcome_days": OUTCOME_DAYS,
        "seed": SEED,
        "model_version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths(repo_root).items()
        },
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    results = run_analysis(ROOT)
    output = ROOT / "ch08_omnichannel" / "assets" / "generated_outputs"
    write_outputs(results, output, ROOT)
    metrics = results["model_metrics"].set_index("split")
    test = metrics.loc["test"]
    print("Omnichannel analysis")
    print(f"  Ledger events: {len(results['event_ledger']):,}")
    print(f"  HCP-account snapshots: {len(results['snapshot_panel']):,}")
    print(
        f"  Test AUC: {test.roc_auc:.3f}; "
        f"average precision: {test.average_precision:.3f}; "
        f"Brier score: {test.brier_score:.3f}"
    )
    print(
        f"  Channel plan relationships: {len(results['channel_plan']):,}"
    )
    print(f"Wrote outputs to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
