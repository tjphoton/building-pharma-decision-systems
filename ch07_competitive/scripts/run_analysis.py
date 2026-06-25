"""Run the complete Chapter 7 competitive access analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from ch07_competitive.generation_modules.ch07_config import (  # noqa: E402
    ADOPTION_BENCHMARK,
    ADOPTION_PROBABILITY_THRESHOLD,
    ACCOUNT_RESTRICTED_PATIENT_THRESHOLD,
    ANALYSIS_DATE,
    BRAND,
    COMPETITORS,
    DATA_CUTOFF,
    DECISION_RULE_VERSION,
    FRICTION_THRESHOLD,
    ITS_CHANGE_WEEK,
    ITS_EFFECT_WEEK,
    ITS_TREATED_PAYER,
    MIN_ACCOUNT_PATIENTS,
    MIN_ACCOUNT_TREATED,
    MIN_SEGMENT_TREATED,
    RESTRICTED_LIVES_THRESHOLD,
    SOURCE_MATURITY_DATE,
    STARTING_REGIMEN_DAYS,
    WASHOUT_DAYS,
)
from ch07_competitive.scripts.accounts import build_account_actions  # noqa: E402
from ch07_competitive.scripts.cohort import (  # noqa: E402
    build_competitive_starts,
    build_switch_evidence,
)
from ch07_competitive.scripts.decomposition import (  # noqa: E402
    payer_region_decisions,
)
from ch07_competitive.scripts.formulary_event import (  # noqa: E402
    controlled_its,
    standardized_cusum,
    synthetic_control,
)
from ch07_competitive.scripts.policy import (  # noqa: E402
    build_policy,
    covered_lives_summary,
    relative_position,
    restriction_lives,
)
from ch07_competitive.scripts.transactions import (  # noqa: E402
    build_attempts,
    friction_summary,
    patient_friction,
)


def input_paths(repo_root: Path) -> dict[str, Path]:
    """Return authoritative Chapter 3 through 7 inputs."""

    upstream = repo_root / "ch03_data" / "output_data" / "generated_data"
    journey = repo_root / "ch05_journey" / "assets" / "generated_outputs"
    hcp = repo_root / "ch06_hcp" / "assets" / "generated_outputs"
    generated = repo_root / "ch07_competitive" / "data" / "generated"
    return {
        "access": generated / "access_history.csv",
        "formulary": upstream / "formulary" / "formulary_status.csv",
        "pharmacy": upstream / "claims_pharmacy" / "pharmacy_claims.csv",
        "ndc_codes": upstream / "reference" / "ndc_codes.csv",
        "hub": upstream / "specialty_pharmacy" / "sp_events.csv",
        "lines": journey / "lines.csv",
        "initiators": journey / "initiators.csv",
        "journeys": journey / "journeys.csv",
        "patient_hcp": hcp / "patient_hcp.csv",
        "account_targets": hcp / "account_targets.csv",
        "enrollment": generated / "plan_region_enrollment.csv",
        "event_panel": generated / "formulary_event_panel.csv",
        "chapter7_manifest": generated / "manifest.json",
    }


def _require_inputs(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Chapter 7 inputs are missing. Generate upstream chapters and run "
            "`uv run python ch07_competitive/generation_modules/generate_ch07_data.py`:\n"
            + "\n".join(missing)
        )


def load_inputs(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Load read-only upstream files and Chapter 7 supplemental data."""

    paths = input_paths(repo_root)
    _require_inputs(paths)
    return {
        "access": pd.read_csv(
            paths["access"], parse_dates=["effective_start", "effective_end"]
        ),
        "formulary": pd.read_csv(
            paths["formulary"], parse_dates=["effective_start", "effective_end"]
        ),
        "pharmacy": pd.read_csv(
            paths["pharmacy"],
            dtype={
                "ndc": str,
                "ndc_prescribed": str,
                "reject_code": str,
                "prescriber_npi": str,
            },
            parse_dates=["date_of_service", "rx_written_date"],
        ),
        "ndc_codes": pd.read_csv(paths["ndc_codes"], dtype={"ndc": str}),
        "hub": pd.read_csv(
            paths["hub"],
            dtype={"prescriber_npi": str},
            parse_dates=["referral_date", "status_date", "ship_date"],
        ),
        "lines": pd.read_csv(paths["lines"], parse_dates=["line_start", "line_end"]),
        "initiators": pd.read_csv(paths["initiators"], parse_dates=["therapy_index"]),
        "journeys": pd.read_csv(
            paths["journeys"],
            parse_dates=["index_date", "followup_end", "first_treatment_date"],
        ),
        "patient_hcp": pd.read_csv(
            paths["patient_hcp"],
            dtype={"npi": str},
            parse_dates=["analysis_date", "index_date", "followup_end"],
        ),
        "account_targets": pd.read_csv(paths["account_targets"]),
        "enrollment": pd.read_csv(paths["enrollment"], parse_dates=["as_of_date"]),
        "event_panel": pd.read_csv(paths["event_panel"], parse_dates=["week_start"]),
    }


def run_analysis(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Return the complete Chapter 7 evidence and decision package."""

    inputs = load_inputs(repo_root)
    corrected_line1, start_segments, source_of_business, transitions = (
        build_competitive_starts(
            inputs["lines"],
            inputs["journeys"],
            inputs["initiators"],
            brand=BRAND,
        )
    )
    switch_evidence = build_switch_evidence(corrected_line1)
    policy = build_policy(
        inputs["access"],
        inputs["formulary"],
        inputs["enrollment"],
        pd.Timestamp(ANALYSIS_DATE),
    )
    lives = covered_lives_summary(policy, brand=BRAND)
    restrictions = restriction_lives(policy, brand=BRAND)
    relative = relative_position(policy, brand=BRAND)

    attempts = build_attempts(
        inputs["pharmacy"],
        inputs["ndc_codes"],
        inputs["journeys"],
        inputs["hub"],
    )
    friction = friction_summary(attempts, brand=BRAND)
    patient_attempts = patient_friction(attempts, brand=BRAND)

    decisions = payer_region_decisions(
        start_segments,
        policy,
        friction,
        brand=BRAND,
        benchmark=ADOPTION_BENCHMARK,
        min_treated=MIN_SEGMENT_TREATED,
        posterior_threshold=ADOPTION_PROBABILITY_THRESHOLD,
        restricted_lives_threshold=RESTRICTED_LIVES_THRESHOLD,
        friction_threshold=FRICTION_THRESHOLD,
        rule_version=DECISION_RULE_VERSION,
        analysis_date=ANALYSIS_DATE,
    )
    brand_policy = policy.loc[policy["product_name"].eq(BRAND)]
    account_actions, account_payer_queue = build_account_actions(
        inputs["patient_hcp"],
        inputs["account_targets"],
        corrected_line1,
        brand_policy,
        patient_attempts,
        brand=BRAND,
        benchmark=ADOPTION_BENCHMARK,
        min_patients=MIN_ACCOUNT_PATIENTS,
        min_treated=MIN_ACCOUNT_TREATED,
        posterior_threshold=ADOPTION_PROBABILITY_THRESHOLD,
        restricted_threshold=ACCOUNT_RESTRICTED_PATIENT_THRESHOLD,
        friction_threshold=FRICTION_THRESHOLD,
        analysis_date=ANALYSIS_DATE,
        rule_version=DECISION_RULE_VERSION,
    )

    its_coef, its_fitted, its_summary = controlled_its(
        inputs["event_panel"],
        treated_payer=ITS_TREATED_PAYER,
        change_week=ITS_CHANGE_WEEK,
        effect_week=ITS_EFFECT_WEEK,
    )
    sc_result, sc_diagnostics = synthetic_control(
        inputs["event_panel"],
        treated_payer=ITS_TREATED_PAYER,
        change_week=ITS_CHANGE_WEEK,
    )
    treated_series = (
        inputs["event_panel"]
        .loc[inputs["event_panel"]["payer_id"].eq(ITS_TREATED_PAYER)]
        .sort_values("week")["brand_share"]
        .reset_index(drop=True)
    )
    cusum = standardized_cusum(treated_series)

    pat_trace = attempts.loc[
        attempts["patient_id"].eq("PAT02034") & attempts["product_name"].eq(BRAND),
        [
            "patient_id",
            "payer_id",
            "region",
            "fill_number",
            "first_submission_date",
            "last_transaction_date",
            "transaction_rows",
            "had_pend",
            "had_reversal",
            "final_outcome",
            "days_to_paid",
            "hub_status",
            "dispense_status",
        ],
    ].copy()

    headline = pd.DataFrame(
        [
            {
                "new_to_therapy_patients": len(corrected_line1),
                "roventra_new_starts": int(corrected_line1["brand_start"].sum()),
                "restricted_lives": int(
                    brand_policy.loc[
                        brand_policy["material_access_barrier"], "enrolled_lives"
                    ].sum()
                ),
                "total_lives": int(restrictions["enrolled_lives"].sum()),
                "payer_region_adoption_flags": int(decisions["adoption_flag"].sum()),
                "payer_region_access_flags": int(decisions["access_flag"].sum()),
            }
        ]
    )

    return {
        "headline": headline,
        "corrected_line1": corrected_line1,
        "competitive_start_evidence": start_segments,
        "source_of_business": source_of_business,
        "treatment_transitions": transitions,
        "switch_evidence": switch_evidence,
        "access_history": inputs["access"],
        "policy_landscape": policy,
        "covered_lives_summary": lives,
        "restriction_lives": restrictions,
        "relative_position": relative,
        "prescription_attempts": attempts,
        "access_friction": friction,
        "patient_friction": patient_attempts,
        "payer_region_decisions": decisions,
        "account_access_adoption_actions": account_actions,
        "account_payer_queue": account_payer_queue,
        "formulary_event_coefficients": its_coef,
        "formulary_event_fitted": its_fitted,
        "formulary_event_effect": its_summary,
        "synthetic_control": sc_result,
        "synthetic_control_diagnostics": sc_diagnostics,
        "changepoint_alerts": cusum,
        "pat02034_attempt_trace": pat_trace,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    repo_root: Path,
) -> None:
    """Write derived CSVs and a complete analysis manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            path = output_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            outputs[name] = {
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": _sha256(path),
            }

    source_hashes = {
        name: _sha256(path)
        for name, path in input_paths(repo_root).items()
        if path.exists()
    }
    manifest = {
        "chapter": 7,
        "brand": BRAND,
        "competitors": list(COMPETITORS),
        "analysis_date": ANALYSIS_DATE,
        "data_cutoff": DATA_CUTOFF,
        "source_maturity_date": SOURCE_MATURITY_DATE,
        "cohort_rule": {
            "washout_days": WASHOUT_DAYS,
            "starting_regimen_days": STARTING_REGIMEN_DAYS,
            "source": "Chapter 5 corrected lines.csv",
        },
        "decision_rule_version": DECISION_RULE_VERSION,
        "source_types": {
            "chapter3": "synthetic source records",
            "chapter5": "derived synthetic journey outputs",
            "chapter6": "derived synthetic HCP-account outputs",
            "chapter7": "synthetic supplemental denominators and planted event",
            "outputs": "derived",
        },
        "outputs": outputs,
        "source_hashes": source_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def print_summary(results: Mapping[str, pd.DataFrame]) -> None:
    """Print a compact, stable run summary."""

    headline = results["headline"].iloc[0]
    actions = results["payer_region_decisions"]["action"].value_counts()
    accounts = results["account_access_adoption_actions"]["action"].value_counts()
    event = results["formulary_event_effect"].iloc[0]
    print("Chapter 7 evidence package")
    print(f"  New-to-therapy patients: {int(headline.new_to_therapy_patients):,}")
    print(f"  Roventra new starts: {int(headline.roventra_new_starts):,}")
    print(
        f"  Restricted lives: {int(headline.restricted_lives):,} "
        f"of {int(headline.total_lives):,}"
    )
    print("  Payer-region actions:")
    for action, count in actions.items():
        print(f"    {action}: {count}")
    print("  Account actions:")
    for action, count in accounts.items():
        print(f"    {action}: {count}")
    print(
        f"  PAY004 effect at week {int(event.effect_week)}: {event.effect_at_week:+.3f}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    results = run_analysis(root)
    target = root / "ch07_competitive" / "assets" / "generated_outputs"
    write_outputs(results, target, root)
    print_summary(results)
    print(f"Wrote Chapter 7 outputs to {target}")
