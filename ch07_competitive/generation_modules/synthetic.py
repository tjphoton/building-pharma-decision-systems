"""Generate isolated Chapter 7 supplemental data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ch07_competitive.generation_modules.ch07_config import (
    ACCESS_HISTORY_PAYER,
    ACCESS_HISTORY_REGION,
    ANALYSIS_DATE,
    BRAND,
    DATA_CUTOFF,
    GENERATOR_VERSION,
    ITS_CHANGE_WEEK,
    ITS_DONORS,
    ITS_LEVEL_EFFECT,
    ITS_SLOPE_EFFECT,
    ITS_TREATED_PAYER,
    ITS_TREATED_REGION,
    SEED,
    SOURCE_MATURITY_DATE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(repo_root: Path) -> dict[str, Path]:
    upstream = repo_root / "ch03_data" / "output_data" / "generated_data"
    journey = repo_root / "ch05_journey" / "assets" / "generated_outputs"
    hcp = repo_root / "ch06_hcp" / "assets" / "generated_outputs"
    return {
        "payers": upstream / "reference" / "payers.csv",
        "access": upstream / "market_access" / "market_access_rules.csv",
        "lines": journey / "lines.csv",
        "journeys": journey / "journeys.csv",
        "patient_hcp": hcp / "patient_hcp.csv",
    }


def _plan_region_enrollment(
    payers: pd.DataFrame,
    access: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create plan-region enrollment denominators for the synthetic case."""

    cells = (
        access.loc[
            access["product_name"].eq(BRAND), ["payer_id", "payer_type", "region"]
        ]
        .drop_duplicates()
        .merge(payers[["payer_id", "payer_name"]], on="payer_id", how="left")
        .sort_values(["payer_id", "region"])
        .reset_index(drop=True)
    )
    base = {
        "Commercial": 420_000,
        "Medicare Advantage": 260_000,
        "Medicare Part D": 300_000,
        "Medicaid": 340_000,
    }
    multipliers = rng.uniform(0.72, 1.28, len(cells))
    cells["enrolled_lives"] = [
        int(round(base.get(pt, 300_000) * mult / 1000) * 1000)
        for pt, mult in zip(cells["payer_type"], multipliers, strict=True)
    ]
    cells["as_of_date"] = ANALYSIS_DATE
    cells["source"] = "Synthetic Chapter 7 plan-region enrollment"
    return cells


def _access_history(access: pd.DataFrame) -> pd.DataFrame:
    """Expand one cell into an effective-dated access history.

    Every cell passes through with its single source record. The demonstration
    cell gains two earlier records so the as-of-date selection has something to
    choose among: covered in January, step edit in July, non-covered in October.
    The record in force on the analysis date is the original source row, so the
    landscape on that date, and every downstream measure, is unchanged.
    """

    is_demo = (
        access["payer_id"].eq(ACCESS_HISTORY_PAYER)
        & access["region"].eq(ACCESS_HISTORY_REGION)
        & access["product_name"].eq(BRAND)
    )
    others = access.loc[~is_demo].copy()
    active = access.loc[is_demo].iloc[0].to_dict()
    history = [
        {
            **active,
            "access_rule_id": f"{active['access_rule_id']}-H1",
            "coverage_status": "Covered",
            "prior_authorization": "No",
            "step_edit": "No",
            "specialty_pharmacy_required": "No",
            "effective_start": "2024-01-01",
            "effective_end": "2024-06-30",
        },
        {
            **active,
            "access_rule_id": f"{active['access_rule_id']}-H2",
            "coverage_status": "Covered",
            "prior_authorization": "No",
            "step_edit": "Yes",
            "specialty_pharmacy_required": "No",
            "effective_start": "2024-07-01",
            "effective_end": "2024-09-30",
        },
        {**active, "effective_start": "2024-10-01", "effective_end": "2025-12-31"},
    ]
    out = pd.concat(
        [others, pd.DataFrame(history)[access.columns]], ignore_index=True
    )
    return out.sort_values(
        ["payer_id", "region", "product_name", "effective_start"]
    ).reset_index(drop=True)


def _event_panel(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a controlled weekly formulary-event panel.

    PAY004 Northeast receives a planted level and slope improvement at week 17.
    Three independent donor payer series share the market trend and seasonality
    without receiving the event.
    """

    weeks = np.arange(1, 53)
    common = 0.225 - 0.00025 * weeks + 0.008 * np.sin(2 * np.pi * weeks / 13)
    payer_offsets = {
        ITS_TREATED_PAYER: 0.003,
        ITS_DONORS[0]: -0.006,
        ITS_DONORS[1]: 0.004,
        ITS_DONORS[2]: -0.001,
    }
    regions = {
        ITS_TREATED_PAYER: ITS_TREATED_REGION,
        ITS_DONORS[0]: "South",
        ITS_DONORS[1]: "West",
        ITS_DONORS[2]: "Midwest",
    }
    rows: list[dict] = []
    for payer_id, offset in payer_offsets.items():
        ar_error = 0.0
        for week, base_share in zip(weeks, common, strict=True):
            ar_error = 0.35 * ar_error + rng.normal(0, 0.006)
            post = payer_id == ITS_TREATED_PAYER and week >= ITS_CHANGE_WEEK
            time_after = max(0, week - ITS_CHANGE_WEEK)
            effect = ITS_LEVEL_EFFECT + ITS_SLOPE_EFFECT * time_after if post else 0.0
            probability = float(
                np.clip(base_share + offset + effect + ar_error, 0.08, 0.60)
            )
            class_starts = int(rng.integers(135, 225))
            brand_starts = int(rng.binomial(class_starts, probability))
            rows.append(
                {
                    "payer_id": payer_id,
                    "region": regions[payer_id],
                    "week": int(week),
                    "week_start": (
                        pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=int(week - 1))
                    ).date(),
                    "brand_starts": brand_starts,
                    "class_starts": class_starts,
                    "brand_share": brand_starts / class_starts,
                    "treated_event": payer_id == ITS_TREATED_PAYER,
                    "post_event": bool(post),
                }
            )
    return pd.DataFrame(rows)


def generate(repo_root: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Generate Chapter 7-only tables and write a provenance manifest."""

    paths = source_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing upstream files:\n" + "\n".join(missing))

    rng = np.random.default_rng(SEED)
    payers = pd.read_csv(paths["payers"])
    access = pd.read_csv(paths["access"])
    tables = {
        "access_history": _access_history(access),
        "plan_region_enrollment": _plan_region_enrollment(payers, access, rng),
        "formulary_event_panel": _event_panel(rng),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = {
            "rows": len(frame),
            "date_range": (
                [str(frame["week_start"].min()), str(frame["week_start"].max())]
                if "week_start" in frame
                else [ANALYSIS_DATE, ANALYSIS_DATE]
            ),
            "sha256": _sha256(path),
        }

    manifest = {
        "chapter": 7,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "analysis_date": ANALYSIS_DATE,
        "data_cutoff": DATA_CUTOFF,
        "source_maturity_date": SOURCE_MATURITY_DATE,
        "source_types": {
            "access_history": "synthetic effective-dated access with planted history",
            "plan_region_enrollment": "synthetic",
            "formulary_event_panel": "synthetic planted-effect panel",
            "upstream_claims": "synthetic",
            "derived_outputs": "derived",
        },
        "planted_event": {
            "payer_id": ITS_TREATED_PAYER,
            "region": ITS_TREATED_REGION,
            "change_week": ITS_CHANGE_WEEK,
            "level_effect": ITS_LEVEL_EFFECT,
            "slope_effect_per_week": ITS_SLOPE_EFFECT,
            "donors": list(ITS_DONORS),
        },
        "outputs": outputs,
        "source_hashes": {name: _sha256(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tables
