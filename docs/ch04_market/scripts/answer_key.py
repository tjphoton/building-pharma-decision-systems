"""Reproduce the Chapter 3 answer key for Chapter 4 validation.

Chapter 3's generator assigns every synthetic patient a ``condition_bucket`` that
records their true underlying condition, then deliberately strips that column
before writing ``patients.csv``: a clean teaching dataset should not ship its own
answer key. Chapter 4 needs that key to measure how wrong each claims-based market
estimate is. We reproduce it here from the same generator and seed, so no Chapter 3
file changes and the key is fully deterministic.

Run once from the repository root::

    uv run python ch04_market/scripts/answer_key.py

which writes ``ch04_market/output_data/answer_key.csv`` (one row per patient).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# The default seed and patient count used by the Chapter 3 generator
# (ch03_data/scripts/generate_all_synthetic_data.py).
CHAPTER3_SEED = 20260609
CHAPTER3_PATIENTS = 20_000

# ICD-10 codes that the Chapter 3 generator treats as the launch condition
# (generation_modules/claims.py, CLAIM_CODES_BY_BUCKET["Launch condition"]).
LAUNCH_CONDITION_CODES = ["E11.9", "E11.65", "E11.40"]

# The fictional launch product after the LaunchRx -> Roventra migration.
LAUNCH_PRODUCT = "Roventra"


def build_answer_key(repo_root: Path) -> pd.DataFrame:
    """Return one row per patient with the true condition bucket.

    Re-runs the first two deterministic steps of the Chapter 3 generator
    (``build_entities`` then ``apply_canonical_overrides``). These are the only
    consumers of the seed before patient identities are fixed, so the resulting
    ``patient_id`` order matches the written ``patients.csv`` exactly.
    """

    sys.path.insert(0, str(repo_root / "ch03_data"))
    from generation_modules.entities import apply_canonical_overrides, build_entities

    rng = random.Random(CHAPTER3_SEED)
    bundle = build_entities(rng, None, None, n_patients=CHAPTER3_PATIENTS)
    apply_canonical_overrides(bundle)

    return pd.DataFrame(
        [
            {
                "patient_id": patient["patient_id"],
                "condition_bucket": patient["condition_bucket"],
                "true_launch_condition": patient["condition_bucket"] == "Launch condition",
            }
            for patient in bundle.patients
        ]
    )


def load_answer_key(repo_root: Path) -> pd.DataFrame:
    """Load the cached answer key, building it on first use."""

    path = repo_root / "ch04_market" / "output_data" / "answer_key.csv"
    if not path.exists():
        key = build_answer_key(repo_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        key.to_csv(path, index=False)
    return pd.read_csv(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    answer_key = build_answer_key(root)
    destination = root / "ch04_market" / "output_data" / "answer_key.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    answer_key.to_csv(destination, index=False)
    counts = answer_key["condition_bucket"].value_counts()
    print(f"Wrote {len(answer_key):,} patient answer-key rows to {destination}")
    print(f"True launch-condition patients: {int(answer_key['true_launch_condition'].sum()):,}")
    print(counts.to_string())
