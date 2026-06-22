"""Generate isolated Chapter 6 supplemental data."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = ROOT / "ch06_hcp" / "generation_modules"
sys.path.insert(0, str(GENERATION_DIR))

from synthetic import generate  # noqa: E402


if __name__ == "__main__":
    output = ROOT / "ch06_hcp" / "data" / "generated"
    tables = generate(ROOT, output)
    print("Chapter 6 supplemental data")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame):,} rows")
    print(f"Wrote Chapter 6-only data to {output.relative_to(ROOT)}")
