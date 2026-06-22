"""Generate Chapter 7 supplemental data."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ch07_competitive.generation_modules.synthetic import generate  # noqa: E402


if __name__ == "__main__":
    output = ROOT / "ch07_competitive" / "data" / "generated"
    tables = generate(ROOT, output)
    print("Chapter 7 supplemental data")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame):,} rows")
    print(f"Wrote Chapter 7-only data to {output.relative_to(ROOT)}")
