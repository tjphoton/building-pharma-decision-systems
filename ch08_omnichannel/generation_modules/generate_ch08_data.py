"""Command-line entry point for omnichannel supplemental data."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from synthetic import generate  # noqa: E402


def main() -> None:
    output = ROOT / "ch08_omnichannel" / "data" / "generated"
    counts = generate(ROOT, output)
    print("Omnichannel supplemental data")
    for filename, rows in counts.items():
        print(f"  {filename.removesuffix('.csv')}: {rows:,} rows")
    print(f"Wrote omnichannel data to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
