#!/usr/bin/env python3
"""Plot the Chapter 3 medical-claim receipt-lag distribution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_quality import claim_lag_analysis
from visual_style import BLUE_DARK, GOLD, GOLD_DARK, INK, MUTED, WHITE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation_modules.entities import validate_manifest_contract  # noqa: E402

DATA_PATH = ROOT / "output_data" / "generated_data" / "claims_medical" / "medical_claims.csv"
FIGURE_DIR = ROOT / "assets" / "figures"
ANALYSIS_DIR = ROOT / "output_data" / "analysis_results" / "data_quality"


def normalize_svg(path: Path) -> None:
    """Remove renderer-added trailing whitespace from an SVG."""
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def create_claim_lag_figure(medical_claims: pd.DataFrame) -> pd.DataFrame:
    summary = claim_lag_analysis(medical_claims)
    counts = summary["claim_count"].astype(int)
    percents = summary["percent"].astype(float)
    labels = summary["lag_bucket"].astype(str)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    bars = ax.bar(
        labels,
        percents,
        color=[BLUE_DARK, "#77A9CC", GOLD, "#E8B55A"],
        edgecolor=[BLUE_DARK, BLUE_DARK, GOLD_DARK, GOLD_DARK],
        linewidth=1.2,
        width=0.66,
    )
    ax.set_ylabel("Share of medical claim rows (%)", fontsize=13, labelpad=12)
    ax.set_ylim(0, max(percents) * 1.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA6B2")
    ax.grid(axis="y", color="#DCE3EA", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12, length=0, pad=10)
    ax.tick_params(axis="y", labelsize=11, length=0)
    for bar, count, percent in zip(bars, counts, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(percents) * 0.025,
            f"{percent:.1f}%\n({count:,} rows)",
            ha="center",
            va="bottom",
            fontsize=11.5,
            weight="bold",
            color=INK,
        )

    fig.suptitle(
        "Medical Claim Receipt Lag Is Right-Skewed",
        x=0.5,
        y=0.96,
        ha="center",
        fontsize=22,
        weight="bold",
    )
    fig.tight_layout(rect=[0.055, 0.03, 0.98, 0.91])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        path = FIGURE_DIR / f"figure-3-7-claim-receipt-lag-distribution.{suffix}"
        fig.savefig(
            path,
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        if suffix == "svg":
            normalize_svg(path)
    plt.close(fig)
    return summary


def main() -> None:
    if not DATA_PATH.exists():
        print("Generated medical claims not found. Run generate_all_synthetic_data.py first.")
        sys.exit(1)
    validate_manifest_contract(DATA_PATH.parents[1])
    medical_claims = pd.read_csv(DATA_PATH)
    summary = create_claim_lag_figure(medical_claims)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ANALYSIS_DIR / "dq_claim_lag.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote claim-lag figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
