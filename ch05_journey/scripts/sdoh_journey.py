"""Chapter 5 SDOH persistence and refill-gap extension."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CH99_SCRIPTS = Path(__file__).resolve().parents[2] / "ch99_SDOH" / "scripts"
sys.path.insert(0, str(CH99_SCRIPTS))

from sdoh_pipeline import (  # noqa: E402
    GOLD,
    GREEN,
    RED,
    account_support_flag,
    build_area_table,
    build_patient_table,
    journey_sdoh_summary,
    persistence_curve,
)


def build_sdoh_journey_outputs() -> dict[str, pd.DataFrame]:
    """Build Chapter 5 SDOH persistence and support-planning tables."""

    areas = build_area_table()
    patients = build_patient_table(areas)
    return {
        "sdoh_journey_summary": journey_sdoh_summary(patients),
        "sdoh_persistence_curve": persistence_curve(patients),
        "sdoh_account_support_flag": account_support_flag(patients),
    }


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    """Save SVG and PNG versions with trimmed whitespace."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = figures_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220, facecolor="white")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def figure_5_16_sdoh_persistence(outputs: Path, figures_dir: Path) -> None:
    """Plot persistence curves by SDOH barrier group."""

    curve = pd.read_csv(outputs / "sdoh_persistence_curve.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {"Low barrier": GREEN, "Middle barrier": GOLD, "High barrier": RED}
    for label in ["Low barrier", "Middle barrier", "High barrier"]:
        data = curve.loc[curve["barrier_group"].eq(label)]
        ax.plot(data["day"], data["persistent_share_pct"], marker="o",
                linewidth=2.0, color=colors[label], label=label)
        end = data.iloc[-1]
        ax.text(end["day"] + 3, end["persistent_share_pct"], label,
                va="center", fontsize=9, color=colors[label])

    ax.set_title("Refill persistence by SDOH barrier group", fontsize=13)
    ax.set_xlabel("Days after treatment start")
    ax.set_ylabel("Patients still persistent (%)")
    ax.set_xlim(0, 138)
    ax.set_ylim(20, 108)
    ax.grid(axis="y", color="#E8ECEF", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, figures_dir, "figure_5_16_sdoh_persistence")
