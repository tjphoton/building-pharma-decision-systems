"""Build the Chapter 5 figures from generated outputs.

Every value is read from the generated data package or the pipeline outputs,
never hard-coded, so the figures regenerate with the data. Run after
run_analysis.py:

    uv run python ch05_journey/scripts/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from episode_construction import DX_COLS, LAUNCH_CONDITION_CODES
from survival import aalen_johansen_curve, km_curve

CHAPTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_DIR.parent / "ch03_data" / "output_data" / "generated_data"
OUT_DIR = CHAPTER_DIR / "assets" / "generated_outputs"
FIG_DIR = CHAPTER_DIR / "assets" / "figures"

# Pastel semantic palette from AGENTS.md visual rules
BLUE = "#7ba6d0"   # source data / observed records
GOLD = "#e0b663"   # rules / calculations
GREEN = "#7fb685"  # analytical outputs
ORANGE = "#f0a06b"  # second competitor
RED = "#d97b6c"    # defects / cautions
GRAY = "#9a9a9a"   # supporting context

TIMELINE_PATIENT = "PAT00839"


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# Figure 5.1 – Cohort eligibility: patient timelines
# ---------------------------------------------------------------------------

def figure_5_1_cohort_eligibility() -> None:
    """Left-panel patient timelines for cohort entry."""

    all_indexed = pd.read_csv(
        OUT_DIR / "all_indexed.csv",
        parse_dates=["coverage_start", "coverage_end", "index_date", "followup_end"],
    )

    # Select 3 visually distinct patients
    passing  = all_indexed.loc[
        all_indexed["lookback_days"].ge(250) & all_indexed["followup_days"].ge(150)
    ].iloc[0]
    short_lb = all_indexed.loc[
        all_indexed["lookback_days"].between(30, 100) & all_indexed["followup_days"].ge(90)
    ].iloc[0]
    short_fu = all_indexed.loc[
        all_indexed["lookback_days"].ge(180) & all_indexed["followup_days"].between(20, 65)
    ].iloc[0]

    patients = [
        (passing,  "Retained",                    GREEN, True,  True),
        (short_lb, "Excluded: lookback too short", RED,   False, True),
        (short_fu, "Excluded: follow-up too short", GOLD, True,  False),
    ]

    fig, ax_left = plt.subplots(figsize=(13.0, 5.5))

    # ---- Left panel: 3 patient timelines ----
    label_x = -665
    for i, (pt, label, color, lb_ok, fu_ok) in enumerate(patients):
        y = 2 - i
        lb_days = int(pt["lookback_days"])
        fu_days = int(pt["followup_days"])

        # Lookback zone
        ax_left.barh(y, lb_days, left=-lb_days, height=0.32,
                     color=BLUE if lb_ok else RED, alpha=0.62,
                     edgecolor="#888888", linewidth=0.5)
        # Follow-up zone
        ax_left.barh(y, fu_days, left=0, height=0.32,
                     color=GREEN if fu_ok else GOLD, alpha=0.62,
                     edgecolor="#888888", linewidth=0.5)
        # Index marker
        ax_left.plot(0, y, marker="|", color="#333333",
                     markersize=11, markeredgewidth=1.8)
        # Labels
        ax_left.text(label_x, y, label.replace(": ", "\n"), ha="left", va="center",
                     fontsize=10, color=color, fontweight="bold")
        ax_left.text((-lb_days / 2) - 25, y + 0.23, f"{lb_days}d lookback",
                     ha="center", fontsize=7.5, color="#555555")
        follow_label_x = fu_days / 2 if fu_days >= 90 else 88
        follow_ha = "center" if fu_days >= 90 else "right"
        ax_left.text(follow_label_x, y + 0.23, f"{fu_days}d follow-up",
                     ha=follow_ha, fontsize=7.5, color="#555555")

    line_bottom = -0.72
    line_top = 2.16
    ax_left.vlines(-180, line_bottom, line_top, color=BLUE,
                   linestyle="--", linewidth=1.1, alpha=0.7)
    ax_left.vlines(90, line_bottom, line_top, color=GREEN,
                   linestyle="--", linewidth=1.1, alpha=0.7)
    ax_left.vlines(0, line_bottom, line_top, color="#666666",
                   linestyle=":", linewidth=0.8)
    ax_left.text(-186, -0.62, "-180d\nthreshold", ha="right",
                 fontsize=8.5, color="#3a6da0")
    ax_left.text( 96, -0.62, "+90d\nthreshold",  ha="left",
                 fontsize=8.5, color="#3a7a50")
    ax_left.text( 4, -0.62, "Diagnosis\nindex", ha="left",
                 fontsize=8.5,  color="#444444")
    ax_left.text(-190, 2.72, "Which patient qualifies?", ha="center",
                 fontsize=13.5, color="#222222", fontweight="bold")
    ax_left.set_xlim(-700, 320)
    ax_left.set_ylim(-0.78, 2.85)
    ax_left.set_yticks([])
    ax_left.set_xlabel("Days relative to diagnosis index", fontsize=11)
    ax_left.spines[["left", "right", "top"]].set_visible(False)

    _save(fig, "figure_5_1_cohort_eligibility")


# ---------------------------------------------------------------------------
# Figure 5.2 - Line-of-therapy rule schematic
# ---------------------------------------------------------------------------

def figure_5_2_treatment_sequence_rules() -> None:
    """Conceptual timelines showing how fill timing becomes line logic."""

    fig, axes = plt.subplots(4, 2, figsize=(13.0, 10.1))
    axes = axes.ravel()

    def setup(ax: plt.Axes, title: str, xlim: tuple[int, int] = (-190, 130)) -> None:
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.08, 1.30)
        ax.axis("off")
        ax.text(xlim[0], 1.24, title, ha="left", va="top",
                fontsize=12.0, fontweight="bold", color="#222222")

    def baseline(ax: plt.Axes, y: float, x0: float, x1: float) -> None:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    lw=1.1, mutation_scale=9))

    def supply(ax: plt.Axes, start: float, end: float, y: float, label: str,
               color: str = BLUE, height: float = 0.12) -> None:
        ax.add_patch(mpatches.Rectangle(
            (start, y - height / 2), end - start, height,
            facecolor=color, edgecolor="#777777", linewidth=0.5, alpha=0.72,
        ))
        ax.text((start + end) / 2, y + 0.14, label, ha="center", va="bottom",
                fontsize=8.8, color="#444444")

    def window(ax: plt.Axes, start: float, end: float, y: float, label: str,
               color: str = GOLD, height: float = 0.07,
               label_x: float | None = None, label_dy: float = -0.13) -> None:
        ax.add_patch(mpatches.Rectangle(
            (start, y - height / 2), end - start, height,
            facecolor=color, edgecolor="none", alpha=0.35,
        ))
        ax.text((start + end) / 2 if label_x is None else label_x, y + label_dy, label, ha="center", va="top",
                fontsize=8.4, color="#7a5a16")

    def marker(ax: plt.Axes, x: float, y: float, label: str,
               color: str = "#333333", above: bool = False) -> None:
        ax.vlines(x, y - 0.18, y + 0.18, color=color, linewidth=1.3)
        va = "bottom" if above else "top"
        dy = 0.22 if above else -0.21
        ax.text(x, y + dy, label, ha="center", va=va,
                fontsize=8.4, color=color)

    def result(ax: plt.Axes, x: float, y: float, text: str,
               color: str = GREEN) -> None:
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="#244a2a",
                bbox=dict(boxstyle="round,pad=0.30", facecolor=color,
                          edgecolor="#5a8c60", linewidth=0.7, alpha=0.28))

    # 1. Therapy index and washout
    ax = axes[0]
    setup(ax, "1. Start a new line")
    baseline(ax, 0.46, -180, 72)
    window(ax, -180, 0, 0.46, "180d washout: no prior basket fill",
           color=GRAY, height=0.08)
    marker(ax, -180, 0.46, "-180d", color="#777777")
    marker(ax, 0, 0.46, "therapy index\nfirst treatment fill", color="#333333")
    supply(ax, 0, 28, 0.64, "Product A\n28d supply")
    result(ax, 54, 0.67, "Line 1 = A")

    # 2. Same product refill before the gap closes
    ax = axes[1]
    setup(ax, "2. Refill before the gap closes", xlim=(-10, 120))
    baseline(ax, 0.46, 0, 112)
    supply(ax, 0, 28, 0.64, "A supply")
    window(ax, 28, 88, 0.46, "60d allowable gap", height=0.08, label_x=55)
    marker(ax, 70, 0.46, "A refill", color="#3a6da0")
    supply(ax, 70, 98, 0.64, "A supply")
    result(ax, 102, 0.82, "same line")

    # 3. Combination window
    ax = axes[2]
    setup(ax, "3. Product inside 30 days", xlim=(-10, 115))
    baseline(ax, 0.46, 0, 106)
    window(ax, 0, 30, 0.34, "30d regimen window", height=0.08)
    ax.add_patch(mpatches.Rectangle(
        (0, 0.62 - 0.12 / 2), 28, 0.12,
        facecolor=BLUE, edgecolor="#777777", linewidth=0.5, alpha=0.72,
    ))
    ax.text(1, 0.70, "A supply", ha="left", va="bottom",
            fontsize=8.8, color="#444444")
    ax.vlines(18, 0.38, 0.80, color="#2f6f45", linewidth=1.3)
    ax.text(22, 0.76, "B fill", ha="left", va="center",
            fontsize=8.4, color="#2f6f45",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.2))
    supply(ax, 18, 46, 0.88, "B supply", color=GREEN)
    result(ax, 82, 0.68, "Line 1 = A + B")

    # 4. Addition after the regimen window while A is still active
    ax = axes[3]
    setup(ax, "4. New product while A is active", xlim=(-10, 125))
    baseline(ax, 0.46, 0, 116)
    window(ax, 0, 30, 0.34, "30d window", height=0.07)
    supply(ax, 0, 28, 0.62, "A")
    supply(ax, 25, 53, 0.62, "A refill")
    marker(ax, 45, 0.46, "B fill", color="#2f6f45")
    supply(ax, 45, 73, 0.78, "B", color=GREEN)
    result(ax, 96, 0.68, "Addition:\nLine 2 = A + B")

    # 5. Switch after active supply is gone
    ax = axes[4]
    setup(ax, "5. New product after A supply ends", xlim=(-10, 120))
    baseline(ax, 0.46, 0, 112)
    supply(ax, 0, 28, 0.64, "A supply")
    window(ax, 28, 88, 0.46, "within 60d gap", height=0.08, label_dy=-0.18)
    ax.vlines(55, 0.32, 0.58, color="#2f6f45", linewidth=1.3)
    ax.text(58, 0.59, "B fill", ha="left", va="center",
            fontsize=8.4, color="#2f6f45",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.2))
    supply(ax, 55, 83, 0.78, "B supply", color=GREEN)
    result(ax, 100, 0.68, "Switch:\nLine 2 = B")

    # 6. Restart after the gap closes
    ax = axes[5]
    setup(ax, "6. Restart after the gap closes", xlim=(-10, 125))
    baseline(ax, 0.46, 0, 116)
    supply(ax, 0, 28, 0.64, "A supply")
    window(ax, 28, 88, 0.46, "60d gap", height=0.08, label_dy=-0.18)
    marker(ax, 88, 0.46, "gap closes", color="#8a5a00")
    marker(ax, 102, 0.46, "A fill", color="#3a6da0")
    supply(ax, 102, 120, 0.64, "A", color=BLUE)
    result(ax, 98, 0.92, "Restart:\nLine 2 = A")

    # 7. Censoring when observation ends before the gap resolves
    ax = axes[6]
    setup(ax, "7. Observation ends before gap closes", xlim=(-10, 125))
    baseline(ax, 0.46, 0, 116)
    supply(ax, 0, 28, 0.64, "A supply")
    window(ax, 28, 88, 0.46, "gap still open", height=0.08, label_dy=-0.18)
    ax.vlines(70, 0.28, 0.64, color="#666666", linewidth=1.3)
    ax.text(72, 0.28, "observation ends", ha="left", va="top",
            fontsize=8.4, color="#666666")
    result(ax, 104, 0.70, "Censored", color=GRAY)

    # 8. Discontinuation after the gap closes with no restart
    ax = axes[7]
    setup(ax, "8. Discontinuation after the gap closes", xlim=(-10, 125))
    baseline(ax, 0.46, 0, 116)
    supply(ax, 0, 28, 0.64, "A supply")
    window(ax, 28, 88, 0.46, "60d gap", height=0.08, label_dy=-0.18)
    marker(ax, 88, 0.46, "gap closes", color="#8a5a00")
    marker(ax, 112, 0.46, "still observed", color="#666666")
    result(ax, 100, 0.82, "Discontinued", color=RED)

    fig.text(0.5, 0.985, "Line of therapy treatment sequence rules",
             ha="center", va="top", fontsize=16.5, fontweight="bold", color="#222222")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.03,
                        hspace=0.36, wspace=0.20)

    _save(fig, "figure_5_2_treatment_sequence_rules")


# ---------------------------------------------------------------------------
# Figure 5.3 - PAT00839 switch example
# ---------------------------------------------------------------------------

def figure_5_3_switch_example() -> None:
    """Patient-specific switch example with cleaner dated sequencing."""

    lines = pd.read_csv(OUT_DIR / "lines.csv", parse_dates=["line_start", "line_end"])
    medical = pd.read_csv(
        DATA_DIR / "claims_medical" / "medical_claims_mature.csv",
        parse_dates=["claim_date"],
    )
    pharmacy = pd.read_csv(
        DATA_DIR / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
        parse_dates=["date_of_service"],
    )
    ndc_codes = pd.read_csv(DATA_DIR / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    basket = set(pd.read_csv(DATA_DIR / "reference" / "products.csv")["product_name"])
    pharmacy["product_name"] = pharmacy["ndc_prescribed"].map(
        ndc_codes.set_index("ndc")["drug_name"]
    )

    pid = "PAT00839"
    patient_lines = lines.loc[lines["patient_id"].eq(pid)].sort_values("line_number")
    patient_fills = pharmacy.loc[
        pharmacy["patient_id"].eq(pid) & pharmacy["product_name"].isin(basket)
    ].sort_values("date_of_service")
    dx_mask = (
        medical["patient_id"].eq(pid)
        & medical[DX_COLS].isin(set(LAUNCH_CONDITION_CODES)).any(axis=1)
    )
    diagnosis_index = medical.loc[dx_mask, "claim_date"].min()
    therapy_index = patient_fills.loc[
        patient_fills["transaction_type"].eq("PAID"), "date_of_service"
    ].min()
    line1 = patient_lines.iloc[0]
    line2 = patient_lines.iloc[1]
    switch_date = line2["line_start"]
    line2_gap_end = line2["line_end"] + pd.Timedelta(days=60)
    observation_end = pd.Timestamp("2024-12-31")

    fig, ax = plt.subplots(figsize=(13.0, 5.5))
    start = diagnosis_index - pd.Timedelta(days=10)
    end = observation_end + pd.Timedelta(days=10)
    x = mdates.date2num
    ax.set_xlim(x(start), x(end))
    ax.set_ylim(0.0, 4.3)
    ax.set_yticks([3.35, 2.45, 1.45, 0.65])
    ax.set_yticklabels(["Timeline", "Claims", "Line 1", "Line 2"], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)

    for y in [3.35, 2.45, 1.45, 0.65]:
        ax.hlines(y, x(start), x(end), color="#c8c8c8", linewidth=1.0, zorder=0)

    def badge(xpos: float, ypos: float, text: str, face: str, edge: str = "#444444", txt: str = "white") -> None:
        ax.text(
            xpos,
            ypos,
            text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=txt,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=edge, linewidth=1.1),
            zorder=8,
        )

    def bar(y: float, start_date: pd.Timestamp, end_date: pd.Timestamp, color: str, label: str) -> None:
        ax.barh(
            y,
            x(end_date + pd.Timedelta(days=1)) - x(start_date),
            left=x(start_date),
            height=0.34,
            color=color,
            edgecolor="#4d4d4d",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            x(start_date + (end_date - start_date) / 2),
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    def vertical_label(date: pd.Timestamp, y: float, label: str, align: str = "above", color: str = "#444444") -> None:
        xpos = x(date)
        ax.vlines(xpos, y - 0.28, y + 0.28, color=color, linewidth=1.5, zorder=5)
        offset = 0.38 if align == "above" else -0.40
        va = "bottom" if align == "above" else "top"
        ax.text(xpos, y + offset, label, ha="center", va=va, fontsize=12, color=color)

    bar(1.45, line1["line_start"], line1["line_end"], BLUE, "L1 Nexoral")
    bar(0.65, line2["line_start"], line2["line_end"], GREEN, "L2 Vexpro")

    gap_start = line2["line_end"] + pd.Timedelta(days=1)
    ax.add_patch(
        mpatches.Rectangle(
            (x(gap_start), 0.48),
            x(line2_gap_end) - x(gap_start),
            0.18,
            facecolor=GOLD,
            edgecolor="none",
            alpha=0.35,
            zorder=2,
        )
    )
    ax.text(
        x(gap_start + (line2_gap_end - gap_start) / 2),
        0.28,
        "60-day allowable gap",
        ha="center",
        va="center",
        fontsize=11,
        color="#8a5a00",
    )

    vertical_label(diagnosis_index, 3.35, "Diagnosis index")
    vertical_label(therapy_index, 3.35, "Therapy index")
    vertical_label(observation_end, 3.35, "Observation end")

    ax.annotate(
        "",
        xy=(x(therapy_index), 3.95),
        xytext=(x(diagnosis_index), 3.95),
        arrowprops=dict(arrowstyle="<->", color="#666666", linewidth=1.2),
    )
    ax.text(
        x(diagnosis_index + (therapy_index - diagnosis_index) / 2),
        4.06,
        "146 days to first treatment fill",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    paid_rows = patient_fills.loc[patient_fills["transaction_type"].eq("PAID")]
    for i, row in enumerate(paid_rows.itertuples(index=False)):
        color = BLUE if row.product_name == "Nexoral" else GREEN
        ax.scatter(x(row.date_of_service), 2.45, s=170, marker="o", color=color, zorder=6)
        ax.text(
            x(row.date_of_service),
            2.74 if i else 2.70,
            row.product_name,
            ha="center",
            va="bottom",
            fontsize=12,
            color=color,
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(x(switch_date), 0.88),
        xytext=(x(switch_date + pd.Timedelta(days=12)), 1.56),
        arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.2, mutation_scale=10),
    )
    badge(x(switch_date + pd.Timedelta(days=12)), 1.92, "SWITCH", GOLD, txt="#222222")
    badge(x(line2_gap_end + pd.Timedelta(days=32)), 0.65, "DISCONTINUED", RED)

    ax.text(
        x(switch_date + pd.Timedelta(days=12)),
        1.08,
        "Vexpro arrives after Nexoral supply ends.",
        ha="left",
        va="center",
        fontsize=11,
        color="#555555",
    )
    xticks = [
        diagnosis_index,
        therapy_index,
        switch_date,
        line2_gap_end,
        observation_end,
    ]
    ax.set_xticks([x(d) for d in xticks])
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %-d\n%Y") for d in xticks], fontsize=11)
    ax.spines["bottom"].set_color("#cfcfcf")
    ax.set_title("PAT00839 switch example", fontsize=17)
    fig.tight_layout()

    _save(fig, "figure_5_3_switch_example")


# ---------------------------------------------------------------------------
# Figure 5.4 - PAT03874 addition example
# ---------------------------------------------------------------------------

def figure_5_4_addition_example() -> None:
    """Patient-specific addition example with visible overlap and cleaner labels."""

    lines = pd.read_csv(OUT_DIR / "lines.csv", parse_dates=["line_start", "line_end"])
    medical = pd.read_csv(
        DATA_DIR / "claims_medical" / "medical_claims_mature.csv",
        parse_dates=["claim_date"],
    )
    pharmacy = pd.read_csv(
        DATA_DIR / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
        parse_dates=["date_of_service"],
    )
    ndc_codes = pd.read_csv(DATA_DIR / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    basket = set(pd.read_csv(DATA_DIR / "reference" / "products.csv")["product_name"])
    pharmacy["product_name"] = pharmacy["ndc_prescribed"].map(
        ndc_codes.set_index("ndc")["drug_name"]
    )

    pid = "PAT03874"
    patient_lines = lines.loc[lines["patient_id"].eq(pid)].sort_values("line_number")
    patient_fills = pharmacy.loc[
        pharmacy["patient_id"].eq(pid) & pharmacy["product_name"].isin(basket)
    ].sort_values("date_of_service")
    dx_mask = (
        medical["patient_id"].eq(pid)
        & medical[DX_COLS].isin(set(LAUNCH_CONDITION_CODES)).any(axis=1)
    )
    diagnosis_index = medical.loc[dx_mask, "claim_date"].min()
    therapy_index = patient_fills.loc[
        patient_fills["transaction_type"].eq("PAID"), "date_of_service"
    ].min()
    line1 = patient_lines.iloc[0]
    line2 = patient_lines.iloc[1]
    addition_date = line2["line_start"]
    regimen_window_end = therapy_index + pd.Timedelta(days=30)
    observation_end = pd.Timestamp("2024-12-31")

    fig, ax = plt.subplots(figsize=(13.0, 5.5))
    start = diagnosis_index - pd.Timedelta(days=10)
    end = observation_end + pd.Timedelta(days=10)
    x = mdates.date2num
    ax.set_xlim(x(start), x(end))
    ax.set_ylim(0.0, 4.3)
    ax.set_yticks([3.35, 2.45, 1.45, 0.65])
    ax.set_yticklabels(["Timeline", "Claims", "Line 1", "Line 2"], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)

    for y in [3.35, 2.45, 1.45, 0.65]:
        ax.hlines(y, x(start), x(end), color="#c8c8c8", linewidth=1.0, zorder=0)

    def badge(xpos: float, ypos: float, text: str, face: str, edge: str = "#444444", txt: str = "white") -> None:
        ax.text(
            xpos,
            ypos,
            text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=txt,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=edge, linewidth=1.1),
            zorder=8,
        )

    def bar(y: float, start_date: pd.Timestamp, end_date: pd.Timestamp, color: str, label: str) -> None:
        ax.barh(
            y,
            x(end_date + pd.Timedelta(days=1)) - x(start_date),
            left=x(start_date),
            height=0.34,
            color=color,
            edgecolor="#4d4d4d",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            x(start_date + (end_date - start_date) / 2),
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    def vertical_label(date: pd.Timestamp, y: float, label: str, align: str = "above", color: str = "#444444") -> None:
        xpos = x(date)
        ax.vlines(xpos, y - 0.28, y + 0.28, color=color, linewidth=1.5, zorder=5)
        offset = 0.38 if align == "above" else -0.40
        va = "bottom" if align == "above" else "top"
        ax.text(xpos, y + offset, label, ha="center", va=va, fontsize=12, color=color)

    bar(1.45, line1["line_start"], line1["line_end"], BLUE, "L1 Vexpro")
    bar(0.65, line2["line_start"], observation_end, GREEN, "L2 Nexoral + Vexpro")

    ax.add_patch(
        mpatches.Rectangle(
            (x(therapy_index), 1.18),
            x(regimen_window_end) - x(therapy_index),
            0.18,
            facecolor=GOLD,
            edgecolor="none",
            alpha=0.35,
            zorder=2,
        )
    )
    ax.text(
        x(therapy_index + (regimen_window_end - therapy_index) / 2 - pd.Timedelta(days=8)),
        0.92,
        "30-day regimen window",
        ha="center",
        va="center",
        fontsize=11,
        color="#8a5a00",
    )

    vertical_label(diagnosis_index, 3.35, "Diagnosis index")
    vertical_label(therapy_index, 3.35, "Therapy index")
    vertical_label(observation_end, 3.35, "Observation end")
    ax.annotate(
        "",
        xy=(x(therapy_index), 3.95),
        xytext=(x(diagnosis_index), 3.95),
        arrowprops=dict(arrowstyle="<->", color="#666666", linewidth=1.2),
    )
    ax.text(
        x(diagnosis_index + (therapy_index - diagnosis_index) / 2),
        4.06,
        "58 days to first treatment fill",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    for i, row in enumerate(
        patient_fills.loc[patient_fills["transaction_type"].eq("PAID")].itertuples(index=False)
    ):
        color = BLUE if row.product_name == "Vexpro" else GREEN
        ax.scatter(x(row.date_of_service), 2.45, s=170, marker="o", color=color, zorder=6)
        ax.text(
            x(row.date_of_service),
            2.74 if i else 2.70,
            row.product_name,
            ha="center",
            va="bottom",
            fontsize=12,
            color=color,
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(x(addition_date), 0.88),
        xytext=(x(addition_date + pd.Timedelta(days=16)), 1.58),
        arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.2, mutation_scale=10),
    )
    badge(x(addition_date + pd.Timedelta(days=16)), 1.92, "ADDITION", GOLD, txt="#222222")
    badge(x(observation_end - pd.Timedelta(days=7)), 1.06, "CENSORED", GRAY, txt="#333333")

    ax.text(
        x(addition_date + pd.Timedelta(days=9)),
        1.12,
        "Nexoral arrives after the 30-day window\nwhile Vexpro supply is still active.",
        ha="left",
        va="center",
        fontsize=11,
        color="#555555",
    )

    xticks = [
        diagnosis_index,
        therapy_index,
        regimen_window_end,
        addition_date,
        observation_end,
    ]
    ax.set_xticks([x(d) for d in xticks])
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %-d\n%Y") for d in xticks], fontsize=11)
    ax.spines["bottom"].set_color("#cfcfcf")
    ax.set_title("PAT03874 addition example", fontsize=17)
    fig.tight_layout()

    _save(fig, "figure_5_4_addition_example")


# ---------------------------------------------------------------------------
# Supplemental - patient journey in 3 analytical layers
# ---------------------------------------------------------------------------

def figure_5_2_patient_journey_timeline() -> None:
    """The same patient shown across 3 layers: source events, exposure, classification."""

    lines    = pd.read_csv(OUT_DIR / "lines.csv",
                           parse_dates=["line_start", "line_end"])
    journeys = pd.read_csv(OUT_DIR / "journeys.csv",
                           parse_dates=["index_date", "followup_end"])
    pharmacy = pd.read_csv(
        DATA_DIR / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
        parse_dates=["date_of_service"],
    )
    ndc_codes = pd.read_csv(DATA_DIR / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    pharmacy["product_name"] = pharmacy["ndc_prescribed"].map(
        ndc_codes.set_index("ndc")["drug_name"]
    )
    sp = pd.read_csv(
        DATA_DIR / "specialty_pharmacy" / "sp_events.csv",
        parse_dates=["referral_date", "status_date", "ship_date"],
    )

    pid  = TIMELINE_PATIENT
    mine = lines.loc[lines["patient_id"].eq(pid)].sort_values("line_number")
    me   = journeys.loc[journeys["patient_id"].eq(pid)].iloc[0]
    fills = pharmacy.loc[pharmacy["patient_id"].eq(pid)].copy()
    hub_rows = sp.loc[sp["patient_id"].eq(pid)]
    hub = hub_rows.iloc[0] if not hub_rows.empty else None

    idx  = me["index_date"]
    fup  = me["followup_end"]
    span = (fup - idx).days

    def _xn(d: pd.Timestamp) -> float:
        """Normalize date to [0, 1] on the figure x-axis."""
        return (d - idx).days / span

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(-0.10, 1.15)
    ax.set_ylim(-0.7, 3.6)
    ax.axis("off")

    Y_SPINE    = 3.2
    Y_EVENTS   = 2.4
    Y_EXPOSURE = 1.5
    Y_CLASS    = 0.5
    BAND_H     = 0.65

    # Band backgrounds
    for yc, ht, band_label in [
        (Y_EVENTS,   BAND_H, "SOURCE\nEVENTS"),
        (Y_EXPOSURE, BAND_H, "EXPOSURE"),
        (Y_CLASS,    BAND_H, "CLASSIFI-\nCATION"),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, yc - ht / 2), 1.0, ht,
            boxstyle="round,pad=0.015", linewidth=0.6,
            edgecolor="#cccccc", facecolor="#f8f8f8", zorder=0,
        ))
        ax.text(-0.07, yc, band_label, ha="center", va="center",
                fontsize=7.5, color="#666666", fontweight="bold")

    # Spine timeline
    ax.annotate("", xy=(1.04, Y_SPINE), xytext=(-0.01, Y_SPINE),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.2, mutation_scale=10))
    ax.plot(0, Y_SPINE, marker="o", color="#5555aa", markersize=8, zorder=5)
    ax.text(0, Y_SPINE + 0.20, "Diagnosis\nindex", ha="center",
            fontsize=8, color="#444444")
    ax.plot(1, Y_SPINE, marker="o", color="#444444", markersize=8, zorder=5)
    ax.text(1, Y_SPINE + 0.20, "Follow-up\nend", ha="center",
            fontsize=8, color="#444444")

    # Source events
    for _, fill in fills.iterrows():
        if fill["date_of_service"] < idx or fill["date_of_service"] > fup:
            continue
        xp = _xn(fill["date_of_service"])
        if fill["transaction_type"] == "PAID":
            col = BLUE if fill["product_name"] != "Roventra" else GREEN
            ax.plot(xp, Y_EVENTS, marker="o", color=col,
                    markersize=8, zorder=5)
            ax.text(xp, Y_EVENTS + 0.25,
                    f"{fill['product_name'][:6]} fill",
                    ha="center", fontsize=6.5, color=col)
        else:
            ax.plot(xp, Y_EVENTS, marker="x", color=RED,
                    markersize=10, markeredgewidth=2.2, zorder=6)
            ax.text(xp, Y_EVENTS - 0.30, "Pended Rx",
                    ha="center", fontsize=7, color=RED)

    if hub is not None:
        for ev_label, ev_date, ev_marker in [
            ("Hub\nreferral", hub["referral_date"], "^"),
            ("Shipped",       hub["ship_date"],      "v"),
        ]:
            if pd.isna(ev_date) or ev_date < idx or ev_date > fup:
                continue
            xp = _xn(ev_date)
            ax.plot(xp, Y_EVENTS, marker=ev_marker, color=GRAY,
                    markersize=9, zorder=5)
            ax.text(xp, Y_EVENTS + 0.25, ev_label,
                    ha="center", fontsize=6.5, color="#666666")

    ax.text(0.5, Y_EVENTS + 0.53, "Observed records in source tables",
            ha="center", fontsize=8, color="#555555", style="italic")

    # Exposure bars
    line_colors = {1: BLUE, 2: GREEN}
    for _, row in mine.iterrows():
        x0 = _xn(row["line_start"])
        x1 = _xn(row["line_end"])
        lc = line_colors.get(int(row["line_number"]), GOLD)
        ax.barh(Y_EXPOSURE, x1 - x0, left=x0, height=0.28,
                color=lc, edgecolor="#444444", linewidth=0.8, zorder=3)
        ax.text((x0 + x1) / 2, Y_EXPOSURE, row["regimen"],
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")

    if len(mine) >= 2:
        sw_x = _xn(mine.iloc[1]["line_start"])
        ax.annotate("Switch",
                    xy=(sw_x, Y_EXPOSURE + 0.18),
                    xytext=(sw_x, Y_EXPOSURE + 0.42),
                    ha="center", fontsize=8, color="#444444",
                    arrowprops=dict(arrowstyle="-|>", color="#444444",
                                   lw=0.9, mutation_scale=8))

    ax.text(0.5, Y_EXPOSURE + 0.56,
            "Analysis-constructed exposure from completed fills",
            ha="center", fontsize=8, color="#555555", style="italic")

    # Classification boxes
    def _box(xc: float, yc: float, text: str,
             fc: str, tc: str = "white") -> None:
        ax.text(xc, yc, text, ha="center", va="center",
                fontsize=7.5, color=tc, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=fc,
                          edgecolor="#444444", linewidth=0.8, alpha=0.88))

    x_dx    = 0.04
    x_l1    = (_xn(mine.iloc[0]["line_start"]) + _xn(mine.iloc[0]["line_end"])) / 2
    x_sw    = _xn(mine.iloc[1]["line_start"]) if len(mine) >= 2 else 0.72
    x_l2    = (_xn(mine.iloc[1]["line_start"]) + _xn(mine.iloc[1]["line_end"])) / 2 \
              if len(mine) >= 2 else 0.82
    x_disc  = min(_xn(mine.iloc[-1]["line_end"]) + 0.09, 1.0)

    _box(x_dx,  Y_CLASS, "Diagnosis\ncohort",              "#7777aa")
    _box(x_l1,  Y_CLASS, f"L1: {mine.iloc[0]['regimen']}", BLUE)
    _box(x_sw,  Y_CLASS, "Switch",                         GOLD, "#444444")
    _box(x_l2,  Y_CLASS, f"L2: {mine.iloc[1]['regimen']}" if len(mine) >= 2 else "L2", GREEN)
    _box(x_disc, Y_CLASS, "Discontinued",                  RED)

    for x_s, x_e in [(x_dx + 0.06, x_l1 - 0.06),
                     (x_l1 + 0.06, x_sw  - 0.03),
                     (x_sw  + 0.03, x_l2 - 0.06),
                     (x_l2  + 0.06, x_disc - 0.06)]:
        if x_s < x_e:
            ax.annotate("", xy=(x_e, Y_CLASS), xytext=(x_s, Y_CLASS),
                        arrowprops=dict(arrowstyle="-|>", color="#444444",
                                       lw=0.9, mutation_scale=8))

    ax.text(0.5, Y_CLASS + 0.53,
            "Inferred patient state from explicit sequencing rules",
            ha="center", fontsize=8, color="#555555", style="italic")

    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1.0,
               ymin=0.05, ymax=0.82)

    # Note box
    note = (
        "The pended Rx is retained as access evidence, never counted as treatment.\n"
        f"Line 2 ends as Discontinued: the 60-day gap fires before follow-up ends on {fup.strftime('%Y-%m-%d')}."
    )
    ax.text(0.5, -0.45, note, ha="center", va="center", fontsize=8,
            color="#5a3800",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e8",
                      edgecolor=GOLD, linewidth=1.2))

    ax.set_title(
        f"{TIMELINE_PATIENT}: the same journey across 3 analytical layers",
        fontsize=11,
    )
    _save(fig, "figure_5_2_patient_journey_timeline")


# ---------------------------------------------------------------------------
# Figure 5.3 – Two-panel pathway: pre-initiation + post-initiation alluvial
# ---------------------------------------------------------------------------

def _alluvial(
    ax: plt.Axes,
    sources: list[tuple[str, int]],
    targets: list[tuple[str, int]],
    flows: list[tuple[str, str, int]],
    src_colors: dict[str, str],
    tgt_colors: dict[str, str],
    flow_alpha: float = 0.35,
    flow_alpha_overrides: dict[str, float] | None = None,
    flow_color_overrides: dict[str, str] | None = None,
    flow_linewidth_overrides: dict[str, float] | None = None,
) -> None:
    """Alluvial chart: sources and targets ordered from top to bottom."""

    def format_pct(count: int, total_count: int) -> str:
        pct = 100 * count / total_count
        return f"{pct:.0f}%" if pct >= 10 else f"{pct:.1f}%"

    def stack_positions(items: list[tuple[str, int]], gap: float = 0.022) -> dict[str, tuple[float, float]]:
        total_count = sum(count for _, count in items)
        if total_count == 0:
            return {}
        usable = 1.0 - gap * max(len(items) - 1, 0)
        y_top = 1.0
        positions: dict[str, tuple[float, float]] = {}
        for label, count in items:
            h = usable * (count / total_count)
            y1 = y_top
            y0 = y1 - h
            positions[label] = (y0, y1)
            y_top = y0 - gap
        return positions

    def spaced_centers(
        order: list[str],
        positions: dict[str, tuple[float, float]],
        min_sep: float = 0.10,
        floor: float = 0.02,
    ) -> dict[str, float]:
        centers = {label: (positions[label][0] + positions[label][1]) / 2 for label in order}
        adjusted: dict[str, float] = {}
        prev = 1.02
        for label in order:
            y = min(centers[label], prev - min_sep)
            adjusted[label] = y
            prev = y
        min_y = min(adjusted.values())
        if min_y < floor:
            bump = floor - min_y
            for label in order:
                adjusted[label] += bump
        return adjusted

    total = sum(c for _, c in sources)
    if total == 0:
        return

    src_pos = stack_positions(sources)
    tgt_pos = stack_positions(targets)
    src_counts = dict(sources)
    tgt_counts = dict(targets)
    src_label_y = spaced_centers([label for label, _ in sources], src_pos, min_sep=0.14, floor=0.06)
    tgt_label_y = spaced_centers([label for label, _ in targets], tgt_pos, min_sep=0.10, floor=0.02)

    bar_x0 = 0.10
    bar_w = 0.16
    left_label_x = -0.30
    right_label_x = 1.25

    for label, (y0, y1) in src_pos.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x0, y0), bar_w, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=src_colors.get(label, GRAY), edgecolor="#4a4a4a",
            linewidth=0.7, alpha=0.92,
        ))
        y_mid = (y0 + y1) / 2
        ax.plot([left_label_x + 0.03, bar_x0 - 0.01], [src_label_y[label], y_mid],
                color="#b8bcc6", linewidth=0.8)
        ax.text(left_label_x, src_label_y[label],
                f"{label} ({format_pct(src_counts[label], total)})\n"
                f"n = {src_counts[label]:,}",
                ha="right", va="center", fontsize=8.9, color="#333333")

    for label, (y0, y1) in tgt_pos.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (1 - bar_x0 - bar_w, y0), bar_w, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=tgt_colors.get(label, GRAY), edgecolor="#4a4a4a",
            linewidth=0.7, alpha=0.92,
        ))
        y_mid = (y0 + y1) / 2
        ax.plot([1 + 0.01, right_label_x - 0.03], [y_mid, tgt_label_y[label]],
                color="#b8bcc6", linewidth=0.8)
        ax.text(right_label_x, tgt_label_y[label],
                f"{label} ({format_pct(tgt_counts[label], total)})\n"
                f"n = {tgt_counts[label]:,}",
                ha="left", va="center", fontsize=9.2, color="#333333")

    src_cursor = {k: v[1] for k, v in src_pos.items()}
    tgt_cursor = {k: v[1] for k, v in tgt_pos.items()}
    t = np.linspace(0, 1, 120)
    s = t * t * (3 - 2 * t)  # smoothstep S-curve

    ordered_flows = sorted(
        flows,
        key=lambda x: (
            sources.index((x[0], next(c for lbl, c in sources if lbl == x[0]))),
            targets.index((x[1], next(c for lbl, c in targets if lbl == x[1]))),
            -x[2],
        ),
    )
    for src, tgt, count in ordered_flows:
        if count <= 0:
            continue
        h = (1.0 - 0.022 * max(len(sources) - 1, 0)) * (count / total)
        y1s = src_cursor[src]
        y0s = y1s - h
        y1t = tgt_cursor[tgt]
        y0t = y1t - h

        x = bar_x0 + bar_w + s * (1 - 2 * (bar_x0 + bar_w))
        y_bot = y0s + s * (y0t - y0s)
        y_top = y1s + s * (y1t - y1s)
        alpha = (flow_alpha_overrides or {}).get(src, flow_alpha)
        color = (flow_color_overrides or {}).get(src, src_colors.get(src, GRAY))
        lw = (flow_linewidth_overrides or {}).get(src, 0)
        ax.fill_between(x, y_bot, y_top, color=color, alpha=alpha, linewidth=lw)

        src_cursor[src] = y0s
        tgt_cursor[tgt] = y0t

    ax.set_xlim(-0.67, 1.64)
    ax.set_ylim(-0.05, 1.03)
    ax.axis("off")


def figure_5_3_pathway_sankey() -> None:
    """Single-panel alluvial chart for observed line-1 treatment pathways."""

    lines      = pd.read_csv(OUT_DIR / "lines.csv")
    initiators = pd.read_csv(OUT_DIR / "initiators.csv")

    new_ids = set(initiators.loc[initiators["new_to_therapy"], "patient_id"])
    line1   = lines.loc[
        lines["line_number"].eq(1) & lines["patient_id"].isin(new_ids)
    ].copy()
    line1["src"] = line1["regimen"].where(
        ~line1["regimen"].str.contains(" + ", regex=False), "Combination"
    )
    line1["tgt"] = line1["end_reason"].map({
        "Censored":     "Still on line 1",
        "Discontinued": "Discontinued",
        "Switch":       "Advanced to line 2",
        "Addition":     "Advanced to line 2",
    })
    flows_df = line1.groupby(["src", "tgt"]).size().reset_index(name="n")

    src_counts = line1.groupby("src")["patient_id"].count().sort_values(ascending=False)
    tgt_counts = line1.groupby("tgt")["patient_id"].count().sort_values(ascending=False)
    src_present = src_counts.index.tolist()
    tgt_order = tgt_counts.index.tolist()

    s_colors = {"Roventra": GREEN, "Nexoral": BLUE, "Vexpro": ORANGE, "Combination": GOLD}
    t_colors = {"Still on line 1": GREEN, "Advanced to line 2": BLUE, "Discontinued": RED}

    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    sources = [(s, int(src_counts[s])) for s in src_present]
    targets = [(t, int(tgt_counts.get(t, 0))) for t in tgt_order]
    flows   = [(r["src"], r["tgt"], int(r["n"])) for _, r in flows_df.iterrows()]
    _alluvial(ax, sources, targets, flows, s_colors, t_colors,
              flow_alpha=0.38,
              flow_color_overrides={"Combination": "#7A4A00"},
              flow_alpha_overrides={"Combination": 0.55})
    fig.subplots_adjust(left=0.06, right=0.96, top=0.98, bottom=0.06)
    _save(fig, "figure_5_3_pathway_sankey")


def figure_5_3_pathway_chord() -> None:
    """Chord-style alternative for line-1 pathways."""

    lines = pd.read_csv(OUT_DIR / "lines.csv")
    initiators = pd.read_csv(OUT_DIR / "initiators.csv")

    new_ids = set(initiators.loc[initiators["new_to_therapy"], "patient_id"])
    line1 = lines.loc[
        lines["line_number"].eq(1) & lines["patient_id"].isin(new_ids)
    ].copy()
    line1["src"] = line1["regimen"].where(
        ~line1["regimen"].str.contains(" + ", regex=False), "Combination"
    )
    line1["tgt"] = line1["end_reason"].map({
        "Censored": "Still on line 1",
        "Discontinued": "Discontinued",
        "Switch": "Advanced to line 2",
        "Addition": "Advanced to line 2",
    })
    flows_df = line1.groupby(["src", "tgt"]).size().reset_index(name="n")
    src_counts = line1.groupby("src").size().sort_values(ascending=False)
    tgt_counts = line1.groupby("tgt").size().sort_values(ascending=False)

    groups = [(name, int(count), "src") for name, count in src_counts.items()]
    groups += [(name, int(count), "tgt") for name, count in tgt_counts.items()]
    total = sum(count for _, count, _ in groups)
    gap = np.deg2rad(5)
    usable = 2 * np.pi - gap * len(groups)
    angle_cursor = np.pi / 2
    group_angles: dict[str, tuple[float, float, str]] = {}
    for name, count, kind in groups:
        span = usable * (count / total)
        start = angle_cursor
        end = angle_cursor - span
        group_angles[name] = (start, end, kind)
        angle_cursor = end - gap

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"aspect": "equal"})
    outer_r = 1.0
    inner_r = 0.84
    src_colors = {"Roventra": GREEN, "Nexoral": BLUE, "Vexpro": ORANGE, "Combination": GOLD}
    tgt_colors = {"Still on line 1": GREEN, "Discontinued": RED, "Advanced to line 2": BLUE}

    def pol2xy(theta: float, radius: float) -> tuple[float, float]:
        return radius * np.cos(theta), radius * np.sin(theta)

    def draw_arc(theta0: float, theta1: float, radius: float, color: str) -> None:
        arc = np.linspace(theta0, theta1, 80)
        xs, ys = pol2xy(arc, radius)
        ax.plot(xs, ys, color=color, linewidth=14, solid_capstyle="butt")

    def draw_ribbon(theta_a0: float, theta_a1: float, theta_b0: float, theta_b1: float, color: str) -> None:
        a0 = np.array(pol2xy(theta_a0, inner_r))
        a1 = np.array(pol2xy(theta_a1, inner_r))
        b0 = np.array(pol2xy(theta_b0, inner_r))
        b1 = np.array(pol2xy(theta_b1, inner_r))
        control = np.array([0.0, 0.0])
        path = mpatches.PathPatch(
            mpatches.Path(
                [
                    a0,
                    control,
                    b0,
                    b1,
                    control,
                    a1,
                    a0,
                ],
                [
                    mpatches.Path.MOVETO,
                    mpatches.Path.CURVE3,
                    mpatches.Path.CURVE3,
                    mpatches.Path.LINETO,
                    mpatches.Path.CURVE3,
                    mpatches.Path.CURVE3,
                    mpatches.Path.CLOSEPOLY,
                ],
            ),
            facecolor=color,
            edgecolor="none",
            alpha=0.34,
        )
        ax.add_patch(path)

    for name, count, kind in groups:
        theta0, theta1, _ = group_angles[name]
        color = src_colors.get(name, tgt_colors.get(name, GRAY))
        draw_arc(theta0, theta1, outer_r, color)
        theta_mid = (theta0 + theta1) / 2
        lx, ly = pol2xy(theta_mid, 1.19)
        ax.text(lx, ly, f"{name}\n{count:,}",
                ha="center", va="center", fontsize=9, color="#333333")

    src_cursor = {name: group_angles[name][0] for name in src_counts.index}
    tgt_cursor = {name: group_angles[name][0] for name in tgt_counts.index}
    for src_name, tgt_name, count in flows_df.sort_values("n", ascending=False).itertuples(index=False):
        src_span = (group_angles[src_name][0] - group_angles[src_name][1]) * (count / src_counts[src_name])
        tgt_span = (group_angles[tgt_name][0] - group_angles[tgt_name][1]) * (count / tgt_counts[tgt_name])
        src_theta0 = src_cursor[src_name]
        src_theta1 = src_theta0 - src_span
        tgt_theta0 = tgt_cursor[tgt_name]
        tgt_theta1 = tgt_theta0 - tgt_span
        draw_ribbon(src_theta0, src_theta1, tgt_theta0, tgt_theta1, src_colors.get(src_name, GRAY))
        src_cursor[src_name] = src_theta1
        tgt_cursor[tgt_name] = tgt_theta1

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.30)
    ax.axis("off")
    fig.suptitle("Line 1 pathways: chord alternative", fontsize=12, y=0.97)
    fig.text(0.5, 0.02,
             "Synthetic data. This is an alternative view. The Sankey remains easier to read for exact pathway volume.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    _save(fig, "figure_5_3_pathway_chord")


# ---------------------------------------------------------------------------
# Figure 5.4 – Washout examples
# ---------------------------------------------------------------------------

def figure_5_4_washout_relabel() -> None:
    """Contrast a relabeled continuation with a true new start."""

    journeys = pd.read_csv(
        OUT_DIR / "journeys.csv",
        parse_dates=["index_date", "followup_end"],
    )
    initiators = pd.read_csv(
        OUT_DIR / "initiators.csv",
        parse_dates=["therapy_index"],
    )
    pharmacy = pd.read_csv(
        DATA_DIR / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
        parse_dates=["date_of_service"],
    )
    ndc_codes = pd.read_csv(DATA_DIR / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    basket = set(pd.read_csv(DATA_DIR / "reference" / "products.csv")["product_name"])
    pharmacy["product_name"] = pharmacy["ndc_prescribed"].map(
        ndc_codes.set_index("ndc")["drug_name"]
    )

    cohort_ids = set(journeys["patient_id"])
    paid_all = pharmacy.loc[
        pharmacy["transaction_type"].eq("PAID")
        & pharmacy["product_name"].isin(basket)
        & pharmacy["patient_id"].isin(cohort_ids)
    ].copy()

    prior = paid_all.merge(
        initiators[["patient_id", "therapy_index", "new_to_therapy"]],
        on="patient_id",
        how="inner",
    ).merge(
        journeys[["patient_id", "index_date", "followup_end"]],
        on="patient_id",
        how="inner",
    )
    prior["window_start"] = prior["therapy_index"] - pd.to_timedelta(180, unit="D")
    in_washout = prior.loc[
        prior["date_of_service"].ge(prior["window_start"])
        & prior["date_of_service"].lt(prior["therapy_index"])
    ].copy()

    in_washout_agg = in_washout.groupby("patient_id", as_index=False).agg(
        prior_fills=("claim_id", "count"),
        last_prior=("date_of_service", "max"),
        therapy_index=("therapy_index", "first"),
        index_date=("index_date", "first"),
        followup_end=("followup_end", "first"),
    ).assign(days_to_therapy=lambda d: (d["therapy_index"] - d["index_date"]).dt.days)

    # Roventra-only washout patients (so corrected_text is accurate)
    roventra_wash_ids = set(in_washout.loc[in_washout["product_name"].eq("Roventra"), "patient_id"])
    # Visible after-fills: within therapy_index + 70 days
    after_70 = prior.loc[
        prior["date_of_service"].gt(prior["therapy_index"])
        & prior["date_of_service"].le(prior["therapy_index"] + pd.to_timedelta(70, unit="D"))
    ]
    after_counts_70 = after_70.groupby("patient_id", as_index=False).agg(visible_after=("claim_id", "count"))

    prevalent = (
        in_washout_agg
        .merge(after_counts_70, on="patient_id", how="inner")
        .loc[lambda d:
             d["patient_id"].isin(roventra_wash_ids)
             & d["days_to_therapy"].ge(20)
             & d["visible_after"].ge(2)]
        .sort_values(["prior_fills", "visible_after", "days_to_therapy"], ascending=[False, False, False])
        .reset_index(drop=True)
        .iloc[0]
    )

    line_counts = (
        paid_all.groupby("patient_id", as_index=False)
        .agg(paid_fills=("claim_id", "count"))
        .merge(initiators, on="patient_id", how="inner")
        .merge(journeys[["patient_id", "index_date", "followup_end"]], on="patient_id", how="inner")
    )
    new_start = (
        line_counts.loc[line_counts["new_to_therapy"]]
        .assign(days_from_dx=lambda d: (d["therapy_index"] - d["index_date"]).dt.days)
        .sort_values(["paid_fills", "days_from_dx"], ascending=[False, True])
        .reset_index(drop=True)
        .iloc[0]
    )

    selected = [
        {
            "patient_id": prevalent["patient_id"],
            "title": "Relabeled continuation",
            "therapy_index": prevalent["therapy_index"],
            "index_date": prevalent["index_date"],
            "followup_end": prevalent["followup_end"],
            "headline": f"{int(prevalent['prior_fills'])} prior treatment fills inside the 180-day washout",
            "naive_text": "No washout: counts the post-diagnosis fill as a new line 1 start",
            "corrected_text": "180-day washout: exclude as a continuing Roventra user",
            "corrected_color": RED,
        },
        {
            "patient_id": new_start["patient_id"],
            "title": "True new start",
            "therapy_index": new_start["therapy_index"],
            "index_date": new_start["index_date"],
            "followup_end": new_start["followup_end"],
            "headline": "No treatment-basket fills inside the 180-day washout",
            "naive_text": "No washout: counts as a new line 1 start",
            "corrected_text": "180-day washout: still retained as a true new start",
            "corrected_color": GREEN,
        },
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.9))
    x = mdates.date2num

    for ax, spec in zip(axes, selected):
        pid = spec["patient_id"]
        patient_paid = paid_all.loc[paid_all["patient_id"].eq(pid)].sort_values("date_of_service")
        therapy_index = pd.Timestamp(spec["therapy_index"])
        index_date = pd.Timestamp(spec["index_date"])
        washout_start = therapy_index - pd.Timedelta(days=180)
        end = therapy_index + pd.Timedelta(days=78)
        start = washout_start - pd.Timedelta(days=8)

        ax.set_xlim(x(start), x(end))
        ax.set_ylim(0.0, 3.1)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.hlines(2.28, x(start), x(end), color="#c8c8c8", linewidth=1.0, zorder=0)

        ax.add_patch(
            mpatches.Rectangle(
                (x(washout_start), 1.95),
                x(therapy_index) - x(washout_start),
                0.66,
                facecolor=GOLD,
                edgecolor="none",
                alpha=0.24,
                zorder=1,
            )
        )
        ax.text(
            x(washout_start + (therapy_index - washout_start) / 2),
            1.82,
            "180-day washout window",
            ha="center",
            va="top",
            fontsize=10.5,
            color="#8a5a00",
        )

        visible_fills = patient_paid.loc[
            patient_paid["date_of_service"].ge(start)
            & patient_paid["date_of_service"].le(end)
        ]
        for row in visible_fills.itertuples(index=False):
            xpos = x(row.date_of_service)
            ax.scatter(xpos, 2.28, s=130, color=GREEN, edgecolor="#4d4d4d", linewidth=0.8, zorder=4)
            if pd.Timestamp(row.date_of_service) != therapy_index:
                ax.text(
                    xpos,
                    2.10,
                    pd.Timestamp(row.date_of_service).strftime("%b %-d"),
                    ha="center",
                    va="top",
                    fontsize=10.5,
                    color="#444444",
                )

        if abs((therapy_index - index_date).days) <= 14:
            xpos = x(therapy_index)
            ax.vlines(xpos, 1.95, 2.64, color="#555555", linewidth=1.4, zorder=3)
            ax.text(
                xpos,
                2.82,
                f"Diagnosis index {index_date.strftime('%b %-d, %Y')}\nFirst treatment fill {therapy_index.strftime('%b %-d, %Y')}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#444444",
            )
        else:
            dx_x = x(index_date)
            tx_x = x(therapy_index)
            ax.vlines(dx_x, 2.35, 2.64, color="#555555", linewidth=1.4, zorder=3)
            ax.vlines(tx_x, 1.95, 2.64, color="#555555", linewidth=1.4, zorder=3)
            ax.text(
                dx_x - 5,
                2.80,
                f"Diagnosis index\n{index_date.strftime('%b %-d, %Y')}",
                ha="center",
                va="bottom",
                fontsize=10.3,
                color="#444444",
            )
            ax.text(
                tx_x + 5,
                2.80,
                f"First treatment fill\n{therapy_index.strftime('%b %-d, %Y')}",
                ha="center",
                va="bottom",
                fontsize=10.3,
                color="#444444",
            )

        ax.text(
            x(start),
            3.02,
            f"{spec['title']}: {pid}",
            ha="left",
            va="top",
            fontsize=14.5,
            color="#222222",
            fontweight="bold",
        )
        # Headline centered in the top half of the yellow box (y=2.28..2.61 → mid=2.445)
        x_hl = x(washout_start + (therapy_index - washout_start) / 2)
        ax.text(
            x_hl,
            2.445,
            spec["headline"],
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )

        ax.text(
            0.62,
            0.47,
            spec["naive_text"],
            ha="left",
            va="center",
            fontsize=10.4,
            color="white",
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.06",
                      facecolor=BLUE, edgecolor="#4d4d4d", linewidth=0.9, alpha=0.92),
        )

        ax.text(
            0.62,
            0.36,
            spec["corrected_text"],
            ha="left",
            va="center",
            fontsize=10.4,
            color="white" if spec["corrected_color"] != GREEN else "#173d24",
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.06",
                      facecolor=spec["corrected_color"], edgecolor="#4d4d4d", linewidth=0.9, alpha=0.92),
        )

        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)

    fig.suptitle("The washout rule removes continuations and preserves true new starts", fontsize=24, y=0.965)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.91])
    _save(fig, "figure_5_4_washout_relabel")


# ---------------------------------------------------------------------------
# Figure 5.5 – Rule sensitivity grid
# ---------------------------------------------------------------------------

def figure_5_5_rule_sensitivity() -> None:
    """Three panels showing line-1 discontinued share under each sequencing rule."""

    grid = pd.read_csv(OUT_DIR / "lot_sensitivity.csv")
    base = {"washout_days": 180, "regimen_window_days": 30, "allowable_gap_days": 60}
    panels = [
        ("washout_days",        "Washout (days)"),
        ("regimen_window_days", "Regimen window (days)"),
        ("allowable_gap_days",  "Allowable gap (days)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    for ax, (column, label) in zip(axes, panels):
        others = [key for key in base if key != column]
        rows   = grid.loc[
            grid[others[0]].eq(base[others[0]]) & grid[others[1]].eq(base[others[1]])
        ].sort_values(column)
        ax.plot(
            rows[column].astype(str),
            rows["line1_discontinued_share"],
            marker="o", color=GOLD, linewidth=1.6, markersize=7,
            markeredgecolor="#444444",
        )
        for _, row in rows.iterrows():
            ax.annotate(
                f"{row['line1_discontinued_share']:.1%}",
                (str(row[column]), row["line1_discontinued_share"] + 0.018),
                ha="center", fontsize=11,
            )
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylim(0.20, 0.60)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Line-1 discontinued share", fontsize=11)
    fig.suptitle("The ruler moves the result: discontinuation under each rule",
                 fontsize=14)
    fig.tight_layout()
    _save(fig, "figure_5_5_rule_sensitivity")


# ---------------------------------------------------------------------------
# Figure 5.7 - Stage clocks inside time to treatment
# ---------------------------------------------------------------------------

def figure_5_7_ttt_stage_clocks() -> None:
    """Decompose one diagnosis-to-treatment clock into actionable stages."""

    labels = ["Symptoms", "Diagnosis", "Test order", "Test result", "Prescription",
              "PA approval", "Treatment start"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    for i in range(len(labels) - 1):
        ax.annotate("", xy=(x[i + 1] - 0.18, 1.6), xytext=(x[i] + 0.18, 1.6),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))
    colors = [GRAY, BLUE, BLUE, BLUE, GOLD, GOLD, GREEN]
    for xi, label, color in zip(x, labels, colors):
        ax.scatter(xi, 1.6, s=900, marker="s", color=color,
                   edgecolor="#444444", linewidth=1.2, zorder=3)
        ax.text(xi, 2.02, label, ha="center", va="bottom", fontsize=11)

    spans = [
        (0, 1, "Diagnostic", GRAY), (1, 3, "Testing", BLUE),
        (3, 4, "Prescribing", GOLD), (4, 6, "Access", ORANGE),
    ]
    for start, end, label, color in spans:
        ax.annotate("", xy=(end, 0.55), xytext=(start, 0.55),
                    arrowprops=dict(arrowstyle="|-|", color=color, lw=3))
        ax.text((start + end) / 2, 0.25, f"{label} delay", ha="center", fontsize=11)
    ax.annotate("", xy=(6, -0.35), xytext=(1, -0.35),
                arrowprops=dict(arrowstyle="|-|", color=GREEN, lw=3))
    ax.text(3.5, -0.65, "Diagnosis-to-treatment time",
            ha="center", fontsize=11, color="#315c37")
    ax.set_xlim(-0.55, 6.55)
    ax.set_ylim(-1.0, 2.55)
    ax.axis("off")
    ax.set_title("Total time to treatment contains several actionable clocks", fontsize=16)
    _save(fig, "figure_5_7_ttt_stage_clocks")


# ---------------------------------------------------------------------------
# Figure 5.8 - Five-patient records
# ---------------------------------------------------------------------------

def figure_5_8_patient_records() -> None:
    """Patient timelines for the five-patient censoring example."""

    days = [19, 31, 59, 90, 90]
    treated = [True, True, True, False, False]
    patients = list("ABCDE")
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    ypos = np.arange(5)[::-1]
    for y, patient, day, event in zip(ypos, patients, days, treated):
        ax.hlines(y, 0, day, color=BLUE, linewidth=7, alpha=0.65)
        ax.scatter(day, y, s=100, marker="o" if event else "X",
                   color=GREEN if event else GRAY, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.text(day, y + 0.16, f"day {day}", ha="center", fontsize=11)
    ax.set_yticks(ypos, [f"Patient {p}" for p in patients], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Days from diagnosis", fontsize=12)
    ax.set_title("Patient records", fontsize=17, pad=14)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, 4.55)
    fig.tight_layout(pad=0.8)
    _save(fig, "figure_5_8_patient_records")


# ---------------------------------------------------------------------------
# Figure 5.9 - Five-patient Kaplan-Meier estimate
# ---------------------------------------------------------------------------

def figure_5_9_km_estimate() -> None:
    """Toy-case cumulative initiation beside the Kaplan-Meier untreated curve."""

    days = [19, 31, 59, 90, 90]
    treated = [True, True, True, False, False]
    curve = km_curve(pd.Series(days), pd.Series(treated))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6),
                             gridspec_kw={"width_ratios": [1, 1]})
    left, right = axes

    left.step(curve["day"], curve["survival"], where="post",
              color=BLUE, linewidth=2.2)
    left.axhline(0.5, color=GOLD, linestyle="--", linewidth=1.5)
    left.vlines(59, 0, 0.8, color=GOLD, linestyle="--", linewidth=1.5)
    left.text(62, 0.52, "median = 59 days", fontsize=11, color="#765714")
    for day, value, label in [(19, .8, "4/5"), (31, .6, "3/5"), (59, .4, "2/5")]:
        left.plot(day, value, "o", color=GREEN, markersize=7)
        left.annotate(label, (day, value), xytext=(-12, 0),
                      textcoords="offset points", ha="right", va="center",
                      fontsize=11)
    left.plot([90, 90], [.385, .415], "|", color=GRAY, markersize=12)
    left.set_xlim(0, 100)
    left.set_ylim(0, 1.05)
    left.set_xlabel("Days from diagnosis", fontsize=12)
    left.set_ylabel("Probability still untreated", fontsize=12)
    left.set_title("Kaplan-Meier Curve", fontsize=17, pad=14)
    left.tick_params(labelsize=11)
    left.spines[["right", "top"]].set_visible(False)

    right.step(curve["day"], 1 - curve["survival"], where="post",
               color=GREEN, linewidth=2.4)
    for day, value, label in [(19, .2, "20%"), (31, .4, "40%"), (59, .6, "60%")]:
        right.plot(day, value, "o", color=BLUE, markersize=8, zorder=5)
        right.annotate(label, (day, value), xytext=(0, 12),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=11)
    right.plot([90, 90], [.585, .615], "|", color=GRAY, markersize=12)
    right.set_xlim(0, 100)
    right.set_ylim(0, 1.05)
    right.set_xlabel("Days from diagnosis", fontsize=12)
    right.set_ylabel("Cumulative initiation", fontsize=12)
    right.set_title("Treatment initiation", fontsize=17, pad=14)
    right.tick_params(labelsize=11)
    right.spines[["right", "top"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    _save(fig, "figure_5_9_km_estimate")


# ---------------------------------------------------------------------------
# Figure 5.10 - KM treatment initiation curve
# ---------------------------------------------------------------------------

def figure_5_10_initiation_curve() -> None:
    """KM cumulative initiation with confidence band and risk counts."""

    curve = pd.read_csv(OUT_DIR / "initiation_curve.csv")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.step(curve["day"], curve["cumulative_initiation"],
            where="post", color=BLUE, linewidth=2.4)
    if {"cumulative_initiation_lower_95", "cumulative_initiation_upper_95"}.issubset(curve.columns):
        ax.fill_between(curve["day"], curve["cumulative_initiation_lower_95"], curve["cumulative_initiation_upper_95"],
                        step="post", color=BLUE, alpha=0.18)
    for day in (90, 180, 270):
        at = curve.loc[curve["day"].le(day), "cumulative_initiation"].iloc[-1]
        ax.plot(day, at, marker="o", color=GREEN, markersize=10, zorder=5)
        ax.annotate(f"{at:.1%}", (day, at + 0.048), ha="center", fontsize=13)
    median_day = int(curve.loc[curve["cumulative_initiation"].ge(0.5), "day"].iloc[0])
    ax.plot(median_day, 0.5, marker="o", color=GOLD, markersize=10, zorder=6)
    ax.annotate(
        f"50.0%\nDay {median_day}",
        (median_day, 0.5),
        xytext=(62, -58),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=12,
        linespacing=1.25,
        arrowprops={
            "arrowstyle": "->",
            "color": GRAY,
            "lw": 1.4,
            "connectionstyle": "arc3,rad=0",
        },
        bbox={"facecolor": "white", "edgecolor": GRAY, "pad": 4.0},
    )
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 0.85)
    ax.set_xticks([0, 90, 180, 270])
    ax.set_xlabel("Days from diagnosis index", fontsize=13, labelpad=8)
    ax.set_ylabel("Cumulative initiation", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title("Accumunative treatment initiation", fontsize=16)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.14)
    _save(fig, "figure_5_10_initiation_curve")


# ---------------------------------------------------------------------------
# Figure 5.11 - Competing-risk comparison
# ---------------------------------------------------------------------------

def figure_5_11_competing_risk() -> None:
    """Competing-risk cumulative incidence and coherent state probabilities."""

    days = pd.Series([19, 31, 59, 90, 90])
    outcomes = pd.Series(["Treated", "Died", "Treated", "Censored", "Censored"])
    aj = aalen_johansen_curve(days, outcomes)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2))
    left, right = axes
    death_color = "#c85f55"

    state_areas = left.stackplot(
        aj["day"], aj["event_free"], aj["cumulative_interest"],
        aj["cumulative_competing"], step="post",
        colors=["#b8d2e8", "#9bc7a0", "#e5a29b"], alpha=1.0,
        edgecolor="white", linewidth=1.0,
        labels=["Untreated and alive", "Treated", "Died"],
    )
    for area, hatch in zip(state_areas, [None, "///", "xx"]):
        area.set_hatch(hatch)
    left.set_xlim(0, 95)
    left.set_ylim(0, 1)
    left.set_xlabel("Days from diagnosis", fontsize=14)
    left.set_ylabel("Probability", fontsize=14)
    left.set_title("State probabilities", fontsize=18, pad=14)
    left.legend(frameon=False, fontsize=13, loc="lower left")
    left.spines[["right", "top"]].set_visible(False)
    left.tick_params(labelsize=13)

    treat_line = right.step(
        aj["day"], aj["cumulative_interest"], where="post",
        color="#4f9258", linewidth=4.0,
        label="Treatment cumulative incidence", zorder=3,
    )[0]
    death_line = right.step(
        aj["day"], aj["cumulative_competing"], where="post",
        color=death_color, linewidth=3.2, linestyle=":",
        label="Death cumulative incidence", zorder=5,
    )[0]
    for line in (treat_line, death_line):
        line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth() + 2.2,
                                         foreground="white"), pe.Normal()])
    right.scatter([19, 59], [0.2, 0.4], s=70, color="#4f9258",
                  edgecolor="white", linewidth=0.8, zorder=7)
    right.scatter([31], [0.2], s=76, marker="D", color=death_color,
                  edgecolor="white", linewidth=0.8, zorder=7)
    right.annotate(
        "Death on day 31",
        xy=(31, 0.2),
        xytext=(8, 0.30),
        fontsize=13,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.3},
    )
    right.set_xlim(0, 95)
    right.set_ylim(0, 0.55)
    right.set_xlabel("Days from diagnosis", fontsize=14)
    right.set_ylabel("Cumulative probability", fontsize=14)
    right.set_title("Treatment probabilities", fontsize=18, pad=14)
    right.legend(frameon=False, fontsize=13, loc="lower right",
                 handlelength=3.0)
    right.spines[["right", "top"]].set_visible(False)
    right.tick_params(labelsize=13)
    fig.tight_layout(pad=1.0, w_pad=3.0)
    _save(fig, "figure_5_11_competing_risk")


# ---------------------------------------------------------------------------
# Figure 5.12 - One patient's persistence and adherence measures
# ---------------------------------------------------------------------------

def figure_5_12_patient_medication_use() -> None:
    """Show how persistence, PDC, and MPR use one patient's fill history."""

    pharmacy = pd.read_csv(
        DATA_DIR / "claims_pharmacy" / "pharmacy_claims.csv",
        parse_dates=["date_of_service"],
    )
    adherence = pd.read_csv(
        OUT_DIR / "adherence_index_product.csv",
        parse_dates=["observation_start", "observation_end"],
    )
    basket_adherence = pd.read_csv(OUT_DIR / "adherence_market_basket.csv")
    patient_id = "PAT00036"
    fills = pharmacy.loc[
        pharmacy["patient_id"].eq(patient_id)
        & pharmacy["transaction_type"].eq("PAID")
    ].sort_values("date_of_service")
    patient = adherence.loc[adherence["patient_id"].eq(patient_id)].iloc[0]
    basket_patient = basket_adherence.loc[
        basket_adherence["patient_id"].eq(patient_id)
    ].iloc[0]
    start = patient["observation_start"]
    fill_days = (fills["date_of_service"] - start).dt.days.to_list()
    days_supply = fills["days_supply"].astype(int).to_list()

    coverage: list[tuple[int, int]] = []
    available_through = -1
    for fill_day, supply in zip(fill_days, days_supply, strict=True):
        supply_start = max(fill_day, available_through + 1)
        supply_end = supply_start + supply - 1
        coverage.append((supply_start, supply_end))
        available_through = supply_end

    window_days = int(patient["observation_days"])
    visible_end = window_days - 1
    cutoff_x = window_days
    covered_days = round(float(patient["pdc"]) * window_days)
    uncovered_days = window_days - covered_days
    fig, ax = plt.subplots(figsize=(12.0, 5.7))
    ax.set_xlim(-10, 125)
    ax.set_ylim(-0.8, 3.35)
    ax.axis("off")

    row_y = {"fills": 2.9, "coverage": 1.9, "persistence": 0.9}
    for label, y in row_y.items():
        ax.hlines(y, 0, cutoff_x, color="#b8bcc6", linewidth=1.2)
        ax.text(-7, y, label.title(), ha="right", va="center",
                fontsize=12, fontweight="bold", color="#333333")

    for number, (fill_day, supply) in enumerate(
        zip(fill_days, days_supply, strict=True), start=1
    ):
        ax.plot(fill_day, row_y["fills"], "o", color=BLUE, markersize=9)
        ax.text(fill_day, row_y["fills"] + 0.22, f"Fill {number}\n{supply} days",
                ha="center", va="bottom", fontsize=9.5, color="#345f87")

    for supply_start, supply_end in coverage:
        clipped_end = min(supply_end, visible_end)
        if supply_start <= visible_end:
            ax.barh(row_y["coverage"], clipped_end - supply_start + 1,
                    left=supply_start, height=0.24, color=BLUE,
                    edgecolor="white", linewidth=0.8)
        if supply_end > visible_end:
            ax.barh(row_y["coverage"], supply_end - visible_end,
                    left=cutoff_x, height=0.24, color=GOLD, alpha=0.55,
                    edgecolor="white", linewidth=0.8)
    first_gap_start = min(end_day for _, end_day in coverage) + 1
    for (_, previous_end), (next_start, _) in zip(coverage, coverage[1:]):
        if next_start > previous_end + 1:
            first_gap_start = previous_end + 1
            break
    ax.barh(row_y["coverage"], uncovered_days, left=first_gap_start,
            height=0.24, color=RED, edgecolor="white", linewidth=0.8)
    ax.text(first_gap_start + uncovered_days / 2 - 4, row_y["coverage"] - 0.25,
            f"{uncovered_days} uncovered days",
            ha="center", va="top", fontsize=9.5, color="#9a473c")
    ax.text(106, row_y["coverage"] - 0.25, "Supply after cutoff",
            ha="center", va="top", fontsize=9.5, color="#7a5a16")

    ax.barh(row_y["persistence"], cutoff_x, left=0, height=0.20,
            color=GREEN, edgecolor="white")
    ax.plot(cutoff_x, row_y["persistence"], marker="|", color="#333333",
            markersize=17, markeredgewidth=2)
    ax.annotate(
        "Treatment continues through cutoff",
        xy=(cutoff_x, row_y["persistence"]),
        xytext=(75, row_y["persistence"] + 0.30),
        ha="center", va="bottom", fontsize=9.5, color="#3d6f45",
        arrowprops=dict(arrowstyle="->", color="#3d6f45", linewidth=1.0),
    )

    for x, label in [(0, "First fill"), (cutoff_x, "95-day cutoff")]:
        ax.vlines(x, 0.55, 2.55, color=GRAY, linestyle="--", linewidth=0.9)
        ax.text(x, 0.42, label, ha="center", va="top", fontsize=9.5,
                color="#555555")

    metrics = [
        (7, f"Persistence\n{window_days} observed days\nremains on regimen", GREEN),
        (37, f"Index-product PDC\n{covered_days} / {window_days} = {patient['pdc']:.1%}", BLUE),
        (68, f"Basket PDC\n{covered_days} / {window_days} = {basket_patient['pdc']:.1%}", BLUE),
        (100, f"MPR\n{days_supply[0]} × {len(days_supply)} / {window_days} = {patient['mpr']:.1%}", GOLD),
    ]
    for x, text_value, color in metrics:
        ax.text(x, -0.25, text_value, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=color, linewidth=1.5))

    _save(fig, "figure_5_12_patient_medication_use")


# ---------------------------------------------------------------------------
# Figure 5.13 - Initial-regimen persistence
# ---------------------------------------------------------------------------

def figure_5_13_persistence() -> None:
    """Plot initial-regimen persistence with confidence interval."""

    persistence = pd.read_csv(OUT_DIR / "line1_persistence.csv")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.step(persistence["day"], persistence["survival"], where="post",
            color=GREEN, linewidth=2.4)
    ax.fill_between(
        persistence["day"], persistence["lower_95"], persistence["upper_95"],
        step="post", color=GREEN, alpha=0.18,
    )
    label_positions = {
        60: (48, 0.88),
        90: (94, 0.72),
        113: (130, 0.56),
    }
    for day in (60, 90, 113):
        row = persistence.loc[persistence["day"].le(day)].iloc[-1]
        ax.plot(day, row["survival"], "o", color=GREEN, markersize=5)
        ax.annotate(
            f"{row['survival']:.0%}\n{int(row['at_risk']):,} at risk",
            (day, row["survival"]), xytext=label_positions[day],
            textcoords="data", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8),
        )
    ax.set(xlim=(0, 190), ylim=(0, 1.02), xlabel="Days from first fill",
           ylabel="Probability of remaining on initial regimen",
           title="Initial-regimen persistence")
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.title.set_fontsize(16)
    fig.tight_layout(pad=1.2)
    _save(fig, "figure_5_13_persistence")


# ---------------------------------------------------------------------------
# Figure 5.14 - Index-product PDC distribution
# ---------------------------------------------------------------------------

def figure_5_14_pdc_distribution() -> None:
    """Plot the patient-level index-product PDC distribution."""

    adherence = pd.read_csv(OUT_DIR / "adherence_index_product.csv")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    bins = np.linspace(0, 1, 11)
    ax.hist(adherence["pdc"], bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(0.80, color=GOLD, linestyle="--", linewidth=2)
    ax.text(0.78, ax.get_ylim()[1] * 0.94, "0.80 threshold",
            ha="right", va="top", fontsize=9, color="#7a5a16")
    ax.set(xlim=(0, 1), xlabel="Index-product PDC", ylabel="Patients",
           title="Index-product PDC distribution")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.title.set_fontsize(16)
    fig.tight_layout(pad=1.2)
    _save(fig, "figure_5_14_pdc_distribution")


# ---------------------------------------------------------------------------
# Figure 5.15 - Payer adherence comparison
# ---------------------------------------------------------------------------

def figure_5_15_payer_adherence() -> None:
    """Compare payer adherence rates with Wilson confidence intervals."""

    adherence = pd.read_csv(OUT_DIR / "adherence_index_product.csv")
    payer = pd.read_csv(OUT_DIR / "adherence_by_payer.csv").query(
        "payer_id != 'All'"
    )
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    z_value = 1.959963984540054
    rates = payer["adherent_pdc_rate"].to_numpy()
    counts = payer["treated_patients"].to_numpy()
    denominator = 1 + z_value**2 / counts
    centers = (rates + z_value**2 / (2 * counts)) / denominator
    half_widths = (
        z_value
        * np.sqrt(rates * (1 - rates) / counts + z_value**2 / (4 * counts**2))
        / denominator
    )
    y_pos = np.arange(len(payer))
    ax.errorbar(
        rates, y_pos, xerr=[rates - (centers - half_widths),
                            (centers + half_widths) - rates],
        fmt="o", color=BLUE, ecolor=GRAY, capsize=3,
    )
    ax.axvline(adherence["adherent_pdc"].mean(), color=GOLD,
               linestyle="--", linewidth=1.5)
    ax.set_yticks(y_pos, payer["payer_id"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    )
    ax.set(xlim=(0.08, 0.25), xlabel="Patients with PDC at or above 0.80",
           title="Adherence rate by payer")
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.title.set_fontsize(16)
    fig.tight_layout(pad=1.2)
    _save(fig, "figure_5_15_payer_adherence")


if __name__ == "__main__":
    figure_5_1_cohort_eligibility()
    figure_5_2_treatment_sequence_rules()
    figure_5_3_switch_example()
    figure_5_4_addition_example()
    figure_5_3_pathway_sankey()
    figure_5_4_washout_relabel()
    figure_5_7_ttt_stage_clocks()
    figure_5_8_patient_records()
    figure_5_9_km_estimate()
    figure_5_10_initiation_curve()
    figure_5_11_competing_risk()
    figure_5_12_patient_medication_use()
    figure_5_13_persistence()
    figure_5_14_pdc_distribution()
    figure_5_15_payer_adherence()
    print(f"\nfigures in {FIG_DIR}")
