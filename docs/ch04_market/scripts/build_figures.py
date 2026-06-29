"""Build the publication figures for Chapter 4 from generated analysis outputs.

Figures use short, neutral titles. All interpretation lives in the chapter captions
and prose, per the project visual standard: no takeaway sentences, subtitles, or
footer boxes on the canvas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from sdoh_market import figure_4_5_under_observation, figure_4_6_model_lift

BLUE = "#BFD7EA"
BLUE_DARK = "#2F6690"
GOLD = "#F6D58A"
GOLD_DARK = "#C9941F"
GREEN = "#B8D8BA"
GREEN_DARK = "#3A7D44"
ORANGE = "#F4A582"
RED = "#C74B50"
GRAY = "#E8ECEF"
GRAY_TEXT = "#59636B"
TEXT = "#252A2E"


def save(
    fig: plt.Figure, figures_dir: Path, name: str, pad_inches: float = 0.1
) -> None:
    """Save canonical SVG and companion PNG assets."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    svg_path = figures_dir / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white", pad_inches=pad_inches)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    fig.savefig(
        figures_dir / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=pad_inches,
    )
    plt.close(fig)


def add_card(ax, x, y, width, height, title, value, fill, note="") -> None:
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=TEXT,
        facecolor=fill,
    )
    ax.add_patch(card)
    ax.text(
        x + width / 2, y + height * 0.67, title, ha="center", va="center", fontsize=10
    )
    ax.text(
        x + width / 2,
        y + height * 0.40,
        value,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )
    if note:
        ax.text(
            x + width / 2,
            y + height * 0.15,
            note,
            ha="center",
            va="center",
            fontsize=8,
            color=GRAY_TEXT,
        )


def market_sizes(outputs: Path, figures_dir: Path) -> None:
    panel = pd.read_csv(outputs / "panel_market_sizes.csv").set_index("stage")
    stages = [
        (
            "True condition",
            "Answer key",
            int(panel.loc["True condition", "panel_count"]),
        ),
        (
            "Diagnosis recorded",
            "Any claim status",
            int(panel.loc["Launch diagnosis coded", "panel_count"]),
        ),
        (
            "Age eligible",
            "Age 35 or older",
            int(panel.loc["Age-eligible diagnosed", "panel_count"]),
        ),
        (
            "Untreated opportunity",
            "No net Roventra use",
            int(panel.loc["Untreated opportunity", "panel_count"]),
        ),
    ]
    base = stages[0][2]
    dot_counts = [round(value / base * 100) for _, _, value in stages]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.5, 5.0)
    ax.axis("off")
    ax.set_title(
        "Four market sizes for one disease", fontsize=18, fontweight="bold", pad=18
    )

    dot_x = np.tile(np.arange(10), 10)
    dot_y = np.repeat(np.arange(10), 10)
    dot_rank = dot_x * 10 + dot_y
    colors = [BLUE_DARK, "#4D84AB", "#79A66F", GREEN_DARK]
    for row_index, ((title, rule, value), retained, color) in enumerate(
        zip(stages, dot_counts, colors)
    ):
        y_offset = 4.0 - row_index * 1.35
        ax.text(
            0.1, y_offset + 0.17, title, fontsize=11, fontweight="bold", va="center"
        )
        ax.text(0.1, y_offset - 0.17, rule, fontsize=8.5, color=GRAY_TEXT, va="center")
        ax.scatter(
            3.35 + dot_x * 0.29,
            y_offset - 0.26 + dot_y * 0.058,
            s=14,
            color=[color if rank < retained else GRAY for rank in dot_rank],
            edgecolors="none",
        )
        ax.text(
            6.55, y_offset, f"{value:,}", fontsize=14, fontweight="bold", va="center"
        )
        ax.text(
            7.55,
            y_offset,
            f"{value / base:.0%} of true condition",
            fontsize=9,
            color=GRAY_TEXT,
            va="center",
        )
        if row_index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (10.8, y_offset - 0.30),
                    (10.8, y_offset - 1.02),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.2,
                    color=GRAY_TEXT,
                )
            )
            ax.text(
                11.05,
                y_offset - 0.68,
                ["coded", "eligible", "untreated"][row_index],
                fontsize=8.5,
                color=GRAY_TEXT,
                va="center",
            )
    save(fig, figures_dir, "figure_4_1_market_sizes")


def diagnosis_visibility(outputs: Path, figures_dir: Path) -> None:
    vis = pd.read_csv(outputs / "diagnosis_visibility.csv").set_index("measure")[
        "patient_count"
    ]
    true_n = int(vis["True condition patients"])
    coded = int(vis["True patients with any launch diagnosis code"])
    captured = int(vis["True positives in paid-claims phenotype"])
    false_pos = int(vis["False positives in paid-claims phenotype"])
    never_coded = true_n - coded
    coded_not_paid = coded - captured

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    labels = ["True condition", "Coded", "Paid-claims phenotype"]
    values = [true_n, coded, captured]
    colors = [BLUE_DARK, "#4D84AB", GREEN_DARK]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor=TEXT, linewidth=0.9, zorder=3)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(
            value + true_n * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.text(
        coded + never_coded / 2,
        0.5,
        f"{never_coded:,} never coded",
        ha="center",
        va="center",
        fontsize=9,
        color="#7A3E2C",
    )
    ax.text(
        captured + coded_not_paid / 2,
        1.5,
        f"{coded_not_paid:,} coded, not paid",
        ha="center",
        va="center",
        fontsize=9,
        color="#7A3E2C",
    )
    ax.add_patch(
        Rectangle(
            (coded, -0.34),
            never_coded,
            0.68,
            facecolor=ORANGE,
            edgecolor=RED,
            linewidth=0.9,
            alpha=0.80,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (captured, 0.66),
            coded_not_paid,
            0.68,
            facecolor=ORANGE,
            edgecolor=RED,
            linewidth=0.9,
            alpha=0.80,
            zorder=2,
        )
    )
    ax.text(
        captured,
        2.47,
        f"+{false_pos} false positives",
        fontsize=9,
        color=RED,
        ha="left",
        va="center",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, true_n * 1.12)
    ax.set_xlabel("Patients")
    ax.set_title("Diagnosis visibility gates", fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="x", color=GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, figures_dir, "figure_4_2_diagnosis_visibility")


def phenotype_tradeoff(outputs: Path, figures_dir: Path) -> None:
    diagnostics = pd.read_csv(outputs / "phenotype_diagnostics.csv").set_index(
        "phenotype"
    )
    base = diagnostics.loc["base_phenotype"]
    strict = diagnostics.loc["strict_phenotype"]

    tp1, fp1 = int(base["true_positive"]), int(base["false_positive"])
    fn1, tn1 = int(base["false_negative"]), int(base["true_negative"])
    tp2, fp2 = int(strict["true_positive"]), int(strict["false_positive"])
    fn2, tn2 = int(strict["false_negative"]), int(strict["true_negative"])
    total = tp1 + fp1 + fn1 + tn1

    fp_color = RED
    dot_colors_map = {"tp": BLUE_DARK, "fn": GOLD_DARK, "fp": fp_color, "tn": GRAY}

    fig = plt.figure(figsize=(14.0, 12.0))
    # hspace=0.25 creates a gap between dot grids and metric panels;
    # the legend floats in that gap via fig.legend() with explicit y-coordinate.
    # top=0.87 shifts panels down so the title has breathing room at the top.
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[4.8, 1.1],
        hspace=0.25,
        wspace=0.20,
        left=0.07,
        right=0.93,
        top=0.87,
        bottom=0.04,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1m = fig.add_subplot(gs[1, 0])
    ax2m = fig.add_subplot(gs[1, 1])

    main_sp = 0.90
    main_s = 260

    def draw_main_panel(ax, label, tp, fp, fn, tn):
        raw = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        scaled = {k: round(v / total * 100) for k, v in raw.items()}
        scaled["tn"] += 100 - sum(scaled.values())
        dot_list = (
            [dot_colors_map["tp"]] * scaled["tp"]
            + [dot_colors_map["fn"]] * scaled["fn"]
            + [dot_colors_map["fp"]] * scaled["fp"]
            + [dot_colors_map["tn"]] * scaled["tn"]
        )
        for idx, color in enumerate(dot_list):
            col = (idx % 10) * main_sp
            row_pos = (9 - idx // 10) * main_sp
            ax.scatter(
                col,
                row_pos,
                color=color,
                s=main_s,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        ax.set_xlim(-0.6, 9 * main_sp + 0.6)
        ax.set_ylim(-0.4, 9 * main_sp + 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=32, fontweight="bold", pad=26)

    def draw_metric_panel(ax, tp, fp, fn, metric_label):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        sens = tp / (tp + fn)
        ppv = tp / (tp + fp) if tp + fp else 1.0
        ax.add_patch(
            FancyBboxPatch(
                (0.03, 0.06),
                0.44,
                0.72,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor="#F7F9FB",
                edgecolor=GRAY_TEXT,
                linewidth=1.0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.53, 0.06),
                0.44,
                0.72,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor="#F7F9FB",
                edgecolor=GRAY_TEXT,
                linewidth=1.0,
            )
        )
        for x0, value, denom, label, color in [
            (0.03, sens, tp + fn, "Sensitivity", GOLD_DARK),
            (0.53, ppv, tp + fp, "Precision", fp_color),
        ]:
            ax.text(
                x0 + 0.22,
                0.62,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=color,
            )
            ax.text(
                x0 + 0.22,
                0.40,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=TEXT,
            )
            ax.text(
                x0 + 0.22,
                0.17,
                f"{tp:,} / {denom:,}",
                ha="center",
                va="center",
                fontsize=11,
                color=GRAY_TEXT,
            )
        ax.text(
            0.50,
            0.97,
            metric_label,
            ha="center",
            va="center",
            fontsize=22,
            color=TEXT,
            fontweight="bold",
            clip_on=False,
        )

    draw_main_panel(ax1, "At least 1 diagnosis", tp1, fp1, fn1, tn1)
    draw_main_panel(ax2, "At least 2 diagnoses", tp2, fp2, fn2, tn2)
    draw_metric_panel(ax1m, tp1, fp1, fn1, "Keep more true patients")
    draw_metric_panel(ax2m, tp2, fp2, fn2, "Keep fewer false positives")

    # --- highlight annotations on main panels ---
    def _scaled(tp: int, fn: int, fp: int, tn: int) -> dict:
        raw = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        s = {k: round(v / total * 100) for k, v in raw.items()}
        s["tn"] += 100 - sum(s.values())
        return s

    s1 = _scaled(tp1, fn1, fp1, tn1)
    s2 = _scaled(tp2, fn2, fp2, tn2)
    pad = 0.38
    ann_kw = dict(lw=2.5)  # shared arrow line thickness

    # Left panel: box around the single FP (red) dot; straight vertical arrow with gap
    fp1_idx = s1["tp"] + s1["fn"]
    fp1_col = (fp1_idx % 10) * main_sp
    fp1_row = (9 - fp1_idx // 10) * main_sp
    ax1.add_patch(
        Rectangle(
            (fp1_col - pad, fp1_row - pad),
            2 * pad,
            2 * pad,
            linewidth=2.5,
            edgecolor=RED,
            facecolor="none",
            zorder=5,
        )
    )
    ax1.annotate(
        "False positive\n(bad!)",
        xy=(fp1_col, fp1_row - pad),
        xytext=(fp1_col, fp1_row - 3.0),
        fontsize=16,
        color=RED,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=RED, shrinkB=6, **ann_kw),
    )

    # Right panel: wide box around the all-gold FN row
    fn2_start = s2["tp"]  # first FN index = 30
    fn2_full_row = fn2_start // 10  # row index = 3
    fn2_row_pos = (9 - fn2_full_row) * main_sp  # y-position = 5.4
    ax2.add_patch(
        Rectangle(
            (-pad - 0.06, fn2_row_pos - pad),
            9 * main_sp + 2 * (pad + 0.06),
            2 * pad,
            linewidth=2.5,
            edgecolor=GOLD_DARK,
            facecolor="none",
            zorder=5,
        )
    )
    # x = midpoint between col 1 (0.90) and col 2 (1.80), away from TN annotation
    fn_arrow_x = 1.5 * main_sp
    ax2.annotate(
        "Missed patients\n(bad!)",
        xy=(fn_arrow_x, fn2_row_pos - pad),
        xytext=(fn_arrow_x, fn2_row_pos - 3.0),
        fontsize=16,
        color=GOLD_DARK,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GOLD_DARK, shrinkB=6, **ann_kw),
    )

    # Right panel: box around the TN dot at the same position as the rejected FP
    tn2_idx = s2["tp"] + s2["fn"] + s2["fp"]  # first TN index = 46
    tn2_col = (tn2_idx % 10) * main_sp  # col 6 → x = 5.4
    tn2_row = (9 - tn2_idx // 10) * main_sp  # row 4 → y = 4.5
    ax2.add_patch(
        Rectangle(
            (tn2_col - pad, tn2_row - pad),
            2 * pad,
            2 * pad,
            linewidth=2.5,
            edgecolor=GRAY_TEXT,
            facecolor="none",
            zorder=5,
        )
    )
    ax2.annotate(
        "True negative\n(good!)",
        xy=(tn2_col + pad, tn2_row),
        xytext=(tn2_col + 2.2, tn2_row - 1.8),
        fontsize=16,
        color=GRAY_TEXT,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY_TEXT, **ann_kw),
    )

    # Legend: 2 cols × 2 rows, column-major fill.
    # Left col (good): Found → Excluded.  Right col (bad): Missed → Ghost.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BLUE_DARK,
            markersize=15,
            label="Found (true positive)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GRAY,
            markersize=15,
            label="Excluded (true negative)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GOLD_DARK,
            markersize=15,
            label="Missed (false negative)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=RED,
            markersize=15,
            label="Ghost (false positive)",
        ),
    ]
    # Place legend centered in the gap between dot grids and metric panels.
    # y=0.228 is the visual midpoint of that gap with top=0.87, hspace=0.25.
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.50, 0.28),
        ncol=2,
        fontsize=20,
        frameon=False,
    )

    # pad_inches=0.30 preserves white space above titles when bbox_inches='tight' crops.
    save(fig, figures_dir, "figure_4_2_phenotype_tradeoff", pad_inches=0.30)


def national_anchor(outputs: Path, figures_dir: Path) -> None:
    bridge = pd.read_csv(outputs / "nhanes_bridge.csv")
    bridge = bridge.sort_values("region")
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    y = np.arange(len(bridge))
    target_m = bridge["target_diagnosed_population"] / 1e6
    bars = ax.barh(
        y, target_m, color=BLUE, edgecolor=BLUE_DARK, linewidth=1.0, zorder=3
    )
    ax.invert_yaxis()
    for bar, row in zip(bars, bridge.itertuples()):
        ax.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{row.target_diagnosed_population / 1e6:.1f}M from {int(row.observed_diagnosed_patients):,} panel patients",
            va="center",
            fontsize=10,
        )
        ax.text(
            0.05,
            bar.get_y() + bar.get_height() / 2,
            f"weight {row.population_weight:,.0f}",
            va="center",
            fontsize=9,
            color=GRAY_TEXT,
        )
    total = bridge["target_diagnosed_population"].sum() / 1e6
    ax.set_yticks(y, bridge["region"])
    ax.set_xlim(0, target_m.max() * 1.45)
    ax.set_xlabel("Diagnosed adults, millions")
    ax.set_title("Regional calibration weights", fontsize=18, fontweight="bold", pad=14)
    ax.text(
        target_m.max() * 1.42,
        -0.55,
        f"National anchor: {total:.1f}M",
        fontsize=10.5,
        fontweight="bold",
        color=GREEN_DARK,
        ha="right",
    )
    ax.grid(axis="x", color=GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, figures_dir, "figure_4_4_national_anchor")


def market_funnel(outputs: Path, figures_dir: Path) -> None:
    funnel = pd.read_csv(outputs / "funnel.csv")
    labels = ["Diagnosed", "Age eligible", "Untreated", "Reachable", "Expected starts"]
    values = funnel["population_estimate"].to_numpy() / 1e6
    colors = [BLUE_DARK, "#6A93B5", GOLD, GREEN_DARK, "#76A879"]

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.barh(
        labels[::-1], values[::-1], color=colors[::-1], edgecolor=TEXT, linewidth=0.8
    )
    for bar, value in zip(bars, values[::-1]):
        ax.text(
            value + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}M",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.axhline(1.5, color=TEXT, linewidth=1.1, linestyle=":")
    ax.set_xlim(0, 31)
    ax.set_xlabel("U.S. adults represented, millions", fontsize=11)
    ax.set_title("Patient opportunity funnel", fontsize=18, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, figures_dir, "figure_4_3_market_funnel")


def capture_recapture_figure(outputs: Path, figures_dir: Path) -> None:
    cr = pd.read_csv(outputs / "capture_recapture.csv")
    clean = cr.iloc[1]
    n1 = int(clean["source_a_count"])
    n2 = int(clean["source_b_count"])
    overlap = int(clean["overlap_count"])
    estimate = float(clean["chapman_estimate"])
    unseen = estimate - (n1 + n2 - overlap)

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    fig.suptitle(
        "Capture-recapture from two sources", fontsize=18, fontweight="bold", y=0.99
    )
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.0, 7.5)
    ax.axis("off")

    # Circles shifted upward to leave room for text below
    ax.add_patch(
        Circle(
            (4.3, 4.5), 2.75, facecolor="#F3F5F7", edgecolor="#7A858E", linewidth=1.6
        )
    )
    ax.add_patch(
        Circle((3.45, 4.6), 1.75, facecolor=BLUE, edgecolor=BLUE_DARK, alpha=0.82)
    )
    ax.add_patch(
        Circle((5.25, 4.6), 1.55, facecolor=GREEN, edgecolor=GREEN_DARK, alpha=0.82)
    )

    # Leader lines end inside each colored circle
    ax.annotate(
        f"Paid diagnosis\n$n_1$ = {n1:,}",
        xy=(2.85, 5.1),  # inside blue circle (center 3.45, 4.6, r=1.75)
        xytext=(1.3, 7.0),
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="-", color=BLUE_DARK, lw=1.2),
    )
    ax.annotate(
        f"Paid Roventra\n$n_2$ = {n2:,}",
        xy=(5.9, 5.1),  # inside green circle (center 5.25, 4.6, r=1.55)
        xytext=(9.0, 7.0),
        fontsize=10.5,
        fontweight="bold",
        ha="right",
        arrowprops=dict(arrowstyle="-", color=GREEN_DARK, lw=1.2),
    )
    ax.text(
        4.32,
        4.6,
        f"Both\n$m$ = {overlap:,}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        4.3,
        2.2,
        f"Estimated missed by both: {unseen:,.0f}",
        ha="center",
        fontsize=10,
        color=GRAY_TEXT,
    )
    ax.text(
        4.3,
        0.5,
        f"Chapman estimate: {estimate:,.0f}",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=GREEN_DARK,
    )

    save(fig, figures_dir, "figure_4_4_capture_recapture")


def uncertainty_tornado(outputs: Path, figures_dir: Path) -> None:
    layers = pd.read_csv(outputs / "uncertainty_layers.csv").sort_values("width")
    point = float(layers["point"].iloc[0]) / 1e6
    y = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    for i, row in enumerate(layers.itertuples()):
        low, high = row.low / 1e6, row.high / 1e6
        ax.barh(
            i,
            high - low,
            left=low,
            height=0.55,
            color=BLUE,
            edgecolor=BLUE_DARK,
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            low - 0.05,
            i,
            f"{low:.2f}M",
            va="center",
            ha="right",
            fontsize=9,
            color=GRAY_TEXT,
        )
        ax.text(
            high + 0.05,
            i,
            f"{high:.2f}M",
            va="center",
            ha="left",
            fontsize=9,
            color=GRAY_TEXT,
        )
    ax.axvline(point, color=TEXT, linewidth=1.2, linestyle=":")
    ax.text(
        point,
        -0.95,
        f"point {point:.2f}M",
        ha="center",
        va="top",
        fontsize=9,
        color=TEXT,
    )
    ax.set_yticks(y, layers["layer"])
    ax.set_ylim(-1.1, len(layers) - 0.3)
    ax.set_xlim(3.9, 6.0)
    ax.set_xlabel("Reachable opportunity, millions", fontsize=11)
    ax.set_title(
        "Uncertainty layers on reachable opportunity",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, figures_dir, "figure_4_8_uncertainty_tornado")


def account_stability(outputs: Path, figures_dir: Path) -> None:
    ranks = pd.read_csv(outputs / "account_rank_stability.csv").sort_values(
        "point_rank"
    )
    y = np.arange(len(ranks))
    low = ranks["median_bootstrap_rank"] - ranks["rank_p5"]
    high = ranks["rank_p95"] - ranks["median_bootstrap_rank"]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axvspan(0.5, 5.5, color=GREEN, alpha=0.35, zorder=0)
    ax.errorbar(
        ranks["median_bootstrap_rank"],
        y,
        xerr=[low, high],
        fmt="o",
        color=BLUE_DARK,
        ecolor=BLUE_DARK,
        capsize=6,
        markersize=8,
        linewidth=2,
    )
    ax.scatter(
        ranks["point_rank"], y, marker="D", color=GOLD, edgecolor=TEXT, s=60, zorder=3
    )
    ax.set_yticks(y, ranks["account_name"])
    ax.invert_yaxis()
    for row_index, row in ranks.reset_index(drop=True).iterrows():
        ax.text(
            91,
            row_index,
            f"{row['share_of_replicates_in_top5']:.0%} in top 5",
            va="center",
            fontsize=9,
            color=GRAY_TEXT,
        )
    ax.set_xlim(0, 101)
    ax.set_xlabel("Bootstrap rank, lower is better", fontsize=11)
    ax.set_title("Account rank stability", fontsize=18, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(axis="x", color=GRAY, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GOLD,
            markeredgecolor=TEXT,
            markersize=8,
            label="Point rank",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE_DARK,
            markerfacecolor=BLUE_DARK,
            markersize=7,
            linewidth=2,
            label="Bootstrap median and 90% interval",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.58, 1.02),
        ncol=2,
        fontsize=9,
    )
    save(fig, figures_dir, "figure_4_9_account_rank_stability")


def state_map(outputs: Path, figures_dir: Path) -> None:
    import plotly.express as px

    state_df = pd.read_csv(outputs / "state_opportunity.csv")
    fig = px.choropleth(
        state_df,
        locations="state",
        locationmode="USA-states",
        color="reachable_opportunity",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"reachable_opportunity": "Reachable opportunity"},
    )
    fig.update_layout(
        title=dict(
            text="Reachable opportunity by state",
            x=0.5,
            xanchor="center",
            font=dict(size=22, color=TEXT),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=560,
        width=1000,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(figures_dir / "figure_4_10_state_opportunity_map.png"), scale=2)
    fig.write_image(str(figures_dir / "figure_4_10_state_opportunity_map.svg"))


def main() -> None:
    chapter_dir = Path(__file__).resolve().parents[1]
    outputs = chapter_dir / "assets" / "generated_outputs"
    figures_dir = chapter_dir / "assets" / "figures"
    market_sizes(outputs, figures_dir)
    phenotype_tradeoff(outputs, figures_dir)
    market_funnel(outputs, figures_dir)
    capture_recapture_figure(outputs, figures_dir)
    figure_4_5_under_observation(outputs, figures_dir)
    figure_4_6_model_lift(outputs, figures_dir)
    print(f"Wrote Chapter 4 figures to {figures_dir}")


if __name__ == "__main__":
    main()
