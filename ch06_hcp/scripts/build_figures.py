"""Build the quantitative and conceptual figures for Chapter 6."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "ch06_hcp"
OUT = CHAPTER / "assets" / "generated_outputs"
FIG = CHAPTER / "assets" / "figures"
SOURCE = FIG / "source"

INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"
BLUE = "#5477C4"
BLUE_LIGHT = "#CEDFFE"
GOLD = "#B8A037"
GOLD_LIGHT = "#FFEA8F"
GREEN = "#71B436"
GREEN_LIGHT = "#BEEB96"
ORANGE = "#CC6F47"
ORANGE_LIGHT = "#FFBDA1"
GRAY = "#7A828F"
GRAY_LIGHT = "#E2E5EA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def header(fig: plt.Figure, title: str) -> None:
    """Add the centered figure title required by the book style."""

    fig.suptitle(
        textwrap.fill(title, 70),
        x=0.5,
        y=0.965,
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
    )


def finish(fig: plt.Figure, stem: str) -> None:
    """Export an editable SVG and a chapter-ready PNG."""

    FIG.mkdir(parents=True, exist_ok=True)
    svg_path = FIG / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(FIG / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def write_targeting_excalidraw() -> None:
    """Write the editable Excalidraw source for the conceptual workflow."""

    updated = 1781841600000

    def common(element_id: str, element_type: str, x: float, y: float, width: float, height: float) -> dict:
        return {
            "id": element_id,
            "type": element_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": INK,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": 6000 + len(element_id),
            "version": 1,
            "versionNonce": 7000 + len(element_id),
            "isDeleted": False,
            "boundElements": [],
            "updated": updated,
            "link": None,
            "locked": False,
        }

    def rectangle(element_id: str, x: float, y: float, width: float, height: float, stroke: str, fill: str) -> dict:
        element = common(element_id, "rectangle", x, y, width, height)
        element.update(
            {
                "strokeColor": stroke,
                "backgroundColor": fill,
                "roundness": {"type": 3},
            }
        )
        return element

    def text_element(element_id: str, x: float, y: float, width: float, height: float, text: str, font_size: int, bold: bool = False) -> dict:
        element = common(element_id, "text", x, y, width, height)
        element.update(
            {
                "text": text,
                "fontSize": font_size,
                "fontFamily": 2,
                "textAlign": "center",
                "verticalAlign": "middle",
                "containerId": None,
                "originalText": text,
                "autoResize": True,
                "lineHeight": 1.25,
                "strokeColor": INK if bold else MUTED,
            }
        )
        return element

    def arrow(element_id: str, x: float, y: float, dx: float, dy: float) -> dict:
        element = common(element_id, "arrow", x, y, abs(dx), abs(dy))
        element.update(
            {
                "points": [[0, 0], [dx, dy]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
            }
        )
        return element

    elements = [
        text_element("ch06-title", 350, 28, 500, 55, "Targeting workflow", 36, bold=True),
    ]
    boxes = [
        ("opportunity", 60, "#5477C4", "#CEDFFE", "Opportunity", "Patients\nTreatment mix"),
        ("account", 300, "#5477C4", "#CEDFFE", "Account context", "Account\nTerritory"),
        ("gates", 540, "#B8A037", "#FFEA8F", "Action gates", "Evidence\nPermission\nCapacity"),
        ("hcp", 780, "#71B436", "#BEEB96", "HCP action", "Prioritize\nMaintain\nHold"),
        ("plan", 1030, "#71B436", "#BEEB96", "Call plan", ""),
    ]
    for index, (name, x, stroke, fill, title, body) in enumerate(boxes):
        width = 190 if name != "plan" else 105
        elements.append(rectangle(f"ch06-{name}-box", x, 175, width, 160, stroke, fill))
        elements.append(text_element(f"ch06-{name}-title", x, 195, width, 40, title, 24, bold=True))
        if body:
            elements.append(text_element(f"ch06-{name}-body", x, 240, width, 75, body, 20))
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][1]
            elements.append(arrow(f"ch06-arrow-{index + 1}", x + width, 255, next_x - x - width, 0))
    elements.extend(
        [
            text_element(
                "ch06-review-text",
                300,
                430,
                600,
                36,
                "Review loop: record overrides, refresh evidence, and measure execution",
                18,
            ),
            arrow("ch06-review-arrow", 900, 390, -720, 0),
        ]
    )
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "hands-on-pharma-decision-science",
        "elements": elements,
        "appState": {
            "gridSize": 20,
            "gridStep": 5,
            "gridModeEnabled": False,
            "viewBackgroundColor": "#FFFFFF",
        },
        "files": {},
    }
    SOURCE.mkdir(parents=True, exist_ok=True)
    (SOURCE / "figure_6_1_targeting_workflow.excalidraw").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def figure_targeting_workflow() -> None:
    write_targeting_excalidraw()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.subplots_adjust(top=0.84, left=0.06, right=0.96, bottom=0.12)
    header(fig, "Targeting workflow")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.15)
    ax.axis("off")
    boxes = [
        (0.3, 1.55, 2.15, 1.45, "Opportunity", "Patients\nTreatment mix", BLUE_LIGHT, BLUE),
        (3.05, 1.55, 2.15, 1.45, "Account context", "Account\nTerritory", BLUE_LIGHT, BLUE),
        (5.8, 1.55, 2.15, 1.45, "Action gates", "Evidence\nPermission\nCapacity", GOLD_LIGHT, GOLD),
        (8.55, 1.55, 2.15, 1.45, "HCP action", "Prioritize\nMaintain\nHold", GREEN_LIGHT, GREEN),
    ]
    for x, y, w, h, title, body, fill, edge in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fill, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", weight="bold", fontsize=13)
        ax.text(x + w / 2, y + 0.52, body, ha="center", va="center", fontsize=11.5)
    for x1, x2 in [(2.45, 3.05), (5.2, 5.8), (7.95, 8.55), (10.7, 11.35)]:
        ax.add_patch(
            FancyArrowPatch((x1, 2.27), (x2, 2.27), arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.4, color=INK)
        )
    final = FancyBboxPatch(
        (11.35, 1.55), 0.6, 1.45, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=1.5,
    )
    ax.add_patch(final)
    ax.text(11.65, 2.27, "Call\nplan", ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(6.0, 0.06, "Review loop: record overrides, refresh evidence, and measure execution", ha="center",
            color=MUTED, fontsize=11.5)
    ax.add_patch(
        FancyArrowPatch((10.2, 1.25), (1.6, 1.25), connectionstyle="arc3,rad=-0.10",
                        arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=GRAY)
    )
    finish(fig, "figure_6_1_targeting_workflow")


def figure_volume_trap(hcp: pd.DataFrame) -> None:
    top = hcp.nlargest(15, "cohort_patients").sort_values("cohort_patients")
    labels = [f"HCP {str(value)[-4:]}" for value in top["npi"]]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 8.2))
    fig.subplots_adjust(top=0.86, left=0.17, right=0.94, bottom=0.11)
    header(fig, "Top 15 HCPs by attributed patient volume")
    ax.barh(y, top["cohort_patients"], color=BLUE_LIGHT, edgecolor=BLUE, label="Cohort patients")
    ax.barh(y, top["opportunity_patients"], color=GOLD_LIGHT, edgecolor=GOLD, label="Opportunity patients")
    blocked = ~top["contact_permitted"]
    ax.scatter(
        top.loc[blocked, "cohort_patients"] + 0.8,
        y[blocked],
        marker="X",
        s=95,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Latest consent: Opt-out",
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Patients")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    clean_axes(ax, "x")
    finish(fig, "figure_6_2_volume_trap")


def figure_decile_diagnostic(deciles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.96, bottom=0.13, wspace=0.28)
    header(fig, "HCP volume deciles and current contact permission")
    x = deciles["volume_decile"]
    axes[0].bar(x, deciles["cohort_patients"], color=BLUE_LIGHT, edgecolor=BLUE, label="Cohort")
    axes[0].bar(x, deciles["opportunity_patients"], color=GOLD_LIGHT, edgecolor=GOLD, label="Opportunity")
    axes[0].set_title("Patient evidence", loc="left", weight="bold")
    axes[0].set_xlabel("Volume decile")
    axes[0].set_ylabel("Patients")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend(frameon=False, loc="upper left")
    clean_axes(axes[0], "y")

    share = deciles["contactable_share"] * 100
    axes[1].plot(x, share, color=ORANGE, marker="o", markersize=7, linewidth=1.8)
    for xi, yi in zip(x, share, strict=True):
        axes[1].text(xi, yi + 1.8, f"{yi:.0f}%", ha="center", fontsize=9.5, color=INK)
    axes[1].set_title("Contact permission", loc="left", weight="bold")
    axes[1].set_xlabel("Volume decile")
    axes[1].set_ylabel("HCPs with current permission")
    axes[1].set_ylim(50, 90)
    axes[1].set_xticks(range(1, 11))
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    clean_axes(axes[1], "y")
    finish(fig, "figure_6_3_decile_diagnostic")


def figure_account_action_map(accounts: pd.DataFrame) -> None:
    colors = {
        "Increase priority": BLUE,
        "Maintain": GREEN,
        "Access review": ORANGE,
        "Hold contact": GOLD,
        "Monitor": GRAY,
    }
    markers = {
        "Increase priority": "o",
        "Maintain": "s",
        "Access review": "X",
        "Hold contact": "^",
        "Monitor": ".",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(top=0.84, left=0.11, right=0.95, bottom=0.12)
    header(fig, "Account targeting actions")
    for action in colors:
        part = accounts.loc[accounts["account_action"].eq(action)]
        ax.scatter(
            part["roventra_share"] * 100,
            part["opportunity_patients"],
            s=part["cohort_patients"] * 5,
            color=colors[action],
            marker=markers[action],
            alpha=0.78 if action != "Monitor" else 0.45,
            edgecolor="white",
            linewidth=0.8,
            label=action,
        )
    benchmark = accounts["launch_share_benchmark"].iloc[0] * 100
    ax.axvline(benchmark, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.text(
        benchmark + 0.8,
        1.0,
        f"Benchmark {benchmark:.1f}%",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=10,
        color=INK,
    )
    labels = pd.concat(
        [
            accounts.loc[accounts["account_action"].eq("Increase priority")].nlargest(3, "opportunity_patients"),
            accounts.loc[accounts["account_action"].eq("Access review")],
            accounts.loc[accounts["account_action"].eq("Hold contact")].nlargest(2, "opportunity_patients"),
        ]
    ).drop_duplicates("account_id")
    offsets = [(8, 8), (8, -14), (-8, 9), (8, 10), (-8, -15), (8, 8), (-8, 9)]
    for (_, row), offset in zip(labels.iterrows(), offsets, strict=False):
        ax.annotate(
            row["account_name"],
            (row["roventra_share"] * 100, row["opportunity_patients"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha="left" if offset[0] > 0 else "right",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": GRID, "alpha": 0.92},
        )
    ax.set_xlabel("Roventra share among treated patients")
    ax.set_ylabel("Opportunity patients")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    clean_axes(ax, "both")
    finish(fig, "figure_6_4_account_action_map")


def figure_gate_attrition(gates: pd.DataFrame) -> None:
    plot = gates.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.subplots_adjust(top=0.84, left=0.23, right=0.94, bottom=0.12)
    header(fig, "Sequential account gates")
    colors = [GREEN] + [GOLD_LIGHT] * (len(plot) - 2) + [BLUE_LIGHT]
    bars = ax.barh(plot["stage"], plot["accounts"], color=colors, edgecolor=INK, linewidth=0.7)
    for bar, value in zip(bars, plot["accounts"], strict=True):
        ax.text(value + 2, bar.get_y() + bar.get_height() / 2, f"{value}", va="center", weight="bold")
    ax.set_xlim(0, 180)
    ax.set_xlabel("Accounts remaining")
    clean_axes(ax, "x")
    finish(fig, "figure_6_5_gate_attrition")


def figure_territory_allocation(territory: pd.DataFrame) -> None:
    plot = territory.sort_values("opportunity_share")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(top=0.84, left=0.14, right=0.94, bottom=0.12)
    header(fig, "Territory opportunity and planned calls")
    ax.hlines(y, plot["opportunity_share"] * 100, plot["call_share"] * 100, color=GRAY_LIGHT, linewidth=3)
    ax.scatter(plot["opportunity_share"] * 100, y, color=BLUE, s=85, label="Actionable opportunity")
    ax.scatter(plot["call_share"] * 100, y, color=GREEN, s=85, marker="s", label="Recommended calls")
    ax.set_yticks(y, plot["territory"])
    ax.set_xlabel("Share of chapter total")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(loc="lower right", frameon=False)
    clean_axes(ax, "x")
    finish(fig, "figure_6_6_territory_allocation")


def main() -> None:
    figure_targeting_workflow()
    figure_volume_trap(pd.read_csv(OUT / "hcp_targets.csv"))
    figure_decile_diagnostic(pd.read_csv(OUT / "decile_summary.csv"))
    figure_account_action_map(pd.read_csv(OUT / "account_targets.csv"))
    figure_gate_attrition(pd.read_csv(OUT / "gate_summary.csv"))
    figure_territory_allocation(pd.read_csv(OUT / "territory_summary.csv"))
    print(f"Wrote Chapter 6 figures to {FIG}")


if __name__ == "__main__":
    main()
