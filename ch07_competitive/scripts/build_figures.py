"""Build publication figures for Chapter 7."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


BLUE = "#3F72C8"
GOLD = "#C89400"
GREEN = "#2E8B57"
RED = "#C83E2B"
ORANGE = "#D97706"
PURPLE = "#7A5AA6"
GRAY = "#7C8795"
LIGHT_BLUE = "#DCE8F8"
LIGHT_GOLD = "#F5E8B8"
LIGHT_GREEN = "#DCEEDF"
LIGHT_RED = "#F6DDD7"
LIGHT_GRAY = "#EDF0F3"

ACTION_COLORS = {
    "Access work": RED,
    "Adoption review": ORANGE,
    "Dual workstream": PURPLE,
    "Defend and learn": GREEN,
    "Monitor": GRAY,
}
ACTION_MARKERS = {
    "Access work": "^",
    "Adoption review": "D",
    "Dual workstream": "X",
    "Defend and learn": "o",
    "Monitor": "s",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    }
)


def _save(fig: plt.Figure, filename: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{filename}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{filename}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def figure_7_1_evidence_chain(directory: Path) -> None:
    """Show the evidence chain from policy to action."""

    labels = [
        ("Policy state", "Tier, PA, step", BLUE, LIGHT_BLUE),
        ("Lives exposed", "Plan enrollment", BLUE, LIGHT_BLUE),
        ("Attempt outcome", "Paid or unresolved", RED, LIGHT_RED),
        ("Corrected starts", "180-day washout", GOLD, LIGHT_GOLD),
        ("Action", "Owner and reason", GREEN, LIGHT_GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.7))
    x_positions = np.linspace(0.5, 10.0, len(labels))
    for index, (title, detail, edge, face) in enumerate(labels):
        x = x_positions[index]
        box = FancyBboxPatch(
            (x - 0.9, 0.55),
            1.8,
            0.95,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 1.15, title, ha="center", va="center", fontweight="bold")
        ax.text(x, 0.82, detail, ha="center", va="center", fontsize=8, color="#333333")
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 1.0, 1.02),
                xytext=(x + 1.0, 1.02),
                arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.4},
            )
    ax.set_xlim(-0.6, 11.1)
    ax.set_ylim(0.25, 1.9)
    ax.set_title("From Access Evidence to a Defensible Action", fontweight="bold")
    ax.axis("off")
    _save(fig, "figure_7_1_evidence_chain", directory)


def figure_7_2_access_lives(restrictions: pd.DataFrame, directory: Path) -> None:
    """Show mutually exclusive lives by access state."""

    order = ["Non-covered", "Step edit", "Prior authorization", "Unrestricted"]
    colors = {
        "Non-covered": RED,
        "Step edit": ORANGE,
        "Prior authorization": GOLD,
        "Unrestricted": GREEN,
    }
    data = restrictions.set_index("access_state").reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = 0
    total = data["enrolled_lives"].sum()
    for state in order:
        value = data.loc[state, "enrolled_lives"]
        if value <= 0:
            continue
        ax.barh(["Roventra"], [value], left=left, color=colors[state], label=state)
        ax.text(
            left + value / 2,
            0,
            f"{state}\n{value / 1_000_000:.1f}M\n{value / total:.0%}",
            ha="center",
            va="center",
            color="white" if state in {"Non-covered", "Step edit"} else "#222222",
            fontsize=8,
            fontweight="bold",
        )
        left += value
    ax.set_xlabel("Synthetic enrolled lives")
    ax.set_title("Roventra Lives by Access State", fontweight="bold")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value / 1_000_000:.0f}M")
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    _save(fig, "figure_7_2_access_lives", directory)


def figure_7_3_payer_region_matrix(decisions: pd.DataFrame, directory: Path) -> None:
    """Align share uncertainty, restricted lives, friction, and action."""

    action_order = {
        "Dual workstream": 0,
        "Access work": 1,
        "Adoption review": 2,
        "Defend and learn": 3,
        "Monitor": 4,
    }
    data = decisions.copy()
    data["action_order"] = data["action"].map(action_order)
    data = data.sort_values(
        ["action_order", "restricted_lives", "treated_patients"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    data["label"] = data["payer_id"] + " " + data["region"].str[:2].str.upper()
    y = np.arange(len(data))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13, 10),
        sharey=True,
        gridspec_kw={"width_ratios": [2.7, 1.6, 1.4, 1.4]},
    )
    low = data["brand_share"] - data["share_lower_95"]
    high = data["share_upper_95"] - data["brand_share"]
    axes[0].errorbar(
        data["brand_share"],
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        capsize=2,
        markersize=4,
    )
    axes[0].axvline(0.82, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_xlim(0.60, 1.01)
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axes[0].set_xlabel("Roventra share")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["label"], fontsize=7)

    axes[1].barh(y, data["restricted_lives"] / 1000, color=RED, alpha=0.75)
    axes[1].set_xlabel("Restricted lives (000s)")

    axes[2].barh(y, data["pend_exposure_rate"], color=GOLD, alpha=0.8)
    axes[2].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axes[2].set_xlabel("Attempts with pend")

    for index, row in data.iterrows():
        axes[3].scatter(
            0.5,
            index,
            marker=ACTION_MARKERS[row["action"]],
            color=ACTION_COLORS[row["action"]],
            s=45,
        )
    axes[3].set_xlim(0, 1)
    axes[3].set_xticks([])
    axes[3].set_xlabel("Action")
    for ax in axes:
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Payer-Region Evidence Matrix", fontweight="bold", y=0.995)
    fig.tight_layout()
    _save(fig, "figure_7_3_payer_region_matrix", directory)


def figure_7_4_attempt_trace(trace: pd.DataFrame, directory: Path) -> None:
    """Show PAT02034 transactions collapsing into completed attempts."""

    data = trace.sort_values("fill_number").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for row_index, row in data.iterrows():
        y = len(data) - row_index
        start = pd.Timestamp(row["first_submission_date"])
        end = pd.Timestamp(row["last_transaction_date"])
        ax.plot([start, end], [y, y], color=BLUE, linewidth=3)
        ax.scatter(start, y, color=RED if row["had_pend"] else BLUE, s=55, zorder=3)
        ax.scatter(end, y, color=GREEN, marker="s", s=55, zorder=3)
        chain_label = "PENDED to PAID" if row["had_pend"] else "Submitted and PAID"
        if row["had_reversal"]:
            chain_label = "PAID, REVERSED, PAID"
        midpoint = start + (end - start) / 2
        ax.text(midpoint, y + 0.24, chain_label, ha="center", fontsize=8)
        ax.text(
            midpoint,
            y - 0.28,
            f"{int(row['transaction_rows'])} transaction rows",
            ha="center",
            fontsize=8,
            color=GRAY,
        )
    ax.set_yticks(range(1, len(data) + 1))
    ax.set_yticklabels(
        [
            f"Fill {fill_number}"
            for fill_number in reversed(data["fill_number"].tolist())
        ]
    )
    ax.set_xlabel("2024 service date")
    ax.set_title(
        "PAT02034: Transactions Collapse into 4 Completed Attempts", fontweight="bold"
    )
    ax.set_ylim(0.45, len(data) + 0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "figure_7_4_attempt_trace", directory)


def figure_7_5_decision_map(decisions: pd.DataFrame, directory: Path) -> None:
    """Show access quality and posterior adoption evidence with action encoding."""

    rng = np.random.default_rng(7)
    data = decisions.copy()
    data["x_plot"] = data["access_quality_weight"] + rng.normal(0, 0.012, len(data))
    fig, ax = plt.subplots(figsize=(9, 6))
    for action, group in data.groupby("action"):
        ax.scatter(
            group["x_plot"],
            group["posterior_mean_share"],
            s=np.sqrt(group["enrolled_lives"]) * 1.5,
            marker=ACTION_MARKERS[action],
            color=ACTION_COLORS[action],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=action,
        )
    ax.axhline(0.82, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("Access-quality score (scenario weighted)")
    ax.set_ylabel("Partially pooled Roventra share")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.74, 0.90)
    for _, row in data.nlargest(3, "restricted_lives").iterrows():
        ax.annotate(
            f"{row['payer_id']} {row['region'][:2].upper()}",
            (row["x_plot"], row["posterior_mean_share"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_title(
        "Access and Adoption Evidence Produce Different Actions", fontweight="bold"
    )
    ax.legend(frameon=False, ncol=2, fontsize=8, markerscale=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "figure_7_5_decision_map", directory)


def figure_7_6_formulary_event(
    fitted: pd.DataFrame,
    effect: pd.DataFrame,
    change_week: int,
    directory: Path,
) -> None:
    """Show observed and counterfactual shares plus the estimated effect."""

    summary = effect.iloc[0]
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    top.plot(
        fitted["week"], fitted["actual"], color=BLUE, linewidth=1.2, label="Observed"
    )
    top.plot(
        fitted["week"],
        fitted["fitted"],
        color=BLUE,
        linewidth=2.2,
        label="Controlled ITS",
    )
    top.plot(
        fitted["week"],
        fitted["counterfactual"],
        color=GRAY,
        linestyle="--",
        linewidth=1.8,
        label="Counterfactual",
    )
    top.axvline(change_week, color=RED, linestyle=":", linewidth=1.5)
    top.set_ylabel("Roventra share")
    top.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    top.legend(frameon=False, ncol=3)
    top.spines[["top", "right"]].set_visible(False)

    bottom.plot(fitted["week"], fitted["effect"], color=GREEN, linewidth=2)
    bottom.axhline(0, color=GRAY, linewidth=0.8)
    bottom.axvline(change_week, color=RED, linestyle=":", linewidth=1.5)
    bottom.errorbar(
        [summary["effect_week"]],
        [summary["effect_at_week"]],
        yerr=[
            [summary["effect_at_week"] - summary["effect_at_week_lower_95"]],
            [summary["effect_at_week_upper_95"] - summary["effect_at_week"]],
        ],
        fmt="o",
        color=GREEN,
        capsize=4,
    )
    bottom.set_xlabel("Week in 2024")
    bottom.set_ylabel("Estimated effect")
    bottom.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:+.0%}")
    )
    bottom.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "PAY004 Formulary Improvement: Controlled Interrupted Time Series",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "figure_7_6_formulary_event", directory)


def figure_7_7_account_actions(accounts: pd.DataFrame, directory: Path) -> None:
    """Show account action counts and the largest restricted patient pools."""

    counts = accounts["action"].value_counts().reindex(ACTION_COLORS).dropna()
    top_accounts = accounts.nlargest(8, "restricted_patients").sort_values(
        "restricted_patients"
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.barh(
        counts.index,
        counts.values,
        color=[ACTION_COLORS[action] for action in counts.index],
    )
    for index, value in enumerate(counts.values):
        left.text(value + 0.8, index, f"{int(value)}", va="center")
    left.set_xlabel("Accounts")
    left.set_title("Action Queue")
    left.spines[["top", "right", "left"]].set_visible(False)
    left.tick_params(axis="y", length=0)

    right.barh(
        top_accounts["account_id"],
        top_accounts["restricted_patients"],
        color=[ACTION_COLORS[action] for action in top_accounts["action"]],
    )
    right.set_xlabel("Attributed patients with material access barrier")
    right.set_title("Largest Restricted Patient Pools")
    right.spines[["top", "right", "left"]].set_visible(False)
    right.tick_params(axis="y", length=0)
    fig.suptitle(
        "Account Actions Preserve Access and Adoption Evidence", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "figure_7_7_account_actions", directory)


def figure_7_8_switch_support(evidence: pd.DataFrame, directory: Path) -> None:
    """Show switch-event support and the honest median status."""

    data = evidence.sort_values("patients")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(data["first_regimen"], data["switch_events"], color=BLUE)
    for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['switch_events'])} switches; median not reached",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Observed switch events")
    ax.set_title(
        "The Cohort Does Not Support a Comparative Switch Median", fontweight="bold"
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save(fig, "figure_7_8_switch_support", directory)


def build_all(results: dict[str, pd.DataFrame], directory: Path) -> None:
    """Build every Chapter 7 figure from the shared results dictionary."""

    from ch07_competitive.generation_modules.ch07_config import ITS_CHANGE_WEEK

    figure_7_1_evidence_chain(directory)
    figure_7_2_access_lives(results["restriction_lives"], directory)
    figure_7_3_payer_region_matrix(results["payer_region_decisions"], directory)
    figure_7_4_attempt_trace(results["pat02034_attempt_trace"], directory)
    figure_7_5_decision_map(results["payer_region_decisions"], directory)
    figure_7_6_formulary_event(
        results["formulary_event_fitted"],
        results["formulary_event_effect"],
        ITS_CHANGE_WEEK,
        directory,
    )
    figure_7_7_account_actions(results["account_access_adoption_actions"], directory)
    figure_7_8_switch_support(results["switch_evidence"], directory)
    print(f"Built 8 Chapter 7 figures in {directory}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from ch07_competitive.scripts.run_analysis import run_analysis

    build_all(
        run_analysis(root),
        root / "ch07_competitive" / "assets" / "figures",
    )
