"""Build Chapter 14 figures from the current forecasting analysis."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import forecasting as fc  # noqa: E402
from forecast_config import GENERIC_ENTRY_DATE, LAUNCH_DATE  # noqa: E402
from run_analysis import run_analysis  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

TOKENS = {
    "ink": "#1f2937",
    "muted": "#667085",
    "grid": "#d9dee8",
    "blue": "#3a6ea5",
    "gold": "#b4892f",
    "green": "#2e7d32",
    "orange": "#b65d1f",
    "red": "#b42318",
    "purple": "#7c5cbf",
    "teal": "#1b7f79",
    "magenta": "#a13d5a",
    "gray": "#98a2b3",
    "soft_blue": "#dbeafe",
    "soft_gold": "#fef3c7",
    "soft_green": "#dcfce7",
    "soft_orange": "#fed7aa",
    "soft_red": "#fee4e2",
    "soft_gray": "#f3f4f6",
}


def _save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    svg = output_dir / f"{name}.svg"
    png = output_dir / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(color=TOKENS["grid"], alpha=0.75, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["grid"])
    ax.spines["bottom"].set_color(TOKENS["grid"])
    ax.tick_params(colors=TOKENS["muted"])
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])
    ax.title.set_color(TOKENS["ink"])


TRAIN_WEEKS = 44


def _holdout_base_chart(ax: plt.Axes, weeks: np.ndarray, trx: np.ndarray, train_weeks: int = TRAIN_WEEKS) -> None:
    """Shared base for every 14.3 method figure: solid training history, dashed held-out actual, shaded scoring window."""
    is_train = weeks <= train_weeks
    ax.axvspan(train_weeks, weeks.max(), color=TOKENS["soft_gray"], zorder=0)
    ax.plot(weeks[is_train], trx[is_train], color=TOKENS["ink"], linewidth=2.2, label="Observed (training)")
    ax.plot(
        weeks[~is_train], trx[~is_train], color=TOKENS["gray"], linewidth=1.8, linestyle="--",
        marker="o", markersize=4, label="Actual (held out)",
    )
    ax.axvline(train_weeks, color=TOKENS["muted"], linestyle=":", linewidth=1.2)
    _style(ax)
    ax.set_xlim(weeks.min(), weeks.max())
    ax.set_xlabel("Week since launch")
    ax.set_ylabel("Weekly TRx")


def _method_forecast_figure(
    observed: pd.DataFrame,
    lines: list[tuple[str, np.ndarray, str, dict]],
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    """A held-out method comparison: base chart plus one or more forecast lines over the scoring window."""
    weeks = np.arange(1, len(observed) + 1, dtype=float)
    trx = observed["trx"].to_numpy()
    holdout_weeks = weeks[-(len(weeks) - TRAIN_WEEKS):]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _holdout_base_chart(ax, weeks, trx)
    for label, values, color, style in lines:
        ax.plot(
            holdout_weeks, values, color=color, linewidth=2.2,
            marker=style.get("marker", "s"), markersize=4,
            linestyle=style.get("linestyle", "-"), label=label,
        )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(title, fontsize=13.5, weight="600", pad=12)
    _save(fig, filename, output_dir)


def _box(ax, x, y, w, h, text, facecolor, edgecolor=None, fontsize=10, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    edgecolor = edgecolor or TOKENS["ink"]
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.25, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TOKENS["ink"],
        wrap=True,
        weight=weight,
    )


def figure_14_1_lifecycle(national: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.plot(national["week_start"], national["trx"], color=TOKENS["blue"], linewidth=1.8)
    observed_end = pd.Timestamp("2025-02-28")
    ax.axvspan(LAUNCH_DATE, observed_end, color=TOKENS["soft_blue"], alpha=0.45)
    ymax = float(national["trx"].max()) * 1.18
    ax.set_ylim(0, ymax)
    label_gap = pd.Timedelta(weeks=3)
    ax.axvline(LAUNCH_DATE, color=TOKENS["green"], linestyle="--", linewidth=1.6)
    ax.text(LAUNCH_DATE + label_gap, ymax * 0.93, "Launch", color=TOKENS["green"], fontsize=10, weight="600", ha="left")
    peak_week = national.loc[national["trx"].idxmax(), "week_start"]
    ax.axvline(peak_week, color=TOKENS["gold"], linestyle="--", linewidth=1.6)
    ax.text(peak_week + label_gap, ymax * 0.93, "Peak", color=TOKENS["gold"], fontsize=10, weight="600", ha="left")
    ax.axvline(GENERIC_ENTRY_DATE, color=TOKENS["red"], linestyle="--", linewidth=1.6)
    ax.text(GENERIC_ENTRY_DATE + label_gap, ymax * 0.93, "Generic entry", color=TOKENS["red"], fontsize=10, weight="600", ha="left")
    ax.text(
        LAUNCH_DATE + (observed_end - LAUNCH_DATE) * 0.5,
        ymax * 0.42,
        "Observed\nwindow",
        color=TOKENS["blue"],
        fontsize=9.5,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    _style(ax)
    ax.set_title("Roventra Lifecycle", fontsize=14, weight="600", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_1_lifecycle_series", output_dir)


def figure_14_2_funnel_timeline(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(observed["week_start"], observed["nbrx"].cumsum(), color=TOKENS["blue"], linewidth=2.0, label="Cumulative new starts (NBRx)")
    ax.plot(observed["week_start"], observed["on_therapy_stock"], color=TOKENS["gold"], linewidth=2.0, label="On-therapy stock")
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Cumulative New Starts and On-Therapy Stock", fontsize=14, weight="600", pad=14)
    ax.set_xlabel("Week")
    ax.set_ylabel("Patients")
    ax.set_ylim(bottom=0)
    _save(fig, "figure_14_2_funnel_timeline", output_dir)


def figure_14_3_bass_toy_shapes(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    toy_months = np.linspace(0, 24, 200)
    innovation_only = fc.bass_cumulative_fraction(toy_months, 0.04, 1e-6) * 100
    imitation_only = fc.bass_cumulative_fraction(toy_months, 0.0005, 0.45) * 100
    blended = fc.bass_cumulative_fraction(toy_months, 0.02, 0.22) * 100
    ax.plot(toy_months, innovation_only, color=TOKENS["gold"], linewidth=2.0, linestyle=":", label="Innovation only")
    ax.plot(toy_months, imitation_only, color=TOKENS["green"], linewidth=2.0, linestyle="--", label="Imitation only")
    ax.plot(toy_months, blended, color=TOKENS["blue"], linewidth=2.0, label="Blended")
    _style(ax)
    ax.set_title("Prescriber Adoption Shapes", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Cumulative adopters")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right")
    _save(fig, "figure_14_3_bass_toy_shapes", output_dir)


def figure_14_4_analog_shapes(results: dict, output_dir: Path) -> None:
    analog_shapes = results["analog_launch_shapes"]
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    palette = [TOKENS["blue"], TOKENS["gold"]]
    for color, column in zip(palette, analog_shapes.columns[1:], strict=True):
        ax.plot(analog_shapes["month"], analog_shapes[column], color=color, linewidth=2.2, label=column)
    _style(ax)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right")
    ax.set_title("Analog Launch Shapes Before Selection", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Cumulative starts as a share of each analog's ceiling")
    ax.set_ylim(0, 1.04)
    _save(fig, "figure_14_4_analog_shapes", output_dir)


def figure_14_5_analog_selection(results: dict, output_dir: Path) -> None:
    overlay = results["analog_selection_overlay"]
    weeks_per_month = 52.0 / 12.0
    full_weeks = np.arange(1, 81, dtype=float)
    full_months = full_weeks / weeks_per_month
    analog_colors = [
        ("Comparable A (fast KOL-driven uptake)", TOKENS["blue"]),
        ("Comparable B (slower primary-care-driven uptake)", TOKENS["gold"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    ax_left, ax_right = axes
    panels = [(ax_left, 26, "First 26 Weeks"), (ax_right, 52, "First 52 Weeks")]
    for ax, cutoff_week, title in panels:
        window = overlay.loc[overlay["week"] <= cutoff_week]
        ax.plot(
            window["week"], window["Roventra actual"], color=TOKENS["ink"], linewidth=2.0,
            marker="o", markersize=4, label="Roventra actual",
        )
        for name, color in analog_colors:
            params = fc.ANALOG_LAUNCHES[name]
            projected = fc.bass_cumulative_fraction(full_months, params["p"], params["q"])
            ax.plot(full_weeks, projected, color=color, linewidth=2.2, label=name)
        _style(ax)
        ax.set_title(title, fontsize=13, weight="600", pad=12)
        ax.set_xlabel("Week since launch")
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Cumulative starts as a share of ceiling")

    # Label the curves directly in the left panel, hugging the right edge where
    # each curve has settled (steep-rising A up top, near-flat B down low), so
    # a fixed vertical offset clears the line across the whole label width.
    ax_left.text(
        32, 0.66, "Comparable A\n(fast KOL-driven uptake)",
        color=TOKENS["blue"], fontsize=9.5, ha="right", va="bottom", linespacing=1.4,
    )
    ax_left.text(
        68, 0.20, "Comparable B\n(slower primary-care-\ndriven uptake)",
        color=TOKENS["gold"], fontsize=9, ha="right", va="bottom", linespacing=1.3,
    )
    ax_left.text(
        20, 0.224, "Roventra actual",
        color=TOKENS["ink"], fontsize=9.5, ha="left", va="top",
    )
    fig.suptitle("Roventra Against Both Analogs, at Each Checkpoint", fontsize=14.5, weight="600", y=1.02)
    _save(fig, "figure_14_5_analog_selection_zoom", output_dir)


def figure_14_6_bass_fit(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    bass = results["bass_fit"].iloc[0]
    weeks = (observed["week_start"] - LAUNCH_DATE).dt.days / 7.0
    months = (weeks * 12.0 / 52.0).to_numpy()
    cumulative_actual = observed["nbrx"].cumsum().to_numpy()

    # Extend far enough to show the curve flatten at the ceiling, then a bit past that.
    near_ceiling_month = next(
        t for t in np.arange(1.0, 200.0, 0.5)
        if fc.bass_cumulative_fraction(np.array([t]), bass["p"], bass["q"])[0] >= 0.995
    )
    x_max = near_ceiling_month + 10.0
    projection_months = np.linspace(0, x_max, 300)
    projected = fc.bass_cumulative_fraction(projection_months, bass["p"], bass["q"]) * bass["m"]
    in_window = months <= x_max

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.plot(months[in_window], cumulative_actual[in_window], "o", color=TOKENS["blue"], markersize=4, label="Observed cumulative NBRx")
    ax.plot(projection_months, projected, color=TOKENS["gold"], linewidth=2.0, label="Fitted Bass curve")
    ax.axhline(bass["m"], color=TOKENS["gray"], linestyle="--", linewidth=1.4)
    ax.text(
        x_max, bass["m"], f"  Fitted ceiling m = {int(round(bass['m']))}",
        color=TOKENS["muted"], fontsize=13, ha="right", va="bottom",
    )
    formula_text = (
        r"$m\cdot F(t)=m\cdot\dfrac{1-e^{-(p+q)t}}{1+\frac{q}{p}e^{-(p+q)t}}$"
        "\n\n"
        + rf"$p={bass['p']:.3f},\ q={bass['q']:.3f},\ m={int(round(bass['m']))}$"
    )
    ax.text(
        0.75,
        0.12,
        formula_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=17,
        color=TOKENS["ink"],
        bbox={"facecolor": "white", "edgecolor": TOKENS["grid"], "boxstyle": "round,pad=0.5"},
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=13, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    ax.set_title("Fitted Bass Adoption Curve", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Cumulative new-to-therapy starts")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, bass["m"] * 1.08)
    _save(fig, "figure_14_6_bass_fit", output_dir)


def figure_14_7_opening_window(observed: pd.DataFrame, output_dir: Path) -> None:
    weeks = np.arange(1, len(observed) + 1, dtype=float)
    trx = observed["trx"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _holdout_base_chart(ax, weeks, trx)
    ymax = ax.get_ylim()[1]
    ax.text(TRAIN_WEEKS / 2, ymax * 0.93, "Train on this", ha="center", color=TOKENS["ink"], fontsize=10)
    ax.text(
        (TRAIN_WEEKS + weeks.max()) / 2, ymax * 0.93, "Score against this",
        ha="center", color=TOKENS["muted"], fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("52 Weeks of Observed Prescribing", fontsize=13.5, weight="600", pad=12)
    _save(fig, "figure_14_7_opening_window", output_dir)


def figure_14_8_baseline_naive(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    naive = results["holdout_forecasts"]["naive"].to_numpy()
    _method_forecast_figure(
        observed,
        [(f"Naive forecast (repeat week {TRAIN_WEEKS})", naive, TOKENS["orange"], {})],
        "Baseline: Naive Forecast Against the Held-Out Weeks",
        "figure_14_8_baseline_naive",
        output_dir,
    )


def figure_14_9_backtest_schematic(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    n_weeks = 52
    horizon = 8
    n_folds = 4
    train_fill, train_edge = "#eceff3", "#98a2b3"
    test_fill, test_edge = "#dde1e7", "#667085"
    last_fold = n_folds - 1
    last_test_end = n_weeks - last_fold * horizon
    last_test_start = last_test_end - horizon
    for fold in range(n_folds):
        test_end = n_weeks - fold * horizon
        test_start = test_end - horizon
        y = n_folds - fold
        ax.barh(y, test_start, left=0, color=train_fill, edgecolor=train_edge, height=0.6)
        ax.barh(y, horizon, left=test_start, color=test_fill, edgecolor=test_edge, height=0.6)
        ax.axvline(
            test_start,
            ymin=(y - 0.28) / (n_folds + 1),
            ymax=(y + 0.28) / (n_folds + 1),
            color=TOKENS["ink"],
            linewidth=1.1,
        )
        ax.text(-2, y, f"Fold {fold}", ha="right", va="center", fontsize=9.5)
    ax.set_xlim(-8, n_weeks + 1)
    ax.set_ylim(0.1, n_folds + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Week in observed history")
    ax.set_xticks([0, 12, 24, 36, 44, 52])
    ax.set_title("Rolling-Origin Backtest", fontsize=13.5, weight="600", pad=12)
    ax.text(last_test_start / 2, 0.4, "Train on past data", ha="center", color=TOKENS["muted"], fontsize=9.5)
    ax.text((last_test_start + last_test_end) / 2, 0.4, "Score next 8 weeks", ha="center", color=test_edge, fontsize=9.5)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(TOKENS["grid"])
    ax.tick_params(axis="x", colors=TOKENS["muted"])
    _save(fig, "figure_14_9_backtest_schematic", output_dir)


def figure_14_10_ets_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    ets = results["holdout_forecasts"]["ets"].to_numpy()
    _method_forecast_figure(
        observed,
        [("ETS forecast", ets, TOKENS["green"], {})],
        "Exponential Smoothing (ETS) Against the Held-Out Weeks",
        "figure_14_10_ets_forecast",
        output_dir,
    )


def figure_14_11_sarima_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    sarima = results["holdout_forecasts"]["sarima"].to_numpy()
    _method_forecast_figure(
        observed,
        [("SARIMA forecast", sarima, TOKENS["red"], {})],
        "SARIMA Against the Held-Out Weeks",
        "figure_14_11_sarima_forecast",
        output_dir,
    )


def figure_14_12_prophet_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    prophet_without = results["prophet_no_covariates_holdout"]
    prophet_with = results["holdout_forecasts"]["prophet"].to_numpy()
    _method_forecast_figure(
        observed,
        [
            ("Prophet, no covariates", prophet_without, TOKENS["orange"], {"linestyle": "--", "marker": "D"}),
            ("Prophet, with access + promo", prophet_with, TOKENS["blue"], {}),
        ],
        "Prophet, With and Without Planned Covariates",
        "figure_14_12_prophet_forecast",
        output_dir,
    )


def figure_14_13_gbt_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    gbt = results["holdout_forecasts"]["gbt"].to_numpy()
    _method_forecast_figure(
        observed,
        [("GBT forecast", gbt, TOKENS["gold"], {})],
        "Gradient-Boosted Trees Against the Held-Out Weeks",
        "figure_14_13_gbt_forecast",
        output_dir,
    )


def figure_14_14_tft_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    tft = results["holdout_forecasts"]["tft"].to_numpy()
    _method_forecast_figure(
        observed,
        [("TFT forecast", tft, TOKENS["purple"], {})],
        "Temporal Fusion Transformer Against the Held-Out Weeks",
        "figure_14_14_tft_forecast",
        output_dir,
    )


def figure_14_15_foundation_forecast(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    chronos = results["holdout_forecasts"]["chronos"].to_numpy()
    timesfm = results["holdout_forecasts"]["timesfm"].to_numpy()
    _method_forecast_figure(
        observed,
        [
            ("Chronos (median)", chronos, TOKENS["blue"], {}),
            ("TimesFM", timesfm, TOKENS["gold"], {"marker": "^"}),
        ],
        "Zero-Shot Foundation Models Against the Held-Out Weeks",
        "figure_14_15_foundation_forecast",
        output_dir,
    )


def figure_14_16_holdout_forecasts(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    holdout_forecasts = results["holdout_forecasts"]
    weeks = np.arange(1, len(observed) + 1, dtype=float)
    trx = observed["trx"].to_numpy()
    context_weeks = 12
    window_start = TRAIN_WEEKS - context_weeks
    is_context = (weeks >= window_start) & (weeks <= TRAIN_WEEKS)
    is_holdout = weeks > TRAIN_WEEKS
    holdout_weeks = weeks[is_holdout]

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axvspan(TRAIN_WEEKS, weeks.max(), color=TOKENS["soft_gray"], zorder=0)
    ax.axvline(TRAIN_WEEKS, color=TOKENS["muted"], linestyle=":", linewidth=1.2)
    # The actual series stays bold and solid throughout; every method's forecast
    # is muted and dashed so the one line that matters never gets lost in the pile.
    ax.plot(weeks[is_context], trx[is_context], color=TOKENS["ink"], linewidth=3.2, label="Observed (training)")
    ax.plot(
        holdout_weeks, trx[is_holdout], color=TOKENS["ink"], linewidth=3.2,
        marker="o", markersize=6, label="Actual (held out)", zorder=10,
    )
    # Same color per method as its own dedicated figure earlier in the section.
    style_map = {
        "naive": (TOKENS["orange"], "s"),
        "ets": (TOKENS["green"], "s"),
        "sarima": (TOKENS["red"], "s"),
        "prophet": (TOKENS["blue"], "s"),
        "gbt": (TOKENS["gold"], "s"),
        "tft": (TOKENS["purple"], "s"),
        "chronos": (TOKENS["teal"], "s"),
        "timesfm": (TOKENS["magenta"], "^"),
    }
    labels = {
        "naive": "Naive / seasonal-naive",
        "ets": "ETS",
        "sarima": "SARIMA",
        "prophet": "Prophet",
        "gbt": "GBT",
        "tft": "TFT",
        "chronos": "Chronos",
        "timesfm": "TimesFM",
    }
    for method, (color, marker) in style_map.items():
        ax.plot(
            holdout_weeks, holdout_forecasts[method], color=color, linestyle="--", marker=marker,
            markersize=4, linewidth=1.4, alpha=0.6, label=labels[method],
        )
    _style(ax)
    ax.set_xlim(window_start, weeks.max())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper left")
    ax.set_title("Every Method Against the Held-Out Weeks", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Week since launch")
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_16_foundation_vs_classical", output_dir)


def figure_14_17_fan_chart(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    interval = results["conformal_interval"]
    actual_holdout = results["holdout_actual"]["actual"].to_numpy()
    coverage = results["conformal_coverage"]["empirical"].iloc[0]

    weeks = np.arange(1, len(observed) + 1, dtype=float)
    trx = observed["trx"].to_numpy()
    context_weeks = 12
    window_start = TRAIN_WEEKS - context_weeks
    is_context = (weeks >= window_start) & (weeks <= TRAIN_WEEKS)
    holdout_weeks = weeks[weeks > TRAIN_WEEKS]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axvspan(TRAIN_WEEKS, weeks.max(), color=TOKENS["soft_gray"], zorder=0)
    ax.axvline(TRAIN_WEEKS, color=TOKENS["muted"], linestyle=":", linewidth=1.2)
    ax.plot(weeks[is_context], trx[is_context], color=TOKENS["ink"], linewidth=2.4, label="Observed (training)")
    ax.fill_between(holdout_weeks, interval["lower"], interval["upper"], color=TOKENS["soft_blue"], label="80% interval")
    ax.plot(holdout_weeks, interval["point_forecast"], color=TOKENS["blue"], linewidth=2.0, label="ETS point forecast")
    ax.plot(
        holdout_weeks, actual_holdout, color=TOKENS["ink"], linewidth=2.4,
        marker="o", markersize=6, label="Actual (held out)", zorder=10,
    )
    _style(ax)
    ax.set_xlim(window_start, weeks.max())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(f"Calibrated Interval (Empirical Coverage {coverage:.0%})", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Week since launch")
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_17_calibrated_fan_chart", output_dir)


def figure_14_18_production_refit(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    national = results["national_series"]
    production = results["production_forecast"]["forecast"].to_numpy()
    production_interval = results["production_interval"]
    horizon = len(production)
    n_observed = len(observed)
    total_weeks = n_observed + horizon
    weeks = np.arange(1, total_weeks + 1, dtype=float)
    trx_truth = national["trx"].to_numpy()[:total_weeks]
    window_start = 32.0
    history_mask = weeks <= n_observed
    future_mask = weeks > n_observed
    future_weeks = weeks[future_mask]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axvspan(n_observed, total_weeks, color=TOKENS["soft_gray"], alpha=0.5, zorder=0)
    ax.axvline(n_observed, color=TOKENS["muted"], linestyle=":", linewidth=1.2)
    ax.plot(weeks[history_mask], trx_truth[history_mask], color=TOKENS["ink"], linewidth=2.4, label="Observed, all 52 weeks (training)")
    ax.fill_between(
        future_weeks, production_interval["lower"], production_interval["upper"],
        color=TOKENS["soft_blue"], label="80% interval", zorder=1,
    )
    ax.plot(
        future_weeks, production, color=TOKENS["blue"], linewidth=2.2,
        marker="s", markersize=5, label="ETS production forecast",
    )
    ax.plot(
        future_weeks, trx_truth[future_mask], color=TOKENS["ink"], linewidth=2.2, linestyle="--",
        marker="o", markersize=6, label="Actual TRx, weeks 53-60", zorder=10,
    )
    _style(ax)
    ax.set_xlim(window_start - 1, total_weeks + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9.2, loc="upper left")
    ax.set_title("The Production Forecast Against the Weeks That Followed", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Week since launch")
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_18_production_refit", output_dir)


def figure_14_19_hierarchy(results: dict, output_dir: Path) -> None:
    base = results["hierarchy_base_forecast"].iloc[0]
    bu = results["reconciled_bottom_up"]["h1"]
    td = results["reconciled_top_down"]["h1"]
    ols = results["reconciled_ols"]["h1"]
    territory_order = [c for c in base.index if c != "National"]

    methods = [("Unreconciled", base), ("Bottom-up", bu), ("Top-down", td), ("OLS", ols)]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(methods))
    bar_width = 0.34
    y_lo, y_hi = 420, 450
    y_span = y_hi - y_lo
    national_color = TOKENS["blue"]
    territory_color = TOKENS["soft_gold"]
    gap_color = TOKENS["red"]
    coherent_color = TOKENS["green"]

    for xpos, (label, series) in zip(x, methods, strict=True):
        national_value = float(series["National"])
        territory_sum = float(series[territory_order].sum())
        ax.bar(
            xpos - bar_width / 2 - 0.02, national_value, width=bar_width, color=national_color,
            zorder=2, label="National forecast" if xpos == 0 else None,
        )
        ax.bar(
            xpos + bar_width / 2 + 0.02, territory_sum, width=bar_width, color=territory_color,
            edgecolor=TOKENS["gold"], hatch="///", linewidth=1.1, zorder=2,
            label="Sum of 12 territories" if xpos == 0 else None,
        )
        for xoff, value in [(-bar_width / 2 - 0.02, national_value), (bar_width / 2 + 0.02, territory_sum)]:
            ax.text(xpos + xoff, value + y_span * 0.012, f"{value:,.1f}", ha="center", va="bottom", fontsize=9, color=TOKENS["ink"], weight="600")
        ax.text(
            xpos, max(national_value, territory_sum) + y_span * 0.07, label,
            ha="center", va="bottom", color=TOKENS["ink"], fontsize=10.5, weight="700",
        )
        gap = national_value - territory_sum
        if abs(gap) > 1.0:
            gap_text, gap_text_color = f"{gap:+.1f} ({gap / territory_sum:+.1%})", gap_color
        else:
            gap_text, gap_text_color = "coherent", coherent_color
        ax.text(
            xpos, -0.04, gap_text, transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=gap_text_color, fontsize=9.5, weight="600",
        )

    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style(ax)
    ax.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("Only Reconciliation Makes National Match the Territory Sum", fontsize=13.5, weight="600", pad=14)
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_19_forecast_hierarchy", output_dir)


def figure_14_20_erosion_schematic(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    weeks = np.linspace(0, 100, 300)
    # Same toy brand as the 14.5.3 half-life walkthrough (1,000 TRx, 10%
    # residual, 10-week half-life), run through the actual _erosion_curve()
    # formula: 1 smooth exponential, no piecewise kink.
    pre_entry, residual_fraction, half_life_weeks = 1000.0, 0.10, 10.0
    curve = fc._erosion_curve(weeks, pre_entry, residual_fraction, half_life_weeks)
    ax.plot(weeks, curve, color=TOKENS["blue"], linewidth=2.3)
    ax.text(15, 800, "Automatic\nsubstitution", ha="center", va="center", color=TOKENS["ink"], fontsize=10, weight="600")
    ax.text(56, curve.max() * 0.25, "Residual / loyalty share", ha="center", va="center", color=TOKENS["ink"], fontsize=10, weight="600")
    _style(ax)
    ax.set_title("One Decay Curve Reads as Fast, Then Slow", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Weeks since generic entry")
    ax.set_ylabel("Illustrative TRx")
    _save(fig, "figure_14_20_erosion_schematic", output_dir)


def figure_14_21_analog_erosion(results: dict, output_dir: Path) -> None:
    analog = results["analog_erosion_comparison"]
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.plot(
        analog["weeks_since_entry"],
        analog["Comparable erosion A (fast generic substitution)"],
        color=TOKENS["blue"],
        linewidth=2.2,
    )
    ax.plot(
        analog["weeks_since_entry"],
        analog["Comparable erosion B (slower substitution, branded loyalty)"],
        color=TOKENS["gold"],
        linewidth=2.2,
    )
    ax.text(28, 55, "Comparable A", ha="center", va="center", color=TOKENS["blue"], fontsize=9.5, weight="600")
    ax.text(45, 100, "Comparable B", ha="center", va="center", color=TOKENS["gold"], fontsize=9.5, weight="600")
    _style(ax)
    ax.set_title("Comparable Erosion Shapes Diverge Fast", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Weeks since generic entry")
    ax.set_ylabel("Projected TRx from Roventra's pre-entry level")
    _save(fig, "figure_14_21_analog_erosion_curves", output_dir)


def figure_14_22_erosion(results: dict, output_dir: Path) -> None:
    tail = results["erosion_tail"]
    analog = results["analog_erosion"]
    chronos_check = results["chronos_erosion_cross_check"]
    weeks = tail["weeks_since_entry"].to_numpy()
    actual = tail["trx"].to_numpy()
    mature_fit = fc.fit_erosion(weeks[:78], actual[:78])
    fitted = fc._erosion_curve(
        weeks,
        mature_fit["pre_entry_reference"],
        mature_fit["residual_fraction"],
        mature_fit["half_life_weeks"],
    )
    cross_check_weeks = np.arange(78, 78 + len(chronos_check))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax_left.plot(weeks, actual, color=TOKENS["ink"], linewidth=2.0, label="Actual post-entry TRx")
    ax_left.plot(analog["weeks_since_entry"], analog["projected"], color=TOKENS["gray"], linestyle="--", linewidth=1.8, label="Analog erosion projection")
    ax_left.plot(weeks, fitted, color=TOKENS["orange"], linewidth=1.8, label="Parametric fit")
    ax_left.fill_between(cross_check_weeks, chronos_check["low"], chronos_check["high"], color=TOKENS["soft_green"], alpha=0.9, label="Chronos cross-check interval")
    ax_left.plot(cross_check_weeks, chronos_check["median"], color=TOKENS["green"], linewidth=1.8)
    _style(ax_left)
    ax_left.set_title("Full post-entry curve", fontsize=12.2, weight="600", pad=10)
    ax_left.set_xlabel("Weeks since generic entry")
    ax_left.set_ylabel("TRx")
    ax_left.legend(frameon=False, fontsize=8.9, loc="upper right")

    mask = weeks >= 70
    zoom_weeks = weeks[mask]
    zoom_actual = actual[mask]
    zoom_fitted = fitted[mask]
    ax_right.plot(zoom_weeks, zoom_actual, color=TOKENS["ink"], linewidth=2.0)
    ax_right.plot(zoom_weeks, zoom_fitted, color=TOKENS["orange"], linewidth=1.8)
    ax_right.fill_between(cross_check_weeks, chronos_check["low"], chronos_check["high"], color=TOKENS["soft_green"], alpha=0.9)
    ax_right.plot(cross_check_weeks, chronos_check["median"], color=TOKENS["green"], linewidth=1.8)
    _style(ax_right)
    zoom_x0, zoom_x1 = 70, 105
    zoom_y0 = min(float(chronos_check["low"].min()), float(zoom_actual.min())) * 0.9
    zoom_y1 = max(float(chronos_check["high"].max()), float(zoom_actual.max())) * 1.12
    ax_right.set_xlim(zoom_x0, zoom_x1)
    ax_right.set_ylim(zoom_y0, zoom_y1)
    ax_right.set_title("Tail zoom", fontsize=12.2, weight="600", pad=10)
    ax_right.set_xlabel("Weeks since generic entry")
    ax_right.set_ylabel("TRx")

    from matplotlib.patches import ConnectionPatch, Rectangle

    ax_left.add_patch(
        Rectangle(
            (zoom_x0, zoom_y0), zoom_x1 - zoom_x0, zoom_y1 - zoom_y0,
            fill=False, edgecolor=TOKENS["ink"], linewidth=1.3, linestyle="--", zorder=5,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            xyA=((zoom_x0 + zoom_x1) / 2, 45), coordsA=ax_left.transData,
            xyB=(67, 26), coordsB=ax_right.transData,
            arrowstyle="-|>", mutation_scale=18, color=TOKENS["ink"], linewidth=1.6, zorder=6,
        )
    )

    fig.suptitle("Post-Entry Erosion with a Tail Cross-Check", fontsize=14, weight="600", y=0.98)
    fig.tight_layout()
    _save(fig, "figure_14_22_erosion_curve", output_dir)


def figure_14_23_consensus_vs_actual(observed: pd.DataFrame, results: dict, output_dir: Path) -> None:
    national = results["national_series"]
    production = results["production_forecast"]["forecast"].to_numpy()
    production_interval = results["production_interval"]
    consensus = results["ensemble_consensus"]["consensus"].to_numpy()
    horizon = len(production)
    n_observed = len(observed)
    total_weeks = n_observed + horizon
    weeks = np.arange(1, total_weeks + 1, dtype=float)
    trx_truth = national["trx"].to_numpy()[:total_weeks]
    window_start = 32.0
    history_mask = weeks <= n_observed
    future_mask = weeks > n_observed
    future_weeks = weeks[future_mask]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axvspan(n_observed, total_weeks, color=TOKENS["soft_gray"], alpha=0.5, zorder=0)
    ax.axvline(n_observed, color=TOKENS["muted"], linestyle=":", linewidth=1.2)
    ax.plot(weeks[history_mask], trx_truth[history_mask], color=TOKENS["ink"], linewidth=2.4, label="Observed, all 52 weeks (training)")
    ax.fill_between(
        future_weeks, production_interval["lower"], production_interval["upper"],
        color=TOKENS["soft_blue"], label="ETS 80% interval (context)", zorder=1,
    )
    ax.plot(future_weeks, production, color=TOKENS["blue"], linewidth=1.8, linestyle="--", label="ETS production forecast")
    ax.plot(
        future_weeks, consensus, color=TOKENS["orange"], linewidth=2.4,
        marker="s", markersize=5, label="Accuracy-weighted consensus", zorder=5,
    )
    ax.plot(
        future_weeks, trx_truth[future_mask], color=TOKENS["ink"], linewidth=2.2, linestyle="--",
        marker="o", markersize=6, label="Actual TRx, weeks 53-60", zorder=10,
    )
    _style(ax)
    ax.set_xlim(window_start - 1, total_weeks + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9.0, loc="upper left")
    ax.set_title("The Consensus Sits Inside ETS's Own Interval", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Week since launch")
    ax.set_ylabel("Weekly TRx")
    _save(fig, "figure_14_23_consensus_vs_actual", output_dir)


def figure_14_24_waterfall(results: dict, output_dir: Path) -> None:
    waterfall = results["consensus_waterfall"]
    totals = waterfall["running_total"].to_numpy(dtype=float)
    pct = waterfall["adjustment_pct"].to_numpy(dtype=float)
    steps = waterfall["step"].to_list()
    adjustments = [0.0]
    adjustments.extend(totals[i] - totals[i - 1] for i in range(1, len(totals)))
    final_total = totals[-1]
    net_pct = final_total / totals[0] - 1.0

    display_steps = [steps[0], *steps[1:], "Committed number"]
    y = np.arange(len(display_steps))
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    neutral = TOKENS["blue"]
    xmin = min(totals.min(), final_total) - 220
    xmax = max(totals.max(), final_total) + 90
    bar_height = 0.5

    ax.axvline(totals[0], color=TOKENS["muted"], linestyle="--", linewidth=1.1, alpha=0.55, zorder=0)
    ax.barh(y[0], totals[0], height=bar_height, color=neutral, zorder=2)
    ax.text(totals[0] + 12, y[0], f"{totals[0]:,.1f}", ha="left", va="center", fontsize=9.5, color=TOKENS["ink"], weight="600")

    running = totals[0]
    for idx, delta in enumerate(adjustments[1:], start=1):
        color = TOKENS["green"] if delta > 0 else TOKENS["red"]
        start_x, end_x = running, running + delta
        ax.plot([start_x, start_x], [y[idx - 1] + bar_height / 2, y[idx]], color=TOKENS["muted"], linestyle=":", linewidth=1.0, zorder=1)
        ax.annotate(
            "", xy=(end_x, y[idx]), xytext=(start_x, y[idx]),
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2.4, "mutation_scale": 20},
            zorder=3,
        )
        ax.text(
            (start_x + end_x) / 2, y[idx] - 0.30, f"{delta:+.1f} ({pct[idx]:+.1%})",
            ha="center", va="bottom", fontsize=8.9, color=color, weight="600",
        )
        running = end_x

    ax.plot([running, running], [y[-2] + bar_height / 2, y[-1]], color=TOKENS["muted"], linestyle=":", linewidth=1.0, zorder=1)
    ax.barh(y[-1], final_total, height=bar_height, color=neutral, zorder=2)
    ax.text(final_total + 12, y[-1] - 0.10, f"{final_total:,.1f}", ha="left", va="bottom", fontsize=9.5, color=TOKENS["ink"], weight="600")
    net_color = TOKENS["green"] if net_pct > 0 else TOKENS["red"]
    ax.text(final_total + 12, y[-1] + 0.12, f"Net {net_pct:+.1%}", ha="left", va="top", fontsize=9.5, color=net_color, weight="700")

    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(display_steps, fontsize=9.5)
    ax.set_ylim(y[-1] + 0.7, y[0] - 0.7)
    _style(ax)
    ax.grid(False)
    ax.set_title("From Analytics Consensus to Committed Number", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Total volume, weeks 53-60")
    fig.tight_layout()
    _save(fig, "figure_14_24_consensus_waterfall", output_dir)


def figure_14_25_scenarios(results: dict, output_dir: Path) -> None:
    scenarios = results["scenario_forecast"]
    months = np.arange(0, 37)
    p = 0.008
    q = 0.244
    scenario_table = scenarios.set_index("scenario")
    low = fc.bass_cumulative_fraction(months, p, q) * scenario_table.loc["Low", "ceiling"]
    base = fc.bass_cumulative_fraction(months, p, q) * scenario_table.loc["Base", "ceiling"]
    high = fc.bass_cumulative_fraction(months, p, q) * scenario_table.loc["High", "ceiling"]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.fill_between(months, low, high, color=TOKENS["soft_green"], alpha=0.18, label="Low to High scenarios")
    ax.plot(months, high, color=TOKENS["green"], linewidth=1.8, label="High")
    ax.plot(months, base, color=TOKENS["blue"], linewidth=2.2, label="Base")
    ax.plot(months, low, color=TOKENS["red"], linewidth=1.8, label="Low")
    ax.annotate(
        "High requires 40% peak share",
        xy=(30, high[30]),
        xytext=(21, high[30] * 0.72),
        arrowprops={"arrowstyle": "->", "color": TOKENS["green"], "linewidth": 1.2},
        color=TOKENS["green"],
        fontsize=9.5,
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Scenario Fan from Driver Assumptions", fontsize=13.5, weight="600", pad=14)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Cumulative treated patients")
    _save(fig, "figure_14_25_scenario_fan", output_dir)


def build_all_figures(output_dir: Path | None = None) -> None:
    output_dir = output_dir or (ROOT / "ch14_forecasting" / "assets" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis()
    national = results["national_series"]
    observed = results["observed_series"]

    figure_14_1_lifecycle(national, output_dir)
    figure_14_2_funnel_timeline(observed, results, output_dir)
    figure_14_3_bass_toy_shapes(output_dir)
    figure_14_4_analog_shapes(results, output_dir)
    figure_14_5_analog_selection(results, output_dir)
    figure_14_6_bass_fit(observed, results, output_dir)
    figure_14_7_opening_window(observed, output_dir)
    figure_14_8_baseline_naive(observed, results, output_dir)
    figure_14_9_backtest_schematic(output_dir)
    figure_14_10_ets_forecast(observed, results, output_dir)
    figure_14_11_sarima_forecast(observed, results, output_dir)
    figure_14_12_prophet_forecast(observed, results, output_dir)
    figure_14_13_gbt_forecast(observed, results, output_dir)
    figure_14_14_tft_forecast(observed, results, output_dir)
    figure_14_15_foundation_forecast(observed, results, output_dir)
    figure_14_16_holdout_forecasts(observed, results, output_dir)
    figure_14_17_fan_chart(observed, results, output_dir)
    figure_14_18_production_refit(observed, results, output_dir)
    figure_14_19_hierarchy(results, output_dir)
    figure_14_20_erosion_schematic(output_dir)
    figure_14_21_analog_erosion(results, output_dir)
    figure_14_22_erosion(results, output_dir)
    figure_14_23_consensus_vs_actual(observed, results, output_dir)
    figure_14_24_waterfall(results, output_dir)
    figure_14_25_scenarios(results, output_dir)
    print(f"Wrote 25 code-built figures to {output_dir}")


if __name__ == "__main__":
    build_all_figures()
