#!/usr/bin/env python3
"""Show how each planted Chapter 3 data defect changes a business conclusion.

Two before/after demonstrations against the generated package:

1. Claim maturity. Counting medical claims by service month from an early-
   January snapshot makes November and December look like a demand decline.
   The mature dataset shows the decline never happened.
2. Product mapping. A naive inner join on the dispensed NDC silently drops
   pack-size variant fills; a LEFT JOIN on the prescribed NDC surfaces them
   and keeps all rows for correct volume counts.

Writes detail CSVs to ``output_data/analysis_results/defect_impact/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visual_style import BLUE_DARK, INK, MUTED, RED_DARK, WHITE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation_modules.entities import validate_manifest_contract  # noqa: E402

DATA_DIR = ROOT / "output_data" / "generated_data"
ANALYSIS_DIR = ROOT / "output_data" / "analysis_results" / "defect_impact"
FIGURE_DIR = ROOT / "assets" / "figures"

DX_COLS = [f"diagnosis_{i}" for i in range(1, 11)]


def normalize_svg(path: Path) -> None:
    """Remove renderer-added trailing whitespace from an SVG."""
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def claim_maturity_comparison(
    early: pd.DataFrame, mature: pd.DataFrame, providers: pd.DataFrame
) -> pd.DataFrame:
    """Compare endocrinology T2D visits per 2024 service month at a snapshot vs. at maturity."""
    endo_npis = providers.loc[providers["specialty_1"].eq("Endocrinology"), "npi"]
    t2d_mask_early = (
        early[DX_COLS].apply(lambda col: col.astype(str).str.startswith("E11") & col.notna()).any(axis=1)
        & early["claim_date"].dt.year.eq(2024)
        & early["rendering_npi"].isin(endo_npis)
    )
    t2d_mask_mature = (
        mature[DX_COLS].apply(lambda col: col.astype(str).str.startswith("E11") & col.notna()).any(axis=1)
        & mature["claim_date"].dt.year.eq(2024)
        & mature["rendering_npi"].isin(endo_npis)
    )
    snapshot_by_month = (
        early.loc[t2d_mask_early, "claim_date"]
        .dt.to_period("M")
        .astype(str)
        .value_counts()
        .sort_index()
    )
    mature_by_month = (
        mature.loc[t2d_mask_mature, "claim_date"]
        .dt.to_period("M")
        .astype(str)
        .value_counts()
        .sort_index()
    )
    result = pd.DataFrame(
        {
            "service_month": mature_by_month.index,
            "claims_received_by_snapshot": snapshot_by_month.reindex(mature_by_month.index, fill_value=0).values,
            "claims_eventually_received": mature_by_month.values,
        }
    )
    result["completeness_pct"] = (
        100 * result["claims_received_by_snapshot"] / result["claims_eventually_received"]
    ).round(2)
    result["apparent_change_vs_truth"] = (result["completeness_pct"] - 100).round(2)
    return result


def product_mapping_comparison(
    pharmacy_claims: pd.DataFrame, ndc_codes: pd.DataFrame
) -> pd.DataFrame:
    """Compare paid fills: inner join on dispensed NDC (naive) vs join on prescribed NDC (correct).

    Joining on the dispensed NDC silently drops pack-size variant fills when the
    dispensed code differs from the prescribed code. Joining on the prescribed NDC
    (which never changes for the same product) surfaces all fills correctly.
    """
    paid = pharmacy_claims.loc[pharmacy_claims["transaction_type"].eq("PAID")].copy()
    ref = ndc_codes[["ndc", "drug_name"]]

    # Naive join on dispensed NDC: pack-size variants drop silently
    naive_counts = (
        paid.merge(ref, on="ndc", how="inner")
        .groupby("drug_name")
        .size()
        .rename("paid_fills_dispensed_join")
    )

    # Correct join on prescribed NDC: all fills retained, stable attribution
    correct = paid.merge(ref.rename(columns={"ndc": "ndc_prescribed"}), on="ndc_prescribed", how="left")
    correct["drug_name"] = correct["drug_name"].fillna("unmapped")
    correct_counts = correct.groupby("drug_name").size().rename("paid_fills_prescribed_join")

    result = (
        pd.concat([naive_counts, correct_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    result["fills_missed_by_dispensed_join"] = (
        result["paid_fills_prescribed_join"] - result["paid_fills_dispensed_join"]
    )
    return result.sort_values("paid_fills_prescribed_join", ascending=False).reset_index(drop=True)


def save_maturity_figure(maturity: pd.DataFrame, output_stem: Path) -> None:
    """Plot the false-decline comparison used in Chapter 3."""
    months = [m[5:] + "/24" for m in maturity["service_month"]]
    snapshot = maturity["claims_received_by_snapshot"]
    mature = maturity["claims_eventually_received"]
    december = maturity.iloc[-1]
    apparent_drop = 100 * (1 - december["claims_received_by_snapshot"] / december["claims_eventually_received"])

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
    ax.plot(
        months, mature, marker="o", linewidth=2.4, color=BLUE_DARK,
        label="Claims eventually received (mature)",
    )
    ax.plot(
        months, snapshot, marker="o", linewidth=2.4, color=RED_DARK,
        linestyle="--", label="Claims received by Jan 05, 2025 (snapshot)",
    )
    ax.fill_between(months, snapshot, mature, color=RED_DARK, alpha=0.10)
    ax.annotate(
        f"Not yet received at the snapshot:\nDecember looks {apparent_drop:.0f}% lower than it ends up",
        xy=(11, (december["claims_received_by_snapshot"] + december["claims_eventually_received"]) / 2),
        xytext=(6.0, mature.min() * 0.78),
        fontsize=11.5,
        color=RED_DARK,
        arrowprops={"arrowstyle": "->", "color": RED_DARK, "linewidth": 1.4},
    )
    ax.set_ylabel("Endocrinology visits (T2D diagnosis) by service month", fontsize=13, labelpad=12)
    ax.set_ylim(0, mature.max() * 1.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA6B2")
    ax.grid(axis="y", color="#DCE3EA", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.legend(loc="lower left", frameon=False, fontsize=11.5)
    fig.suptitle(
        "The December Decline That Never Happened",
        x=0.5,
        y=0.96,
        ha="center",
        fontsize=22,
        weight="bold",
    )
    fig.tight_layout(rect=[0.055, 0.03, 0.98, 0.91])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        if suffix == "svg":
            normalize_svg(path)
    plt.close(fig)


def run_defect_impact(data_dir: Path) -> dict[str, pd.DataFrame]:
    providers = pd.read_csv(data_dir / "reference" / "providers.csv")
    early = pd.read_csv(
        data_dir / "claims_medical" / "medical_claims.csv",
        parse_dates=["claim_date"],
    )
    mature = pd.read_csv(
        data_dir / "claims_medical" / "medical_claims_mature.csv",
        parse_dates=["claim_date"],
    )
    pharmacy_claims = pd.read_csv(
        data_dir / "claims_pharmacy" / "pharmacy_claims.csv",
        dtype={"ndc": str, "ndc_prescribed": str},
    )
    ndc_codes = pd.read_csv(data_dir / "reference" / "ndc_codes.csv", dtype={"ndc": str})
    return {
        "defect_impact_maturity": claim_maturity_comparison(early, mature, providers),
        "defect_impact_product_mapping": product_mapping_comparison(pharmacy_claims, ndc_codes),
    }


if __name__ == "__main__":
    if not (DATA_DIR / "claims_medical" / "medical_claims.csv").exists():
        print("Generated claims not found. Run generate_all_synthetic_data.py first.")
        sys.exit(1)
    validate_manifest_contract(DATA_DIR)
    results = run_defect_impact(DATA_DIR)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(ANALYSIS_DIR / f"{name}.csv", index=False)
        print(f"--- {name} ---")
        print(frame.to_string(index=False))
        print()
    print(f"Wrote detail tables to {ANALYSIS_DIR}")
