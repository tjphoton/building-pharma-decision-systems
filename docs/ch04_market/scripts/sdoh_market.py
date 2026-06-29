"""Chapter 4 SDOH market-sizing extension.

The source rows are synthetic and area-level. They are used to teach how
structural context changes hidden-patient estimates and patient finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CH99_SCRIPTS = Path(__file__).resolve().parents[2] / "ch99_SDOH" / "scripts"
sys.path.insert(0, str(CH99_SCRIPTS))

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

from sdoh_pipeline import (  # noqa: E402
    BLUE,
    GOLD,
    GRAY,
    SEED,
    TEXT,
    account_market_flag,
    build_area_table,
    build_patient_table,
    market_sdoh_summary,
    model_enrichment,
)

# State-level SDOH mapping for the 12 ch04 states.
# Values are approximate synthetic representations of real ACS/SVI/HRSA data.
# Q1 = lowest barrier, Q5 = highest.
_STATE_SDOH = pd.DataFrame(
    [
        ("NJ", 1, 0.052, 0.120, 0.810, 0.05),
        ("NY", 1, 0.054, 0.115, 0.815, 0.12),
        ("WA", 2, 0.064, 0.138, 0.780, 0.22),
        ("PA", 2, 0.058, 0.145, 0.770, 0.25),
        ("CA", 3, 0.075, 0.155, 0.730, 0.18),
        ("IL", 3, 0.063, 0.148, 0.740, 0.23),
        ("OH", 3, 0.060, 0.152, 0.735, 0.30),
        ("MI", 3, 0.058, 0.160, 0.725, 0.28),
        ("AZ", 4, 0.105, 0.175, 0.680, 0.35),
        ("FL", 4, 0.120, 0.168, 0.670, 0.20),
        ("GA", 4, 0.115, 0.178, 0.660, 0.40),
        ("TX", 5, 0.165, 0.195, 0.610, 0.45),
    ],
    columns=[
        "state", "sdoh_barrier_quintile",
        "uninsured_share", "transportation_burden",
        "primary_care_access_index", "rural_share",
    ],
)


def build_ch04_sdoh_patient_scores(
    tables: dict, patients: pd.DataFrame
) -> pd.DataFrame:
    """Score undiagnosed ch04 patients with claims-only and claims+SDOH models.

    Joins state-level SDOH features to the ch04 patient population, trains both
    model variants, and returns individual undiagnosed patient scores ordered by
    SDOH-enriched score. Used in Listing 4.9 for HCP targeting.
    """
    from patient_finding import RANDOM_STATE, build_features  # local to avoid circular import

    f = build_features(tables, patients)
    state_col = patients[["patient_id", "state"]].copy()
    f = f.merge(state_col, on="patient_id", how="left")
    f = f.merge(_STATE_SDOH, on="state", how="left")

    sdoh_cols = [
        "uninsured_share", "transportation_burden",
        "primary_care_access_index", "rural_share",
    ]
    cat_cols = ["age_band", "region", "sex"]
    meta_cols = ["patient_id", "diagnosed", "true_condition", "state", "sdoh_barrier_quintile"]

    X_claims = pd.get_dummies(
        f.drop(columns=meta_cols + sdoh_cols), columns=cat_cols
    )
    X_sdoh = pd.get_dummies(
        f.drop(columns=meta_cols), columns=cat_cols
    )
    y = f["diagnosed"].astype(int)

    clf_claims = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_claims, y)
    clf_sdoh = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_sdoh, y)

    undx_mask = f["diagnosed"].eq(0)
    keep_cols = [
        "patient_id", "state", "sdoh_barrier_quintile", "true_condition",
        "age_band", "region", "sex", "n_class_fills", "max_a1c", "diabetes_rx_proxy",
    ]
    out = f.loc[undx_mask, keep_cols].copy()
    out["claims_score"] = clf_claims.predict_proba(X_claims.loc[undx_mask])[:, 1]
    out["sdoh_score"] = clf_sdoh.predict_proba(X_sdoh.loc[undx_mask])[:, 1]
    return out.sort_values("sdoh_score", ascending=False).reset_index(drop=True)


def build_sdoh_market_outputs() -> dict[str, pd.DataFrame]:
    """Build the Chapter 4 SDOH tables."""

    areas = build_area_table()
    patients = build_patient_table(areas)
    enrichment = model_enrichment(patients)
    return {
        "sdoh_area": areas,
        "sdoh_patient": patients,
        "sdoh_market_summary": market_sdoh_summary(patients),
        "sdoh_model_summary": enrichment["model_summary"],
        "sdoh_model_top_decile": enrichment["top_decile_by_quintile"],
        "sdoh_account_market_flag": account_market_flag(patients),
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


def figure_4_5_under_observation(outputs: Path, figures_dir: Path) -> None:
    """Plot observed diagnosis share and unobserved patients by SDOH quintile."""

    summary = pd.read_csv(outputs / "sdoh_market_summary.csv")
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    x = summary["sdoh_barrier_quintile"]
    bars = ax1.bar(x, summary["estimated_unobserved"], color=GOLD, edgecolor="#8a6a24")
    ax1.set_xlabel("SDOH barrier quintile")
    ax1.set_ylabel("Estimated unobserved patients", color="#6b4d0d")
    ax1.tick_params(axis="y", labelcolor="#6b4d0d")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, summary["observed_share_pct"], color=BLUE, marker="o", linewidth=2.0)
    ax2.set_ylabel("Observed diagnosis share (%)", color="#2f5f8f")
    ax2.tick_params(axis="y", labelcolor="#2f5f8f")
    ax2.set_ylim(40, 90)

    for bar, share in zip(bars, summary["observed_share_pct"], strict=True):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{int(bar.get_height())}", ha="center", fontsize=9, color=TEXT)
        ax2.text(bar.get_x() + bar.get_width() / 2, share + 2.5,
                 f"{share:.1f}%", ha="center", fontsize=9, color="#2f5f8f")

    ax1.set_title("Observed diagnoses by SDOH barrier quintile", fontsize=13)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, figures_dir, "figure_4_5_sdoh_under_observation")


def figure_4_6_model_lift(outputs: Path, figures_dir: Path) -> None:
    """Plot true undiagnosed patients selected by each model."""

    top_df = pd.read_csv(outputs / "sdoh_model_top_decile.csv")
    summary = pd.read_csv(outputs / "sdoh_model_summary.csv").set_index("model")
    base_auc = summary.loc["Claims only", "auc"]
    sdoh_auc = summary.loc["Claims + SDOH", "auc"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = top_df["sdoh_barrier_quintile"]
    width = 0.34
    ax.bar(x - width / 2, top_df["claims_only_true"], width=width,
           color=GRAY, edgecolor="#666666", label="Claims only")
    ax.bar(x + width / 2, top_df["claims_sdoh_true"], width=width,
           color=BLUE, edgecolor="#2f5f8f", label="Claims + SDOH")

    for xi, base_val, sdoh_val in zip(
        x, top_df["claims_only_true"], top_df["claims_sdoh_true"], strict=True
    ):
        ax.text(xi - width / 2, base_val + 0.8, f"{int(base_val)}", ha="center", fontsize=9)
        ax.text(xi + width / 2, sdoh_val + 0.8, f"{int(sdoh_val)}", ha="center", fontsize=9)

    ax.set_xlabel("SDOH barrier quintile")
    ax.set_ylabel("True undiagnosed patients in top decile")
    ax.set_title(f"Patient finding by SDOH quintile, AUC {base_auc:.3f} to {sdoh_auc:.3f}", fontsize=12)
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, figures_dir, "figure_4_6_sdoh_model_lift")
