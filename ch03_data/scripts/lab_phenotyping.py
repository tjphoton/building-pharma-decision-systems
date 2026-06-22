"""Build patient phenotype cohorts from lab results delivered by the longitudinal claims vendor."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_modules.entities import validate_manifest_contract  # noqa: E402

A1C_DIABETIC = 6.5
A1C_PREDIABETIC = 5.7
PSA_HIGH = 10.0
PSA_BORDERLINE = 4.0
PDL1_HIGH = 50.0
PDL1_LOW_POSITIVE = 1.0


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def _latest_lab(lab_df: pd.DataFrame, test_name: str, value_name: str) -> pd.DataFrame:
    labs = lab_df.loc[lab_df["test_name"].eq(test_name)].copy()
    labs["service_date"] = pd.to_datetime(labs["service_date"], errors="coerce")
    labs["result"] = pd.to_numeric(labs["result"], errors="coerce")
    labs = labs.dropna(subset=["patient_id", "service_date", "result"])
    if labs.empty:
        return pd.DataFrame(columns=["patient_id", value_name, "result_date"])
    return (
        labs.sort_values(["patient_id", "service_date", "result"])
        .groupby("patient_id", as_index=False)
        .tail(1)[["patient_id", "result", "service_date"]]
        .rename(columns={"result": value_name, "service_date": "result_date"})
    )


def classify_by_a1c(lab_df: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_lab(lab_df, "Hemoglobin A1c", "latest_a1c")
    latest["a1c_class"] = pd.cut(
        latest["latest_a1c"],
        bins=[float("-inf"), A1C_PREDIABETIC, A1C_DIABETIC, float("inf")],
        labels=["normal", "prediabetes", "diabetes"],
        right=False,
    ).astype("string")
    return latest[["patient_id", "latest_a1c", "a1c_class", "result_date"]]


def classify_by_psa(lab_df: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_lab(lab_df, "PSA", "latest_psa")
    latest["psa_class"] = pd.cut(
        latest["latest_psa"],
        bins=[float("-inf"), PSA_BORDERLINE, PSA_HIGH, float("inf")],
        labels=["lower", "intermediate", "higher"],
        right=False,
    ).astype("string")
    return latest[["patient_id", "latest_psa", "psa_class", "result_date"]]


def classify_by_pdl1(lab_df: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_lab(lab_df, "PD-L1 TPS", "latest_pdl1")
    latest["pdl1_class"] = pd.cut(
        latest["latest_pdl1"],
        bins=[float("-inf"), PDL1_LOW_POSITIVE, PDL1_HIGH, float("inf")],
        labels=["less_than_1_percent", "1_to_49_percent", "50_percent_or_higher"],
        right=False,
    ).astype("string")
    return latest[["patient_id", "latest_pdl1", "pdl1_class", "result_date"]]


def build_lab_phenotype_cohort(
    lab_df: pd.DataFrame,
    patients_df: pd.DataFrame,
) -> pd.DataFrame:
    cohort = patients_df[["patient_id"]].copy()
    for classified, value_col, class_col in [
        (classify_by_a1c(lab_df), "latest_a1c", "a1c_class"),
        (classify_by_psa(lab_df), "latest_psa", "psa_class"),
        (classify_by_pdl1(lab_df), "latest_pdl1", "pdl1_class"),
    ]:
        result_date_col = f"{class_col.replace('_class', '')}_result_date"
        cohort = cohort.merge(
            classified[["patient_id", value_col, class_col, "result_date"]].rename(
                columns={"result_date": result_date_col}
            ),
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        cohort[class_col] = cohort[class_col].fillna("no_result")
    return cohort


def lab_phenotype_summary(cohort_df: pd.DataFrame) -> pd.DataFrame:
    labels = {"a1c_class": "A1C", "psa_class": "PSA", "pdl1_class": "PD-L1"}
    records: list[dict] = []
    for column, biomarker in labels.items():
        counts = cohort_df[column].value_counts(dropna=False)
        for phenotype_class, count in counts.items():
            records.append(
                {
                    "biomarker": biomarker,
                    "phenotype_class": str(phenotype_class),
                    "patient_count": int(count),
                    "percent_of_all_patients": round(100 * count / max(len(cohort_df), 1), 2),
                }
            )
    return pd.DataFrame(records)


def save_a1c_figure(summary: pd.DataFrame, output_stem: Path) -> None:
    order = ["no_result", "normal", "prediabetes", "diabetes"]
    labels = ["No A1C result", "Normal", "Prediabetes", "Diabetes"]
    a1c = summary.loc[summary["biomarker"].eq("A1C")].set_index("phenotype_class")
    counts = [int(a1c.loc[item, "patient_count"]) if item in a1c.index else 0 for item in order]
    percents = [float(a1c.loc[item, "percent_of_all_patients"]) if item in a1c.index else 0 for item in order]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#172033",
            "axes.labelcolor": "#344054",
            "xtick.color": "#344054",
            "ytick.color": "#5F6B7A",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    colors = ["#B8C5D1", "#D8E8F3", "#77A9CC", "#245B8A"]
    edge_colors = ["#7C8B99", "#8BB5D0", "#477FA7", "#173F63"]
    bars = ax.bar(labels, counts, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.66)
    ax.set_ylabel("Patients", fontsize=13, labelpad=12)
    ax.set_ylim(0, max(counts) * 1.20)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA6B2")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="y", color="#DCE3EA", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12, length=0, pad=10)
    ax.tick_params(axis="y", labelsize=11, length=0)
    for bar, percent in zip(bars, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.025,
            f"{int(bar.get_height()):,}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
            color="#172033",
        )
    fig.suptitle(
        "A1C Patient Groups in the Synthetic Population",
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
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        if suffix == "svg":
            normalize_svg(path)
    plt.close(fig)


def run_phenotyping(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_manifest_contract(data_dir)
    lab = pd.read_csv(data_dir / "claims_lab" / "lab_results.csv")
    patients = pd.read_csv(data_dir / "reference" / "patients.csv")
    cohort = build_lab_phenotype_cohort(lab, patients)
    return cohort, lab_phenotype_summary(cohort)


if __name__ == "__main__":
    output_root = Path(__file__).resolve().parents[1] / "output_data"
    figure_dir = Path(__file__).resolve().parents[1] / "assets" / "figures"
    data_dir = output_root / "generated_data"
    output_dir = output_root / "analysis_results" / "lab_phenotyping"
    if not (data_dir / "claims_lab" / "lab_results.csv").exists():
        print("Lab results not found. Run generate_all_synthetic_data.py first.")
        sys.exit(1)

    cohort, summary = run_phenotyping(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_dir / "lab_phenotype_cohort.csv", index=False)
    summary.to_csv(output_dir / "lab_phenotype_summary.csv", index=False)
    save_a1c_figure(summary, output_dir / "a1c_patient_group_distribution")
    save_a1c_figure(summary, figure_dir / "figure-3-8-a1c-patient-group-distribution")
    print(summary.to_string(index=False))
    print(f"Wrote results to {output_dir}")
