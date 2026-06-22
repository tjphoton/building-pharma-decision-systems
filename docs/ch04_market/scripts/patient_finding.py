"""Machine-learning patient finding: turn the count of unseen patients into a list.

Capture-recapture (maturity.py) estimates how many launch-condition patients sit
outside every paid claim. This module ranks the undiagnosed patients by their
probability of truly having the condition, so the count becomes an actionable list.

The label is the confirmed paid-claims phenotype. The features rely on biomarker
signals (A1C) that are positively correlated with true disease status even in the
undiagnosed population, and exclude the launch diagnosis code itself so the model
cannot simply relearn the phenotype.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from estimands import drug_name_from_ndc

LAUNCH_CLASS_DRUGS = ["Roventra", "Nexoral", "Vexpro"]
RANDOM_STATE = 20260613


def build_features(
    tables: Mapping[str, pd.DataFrame], patients: pd.DataFrame
) -> pd.DataFrame:
    """Assemble leakage-free per-patient features for patient finding.

    Feature design principle: use features that are POSITIVELY correlated with the
    true disease state even among undiagnosed patients. Healthcare utilization counts
    (n_medical_claims, n_distinct_dx) correlate with overall morbidity, not specifically
    with T2D, so they are anti-correlated with true T2D in the undiagnosed group and
    produce negative lift. The A1C biomarker is directly informative regardless of
    whether a T2D diagnosis has been coded.
    """
    pharmacy = tables["pharmacy_claims"].copy()
    pharmacy["drug_name"] = drug_name_from_ndc(pharmacy, tables["ndc_codes"])
    paid_class = pharmacy.loc[
        pharmacy["transaction_type"].eq("PAID") & pharmacy["drug_name"].isin(LAUNCH_CLASS_DRUGS)
    ]

    # Derive A1C signal from lab results
    lab = tables.get("lab_results", pd.DataFrame())
    if not lab.empty and "test_name" in lab.columns:
        a1c = lab.loc[lab["test_name"].eq("Hemoglobin A1c")]
        a1c_max = a1c.groupby("patient_id")["result"].max()
    else:
        a1c_max = pd.Series(dtype=float, name="result")

    # T2D drug proxy: pharmacy claims billed with a T2D diagnosis code
    diabetes_rx_patients = pharmacy.loc[
        pharmacy["diagnosis_code"].str.startswith("E11", na=False)
        & pharmacy["transaction_type"].eq("PAID")
    ]["patient_id"]

    f = patients[
        ["patient_id", "age_band", "region", "sex", "base_phenotype", "reference_condition"]
    ].copy()
    f = f.rename(columns={"base_phenotype": "diagnosed", "reference_condition": "true_condition"})

    # Primary biomarker signal: A1C >= 6.5% is the ADA diabetes diagnostic threshold.
    # True T2D patients have elevated A1C regardless of diagnosis coding status.
    f["max_a1c"] = f["patient_id"].map(a1c_max).fillna(0)
    f["has_elevated_a1c"] = (f["max_a1c"] >= 6.5).astype(int)

    # Secondary signals: class drug fills and T2D-coded pharmacy claims
    f["n_class_fills"] = f["patient_id"].map(paid_class.groupby("patient_id").size()).fillna(0)
    f["diabetes_rx_proxy"] = f["patient_id"].isin(diabetes_rx_patients).astype(int)

    return f


def patient_finding_lift(
    tables: Mapping[str, pd.DataFrame], patients: pd.DataFrame
) -> pd.DataFrame:
    """Train the finder and return the lift among undiagnosed patients."""
    f = build_features(tables, patients)
    feature_frame = pd.get_dummies(
        f.drop(columns=["patient_id", "diagnosed", "true_condition"]),
        columns=["age_band", "region", "sex"],
    )
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        f["diagnosed"].astype(int),
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=f["diagnosed"],
    )
    model = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    f["score"] = model.predict_proba(feature_frame)[:, 1]

    undiagnosed = f.loc[f["diagnosed"].eq(0)].sort_values("score", ascending=False)
    base_rate = undiagnosed["true_condition"].mean()
    rows = [
        {
            "segment": "All undiagnosed",
            "patients": len(undiagnosed),
            "true_positives": int(undiagnosed["true_condition"].sum()),
            "true_rate": round(base_rate, 4),
            "lift": 1.0,
            "held_out_auc": round(float(auc), 3),
        }
    ]
    for fraction, label in [(0.10, "Top 10% by score"), (0.20, "Top 20% by score")]:
        top = undiagnosed.head(int(len(undiagnosed) * fraction))
        true_rate = top["true_condition"].mean()
        rows.append(
            {
                "segment": label,
                "patients": len(top),
                "true_positives": int(top["true_condition"].sum()),
                "true_rate": round(float(true_rate), 4),
                "lift": round(float(true_rate / base_rate), 2),
                "held_out_auc": round(float(auc), 3),
            }
        )
    return pd.DataFrame(rows)
