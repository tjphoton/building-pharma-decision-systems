"""Specialty pharmacy hub pathway analytics for Chapter 5."""

from __future__ import annotations

import pandas as pd


def sp_hub_funnel(
    specialty_pharmacy: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Referral-to-shipment funnel for cohort patients, with stage timing."""
    windows = cohort[["patient_id", "index_date", "followup_end"]]
    sp = specialty_pharmacy.merge(windows, on="patient_id", how="inner", validate="many_to_one")
    sp = sp.loc[
        sp["referral_date"].between(sp["index_date"], sp["followup_end"], inclusive="both")
    ].copy()
    sp = sp.sort_values(["patient_id", "referral_date"]).drop_duplicates("patient_id", keep="first")
    sp["days_to_decision"] = (sp["status_date"] - sp["referral_date"]).dt.days
    sp["days_to_ship"] = (sp["ship_date"] - sp["referral_date"]).dt.days

    total = len(sp)
    approved = sp["hub_status"].eq("Approved") | sp["hub_status"].eq("PA Approved")
    shipped = sp["dispense_status"].eq("Shipped")
    rows = [
        {
            "stage": "Referral received",
            "patients": total,
            "share_of_referrals": 1.0,
            "median_days_from_referral": 0.0,
        },
        {
            "stage": "Authorization approved",
            "patients": int(approved.sum()),
            "share_of_referrals": round(approved.mean(), 3),
            "median_days_from_referral": float(sp.loc[approved, "days_to_decision"].median()),
        },
        {
            "stage": "Shipped",
            "patients": int(shipped.sum()),
            "share_of_referrals": round(shipped.mean(), 3),
            "median_days_from_referral": float(sp.loc[shipped, "days_to_ship"].median()),
        },
        {
            "stage": "Abandoned",
            "patients": int(sp["dispense_status"].eq("Abandoned").sum()),
            "share_of_referrals": round(sp["dispense_status"].eq("Abandoned").mean(), 3),
            "median_days_from_referral": float(
                sp.loc[sp["dispense_status"].eq("Abandoned"), "days_to_decision"].median()
            ),
        },
    ]
    return pd.DataFrame(rows)


def abandonment_outcomes(
    specialty_pharmacy: pd.DataFrame,
    cohort: pd.DataFrame,
    paid_basket_fills: pd.DataFrame,
    product: str = "Roventra",
) -> pd.DataFrame:
    """What happens after a hub abandonment, by recorded reason."""
    windows = cohort[["patient_id", "index_date", "followup_end"]]
    sp = specialty_pharmacy.merge(windows, on="patient_id", how="inner", validate="many_to_one")
    sp = sp.loc[
        sp["dispense_status"].eq("Abandoned")
        & sp["referral_date"].between(sp["index_date"], sp["followup_end"], inclusive="both")
    ].copy()
    sp = sp.sort_values(["patient_id", "referral_date"]).drop_duplicates("patient_id", keep="first")

    fills = paid_basket_fills[["patient_id", "date_of_service", "product_name"]]
    joined = sp.merge(fills, on="patient_id", how="left")
    after = joined.loc[
        joined["date_of_service"].gt(joined["referral_date"])
        & joined["date_of_service"].le(joined["followup_end"])
    ]

    recovered = set(after.loc[after["product_name"].eq(product), "patient_id"])
    competitor = set(after.loc[~after["product_name"].eq(product), "patient_id"]) - recovered

    sp["outcome"] = "No further treatment-basket fill"
    sp.loc[sp["patient_id"].isin(competitor), "outcome"] = "Moved to competitor"
    sp.loc[sp["patient_id"].isin(recovered), "outcome"] = f"Later {product} fill"

    summary = (
        sp.groupby(["discontinue_reason", "outcome"], as_index=False)
        .agg(patients=("patient_id", "nunique"))
        .sort_values(["discontinue_reason", "patients"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary
