"""Effective-dated access policy and enrollment-denominator measures."""

from __future__ import annotations

import numpy as np
import pandas as pd


ACCESS_QUALITY = {
    "Covered": 1.00,
    "Covered with PA": 0.70,
    "Covered with Step Edit": 0.50,
    "Non-covered": 0.05,
}


def build_policy(
    access: pd.DataFrame,
    formulary: pd.DataFrame,
    enrollment: pd.DataFrame,
    analysis_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create one canonical policy row per payer-region-product."""

    active_access = access.loc[
        access["effective_start"].le(analysis_date)
        & access["effective_end"].ge(analysis_date)
    ].copy()
    active_formulary = formulary.loc[
        formulary["effective_start"].le(analysis_date)
        & formulary["effective_end"].ge(analysis_date)
    ].copy()
    f = active_formulary.rename(
        columns={
            "plan_id": "payer_id",
            "prior_authorization": "formulary_pa",
            "step_therapy": "formulary_step",
            "specialty_pharmacy": "formulary_sp",
        }
    )
    keep = [
        "payer_id",
        "product_name",
        "tier",
        "formulary_pa",
        "formulary_step",
        "quantity_limit",
        "formulary_sp",
    ]
    policy = active_access.merge(
        f[keep], on=["payer_id", "product_name"], how="left", validate="many_to_one"
    )
    policy = policy.merge(
        enrollment[["payer_id", "region", "enrolled_lives", "as_of_date"]],
        on=["payer_id", "region"],
        how="left",
        validate="many_to_one",
    )
    policy["prior_authorization_flag"] = policy["prior_authorization"].eq(
        "Yes"
    ) | policy["formulary_pa"].eq("Yes")
    policy["step_therapy_flag"] = policy["step_edit"].eq("Yes") | policy[
        "formulary_step"
    ].eq("Yes")
    policy["specialty_pharmacy_flag"] = policy["specialty_pharmacy_required"].eq(
        "Yes"
    ) | policy["formulary_sp"].eq("Yes")
    policy["quantity_limit_flag"] = policy["quantity_limit"].eq("Yes")
    policy["unrestricted"] = policy["coverage_status"].eq("Covered") & ~policy[
        [
            "prior_authorization_flag",
            "step_therapy_flag",
            "specialty_pharmacy_flag",
            "quantity_limit_flag",
        ]
    ].any(axis=1)
    policy["workable_coverage"] = ~policy["coverage_status"].eq("Non-covered")
    policy["access_quality_weight"] = policy["coverage_status"].map(ACCESS_QUALITY)
    policy["access_state"] = np.select(
        [
            policy["coverage_status"].eq("Non-covered"),
            policy["step_therapy_flag"],
            policy["prior_authorization_flag"],
            policy["unrestricted"],
        ],
        ["Non-covered", "Step edit", "Prior authorization", "Unrestricted"],
        default="Other restriction",
    )
    policy["material_access_barrier"] = policy["access_state"].isin(
        ["Non-covered", "Step edit"]
    )
    return policy.sort_values(["payer_id", "region", "product_name"]).reset_index(
        drop=True
    )


def covered_lives_summary(
    policy: pd.DataFrame, brand: str = "Roventra"
) -> pd.DataFrame:
    """Calculate plan coverage, covered lives, unrestricted lives, and quality."""

    b = policy.loc[policy["product_name"].eq(brand)].copy()
    rows: list[dict] = []
    for payer_type, group in [("All", b), *list(b.groupby("payer_type"))]:
        total_lives = int(group["enrolled_lives"].sum())
        rows.append(
            {
                "payer_type": payer_type,
                "plans": len(group),
                "covered_plans": int(group["workable_coverage"].sum()),
                "plan_coverage_rate": group["workable_coverage"].mean(),
                "total_lives": total_lives,
                "covered_lives": int(
                    group.loc[group["workable_coverage"], "enrolled_lives"].sum()
                ),
                "covered_lives_rate": (
                    group.loc[group["workable_coverage"], "enrolled_lives"].sum()
                    / total_lives
                ),
                "unrestricted_lives": int(
                    group.loc[group["unrestricted"], "enrolled_lives"].sum()
                ),
                "unrestricted_lives_rate": (
                    group.loc[group["unrestricted"], "enrolled_lives"].sum()
                    / total_lives
                ),
                "access_quality_score": (
                    (group["enrolled_lives"] * group["access_quality_weight"]).sum()
                    / total_lives
                ),
            }
        )
    return pd.DataFrame(rows)


def restriction_lives(policy: pd.DataFrame, brand: str = "Roventra") -> pd.DataFrame:
    """Report brand lives in each mutually exclusive access state."""

    b = policy.loc[policy["product_name"].eq(brand)]
    result = b.groupby("access_state", as_index=False).agg(
        payer_region_cells=("payer_id", "size"),
        enrolled_lives=("enrolled_lives", "sum"),
    )
    result["lives_share"] = result["enrolled_lives"] / result["enrolled_lives"].sum()
    return result.sort_values("enrolled_lives", ascending=False).reset_index(drop=True)


def relative_position(policy: pd.DataFrame, brand: str = "Roventra") -> pd.DataFrame:
    """Compare brand access quality with the strongest competitor."""

    keys = ["payer_id", "region"]
    b = policy.loc[
        policy["product_name"].eq(brand),
        keys + ["access_state", "access_quality_weight", "enrolled_lives"],
    ].rename(
        columns={
            "access_state": "brand_access_state",
            "access_quality_weight": "brand_access_quality",
        }
    )
    competitors = policy.loc[~policy["product_name"].eq(brand)].copy()
    best = (
        competitors.sort_values(
            [*keys, "access_quality_weight"], ascending=[True, True, False]
        )
        .drop_duplicates(keys)[
            keys + ["product_name", "access_state", "access_quality_weight"]
        ]
        .rename(
            columns={
                "product_name": "best_competitor",
                "access_state": "competitor_access_state",
                "access_quality_weight": "competitor_access_quality",
            }
        )
    )
    out = b.merge(best, on=keys, how="left", validate="one_to_one")
    out["access_quality_gap"] = (
        out["brand_access_quality"] - out["competitor_access_quality"]
    )
    out["position"] = np.select(
        [out["access_quality_gap"].gt(0), out["access_quality_gap"].eq(0)],
        ["Brand favored", "Parity"],
        default="Competitor favored",
    )
    return out
