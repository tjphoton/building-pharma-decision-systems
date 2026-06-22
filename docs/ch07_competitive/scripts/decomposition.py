"""Payer-region uncertainty and two-axis access-adoption decisions."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import beta


def wilson_interval(
    successes: pd.Series, totals: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return Wilson 95% interval bounds."""

    z = NormalDist().inv_cdf(0.975)
    n = pd.to_numeric(totals, errors="coerce").astype(float)
    x = pd.to_numeric(successes, errors="coerce").astype(float)
    p = x / n.replace(0, np.nan)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half, center + half


def payer_region_decisions(
    starts: pd.DataFrame,
    policy: pd.DataFrame,
    friction: pd.DataFrame,
    *,
    brand: str,
    benchmark: float,
    min_treated: int,
    posterior_threshold: float,
    restricted_lives_threshold: float,
    friction_threshold: float,
    rule_version: str,
    analysis_date: str,
) -> pd.DataFrame:
    """Join evidence and assign access, adoption, mixed, or monitoring action."""

    b = policy.loc[policy["product_name"].eq(brand)].copy()
    columns = [
        "payer_id",
        "region",
        "payer_type",
        "coverage_status",
        "access_state",
        "enrolled_lives",
        "unrestricted",
        "workable_coverage",
        "access_quality_weight",
        "material_access_barrier",
    ]
    out = b[columns].merge(starts, on=["payer_id", "region"], how="left")
    out = out.merge(friction, on=["payer_id", "region"], how="left")
    numeric = [
        "treated_patients",
        "brand_starts",
        "competitor_starts",
        "submitted_attempts",
        "completed_attempts",
        "unresolved_attempts",
        "attempts_with_pend",
    ]
    out[numeric] = out[numeric].fillna(0)
    out["brand_share"] = out["brand_starts"] / out["treated_patients"].replace(
        0, np.nan
    )
    out["share_lower_95"], out["share_upper_95"] = wilson_interval(
        out["brand_starts"], out["treated_patients"]
    )

    total_brand = out["brand_starts"].sum()
    total_comp = out["competitor_starts"].sum()
    prior_strength = 40.0
    prior_mean = total_brand / (total_brand + total_comp)
    alpha0 = prior_mean * prior_strength
    beta0 = (1 - prior_mean) * prior_strength
    out["posterior_mean_share"] = (out["brand_starts"] + alpha0) / (
        out["treated_patients"] + alpha0 + beta0
    )
    out["probability_below_benchmark"] = beta.cdf(
        benchmark,
        out["brand_starts"] + alpha0,
        out["competitor_starts"] + beta0,
    )
    out["restricted_lives"] = np.where(
        out["material_access_barrier"], out["enrolled_lives"], 0
    )
    out["restricted_lives_rate"] = out["restricted_lives"] / out["enrolled_lives"]
    out["unresolved_rate"] = (
        out["unresolved_attempts"] / out["submitted_attempts"].replace(0, np.nan)
    ).fillna(0)
    out["evidence_sufficient"] = out["treated_patients"].ge(min_treated)
    out["access_flag"] = out["material_access_barrier"] & out[
        "restricted_lives_rate"
    ].ge(restricted_lives_threshold) | out["unresolved_rate"].ge(friction_threshold)
    out["adoption_flag"] = out["evidence_sufficient"] & out[
        "probability_below_benchmark"
    ].ge(posterior_threshold)
    out["action"] = np.select(
        [
            ~out["evidence_sufficient"],
            out["access_flag"] & out["adoption_flag"],
            out["access_flag"] & ~out["adoption_flag"],
            ~out["access_flag"] & out["adoption_flag"],
        ],
        ["Monitor", "Dual workstream", "Access work", "Adoption review"],
        default="Defend and learn",
    )
    out["owner"] = out["action"].map(
        {
            "Monitor": "Commercial analytics",
            "Dual workstream": "Market access and field analytics",
            "Access work": "Market access",
            "Adoption review": "Field analytics",
            "Defend and learn": "Brand and field analytics",
        }
    )
    out["reason_code"] = np.select(
        [
            ~out["evidence_sufficient"],
            out["access_flag"] & out["adoption_flag"],
            out["access_flag"] & ~out["adoption_flag"],
            ~out["access_flag"] & out["adoption_flag"],
        ],
        [
            "MONITOR_SPARSE_TREATED_DENOMINATOR",
            "DUAL_ACCESS_AND_ADOPTION",
            "ROUTE_ACCESS_BARRIER",
            "INVESTIGATE_ADOPTION_GAP",
        ],
        default="DEFEND_SUPPORTED_ADOPTION",
    )
    out["analysis_date"] = analysis_date
    out["decision_rule_version"] = rule_version
    out["refresh_date"] = str(
        (pd.Timestamp(analysis_date) + pd.offsets.QuarterEnd()).date()
    )
    return out.sort_values(
        ["action", "restricted_lives", "treated_patients"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
