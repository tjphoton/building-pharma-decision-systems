"""Channel economics: credit, modeled incremental response, and cost.

Attribution answers which channels sit on converting paths. It says nothing about
what a touch costs or whether the response would have happened anyway. This module
joins three numbers per channel so a high-credit channel can still be revealed as a
poor place to add budget: the attribution credit share, a covariate-adjusted
estimate of the incremental response from one more touch, and the cost of buying
that incremental response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ch08_omnichannel.generation_modules.ch08_config import (
    CHANNEL_METADATA,
    CHANNEL_UNIT_COST,
)


CHANNEL_PREFIX = {
    channel: metadata["prefix"] for channel, metadata in CHANNEL_METADATA.items()
}

# Pre-touch context held fixed while one channel's frequency is nudged by one.
_ECON_COVARIATES = [
    "review_opportunity",
    "evidence_need_score",
    "access_resource_score",
    "digital_response_rate",
    "field_response_rate",
    "total_pressure_90",
]

# Below this modeled incremental response per touch, a cost-per-incremental ratio
# is not meaningful: the touch buys no measurable behavior change.
_MIN_INCREMENTAL = 0.002


def channel_economics(
    panel: pd.DataFrame,
    markov: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-channel credit share, incremental response, and unit economics.

    The incremental estimate is a covariate-adjusted average marginal effect: fit
    one pooled response model on every channel's 90-day frequency plus pre-touch
    context, then read the mean change in predicted response when a single channel
    gains one touch. It is observational and needs a holdout before a budget shift;
    the teaching point is the contrast with attribution credit and with cost.
    """

    frequency_columns = [
        f"{prefix}_frequency_90" for prefix in CHANNEL_PREFIX.values()
    ]
    features = frequency_columns + _ECON_COVARIATES
    model = LogisticRegression(C=0.2, max_iter=2_000, random_state=20260622)
    model.fit(panel[features], panel["future_response"])
    base_probability = model.predict_proba(panel[features])[:, 1]

    credit = markov.set_index("channel")["markov_credit"]
    rows: list[dict[str, object]] = []
    for channel, prefix in CHANNEL_PREFIX.items():
        column = f"{prefix}_frequency_90"
        nudged = panel[features].copy()
        nudged[column] = nudged[column] + 1
        nudged_probability = model.predict_proba(nudged)[:, 1]
        incremental = float(np.mean(nudged_probability - base_probability))
        unit_cost = CHANNEL_UNIT_COST[channel]
        measurable = incremental >= _MIN_INCREMENTAL
        rows.append(
            {
                "channel": channel,
                "markov_credit": float(credit.get(channel, np.nan)),
                "incremental_per_touch": incremental,
                "unit_cost": unit_cost,
                "cost_per_incremental_response": (
                    unit_cost / incremental if measurable else np.nan
                ),
                "measurable_incremental": measurable,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values("markov_credit", ascending=False).reset_index(
        drop=True
    )


def channel_affinity_trace(
    panel: pd.DataFrame,
    plan: pd.DataFrame,
    analysis_date: pd.Timestamp,
    npis: list[str],
) -> pd.DataFrame:
    """Show that the right channel differs by HCP, and that affinity reaches the plan.

    Each HCP carries a stable digital and field response rate from the targeting
    chapter. The dominant rate names the channel an HCP actually responds to, and
    the released plan routes follow-up to that channel rather than to every channel.
    """

    latest = panel.loc[panel["snapshot_date"].eq(analysis_date)].copy()
    latest = latest.loc[latest["npi"].isin(npis)]
    latest["channel_affinity"] = np.where(
        latest["digital_response_rate"] >= latest["field_response_rate"],
        "Digital responder",
        "Field responder",
    )
    merged = latest.merge(
        plan[["npi", "recommended_action", "recommended_channel"]],
        on="npi",
        how="left",
        validate="one_to_one",
    )
    order = {npi: position for position, npi in enumerate(npis)}
    merged["order"] = merged["npi"].map(order)
    return (
        merged.sort_values("order")[
            [
                "npi",
                "digital_response_rate",
                "field_response_rate",
                "channel_affinity",
                "last_response_channel",
                "recommended_channel",
            ]
        ].reset_index(drop=True)
    )
