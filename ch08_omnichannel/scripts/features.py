"""Build leakage-free HCP-account snapshot features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ch08_omnichannel.generation_modules.ch08_config import (
    CHANNEL_METADATA,
    HIGH_PRESSURE_MIN,
    PRESSURE_LOW_MAX,
    PRESSURE_MODERATE_MAX,
    RECENCY_DECAY_DAYS,
    SNAPSHOT_END,
    SNAPSHOT_FREQUENCY,
    SNAPSHOT_START,
)


CHANNEL_PREFIX = {
    channel: metadata["prefix"] for channel, metadata in CHANNEL_METADATA.items()
}


def snapshot_dates() -> pd.DatetimeIndex:
    """Return the monthly snapshot dates used in the chapter."""

    return pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq=SNAPSHOT_FREQUENCY)


def _days_since(
    snapshot_date: pd.Timestamp,
    dates: pd.Series,
    default: int = 365,
) -> int:
    if dates.empty:
        return default
    return int(min((snapshot_date - dates.max()).days, default))


def _last_value(events: pd.DataFrame, column: str, default: str) -> str:
    if events.empty:
        return default
    return str(events.sort_values(["event_date", "event_id"]).iloc[-1][column])


def _decayed_count(
    snapshot_date: pd.Timestamp,
    events: pd.DataFrame,
    decay_days: int = RECENCY_DECAY_DAYS,
) -> float:
    if events.empty:
        return 0.0
    age = (snapshot_date - events["event_date"]).dt.days.clip(lower=0)
    return float(np.exp(-age / decay_days).sum())


def _beta_binomial_rate(
    successes: int,
    trials: int,
    prior_rate: float,
    prior_weight: int,
) -> float:
    return float((successes + prior_weight * prior_rate) / (trials + prior_weight))


def build_snapshot_panel(
    ledger: pd.DataFrame,
    hcp_features: pd.DataFrame,
    hcp_segments: pd.DataFrame,
    engagement_signals: pd.DataFrame,
    account_targets: pd.DataFrame,
    account_actions: pd.DataFrame,
    lookback_days: int = 90,
    outcome_days: int = 28,
) -> pd.DataFrame:
    """Create repeated snapshots using past events and a future response target."""

    static = hcp_features.merge(
        engagement_signals[
            [
                "npi",
                "evidence_need_score",
                "access_resource_score",
                "digital_response_rate",
                "field_response_rate",
            ]
        ],
        on="npi",
        how="left",
        validate="one_to_one",
    )
    static = static.merge(
        hcp_segments[
            ["npi", "segment_name", "engagement_pattern"]
        ],
        on="npi",
        how="left",
        validate="one_to_one",
    )
    static = static.merge(
        account_targets[["account_id", "account_action", "reason_code"]],
        on="account_id",
        how="left",
        validate="many_to_one",
    )
    static = static.merge(
        account_actions[
            ["account_id", "action", "access_flag", "adoption_flag"]
        ].rename(columns={"action": "competitive_action"}),
        on="account_id",
        how="left",
        validate="many_to_one",
    )
    static["segment_name"] = static["segment_name"].fillna("Not clustered")
    static["engagement_pattern"] = static["engagement_pattern"].fillna(
        "Standard channel review"
    )

    rows: list[dict[str, object]] = []
    for snapshot_date in snapshot_dates():
        lookback_start = snapshot_date - pd.Timedelta(days=lookback_days - 1)
        recent_start = snapshot_date - pd.Timedelta(days=29)
        peer_start = snapshot_date - pd.Timedelta(days=179)
        outcome_end = snapshot_date + pd.Timedelta(days=outcome_days)
        market_history = ledger.loc[
            ledger["event_date"].between(lookback_start, snapshot_date)
            & ledger["delivered"]
        ]
        market_response_rate = (
            float(market_history["meaningful_response"].mean())
            if not market_history.empty
            else 0.24
        )
        for hcp in static.itertuples(index=False):
            mine = ledger.loc[
                ledger["npi"].eq(str(hcp.npi))
                & ledger["account_id"].eq(hcp.account_id)
            ]
            history = mine.loc[
                mine["event_date"].between(lookback_start, snapshot_date)
            ]
            recent = mine.loc[
                mine["event_date"].between(recent_start, snapshot_date)
            ]
            peer_history = mine.loc[
                mine["event_date"].between(peer_start, snapshot_date)
            ]
            future = mine.loc[
                mine["event_date"].gt(snapshot_date)
                & mine["event_date"].le(outcome_end)
            ]
            features: dict[str, object] = {
                "snapshot_date": snapshot_date,
                "outcome_end": outcome_end,
                "npi": str(hcp.npi),
                "account_id": hcp.account_id,
                "territory": hcp.territory,
                "account_action": hcp.account_action,
                "competitive_action": hcp.competitive_action,
                "contact_permission_status": hcp.contact_permission_status,
                "contact_permitted": bool(hcp.contact_permitted),
                "segment_name": hcp.segment_name,
                "engagement_pattern": hcp.engagement_pattern,
                "review_opportunity": int(hcp.review_opportunity),
                "evidence_need_score": float(hcp.evidence_need_score),
                "access_resource_score": float(hcp.access_resource_score),
                "digital_response_rate": float(hcp.digital_response_rate),
                "field_response_rate": float(hcp.field_response_rate),
                "days_since_event": _days_since(
                    snapshot_date, history["event_date"]
                ),
                "days_since_response": _days_since(
                    snapshot_date,
                    history.loc[
                        history["meaningful_response"], "event_date"
                    ],
                ),
                "distinct_topics_90": int(history["content_topic"].nunique()),
                "repeated_topics_90": int(
                    max(len(history) - history["content_topic"].nunique(), 0)
                ),
                "channel_diversity_90": int(history["channel"].nunique()),
                "total_pressure_30": int(len(recent)),
                "total_pressure_90": int(len(history)),
                "last_channel": _last_value(history, "channel", "None"),
                "last_response_channel": _last_value(
                    history.loc[history["meaningful_response"]],
                    "channel",
                    "None",
                ),
                "future_response": int(future["meaningful_response"].any()),
                "future_response_count": int(
                    future["meaningful_response"].sum()
                ),
                "future_events": int(len(future)),
                "meaningful_responses_90": int(
                    history["meaningful_response"].sum()
                ),
            }
            for channel, prefix in CHANNEL_PREFIX.items():
                features[f"{prefix}_frequency_30"] = int(
                    recent["channel"].eq(channel).sum()
                )
                features[f"{prefix}_frequency_90"] = int(
                    history["channel"].eq(channel).sum()
                )
                features[f"{prefix}_responses_90"] = int(
                    (
                        history["channel"].eq(channel)
                        & history["meaningful_response"]
                    ).sum()
                )
            features["live_program_attendance_180"] = int(
                (
                    peer_history["channel"].isin(
                        ["Peer program", "Speaker program", "Conference"]
                    )
                    & peer_history["meaningful_response"]
                ).sum()
            )
            features["peer_attendance_180"] = int(
                (
                    peer_history["channel"].eq("Peer program")
                    & peer_history["meaningful_response"]
                ).sum()
            )
            features["speaker_attendance_180"] = int(
                (
                    peer_history["channel"].eq("Speaker program")
                    & peer_history["meaningful_response"]
                ).sum()
            )
            features["event_attendance_180"] = int(
                (
                    peer_history["channel"].eq("Conference")
                    & peer_history["meaningful_response"]
                ).sum()
            )
            delivered_history = history.loc[history["delivered"]]
            response_count = int(delivered_history["meaningful_response"].sum())
            delivered_count = int(len(delivered_history))
            features["shrunken_response_rate_90"] = _beta_binomial_rate(
                response_count,
                delivered_count,
                market_response_rate,
                prior_weight=8,
            )
            features["decayed_total_pressure_90"] = _decayed_count(
                snapshot_date, history
            )
            features["decayed_response_90"] = _decayed_count(
                snapshot_date,
                history.loc[history["meaningful_response"]],
            )
            features["email_clicks_90"] = int(
                (
                    history["channel"].eq("Email")
                    & history["meaningful_response"]
                ).sum()
            )
            features["web_actions_90"] = int(
                (
                    history["channel"].eq("Web")
                    & history["meaningful_response"]
                ).sum()
            )
            features["paid_clicks_90"] = int(
                (
                    history["channel"].eq("Paid media")
                    & history["click_flag"].eq(1)
                ).sum()
            )
            features["account_support_resolutions_90"] = int(
                (
                    history["channel"].eq("Account support")
                    & history["resolution_flag"].eq(1)
                ).sum()
            )
            features["funnel_engagements_90"] = int(
                history[
                    [
                        "open_flag",
                        "click_flag",
                        "attendance_flag",
                        "registration_flag",
                        "landing_visit_flag",
                        "download_flag",
                        "followup_requested_flag",
                    ]
                ].sum().sum()
            )
            rows.append(features)
    panel = pd.DataFrame(rows)
    panel["pressure_band"] = pd.cut(
        panel["total_pressure_30"],
        bins=[-1, PRESSURE_LOW_MAX, PRESSURE_MODERATE_MAX, np.inf],
        labels=["Low", "Moderate", "High"],
    ).astype(str)
    panel["high_pressure"] = panel["total_pressure_30"].ge(HIGH_PRESSURE_MIN)
    panel["review_opportunity_band"] = pd.cut(
        panel["review_opportunity"],
        bins=[-1, 4, 9, np.inf],
        labels=["0-4", "5-9", "10+"],
    ).astype(str)
    return panel.sort_values(["snapshot_date", "npi"]).reset_index(drop=True)


def pressure_response_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize future response by observed channel frequency."""

    rows: list[dict[str, object]] = []
    for channel, prefix in CHANNEL_PREFIX.items():
        frequency = panel[f"{prefix}_frequency_90"].clip(upper=5)
        adjusted = _adjusted_pressure_curve(panel, frequency)
        for bucket, group in panel.assign(frequency_bucket=frequency).groupby(
            "frequency_bucket"
        ):
            bucket_value = int(bucket)
            rows.append(
                {
                    "channel": channel,
                    "frequency_bucket": "5+" if bucket_value == 5 else str(bucket_value),
                    "frequency_order": bucket_value,
                    "snapshots": len(group),
                    "future_responses": int(group["future_response"].sum()),
                    "future_response_rate": group["future_response"].mean(),
                    "adjusted_response_rate": adjusted.get(bucket_value, np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["channel", "frequency_order"]
    ).reset_index(drop=True)


def _adjusted_pressure_curve(
    panel: pd.DataFrame,
    frequency: pd.Series,
) -> dict[int, float]:
    work = panel.copy()
    work["frequency"] = frequency.astype(int)
    covariates = [
        "frequency",
        "review_opportunity",
        "evidence_need_score",
        "access_resource_score",
        "digital_response_rate",
        "field_response_rate",
    ]
    if work["future_response"].nunique() < 2:
        return {}
    model = LogisticRegression(C=0.5, max_iter=1_000, random_state=20260622)
    model.fit(work[covariates], work["future_response"])
    adjusted: dict[int, float] = {}
    for bucket in sorted(work["frequency"].unique()):
        scenario = work[covariates].copy()
        scenario["frequency"] = bucket
        adjusted[int(bucket)] = float(model.predict_proba(scenario)[:, 1].mean())
    return adjusted


_SATURATION_COVARIATES = [
    "total_pressure_30",
    "review_opportunity",
    "evidence_need_score",
    "access_resource_score",
    "digital_response_rate",
    "field_response_rate",
]


def saturation_summary(panel: pd.DataFrame, cap_threshold: float = 0.05) -> pd.DataFrame:
    """Separate the selection effect of contact volume from its marginal return.

    Observed reach is the raw probability of a meaningful response in the next 28
    days by recent contact band. It rises steeply, but more-contacted HCPs are also
    more responsive to begin with, so most of that slope is who was contacted, not
    how often. The adjusted reach holds opportunity, evidence need, access need, and
    channel affinity fixed and moves only recent contact; its marginal gain is the
    honest return on one more touch, and it shrinks as contact rises.
    """

    work = panel.copy()
    work["frequency_band"] = work["total_pressure_30"].clip(upper=5)
    model = LogisticRegression(C=0.5, max_iter=1_000, random_state=20260622)
    model.fit(work[_SATURATION_COVARIATES], work["future_response"])
    adjusted: dict[int, float] = {}
    for band in range(6):
        scenario = work[_SATURATION_COVARIATES].copy()
        scenario["total_pressure_30"] = band
        adjusted[band] = float(model.predict_proba(scenario)[:, 1].mean())
    grouped = (
        work.groupby("frequency_band", as_index=False)
        .agg(
            snapshots=("npi", "size"),
            observed_reach=("future_response", "mean"),
        )
        .sort_values("frequency_band")
        .reset_index(drop=True)
    )
    grouped["adjusted_reach"] = grouped["frequency_band"].map(adjusted)
    grouped["adjusted_marginal_gain"] = grouped["adjusted_reach"].diff()
    grouped["recent_events"] = grouped["frequency_band"].map(
        lambda value: "5+" if value == 5 else str(int(value))
    )
    return grouped[
        [
            "recent_events",
            "frequency_band",
            "snapshots",
            "observed_reach",
            "adjusted_reach",
            "adjusted_marginal_gain",
        ]
    ]


def response_shrinkage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Show how sparse observed response rates shrink toward the channel mean."""

    latest = panel.loc[panel["snapshot_date"].eq(panel["snapshot_date"].max())].copy()
    latest["observed_response_rate_90"] = np.where(
        latest["total_pressure_90"].gt(0),
        latest["meaningful_responses_90"] / latest["total_pressure_90"],
        np.nan,
    )
    sparse = (
        latest.loc[latest["total_pressure_90"].between(1, 2)]
        .sort_values(["observed_response_rate_90", "total_pressure_90"], ascending=[False, True])
        .head(3)
        .assign(evidence_level="Sparse")
    )
    established = (
        latest.loc[latest["total_pressure_90"].ge(8)]
        .assign(distance_from_market=lambda frame: (frame["observed_response_rate_90"] - frame["shrunken_response_rate_90"]).abs())
        .sort_values(["distance_from_market", "total_pressure_90"], ascending=[False, False])
        .head(3)
        .assign(evidence_level="Established")
    )
    examples = pd.concat([sparse, established], ignore_index=True)
    return examples[
        [
            "evidence_level",
            "npi",
            "account_id",
            "meaningful_responses_90",
            "total_pressure_90",
            "observed_response_rate_90",
            "shrunken_response_rate_90",
        ]
    ].reset_index(drop=True)
