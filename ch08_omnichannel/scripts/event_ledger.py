"""Build the source-aware Chapter 8 event ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ch08_omnichannel.generation_modules.ch08_config import CHANNELS


VALID_CHANNELS = set(CHANNELS)


def build_event_ledger(events: pd.DataFrame) -> pd.DataFrame:
    """Map source events into common analytical fields."""

    ledger = events.copy()
    ledger["delivered"] = ledger["delivery_status"].eq("Delivered")
    field_response = (
        ledger["channel"].isin(["Field", "Phone"])
        & ledger["field_outcome"].isin(["Positive", "Follow-up"])
    )
    digital_response = (
        ledger["channel"].isin(["Email", "Web", "Paid media"])
        & ledger["click_flag"].eq(1)
    )
    direct_mail_response = (
        ledger["channel"].eq("Direct mail")
        & ledger["landing_visit_flag"].eq(1)
    )
    event_response = (
        ledger["channel"].isin(["Peer program", "Speaker program", "Conference"])
        & ledger["attendance_flag"].eq(1)
    )
    account_response = (
        ledger["channel"].eq("Account support")
        & ledger["resolution_flag"].eq(1)
    )
    ledger["meaningful_response"] = (
        field_response
        | digital_response
        | direct_mail_response
        | event_response
        | account_response
    )
    ledger["response_type"] = np.select(
        [
            ledger["channel"].isin(["Field", "Phone"]) & field_response,
            ledger["channel"].eq("Email") & ledger["click_flag"].eq(1),
            ledger["channel"].eq("Email") & ledger["open_flag"].eq(1),
            ledger["channel"].eq("Web") & ledger["click_flag"].eq(1),
            ledger["channel"].isin(["Peer program", "Speaker program", "Conference"])
            & ledger["attendance_flag"].eq(1),
            ledger["channel"].eq("Paid media") & ledger["click_flag"].eq(1),
            direct_mail_response,
            ledger["channel"].eq("Account support") & ledger["resolution_flag"].eq(1),
            ledger["registration_flag"].eq(1),
            ledger["viewable_impression_flag"].eq(1),
            ledger["landing_visit_flag"].eq(1),
        ],
        [
            ledger["field_outcome"],
            "Clicked",
            "Opened",
            "Qualified action",
            "Attended",
            "Clicked",
            "Landing visit",
            "Resolved",
            "Registered",
            "Viewable impression",
            "Landing visit",
        ],
        default=np.where(ledger["delivered"], "No response", "Not delivered"),
    )
    ledger["response_strength"] = np.select(
        [
            ledger["field_outcome"].eq("Positive"),
            ledger["field_outcome"].eq("Follow-up"),
            ledger["resolution_flag"].eq(1),
            ledger["attendance_flag"].eq(1),
            ledger["click_flag"].eq(1),
            ledger["download_flag"].eq(1),
            ledger["registration_flag"].eq(1),
            ledger["landing_visit_flag"].eq(1),
            ledger["open_flag"].eq(1),
            ledger["viewable_impression_flag"].eq(1),
        ],
        [1.0, 0.8, 0.85, 0.8, 0.7, 0.65, 0.45, 0.35, 0.1, 0.05],
        default=0.0,
    )
    if not set(ledger["channel"]).issubset(VALID_CHANNELS):
        invalid = sorted(set(ledger["channel"]) - VALID_CHANNELS)
        raise ValueError(f"Unmapped channels: {invalid}")
    columns = [
        "event_id",
        "event_date",
        "npi",
        "account_id",
        "territory",
        "source_system",
        "channel",
        "event_type",
        "content_topic",
        "delivered",
        "response_type",
        "response_strength",
        "meaningful_response",
        "permission_status_at_event",
        "campaign_id",
        "sequence_id",
        "sent_flag",
        "bounce_flag",
        "open_flag",
        "click_flag",
        "attendance_flag",
        "viewable_impression_flag",
        "landing_visit_flag",
        "registration_flag",
        "download_flag",
        "followup_requested_flag",
        "transfer_flag",
        "resolution_flag",
        "field_outcome",
    ]
    return ledger[columns].sort_values(
        ["npi", "account_id", "event_date", "event_id"]
    ).reset_index(drop=True)


def channel_delivery_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize delivery, response, and reach by channel."""

    summary = (
        ledger.groupby("channel", as_index=False)
        .agg(
            events=("event_id", "nunique"),
            delivered_events=("delivered", "sum"),
            opened_events=("open_flag", "sum"),
            clicked_events=("click_flag", "sum"),
            attended_events=("attendance_flag", "sum"),
            registered_events=("registration_flag", "sum"),
            viewable_impressions=("viewable_impression_flag", "sum"),
            landing_visits=("landing_visit_flag", "sum"),
            downloads=("download_flag", "sum"),
            followup_requests=("followup_requested_flag", "sum"),
            resolved_events=("resolution_flag", "sum"),
            meaningful_responses=("meaningful_response", "sum"),
            reached_relationships=(
                "npi",
                lambda values: int(
                    ledger.loc[values.index, ["npi", "account_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            ),
        )
    )
    summary["delivery_rate"] = summary["delivered_events"] / summary["events"]
    summary["response_rate_per_delivered"] = (
        summary["meaningful_responses"] / summary["delivered_events"]
    )
    return summary.sort_values("events", ascending=False).reset_index(drop=True)


def email_quality_summary(
    ledger: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Compare email open and click metrics with the synthetic answer key."""

    email = ledger.loc[ledger["channel"].eq("Email")].merge(
        truth[["event_id", "machine_open_answer_key"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    delivered = int(email["delivered"].sum())
    opens = int(email["open_flag"].sum())
    machine_opens = int(email["machine_open_answer_key"].fillna(False).sum())
    clicks = int(email["click_flag"].sum())
    human_opens = opens - machine_opens
    rows = [
        ("Raw open rate", opens, delivered),
        ("Human open rate (answer key)", human_opens, delivered),
        ("Click rate", clicks, delivered),
        ("Click-to-open rate", clicks, opens),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "events": numerator,
                "base_events": base,
                "rate": numerator / base if base else np.nan,
            }
            for metric, numerator, base in rows
        ]
    )
