"""Observed sequence and attribution diagnostics."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
import pandas as pd


def observed_sequences(
    ledger: pd.DataFrame,
    scored_snapshots: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    lookback_days: int = 90,
    maximum_events: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build recent channel sequences and adjacent transition counts."""

    start = snapshot_date - pd.Timedelta(days=lookback_days - 1)
    recent = ledger.loc[
        ledger["event_date"].between(start, snapshot_date)
    ].copy()
    outcome = scored_snapshots.loc[
        scored_snapshots["snapshot_date"].eq(snapshot_date),
        ["npi", "account_id", "future_response"],
    ]
    rows: list[dict[str, object]] = []
    transitions: dict[tuple[str, str], int] = {}
    for (npi, account_id), group in recent.groupby(["npi", "account_id"]):
        ordered = group.sort_values(["event_date", "event_id"]).tail(
            maximum_events
        )
        channels = ordered["channel"].tolist()
        for first, second in pairwise(channels):
            transitions[(first, second)] = transitions.get(
                (first, second), 0
            ) + 1
        rows.append(
            {
                "npi": npi,
                "account_id": account_id,
                "events_in_sequence": len(ordered),
                "channel_sequence": " > ".join(channels),
                "response_sequence": " > ".join(
                    ordered["response_type"].tolist()
                ),
                "last_event_date": ordered["event_date"].max(),
            }
        )
    detail = pd.DataFrame(rows).merge(
        outcome,
        on=["npi", "account_id"],
        how="left",
        validate="one_to_one",
    )
    summary = (
        detail.groupby("channel_sequence", as_index=False)
        .agg(
            relationships=("npi", "size"),
            future_responses=("future_response", "sum"),
            future_response_rate=("future_response", "mean"),
        )
        .sort_values(
            ["relationships", "future_response_rate"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    transition_table = pd.DataFrame(
        [
            {"from_channel": key[0], "to_channel": key[1], "transitions": value}
            for key, value in transitions.items()
        ]
    ).sort_values("transitions", ascending=False)
    return detail, summary, transition_table.reset_index(drop=True)


def sequence_pattern_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Group exact paths into interpretable sequence patterns."""

    work = detail.copy()
    work["sequence_pattern"] = np.select(
        [
            work["channel_sequence"].str.contains("Field")
            & work["channel_sequence"].str.contains("Email|Web", regex=True),
            work["channel_sequence"].str.contains("Email > Email"),
            work["channel_sequence"].str.contains(
                "Peer program|Speaker program|Conference", regex=True
            ),
            work["channel_sequence"].str.contains("Account support"),
        ],
        [
            "Field plus digital",
            "Repeated email",
            "Live program",
            "Account support",
        ],
        default="Other sequence",
    )
    summary = (
        work.groupby("sequence_pattern", as_index=False)
        .agg(
            relationships=("npi", "size"),
            future_responses=("future_response", "sum"),
            future_response_rate=("future_response", "mean"),
        )
        .sort_values(["future_response_rate", "relationships"], ascending=[False, False])
        .reset_index(drop=True)
    )
    overall = work["future_response"].mean()
    summary["lift_vs_all_sequences"] = summary["future_response_rate"] / overall
    summary["pattern_rule"] = summary["sequence_pattern"].map(
        {
            "Field plus digital": "last 3 events include field and email or web",
            "Repeated email": "last 3 events include consecutive email touches",
            "Live program": "last 3 events include peer, speaker, or conference",
            "Account support": "last 3 events include account support",
            "Other sequence": "none of the named rules",
        }
    )
    return summary


def sequence_pattern_examples(detail: pd.DataFrame) -> pd.DataFrame:
    """Return one readable example path for each sequence pattern."""

    work = detail.copy()
    patterns = sequence_pattern_summary(detail)[["sequence_pattern"]]
    work["sequence_pattern"] = np.select(
        [
            work["channel_sequence"].str.contains("Field")
            & work["channel_sequence"].str.contains("Email|Web", regex=True),
            work["channel_sequence"].str.contains("Email > Email"),
            work["channel_sequence"].str.contains(
                "Peer program|Speaker program|Conference", regex=True
            ),
            work["channel_sequence"].str.contains("Account support"),
        ],
        [
            "Field plus digital",
            "Repeated email",
            "Live program",
            "Account support",
        ],
        default="Other sequence",
    )
    ranked = (
        work.sort_values(
            ["sequence_pattern", "future_response", "events_in_sequence"],
            ascending=[True, False, False],
        )
        .groupby("sequence_pattern", as_index=False)
        .first()
    )
    return patterns.merge(
        ranked[
            [
                "sequence_pattern",
                "channel_sequence",
                "response_sequence",
                "future_response",
            ]
        ],
        on="sequence_pattern",
        how="left",
        validate="one_to_one",
    )


def attribution_comparison(
    ledger: pd.DataFrame,
    start_date: pd.Timestamp = pd.Timestamp("2024-06-01"),
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Allocate the same response events with four common accounting rules."""

    rows: list[dict[str, object]] = []
    for (npi, account_id), group in ledger.groupby(["npi", "account_id"]):
        ordered = group.sort_values(["event_date", "event_id"])
        responses = ordered.loc[
            ordered["meaningful_response"]
            & ordered["event_date"].ge(start_date)
        ]
        if responses.empty:
            continue
        response = responses.iloc[0]
        window_start = response["event_date"] - pd.Timedelta(
            days=lookback_days
        )
        touches = ordered.loc[
            ordered["event_date"].between(window_start, response["event_date"])
            & ordered["delivered"]
        ].copy()
        if touches.empty:
            continue
        elapsed = (
            response["event_date"] - touches["event_date"]
        ).dt.days.to_numpy()
        decay = np.exp(-elapsed / 30)
        decay = decay / decay.sum()
        for position, (_, touch) in enumerate(touches.iterrows()):
            rows.append(
                {
                    "npi": npi,
                    "account_id": account_id,
                    "channel": touch["channel"],
                    "first_touch": float(position == 0),
                    "last_touch": float(position == len(touches) - 1),
                    "linear": 1 / len(touches),
                    "time_decay": float(decay[position]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby("channel", as_index=False)[
        ["first_touch", "last_touch", "linear", "time_decay"]
    ].sum()
    for column in ["first_touch", "last_touch", "linear", "time_decay"]:
        summary[column] = 100 * summary[column] / summary[column].sum()
    return summary.sort_values("linear", ascending=False).reset_index(drop=True)


def _build_journeys(
    ledger: pd.DataFrame,
    start_date: pd.Timestamp,
    lookback_days: int,
) -> list[tuple[list[str], bool]]:
    """Return one delivered-channel path per relationship with a conversion flag.

    A converting path is the ordered delivered channels in the prior `lookback_days`
    before the first meaningful response. A non-converting path is the relationship's
    delivered channels from `start_date` onward with no meaningful response.
    """

    journeys: list[tuple[list[str], bool]] = []
    for _, group in ledger.groupby(["npi", "account_id"]):
        ordered = group.sort_values(["event_date", "event_id"])
        delivered = ordered.loc[ordered["delivered"]]
        responses = delivered.loc[
            delivered["meaningful_response"]
            & delivered["event_date"].ge(start_date)
        ]
        if not responses.empty:
            response = responses.iloc[0]
            window_start = response["event_date"] - pd.Timedelta(days=lookback_days)
            window = delivered.loc[
                delivered["event_date"].between(
                    window_start, response["event_date"]
                )
            ]
            journeys.append((window["channel"].tolist(), True))
        else:
            window = delivered.loc[delivered["event_date"].ge(start_date)]
            if window.empty:
                continue
            journeys.append((window["channel"].tolist(), False))
    return journeys


def _conversion_probability(
    channels: list[str],
    transitions: dict[tuple[str, str], float],
) -> float:
    """Absorbing-Markov conversion probability starting from the start state."""

    transient = ["start", *channels]
    index = {state: position for position, state in enumerate(transient)}
    size = len(transient)
    q_matrix = np.zeros((size, size))
    to_conversion = np.zeros(size)
    for (source, target), probability in transitions.items():
        if source not in index:
            continue
        row = index[source]
        if target == "conversion":
            to_conversion[row] += probability
        elif target in index:
            q_matrix[row, index[target]] += probability
    fundamental = np.linalg.inv(np.eye(size) - q_matrix)
    absorbed = fundamental @ to_conversion
    return float(absorbed[index["start"]])


def markov_attribution(
    ledger: pd.DataFrame,
    start_date: pd.Timestamp = pd.Timestamp("2024-06-01"),
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Allocate channel credit by Markov removal effect.

    The removal effect of a channel is the relative drop in modeled conversion
    probability when every path through that channel is redirected to no response.
    Credits normalize the removal effects to 100%.
    """

    journeys = _build_journeys(ledger, start_date, lookback_days)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    channels: set[str] = set()
    for path, converted in journeys:
        states = ["start", *path, "conversion" if converted else "null"]
        channels.update(path)
        for source, target in pairwise(states):
            counts[(source, target)] += 1
    from_totals: dict[str, int] = defaultdict(int)
    for (source, _target), count in counts.items():
        from_totals[source] += count
    transitions = {
        edge: count / from_totals[edge[0]] for edge, count in counts.items()
    }
    ordered_channels = sorted(channels)
    base = _conversion_probability(ordered_channels, transitions)

    rows: list[dict[str, object]] = []
    for channel in ordered_channels:
        removed: dict[tuple[str, str], float] = defaultdict(float)
        for (source, target), probability in transitions.items():
            if source == channel:
                continue
            if target == channel:
                removed[(source, "null")] += probability
            else:
                removed[(source, target)] += probability
        remaining = [name for name in ordered_channels if name != channel]
        without = _conversion_probability(remaining, dict(removed))
        rows.append(
            {
                "channel": channel,
                "removal_effect": (base - without) / base if base else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["markov_credit"] = (
        100 * result["removal_effect"] / result["removal_effect"].sum()
    )
    return result.sort_values("markov_credit", ascending=False).reset_index(
        drop=True
    )
