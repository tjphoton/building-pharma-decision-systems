"""Generate Chapter 8-only HCP omnichannel events."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ch08_config import (
        ANALYSIS_DATE,
        CHANNELS,
        CHANNEL_BASE_WEIGHTS,
        CHANNEL_METADATA,
        DATA_END,
        DATA_START,
        FIELD_THEN_DIGITAL_BASE_LOGIT,
        FIELD_THEN_DIGITAL_EVIDENCE_GAIN,
        FIELD_THEN_DIGITAL_WINDOW,
        GENERATOR_VERSION,
        LIVE_PROGRAM_EFFECT_WINDOW,
        LIVE_PROGRAM_UPLIFT_BASE_LOGIT,
        LIVE_PROGRAM_UPLIFT_EVIDENCE_GAIN,
        MONTHLY_LAUNCH_RAMP,
        OPT_OUT_END,
        OPT_OUT_START,
        RESPONSE_ACCESS_WEIGHT,
        RESPONSE_AFFINITY_WEIGHT,
        RESPONSE_BASE_LOGIT,
        RESPONSE_EVIDENCE_WEIGHT,
        RESPONSE_FATIGUE_WEIGHT,
        RESPONSE_MEMORY_WEIGHT,
        RESPONSE_TOPIC_WEIGHT,
        SEED,
        TOPICS,
        TRACE_OVERRIDES,
        TRACE_OVERRIDE_RESPONSE_PROBABILITY,
    )
except ImportError:  # Direct script execution.
    from ch08_config import (
        ANALYSIS_DATE,
        CHANNELS,
        CHANNEL_BASE_WEIGHTS,
        CHANNEL_METADATA,
        DATA_END,
        DATA_START,
        FIELD_THEN_DIGITAL_BASE_LOGIT,
        FIELD_THEN_DIGITAL_EVIDENCE_GAIN,
        FIELD_THEN_DIGITAL_WINDOW,
        GENERATOR_VERSION,
        LIVE_PROGRAM_EFFECT_WINDOW,
        LIVE_PROGRAM_UPLIFT_BASE_LOGIT,
        LIVE_PROGRAM_UPLIFT_EVIDENCE_GAIN,
        MONTHLY_LAUNCH_RAMP,
        OPT_OUT_END,
        OPT_OUT_START,
        RESPONSE_ACCESS_WEIGHT,
        RESPONSE_AFFINITY_WEIGHT,
        RESPONSE_BASE_LOGIT,
        RESPONSE_EVIDENCE_WEIGHT,
        RESPONSE_FATIGUE_WEIGHT,
        RESPONSE_MEMORY_WEIGHT,
        RESPONSE_TOPIC_WEIGHT,
        SEED,
        TOPICS,
        TRACE_OVERRIDES,
        TRACE_OVERRIDE_RESPONSE_PROBABILITY,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(repo_root: Path) -> dict[str, Path]:
    """Return the read-only source files used by the generator."""

    return {
        "hcp_features": (
            repo_root / "ch06_hcp" / "assets" / "generated_outputs" / "hcp_features.csv"
        ),
        "hcp_segments": (
            repo_root / "ch06_hcp" / "assets" / "generated_outputs" / "hcp_segments.csv"
        ),
        "engagement_signals": (
            repo_root / "ch06_hcp" / "data" / "generated" / "engagement_signals.csv"
        ),
        "account_targets": (
            repo_root
            / "ch06_hcp"
            / "assets"
            / "generated_outputs"
            / "account_targets.csv"
        ),
        "account_actions": (
            repo_root
            / "ch07_competitive"
            / "assets"
            / "generated_outputs"
            / "account_access_adoption_actions.csv"
        ),
    }


def load_sources(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Load upstream records without modifying them."""

    paths = source_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing upstream Chapter 8 inputs:\n" + "\n".join(missing)
        )
    return {
        "hcp_features": pd.read_csv(paths["hcp_features"], dtype={"npi": str}),
        "hcp_segments": pd.read_csv(paths["hcp_segments"], dtype={"npi": str}),
        "engagement_signals": pd.read_csv(
            paths["engagement_signals"], dtype={"npi": str}
        ),
        "account_targets": pd.read_csv(paths["account_targets"]),
        "account_actions": pd.read_csv(paths["account_actions"]),
    }


def _logistic(value: float) -> float:
    return float(1 / (1 + np.exp(-value)))


def _select_channel(
    rng: np.random.Generator,
    priority: bool,
    digital_affinity: float,
    field_affinity: float,
    recent_channel_response: dict[str, float],
) -> str:
    values = []
    for channel in CHANNELS:
        weight = CHANNEL_BASE_WEIGHTS[channel]
        if channel in {"Field", "Phone"}:
            weight += 0.18 * priority + 0.20 * field_affinity
        if channel in {"Email", "Web", "Paid media", "Direct mail"}:
            weight += 0.18 * digital_affinity
        if channel in {"Peer program", "Speaker program", "Conference"}:
            weight += 0.10 * digital_affinity + 0.08 * priority
        if channel == "Account support":
            weight += 0.18 * priority
        weight += 0.38 * recent_channel_response[channel]
        values.append(weight)
    weights = np.array(values)
    weights = weights / weights.sum()
    return str(rng.choice(CHANNELS, p=weights))


def _select_topic(
    rng: np.random.Generator,
    access_need: float,
    evidence_need: float,
) -> str:
    weights = np.array(
        [
            0.24 + 0.34 * evidence_need,
            0.25,
            0.17 + 0.34 * access_need,
            0.18,
        ]
    )
    weights = weights / weights.sum()
    return str(rng.choice(TOPICS, p=weights))


def _event_fields(
    rng: np.random.Generator,
    channel: str,
    response_probability: float,
    forced_outcome: str | None = None,
) -> dict[str, object]:
    """Return source-specific event fields for one channel."""

    delivered = rng.random() > (0.02 if channel != "Paid media" else 0.05)
    meaningful = bool(delivered and rng.random() < response_probability)
    machine_open = False
    human_open = False
    open_flag = 0
    click_flag = 0
    attendance_flag = 0
    field_outcome = ""
    sent_flag = int(channel in {"Email", "Direct mail"} or delivered)
    bounce_flag = int(not delivered and channel in {"Email", "Direct mail"})
    viewable_impression_flag = 0
    landing_visit_flag = 0
    registration_flag = 0
    download_flag = 0
    followup_requested_flag = 0
    transfer_flag = 0
    resolution_flag = 0

    if channel == "Field":
        if meaningful:
            field_outcome = str(rng.choice(["Positive", "Follow-up"], p=[0.55, 0.45]))
            followup_requested_flag = int(field_outcome == "Follow-up")
        else:
            field_outcome = str(rng.choice(["Neutral", "No reach"], p=[0.62, 0.38]))
    elif channel == "Email":
        click_flag = int(meaningful)
        machine_open = bool(delivered and not meaningful and rng.random() < 0.30)
        human_open = bool(delivered and rng.random() < 0.42)
        open_flag = int(click_flag or machine_open or human_open)
    elif channel == "Web":
        click_flag = int(meaningful)
        download_flag = int(delivered and rng.random() < (0.18 + 0.45 * meaningful))
        landing_visit_flag = int(delivered)
    elif channel == "Peer program":
        registration_flag = int(delivered and rng.random() < (0.22 + 0.50 * meaningful))
        attendance_flag = int(meaningful)
        followup_requested_flag = int(meaningful and rng.random() < 0.30)
    elif channel == "Speaker program":
        registration_flag = int(delivered and rng.random() < (0.30 + 0.45 * meaningful))
        attendance_flag = int(meaningful)
        followup_requested_flag = int(meaningful and rng.random() < 0.28)
    elif channel == "Paid media":
        viewable_impression_flag = int(delivered and rng.random() < 0.74)
        click_flag = int(meaningful)
        landing_visit_flag = int(click_flag or (delivered and rng.random() < 0.05))
    elif channel == "Conference":
        registration_flag = int(delivered and rng.random() < 0.35)
        attendance_flag = int(meaningful)
        download_flag = int(meaningful and rng.random() < 0.25)
        followup_requested_flag = int(meaningful and rng.random() < 0.22)
    elif channel == "Direct mail":
        open_flag = int(delivered and rng.random() < 0.36)
        click_flag = int(meaningful)
        landing_visit_flag = int(click_flag)
        followup_requested_flag = int(meaningful and rng.random() < 0.12)
    elif channel == "Phone":
        if meaningful:
            field_outcome = str(rng.choice(["Positive", "Follow-up"], p=[0.45, 0.55]))
            followup_requested_flag = 1
        else:
            field_outcome = str(rng.choice(["No reach", "Neutral"], p=[0.58, 0.42]))
        transfer_flag = int(meaningful and rng.random() < 0.18)
    elif channel == "Account support":
        followup_requested_flag = int(delivered)
        resolution_flag = int(meaningful)
        field_outcome = "Resolved" if meaningful else "Open"
    else:
        raise ValueError(f"Unknown channel: {channel}")

    if forced_outcome is not None:
        delivered = True
        machine_open = False
        human_open = False
        open_flag = 0
        click_flag = 0
        attendance_flag = 0
        field_outcome = ""
        viewable_impression_flag = 0
        landing_visit_flag = 0
        registration_flag = 0
        download_flag = 0
        followup_requested_flag = 0
        transfer_flag = 0
        resolution_flag = 0
        if channel == "Field":
            field_outcome = forced_outcome
            followup_requested_flag = int(forced_outcome == "Follow-up")
        elif channel == "Email":
            if forced_outcome == "Clicked":
                open_flag = 1
                click_flag = 1
            elif forced_outcome == "Open only":
                open_flag = 1
        elif channel == "Web" and forced_outcome == "Qualified action":
            click_flag = 1
            landing_visit_flag = 1
        elif channel == "Peer program" and forced_outcome == "Attended":
            attendance_flag = 1
            registration_flag = 1
        elif channel == "Speaker program" and forced_outcome == "Attended":
            attendance_flag = 1
            registration_flag = 1
        elif channel == "Paid media" and forced_outcome == "Clicked":
            click_flag = 1
            landing_visit_flag = 1
            viewable_impression_flag = 1
        elif channel == "Conference" and forced_outcome == "Follow-up":
            attendance_flag = 1
            followup_requested_flag = 1
        elif channel == "Direct mail" and forced_outcome == "Qualified action":
            open_flag = 1
            click_flag = 1
            landing_visit_flag = 1
        elif channel == "Phone":
            field_outcome = forced_outcome
            followup_requested_flag = int(forced_outcome == "Follow-up")
        elif channel == "Account support" and forced_outcome == "Resolved":
            resolution_flag = 1
            field_outcome = "Resolved"

    meta = CHANNEL_METADATA[channel]
    return {
        "source_system": meta["source_system"],
        "event_type": meta["event_type"],
        "delivery_status": "Delivered" if delivered else "Failed",
        "sent_flag": sent_flag,
        "bounce_flag": bounce_flag,
        "open_flag": open_flag,
        "click_flag": click_flag,
        "attendance_flag": attendance_flag,
        "viewable_impression_flag": viewable_impression_flag,
        "landing_visit_flag": landing_visit_flag,
        "registration_flag": registration_flag,
        "download_flag": download_flag,
        "followup_requested_flag": followup_requested_flag,
        "transfer_flag": transfer_flag,
        "resolution_flag": resolution_flag,
        "field_outcome": field_outcome,
        "machine_open_answer_key": machine_open,
        "human_open_answer_key": bool(channel == "Email" and human_open),
        "meaningful_response_answer_key": meaningful,
        "planted_response_probability": round(response_probability, 6),
    }


def _build_population(
    sources: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> pd.DataFrame:
    hcp = sources["hcp_features"].merge(
        sources["engagement_signals"][
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
    hcp = hcp.merge(
        sources["hcp_segments"][
            [
                "npi",
                "segment_name",
                "engagement_pattern",
            ]
        ],
        on="npi",
        how="left",
        validate="one_to_one",
    )
    hcp = hcp.merge(
        sources["account_targets"][["account_id", "account_action", "reason_code"]],
        on="account_id",
        how="left",
        validate="many_to_one",
    )
    hcp = hcp.merge(
        sources["account_actions"][
            ["account_id", "action", "access_flag", "adoption_flag"]
        ].rename(columns={"action": "competitive_action"}),
        on="account_id",
        how="left",
        validate="many_to_one",
    )
    hcp["segment_name"] = hcp["segment_name"].fillna("Not clustered")
    hcp["engagement_pattern"] = hcp["engagement_pattern"].fillna(
        "Standard channel review"
    )
    if hcp[
        [
            "evidence_need_score",
            "access_resource_score",
            "digital_response_rate",
            "field_response_rate",
        ]
    ].isna().any().any():
        raise ValueError("Every Chapter 8 HCP needs engagement evidence.")
    hcp["latent_topic_affinity"] = rng.beta(2.4, 2.0, len(hcp))
    hcp["fatigue_sensitivity"] = rng.beta(2.0, 4.0, len(hcp))
    opt_out_count = int(hcp["contact_permission_status"].eq("Opt-out").sum())
    if opt_out_count:
        offsets = rng.integers(
            0,
            (OPT_OUT_END - OPT_OUT_START).days + 1,
            size=opt_out_count,
        )
        hcp.loc[
            hcp["contact_permission_status"].eq("Opt-out"),
            "opt_out_date",
        ] = [OPT_OUT_START + pd.Timedelta(days=int(offset)) for offset in offsets]
    hcp["opt_out_date"] = pd.to_datetime(hcp["opt_out_date"])
    return hcp


def build_events(
    sources: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate longitudinal channel events and a separate answer key."""

    population = _build_population(sources, rng)
    month_starts = pd.date_range(DATA_START, DATA_END, freq="MS")
    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    event_number = 1

    for hcp in population.itertuples(index=False):
        permission = str(hcp.contact_permission_status)
        priority = hcp.account_action == "Increase priority"
        base_rate = (
            0.62
            + 0.28 * np.log1p(float(hcp.review_opportunity))
            + 0.34 * priority
            + 0.10 * (hcp.competitive_action == "Adoption review")
            + 0.20 * float(hcp.evidence_need_score)
            + 0.15 * float(hcp.access_resource_score)
        )
        recent_dates: list[pd.Timestamp] = []
        recent_responses: list[tuple[pd.Timestamp, str]] = []
        live_program_response_dates: list[pd.Timestamp] = []
        field_response_dates: list[pd.Timestamp] = []
        evidence_centered = float(hcp.evidence_need_score) - 0.5

        for month_start in month_starts:
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_key = f"{month_start:%Y-%m}"
            intensity = base_rate
            intensity += MONTHLY_LAUNCH_RAMP.get(month_key, 0.0)
            recent_responses = [
                (date, channel)
                for date, channel in recent_responses
                if 0 <= (month_start - date).days <= 90
            ]
            response_count = len(recent_responses)
            if priority:
                intensity += min(0.16 * response_count, 0.70)
            else:
                intensity += min(0.08 * response_count, 0.35)
            event_count = int(min(rng.poisson(intensity), 5))
            offsets = sorted(
                rng.integers(
                    0,
                    max((month_end - month_start).days + 1, 1),
                    size=event_count,
                ).tolist()
            )
            for offset in offsets:
                event_date = month_start + pd.Timedelta(days=int(offset))
                if event_date > DATA_END:
                    continue
                if (
                    permission == "Opt-out"
                    and pd.notna(hcp.opt_out_date)
                    and event_date > hcp.opt_out_date
                ):
                    continue

                recent_dates = [
                    date
                    for date in recent_dates
                    if 0 <= (event_date - date).days <= 30
                ]
                response_memory = {channel: 0.0 for channel in CHANNELS}
                for response_date, response_channel in recent_responses:
                    age = max((event_date - response_date).days, 0)
                    response_memory[response_channel] += float(np.exp(-age / 45))
                channel = _select_channel(
                    rng,
                    priority,
                    float(hcp.digital_response_rate),
                    float(hcp.field_response_rate),
                    response_memory,
                )
                topic = _select_topic(
                    rng,
                    float(hcp.access_resource_score),
                    float(hcp.evidence_need_score),
                )
                affinity = (
                    float(hcp.field_response_rate)
                    if channel == "Field"
                    else float(hcp.digital_response_rate)
                )
                if channel in {"Peer program", "Speaker program", "Conference"}:
                    affinity = 0.55 * affinity + 0.45 * float(hcp.evidence_need_score)
                if channel == "Paid media":
                    affinity *= 0.55
                if channel == "Account support":
                    affinity = 0.45 * affinity + 0.55 * float(hcp.access_resource_score)
                if channel in {"Direct mail", "Phone"}:
                    affinity = 0.60 * affinity + 0.40 * float(hcp.evidence_need_score)
                topic_match = float(hcp.latent_topic_affinity)
                fatigue = max(len(recent_dates) - 2, 0) * float(hcp.fatigue_sensitivity)
                base_logit = (
                    RESPONSE_BASE_LOGIT
                    + RESPONSE_AFFINITY_WEIGHT * affinity
                    + RESPONSE_EVIDENCE_WEIGHT * float(hcp.evidence_need_score)
                    + RESPONSE_ACCESS_WEIGHT * float(hcp.access_resource_score)
                    + RESPONSE_TOPIC_WEIGHT * topic_match
                    + RESPONSE_MEMORY_WEIGHT * response_memory.get(channel, 0.0)
                    - RESPONSE_FATIGUE_WEIGHT * fatigue
                )
                if channel == "Speaker program":
                    base_logit += 0.15 * float(hcp.evidence_need_score)
                if channel == "Account support":
                    base_logit += 0.35 * float(hcp.access_resource_score)
                if channel == "Paid media":
                    base_logit -= 0.35
                if channel == "Direct mail":
                    base_logit -= 0.20
                if priority:
                    base_logit += 0.28

                # Planted live-program treatment effect (heterogeneous by
                # evidence need). The counterfactual base_logit excludes it.
                live_program_response_dates = [
                    date
                    for date in live_program_response_dates
                    if 0 <= (event_date - date).days <= LIVE_PROGRAM_EFFECT_WINDOW
                ]
                live_active = bool(live_program_response_dates)
                live_uplift_logit = (
                    LIVE_PROGRAM_UPLIFT_BASE_LOGIT
                    + LIVE_PROGRAM_UPLIFT_EVIDENCE_GAIN * evidence_centered
                    if live_active
                    else 0.0
                )

                # Planted field-then-digital sequence effect.
                field_response_dates = [
                    date
                    for date in field_response_dates
                    if 0 <= (event_date - date).days <= FIELD_THEN_DIGITAL_WINDOW
                ]
                field_seq_active = bool(
                    field_response_dates and channel in {"Email", "Web"}
                )
                field_seq_logit = (
                    FIELD_THEN_DIGITAL_BASE_LOGIT
                    + FIELD_THEN_DIGITAL_EVIDENCE_GAIN * evidence_centered
                    if field_seq_active
                    else 0.0
                )

                logit = base_logit + live_uplift_logit + field_seq_logit
                response_probability = _logistic(logit)
                counterfactual_probability = _logistic(base_logit)
                fields = _event_fields(rng, channel, response_probability)

                event_id = f"OME{event_number:07d}"
                rows.append(
                    {
                        "event_id": event_id,
                        "event_date": event_date,
                        "npi": str(hcp.npi),
                        "account_id": hcp.account_id,
                        "territory": hcp.territory,
                        "channel": channel,
                        "content_topic": topic,
                        "permission_status_at_event": permission,
                        "campaign_id": f"CMP-{event_date:%Y%m}",
                        "sequence_id": f"{hcp.npi}-{event_date:%Y%m}",
                        **{
                            key: value
                            for key, value in fields.items()
                            if not key.endswith("answer_key")
                            and key != "planted_response_probability"
                        },
                    }
                )
                truth_rows.append(
                    {
                        "event_id": event_id,
                        "machine_open_answer_key": fields["machine_open_answer_key"],
                        "human_open_answer_key": fields["human_open_answer_key"],
                        "meaningful_response_answer_key": fields[
                            "meaningful_response_answer_key"
                        ],
                        "planted_response_probability": fields[
                            "planted_response_probability"
                        ],
                        "planted_counterfactual_probability": round(
                            counterfactual_probability, 6
                        ),
                        "planted_live_active": live_active,
                        "planted_live_uplift_logit": round(live_uplift_logit, 6),
                        "planted_field_sequence_active": field_seq_active,
                        "planted_field_sequence_logit": round(field_seq_logit, 6),
                        "evidence_need_score": round(
                            float(hcp.evidence_need_score), 6
                        ),
                        "latent_topic_affinity": round(
                            float(hcp.latent_topic_affinity), 6
                        ),
                        "fatigue_sensitivity": round(
                            float(hcp.fatigue_sensitivity), 6
                        ),
                    }
                )
                event_number += 1
                recent_dates.append(event_date)
                if fields["meaningful_response_answer_key"]:
                    recent_responses.append((event_date, channel))
                    if channel in {"Peer program", "Speaker program", "Conference"}:
                        live_program_response_dates.append(event_date)
                    if channel == "Field":
                        field_response_dates.append(event_date)

    events = pd.DataFrame(rows)
    truth = pd.DataFrame(truth_rows)
    events = _apply_trace_overrides(events, population, rng, event_number)
    truth = truth.loc[truth["event_id"].isin(events["event_id"])].copy()
    missing_truth = events.loc[
        ~events["event_id"].isin(truth["event_id"]), "event_id"
    ]
    if not missing_truth.empty:
        override_truth = events.loc[events["event_id"].isin(missing_truth)].copy()
        override_meaningful = (
            (
                override_truth["channel"].isin(["Field", "Phone"])
                & override_truth["field_outcome"].isin(["Positive", "Follow-up"])
            )
            | (
                override_truth["channel"].isin(["Email", "Web", "Paid media", "Direct mail"])
                & override_truth["click_flag"].eq(1)
            )
            | (
                override_truth["channel"].isin(["Peer program", "Speaker program", "Conference"])
                & override_truth["attendance_flag"].eq(1)
            )
            | (
                override_truth["channel"].eq("Account support")
                & override_truth["resolution_flag"].eq(1)
            )
        )
        truth = pd.concat(
            [
                truth,
                pd.DataFrame(
                    {
                        "event_id": override_truth["event_id"],
                        "machine_open_answer_key": False,
                        "human_open_answer_key": (
                            override_truth["channel"].eq("Email")
                            & override_truth["open_flag"].eq(1)
                            & override_truth["click_flag"].eq(0)
                        ),
                        "meaningful_response_answer_key": override_meaningful,
                        "planted_response_probability": np.nan,
                        "planted_counterfactual_probability": np.nan,
                        "planted_live_active": False,
                        "planted_live_uplift_logit": np.nan,
                        "planted_field_sequence_active": False,
                        "planted_field_sequence_logit": np.nan,
                        "evidence_need_score": np.nan,
                        "latent_topic_affinity": np.nan,
                        "fatigue_sensitivity": np.nan,
                    }
                ),
            ],
            ignore_index=True,
        )
    return (
        events.sort_values(["event_date", "event_id"]).reset_index(drop=True),
        truth.sort_values("event_id").reset_index(drop=True),
    )


def _apply_trace_overrides(
    events: pd.DataFrame,
    population: pd.DataFrame,
    rng: np.random.Generator,
    event_number: int,
) -> pd.DataFrame:
    """Replace recent trace events with deterministic teaching histories."""

    result = events.copy()
    override_rows: list[dict[str, object]] = []
    for npi, specifications in TRACE_OVERRIDES.items():
        hcp = population.loc[population["npi"].eq(npi)]
        if hcp.empty:
            continue
        row = hcp.iloc[0]
        cutoff = pd.Timestamp("2024-11-01")
        result = result.loc[
            ~(result["npi"].eq(npi) & result["event_date"].ge(cutoff))
        ].copy()
        for date_text, channel, topic, outcome in specifications:
            fields = _event_fields(
                rng,
                channel,
                response_probability=TRACE_OVERRIDE_RESPONSE_PROBABILITY,
                forced_outcome=outcome,
            )
            override_rows.append(
                {
                    "event_id": f"OME{event_number:07d}",
                    "event_date": pd.Timestamp(date_text),
                    "npi": npi,
                    "account_id": row["account_id"],
                    "territory": row["territory"],
                    "channel": channel,
                    "content_topic": topic,
                    "permission_status_at_event": row[
                        "contact_permission_status"
                    ],
                    "campaign_id": f"CMP-{pd.Timestamp(date_text):%Y%m}",
                    "sequence_id": f"{npi}-{pd.Timestamp(date_text):%Y%m}",
                    **{
                        key: value
                        for key, value in fields.items()
                        if not key.endswith("answer_key")
                        and key != "planted_response_probability"
                    },
                }
            )
            event_number += 1
    return pd.concat([result, pd.DataFrame(override_rows)], ignore_index=True)


def generate(repo_root: Path, output_dir: Path) -> dict[str, int]:
    """Write Chapter 8-only synthetic data and a provenance manifest."""

    rng = np.random.default_rng(SEED)
    sources = load_sources(repo_root)
    events, truth = build_events(sources, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "engagement_events.csv"
    truth_path = output_dir / "engagement_truth.csv"
    events.to_csv(events_path, index=False)
    truth.to_csv(truth_path, index=False)

    paths = source_paths(repo_root)
    manifest = {
        "description": "Synthetic HCP omnichannel data for Chapter 8",
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "analysis_date": ANALYSIS_DATE.date().isoformat(),
        "date_range": {
            "start": events["event_date"].min().date().isoformat(),
            "end": events["event_date"].max().date().isoformat(),
        },
        "row_counts": {
            "engagement_events.csv": len(events),
            "engagement_truth.csv": len(truth),
        },
        "planted_effects": {
            "live_program_uplift": {
                "window_days": LIVE_PROGRAM_EFFECT_WINDOW,
                "base_logit": LIVE_PROGRAM_UPLIFT_BASE_LOGIT,
                "evidence_gain": LIVE_PROGRAM_UPLIFT_EVIDENCE_GAIN,
                "moderator": "evidence_need_score (centered at 0.5)",
            },
            "field_then_digital_sequence": {
                "window_days": FIELD_THEN_DIGITAL_WINDOW,
                "base_logit": FIELD_THEN_DIGITAL_BASE_LOGIT,
                "evidence_gain": FIELD_THEN_DIGITAL_EVIDENCE_GAIN,
                "moderator": "evidence_need_score (centered at 0.5)",
            },
        },
        "answer_key_columns": [
            "machine_open_answer_key",
            "human_open_answer_key",
            "meaningful_response_answer_key",
            "planted_response_probability",
            "planted_counterfactual_probability",
            "planted_live_active",
            "planted_live_uplift_logit",
            "planted_field_sequence_active",
            "planted_field_sequence_logit",
        ],
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "engagement_events.csv": _sha256(events_path),
            "engagement_truth.csv": _sha256(truth_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest["row_counts"]
