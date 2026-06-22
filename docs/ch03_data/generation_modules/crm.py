"""CRM, territory alignment, and digital engagement generators for ch03 synthetic data."""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import date

from .entities import (
    EntityBundle,
    CHANNELS,
    DIGITAL_CHANNELS,
    MESSAGE_TOPICS,
    OUTCOMES,
    TOPICS,
    rand_date,
)

CRM_FIELDS = [
    "interaction_id", "interaction_date", "rep_id", "hcp_npi", "account_id",
    "territory", "product_name", "channel", "detail_topic", "call_outcome",
    "duration_min", "sample_qty", "consent_status",
]

DIGITAL_FIELDS = [
    "digital_event_id", "event_date", "hcp_npi", "account_id", "territory",
    "product_name", "channel", "content_topic", "open_flag", "click_flag",
    "webinar_attended",
]


def generate_territory_alignment(bundle: EntityBundle) -> list[dict]:
    """Build territory alignment table: rep -> territory -> account coverage metrics."""
    territory_rep: dict[str, str] = {}
    rep_id = 1
    for account in sorted(bundle.accounts, key=lambda a: a["territory"]):
        t = account["territory"]
        if t not in territory_rep:
            territory_rep[t] = f"REP{rep_id:03d}"
            rep_id += 1

    territory_accounts: dict[str, int] = defaultdict(int)
    for account in bundle.accounts:
        territory_accounts[account["territory"]] += 1

    territory_providers: dict[str, int] = defaultdict(int)
    for provider in bundle.providers:
        acc = next((a for a in bundle.accounts if a["account_id"] == provider["account_id"]), None)
        if acc:
            territory_providers[acc["territory"]] += 1

    rows = []
    for territory, rep in sorted(territory_rep.items()):
        n_accounts = territory_accounts[territory]
        n_hcps = territory_providers[territory]
        rows.append({
            "territory": territory,
            "rep_id": rep,
            "region": next(
                (a["region"] for a in bundle.accounts if a["territory"] == territory),
                "Unknown",
            ),
            "n_accounts_aligned": n_accounts,
            "n_hcps_aligned": n_hcps,
            "target_calls_per_month": max(20, round(n_hcps * 0.4)),
            "target_coverage_pct": 0.80,
        })
    return rows


def generate_crm(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    account_providers: dict[str, list[dict]] = defaultdict(list)
    for p in bundle.providers:
        account_providers[p["account_id"]].append(p)

    territory_rep: dict[str, str] = {}
    rep_seq = 1
    for account in sorted(bundle.accounts, key=lambda a: a["territory"]):
        t = account["territory"]
        if t not in territory_rep:
            territory_rep[t] = f"REP{rep_seq:03d}"
            rep_seq += 1

    rows: list[dict] = []
    interaction_id = 1
    for account in bundle.accounts:
        population = account_providers[account["account_id"]]
        territory = account["territory"]
        rep_id = territory_rep.get(territory, f"REP{rng.randint(1, 18):03d}")
        for provider in population:
            for _ in range(rng.randint(2, 8)):
                interaction_date = rand_date(rng, date(2024, 1, 1), date(2025, 1, 31))
                channel = rng.choice(CHANNELS)
                rows.append(
                    {
                        "interaction_id": f"CRM{interaction_id:07d}",
                        "interaction_date": interaction_date.isoformat(),
                        "rep_id": rep_id,
                        "hcp_npi": provider["npi"],
                        "account_id": account["account_id"],
                        "territory": territory,
                        "product_name": rng.choices(
                            [product["product_name"] for product in bundle.products],
                            weights=[4, 3, 3],
                            k=1,
                        )[0],
                        "channel": channel,
                        "detail_topic": rng.choice(MESSAGE_TOPICS),
                        "call_outcome": rng.choice(OUTCOMES),
                        "duration_min": rng.choice([5, 10, 15, 20, 30, 45]),
                        "sample_qty": rng.choice([0, 0, 0, 1, 2]) if channel == "Sample Drop" else 0,
                        "consent_status": rng.choice(["Allowed", "Allowed", "Opt-out"]),
                    }
                )
                interaction_id += 1
    return rows


def generate_digital(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    account_providers: dict[str, list[dict]] = defaultdict(list)
    for p in bundle.providers:
        account_providers[p["account_id"]].append(p)

    rows: list[dict] = []
    event_id = 1
    for account in bundle.accounts:
        providers = account_providers[account["account_id"]]
        if not providers:
            continue
        for provider in providers[: rng.randint(1, len(providers))]:
            for _ in range(rng.randint(0, 5)):
                event_date = rand_date(rng, date(2024, 1, 1), date(2025, 1, 31))
                open_flag = rng.choice([0, 1, 1, 1])
                click_flag = 1 if open_flag and rng.random() > 0.5 else 0
                rows.append(
                    {
                        "digital_event_id": f"DIG{event_id:07d}",
                        "event_date": event_date.isoformat(),
                        "hcp_npi": provider["npi"],
                        "account_id": account["account_id"],
                        "territory": account["territory"],
                        "product_name": rng.choices(
                            [product["product_name"] for product in bundle.products],
                            weights=[4, 3, 3],
                            k=1,
                        )[0],
                        "channel": rng.choice(DIGITAL_CHANNELS),
                        "content_topic": rng.choice(TOPICS),
                        "open_flag": open_flag,
                        "click_flag": click_flag,
                        "webinar_attended": 1 if click_flag and rng.random() > 0.75 else 0,
                    }
                )
                event_id += 1
    return rows
