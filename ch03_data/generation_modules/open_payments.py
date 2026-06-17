"""Synthetic CMS Open Payments-style records for Chapter 3."""

from __future__ import annotations

import random

from .entities import (
    EntityBundle,
    _PAYMENT_CATEGORIES,
    _PAYMENT_RANGES,
    _PAYMENT_WEIGHTS_KOL,
    _PAYMENT_WEIGHTS_STD,
    _PHARMA_COMPANIES,
)


def generate_open_payments(
    rng: random.Random,
    bundle: EntityBundle,
) -> list[dict]:
    """Simulate physician payment records with a small KOL subgroup."""

    records: list[dict] = []
    n_kol = max(1, int(len(bundle.providers) * 0.15))
    kol_npis = {provider["npi"] for provider in rng.sample(bundle.providers, n_kol)}

    for provider in bundle.providers:
        npi = provider["npi"]
        is_kol = npi in kol_npis
        if is_kol:
            n_payments = rng.randint(3, 8)
        elif rng.random() < 0.25:
            n_payments = rng.randint(1, 3)
        else:
            continue

        weights = _PAYMENT_WEIGHTS_KOL if is_kol else _PAYMENT_WEIGHTS_STD
        for _ in range(n_payments):
            category = rng.choices(_PAYMENT_CATEGORIES, weights=weights, k=1)[0]
            low, high = _PAYMENT_RANGES[category]
            records.append(
                {
                    "npi": npi,
                    "company_name": rng.choice(_PHARMA_COMPANIES),
                    "payment_year": 2024,
                    "payment_category": category,
                    "payment_amount": round(rng.uniform(low, high), 2),
                    "is_kol": is_kol,
                }
            )
    return records
