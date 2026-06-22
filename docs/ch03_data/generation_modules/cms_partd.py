"""CMS Medicare Part D prescriber summary generator for ch03 synthetic data."""
from __future__ import annotations

import random

from .entities import EntityBundle, _PART_D_DRUG_NAMES

PART_D_TYPE_BY_SPECIALTY = {
    "Primary Care": "Family Practice",
    "Oncology": "Oncology/Hematology",
    "Cardiology": "Cardiology",
    "Endocrinology": "Endocrinology",
    "Rheumatology": "Rheumatology",
    "Pulmonology": "Pulmonology",
}


def generate_cms_partd(rng: random.Random, bundle: EntityBundle) -> list[dict]:
    """Simulate CMS Medicare Part D Prescribers public-use file for calendar year 2024.

    Each record represents a provider x drug combination with Medicare Part D
    claim counts, days supply, and total drug costs -- useful for specialty-level
    prescribing volume and competitive share benchmarking.
    """
    records: list[dict] = []
    for provider in bundle.providers:
        specialty = PART_D_TYPE_BY_SPECIALTY.get(provider["specialty"], "Internal Medicine")
        # Not all providers prescribe all products; Roventra less common = realistic
        products_written = rng.choices(
            list(_PART_D_DRUG_NAMES.keys()),
            weights=[2, 4, 4],
            k=rng.randint(1, 3),
        )
        seen: set[str] = set()
        for drug in products_written:
            if drug in seen:
                continue
            seen.add(drug)
            tot_claims = rng.randint(11, 300)
            bene_count = rng.randint(11, tot_claims)
            days_supply = tot_claims * rng.randint(28, 90)
            total_cost = round(tot_claims * rng.uniform(80, 1200), 2)
            records.append({
                "prscrbr_npi": provider["npi"],
                "prscrbr_last_org_name": f"Provider_{provider['hcp_id']}",
                "prscrbr_first_name": "Synthetic",
                "prscrbr_city": provider.get("state", "NY") + " City",
                "prscrbr_state_abrvtn": provider.get("state", "NY"),
                "prscrbr_type": specialty,
                "drug_name": drug,
                "generic_name": _PART_D_DRUG_NAMES[drug]["generic"],
                "drug_class": _PART_D_DRUG_NAMES[drug]["class"],
                "tot_clms": tot_claims,
                "tot_benes": bene_count,
                "tot_day_suply": days_supply,
                "tot_drug_cst": total_cost,
                "year": 2024,
            })
    return records
