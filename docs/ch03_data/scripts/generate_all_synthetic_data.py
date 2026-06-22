#!/usr/bin/env python3
"""Main entry point for generating all ch03 synthetic commercial data.

Usage:
    python generate_all_synthetic_data.py
    python generate_all_synthetic_data.py --patients 250
    python generate_all_synthetic_data.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, datetime
from pathlib import Path

# Allow importing the sibling generation_modules package from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation_modules.claims import (
    LAUNCH_DIAGNOSIS_DOCUMENTATION_RATE,
    LAUNCH_DIAGNOSIS_FALSE_POSITIVE_RATE,
    MEDICAL_HEADER_FIELDS,
    PHARMACY_CLAIM_FIELDS,
    SERVICE_LINE_FIELDS,
    filter_by_snapshot_date,
    generate_medical_headers,
    generate_pharmacy_claims,
    generate_service_lines,
    strip_internal_fields,
)
from generation_modules.cms_partd import generate_cms_partd
from generation_modules.crm import (
    CRM_FIELDS,
    DIGITAL_FIELDS,
    generate_crm,
    generate_digital,
    generate_territory_alignment,
)
from generation_modules.lab import LAB_RESULT_FIELDS, generate_lab_results
from generation_modules.entities import (
    DEFAULT_OUTPUT,
    apply_canonical_overrides,
    build_entities,
    ensure_dir,
    make_reference_tables,
    write_csv,
    write_dataset_folder,
    write_manifest,
)
from generation_modules.formulary import (
    generate_formulary_bundle,
)
from generation_modules.open_payments import generate_open_payments
from generation_modules.scenario import build_launch_treatment_attempts
from generation_modules.specialty_pharmacy import SP_EVENT_FIELDS, generate_specialty_pharmacy


SCENARIO_SETTINGS = {
    "default_seed": 20260609,
    "claims_per_patient_range": [1, 4],
    "launch_patient_treatment_probability": 0.68,
    "nonlaunch_patient_treatment_probability_range": [0, 4],
    "launch_diagnosis_documentation_rate": LAUNCH_DIAGNOSIS_DOCUMENTATION_RATE,
    "launch_diagnosis_false_positive_rate": LAUNCH_DIAGNOSIS_FALSE_POSITIVE_RATE,
    "missing_ndc_mapping_rate": 0.05,
    "formulary_quarterly_change_rate": 0.20,
    "specialty_case_probability_given_launch_attempt": 0.82,
}

# Early snapshot cutoff: five days after month close for December 2024.
# Claims with a lag that puts their receipt date after this cutoff are absent
# from the early file; they arrive later, creating the maturity gap.
EARLY_SNAPSHOT_CUTOFF = date(2025, 1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pharmaceutical data for Chapter 3."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--patients",
        type=int,
        default=20_000,
        help="Number of synthetic patients to generate (default: 20000). ",
    )
    parser.add_argument(
        "--diagnosis-dict",
        type=Path,
        help="Optional diagnosis-code CSV. Uses the built-in synthetic dictionary when omitted.",
    )
    parser.add_argument(
        "--drug-dict",
        type=Path,
        help="Optional drug-code CSV. Uses the built-in synthetic dictionary when omitted.",
    )
    args = parser.parse_args()
    if args.patients <= 0:
        parser.error("--patients must be a positive integer.")
    return args


def _table_contract(rows: list[dict], fields: list[str], event_date_fields: list[str] | None = None) -> dict:
    event_date_fields = event_date_fields or []
    contract: dict[str, object] = {
        "row_count": len(rows),
        "columns": fields,
    }
    if rows and event_date_fields:
        date_ranges: dict[str, dict[str, str]] = {}
        for field in event_date_fields:
            values = [row[field] for row in rows if row.get(field)]
            if values:
                date_ranges[field] = {"min": min(values), "max": max(values)}
        if date_ranges:
            contract["date_ranges"] = date_ranges
    return contract


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    bundle = build_entities(
        rng, args.diagnosis_dict, args.drug_dict, n_patients=args.patients
    )
    apply_canonical_overrides(bundle)

    ensure_dir(args.output_dir)
    make_reference_tables(args.output_dir, bundle)

    print(
        f"Generating data for {args.patients:,} patients, "
        f"{len(bundle.accounts):,} accounts, {len(bundle.providers):,} providers..."
    )

    access, formulary, formulary_history, quarterly_states = generate_formulary_bundle(rng, bundle)

    # launch_attempts: one scenario per T2D patient assigned a Roventra treatment timeline;
    # controls pharmacy fill sequence, SP referral, and lab timing for the teaching case
    launch_attempts = build_launch_treatment_attempts(rng, bundle, quarterly_states)

    # Medical claims: generate all encounters with internal lag metadata
    all_headers = generate_medical_headers(rng, bundle)
    # Early snapshot: only encounters that arrived within 5 days of month close
    early_headers = filter_by_snapshot_date(all_headers, EARLY_SNAPSHOT_CUTOFF)
    # Mature snapshot: all encounters (no date filter)
    mature_headers = all_headers
    # Strip internal fields before writing
    early_rows = strip_internal_fields(early_headers)
    mature_rows = strip_internal_fields(mature_headers)
    service_lines = generate_service_lines(all_headers, rng)

    pharmacy_claims = generate_pharmacy_claims(rng, bundle, launch_attempts)
    lab_results = generate_lab_results(rng, bundle, launch_attempts)
    specialty = generate_specialty_pharmacy(rng, bundle, launch_attempts)
    crm = generate_crm(rng, bundle)
    territory_alignment = generate_territory_alignment(bundle)
    digital = generate_digital(rng, bundle)
    open_payments = generate_open_payments(rng, bundle)
    cms_partd = generate_cms_partd(rng, bundle)

    print(f"  Medical encounters: {len(mature_rows):,} mature / {len(early_rows):,} early snapshot")
    print(f"  Service lines: {len(service_lines):,}")
    print(f"  Pharmacy claims: {len(pharmacy_claims):,}")
    print(f"  Lab results: {len(lab_results):,}")
    print(f"  Patients with Roventra treatment timeline: {len(launch_attempts):,}")
    print(f"  Formulary change events: {len(formulary_history):,}")
    print(f"  Specialty pharmacy hub referrals: {len(specialty):,}")
    print(f"  CRM interactions: {len(crm):,}")
    print(f"  Territory alignment records: {len(territory_alignment):,}")
    print(f"  Digital events: {len(digital):,}")
    print(f"  Open Payments records: {len(open_payments):,}")
    print(f"  CMS Part D records: {len(cms_partd):,}")

    # Write datasets
    med_dir = args.output_dir / "claims_medical"
    ensure_dir(med_dir)
    write_csv(med_dir / "medical_claims.csv", early_rows, MEDICAL_HEADER_FIELDS)
    write_csv(med_dir / "medical_claims_mature.csv", mature_rows, MEDICAL_HEADER_FIELDS)
    write_csv(med_dir / "service_lines.csv", service_lines, SERVICE_LINE_FIELDS)

    write_dataset_folder(
        args.output_dir,
        "claims_pharmacy",
        "Pharmacy claims",
        [("pharmacy_claims.csv", pharmacy_claims, PHARMACY_CLAIM_FIELDS)],
    )
    write_dataset_folder(
        args.output_dir,
        "claims_lab",
        "Lab results (from longitudinal claims vendor via specialty lab partnerships)",
        [("lab_results.csv", lab_results, LAB_RESULT_FIELDS)],
    )
    write_dataset_folder(
        args.output_dir,
        "specialty_pharmacy",
        "Specialty pharmacy hub events",
        [("sp_events.csv", specialty, SP_EVENT_FIELDS)],
    )
    write_dataset_folder(
        args.output_dir,
        "crm_veeva",
        "Veeva CRM interactions and territory alignment",
        [
            ("crm_interactions.csv", crm, CRM_FIELDS),
            ("territory_alignment.csv", territory_alignment, list(territory_alignment[0].keys()) if territory_alignment else []),
        ],
    )
    write_dataset_folder(
        args.output_dir,
        "digital_engagement",
        "Digital engagement events",
        [("digital_engagement.csv", digital, DIGITAL_FIELDS)],
    )
    write_dataset_folder(
        args.output_dir,
        "market_access",
        "Market access rules",
        [("market_access_rules.csv", access, list(access[0].keys()) if access else [])],
    )
    write_dataset_folder(
        args.output_dir,
        "formulary",
        "Formulary rules",
        [
            ("formulary_status.csv", formulary, list(formulary[0].keys()) if formulary else []),
            ("formulary_history.csv", formulary_history, list(formulary_history[0].keys()) if formulary_history else []),
        ],
    )
    write_dataset_folder(
        args.output_dir,
        "open_payments",
        "CMS Open Payments (synthetic)",
        [("open_payments.csv", open_payments, list(open_payments[0].keys()) if open_payments else [])],
    )
    write_dataset_folder(
        args.output_dir,
        "cms_part_d",
        "CMS Medicare Part D Prescribers (synthetic proxy)",
        [("prescriber_summary.csv", cms_partd, list(cms_partd[0].keys()) if cms_partd else [])],
    )

    ref_ndc_rows = [{"ndc": row["NDC"], "brand_generic": row["BRAND_GENERIC"], "drug_name": row["CUI_L1_NAME"], "ingredient": row["INGREDIENT_NAME_ARRAY"]} for row in bundle.ndc_codes]
    ref_pat_rows = [
        {
            "patient_id": p["patient_id"],
            "state": p["state"],
            "region": p["region"],
            "age_band": p["age_band"],
            "sex": p["sex"],
            "true_launch_condition": p["condition_bucket"] == "Launch condition",
        }
        for p in bundle.patients
    ]

    table_contracts = {
        "reference/patients.csv": _table_contract(ref_pat_rows, ["patient_id", "state", "region", "age_band", "sex", "true_launch_condition"]),
        "reference/patient_enrollments.csv": _table_contract(
            [{}],  # placeholder; actual rows built by make_reference_tables
            ["patient_id", "eligibility_start_date", "eligibility_end_date", "payer_id", "payer_type", "has_medical_coverage", "has_pharmacy_coverage", "product_type"],
            ["eligibility_start_date", "eligibility_end_date"],
        ),
        "reference/providers.csv": _table_contract(bundle.providers, ["npi", "specialty_1", "specialty_2", "provider_state", "provider_type", "credential", "primary_facility_npi"]),
        "reference/hcp_targets.csv": _table_contract(bundle.providers, ["npi", "account_id", "territory", "state", "region", "specialty_1"]),
        "reference/accounts.csv": _table_contract(bundle.accounts, list(bundle.accounts[0].keys()) if bundle.accounts else []),
        "reference/payers.csv": _table_contract(bundle.payers, list(bundle.payers[0].keys()) if bundle.payers else []),
        "reference/products.csv": _table_contract(bundle.products, list(bundle.products[0].keys()) if bundle.products else []),
        "reference/diagnosis_codes.csv": _table_contract(bundle.diagnosis_codes, list(bundle.diagnosis_codes[0].keys()) if bundle.diagnosis_codes else []),
        "reference/ndc_codes.csv": _table_contract(ref_ndc_rows, ["ndc", "brand_generic", "drug_name", "ingredient"]),
        "claims_medical/medical_claims.csv": _table_contract(early_rows, MEDICAL_HEADER_FIELDS, ["claim_date"]),
        "claims_medical/medical_claims_mature.csv": _table_contract(mature_rows, MEDICAL_HEADER_FIELDS, ["claim_date"]),
        "claims_medical/service_lines.csv": _table_contract(service_lines, SERVICE_LINE_FIELDS, ["service_from"]),
        "claims_pharmacy/pharmacy_claims.csv": _table_contract(pharmacy_claims, PHARMACY_CLAIM_FIELDS, ["date_of_service"]),
        "claims_lab/lab_results.csv": _table_contract(lab_results, LAB_RESULT_FIELDS, ["service_date"]),
        "specialty_pharmacy/sp_events.csv": _table_contract(specialty, SP_EVENT_FIELDS, ["referral_date", "status_date"]),
        "crm_veeva/crm_interactions.csv": _table_contract(crm, CRM_FIELDS, ["interaction_date"]),
        "crm_veeva/territory_alignment.csv": _table_contract(territory_alignment, list(territory_alignment[0].keys()) if territory_alignment else []),
        "digital_engagement/digital_engagement.csv": _table_contract(digital, DIGITAL_FIELDS, ["event_date"]),
        "market_access/market_access_rules.csv": _table_contract(access, list(access[0].keys()) if access else [], ["effective_start", "effective_end"]),
        "formulary/formulary_status.csv": _table_contract(formulary, list(formulary[0].keys()) if formulary else [], ["effective_start", "effective_end"]),
        "formulary/formulary_history.csv": _table_contract(formulary_history, list(formulary_history[0].keys()) if formulary_history else [], ["effective_date"]),
        "open_payments/open_payments.csv": _table_contract(open_payments, list(open_payments[0].keys()) if open_payments else []),
        "cms_part_d/prescriber_summary.csv": _table_contract(cms_partd, list(cms_partd[0].keys()) if cms_partd else []),
    }

    write_manifest(
        args.output_dir,
        bundle,
        run_config={
            "seed": args.seed,
            "patients_requested": args.patients,
            "output_dir": str(args.output_dir),
            "early_snapshot_cutoff": EARLY_SNAPSHOT_CUTOFF.isoformat(),
            "diagnosis_dict": str(args.diagnosis_dict) if args.diagnosis_dict else None,
            "drug_dict": str(args.drug_dict) if args.drug_dict else None,
            "invoked_at_local": datetime.now().isoformat(timespec="seconds"),
        },
        scenario_settings=SCENARIO_SETTINGS,
        table_contracts=table_contracts,
        diagnosis_path=args.diagnosis_dict,
        drug_path=args.drug_dict,
    )
    print(f"Done. Data written to {args.output_dir}")


if __name__ == "__main__":
    main()
