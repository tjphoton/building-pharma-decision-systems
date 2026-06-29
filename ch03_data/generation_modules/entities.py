"""Entity definitions, constants, lookup helpers, and utility functions for ch03 data generation."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output_data" / "generated_data"
MANIFEST_SCHEMA_VERSION = "2026-06-15"

REGIONS = ["Northeast", "South", "Midwest", "West"]
STATES = [
    ("New York", "NY", "Northeast"),
    ("New Jersey", "NJ", "Northeast"),
    ("Pennsylvania", "PA", "Northeast"),
    ("Florida", "FL", "South"),
    ("Georgia", "GA", "South"),
    ("Texas", "TX", "South"),
    ("Illinois", "IL", "Midwest"),
    ("Michigan", "MI", "Midwest"),
    ("Ohio", "OH", "Midwest"),
    ("California", "CA", "West"),
    ("Arizona", "AZ", "West"),
    ("Washington", "WA", "West"),
]
SPECIALTIES = [
    "Oncology",
    "Cardiology",
    "Endocrinology",
    "Rheumatology",
    "Pulmonology",
    "Primary Care",
]

# Secondary specialty by primary (mirrors Komodo SPECIALTY2 convention).
# ~28% of eligible subspecialists carry a secondary in real data.
SECONDARY_BY_SPECIALTY: dict[str, str] = {
    "Cardiology": "Internal Medicine",
    "Rheumatology": "Internal Medicine",
    "Endocrinology": "Internal Medicine",
    "Pulmonology": "Internal Medicine",
}

# Default credential by specialty for the vendor provider directory.
CREDENTIAL_BY_SPECIALTY = {
    "Oncology": "MD",
    "Cardiology": "MD",
    "Endocrinology": "MD",
    "Rheumatology": "MD",
    "Pulmonology": "MD",
    "Primary Care": "MD",
}

CONDITION_BUCKETS = [
    ("Launch condition", 6),
    ("Oncology", 2),
    ("Cardiology", 2),
    ("Rheumatology", 1),
    ("Other", 2),
]
ACCOUNT_TYPES = ["Health System", "Community Practice", "Academic Center", "Clinic"]
PAYER_TYPES = ["Commercial", "Medicare Advantage", "Medicare Part D", "Medicaid"]
CHANNELS = ["Face to Face", "Remote", "Phone", "Sample Drop"]
DIGITAL_CHANNELS = ["Email", "Web Visit", "Webinar", "Paid Media", "Content Hub"]
TOPICS = [
    "Patient identification",
    "Coverage",
    "Dose and safety",
    "Referral workflow",
    "Prior authorization",
    "Adherence",
]
OUTCOMES = ["Positive", "Neutral", "Follow-up", "No Reach", "Abandoned"]
MESSAGE_TOPICS = [
    "Patient identification",
    "Coverage and reimbursement",
    "Dose and administration",
    "Safety and tolerability",
    "Referral workflow and SP onboarding",
    "Prior authorization support",
    "Adherence program enrollment",
    "Clinical data - efficacy",
    "Clinical data - REMS requirements",
    "Competitive differentiation",
]
PRODUCTS = [
    {"product_name": "Roventra", "brand_generic": "Brand", "therapy_area": "Launch product"},
    {"product_name": "Nexoral", "brand_generic": "Brand", "therapy_area": "Competitor"},
    {"product_name": "Vexpro", "brand_generic": "Brand", "therapy_area": "Competitor"},
]

# Lab test catalog: (test_name, loinc_code, result_unit)
LAB_TESTS = [
    ("Hemoglobin A1c", "4548-4", "percent"),
    ("LDL Cholesterol", "2089-1", "mg/dL"),
    ("Systolic BP", "8480-6", "mmHg"),
    ("Diastolic BP", "8462-4", "mmHg"),
    ("AST", "1920-8", "U/L"),
    ("ALT", "1742-6", "U/L"),
    ("PSA", "2857-1", "ng/mL"),
    ("PD-L1 TPS", "85319-2", "percent"),
]

# Short alias map for internal use (test_name → short key used in legacy code)
_LAB_ALIAS = {
    "Hemoglobin A1c": "A1C",
    "LDL Cholesterol": "LDL",
    "Systolic BP": "BP_SYSTOLIC",
    "Diastolic BP": "BP_DIASTOLIC",
    "AST": "AST",
    "ALT": "ALT",
    "PSA": "PSA",
    "PD-L1 TPS": "PD_L1_PERCENT",
}

# Reference ranges: (low_normal, high_normal)
LAB_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "A1C":          (4.0,  5.6),
    "LDL":          (0,    100),
    "BP_SYSTOLIC":  (90,   120),
    "BP_DIASTOLIC": (60,   80),
    "AST":          (10,   40),
    "ALT":          (7,    56),
    "PSA":          (0,    4.0),
    "PD_L1_PERCENT":(0,    1),
}

# EHR_LABS kept for backward compatibility with lab.py
EHR_LABS = [
    ("A1C", "percent"),
    ("LDL", "mg/dL"),
    ("BP_SYSTOLIC", "mmHg"),
    ("BP_DIASTOLIC", "mmHg"),
    ("AST", "U/L"),
    ("ALT", "U/L"),
    ("PSA", "ng/mL"),
    ("PD_L1_PERCENT", "percent"),
]

PROBLEM_LIST_CODES = {
    "Launch condition": ["E11.9", "E11.65", "E11.40"],
    "Oncology":         ["C34.10", "C34.90", "Z85.118"],
    "Cardiology":       ["I50.9", "I10", "I25.10"],
    "Rheumatology":     ["M06.9", "M05.79", "M32.9"],
    "Other":            ["F41.9", "G89.3", "K21.9"],
}

_PHARMA_COMPANIES = [
    "NovaBio Pharmaceuticals",
    "Helix Therapeutics",
    "Meridian Health Sciences",
    "ClearPath BioMed",
    "Apex Oncology",
]
_PAYMENT_CATEGORIES = [
    "Speaking Fees",
    "Consulting Fees",
    "Research Grants",
    "Education/Training",
    "Travel Reimbursement",
]
_PAYMENT_WEIGHTS_KOL = [30, 25, 20, 15, 10]
_PAYMENT_WEIGHTS_STD = [10, 15, 10, 35, 30]
_PAYMENT_RANGES = {
    "Speaking Fees": (1_500, 15_000),
    "Consulting Fees": (2_000, 25_000),
    "Research Grants": (5_000, 100_000),
    "Education/Training": (200, 1_500),
    "Travel Reimbursement": (100, 1_200),
}

_PART_D_SPECIALTIES = [
    "Internal Medicine", "Family Practice", "Cardiology",
    "Endocrinology", "Oncology/Hematology", "Rheumatology",
    "Pulmonology", "Psychiatry", "Geriatric Medicine",
]
_PART_D_DRUG_NAMES = {
    "Roventra": {"generic": "roventra", "class": "Launch product"},
    "Nexoral":  {"generic": "nexoral",  "class": "Competitor A"},
    "Vexpro":   {"generic": "vexpro",   "class": "Competitor B"},
}


@dataclass
class EntityBundle:
    patients: list[dict]
    providers: list[dict]
    accounts: list[dict]
    payers: list[dict]
    products: list[dict]
    diagnosis_codes: list[dict]
    ndc_codes: list[dict]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_git_commit(root: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def validate_manifest_contract(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Package manifest not found at {manifest_path}. "
            "Run generate_all_synthetic_data.py first."
        )
    manifest = load_manifest(manifest_path)
    required_top_level = [
        "schema_version",
        "generated_at_utc",
        "run_config",
        "scenario_settings",
        "table_contracts",
    ]
    missing = [key for key in required_top_level if key not in manifest]
    if missing:
        raise ValueError(
            "Package manifest is missing required contract fields: "
            + ", ".join(missing)
        )
    table_contracts = manifest.get("table_contracts", {})
    if not table_contracts:
        raise ValueError("Package manifest contains no table contracts.")
    for relative_path, contract in table_contracts.items():
        csv_path = output_dir / relative_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Manifest contract references missing file: {csv_path}"
            )
        expected_row_count = contract.get("row_count")
        if expected_row_count is None:
            raise ValueError(
                f"Manifest contract for {relative_path} is missing row_count."
            )
        expected_hash = contract.get("file_sha256")
        if not expected_hash:
            raise ValueError(
                f"Manifest contract for {relative_path} is missing file_sha256."
            )
        observed_hash = sha256_file(csv_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Generated file does not match its manifest hash: {relative_path}. "
                "Regenerate the Chapter 3 package before analysis."
            )
    return manifest


def rand_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 0)))


def choice_weighted(rng: random.Random, values: list[str], weights: list[int]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def load_rows_from_csv(path: Path, limit: int | None = None) -> list[dict]:
    if not path or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Built-in code/lookup helpers
# ---------------------------------------------------------------------------

def built_in_diagnosis_codes() -> list[dict]:
    return [
        {"CODE": "E11.9", "CODE_ORIG": "E11.9", "CODE_DESCRIPTION": "Type 2 diabetes mellitus without complications"},
        {"CODE": "I10", "CODE_ORIG": "I10", "CODE_DESCRIPTION": "Essential hypertension"},
        {"CODE": "E78.5", "CODE_ORIG": "E78.5", "CODE_DESCRIPTION": "Hyperlipidemia, unspecified"},
        {"CODE": "J45.909", "CODE_ORIG": "J45.909", "CODE_DESCRIPTION": "Unspecified asthma, uncomplicated"},
        {"CODE": "M06.9", "CODE_ORIG": "M06.9", "CODE_DESCRIPTION": "Rheumatoid arthritis, unspecified"},
        {"CODE": "C50.919", "CODE_ORIG": "C50.919", "CODE_DESCRIPTION": "Malignant neoplasm of unspecified site of unspecified female breast"},
        {"CODE": "C34.90", "CODE_ORIG": "C34.90", "CODE_DESCRIPTION": "Malignant neoplasm of unspecified part of unspecified bronchus or lung"},
        {"CODE": "I50.9", "CODE_ORIG": "I50.9", "CODE_DESCRIPTION": "Heart failure, unspecified"},
        {"CODE": "N18.9", "CODE_ORIG": "N18.9", "CODE_DESCRIPTION": "Chronic kidney disease, unspecified"},
        {"CODE": "G89.3", "CODE_ORIG": "G89.3", "CODE_DESCRIPTION": "Neoplasm related pain"},
        {"CODE": "K21.9", "CODE_ORIG": "K21.9", "CODE_DESCRIPTION": "Gastro-esophageal reflux disease without esophagitis"},
        {"CODE": "F41.9", "CODE_ORIG": "F41.9", "CODE_DESCRIPTION": "Anxiety disorder, unspecified"},
    ]


def load_diagnosis_codes(path: Path | None) -> list[dict]:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Diagnosis dictionary not found: {path}")
        rows = load_rows_from_csv(path, limit=40)
        mapped: list[dict] = []
        for row in rows:
            code = row.get("CODE") or row.get("code") or row.get("Code") or ""
            desc = row.get("CODE_DESCRIPTION") or row.get("DESCRIPTION") or row.get("description") or code
            orig = row.get("CODE_ORIG") or code
            if code:
                mapped.append({"CODE": code, "CODE_ORIG": orig, "CODE_DESCRIPTION": desc})
        if mapped:
            return mapped
        raise ValueError(
            f"Diagnosis dictionary contains no usable CODE values: {path}"
        )
    return built_in_diagnosis_codes()


def built_in_ndc_codes() -> list[dict]:
    return [
        {"NDC": "90000-1001-11", "BRAND_GENERIC": "Brand", "CUI_L1_NAME": "Roventra", "INGREDIENT_NAME_ARRAY": "synthetic ingredient a", "ALL_INGREDIENTS": "synthetic ingredient a"},
        {"NDC": "90000-1002-11", "BRAND_GENERIC": "Brand", "CUI_L1_NAME": "Nexoral", "INGREDIENT_NAME_ARRAY": "synthetic ingredient b", "ALL_INGREDIENTS": "synthetic ingredient b"},
        {"NDC": "90000-1003-11", "BRAND_GENERIC": "Brand", "CUI_L1_NAME": "Vexpro", "INGREDIENT_NAME_ARRAY": "synthetic ingredient c", "ALL_INGREDIENTS": "synthetic ingredient c"},
        {"NDC": "90000-1004-11", "BRAND_GENERIC": "Generic", "CUI_L1_NAME": "Supportive Med", "INGREDIENT_NAME_ARRAY": "supportive ingredient", "ALL_INGREDIENTS": "supportive ingredient"},
    ]


def load_ndc_codes(path: Path | None) -> list[dict]:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Drug dictionary not found: {path}")
        rows = load_rows_from_csv(path, limit=40)
        mapped: list[dict] = []
        for row in rows:
            ndc = row.get("NDC") or row.get("ndc") or row.get("Ndc") or ""
            if not ndc:
                continue
            mapped.append(
                {
                    "NDC": ndc,
                    "BRAND_GENERIC": row.get("BRAND_GENERIC", ""),
                    "CUI_L1_NAME": row.get("CUI_L1_NAME", row.get("BRAND", row.get("CUI_L2_NAME_ARRAY", ""))),
                    "INGREDIENT_NAME_ARRAY": row.get("INGREDIENT_NAME_ARRAY", ""),
                    "ALL_INGREDIENTS": row.get("ALL_INGREDIENTS", ""),
                }
            )
        if len(mapped) >= 2:
            return mapped
        raise ValueError(
            f"Drug dictionary must contain at least two usable NDC values: {path}"
        )
    return built_in_ndc_codes()


# ---------------------------------------------------------------------------
# Entity construction
# ---------------------------------------------------------------------------

def _entity_counts(n_patients: int) -> tuple[int, int]:
    n_accounts = max(24, min(n_patients // 80, 1_000))
    n_providers = max(60, min(n_patients // 30, 5_000))
    return n_accounts, n_providers


def build_entities(
    rng: random.Random,
    diagnosis_path: Path | None,
    drug_path: Path | None,
    n_patients: int = 20_000,
) -> EntityBundle:
    accounts: list[dict] = []
    providers: list[dict] = []
    patients: list[dict] = []
    payers: list[dict] = []
    products = PRODUCTS[:]
    diagnosis_codes = load_diagnosis_codes(diagnosis_path)
    ndc_codes = load_ndc_codes(drug_path)

    n_accounts, n_providers = _entity_counts(n_patients)
    id_width = max(3, len(str(n_accounts)))
    hcp_width = max(4, len(str(n_providers)))
    pat_width = max(5, len(str(n_patients)))

    for i, (payer_type, region) in enumerate(
        [
            ("Commercial", "Northeast"),
            ("Commercial", "South"),
            ("Commercial", "Midwest"),
            ("Medicare Advantage", "Northeast"),
            ("Medicare Advantage", "South"),
            ("Medicare Part D", "Midwest"),
            ("Medicaid", "West"),
            ("Commercial", "West"),
        ],
        start=1,
    ):
        payers.append(
            {
                "payer_id": f"PAY{i:03d}",
                "payer_name": f"{region} {payer_type} Plan {i}",
                "payer_type": payer_type,
                "region": region,
            }
        )

    for i in range(1, n_accounts + 1):
        city, state, region = rng.choice(STATES)
        accounts.append(
            {
                "account_id": f"ACC{i:0{id_width}d}",
                "account_name": f"{city} Care {i:0{id_width}d}",
                "account_type": rng.choice(ACCOUNT_TYPES),
                "city": city,
                "state": state,
                "region": region,
                "territory": f"T{(i % 8) + 1:02d}",
                "capacity": rng.randint(1, 4),
            }
        )

    for i in range(1, n_providers + 1):
        account = rng.choice(accounts)
        specialty = choice_weighted(rng, SPECIALTIES, [12, 10, 8, 8, 6, 16])
        secondary = SECONDARY_BY_SPECIALTY.get(specialty, "")
        if secondary and rng.random() >= 0.28:
            secondary = ""
        providers.append(
            {
                "hcp_id": f"HCP{i:0{hcp_width}d}",
                "npi": f"9{i:09d}"[-10:],
                "specialty": specialty,
                "specialty_secondary": secondary,
                "credential": CREDENTIAL_BY_SPECIALTY.get(specialty, "MD"),
                "account_id": account["account_id"],
                "territory": account["territory"],
                "state": account["state"],
                "region": account["region"],
            }
        )

    account_providers: dict[str, list[dict]] = defaultdict(list)
    for p in providers:
        account_providers[p["account_id"]].append(p)

    # Build Primary Care pool for referral network generation.
    # 20 super-referrers are designated by sorting PC providers by NPI (reproducible)
    # and taking the first 20 (pc_sorted[:20]); biased selection in patient assignment
    # gives them higher patient counts, creating useful betweenness centrality in the
    # referral graph that Chapter 6's network analysis depends on.
    pc_providers = [p for p in providers if p["specialty"] == "Primary Care"]
    pc_sorted = sorted(pc_providers, key=lambda p: p["npi"])

    for i in range(1, n_patients + 1):
        account = rng.choice(accounts)
        provider_pool = account_providers[account["account_id"]] or providers
        hcp = rng.choice(provider_pool)
        payer = rng.choice(payers)
        state = rng.choice([s for s in STATES if s[2] == account["region"]] or STATES)
        age_band = choice_weighted(rng, ["18-34", "35-49", "50-64", "65+"], [2, 4, 5, 3])
        condition = choice_weighted(
            rng,
            [bucket for bucket, _ in CONDITION_BUCKETS],
            [weight for _, weight in CONDITION_BUCKETS],
        )
        coverage_start = date(2023, rng.randint(1, 12), rng.randint(1, 28))
        coverage_end = date(2025, rng.randint(1, 12), rng.randint(1, 28))

        # Assign a separate primary_care_npi for patients whose prescriber is a specialist.
        # Uses hash-based selection (not the main RNG) so the shared seed stream is
        # unchanged and ch03–ch06 outputs remain identical after this change.
        # 35% of specialist patients are routed through a super-referrer PCP to produce
        # high-betweenness bridge nodes in the referral graph.
        patient_id_str = f"PAT{i:0{pat_width}d}"
        if hcp["specialty"] == "Primary Care" or not pc_providers:
            primary_care_npi = ""
        else:
            _h1 = int.from_bytes(
                hashlib.sha256(f"pcp-assign:{patient_id_str}".encode()).digest()[:8], "big"
            )
            _frac = _h1 / 2**64
            if _frac < 0.35:
                _h2 = int.from_bytes(
                    hashlib.sha256(f"pcp-super:{patient_id_str}".encode()).digest()[:8], "big"
                )
                primary_care_npi = pc_sorted[_h2 % 20]["npi"]
            else:
                _h2 = int.from_bytes(
                    hashlib.sha256(f"pcp-any:{patient_id_str}".encode()).digest()[:8], "big"
                )
                primary_care_npi = pc_providers[_h2 % len(pc_providers)]["npi"]

        patients.append(
            {
                "patient_id": f"PAT{i:0{pat_width}d}",
                "account_id": account["account_id"],
                "prescriber_npi": hcp["npi"],
                "primary_care_npi": primary_care_npi,
                "payer_id": payer["payer_id"],
                "state": state[1],
                "region": account["region"],
                "age_band": age_band,
                "sex": rng.choice(["F", "M"]),
                "condition_bucket": condition,
                # coverage dates stored internally for claims generation; removed from output patients.csv
                "_coverage_start": coverage_start.isoformat(),
                "_coverage_end": coverage_end.isoformat(),
            }
        )

    return EntityBundle(
        patients=patients,
        providers=providers,
        accounts=accounts,
        payers=payers,
        products=products,
        diagnosis_codes=diagnosis_codes,
        ndc_codes=ndc_codes,
    )


def apply_canonical_overrides(bundle: EntityBundle) -> None:
    """Force the teaching-case entities to match the cross-chapter canonical record.

    Falls back gracefully on small test runs where canonical IDs may not exist.
    """
    acc089 = next((a for a in bundle.accounts if a["account_id"] == "ACC089"), None)
    if acc089 is None:
        # Small test run — use the first available account; skip teaching-entity pinning
        acc089 = bundle.accounts[0] if bundle.accounts else None
        if acc089 is None:
            return

    hcp0280_npi: str = ""
    for provider in bundle.providers:
        if provider["hcp_id"] == "HCP0280":
            provider["specialty"] = "Endocrinology"
            provider["specialty_secondary"] = ""
            provider["account_id"] = acc089["account_id"]
            provider["territory"] = acc089["territory"]
            provider["state"] = acc089["state"]
            provider["region"] = acc089["region"]
            hcp0280_npi = provider["npi"]
            break

    fallback_npi = bundle.providers[0]["npi"] if bundle.providers else ""
    for patient in bundle.patients:
        if patient["patient_id"] == "PAT02034":
            patient["sex"] = "F"
            patient["account_id"] = acc089["account_id"]
            patient["prescriber_npi"] = hcp0280_npi if hcp0280_npi else fallback_npi
            patient["payer_id"] = "PAY002"
            patient["state"] = acc089["state"]
            patient["region"] = acc089["region"]
            patient["condition_bucket"] = "Launch condition"
            break


# ---------------------------------------------------------------------------
# Reference table writers
# ---------------------------------------------------------------------------

def write_dataset_folder(
    base: Path,
    name: str,
    description: str,
    files: list[tuple[str, list[dict], list[str]]],
) -> None:
    folder = base / name
    ensure_dir(folder)
    write_text(
        folder / "README.md",
        f"""# {description}

Synthetic pharmaceutical data for Chapter 3.

Files:
{chr(10).join(f'- `{file_name}`' for file_name, _, _ in files)}
""",
    )
    for file_name, rows, fieldnames in files:
        if rows:
            write_csv(folder / file_name, rows, fieldnames)


def _make_patient_enrollment_rows(bundle: EntityBundle) -> list[dict]:
    """Build one enrollment row per patient from internal coverage dates."""
    payer_type_map = {p["payer_id"]: p["payer_type"] for p in bundle.payers}
    rows = []
    for p in bundle.patients:
        payer_type = payer_type_map.get(p["payer_id"], "Commercial")
        rows.append({
            "patient_id": p["patient_id"],
            "eligibility_start_date": p["_coverage_start"],
            "eligibility_end_date": p["_coverage_end"],
            "payer_id": p["payer_id"],
            "payer_type": payer_type,
            "has_medical_coverage": True,
            "has_pharmacy_coverage": True,
            "product_type": "PPO",
        })
    return rows


def make_reference_tables(output_dir: Path, bundle: EntityBundle) -> None:
    ref_dir = output_dir / "reference"
    ensure_dir(ref_dir)

    # patients.csv — patient demographics only; join keys (payer, account, prescriber) come from claims
    pat_output_fields = [
        "patient_id", "state", "region", "age_band", "sex", "true_launch_condition",
    ]
    pat_rows = [
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
    write_csv(ref_dir / "patients.csv", pat_rows, pat_output_fields)

    # patient_enrollments.csv — eligibility periods; demographics join from patients.csv
    enrollment_rows = _make_patient_enrollment_rows(bundle)
    enrollment_fields = [
        "patient_id", "eligibility_start_date", "eligibility_end_date",
        "payer_id", "payer_type",
        "has_medical_coverage", "has_pharmacy_coverage", "product_type",
    ]
    write_csv(ref_dir / "patient_enrollments.csv", enrollment_rows, enrollment_fields)

    # providers.csv — vendor-delivered provider directory (NPI-keyed)
    prov_dir_fields = [
        "npi", "specialty_1", "specialty_2", "provider_state",
        "provider_type", "credential", "primary_facility_npi",
    ]
    prov_dir_rows = [
        {
            "npi": p["npi"],
            "specialty_1": p["specialty"],
            "specialty_2": p.get("specialty_secondary", ""),
            "provider_state": p["state"],
            "provider_type": "Individual",
            "credential": p["credential"],
            "primary_facility_npi": "",
        }
        for p in bundle.providers
    ]
    write_csv(ref_dir / "providers.csv", prov_dir_rows, prov_dir_fields)

    # hcp_targets.csv — internal commercial target list (~42% of universe, NPI-keyed)
    # Deterministic subset: seeded selection always includes HCP0280's NPI for teaching case
    _target_rng = random.Random(42)
    n_targeted = max(1, round(len(bundle.providers) * 0.42))
    _targeted_npis: set[str] = {p["npi"] for p in _target_rng.sample(bundle.providers, n_targeted)}
    _hcp0280 = next((p for p in bundle.providers if p["hcp_id"] == "HCP0280"), None)
    if _hcp0280:
        _targeted_npis.add(_hcp0280["npi"])
    hcp_target_fields = ["npi", "account_id", "territory", "state", "region", "specialty_1"]
    hcp_target_rows = [
        {
            "npi": p["npi"],
            "account_id": p["account_id"],
            "territory": p["territory"],
            "state": p["state"],
            "region": p["region"],
            "specialty_1": p["specialty"],
        }
        for p in bundle.providers
        if p["npi"] in _targeted_npis
    ]
    write_csv(ref_dir / "hcp_targets.csv", hcp_target_rows, hcp_target_fields)

    # accounts.csv — unchanged
    write_csv(ref_dir / "accounts.csv", bundle.accounts, list(bundle.accounts[0].keys()))

    # payers.csv
    write_csv(ref_dir / "payers.csv", bundle.payers, list(bundle.payers[0].keys()))

    # products.csv
    write_csv(ref_dir / "products.csv", bundle.products, list(bundle.products[0].keys()))

    # diagnosis_codes.csv — unchanged
    write_csv(ref_dir / "diagnosis_codes.csv", bundle.diagnosis_codes, list(bundle.diagnosis_codes[0].keys()))

    # ndc_codes.csv
    ndc_rows = [
        {
            "ndc": row["NDC"],
            "brand_generic": row["BRAND_GENERIC"],
            "drug_name": row["CUI_L1_NAME"],
            "ingredient": row["INGREDIENT_NAME_ARRAY"],
        }
        for row in bundle.ndc_codes
    ]
    write_csv(ref_dir / "ndc_codes.csv", ndc_rows, ["ndc", "brand_generic", "drug_name", "ingredient"])

    write_text(
        ref_dir / "README.md",
        """# Reference tables

- `patients.csv`: patient-level internal analytical attributes
- `patient_enrollments.csv`: coverage eligibility periods (one row per patient-payer period)
- `providers.csv`: vendor-delivered provider directory, keyed by NPI
- `hcp_targets.csv`: internal commercial target list, maps NPI to territory and account
- `accounts.csv`: account hierarchy and sales regions
- `payers.csv`: payer references
- `products.csv`: launch and competitor product list
- `diagnosis_codes.csv`: ICD-10 code dictionary
- `ndc_codes.csv`: drug code dictionary
""",
    )


def write_manifest(
    output_dir: Path,
    bundle: EntityBundle,
    run_config: dict | None = None,
    scenario_settings: dict | None = None,
    table_contracts: dict | None = None,
    diagnosis_path: Path | None = None,
    drug_path: Path | None = None,
) -> None:
    contracts = {
        relative_path: dict(contract)
        for relative_path, contract in (table_contracts or {}).items()
    }
    for relative_path, contract in contracts.items():
        csv_path = output_dir / relative_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Cannot write manifest contract for missing file: {csv_path}"
            )
        contract["file_sha256"] = sha256_file(csv_path)

    manifest = {
        "description": "Synthetic pharmaceutical data package for Chapter 3",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_root": str(ROOT),
        "generator_version": {
            "git_commit": current_git_commit(),
            "git_commit_source": "git rev-parse HEAD",
        },
        "run_config": run_config or {},
        "scenario_settings": scenario_settings or {},
        "folders": {
            "reference": "Shared lookup tables and code dictionaries",
            "claims_medical": "Medical claims (headers and service lines)",
            "claims_pharmacy": "Pharmacy claims",
            "claims_lab": "Lab results",
            "specialty_pharmacy": "Specialty pharmacy hub events",
            "crm_veeva": "CRM interactions and territory alignment",
            "digital_engagement": "Digital engagement events",
            "market_access": "Market access rules",
            "formulary": "Formulary rules",
            "open_payments": "Synthetic Open Payments-style records",
            "cms_part_d": "Synthetic Medicare Part D prescriber summaries",
        },
        "reference_sources": {
            "diagnosis_codes": (
                str(diagnosis_path) if diagnosis_path else "built-in synthetic dictionary"
            ),
            "drug_codes": (
                str(drug_path) if drug_path else "built-in synthetic dictionary"
            ),
        },
        "counts": {
            "patients": len(bundle.patients),
            "providers": len(bundle.providers),
            "accounts": len(bundle.accounts),
            "payers": len(bundle.payers),
            "products": len(bundle.products),
        },
        "table_contracts": contracts,
    }
    manifest_path = output_dir / "manifest.json"
    write_text(manifest_path, json.dumps(manifest, indent=2))
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    write_text(manifest_path, json.dumps(manifest, indent=2))
    write_text(
        output_dir / "README.md",
        """# Chapter 3 Synthetic Data Package

This folder contains the synthetic datasets generated for Chapter 3.

The manifest records the seed, patient count, schema version, key scenario
settings, row counts, date ranges, and table hashes for the current files.

From the repository root, run:

`uv run python ch03_data/scripts/generate_all_synthetic_data.py`
""",
    )
