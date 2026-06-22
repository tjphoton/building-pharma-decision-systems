#!/usr/bin/env python3
"""Download small public reference extracts used across the book.

The script uses official CMS APIs and writes bounded, specialty-filtered files
that are small enough for the chapter notebooks. Use ``--dry-run`` to create
the same file contracts from the local synthetic data without network access.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CHAPTER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = CHAPTER_DIR / "output_data"
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_ROOT / "generated_data"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "public_reference"

CMS_CATALOG_URL = "https://data.cms.gov/data.json"
OPEN_PAYMENTS_CATALOG_URL = "https://openpaymentsdata.cms.gov/data.json"

PART_D_DATASET_TITLE = "Medicare Part D Prescribers - by Provider and Drug"
OPEN_PAYMENTS_DATASET_TITLE = "2022 General Payment Data"

PART_D_SPECIALTIES = (
    "Cardiology",
    "Endocrinology",
    "Hematology-Oncology",
    "Rheumatology",
    "Pulmonary Disease",
)
OPEN_PAYMENTS_SPECIALTIES = (
    "Allopathic & Osteopathic Physicians|Internal Medicine|Cardiovascular Disease",
    "Allopathic & Osteopathic Physicians|Internal Medicine|Endocrinology, Diabetes & Metabolism",
    "Allopathic & Osteopathic Physicians|Internal Medicine|Medical Oncology",
    "Allopathic & Osteopathic Physicians|Internal Medicine|Rheumatology",
    "Allopathic & Osteopathic Physicians|Internal Medicine|Pulmonary Disease",
)

PART_D_COLUMNS = {
    "Prscrbr_Type": "prscrbr_type",
    "Brnd_Name": "drug_name",
    "Tot_Clms": "tot_clms",
    "Tot_Benes": "tot_benes",
    "Tot_Drug_Cst": "tot_drug_cst",
}
OPEN_PAYMENTS_COLUMNS = {
    "covered_recipient_npi": "hcp_id",
    "covered_recipient_specialty_1": "physician_specialty",
    "applicable_manufacturer_or_applicable_gpo_making_payment_name": "company_name",
    "total_amount_of_payment_usdollars": "payment_amount",
    "nature_of_payment_or_transfer_of_value": "payment_category",
}

# Published national prevalence anchors. NHANES does not release region in its
# public files, so Chapter 4 applies national prevalence anchors to regional
# population denominators. Every row below is traceable to a specific official
# publication, period, and table or figure.
NHANES_PREVALENCE_ROWS = (
    {
        "condition": "diabetes",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "diagnosed prevalence",
        "prevalence": 0.113,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 1",
        "publication_title": "Prevalence of Total, Diagnosed, and Undiagnosed Diabetes in Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "516",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db516.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "diabetes",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "total prevalence",
        "prevalence": 0.158,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 1",
        "publication_title": "Prevalence of Total, Diagnosed, and Undiagnosed Diabetes in Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "516",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db516.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "diabetes",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "undiagnosed prevalence",
        "prevalence": 0.045,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 1",
        "publication_title": "Prevalence of Total, Diagnosed, and Undiagnosed Diabetes in Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "516",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db516.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "hypertension",
        "population": "adults age 18+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 18 and older",
        "measure": "prevalence",
        "prevalence": 0.477,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 1",
        "publication_title": "Prevalence and Control of Hypertension Among Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "511",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db511.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "hypertension",
        "population": "adults age 18+ with hypertension",
        "population_definition": "U.S. civilian noninstitutionalized adults age 18 and older with hypertension",
        "measure": "controlled prevalence",
        "prevalence": 0.207,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 4",
        "publication_title": "Prevalence and Control of Hypertension Among Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "511",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db511.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "hyperlipidemia",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "high total cholesterol prevalence",
        "prevalence": 0.113,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 1",
        "publication_title": "Total and High-density Lipoprotein Cholesterol in Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "515",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db515.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "hyperlipidemia",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "low HDL prevalence",
        "prevalence": 0.138,
        "survey_period": "August 2021-August 2023",
        "estimate_type": "crude prevalence",
        "age_adjustment_status": "not age-adjusted",
        "table_or_figure": "Figure 2",
        "publication_title": "Total and High-density Lipoprotein Cholesterol in Adults: United States, August 2021-August 2023",
        "publication_series": "NCHS Data Brief",
        "publication_number": "515",
        "source_url": "https://www.cdc.gov/nchs/products/databriefs/db515.htm",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "obesity",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "prevalence",
        "prevalence": 0.419,
        "survey_period": "2017-March 2020",
        "estimate_type": "age-adjusted prevalence",
        "age_adjustment_status": "age-adjusted to the 2000 U.S. standard population",
        "table_or_figure": "Table 5",
        "publication_title": "Trends in Obesity and Severe Obesity Prevalence in US Youth and Adults by Sex and Age, 2007-2008 to 2017-March 2020",
        "publication_series": "National Health Statistics Reports",
        "publication_number": "158",
        "source_url": "https://www.cdc.gov/nchs/data/nhsr/nhsr158-508.pdf",
        "retrieval_date": "2026-06-11",
    },
    {
        "condition": "severe_obesity",
        "population": "adults age 20+",
        "population_definition": "U.S. civilian noninstitutionalized adults age 20 and older",
        "measure": "prevalence",
        "prevalence": 0.092,
        "survey_period": "2017-March 2020",
        "estimate_type": "age-adjusted prevalence",
        "age_adjustment_status": "age-adjusted to the 2000 U.S. standard population",
        "table_or_figure": "Table 6",
        "publication_title": "Trends in Obesity and Severe Obesity Prevalence in US Youth and Adults by Sex and Age, 2007-2008 to 2017-March 2020",
        "publication_series": "National Health Statistics Reports",
        "publication_number": "158",
        "source_url": "https://www.cdc.gov/nchs/data/nhsr/nhsr158-508.pdf",
        "retrieval_date": "2026-06-11",
    },
)

PUBLIC_REFERENCE_SCHEMA_VERSION = "2026-06-11"


def _frame_contract(
    frame: pd.DataFrame,
    *,
    data_source: str,
    source_url: str,
    extract_year: int | None,
    source_mode: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["data_source"] = data_source
    result["source_url"] = source_url
    result["extract_year"] = extract_year
    result["source_mode"] = source_mode
    result["schema_version"] = PUBLIC_REFERENCE_SCHEMA_VERSION
    result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


def _read_json(url: str, params: dict | list[tuple[str, object]] | None = None) -> object:
    if params:
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pharma-decision-science/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _resolve_distribution(
    catalog_url: str,
    dataset_title: str,
    year: int,
    url_field: str,
) -> str:
    """Resolve a distribution URL from a Project Open Data catalog."""

    catalog = _read_json(catalog_url)
    if not isinstance(catalog, dict):
        raise RuntimeError(f"Unexpected catalog response from {catalog_url}")

    for dataset in catalog.get("dataset", []):
        if dataset.get("title") != dataset_title:
            continue
        for distribution in dataset.get("distribution", []):
            if str(year) in distribution.get("title", "") and distribution.get(url_field):
                return str(distribution[url_field])
    raise RuntimeError(
        f"Could not find {dataset_title!r} year {year} in {catalog_url}"
    )


def download_cms_partd(
    output_path: Path,
    rows_per_specialty: int = 2_000,
) -> pd.DataFrame:
    """Download a bounded 2022 CMS Part D provider-and-drug extract."""

    api_url = _resolve_distribution(
        CMS_CATALOG_URL,
        PART_D_DATASET_TITLE,
        2022,
        "accessURL",
    )
    frames = []
    for specialty in PART_D_SPECIALTIES:
        records = _read_json(
            api_url,
            {
                "offset": 0,
                "size": rows_per_specialty,
                "filter[Prscrbr_Type]": specialty,
            },
        )
        if isinstance(records, list) and records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise RuntimeError("CMS Part D API returned no rows for the selected specialties")

    result = pd.concat(frames, ignore_index=True)
    result = result[list(PART_D_COLUMNS)].rename(columns=PART_D_COLUMNS)
    for column in ("tot_clms", "tot_benes", "tot_drug_cst"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = _frame_contract(
        result,
        data_source="CMS Part D 2022",
        source_url=api_url,
        extract_year=2022,
        source_mode="live",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def download_open_payments(
    output_path: Path,
    rows_per_specialty: int = 500,
) -> pd.DataFrame:
    """Download a bounded 2022 Open Payments general-payment extract."""

    catalog = _read_json(OPEN_PAYMENTS_CATALOG_URL)
    if not isinstance(catalog, dict):
        raise RuntimeError("Unexpected Open Payments catalog response")

    dataset_id = None
    for dataset in catalog.get("dataset", []):
        if dataset.get("title") == OPEN_PAYMENTS_DATASET_TITLE:
            dataset_id = dataset.get("identifier")
            break
    if not dataset_id:
        raise RuntimeError("Could not resolve the 2022 Open Payments dataset")

    api_url = (
        "https://openpaymentsdata.cms.gov/api/1/datastore/query/"
        f"{dataset_id}/0"
    )
    frames = []
    for specialty in OPEN_PAYMENTS_SPECIALTIES:
        params: list[tuple[str, object]] = [
            ("limit", rows_per_specialty),
            ("count", "false"),
            ("conditions[0][property]", "covered_recipient_specialty_1"),
            ("conditions[0][value]", specialty),
            ("conditions[0][operator]", "="),
        ]
        payload = _read_json(api_url, params)
        if isinstance(payload, dict) and payload.get("results"):
            frames.append(pd.DataFrame(payload["results"]))

    if not frames:
        raise RuntimeError("Open Payments API returned no rows for selected specialties")

    result = pd.concat(frames, ignore_index=True)
    result = result[list(OPEN_PAYMENTS_COLUMNS)].rename(columns=OPEN_PAYMENTS_COLUMNS)
    result["payment_amount"] = pd.to_numeric(
        result["payment_amount"],
        errors="coerce",
    )
    result["payment_year"] = 2022
    result = _frame_contract(
        result,
        data_source="CMS Open Payments 2022",
        source_url=api_url,
        extract_year=2022,
        source_mode="live",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def write_nhanes_prevalence(output_path: Path) -> pd.DataFrame:
    """Write the documented NHANES prevalence reference table."""

    result = pd.DataFrame(NHANES_PREVALENCE_ROWS)
    result["data_source"] = "official published prevalence anchor"
    result["source_mode"] = "curated_static"
    result["schema_version"] = PUBLIC_REFERENCE_SCHEMA_VERSION
    result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def _write_synthetic_partd(data_dir: Path, output_path: Path) -> pd.DataFrame:
    source = data_dir / "cms_part_d" / "prescriber_summary.csv"
    if source.exists():
        result = pd.read_csv(source)
        keep = {
            "prscrbr_type": "prscrbr_type",
            "drug_name": "drug_name",
            "tot_clms": "tot_clms",
            "tot_benes": "tot_benes",
            "tot_drug_cst": "tot_drug_cst",
        }
        result = result[list(keep)].rename(columns=keep)
    else:
        providers = pd.read_csv(data_dir / "reference" / "providers.csv")
        rows = []
        for index, provider in providers.head(100).reset_index(drop=True).iterrows():
            for product_index, product in enumerate(("Roventra", "Nexoral", "Vexpro")):
                claims = 12 + ((index + 3 * product_index) % 40)
                rows.append(
                    {
                        "prscrbr_type": PART_D_SPECIALTIES[index % len(PART_D_SPECIALTIES)],
                        "drug_name": product,
                        "tot_clms": claims,
                        "tot_benes": max(11, claims // 2),
                        "tot_drug_cst": float(claims * (120 + 15 * product_index)),
                    }
                )
        result = pd.DataFrame(rows)
    result = _frame_contract(
        result,
        data_source="synthetic proxy",
        source_url="local generated Chapter 3 data",
        extract_year=2024,
        source_mode="dry_run",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def _write_synthetic_open_payments(data_dir: Path, output_path: Path) -> pd.DataFrame:
    source = data_dir / "open_payments" / "open_payments.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Synthetic Open Payments proxy not found at {source}. "
            "Run generate_all_synthetic_data.py first."
        )
    result = pd.read_csv(source)
    if "physician_specialty" not in result.columns:
        result["physician_specialty"] = ""
    contract_columns = [
        "npi",
        "physician_specialty",
        "company_name",
        "payment_amount",
        "payment_category",
        "payment_year",
    ]
    result = result[contract_columns]
    result = _frame_contract(
        result,
        data_source="synthetic proxy",
        source_url="local generated Chapter 3 data",
        extract_year=2024,
        source_mode="dry_run",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def run_download(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Create all three public-reference files and return their data frames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        partd = _write_synthetic_partd(data_dir, output_dir / "cms_partd.csv")
        payments = _write_synthetic_open_payments(
            data_dir,
            output_dir / "open_payments.csv",
        )
    else:
        partd = download_cms_partd(output_dir / "cms_partd.csv")
        payments = download_open_payments(output_dir / "open_payments.csv")
    nhanes = write_nhanes_prevalence(output_dir / "nhanes_prevalence.csv")
    return {
        "cms_partd": partd,
        "open_payments": payments,
        "nhanes_prevalence": nhanes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use local synthetic proxies and skip all network requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_download(args.output_dir, args.data_dir, args.dry_run)
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")
    print(f"Wrote public references to {args.output_dir}")


if __name__ == "__main__":
    main()
