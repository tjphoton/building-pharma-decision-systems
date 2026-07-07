"""Synthetic Roventra lifecycle generator for the forecasting chapter.

Builds one coherent weekly series from launch through loss of exclusivity
and a post-entry tail, at both the national level and a region/territory
hierarchy, using known ground-truth structural parameters so every method
in the chapter can be scored against the true generating process.

The structural model has five layers, applied in this order:

1. Bass diffusion for the structural weekly new-to-therapy rate (NBRx).
2. Access, promotional, and calendar-seasonality multipliers on that rate.
3. A persistence (survival) convolution that turns new starts into an
   on-therapy patient stock.
4. A refill-rate conversion from on-therapy stock to total prescriptions
   (TRx).
5. A loss-of-exclusivity decay applied to both new starts and the stock,
   representing brand share collapsing after generic entry.

`run_analysis.py` for the market-sizing and patient-journey chapters is not
called at generation time; only their named constants (`WASHOUT_DAYS`,
`ACCESS_PROBABILITY`) are imported. See the plan's Progress Tracker for the
reconciliation rationale.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from forecast_config import (  # noqa: E402
    ACCESS_IMPROVEMENT_DATE,
    ACCESS_MIX_AFTER,
    ACCESS_MIX_BEFORE,
    ACCESS_PROBABILITY,
    AS_OF_DATE,
    BACKGROUND_INCIDENCE_WEEKLY,
    BASS_CEILING_M,
    BASS_P,
    BASS_Q,
    GENERIC_ENTRY_DATE,
    HOLIDAY_DIP_FACTOR,
    HOLIDAY_DIP_WEEKS,
    JANUARY_RESET_LIFT,
    LAUNCH_DATE,
    LOE_HALF_LIFE_WEEKS,
    LOE_RESIDUAL_SHARE,
    NOISE_STD_FRACTION,
    PERSISTENCE_SCALE_MONTHS,
    PERSISTENCE_SHAPE,
    PROMO_FLIGHT_END,
    PROMO_FLIGHT_START,
    PROMO_LIFT,
    REFILLS_PER_PATIENT_MONTH,
    REGION_SHARES,
    ROOT,
    SEED,
    SERIES_END_DATE,
    TERRITORIES_PER_REGION,
)

from forecasting import bass_cumulative_fraction, persistence_survival  # noqa: E402

DAYS_PER_MONTH = 30.436875
WEEKS_PER_MONTH = 52.0 / 12.0
WEEKS_PER_YEAR = 52.0


def structural_new_starts(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Weekly new-to-therapy starts implied by the Bass ceiling, before covariates.

    Returns the depleting prevalent-pool component (the classic Bass wave,
    which exhausts itself as the one-time pool of early-diagnosed patients
    is captured) and the non-depleting background-incidence component
    (newly diagnosed patients arriving every week, scaled by the same
    adoption-awareness fraction so it ramps in with launch). Without the
    background term the on-therapy stock would decay back toward zero once
    the Bass wave saturates, years before generic entry; real chronic-
    therapy brands plateau instead, because new patients keep being
    diagnosed after the initial adopting wave is captured.
    """
    months_since_launch = (dates - LAUNCH_DATE).days / DAYS_PER_MONTH
    adoption_fraction = bass_cumulative_fraction(months_since_launch, BASS_P, BASS_Q)
    cumulative = adoption_fraction * BASS_CEILING_M
    bass_wave = np.clip(np.diff(cumulative, prepend=0.0), a_min=0.0, a_max=None)
    background = BACKGROUND_INCIDENCE_WEEKLY * adoption_fraction
    return bass_wave, background


def access_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Step change in new-start rate when the payer mix improves.

    Blends the access-quality weights against the payer mix before and
    after the coverage win, and expresses the after-mix as a ratio to the
    before-mix so the multiplier is 1.0 up to the improvement date.
    """
    blended_before = sum(ACCESS_MIX_BEFORE[k] * ACCESS_PROBABILITY[k] for k in ACCESS_PROBABILITY)
    blended_after = sum(ACCESS_MIX_AFTER[k] * ACCESS_PROBABILITY[k] for k in ACCESS_PROBABILITY)
    ratio = blended_after / blended_before
    return np.where(dates >= ACCESS_IMPROVEMENT_DATE, ratio, 1.0)


def promo_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Lift on new-start rate during the promotional flight window."""
    in_flight = (dates >= PROMO_FLIGHT_START) & (dates <= PROMO_FLIGHT_END)
    return np.where(in_flight, 1.0 + PROMO_LIFT, 1.0)


def seasonal_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """January insurance-reset lift and end-of-year holiday dip."""
    iso_week = dates.isocalendar().week.to_numpy()
    month = dates.month.to_numpy()
    multiplier = np.ones(len(dates))
    multiplier = np.where(month == 1, multiplier * (1.0 + JANUARY_RESET_LIFT), multiplier)
    dip_mask = np.isin(iso_week, list(HOLIDAY_DIP_WEEKS))
    multiplier = np.where(dip_mask, multiplier * HOLIDAY_DIP_FACTOR, multiplier)
    return multiplier


def loe_decay_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Decay toward the residual brand share after generic entry.

    A half-life decay from 1.0 toward `LOE_RESIDUAL_SHARE`, applied to both
    the new-start inflow (fewer new brand starts) and the on-therapy stock
    conversion (existing patients switching to generic), which are separate
    mechanisms and not a double application of the same effect.
    """
    weeks_since_entry = (dates - GENERIC_ENTRY_DATE).days / 7.0
    decay = LOE_RESIDUAL_SHARE + (1.0 - LOE_RESIDUAL_SHARE) * np.exp(
        -np.log(2) * np.clip(weeks_since_entry, a_min=0.0, a_max=None) / LOE_HALF_LIFE_WEEKS
    )
    return np.where(weeks_since_entry < 0, 1.0, decay)


def on_therapy_stock(new_starts: np.ndarray) -> np.ndarray:
    """Convolve weekly new starts with the persistence survival kernel."""
    n = len(new_starts)
    kernel = persistence_survival(np.arange(n, dtype=float))
    stock = np.convolve(new_starts, kernel)[:n]
    return stock


def generate_national_series(seed: int = SEED) -> pd.DataFrame:
    """Build the national weekly NBRx and TRx series with all structural layers applied."""
    dates = pd.date_range(LAUNCH_DATE, SERIES_END_DATE, freq="W-MON")
    rng = np.random.default_rng(seed)

    bass_wave, background = structural_new_starts(dates)
    starts = bass_wave + background
    access_mult = access_multiplier(dates)
    promo_mult = promo_multiplier(dates)
    season_mult = seasonal_multiplier(dates)
    loe_mult = loe_decay_multiplier(dates)

    adjusted_starts = np.clip(starts * access_mult * promo_mult * season_mult * loe_mult, 0.0, None)
    stock = on_therapy_stock(adjusted_starts)
    trx_structural = stock * REFILLS_PER_PATIENT_MONTH * (1.0 / WEEKS_PER_MONTH) * loe_mult

    nbrx_noise = rng.normal(1.0, NOISE_STD_FRACTION, size=len(dates))
    trx_noise = rng.normal(1.0, NOISE_STD_FRACTION, size=len(dates))

    df = pd.DataFrame(
        {
            "week_start": dates,
            "nbrx": np.clip(adjusted_starts * nbrx_noise, 0.0, None),
            "on_therapy_stock": stock,
            "trx": np.clip(trx_structural * trx_noise, 0.0, None),
            "access_multiplier": access_mult,
            "promo_multiplier": promo_mult,
            "seasonal_multiplier": season_mult,
            "loe_multiplier": loe_mult,
        }
    )
    return df


def generate_hierarchy(national: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the national series into region and territory series with independent noise.

    Region and territory shares are fixed weights plus independent weekly
    noise, so a naive bottom-up sum of the parts will not exactly equal the
    national top-down series. That disagreement is deliberate: it is the
    reconciliation problem the demand-supply-planning use case teaches.
    """
    rng = np.random.default_rng(seed + 1)
    region_rows = []
    territory_rows = []
    for region, region_share in REGION_SHARES.items():
        region_nbrx_noise = rng.normal(1.0, 0.05, size=len(national))
        region_trx_noise = rng.normal(1.0, 0.05, size=len(national))
        region_nbrx = national["nbrx"].to_numpy() * region_share * region_nbrx_noise
        region_trx = national["trx"].to_numpy() * region_share * region_trx_noise
        region_rows.append(
            pd.DataFrame(
                {
                    "week_start": national["week_start"],
                    "region": region,
                    "nbrx": region_nbrx,
                    "trx": region_trx,
                }
            )
        )

        territory_shares = rng.dirichlet(np.ones(TERRITORIES_PER_REGION))
        for territory_index, territory_share in enumerate(territory_shares, start=1):
            territory_name = f"{region[:2].upper()}-T{territory_index}"
            territory_nbrx_noise = rng.normal(1.0, 0.08, size=len(national))
            territory_trx_noise = rng.normal(1.0, 0.08, size=len(national))
            territory_rows.append(
                pd.DataFrame(
                    {
                        "week_start": national["week_start"],
                        "region": region,
                        "territory": territory_name,
                        "nbrx": region_nbrx * territory_share * territory_nbrx_noise,
                        "trx": region_trx * territory_share * territory_trx_noise,
                    }
                )
            )
    region_df = pd.concat(region_rows, ignore_index=True)
    territory_df = pd.concat(territory_rows, ignore_index=True)
    return region_df, territory_df


def observed_slice(national: pd.DataFrame) -> pd.DataFrame:
    """The history a reader's forecast can actually see: launch through AS_OF_DATE."""
    return national.loc[national["week_start"] <= AS_OF_DATE].reset_index(drop=True)


def run_generation(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate the full lifecycle: national, observed slice, region, and territory series."""
    national = generate_national_series(seed)
    region, territory = generate_hierarchy(national, seed)
    observed = observed_slice(national)
    return {
        "national_series": national,
        "observed_series": observed,
        "region_series": region,
        "territory_series": territory,
    }


def write_outputs(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write each result table to CSV and record a manifest with provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    for name, table in results.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "seed": SEED,
        "launch_date": str(LAUNCH_DATE.date()),
        "as_of_date": str(AS_OF_DATE.date()),
        "generic_entry_date": str(GENERIC_ENTRY_DATE.date()),
        "series_end_date": str(SERIES_END_DATE.date()),
        "bass_p": BASS_P,
        "bass_q": BASS_Q,
        "bass_ceiling_m": BASS_CEILING_M,
        "persistence_shape": PERSISTENCE_SHAPE,
        "persistence_scale_months": PERSISTENCE_SCALE_MONTHS,
        "refills_per_patient_month": REFILLS_PER_PATIENT_MONTH,
        "loe_residual_share": LOE_RESIDUAL_SHARE,
        "loe_half_life_weeks": LOE_HALF_LIFE_WEEKS,
        "row_counts": {name: len(table) for name, table in results.items()},
        "file_sha256": file_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    output_dir = ROOT / "ch14_forecasting" / "assets" / "generated_outputs"
    results = run_generation()
    write_outputs(results, output_dir)
    print(f"Wrote {len(results)} tables to {output_dir}")
    for name, table in results.items():
        print(f"  {name}: {len(table)} rows")
