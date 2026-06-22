"""Data-quality imperfections used by the Chapter 3 synthetic generators."""

from __future__ import annotations

import random


def claim_lag_days(rng: random.Random) -> int:
    """Return a right-skewed claim receipt lag between 2 and 90 days.

    Used internally to build the two medical snapshot files. The lag value
    is never written to any output CSV.
    """
    lag = round(rng.lognormvariate(2.5, 0.85))
    return max(2, min(lag, 90))


# Pack-size NDC variants absent from ndc_codes.csv — planted to simulate
# codes that entered the claims feed before the reference was refreshed.
UNMAPPED_NDC_VARIANTS: dict[str, str] = {
    "90000-1001-11": "90000-1001-12",
    "90000-1002-11": "90000-1002-12",
    "90000-1003-11": "90000-1003-12",
    "90000-1004-11": "90000-1004-12",
}


def apply_ndc_variation(
    rng: random.Random, ndc_prescribed: str, rate: float = 0.05
) -> tuple[str, str]:
    """Return (ndc_prescribed, ndc_dispensed).

    About 5% of fills get an unmapped pack-size NDC variant as the dispensed
    code, while the prescribed code stays unchanged.
    """
    if rng.random() < rate and ndc_prescribed in UNMAPPED_NDC_VARIANTS:
        return ndc_prescribed, UNMAPPED_NDC_VARIANTS[ndc_prescribed]
    return ndc_prescribed, ndc_prescribed
