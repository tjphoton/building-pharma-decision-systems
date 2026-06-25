"""PDC, MPR, and gap analysis adherence metrics for Chapter 5."""

from __future__ import annotations

import pandas as pd


def _covered_days(
    fills: list[tuple[pd.Timestamp, int, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """Count covered days after same-product early-refill carryover.

    Supply from an early refill starts after the prior supply for that product
    is exhausted. Adjusted intervals are then combined across products.
    """
    intervals: list[tuple[int, int]] = []
    origin = start
    window_len = (end - start).days + 1
    by_product: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for fill_date, days_supply, product in fills:
        by_product.setdefault(product, []).append((fill_date, days_supply))

    for product_fills in by_product.values():
        prior_end: pd.Timestamp | None = None
        for fill_date, days_supply in sorted(product_fills):
            effective_start = fill_date
            if prior_end is not None and fill_date <= prior_end:
                effective_start = prior_end + pd.Timedelta(days=1)
            effective_end = effective_start + pd.Timedelta(
                days=max(int(days_supply), 1) - 1
            )
            prior_end = effective_end
            s = max(0, (effective_start - origin).days)
            e = min(window_len - 1, (effective_end - origin).days)
            if s <= e:
                intervals.append((s, e))
    if not intervals:
        return 0
    intervals.sort()
    covered = 0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            covered += cur_end - cur_start + 1
            cur_start, cur_end = s, e
    covered += cur_end - cur_start + 1
    return covered


def compute_adherence_metrics(
    paid_events: pd.DataFrame,
    cohort: pd.DataFrame,
    observation_days: int = 365,
    minimum_window_days: int = 90,
    product_name: str | None = None,
    product_by_patient: dict[str, str] | None = None,
    scope_label: str | None = None,
) -> pd.DataFrame:
    """Compute PDC and MPR for each treated patient over a fixed observation window.

    PDC (Proportion of Days Covered) counts unique days with supply available,
    preventing overlapping refills from being counted twice. It is the CMS/URAC
    standard for chronic-therapy adherence reporting.

    MPR (Medication Possession Ratio) sums raw days-supply and may exceed 1.0
    when patients stockpile. Both are computed for comparison. Patients whose
    observable window is shorter than minimum_window_days are excluded, since
    a possession ratio over a few days carries no signal.

    Parameters
    ----------
    paid_events:
        Paid pharmacy claims filtered to the treatment cohort.
    cohort:
        Patient cohort with index_date and followup_end columns.
    observation_days:
        Length of the adherence observation window starting from the first fill
        date. Defaults to 365 days (one year).
    product_name:
        If provided, restrict adherence calculation to this product only.
        If None, compute across all market products combined.
    """
    events = paid_events.copy()
    if product_name is not None:
        events = events.loc[events["product_name"].eq(product_name)]

    if events.empty:
        return pd.DataFrame(
            columns=[
                "patient_id", "product_scope", "observation_start",
                "observation_end", "observation_days", "fills",
                "sum_days_supply", "pdc", "mpr", "mpr_capped",
                "adherent_pdc", "adherent_mpr",
            ]
        )

    followup_end_map = cohort.set_index("patient_id")["followup_end"]
    rows: list[dict] = []

    for patient_id, patient_fills in events.groupby("patient_id", sort=True):
        selected_product = None
        if product_by_patient is not None:
            selected_product = product_by_patient.get(patient_id)
            if selected_product is None:
                continue
            patient_fills = patient_fills.loc[
                patient_fills["product_name"].eq(selected_product)
            ]
            if patient_fills.empty:
                continue

        observation_start = patient_fills["date_of_service"].min()
        raw_end = observation_start + pd.Timedelta(days=observation_days - 1)
        observation_end = min(raw_end, followup_end_map.get(patient_id, raw_end))
        window_days = (observation_end - observation_start).days + 1
        if window_days < minimum_window_days:
            # A possession ratio over a few days is noise, not adherence.
            continue

        within_window = patient_fills.loc[
            patient_fills["date_of_service"].between(observation_start, observation_end)
        ]
        fills_list = list(
            zip(
                within_window["date_of_service"],
                within_window["days_supply"].astype(int),
                within_window["product_name"],
            )
        )
        sum_days_supply = int(within_window["days_supply"].astype(int).sum())
        pdc = min(_covered_days(fills_list, observation_start, observation_end) / window_days, 1.0)
        mpr_raw = sum_days_supply / window_days
        mpr_capped = min(mpr_raw, 1.0)

        rows.append(
            {
                "patient_id": patient_id,
                "product_scope": (
                    scope_label
                    or selected_product
                    or product_name
                    or "All market products"
                ),
                "observation_start": observation_start,
                "observation_end": observation_end,
                "observation_days": window_days,
                "fills": len(within_window),
                "sum_days_supply": sum_days_supply,
                "pdc": round(pdc, 4),
                "mpr": round(mpr_raw, 4),
                "mpr_capped": round(mpr_capped, 4),
                "adherent_pdc": pdc >= 0.80,
                "adherent_mpr": mpr_capped >= 0.80,
            }
        )

    return pd.DataFrame(rows)


def adherence_summary(
    adherence: pd.DataFrame,
    cohort: pd.DataFrame,
    by_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize PDC and MPR adherence rates with optional stratification.

    Parameters
    ----------
    adherence:
        Output of compute_adherence_metrics.
    cohort:
        Patient cohort for joining stratification attributes (region, payer_id).
    by_columns:
        Columns from the cohort to stratify by. Defaults to ['region'].
    """
    if by_columns is None:
        by_columns = ["region"]

    joined = adherence.merge(
        cohort[["patient_id"] + by_columns],
        on="patient_id",
        how="left",
    )
    agg = (
        joined.groupby(by_columns, as_index=False)
        .agg(
            treated_patients=("patient_id", "nunique"),
            mean_pdc=("pdc", "mean"),
            median_pdc=("pdc", "median"),
            adherent_pdc_rate=("adherent_pdc", "mean"),
            mean_mpr_capped=("mpr_capped", "mean"),
            adherent_mpr_rate=("adherent_mpr", "mean"),
            mean_fills=("fills", "mean"),
        )
        .round(4)
    )
    overall_row: dict = {col: "All" for col in by_columns}
    overall_row.update(
        {
            "treated_patients": adherence["patient_id"].nunique(),
            "mean_pdc": adherence["pdc"].mean().round(4),
            "median_pdc": adherence["pdc"].median().round(4),
            "adherent_pdc_rate": adherence["adherent_pdc"].mean().round(4),
            "mean_mpr_capped": adherence["mpr_capped"].mean().round(4),
            "adherent_mpr_rate": adherence["adherent_mpr"].mean().round(4),
            "mean_fills": adherence["fills"].mean().round(1),
        }
    )
    overall = pd.DataFrame([overall_row])
    return pd.concat([overall, agg], ignore_index=True)
