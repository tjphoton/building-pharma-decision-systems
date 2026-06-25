"""Line of therapy construction for Chapter 5.

A line of therapy (LoT) is a regimen-level view of the treatment sequence.
Episodes answer "how long was the patient on each product"; lines answer
"which regimen was this, and what number in the sequence". The construction
rules are explicit module constants so every published number can name the
ruler that produced it.

Base sequencing rules
---------------------
Therapy index    First paid market-basket fill on or after the diagnosis index.
Washout          New-to-therapy requires no basket fill in the WASHOUT_DAYS
                 before the therapy index. Patients who fail are prevalent
                 users and are excluded from line numbering.
Regimen window   Distinct basket products first filled within
                 REGIMEN_WINDOW_DAYS of the line start form one combination
                 regimen rather than separate lines.
Allowable gap    A line stays open while the next regimen fill arrives within
                 ALLOWABLE_GAP_DAYS after the current supply runs out.
Addition         A product outside the current regimen, started while at least
                 one regimen product still has supply, advances the line; the
                 new regimen keeps the active backbone plus the new product.
Switch           A product outside the current regimen, started when no
                 regimen product has supply remaining, advances the line; the
                 new regimen is the new product.
Restart          A regimen product returning after the allowable gap closes
                 the prior line as discontinued and opens a new line on the
                 same product (restart_advances_line=False instead resumes
                 the open line; see the sensitivity grid).
Discontinuation  A line whose supply ends, plus the allowable gap, before
                 follow-up ends is discontinued; otherwise the patient is
                 censored on that line.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

WASHOUT_DAYS = 180
REGIMEN_WINDOW_DAYS = 30
ALLOWABLE_GAP_DAYS = 60


def therapy_index_dates(
    paid_basket_fills: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """First paid basket fill on or after each cohort patient's diagnosis index."""

    fills = paid_basket_fills.merge(
        cohort[["patient_id", "index_date", "followup_end"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    post_index = fills.loc[
        fills["date_of_service"].between(
            fills["index_date"], fills["followup_end"], inclusive="both"
        )
    ]
    return (
        post_index.groupby("patient_id", as_index=False)["date_of_service"]
        .min()
        .rename(columns={"date_of_service": "therapy_index"})
    )


def apply_washout(
    paid_basket_fills: pd.DataFrame,
    therapy_index: pd.DataFrame,
    washout_days: int = WASHOUT_DAYS,
) -> pd.DataFrame:
    """Flag each initiator as new to therapy or a prevalent user.

    A patient is new to therapy when no paid basket fill exists in the
    washout_days strictly before the therapy index. With washout_days=0 the
    rule is disabled and every initiator counts as new (the naive analysis).
    """

    flagged = therapy_index.copy()
    if washout_days <= 0:
        flagged["new_to_therapy"] = True
        return flagged

    prior = paid_basket_fills.merge(therapy_index, on="patient_id", how="inner")
    window_start = prior["therapy_index"] - pd.to_timedelta(washout_days, unit="D")
    in_washout = prior.loc[
        prior["date_of_service"].ge(window_start)
        & prior["date_of_service"].lt(prior["therapy_index"])
    ]
    prevalent_ids = set(in_washout["patient_id"])
    flagged["new_to_therapy"] = ~flagged["patient_id"].isin(prevalent_ids)
    return flagged


def construct_lines_of_therapy(
    paid_basket_fills: pd.DataFrame,
    cohort: pd.DataFrame,
    washout_days: int = WASHOUT_DAYS,
    regimen_window_days: int = REGIMEN_WINDOW_DAYS,
    allowable_gap_days: int = ALLOWABLE_GAP_DAYS,
    restart_advances_line: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per patient-line from paid basket fills.

    Returns (lines, initiators). Lines carry the regimen, dates, fill count,
    how the line was entered (entry_reason) and how it ended (end_reason).
    Initiators carries one row per treated cohort patient with the
    new_to_therapy flag, so the attrition from the washout rule stays visible.
    """

    index_dates = therapy_index_dates(paid_basket_fills, cohort)
    initiators = apply_washout(paid_basket_fills, index_dates, washout_days)

    eligible = initiators.loc[initiators["new_to_therapy"]]
    fills = paid_basket_fills.merge(
        eligible[["patient_id", "therapy_index"]], on="patient_id", how="inner"
    ).merge(
        cohort[["patient_id", "followup_end"]], on="patient_id", how="inner"
    )
    fills = fills.loc[
        fills["date_of_service"].ge(fills["therapy_index"])
        & fills["date_of_service"].le(fills["followup_end"])
    ].copy()
    fills["supply_end"] = fills["date_of_service"] + pd.to_timedelta(
        fills["days_supply"].clip(lower=1) - 1, unit="D"
    )
    fills = fills.sort_values(["patient_id", "date_of_service", "product_name"])

    window = pd.Timedelta(days=regimen_window_days)
    gap = pd.Timedelta(days=allowable_gap_days)
    line_rows: list[dict] = []

    for patient_id, patient_fills in fills.groupby("patient_id", sort=True):
        followup_end = patient_fills["followup_end"].iloc[0]
        line_number = 0
        line: dict | None = None
        product_supply_end: dict[str, pd.Timestamp] = {}

        def close_line(reason: str) -> None:
            line["end_reason"] = reason
            line["regimen"] = " + ".join(sorted(line["products"]))
            line_rows.append(line)

        def open_line(row, entry_reason: str, backbone: set[str]) -> None:
            nonlocal line, line_number, product_supply_end
            line_number += 1
            line = {
                "patient_id": patient_id,
                "line_number": line_number,
                "products": set(backbone) | {row.product_name},
                "line_start": row.date_of_service,
                "line_end": row.supply_end,
                "fill_count": 1,
                "entry_reason": entry_reason,
            }
            product_supply_end = {
                product: end
                for product, end in product_supply_end.items()
                if product in backbone and end >= row.date_of_service
            }
            product_supply_end[row.product_name] = row.supply_end

        for row in patient_fills.itertuples(index=False):
            if line is None:
                open_line(row, "Initial therapy", set())
                continue

            in_regimen = row.product_name in line["products"]
            within_gap = row.date_of_service <= line["line_end"] + gap
            within_window = row.date_of_service <= line["line_start"] + window

            if within_window and not in_regimen:
                # Joins the combination regimen rather than advancing the line
                line["products"].add(row.product_name)
                line["line_end"] = max(line["line_end"], row.supply_end)
                line["fill_count"] += 1
                product_supply_end[row.product_name] = row.supply_end
            elif in_regimen and within_gap:
                line["line_end"] = max(line["line_end"], row.supply_end)
                line["fill_count"] += 1
                product_supply_end[row.product_name] = max(
                    product_supply_end.get(row.product_name, row.supply_end),
                    row.supply_end,
                )
            elif in_regimen and not within_gap:
                if restart_advances_line:
                    close_line("Discontinued")
                    open_line(row, "Restart", set())
                else:
                    line["line_end"] = max(line["line_end"], row.supply_end)
                    line["fill_count"] += 1
                    product_supply_end[row.product_name] = row.supply_end
            elif not in_regimen and within_gap:
                backbone = {
                    product
                    for product in line["products"]
                    if product_supply_end.get(product, pd.Timestamp.min)
                    >= row.date_of_service
                }
                reason = "Addition" if backbone else "Switch"
                close_line(reason)
                open_line(row, reason, backbone)
            else:
                close_line("Discontinued")
                open_line(row, "Switch", set())

        if line is not None:
            if line["line_end"] + gap < followup_end:
                close_line("Discontinued")
            else:
                close_line("Censored")

    columns = [
        "patient_id",
        "line_number",
        "regimen",
        "line_start",
        "line_end",
        "fill_count",
        "entry_reason",
        "end_reason",
    ]
    lines = pd.DataFrame(line_rows)
    if lines.empty:
        return pd.DataFrame(columns=columns), initiators
    lines["line_days"] = (lines["line_end"] - lines["line_start"]).dt.days + 1
    return lines[columns + ["line_days"]].reset_index(drop=True), initiators


def lot_entry_shares(lines: pd.DataFrame, product: str = "Roventra") -> pd.DataFrame:
    """Where a product's line entries sit in the sequence: line 1 versus later."""

    contains = lines.loc[lines["regimen"].str.contains(product, regex=False)]
    summary = (
        contains.assign(position=contains["line_number"].le(1).map(
            {True: "Line 1", False: "Line 2 or later"}
        ))
        .groupby("position", as_index=False)
        .agg(line_entries=("patient_id", "count"))
    )
    summary["share"] = (summary["line_entries"] / summary["line_entries"].sum()).round(3)
    return summary


def lot_sensitivity(
    paid_basket_fills: pd.DataFrame,
    cohort: pd.DataFrame,
    washout_grid: Iterable[int] = (0, 90, 180),
    window_grid: Iterable[int] = (14, 30, 45),
    gap_grid: Iterable[int] = (30, 60, 90),
    product: str = "Roventra",
) -> pd.DataFrame:
    """One-dimensional sweeps around the base rules.

    Each row varies a single rule while the other two stay at base, so the
    reader can see which published numbers each ruler moves.
    """

    base = {
        "washout_days": WASHOUT_DAYS,
        "regimen_window_days": REGIMEN_WINDOW_DAYS,
        "allowable_gap_days": ALLOWABLE_GAP_DAYS,
    }
    runs: list[dict] = []
    for washout in washout_grid:
        runs.append({**base, "washout_days": washout, "varied": "washout"})
    for window in window_grid:
        runs.append({**base, "regimen_window_days": window, "varied": "regimen window"})
    for gap_days in gap_grid:
        runs.append({**base, "allowable_gap_days": gap_days, "varied": "allowable gap"})

    rows: list[dict] = []
    seen: set[tuple] = set()
    for run in runs:
        key = (
            run["washout_days"],
            run["regimen_window_days"],
            run["allowable_gap_days"],
        )
        if key in seen:
            continue
        seen.add(key)
        lines, initiators = construct_lines_of_therapy(
            paid_basket_fills,
            cohort,
            washout_days=run["washout_days"],
            regimen_window_days=run["regimen_window_days"],
            allowable_gap_days=run["allowable_gap_days"],
        )
        entries = lot_entry_shares(lines, product=product)
        line1_share = float(
            entries.loc[entries["position"].eq("Line 1"), "share"].sum()
        )
        first_lines = lines.loc[lines["line_number"].eq(1)]
        rows.append(
            {
                "varied": run["varied"],
                "washout_days": run["washout_days"],
                "regimen_window_days": run["regimen_window_days"],
                "allowable_gap_days": run["allowable_gap_days"],
                "new_to_therapy_patients": int(initiators["new_to_therapy"].sum()),
                "lines_per_patient": round(
                    len(lines) / lines["patient_id"].nunique(), 2
                ),
                "combination_line1_share": round(
                    first_lines["regimen"].str.contains(" + ", regex=False).mean(), 3
                ),
                "line1_discontinued_share": round(
                    first_lines["end_reason"].eq("Discontinued").mean(), 3
                ),
                f"{product.lower()}_line1_entry_share": round(line1_share, 3),
            }
        )
    return pd.DataFrame(rows)
