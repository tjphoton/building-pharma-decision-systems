"""Build the Chapter 16 governed analytics environment as local DuckDB databases.

Produces two files under ``ch16_decision/assets/data``:

* ``analytics.duckdb`` - the approved environment the agents may query read-only: a small
  metadata catalog plus production-shaped synthetic commercial tables.
* ``eval_truth.duckdb`` - hidden response truth for evaluation only. It is a separate file
  and is never attached by the agent connection, so the planted effect is unreachable from
  any approved query.

The build is deterministic (fixed seed), so readers regenerate the same environment.
Run from the repository root:

    uv run python ch16_decision/scripts/build_database.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 16
DATA_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"
ANALYTICS_DB = DATA_DIR / "analytics.duckdb"
EVAL_DB = DATA_DIR / "eval_truth.duckdb"

WEEKS = ["2026-W23", "2026-W24", "2026-W25", "2026-W26", "2026-W27"]
DMAS = ["DMA_ATL", "DMA_BOS", "DMA_CHI", "DMA_DAL", "DMA_DEN", "DMA_SEA"]
# Two DMAs have unstable access; the planted effect lives only in stable-access markets.
UNSTABLE_ACCESS = {"DMA_DAL", "DMA_DEN"}
FIRST_DATE = pd.Timestamp("2026-07-14")


def _crosswalk(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(48):
        hcp_id = f"H{i:04d}"
        segment = "community" if i % 3 != 0 else "academic"  # ~2/3 community
        dma = DMAS[i % len(DMAS)]
        access = "unstable" if dma in UNSTABLE_ACCESS else "stable"
        rows.append({"hcp_id": hcp_id, "segment": segment, "dma": dma, "access_state": access})
    return pd.DataFrame(rows)


def _engagement(rng: np.random.Generator, cw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, hcp in cw.iterrows():
        base = int(rng.integers(8, 14))
        for w, week in enumerate(WEEKS):
            # Community engagement climbs over the window; academic stays flat.
            lift = 1.0 + (0.10 * w if hcp.segment == "community" else 0.0)
            emails = base
            opens = int(round(emails * rng.uniform(0.35, 0.55) * lift))
            clicks = int(round(opens * rng.uniform(0.30, 0.50) * lift))
            rows.append({
                "hcp_id": hcp.hcp_id, "week": week, "emails_delivered": emails,
                "opens": min(opens, emails), "clicks": clicks,
            })
    return pd.DataFrame(rows)


def _dtc(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for dma in DMAS:
        base_reach = int(rng.integers(40000, 70000))
        for week in WEEKS:
            rows.append({
                "dma": dma, "week": week,
                "impressions": int(base_reach * rng.uniform(3.0, 4.0)),
                "reach": int(base_reach * rng.uniform(0.95, 1.05)),
                "frequency": round(rng.uniform(2.8, 3.4), 2),
                "spend_usd": int(rng.integers(18000, 30000)),
            })
    return pd.DataFrame(rows)


def _claims(rng: np.random.Generator, cw: pd.DataFrame) -> pd.DataFrame:
    """Closed claims with a reporting lag. Recent weeks are only partly mature as of the
    first decision date."""
    rows = []
    claim_no = 0
    week_dates = {w: FIRST_DATE - pd.Timedelta(weeks=(len(WEEKS) - i))
                  for i, w in enumerate(WEEKS)}
    for _, hcp in cw.iterrows():
        n = int(rng.integers(3, 7))
        for _ in range(n):
            week = WEEKS[int(rng.integers(0, len(WEEKS)))]
            service = week_dates[week] + pd.Timedelta(days=int(rng.integers(0, 7)))
            lag = int(rng.integers(14, 60))  # claims arrive 2 to 8 weeks late
            availability = service + pd.Timedelta(days=lag)
            rows.append({
                "claim_id": f"C{claim_no:05d}", "hcp_id": hcp.hcp_id, "dma": hcp.dma,
                "week": week, "service_date": service.date().isoformat(),
                "availability_date": availability.date().isoformat(),
                "is_mature": bool(availability <= FIRST_DATE), "nrx": 1,
            })
            claim_no += 1
    return pd.DataFrame(rows)


def _rx_weekly(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for dma in DMAS:
        for week in WEEKS:
            rows.append({
                "dma": dma, "week": week,
                "nrx": int(rng.integers(60, 110)), "trx": int(rng.integers(180, 320)),
            })
    return pd.DataFrame(rows)


def _static_tables() -> dict[str, pd.DataFrame]:
    mmm = pd.DataFrame([
        {"channel": "DTC paid media", "saturation": 0.92, "marginal_roi": 0.31,
         "model_version": "mmm_v4.2"},
        {"channel": "HCP digital", "saturation": 0.44, "marginal_roi": 1.70,
         "model_version": "mmm_v4.2"},
        {"channel": "Field", "saturation": 0.61, "marginal_roi": 0.95, "model_version": "mmm_v4.2"},
        {"channel": "Email", "saturation": 0.55, "marginal_roi": 1.10, "model_version": "mmm_v4.2"},
    ])
    experiments = pd.DataFrame([
        {"experiment_id": "EXP-2026-31", "segment": "community", "access_state": "stable",
         "design": "matched-market", "incr_nrx_per_100": 2.1, "ci_low": 0.9, "ci_high": 3.3,
         "status": "completed", "read_out_date": "2026-10-01"},
        {"experiment_id": "EXP-2026-31", "segment": "academic", "access_state": "stable",
         "design": "matched-market", "incr_nrx_per_100": 0.1, "ci_low": -0.8, "ci_high": 1.0,
         "status": "completed", "read_out_date": "2026-10-01"},
    ])
    research = pd.DataFrame([
        {"doc_id": "PR-114", "passage_id": "s3", "source_type": "respondent opinion",
         "finding": "Community practices report higher reliance on digital touchpoints for "
         "dosing guidance than academic centers."},
    ])
    events = pd.DataFrame([
        {"event_id": "EVT-DMA-4", "dma": "DMA_DAL", "event_type": "formulary access change",
         "effective_date": "2026-06-01"},
    ])
    priors = pd.DataFrame([
        {"decision_id": "DEC-2026-0714", "action": "staged matched-market test",
         "expected_low": 123, "expected_high": 349, "observed": 248, "downside_used": 0},
    ])
    model_registry = pd.DataFrame([
        {"model": "MMM", "version": "mmm_v4.2", "run_date": "2026-06-30", "status": "current"},
        {"model": "claims_nowcast", "version": "v1.3", "run_date": "2026-07-10", "status": "current"},
    ])
    return {
        "mmm_channel_results": mmm, "experiment_results": experiments,
        "primary_research": research, "market_events": events, "prior_decisions": priors,
        "model_registry": model_registry,
    }


def _catalog() -> pd.DataFrame:
    rows = [
        ("hcp_digital_engagement", "HCP-week", "2026-07-11", "complete",
         "measurement, targeting", "internal promotional", "HCP email delivery and response"),
        ("dtc_dma_delivery", "DMA-week", "2026-07-11", "complete",
         "measurement", "media vendor", "Consumer reach and frequency by media market"),
        ("closed_claims", "patient-claim", "2026-07-11", "partial (recent weeks maturing)",
         "measurement", "licensed claims", "Adjudicated prescription claims with reporting lag"),
        ("rx_weekly", "DMA-week", "2026-07-11", "complete",
         "measurement", "derived", "Weekly NRx and TRx by market"),
        ("mmm_channel_results", "channel", "2026-06-30", "complete",
         "planning", "model output", "MMM saturation and marginal ROI by channel"),
        ("experiment_results", "segment", "2026-10-01", "complete after read-out",
         "measurement", "experiment registry", "Matched-market incremental NRx reads"),
        ("primary_research", "segment", "2026-05-15", "complete",
         "hypothesis", "approved research", "Qualitative findings with stable passage IDs"),
        ("market_events", "DMA", "2026-07-11", "complete",
         "control", "internal", "Formulary, competitor, and access events"),
        ("prior_decisions", "decision", "2026-07-11", "complete",
         "learning", "decision record", "Prior approved actions and observed outcomes"),
        ("hcp_dma_crosswalk", "HCP", "2026-07-01", "complete",
         "entity mapping", "reference", "HCP to DMA, segment, and access-state mapping"),
    ]
    return pd.DataFrame(rows, columns=[
        "data_product", "entity_level", "refresh_date", "completeness",
        "permitted_use", "source_type", "description",
    ])


def _write(db_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            con.register("df_tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df_tmp")
            con.unregister("df_tmp")
    finally:
        con.close()


def build() -> dict[str, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (ANALYTICS_DB, EVAL_DB):
        path.unlink(missing_ok=True)

    rng = np.random.default_rng(SEED)
    cw = _crosswalk(rng)
    approved = {
        "data_products": _catalog(),
        "hcp_dma_crosswalk": cw,
        "hcp_digital_engagement": _engagement(rng, cw),
        "dtc_dma_delivery": _dtc(rng),
        "closed_claims": _claims(rng, cw),
        "rx_weekly": _rx_weekly(rng),
        **_static_tables(),
    }
    _write(ANALYTICS_DB, approved)

    # Hidden response truth: the planted heterogeneous effect, evaluator-only.
    truth = pd.DataFrame([
        {"segment": "community", "access_state": "stable", "true_incr_nrx_per_100": 2.1},
        {"segment": "community", "access_state": "unstable", "true_incr_nrx_per_100": 0.2},
        {"segment": "academic", "access_state": "stable", "true_incr_nrx_per_100": 0.1},
        {"segment": "academic", "access_state": "unstable", "true_incr_nrx_per_100": 0.0},
    ])
    _write(EVAL_DB, {"response_truth": truth})

    return {name: len(df) for name, df in approved.items()}


if __name__ == "__main__":
    counts = build()
    print(f"Built {ANALYTICS_DB.relative_to(Path.cwd())}")
    for name, n in counts.items():
        print(f"  {name:26} {n:5d} rows")
    print(f"Hidden truth written to {EVAL_DB.relative_to(Path.cwd())} (never attached by agents)")
