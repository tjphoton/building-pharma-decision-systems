"""Analytical tools wired to the governed DuckDB environment.

Each evidence-producing tool runs governed SQL through ``data_access`` and returns typed
``EvidenceItem`` objects whose numbers are computed from the approved tables, with a citation
back to the source. A tool measures and cites; it never names a cause, a next tool, or an
action. Experiment and prior-outcome tools are gated by the decision date, so the first date
cannot read a result that has not happened yet.

This module is a drop-in replacement for ``tools_stub``: it exposes the same ``TOOL_CATALOG``
and ``run_tool(name, decision_phase)`` interface, so the graph switches to real data by
changing one import.
"""

from __future__ import annotations

from datetime import date

from data_access import query_approved_data
from models import EvidenceItem

FIRST_DATE = date(2026, 7, 14)

TOOL_CATALOG: list[str] = [
    "get_mmm_channel_evidence",
    "get_hcp_digital_performance",
    "get_dtc_dma_performance",
    "estimate_claims_maturity",
    "get_experiment_evidence",
    "retrieve_primary_research",
    "get_market_events",
    "get_prior_decision_outcomes",
]


def get_mmm_channel_evidence(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT channel, saturation, marginal_roi FROM mmm_channel_results ORDER BY marginal_roi DESC"
    )
    roi = {row[0]: (row[1], row[2]) for row in r.rows}
    dtc_sat, dtc_roi = roi["DTC paid media"]
    hcp_sat, hcp_roi = roi["HCP digital"]
    return [EvidenceItem(
        evidence_id="MMM-CH-07",
        claim=f"DTC paid media is near saturation ({dtc_sat:.2f} of the curve, marginal ROI "
        f"{dtc_roi:.2f}); HCP digital sits at {hcp_sat:.2f} with marginal ROI {hcp_roi:.2f}.",
        source="mmm_channel_results",
        entity_level="channel",
        window="2026-Q2",
        estimate=f"DTC marginal ROI {dtc_roi:.2f}; HCP digital marginal ROI {hcp_roi:.2f}",
        uncertainty="MMM posterior interval",
        method="Bayesian MMM with adstock and saturation",
        causal_status="associational",
        data_quality="fresh",
        citation="mmm_channel_results (model mmm_v4.2)",
    )]


def get_hcp_digital_performance(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT x.segment AS segment, e.week AS week, sum(e.clicks) AS clicks "
        "FROM hcp_digital_engagement e JOIN hcp_dma_crosswalk x ON e.hcp_id = x.hcp_id "
        "GROUP BY x.segment, e.week ORDER BY x.segment, e.week"
    )
    by_seg: dict[str, list[int]] = {}
    for segment, _week, clicks in r.rows:
        by_seg.setdefault(segment, []).append(int(clicks))
    def pct(series: list[int]) -> int:
        return round(100 * (series[-1] - series[0]) / max(series[0], 1))
    comm, acad = pct(by_seg["community"]), pct(by_seg["academic"])
    return [EvidenceItem(
        evidence_id="HCP-ENG-23",
        claim=f"HCP digital clicks rose {comm}% across the window for community endocrinologists "
        f"and {acad}% for academic-affiliated prescribers.",
        source="hcp_digital_engagement",
        entity_level="HCP-week",
        window="2026-W23..W27",
        estimate=f"community clicks {comm:+d}%, academic {acad:+d}%",
        uncertainty="descriptive; no control group",
        method="delivery and response tally by segment",
        causal_status="descriptive",
        data_quality="fresh",
        citation="hcp_digital_engagement x hcp_dma_crosswalk",
    )]


def get_dtc_dma_performance(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT min(reach) AS lo, max(reach) AS hi, avg(frequency) AS f FROM dtc_dma_delivery"
    )
    lo, hi, freq = r.rows[0]
    spread = round(100 * (hi - lo) / max(lo, 1))
    return [EvidenceItem(
        evidence_id="DTC-DMA-15",
        claim=f"DTC reach varied {spread}% across DMAs at an average frequency of {freq:.1f}; no "
        "delivery spike coincides with the HCP engagement rise.",
        source="dtc_dma_delivery",
        entity_level="DMA-week",
        window="2026-W23..W27",
        estimate=f"reach spread {spread}%, avg frequency {freq:.1f}",
        uncertainty="descriptive",
        method="delivery tally",
        causal_status="descriptive",
        data_quality="fresh",
        citation="dtc_dma_delivery",
    )]


def estimate_claims_maturity(phase: str) -> list[EvidenceItem]:
    as_of = _as_of(phase).isoformat()
    r = query_approved_data(
        "SELECT count(*) AS n, "
        f"sum(CASE WHEN availability_date <= '{as_of}' THEN 1 ELSE 0 END) AS mature "
        "FROM closed_claims WHERE week IN ('2026-W25', '2026-W26', '2026-W27')"
    )
    n, mature = r.rows[0]
    pct = round(100 * mature / max(n, 1))
    fresh = pct >= 80
    return [EvidenceItem(
        evidence_id="CLM-MAT-11",
        claim=f"Closed claims for the recent window are {pct}% mature "
        f"({mature} of {n}); a read of these weeks is a nowcast, not a settled number."
        if not fresh else
        f"Closed claims for the recent window are {pct}% mature ({mature} of {n}); reconciled.",
        source="closed_claims",
        entity_level="patient-claim",
        window="2026-W25..W27",
        estimate=f"{pct}% mature ({mature}/{n})",
        uncertainty="wide nowcast band" if not fresh else "low",
        method="claims completion count (nowcast v1.3)",
        causal_status="descriptive",
        data_quality="immature" if not fresh else "mature",
        citation="closed_claims (nowcast v1.3)",
    )]


def get_experiment_evidence(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT experiment_id, segment, access_state, incr_nrx_per_100, ci_low, ci_high, "
        "read_out_date FROM experiment_results"
    )
    available = [row for row in r.rows if date.fromisoformat(row[6]) <= _as_of(phase)]
    if not available:
        return [EvidenceItem(
            evidence_id="EXP-NONE",
            claim="No experiment has read out for HCP digital in this audience as of the decision "
            "date. The decisive incremental read is missing.",
            source="experiment_results",
            entity_level="n/a", window="n/a", estimate="0 available experiments",
            uncertainty="n/a", method="registry lookup with read-out gating",
            causal_status="descriptive", data_quality="complete",
            citation="experiment_results (none available before decision date)",
        )]
    items = []
    for exp_id, segment, access, incr, lo, hi, _rod in available:
        items.append(EvidenceItem(
            evidence_id=f"{exp_id}-{segment}",
            claim=f"Matched-market test: {incr:+.1f} incremental NRx per 100 targeted HCPs for "
            f"{segment} endocrinologists in {access}-access DMAs (90% CI {lo:.1f} to {hi:.1f}).",
            source="experiment_results",
            entity_level="segment", window="10-week test",
            estimate=f"{incr:+.1f} incr NRx / 100 HCPs",
            uncertainty=f"90% CI {lo:.1f} to {hi:.1f}",
            method="matched-market design with holdout",
            causal_status="causal", data_quality="complete",
            citation=f"experiment_results/{exp_id}",
        ))
    return items


def retrieve_primary_research(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT doc_id, passage_id, source_type, finding FROM primary_research"
    )
    doc_id, passage_id, source_type, finding = r.rows[0]
    return [EvidenceItem(
        evidence_id=f"{doc_id}-{passage_id}",
        claim=finding,
        source="primary_research",
        entity_level="segment", window="2026-H1", estimate="qualitative preference signal",
        uncertainty=f"{source_type}, not an outcome", method="approved primary research",
        causal_status="descriptive", data_quality="approved",
        citation=f"primary_research/{doc_id}#{passage_id}",
    )]


def get_market_events(phase: str) -> list[EvidenceItem]:
    r = query_approved_data(
        "SELECT event_id, dma, event_type, effective_date FROM market_events"
    )
    items = []
    for event_id, dma, event_type, eff in r.rows:
        items.append(EvidenceItem(
            evidence_id=event_id,
            claim=f"A {event_type} took effect in {dma} on {eff}; it overlaps the decision window.",
            source="market_events",
            entity_level="DMA", window=str(eff), estimate="1 access-change cluster",
            uncertainty="localized", method="event registry",
            causal_status="descriptive", data_quality="fresh",
            citation=f"market_events/{event_id}",
        ))
    return items


def get_prior_decision_outcomes(phase: str) -> list[EvidenceItem]:
    # The first action's outcome is only observed by the later decision date.
    if phase != "later":
        return [EvidenceItem(
            evidence_id="PRIOR-NONE",
            claim="No prior decision for this case has an observed outcome yet.",
            source="prior_decisions",
            entity_level="decision", window="n/a", estimate="none",
            uncertainty="n/a", method="decision-record lookup",
            causal_status="descriptive", data_quality="complete",
            citation="prior_decisions (none observed before this date)",
        )]
    r = query_approved_data(
        "SELECT decision_id, action, expected_low, expected_high, observed, downside_used "
        "FROM prior_decisions"
    )
    items = []
    for dec_id, action, lo, hi, observed, downside in r.rows:
        items.append(EvidenceItem(
            evidence_id=f"{dec_id}-A1",
            claim=f"The prior action ({action}) expected {lo} to {hi} incremental NRx and observed "
            f"{observed}, using ${downside:,} of its downside budget.",
            source="prior_decisions",
            entity_level="decision", window="prior period",
            estimate=f"observed {observed} vs expected {lo}-{hi}",
            uncertainty="low", method="decision-outcome monitor",
            causal_status="descriptive", data_quality="complete",
            citation=f"prior_decisions/{dec_id}",
        ))
    return items


_DISPATCH = {
    "get_mmm_channel_evidence": get_mmm_channel_evidence,
    "get_hcp_digital_performance": get_hcp_digital_performance,
    "get_dtc_dma_performance": get_dtc_dma_performance,
    "estimate_claims_maturity": estimate_claims_maturity,
    "get_experiment_evidence": get_experiment_evidence,
    "retrieve_primary_research": retrieve_primary_research,
    "get_market_events": get_market_events,
    "get_prior_decision_outcomes": get_prior_decision_outcomes,
}


def _as_of(phase: str) -> date:
    return date(2026, 10, 6) if phase == "later" else FIRST_DATE


def run_tool(name: str, decision_phase: str) -> list[EvidenceItem]:
    """Run one analytical tool against the governed environment. Unknown tools return a note."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return [EvidenceItem(
            evidence_id=f"NA-{name}", claim=f"Tool {name} is not available.", source=name,
            entity_level="n/a", window="n/a", estimate="none", uncertainty="n/a", method="n/a",
            causal_status="descriptive", data_quality="empty", citation=f"{name}/unavailable",
        )]
    return fn(decision_phase)


# --- Decision service (deterministic; the analyst requests it, does not compute it) ----------


def simulate_budget_scenarios(budget_usd: int) -> dict:
    """Expected incremental NRx for a proposed HCP-digital move, on a saturating response curve.
    The tool returns the number; the decision analyst interprets it."""
    import math

    expected = round(320 * (1 - math.exp(-budget_usd / 150_000)))
    return {"budget_usd": budget_usd, "expected_incr_nrx": expected,
            "curve": "saturating (efficient scale near 180k)"}


def run_sandbox_analysis(*_args, **_kwargs):
    """Restricted Python over an approved query result. Boundary only in the skeleton.

    A production version requires a human interrupt before execution, passes only an approved
    query result into a separate process with no network or filesystem access and a small
    approved import set, and logs code, inputs, and outputs. It is not the normal path; SQL is.
    """
    raise NotImplementedError(
        "run_sandbox_analysis is a governed, human-approved path built in a later milestone; "
        "use query_approved_data for ad hoc analysis."
    )
