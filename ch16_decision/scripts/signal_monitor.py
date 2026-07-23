"""Deterministic signal monitor for the always-on Roventra decision system.

The monitor runs governed read-only queries over the approved analytics environment and emits
a typed :class:`SignalEvent` when a configured pattern holds. It reports an unusual pattern
only: it never diagnoses the cause, selects a tool, or recommends a budget move. A marketer
confirms the candidate into a :class:`DecisionRequest` through :func:`confirm_signal`.

Teaching rule (Section 19.2). Open a candidate when both hold across the 5-week window:

1. Community HCP digital clicks rise at least 30% from the first to the last monitor week.
2. Recent closed claims are below 80% mature, or aggregate weekly NRx growth is no more than 5%.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

from config import FIRST_DECISION_DATE, TRIGGER, TRIGGER_RULE_VERSION
from data_access import query_approved_data
from models import DecisionRequest, SignalEvent


class MonitorError(RuntimeError):
    """Raised when a source needed by the monitor is missing or unreadable."""


@dataclass
class TriggerReadout:
    """The operands and result of the trigger rule, printed in the notebook and manuscript."""

    first_week: str
    last_week: str
    community_clicks_first: int
    community_clicks_last: int
    engagement_rise: float
    claims_maturity: float
    nrx_growth: float
    fires: bool

    def as_lines(self) -> list[str]:
        return [
            f"community clicks {self.first_week}->{self.last_week}: "
            f"{self.community_clicks_first} -> {self.community_clicks_last} "
            f"({self.engagement_rise:+.0%})",
            f"rise >= {TRIGGER['min_engagement_rise']:.0%}? "
            f"{self.engagement_rise >= TRIGGER['min_engagement_rise']}",
            f"recent claims maturity: {self.claims_maturity:.0%} "
            f"(< {TRIGGER['claims_maturity_ceiling']:.0%}? "
            f"{self.claims_maturity < TRIGGER['claims_maturity_ceiling']})",
            f"weekly NRx growth: {self.nrx_growth:+.0%} "
            f"(<= {TRIGGER['max_nrx_growth']:.0%}? {self.nrx_growth <= TRIGGER['max_nrx_growth']})",
            f"candidate opens: {self.fires}",
        ]


def _community_clicks_by_week() -> dict[str, int]:
    try:
        result = query_approved_data(
            "SELECT e.week AS week, sum(e.clicks) AS clicks "
            "FROM hcp_digital_engagement e JOIN hcp_dma_crosswalk x ON e.hcp_id = x.hcp_id "
            "WHERE x.segment = 'community' GROUP BY e.week ORDER BY e.week"
        )
    except Exception as error:  # pragma: no cover - surfaced as a monitor error
        raise MonitorError(f"engagement source unavailable: {error}") from error
    if not result.rows:
        raise MonitorError("hcp_digital_engagement returned no community rows.")
    return {week: int(clicks) for week, clicks in result.rows}


def _claims_maturity(as_of: date) -> float:
    result = query_approved_data(
        "SELECT count(*) AS n, "
        f"sum(CASE WHEN availability_date <= '{as_of.isoformat()}' THEN 1 ELSE 0 END) AS mature "
        "FROM closed_claims WHERE week IN ('2026-W25', '2026-W26', '2026-W27')"
    )
    total, mature = result.rows[0]
    if not total:
        raise MonitorError("closed_claims returned no rows for the recent window.")
    return mature / total


def _nrx_growth() -> float:
    result = query_approved_data(
        "SELECT week, sum(nrx) AS nrx FROM rx_weekly GROUP BY week ORDER BY week"
    )
    if not result.rows:
        raise MonitorError("rx_weekly returned no rows.")
    first, last = int(result.rows[0][1]), int(result.rows[-1][1])
    return (last - first) / max(first, 1)


def _signal_id(measurement_window: str) -> str:
    """A stable ID from the pattern definition, not its observed value, so repeated
    evaluation of the same window yields the same signal and never duplicates a case."""
    basis = "|".join([
        TRIGGER["brand"], TRIGGER["metric"], TRIGGER["population"],
        TRIGGER["geography"], measurement_window, TRIGGER_RULE_VERSION,
    ])
    return "SIG-" + hashlib.sha256(basis.encode()).hexdigest()[:10]


def read_trigger(as_of_date: date) -> TriggerReadout:
    """Compute the trigger operands and whether the rule fires, without building an event."""
    weeks = TRIGGER["monitor_weeks"]
    clicks = _community_clicks_by_week()
    missing = [w for w in weeks if w not in clicks]
    if missing:
        raise MonitorError(f"engagement source missing weeks: {missing}")
    first_week, last_week = weeks[0], weeks[-1]
    first, last = clicks[first_week], clicks[last_week]
    engagement_rise = (last - first) / max(first, 1)
    claims_maturity = _claims_maturity(as_of_date)
    nrx_growth = _nrx_growth()

    engagement_ok = engagement_rise >= TRIGGER["min_engagement_rise"]
    nrx_unsettled = (
        claims_maturity < TRIGGER["claims_maturity_ceiling"]
        or nrx_growth <= TRIGGER["max_nrx_growth"]
    )
    return TriggerReadout(
        first_week=first_week,
        last_week=last_week,
        community_clicks_first=first,
        community_clicks_last=last,
        engagement_rise=engagement_rise,
        claims_maturity=claims_maturity,
        nrx_growth=nrx_growth,
        fires=engagement_ok and nrx_unsettled,
    )


def evaluate_hcp_digital_signal(as_of_date: date | None = None) -> SignalEvent | None:
    """Return a candidate :class:`SignalEvent` when the trigger fires, else ``None``."""
    as_of = as_of_date or FIRST_DECISION_DATE
    readout = read_trigger(as_of)
    if not readout.fires:
        return None
    window = f"{readout.first_week}..{readout.last_week.split('-')[-1]}"
    return SignalEvent(
        signal_id=_signal_id(window),
        brand=TRIGGER["brand"],
        metric=TRIGGER["metric"],
        observed_value=float(readout.community_clicks_last),
        expected_low=float(readout.community_clicks_first),
        expected_high=float(round(readout.community_clicks_first * 1.1)),
        measurement_window=window,
        evidence_date=as_of,
        population=TRIGGER["population"],
        geography=TRIGGER["geography"],
        source=TRIGGER["source"],
        status="candidate",
        trigger_rule_version=TRIGGER_RULE_VERSION,
    )


def confirm_signal(signal: SignalEvent, request: DecisionRequest) -> DecisionRequest:
    """Marketer confirmation: bind the candidate signal to a typed decision request.

    Returns a copy of the request carrying the signal ID, evidence date, and confirmed status.
    The signal is marked ``confirmed`` in place so a second evaluation does not reopen the case.
    """
    signal.status = "confirmed"
    updates: dict = {"signal_id": signal.signal_id}
    if request.evidence_date is None:
        updates["evidence_date"] = signal.evidence_date
    if request.confirmed_at is None:
        updates["confirmed_at"] = f"{signal.evidence_date.isoformat()}T09:00:00Z"
    return request.model_copy(update=updates)


def default_case_id() -> str:
    """The stable case ID shared by the first and later Roventra decisions on one signal."""
    return "CASE-ROVENTRA-HCP-2026"
