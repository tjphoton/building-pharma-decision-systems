"""Time-to-event analyses for Chapter 5, implemented in plain pandas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def km_curve(durations: pd.Series, events: pd.Series) -> pd.DataFrame:
    """Kaplan-Meier estimate from durations and event indicators.

    Returns one row per distinct duration with the number at risk, events,
    censorings, and the survival estimate S(t). The complement 1 - S(t) is
    the cumulative incidence when the event is initiation.
    """

    durations = np.asarray(durations, dtype=int)
    events = np.asarray(events, dtype=bool)
    survival = 1.0
    greenwood_sum = 0.0
    rows = [
        {
            "day": 0,
            "at_risk": len(durations),
            "events": 0,
            "censored": 0,
            "survival": 1.0,
            "lower_95": 1.0,
            "upper_95": 1.0,
            "cumulative_initiation": 0.0,
            "cumulative_initiation_lower_95": 0.0,
            "cumulative_initiation_upper_95": 0.0,
        }
    ]
    for day in np.unique(durations):
        at_risk = int((durations >= day).sum())
        event_count = int(((durations == day) & events).sum())
        censored = int(((durations == day) & ~events).sum())
        if at_risk and event_count:
            survival *= 1 - event_count / at_risk
            if at_risk > event_count:
                greenwood_sum += event_count / (at_risk * (at_risk - event_count))

        if survival <= 0:
            lower, upper = 0.0, 0.0
        elif survival >= 1 or greenwood_sum == 0:
            lower, upper = survival, survival
        else:
            log_survival = np.log(survival)
            log_log_se = np.sqrt(greenwood_sum) / abs(log_survival)
            log_minus_log = np.log(-log_survival)
            z_value = 1.959963984540054
            lower = np.exp(
                -np.exp(log_minus_log + z_value * log_log_se)
            )
            upper = np.exp(
                -np.exp(log_minus_log - z_value * log_log_se)
            )
        rows.append(
            {
                "day": int(day),
                "at_risk": at_risk,
                "events": event_count,
                "censored": censored,
                "survival": survival,
                "lower_95": lower,
                "upper_95": upper,
                "cumulative_initiation": 1 - survival,
                "cumulative_initiation_lower_95": 1 - upper,
                "cumulative_initiation_upper_95": 1 - lower,
            }
        )
    return pd.DataFrame(rows)


def km_median(curve: pd.DataFrame) -> float:
    """First day at which the KM survival estimate drops to 0.5 or below."""

    crossed = curve.loc[curve["survival"].le(0.5), "day"]
    return float(crossed.iloc[0]) if not crossed.empty else float("nan")


def km_estimate_at(curve: pd.DataFrame, day: int) -> float:
    """KM survival estimate at a given day (step function, last value carried)."""

    eligible = curve.loc[curve["day"].le(day), "survival"]
    return float(eligible.iloc[-1]) if not eligible.empty else 1.0


def treatment_initiation_curve(patient_journeys: pd.DataFrame) -> pd.DataFrame:
    """Cumulative treatment initiation with censoring at follow-up end."""

    durations = np.where(
        patient_journeys["initiated_treatment"],
        patient_journeys["days_to_treatment"],
        patient_journeys["followup_days"],
    ).astype(int)
    events = patient_journeys["initiated_treatment"].astype(bool)
    curve = km_curve(pd.Series(durations), pd.Series(events))
    curve["cumulative_initiation"] = 1 - curve["survival"]
    curve["cumulative_initiation_lower_95"] = 1 - curve["upper_95"]
    curve["cumulative_initiation_upper_95"] = 1 - curve["lower_95"]
    return curve


def aalen_johansen_curve(
    durations: pd.Series,
    outcomes: pd.Series,
    event_of_interest: str = "Treated",
    censor_label: str = "Censored",
) -> pd.DataFrame:
    """Cumulative incidence for one event with any other event competing.

    Returns the event-free probability and cumulative incidence for the event
    of interest and all competing events. Censoring removes a record from later
    risk sets without changing any probability on the censoring day.
    """

    durations = np.asarray(durations, dtype=int)
    outcomes = np.asarray(outcomes, dtype=str)
    event_free = 1.0
    cif_interest = 0.0
    cif_competing = 0.0
    rows = [{
        "day": 0, "at_risk": len(durations), "interest_events": 0,
        "competing_events": 0, "censored": 0, "event_free": 1.0,
        "cumulative_interest": 0.0, "cumulative_competing": 0.0,
    }]
    for day in np.unique(durations):
        at_risk = int((durations >= day).sum())
        on_day = durations == day
        interest = int((on_day & (outcomes == event_of_interest)).sum())
        censored = int((on_day & (outcomes == censor_label)).sum())
        competing = int(on_day.sum() - interest - censored)
        prior_event_free = event_free
        cif_interest += prior_event_free * interest / at_risk
        cif_competing += prior_event_free * competing / at_risk
        event_free *= 1 - (interest + competing) / at_risk
        rows.append({
            "day": int(day), "at_risk": at_risk,
            "interest_events": interest, "competing_events": competing,
            "censored": censored, "event_free": event_free,
            "cumulative_interest": cif_interest,
            "cumulative_competing": cif_competing,
        })
    return pd.DataFrame(rows)


def line_persistence_curve(lines: pd.DataFrame, line_number: int = 1) -> pd.DataFrame:
    """Time until departure from a given observed regimen.

    A switch, addition, restart, or confirmed discontinuation ends persistence
    on the current line. Only administrative end of observation is censored.
    """

    selected = lines.loc[lines["line_number"].eq(line_number)]
    curve = km_curve(
        selected["line_days"],
        ~selected["end_reason"].eq("Censored"),
    )
    curve["estimand"] = "Time to departure from initial regimen"
    return curve
