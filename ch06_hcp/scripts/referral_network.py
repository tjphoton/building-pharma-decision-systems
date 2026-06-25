"""Disease-specific referral pathway analysis for Chapter 6."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


ANALYSIS_DATE = pd.Timestamp("2024-12-31")
DEFAULT_TRANSITION_DAYS = 60
MIN_EDGE_PATIENTS = 3


def prepare_referral_episodes(
    episodes: pd.DataFrame,
    *,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
    transition_days: int = DEFAULT_TRANSITION_DAYS,
) -> pd.DataFrame:
    """Apply the declared disease, date, geography, and transition rules."""

    prepared = episodes.loc[
        episodes["condition_code"].eq("E11")
        & episodes["source_npi"].ne(episodes["destination_npi"])
        & episodes["destination_date"].le(analysis_date)
        & episodes["transition_days"].between(5, transition_days)
    ].copy()
    return prepared.sort_values(
        ["destination_date", "patient_id", "episode_id"]
    ).reset_index(drop=True)


def build_referral_graph(
    episodes: pd.DataFrame,
    *,
    min_edge_patients: int = MIN_EDGE_PATIENTS,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Build a directed graph with unique patients as edge strength."""

    edges = episodes.groupby(
        [
            "source_npi",
            "destination_npi",
            "source_specialty",
            "destination_specialty",
            "source_account_id",
            "destination_account_id",
            "region",
        ],
        as_index=False,
    ).agg(
        unique_patients=("patient_id", "nunique"),
        referral_episodes=("episode_id", "nunique"),
        median_transition_days=("transition_days", "median"),
        first_referral_date=("destination_date", "min"),
        last_referral_date=("destination_date", "max"),
    )
    edges = edges.loc[edges["unique_patients"].ge(min_edge_patients)].copy()
    edges["path_cost"] = 1 / edges["unique_patients"]
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source_npi,
            row.destination_npi,
            weight=int(row.unique_patients),
            distance=float(row.path_cost),
        )
    return graph, edges.sort_values(
        ["unique_patients", "source_npi", "destination_npi"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def referral_centrality(
    graph: nx.DiGraph,
    affiliations: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate directed weighted pathway measures with visible definitions."""

    context = (
        affiliations.drop_duplicates("npi")
        .set_index("npi")[["specialty_1", "site_account_id", "region"]]
        .to_dict("index")
    )
    weighted_in = dict(graph.in_degree(weight="weight"))
    weighted_out = dict(graph.out_degree(weight="weight"))
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    betweenness = nx.betweenness_centrality(
        graph, normalized=True, weight="distance"
    )
    rows = []
    for npi in graph.nodes:
        info = context.get(npi, {})
        rows.append(
            {
                "npi": npi,
                "specialty": info.get("specialty_1", "Unknown"),
                "account_id": info.get("site_account_id", "Unknown"),
                "region": info.get("region", "Unknown"),
                "unique_sources": int(in_degree.get(npi, 0)),
                "unique_destinations": int(out_degree.get(npi, 0)),
                "patients_received": int(weighted_in.get(npi, 0)),
                "patients_referred": int(weighted_out.get(npi, 0)),
                "betweenness_centrality": float(betweenness.get(npi, 0)),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["pathway_patient_volume"] = (
        metrics["patients_received"] + metrics["patients_referred"]
    )
    metrics["pathway_breadth"] = (
        metrics["unique_sources"] + metrics["unique_destinations"]
    )
    return metrics.sort_values(
        ["pathway_patient_volume", "pathway_breadth", "npi"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


