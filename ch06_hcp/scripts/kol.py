"""Role-specific scientific influence analysis for Chapter 6."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ANALYSIS_DATE = pd.Timestamp("2024-12-31")
KOL_MODEL_VERSION = "ch06-kol-evidence-v2"
T2D_SCIENTIFIC_SPECIALTIES = {"Endocrinology", "Primary Care", "Cardiology"}


def _domain_scores(
    evidence: pd.DataFrame,
    profiles: pd.DataFrame,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
) -> pd.DataFrame:
    """Aggregate dated evidence and normalize within specialty and career stage."""

    dated = evidence.loc[evidence["event_date"].le(analysis_date)].copy()
    age_years = (analysis_date - dated["event_date"]).dt.days / 365.25
    dated["recency_weight"] = np.exp(-np.log(2) * age_years / 3)
    dated["weighted_evidence"] = (
        dated["contribution_weight"]
        * dated["disease_relevance"]
        * dated["identity_match_confidence"]
        * dated["recency_weight"]
    )
    domain = (
        dated.groupby(["npi", "domain"], as_index=False)
        .agg(
            raw_evidence=("weighted_evidence", "sum"),
            evidence_records=("evidence_id", "nunique"),
            mean_identity_confidence=("identity_match_confidence", "mean"),
            latest_evidence_date=("event_date", "max"),
        )
        .merge(
            profiles[["npi", "specialty_1", "career_stage"]],
            on="npi",
            how="left",
            validate="many_to_one",
        )
    )
    domain["domain_percentile"] = (
        domain.groupby(["domain", "specialty_1", "career_stage"])["raw_evidence"]
        .rank(method="average", pct=True)
        .mul(100)
    )
    return domain


def _peer_connection_scores(
    collaborations: pd.DataFrame,
    profiles: pd.DataFrame,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
) -> pd.DataFrame:
    """Calculate dated scientific-collaboration network evidence."""

    links = collaborations.loc[
        collaborations["collaboration_date"].le(analysis_date)
    ].copy()
    graph = nx.Graph()
    for row in links.itertuples(index=False):
        if graph.has_edge(row.source_npi, row.destination_npi):
            graph[row.source_npi][row.destination_npi]["weight"] += 1
        else:
            graph.add_edge(row.source_npi, row.destination_npi, weight=1)
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    rows = pd.DataFrame(
        {
            "npi": profiles["npi"],
            "collaboration_breadth": profiles["npi"].map(degree).fillna(0).astype(int),
            "collaboration_events": profiles["npi"].map(weighted_degree).fillna(0).astype(int),
        }
    ).merge(
        profiles[["npi", "specialty_1", "career_stage"]],
        on="npi",
        how="left",
        validate="one_to_one",
    )
    rows["peer_connection_percentile"] = (
        rows.groupby(["specialty_1", "career_stage"])["collaboration_events"]
        .rank(method="average", pct=True)
        .mul(100)
    )
    return rows


def build_kol_profiles(
    evidence: pd.DataFrame,
    collaborations: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build visible evidence domains and role-specific KOL candidates."""

    profiles = profiles.loc[
        profiles["specialty_1"].isin(T2D_SCIENTIFIC_SPECIALTIES)
    ].copy()
    evidence = evidence.loc[evidence["npi"].isin(profiles["npi"])]
    collaborations = collaborations.loc[
        collaborations["source_npi"].isin(profiles["npi"])
        & collaborations["destination_npi"].isin(profiles["npi"])
    ]
    domain = _domain_scores(evidence, profiles)
    wide_scores = domain.pivot(
        index="npi", columns="domain", values="domain_percentile"
    ).reset_index()
    wide_counts = domain.pivot(
        index="npi", columns="domain", values="evidence_records"
    ).add_suffix(" records").reset_index()
    peer = _peer_connection_scores(collaborations, profiles)
    result = (
        profiles.merge(wide_scores, on="npi", how="left", validate="one_to_one")
        .merge(wide_counts, on="npi", how="left", validate="one_to_one")
        .merge(
            peer[
                [
                    "npi",
                    "collaboration_breadth",
                    "collaboration_events",
                    "peer_connection_percentile",
                ]
            ],
            on="npi",
            how="left",
            validate="one_to_one",
        )
    )
    for column in ["Research", "Leadership", "Practice expertise"]:
        if column not in result:
            result[column] = 0.0
        result[column] = result[column].fillna(0.0)
    record_columns = [column for column in result if column.endswith(" records")]
    result[record_columns] = result[record_columns].fillna(0).astype(int)
    result["active_evidence_domains"] = (
        result[["Research", "Leadership", "Practice expertise"]].gt(0).sum(axis=1)
        + result["peer_connection_percentile"].gt(0).astype(int)
    )
    role_scores = pd.DataFrame(
        {
            "Evidence-generation collaborator": (
                0.65 * result["Research"] + 0.35 * result["Practice expertise"]
            ),
            "National scientific leader": (
                0.55 * result["Leadership"] + 0.45 * result["Research"]
            ),
            "Regional scientific educator": (
                0.55 * result["peer_connection_percentile"]
                + 0.45 * result["Practice expertise"]
            ),
            "Local practice expert": (
                0.70 * result["Practice expertise"]
                + 0.30 * result["peer_connection_percentile"]
            ),
        },
        index=result.index,
    )
    result["role_fit_score"] = role_scores.max(axis=1).round(1)
    result["proposed_role"] = role_scores.idxmax(axis=1)
    result["kol_candidate"] = result["role_fit_score"].ge(65)
    result.loc[~result["kol_candidate"], "proposed_role"] = "No role proposed"
    evidence_coverage = result["active_evidence_domains"] / 4
    result["evidence_confidence"] = np.select(
        [evidence_coverage.ge(0.75), evidence_coverage.ge(0.5)],
        ["High", "Moderate"],
        default="Low",
    )
    result["review_status"] = np.where(
        result["kol_candidate"], "Medical-affairs review required", "No role proposed"
    )
    result["as_of_date"] = ANALYSIS_DATE
    result["model_version"] = KOL_MODEL_VERSION
    result = result.rename(
        columns={
            "Research": "research_percentile",
            "Leadership": "leadership_percentile",
            "Practice expertise": "practice_expertise_percentile",
        }
    )
    return result.sort_values(
        ["kol_candidate", "active_evidence_domains", "research_percentile", "npi"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True), domain


def build_kol_validation(
    kol_profiles: pd.DataFrame,
    medical_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare proposed roles with independent synthetic medical review.

    Starts from kol_candidates (left side) so all model-flagged candidates
    are present; review rows that exist in medical_reviews are joined in.
    Stats are computed only on candidates that have at least one review decision.
    """
    candidates = kol_profiles.loc[
        kol_profiles["kol_candidate"], ["npi", "proposed_role"]
    ]
    reviews_str = medical_reviews.copy()
    reviews_str["npi"] = reviews_str["npi"].astype(str)

    review = candidates.merge(reviews_str, on="npi", how="left")
    reviewed = review.dropna(subset=["reviewer_id"])

    reviewed = reviewed.copy()
    reviewed["role_match"] = reviewed["reviewed_role"].eq(reviewed["proposed_role"])
    reviewed["confirmed"] = reviewed["decision"].eq("Confirm")

    pivot = reviewed.pivot(index="npi", columns="reviewer_id", values="confirmed")
    common = pivot.dropna()
    kappa = (
        cohen_kappa_score(common.iloc[:, 0], common.iloc[:, 1])
        if len(common) and common.shape[1] == 2
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "validation_measure": "KOL candidates",
                "value": len(candidates),
            },
            {
                "validation_measure": "Reviewed candidates",
                "value": reviewed["npi"].nunique(),
            },
            {
                "validation_measure": "Proposed role match rate",
                "value": reviewed["role_match"].mean(),
            },
            {
                "validation_measure": "Reviewer confirmation rate",
                "value": reviewed["confirmed"].mean(),
            },
            {
                "validation_measure": "Reviewer decision kappa",
                "value": kappa,
            },
        ]
    )
    return reviewed, summary


def build_transparency_review(
    kol_profiles: pd.DataFrame,
    transparency: pd.DataFrame,
) -> pd.DataFrame:
    """Attach payment disclosure only after scientific roles are proposed."""

    candidates = kol_profiles.loc[
        kol_profiles["kol_candidate"],
        ["npi", "proposed_role", "review_status"],
    ]
    result = candidates.merge(transparency, on="npi", how="left", validate="one_to_one")
    result["payment_records"] = result["payment_records"].fillna(0).astype(int)
    result["total_payment_amount"] = result["total_payment_amount"].fillna(0.0)
    result["transparency_use"] = (
        "Disclosure context only; excluded from scientific influence evidence"
    )
    return result
