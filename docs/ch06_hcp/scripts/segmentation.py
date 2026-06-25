"""Validated k-means engagement profiles for Chapter 6."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler


MODEL_VERSION = "ch06-kmeans-v2"
RANDOM_STATE = 42
MIN_COHORT_PATIENTS = 5
MIN_TREATED_PATIENTS = 3


FEATURE_COLUMNS = [
    "evidence_need_score",
    "access_resource_score",
    "digital_response_rate",
    "field_response_rate",
]


def prepare_segmentation_features(
    hcp_features: pd.DataFrame,
    engagement_signals: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, RobustScaler]:
    """Build a continuous, post-gate feature space for Euclidean distance."""

    eligible = hcp_features.loc[
        hcp_features["contact_permitted"]
        & hcp_features["cohort_patients"].ge(MIN_COHORT_PATIENTS)
        & hcp_features["treated_patients"].ge(MIN_TREATED_PATIENTS)
    ].merge(engagement_signals, on="npi", how="left", validate="one_to_one")
    eligible["log_cohort_patients"] = np.log1p(eligible["cohort_patients"])
    eligible["log_recent_contacts"] = np.log1p(eligible["recent_contacts"])
    eligible["opportunity_rate"] = (
        eligible["review_opportunity"] / eligible["cohort_patients"].clip(lower=1)
    )
    eligible["access_signal_rate"] = (
        eligible["access_signal_patients"]
        / eligible["cohort_patients"].clip(lower=1)
    )
    eligible["productive_contact_rate"] = (
        eligible["productive_contacts"]
        / eligible["lifetime_contacts"].clip(lower=1)
    )
    eligible["roventra_share"] = eligible["roventra_share"].fillna(
        eligible["roventra_share"].median()
    )
    if eligible[FEATURE_COLUMNS].isna().any().any():
        missing = eligible[FEATURE_COLUMNS].columns[
            eligible[FEATURE_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(f"Segmentation features contain missing values: {missing}")
    scaler = RobustScaler()
    transformed = scaler.fit_transform(eligible[FEATURE_COLUMNS])
    return eligible.reset_index(drop=True), transformed, scaler


def _seed_stability(matrix: np.ndarray, k: int) -> float:
    labels = []
    for seed in [11, 23, 42, 71, 101]:
        labels.append(
            KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(matrix)
        )
    scores = [
        adjusted_rand_score(labels[0], candidate) for candidate in labels[1:]
    ]
    return float(np.mean(scores))


def _bootstrap_stability(
    matrix: np.ndarray,
    k: int,
    *,
    samples: int = 30,
    random_state: int = RANDOM_STATE,
) -> float:
    base = KMeans(n_clusters=k, random_state=random_state, n_init=30).fit(matrix)
    rng = np.random.default_rng(random_state)
    scores = []
    for sample in range(samples):
        indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        model = KMeans(
            n_clusters=k, random_state=random_state + sample + 1, n_init=20
        ).fit(matrix[indices])
        scores.append(adjusted_rand_score(base.labels_, model.predict(matrix)))
    return float(np.mean(scores))


def evaluate_cluster_counts(
    matrix: np.ndarray,
    candidate_k: range = range(3, 7),
) -> pd.DataFrame:
    """Evaluate separation, size, seed stability, and bootstrap stability."""

    rows = []
    for k in candidate_k:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30).fit(matrix)
        counts = pd.Series(model.labels_).value_counts()
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(matrix, model.labels_),
                "minimum_cluster_size": int(counts.min()),
                "minimum_cluster_share": float(counts.min() / len(matrix)),
                "seed_stability_ari": _seed_stability(matrix, k),
                "bootstrap_stability_ari": _bootstrap_stability(matrix, k),
            }
        )
    evaluation = pd.DataFrame(rows)
    evaluation["selection_score"] = (
        evaluation["silhouette"]
        + 0.25 * evaluation["seed_stability_ari"]
        + 0.25 * evaluation["bootstrap_stability_ari"]
        + 0.10 * evaluation["minimum_cluster_share"].clip(upper=0.15) / 0.15
    )
    minimum_operational_size = max(8, int(np.ceil(0.10 * len(matrix))))
    evaluation["operational_size_pass"] = evaluation["minimum_cluster_size"].ge(
        minimum_operational_size
    )
    return evaluation


def select_cluster_count(evaluation: pd.DataFrame) -> int:
    """Select a stable, operational candidate under a prespecified tie rule."""

    candidates = evaluation.loc[evaluation["operational_size_pass"]]
    if candidates.empty:
        candidates = evaluation
    best_score = candidates["selection_score"].max()
    near_best = candidates.loc[
        candidates["selection_score"].ge(best_score - 0.02)
    ]
    return int(
        near_best.sort_values(
            [
                "bootstrap_stability_ari",
                "seed_stability_ari",
                "silhouette",
                "k",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]["k"]
    )


def _profile_label(row: pd.Series) -> tuple[str, str]:
    """Name a centroid from its dominant observed profile."""

    if row["access_resource_score"] >= 0.70 and row["evidence_need_score"] < 0.62:
        return "Access-resource need", "Access resources, then evidence follow-up"
    if row["evidence_need_score"] >= 0.62 and row["digital_response_rate"] > row[
        "field_response_rate"
    ]:
        return "Digital evidence seeker", "Approved digital evidence, then field review"
    if row["evidence_need_score"] >= 0.62:
        return "Field evidence builder", "Field evidence discussion"
    if row["digital_response_rate"] >= 0.62:
        return "Digital maintenance", "Digital maintenance, then field review"
    if row["field_response_rate"] >= 0.62:
        return "Field maintenance", "Maintenance field follow-up"
    if row["roventra_share"] >= 0.7 and row["opportunity_rate"] < 0.35:
        return "Established adopter", "Maintenance and new-evidence update"
    return "Balanced follow-up", "Standard evidence review"


def _assignment_stability(
    matrix: np.ndarray,
    base_model: KMeans,
    *,
    samples: int = 40,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Estimate per-HCP assignment stability after label alignment."""

    rng = np.random.default_rng(random_state)
    agreement = np.zeros(len(matrix), dtype=float)
    base_labels = base_model.labels_
    for sample in range(samples):
        indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        model = KMeans(
            n_clusters=base_model.n_clusters,
            random_state=random_state + sample + 1,
            n_init=20,
        ).fit(matrix[indices])
        predicted = model.predict(matrix)
        aligned = np.empty_like(predicted)
        for cluster_id in range(base_model.n_clusters):
            members = base_labels[predicted == cluster_id]
            aligned[predicted == cluster_id] = (
                pd.Series(members).mode().iloc[0] if len(members) else cluster_id
            )
        agreement += aligned == base_labels
    return agreement / samples


def fit_hcp_segments(
    feature_table: pd.DataFrame,
    matrix: np.ndarray,
    evaluation: pd.DataFrame,
) -> tuple[KMeans, pd.DataFrame, pd.DataFrame]:
    """Fit the selected model and return deployment rows plus centroid profiles."""

    selected_k = select_cluster_count(evaluation)
    model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=50).fit(
        matrix
    )
    distances = model.transform(matrix)[np.arange(len(matrix)), model.labels_]
    stability = _assignment_stability(matrix, model)
    result = feature_table.copy()
    result["cluster_id"] = model.labels_
    result["centroid_distance"] = distances
    result["assignment_stability"] = stability
    result["model_version"] = MODEL_VERSION

    profile_columns = [
        "cohort_patients",
        "opportunity_rate",
        "roventra_share",
        "access_signal_rate",
        "recent_contacts",
        "productive_contact_rate",
        "evidence_need_score",
        "access_resource_score",
        "digital_response_rate",
        "field_response_rate",
    ]
    profiles = result.groupby("cluster_id", as_index=False).agg(
        hcp_count=("npi", "nunique"),
        **{column: (column, "mean") for column in profile_columns},
    )
    names = {row.cluster_id: _profile_label(pd.Series(row._asdict())) for row in profiles.itertuples()}
    result["segment_name"] = result["cluster_id"].map(
        {cluster: f"C{cluster}: {label}" for cluster, (label, _) in names.items()}
    )
    result["engagement_pattern"] = result["cluster_id"].map(
        {cluster: pattern for cluster, (_, pattern) in names.items()}
    )
    profiles["segment_name"] = profiles["cluster_id"].map(
        {cluster: f"C{cluster}: {label}" for cluster, (label, _) in names.items()}
    )
    profiles["engagement_pattern"] = profiles["cluster_id"].map(
        {cluster: pattern for cluster, (_, pattern) in names.items()}
    )
    keep = [
        "npi",
        "cluster_id",
        "segment_name",
        "centroid_distance",
        "assignment_stability",
        "model_version",
        "engagement_pattern",
        *FEATURE_COLUMNS,
    ]
    return model, result[keep], profiles


def build_policy_baseline(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent comparator for the fitted engagement profiles."""

    result = feature_table[["npi"]].copy()
    result["policy_segment"] = np.select(
        [
            feature_table["access_resource_score"].ge(0.65),
            feature_table["evidence_need_score"].ge(0.65)
            & feature_table["digital_response_rate"].gt(
                feature_table["field_response_rate"]
            ),
            feature_table["evidence_need_score"].ge(0.65),
            feature_table["roventra_share"].ge(0.7),
        ],
        [
            "Access-resource need",
            "Digital evidence seeker",
            "Field evidence builder",
            "Established adopter",
        ],
        default="Balanced follow-up",
    )
    return result


def compare_with_policy_baseline(
    segments: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tab k-means profiles against the transparent policy baseline."""

    merged = segments[["npi", "segment_name"]].merge(
        policy, on="npi", how="inner", validate="one_to_one"
    )
    return (
        pd.crosstab(merged["segment_name"], merged["policy_segment"])
        .reset_index()
        .rename_axis(columns=None)
    )
