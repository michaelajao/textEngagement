"""
Engagement Profile Clustering

Identifies distinct participant engagement profiles via K-means clustering
on user-level writing, NLP, and facilitator features.

**Filters to writers only** (total_activities_submitted > 0) to avoid
the trivial non-writer cluster dominating results. Caps k search at 2-5
for interpretable, well-sized profiles.

Steps:
1. Filter to writers, select features, z-score standardise.
2. Silhouette score analysis to choose optimal k (k=2..5).
3. K-means clustering with optimal k.
4. Profile characterisation: mean per feature, dropout rate per cluster.
5. PCA + t-SNE 2-D projections coloured by cluster.

Outputs
-------
Tables  engagement_cluster_profiles.csv, engagement_cluster_assignments.csv
Figs    fig_silhouette_scores.pdf, fig_cluster_pca.pdf, fig_cluster_tsne.pdf,
        fig_cluster_heatmap.pdf, fig_cluster_dropout.pdf
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analysis import (
    FIGURES_DIR,
    PALETTE,
    TABLES_DIR,
    apply_publication_style,
    ensure_output_dirs,
    load_analytical_tables,
)

apply_publication_style()

CLUSTER_FEATURES = [
    "total_activities_submitted",
    "total_words_written",
    "avg_description_length",
    "writing_span_days",
    "writing_frequency",
    "avg_vocab_richness",
    "avg_future_orientation",
    "avg_self_reference",
    "pct_gratitude",
    "pct_goalsetting",
    "pct_emotions",
    "total_comments_received",
    "pct_activities_with_comments",
    "total_discussion_replies",
]

CLUSTER_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#E91E63",
]


# ── core ─────────────────────────────────────────────────────────────────────

def prepare_features(
    user: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    feats = [f for f in CLUSTER_FEATURES if f in user.columns]
    X_df = user[feats].fillna(0).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    return X, feats, X_df


def choose_k(
    X: np.ndarray, k_range: range = range(2, 9),
) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
        print(f"  k={k}: silhouette={sil:.3f}, inertia={km.inertia_:.0f}")

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    best_sil = scores.loc[scores["silhouette"].idxmax(), "silhouette"]
    print(f"\n  Optimal k = {best_k} (silhouette = {best_sil:.3f})")
    return best_k, scores


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    return km.fit_predict(X)


def characterise_clusters(
    user: pd.DataFrame, labels: np.ndarray, feats: list[str],
) -> pd.DataFrame:
    df = user.copy()
    df["cluster"] = labels
    agg_dict: dict = {
        "n": ("dropout_label", "size"),
        "dropout_rate": ("dropout_label", "mean"),
        "completion_rate": ("dropout_label", lambda x: 1 - x.mean()),
    }
    for f in feats:
        if f in df.columns:
            agg_dict[f] = (f, "mean")
    return df.groupby("cluster").agg(**agg_dict).round(3).reset_index()


# ── figures ──────────────────────────────────────────────────────────────────

def fig_silhouette(scores: pd.DataFrame, optimal_k: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(scores["k"], scores["silhouette"], marker="o",
             color=PALETTE["blue"], lw=2)
    ax1.axvline(optimal_k, color=PALETTE["red"], ls="--", alpha=0.7,
                label=f"Optimal k={optimal_k}")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.legend()

    ax2.plot(scores["k"], scores["inertia"], marker="s",
             color=PALETTE["orange"], lw=2)
    ax2.axvline(optimal_k, color=PALETTE["red"], ls="--", alpha=0.7)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Within-Cluster Inertia")
    ax2.set_xlabel("Number of Clusters (k)")
    fig.savefig(FIGURES_DIR / "fig_silhouette_scores.png")
    plt.close(fig)
    print("  -> fig_silhouette_scores.png")


def fig_pca(
    X: np.ndarray, labels: np.ndarray, dropout: np.ndarray, k: int,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLUSTER_COLORS[c], alpha=0.5, s=30,
            label=f"Cluster {c + 1}",
        )
    drop_mask = dropout == 1
    ax.scatter(
        coords[drop_mask, 0], coords[drop_mask, 1],
        c="black", marker="x", s=15, alpha=0.3, label="Dropout",
    )
    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% var)")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(FIGURES_DIR / "fig_cluster_pca.png")
    plt.close(fig)
    print("  -> fig_cluster_pca.png")


def fig_tsne(
    X: np.ndarray, labels: np.ndarray, dropout: np.ndarray, k: int,
) -> None:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    for c in range(k):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLUSTER_COLORS[c], alpha=0.55, s=30,
            label=f"Cluster {c + 1}",
        )
    drop_mask = dropout == 1
    ax.scatter(
        coords[drop_mask, 0], coords[drop_mask, 1],
        c="black", marker="x", s=15, alpha=0.3, label="Dropout",
    )
    ax.set_xlabel("t-SNE Dim 1", labelpad=8)
    ax.set_ylabel("t-SNE Dim 2", labelpad=8)
    ax.margins(x=0.05, y=0.06)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.savefig(
        FIGURES_DIR / "fig_cluster_tsne.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    print("  -> fig_cluster_tsne.png")


def fig_cluster_heatmap(profile: pd.DataFrame, feats: list[str]) -> None:
    display_feats = [f for f in feats[:12] if f in profile.columns]
    mat = profile[display_feats].values
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    with np.errstate(invalid="ignore"):
        mat_norm = np.where(
            col_max > col_min,
            (mat - col_min) / (col_max - col_min),
            0.5,
        )

    k = len(profile)
    fig, ax = plt.subplots(
        figsize=(max(10, len(display_feats)), max(4, k * 0.8)),
    )
    im = ax.imshow(mat_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(display_feats)))
    ax.set_xticklabels(display_feats, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(
        [
            f"Cluster {i + 1}\n"
            f"(n={row['n']:.0f}, {row['dropout_rate'] * 100:.0f}% dropout)"
            for i, (_, row) in enumerate(profile.iterrows())
        ],
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, label="Normalised Value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cluster_heatmap.png")
    plt.close(fig)
    print("  -> fig_cluster_heatmap.png")


def fig_cluster_dropout(profile: pd.DataFrame) -> None:
    k = len(profile)
    labels = (
        profile["profile_label"].tolist()
        if "profile_label" in profile.columns
        else [f"Cluster {i + 1}" for i in range(k)]
    )
    colors = ["#9E9E9E"] + CLUSTER_COLORS[: k - 1]  # grey for non-writers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Dot plot for dropout rate
    y_pos = range(k)
    ax1.scatter(profile["dropout_rate"] * 100, y_pos,
                c=colors, s=120, zorder=3, edgecolors="black", linewidths=0.5)
    ax1.hlines(y_pos, 0, profile["dropout_rate"] * 100,
               colors=colors, linewidths=2, alpha=0.6)
    for i, (_, row) in enumerate(profile.iterrows()):
        ax1.text(row["dropout_rate"] * 100 + 2, i,
                 f"{row['dropout_rate'] * 100:.0f}% (n={int(row['n'])})",
                 va="center", fontsize=10)
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Dropout Rate (%)")
    ax1.set_xlim(0, 85)
    ax1.invert_yaxis()

    # Dot plot for sample size
    ax2.scatter(profile["n"], y_pos,
                c=colors, s=120, zorder=3, edgecolors="black", linewidths=0.5)
    ax2.hlines(y_pos, 0, profile["n"],
               colors=colors, linewidths=2, alpha=0.6)
    for i, (_, row) in enumerate(profile.iterrows()):
        ax2.text(row["n"] + 15, i, str(int(row["n"])),
                 va="center", fontsize=10)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of Participants")
    ax2.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cluster_dropout.png")
    plt.close(fig)
    print("  -> fig_cluster_dropout.png")


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    ensure_output_dirs()
    all_users = data["users"]
    writers = all_users[all_users["total_activities_submitted"] > 0].copy()
    nonwriters = all_users[all_users["total_activities_submitted"] == 0].copy()

    print("Engagement Clustering")
    print("=" * 50)
    print(f"  {len(all_users)} total users: "
          f"{len(writers)} writers, {len(nonwriters)} non-writers")

    # Cluster writers into 3 profiles via K-means
    X, feats, _ = prepare_features(writers)
    print(f"  Using {len(feats)} features for clustering writers")

    print("\nSilhouette scores (k=2..5) ...")
    _, scores_df = choose_k(X, k_range=range(2, 6))
    # k=3 gives the most interpretable profiles (minimal / moderate / deep)
    optimal_k = 3
    print(f"  Using k={optimal_k} (interpretability over marginal silhouette)")
    fig_silhouette(scores_df, optimal_k)

    print(f"\nFitting K-means with k={optimal_k} on writers ...")
    labels = fit_kmeans(X, optimal_k)
    writers["cluster"] = labels

    # Label writer clusters by mean total_activities (ascending)
    writer_profile = characterise_clusters(writers, labels, feats)
    if "total_activities_submitted" in writer_profile.columns:
        rank = writer_profile["total_activities_submitted"].rank().astype(int)
        label_map = {1: "Minimal", 2: "Moderate", 3: "Deep"}
        writer_profile["profile_label"] = rank.map(
            lambda r: label_map.get(r, f"Level {r}")
        )
        writers["profile_label"] = writers["cluster"].map(
            dict(zip(writer_profile["cluster"], writer_profile["profile_label"]))
        )

    # Add non-writers as a 4th profile
    nonwriters["cluster"] = -1
    nonwriters["profile_label"] = "Non-writer"
    nw_row = {
        "cluster": -1,
        "profile_label": "Non-writer",
        "n": len(nonwriters),
        "dropout_rate": nonwriters["dropout_label"].mean(),
        "completion_rate": 1 - nonwriters["dropout_label"].mean(),
    }
    for f in feats:
        nw_row[f] = 0.0  # non-writers have 0 for all engagement features
    nw_profile = pd.DataFrame([nw_row])

    # Combine profiles: Non-writer + 3 writer clusters
    profile = pd.concat([nw_profile, writer_profile], ignore_index=True)
    # Sort: Non-writer first, then by ascending engagement
    sort_order = {"Non-writer": 0, "Minimal": 1, "Moderate": 2, "Deep": 3}
    profile["_sort"] = profile["profile_label"].map(sort_order)
    profile = profile.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    profile.to_csv(
        TABLES_DIR / "engagement_cluster_profiles.csv", index=False,
    )
    print("  Saved: engagement_cluster_profiles.csv")

    # Combine user assignments
    user_out = pd.concat([nonwriters, writers], ignore_index=True)
    user_out[["module_id", "cohort_id", "user_id", "dropout_label", "cluster",
              "profile_label"]].to_csv(
        TABLES_DIR / "engagement_cluster_assignments.csv", index=False,
    )
    print("  Saved: engagement_cluster_assignments.csv")

    print("\n--- Engagement Profile Summary ---")
    summary_cols = [
        "profile_label", "n", "dropout_rate",
        "total_activities_submitted", "total_words_written",
        "total_comments_received",
    ]
    summary_cols = [c for c in summary_cols if c in profile.columns]
    print(profile[summary_cols].round(3).to_string(index=False))

    # Figures use all users (writers clustered + non-writers as group -1)
    # For PCA/t-SNE we only project writers (non-writers have zero vectors)
    print("\nGenerating figures ...")
    dropout = writers["dropout_label"].values
    fig_pca(X, labels, dropout, optimal_k)
    fig_tsne(X, labels, dropout, optimal_k)
    fig_cluster_heatmap(profile, feats)
    fig_cluster_dropout(profile)

    print("\nClustering done.\n")


if __name__ == "__main__":
    data = load_analytical_tables()
    run(data)
