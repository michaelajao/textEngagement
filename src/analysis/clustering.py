"""
Engagement Profiling — K-Means Clustering
===========================================

Build engagement profiles using the full engagement feature set
(groups["all_features"]; count taken from feature_groups.json, not
hard-coded here). This includes platform engagement, writing,
facilitator, and forum features.

Methods:
  - K-means clustering (k = 2-8, silhouette evaluation)
  - Features z-score standardised
  - NaN imputation rule (documented in Methods): count features
    (ORIGINALS or median == 0) are filled with 0 because a missing
    count means "no events"; ratio/mean features undefined for
    non-writers are filled with the column median so they stay
    neutral after z-scoring.
  - Visualisation: silhouette plot, PCA, t-SNE, heatmap, dropout rates

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/cluster_*.csv
         output/analysis/figures/fig_cluster_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import load_data, save_csv, save_fig, PALETTE, ORIGINALS


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("Engagement Profiling (K-Means Clustering)")
    print("=" * 60)

    ALL_FEATS = groups["all_features"]

    # ── Prepare feature matrix: ALL participants, ALL features ──
    # Use all numeric features from the full feature set
    numeric_feats = df[ALL_FEATS].select_dtypes(include=[np.number]).columns.tolist()
    X_df = df[numeric_feats].copy()

    # Fill NaN: 0 for count features, column median for ratio/proportion features
    for c in X_df.columns:
        if X_df[c].isna().any():
            if X_df[c].median() == 0 or c in ORIGINALS:
                X_df[c] = X_df[c].fillna(0)
            else:
                X_df[c] = X_df[c].fillna(X_df[c].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    print(f"\nClustering ALL {len(df):,} participants on {len(numeric_feats)} features:")
    for f in numeric_feats:
        print(f"  {f}")

    # ── Silhouette analysis: k = 2 to 8 ──
    print("\n--- Silhouette Scores ---")
    sil_results = []
    best_k, best_sil = 2, -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = km.inertia_
        sil_results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(inertia, 1)})
        marker = " <-- best" if sil > best_sil else ""
        if sil > best_sil:
            best_k, best_sil = k, sil
        print(f"  k={k}: silhouette = {sil:.4f}, inertia = {inertia:.0f}{marker}")

    sil_df = pd.DataFrame(sil_results)
    save_csv(sil_df, "cluster_silhouette_scores")

    # Silhouette plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(sil_df["k"], sil_df["silhouette"], "o-", color=PALETTE["blue"])
    ax.axvline(best_k, color=PALETTE["red"], linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend()

    ax = axes[1]
    ax.plot(sil_df["k"], sil_df["inertia"], "o-", color=PALETTE["green"])
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SS)")
    save_fig(fig, "fig_cluster_silhouette")
    plt.close(fig)

    # ── Fit multiple k values for comparison ──
    # Best silhouette is k=2, but engagement is a continuum so we also
    # examine k=3, 4, 5 for richer interpretability.
    # Report all and let the researcher choose.
    for alt_k in [3, 4, 5]:
        if alt_k == best_k:
            continue
        km_alt = KMeans(n_clusters=alt_k, n_init=20, random_state=42)
        alt_labels = km_alt.fit_predict(X)
        alt_engage = pd.DataFrame({"cluster": alt_labels,
            "acts": df["total_activities_submitted"], "dropout": df["dropout_label"]})
        alt_order = alt_engage.groupby("cluster")["acts"].mean().sort_values().index
        alt_summary = alt_engage.groupby("cluster").agg(
            n=("dropout", "count"), dropout_pct=("dropout", lambda x: round(x.mean()*100, 1)))
        alt_summary = alt_summary.loc[alt_order]
        print(f"\n  k={alt_k} profile sizes and dropout rates:")
        for idx, row in alt_summary.iterrows():
            print(f"    Cluster {idx}: n={int(row['n']):,}, dropout={row['dropout_pct']:.1f}%")

    # Use k=5 for the main output. Silhouette scores are close across
    # k=3-7, so k is chosen for interpretability (a five-level engagement
    # gradient incl. the "Light" group) rather than a statistical optimum;
    # the paper reports this choice and the silhouette range explicitly.
    chosen_k = 5
    print(f"\n--- Fitting k={chosen_k} (interpretability-driven choice) ---")

    km_final = KMeans(n_clusters=chosen_k, n_init=30, random_state=42)
    df = df.copy()
    df["cluster"] = km_final.fit_predict(X)

    # ── Order clusters by overall engagement level ──
    # Use a composite: mean of z-scored total_activities + n_page_visits + total_discussion_replies
    engagement_score = df.groupby("cluster")[
        ["total_activities_submitted", "n_page_visits", "total_discussion_replies"]
    ].mean().mean(axis=1)
    order = engagement_score.sort_values().index.tolist()

    # Auto-generate profile labels
    if chosen_k == 2:
        label_names = ["Low engagement", "High engagement"]
    elif chosen_k == 3:
        label_names = ["Minimal", "Moderate", "Deep"]
    elif chosen_k == 4:
        label_names = ["Disengaged", "Minimal", "Moderate", "Deep"]
    elif chosen_k == 5:
        label_names = ["Disengaged", "Minimal", "Light", "Moderate", "Deep"]
    else:
        label_names = [f"Group {i+1}" for i in range(chosen_k)]

    label_map = {order[i]: label_names[i] for i in range(chosen_k)}
    df["profile"] = df["cluster"].map(label_map)

    # ── Profile summary ──
    print("\n--- Profile Summary ---")
    summary_cols = [
        "total_activities_submitted", "n_logins", "n_distinct_pages",
        "n_bookmarks", "total_comments_received", "total_discussion_replies",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]

    profiles = df.groupby("profile").agg(
        n=("dropout_label", "count"),
        dropout_pct=("dropout_label", lambda x: round(x.mean() * 100, 1)),
        **{f"mean_{c}": (c, "mean") for c in summary_cols},
    ).round(1)

    # Reorder by dropout rate descending
    profiles = profiles.sort_values("dropout_pct", ascending=False)
    print(profiles.to_string())
    save_csv(profiles, "cluster_profiles")

    # Save assignments
    assign_cols = ["module_id", "user_id", "cohort_id", "cluster", "profile"]
    save_csv(df[assign_cols], "cluster_assignments")

    # ── PCA visualisation ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PCA scatter by profile
    ax = axes[0]
    color_cycle = [PALETTE["grey"], PALETTE["red"], PALETTE["orange"],
                   PALETTE["blue"], PALETTE["green"], PALETTE["purple"],
                   PALETTE["teal"], PALETTE["dark_blue"]]
    for i, profile in enumerate(profiles.index):
        mask = (df["profile"] == profile).values
        comp_mask = mask & (df["dropout_label"] == 0).values
        drop_mask = mask & (df["dropout_label"] == 1).values
        color = color_cycle[i % len(color_cycle)]
        ax.scatter(X_pca[comp_mask, 0], X_pca[comp_mask, 1],
                   c=color, alpha=0.3, s=12, label=profile)
        ax.scatter(X_pca[drop_mask, 0], X_pca[drop_mask, 1],
                   c=color, alpha=0.8, s=20, marker="x")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)

    # Dropout by profile bar chart
    ax = axes[1]
    prof_order = profiles.index.tolist()
    dp_rates = profiles["dropout_pct"].values
    ns = profiles["n"].astype(int).values
    bars = ax.bar(prof_order, dp_rates, color=color_cycle[:len(prof_order)], edgecolor="white")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n:,}", ha="center", fontsize=9)
    ax.set_ylabel("Dropout Rate (%)")
    ax.set_ylim(0, max(dp_rates) + 10)
    ax.tick_params(axis="x", rotation=15)

    save_fig(fig, "fig_cluster_profiles")
    plt.close(fig)

    # ── t-SNE visualisation ──
    print("\n  Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, profile in enumerate(profiles.index):
        mask = (df["profile"] == profile).values
        comp_mask = mask & (df["dropout_label"] == 0).values
        drop_mask = mask & (df["dropout_label"] == 1).values
        color = color_cycle[i % len(color_cycle)]
        ax.scatter(X_tsne[comp_mask, 0], X_tsne[comp_mask, 1],
                   c=color, alpha=0.3, s=12, label=profile)
        ax.scatter(X_tsne[drop_mask, 0], X_tsne[drop_mask, 1],
                   c=color, alpha=0.8, s=20, marker="x")
    ax.legend(fontsize=8)
    save_fig(fig, "fig_cluster_tsne")
    plt.close(fig)

    # ── Feature heatmap by profile ──
    profile_means = df.groupby("profile")[numeric_feats].mean()
    # Min-max normalise per feature
    norm = (profile_means - profile_means.min()) / (profile_means.max() - profile_means.min() + 1e-9)
    norm = norm.loc[profiles.index]  # Keep same order

    fig, ax = plt.subplots(figsize=(14, max(6, len(numeric_feats) * 0.3)))
    sns.heatmap(norm.T, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 7})
    save_fig(fig, "fig_cluster_heatmap")
    plt.close(fig)

    print(f"\n  Clustering complete: k={chosen_k}, {len(profiles)} profiles")


if __name__ == "__main__":
    run()
