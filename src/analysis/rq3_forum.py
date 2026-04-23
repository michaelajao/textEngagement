"""
RQ3 — Forum Participation and Completion
==========================================

Questions:
  - Is forum posting associated with completion?
  - Is the association independent of writing volume and comment receipt?

Methods:
  - Chi-square: poster vs non-poster
  - Logistic regression: forum replies + writing/comment/login controls

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/rq3_*.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from config import load_data, chi2_or, mann_whitney_compare, save_csv, save_fig, PALETTE, TABLE_DIR


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("RQ3: Forum Participation and Completion")
    print("=" * 60)

    # ── 1. Poster vs non-poster ──
    print("\n--- Poster vs Non-Poster ---")
    ct = pd.crosstab(df["is_poster"], df["dropout_label"])
    ct.index = ["Non-poster", "Poster"]
    ct.columns = ["Completer", "Dropout"]
    chi2, p, OR, ci_lo, ci_hi = chi2_or(ct)

    poster_comp = ct.loc["Poster", "Completer"] / ct.loc["Poster"].sum() * 100
    nonposter_comp = ct.loc["Non-poster", "Completer"] / ct.loc["Non-poster"].sum() * 100

    print(f"  Poster completion:     {poster_comp:.1f}%")
    print(f"  Non-poster completion: {nonposter_comp:.1f}%")
    print(f"  Chi2 = {chi2:.1f}, p = {p:.2e}")
    print(f"  OR = {OR:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")
    save_csv(ct, "rq3_forum_chi2")

    # ── 2. Forum volume among posters ──
    print("\n--- Forum Volume (posters only) ---")
    posters = df[df["is_poster"] == 1]
    forum_feats = ["total_discussion_replies", "forum_span_days", "days_to_first_post"]
    forum_feats = [f for f in forum_feats if f in posters.columns]
    mw_forum = mann_whitney_compare(posters, forum_feats)
    print(mw_forum[["feature", "compl_median", "drop_median", "rank_biserial_r", "p_value"]].to_string(index=False))
    save_csv(mw_forum, "rq3_forum_volume")

    if forum_feats:
        feature_meta = {
            "total_discussion_replies": ("Replies", PALETTE["blue"]),
            "forum_span_days": ("Forum Span (days)", PALETTE["orange"]),
            "days_to_first_post": ("Days to First Post", PALETTE["green"]),
        }
        fig, axes = plt.subplots(1, len(forum_feats), figsize=(5 * len(forum_feats), 4.5))
        if len(forum_feats) == 1:
            axes = [axes]
        for ax, feature in zip(axes, forum_feats):
            title, color = feature_meta.get(feature, (feature, PALETTE["blue"]))
            comp = posters.loc[posters["dropout_label"] == 0, feature].dropna()
            drop = posters.loc[posters["dropout_label"] == 1, feature].dropna()
            bp = ax.boxplot(
                [comp, drop],
                labels=["Completers", "Dropouts"],
                patch_artist=True,
                showfliers=False,
                widths=0.6,
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.45, edgecolor="#444444")
            for median in bp["medians"]:
                median.set(color="#222222", linewidth=1.6)
            ax.set_ylabel(title)
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_fig(fig, "fig_rq3_participation")
        plt.close(fig)

    # ── 3. Early vs late posting ──
    print("\n--- Early vs Late Posting ---")
    posters_df = df.copy()
    posters_df["post_group"] = "Non-poster"
    mask_poster = posters_df["is_poster"] == 1
    mask_early = mask_poster & (posters_df["days_to_first_post"] <= 14)
    mask_late = mask_poster & (posters_df["days_to_first_post"] > 14)
    posters_df.loc[mask_early, "post_group"] = "Early (<=14d)"
    posters_df.loc[mask_late, "post_group"] = "Late (>14d)"

    post_summary = posters_df.groupby("post_group").agg(
        n=("dropout_label", "count"),
        completion_pct=("dropout_label", lambda x: (1 - x.mean()) * 100),
    ).round(1)
    print(post_summary)
    save_csv(post_summary, "rq3_early_posting")

    # ── 4. Logistic regression ──
    print("\n--- Logistic Regression ---")
    X_cols = ["total_discussion_replies", "total_activities_submitted",
              "total_comments_received", "n_logins"]
    reg = df[X_cols + ["dropout_label", "course_name"]].dropna(subset=X_cols)
    X = reg[X_cols].copy()
    X = pd.concat([X, pd.get_dummies(reg["course_name"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    y = reg["dropout_label"]

    model = sm.Logit(y, X).fit(disp=0)
    results = pd.DataFrame({
        "OR": np.exp(model.params), "CI_low": np.exp(model.conf_int()[0]),
        "CI_high": np.exp(model.conf_int()[1]), "p_value": model.pvalues,
    })
    for pred in X_cols:
        r = results.loc[pred]
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {pred:35s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")
    save_csv(results.loc[X_cols], "rq3_logistic_regression")


if __name__ == "__main__":
    run()
