"""
GEE Robustness Model
======================

Account for within-module clustering using Generalised Estimating Equations.

Method:
  - GEE with binomial family, logit link, exchangeable working correlation
  - 8 module-level clusters, robust (sandwich) standard errors
  - Features z-standardised for per-SD odds ratios
  - Writers only
  - Specification shared with the bootstrap and LOMO refits via config.fit_gee

Features (10, covering all dimensions):
  - Volume: total_activities_submitted
  - Quality: avg_vocab_richness
  - Linguistic: avg_sentiment, avg_future_orientation
  - Content: activity_type_entropy
  - Trajectory: word_count_trend
  - Facilitator: total_comments_received
  - Forum: total_discussion_replies
  - Early warning: activities_in_first_7d
  - Platform (NEW): n_distinct_pages

Note: With only 8 clusters, sandwich SEs may be anti-conservative.
Results are exploratory; see sensitivity.py for the cluster bootstrap and
leave-one-module-out checks.

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/gee_*.csv
"""

import numpy as np
import pandas as pd

from config import fit_gee, load_data, save_csv


GEE_FEATS = [
    "total_activities_submitted",
    "avg_vocab_richness",
    "avg_sentiment",
    "avg_future_orientation",
    "activity_type_entropy",
    "word_count_trend",
    "total_comments_received",
    "total_discussion_replies",
    "activities_in_first_7d",
    "n_distinct_pages",  # NEW — platform engagement
]


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("GEE Robustness Model")
    print("=" * 60)

    feats = [f for f in GEE_FEATS if f in writers.columns]
    print(f"\nGEE features ({len(feats)}):")
    for f in feats:
        print(f"  {f}")

    gee_df = writers[["module_id", "dropout_label"] + feats].dropna()
    print(f"\nSample: {len(gee_df):,} writers")
    print(f"Clusters (modules): {gee_df['module_id'].nunique()}")

    # Z-standardised predictors, binomial/logit, exchangeable, sandwich SEs.
    result = fit_gee(gee_df, feats, "module_id")

    # Extract results
    summary = pd.DataFrame({
        "coef": result.params,
        "OR": np.exp(result.params),
        "CI_low": np.exp(result.conf_int().iloc[:, 0]),
        "CI_high": np.exp(result.conf_int().iloc[:, 1]),
        "robust_se": result.bse,
        "p_value": result.pvalues,
    }).drop("const", errors="ignore")
    summary["significant"] = summary["p_value"] < 0.05

    print("\n--- GEE Results (z-standardised, OR < 1 = protective) ---")
    for feat in feats:
        if feat in summary.index:
            r = summary.loc[feat]
            sig = " *" if r["significant"] else ""
            print(f"  {feat:35s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f}{sig}")

    save_csv(summary, "gee_results")

    # BH-FDR correction across GEE coefficients
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(summary["p_value"], method="fdr_bh")
    summary["p_adj"] = p_adj
    summary["sig_fdr"] = summary["p_adj"] < 0.05

    n_sig = summary["sig_fdr"].sum()
    print(f"\n  Significant after BH-FDR: {n_sig}/{len(summary)}")
    save_csv(summary, "gee_results_fdr")


if __name__ == "__main__":
    run()
