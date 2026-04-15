"""
GEE Robustness Model
======================

Account for within-module clustering using Generalised Estimating Equations.

Method:
  - GEE with binomial family, logit link, exchangeable working correlation
  - 10 module-level clusters, robust (sandwich) standard errors
  - Features z-standardised for per-SD odds ratios
  - Writers only

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

Note: With only 10 clusters, sandwich SEs may be anti-conservative.
Results are exploratory.

Inputs:  output/features/user_level_features_v2.csv
Outputs: output/analysis_v2/tables/gee_*.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from config import load_data, save_csv


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

    # Z-standardise
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(gee_df[feats]),
        columns=feats, index=gee_df.index,
    )
    X = sm.add_constant(X)

    # Fit GEE
    gee = sm.GEE(
        gee_df["dropout_label"], X,
        groups=gee_df["module_id"],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = gee.fit()

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
