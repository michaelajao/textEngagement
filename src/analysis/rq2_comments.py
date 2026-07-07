"""
RQ2 — Facilitator Comments and Retention
==========================================

Questions:
  - Do comment recipients complete at higher rates (among writers)?
  - Does the association hold after adjusting for writing volume?
  - Does comment quality (response time, word count) matter beyond presence?

Methods:
  - Chi-square: commented vs uncommented writers
  - Logistic regression: Model 1 (receipt only), Model 2 (+ quality)
  - Stratified analysis: comment effect within activity-level strata

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/rq2_*.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import load_data, chi2_or, save_csv, TABLE_DIR


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("RQ2: Facilitator Comments and Retention")
    print("=" * 60)

    # ── 1. Chi-square: comment receipt among writers ──
    print("\n--- Comment Receipt (writers only) ---")
    ct = pd.crosstab(writers["received_comment"], writers["dropout_label"])
    ct.index = ["No comments", "Received comments"]
    ct.columns = ["Completer", "Dropout"]
    chi2, p, OR, ci_lo, ci_hi = chi2_or(ct)

    rc_comp = ct.loc["Received comments", "Completer"] / ct.loc["Received comments"].sum() * 100
    nc_comp = ct.loc["No comments", "Completer"] / ct.loc["No comments"].sum() * 100

    print(f"  Commented completion:   {rc_comp:.1f}%")
    print(f"  Uncommented completion: {nc_comp:.1f}%")
    print(f"  Chi2 = {chi2:.1f}, p = {p:.2e}")
    print(f"  OR = {OR:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")
    save_csv(ct, "rq2_comment_chi2")

    # ── 2. Stratified by activity level ──
    print("\n--- Stratified by Activity Level ---")
    # Rank-based tertiles: robust to duplicate bin edges (qcut with fixed
    # labels raises if an edge is dropped on a data refresh).
    ranks = writers["total_activities_submitted"].rank(method="first")
    writers["act_tertile"] = pd.cut(
        ranks, bins=3, labels=["Low", "Medium", "High"]
    )
    strat = writers.groupby(["act_tertile", "received_comment"]).agg(
        n=("dropout_label", "count"),
        dropout_pct=("dropout_label", lambda x: x.mean() * 100),
    ).round(1)
    print(strat)
    save_csv(strat, "rq2_stratified")

    # ── 3. Logistic regression — Model 1: receipt only ──
    print("\n--- Model 1: Comment Receipt (adjusted) ---")
    X1_cols = ["received_comment", "total_activities_submitted", "n_logins"]
    reg1 = writers[X1_cols + ["dropout_label", "course_name"]].dropna(subset=X1_cols)
    X1 = reg1[X1_cols].copy()
    X1 = pd.concat([X1, pd.get_dummies(reg1["course_name"], drop_first=True, dtype=float)], axis=1)
    X1 = sm.add_constant(X1)
    y1 = reg1["dropout_label"]

    m1 = sm.Logit(y1, X1).fit(disp=0)
    r1 = pd.DataFrame({
        "OR": np.exp(m1.params), "CI_low": np.exp(m1.conf_int()[0]),
        "CI_high": np.exp(m1.conf_int()[1]), "p_value": m1.pvalues,
    })
    for pred in X1_cols:
        r = r1.loc[pred]
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {pred:35s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")
    save_csv(r1.loc[X1_cols], "rq2_model1")

    # ── 4. Model 2: comment quality among recipients ──
    # The quality covariates (latency, word count) are undefined for
    # non-recipients, so listwise deletion reduces the sample to comment
    # recipients anyway. Fitting on recipients explicitly (and without
    # the receipt indicator, which would be near-constant) makes the
    # model answer the question it can actually answer: among writers
    # who RECEIVED a comment, does measured quality matter?
    print("\n--- Model 2: Comment Quality (recipients only) ---")
    X2_cols = ["avg_response_hours", "avg_comment_word_count",
               "total_activities_submitted", "n_logins"]
    recipients = writers[writers["received_comment"] == 1]
    reg2 = recipients[X2_cols + ["dropout_label", "course_name"]].dropna(subset=X2_cols)
    if len(reg2) > 50:
        X2 = reg2[X2_cols].copy()
        X2 = pd.concat([X2, pd.get_dummies(reg2["course_name"], drop_first=True, dtype=float)], axis=1)
        X2 = sm.add_constant(X2)
        y2 = reg2["dropout_label"]
        m2 = sm.Logit(y2, X2).fit(disp=0)
        r2 = pd.DataFrame({
            "OR": np.exp(m2.params), "CI_low": np.exp(m2.conf_int()[0]),
            "CI_high": np.exp(m2.conf_int()[1]), "p_value": m2.pvalues,
        })
        for pred in X2_cols:
            if pred in r2.index:
                r = r2.loc[pred]
                sig = "*" if r["p_value"] < 0.05 else ""
                print(f"  {pred:35s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")
        save_csv(r2.loc[[c for c in X2_cols if c in r2.index]], "rq2_model2")
    else:
        print("  Insufficient data for Model 2")


if __name__ == "__main__":
    run()
