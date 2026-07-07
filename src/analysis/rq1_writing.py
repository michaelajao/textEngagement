"""
RQ1 — Writing Engagement and Completion
========================================

Questions:
  - Does writing anything at all distinguish completers from dropouts?
  - Is there a dose-response relationship between writing volume and completion?
  - Do early writing signals (first 7 days) predict completion?
  - Does the association hold after adjusting for platform usage and course?

Methods:
  - Chi-square: writer vs non-writer
  - Mann-Whitney U with BH-FDR: feature-by-feature comparisons
  - Spearman trend: dose-response across activity bins
  - Logistic regression: early writing predictors + controls

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/rq1_*.csv
         output/analysis/figures/fig_rq1_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from config import (
    load_data, mann_whitney_compare, chi2_or,
    save_csv, save_fig, PALETTE, TABLE_DIR, FIG_DIR,
)

apply_pub = None
try:
    from src.utils import apply_publication_style
    apply_publication_style()
except Exception:
    pass


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    ALL_FEATS = groups["all_features"]
    print("\n" + "=" * 60)
    print("RQ1: Writing Engagement and Completion")
    print("=" * 60)

    # ── 1. Writer vs Non-Writer ──
    print("\n--- Writer vs Non-Writer ---")
    ct = pd.crosstab(df["is_writer"], df["dropout_label"])
    ct.index = ["Non-writer", "Writer"]
    ct.columns = ["Completer", "Dropout"]
    chi2, p, OR, ci_lo, ci_hi = chi2_or(ct)

    wr_comp = ct.loc["Writer", "Completer"] / ct.loc["Writer"].sum() * 100
    nw_comp = ct.loc["Non-writer", "Completer"] / ct.loc["Non-writer"].sum() * 100

    print(f"  Writer completion:     {wr_comp:.1f}%")
    print(f"  Non-writer completion: {nw_comp:.1f}%")
    print(f"  Chi2 = {chi2:.1f}, p = {p:.2e}")
    print(f"  OR = {OR:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")

    save_csv(ct, "rq1_writer_chi2")

    # ── 2. Univariate comparisons ──
    print("\n--- Univariate Comparisons (BH-FDR) ---")
    numeric_feats = df[ALL_FEATS].select_dtypes(include=[np.number]).columns.tolist()
    mw = mann_whitney_compare(df, numeric_feats)
    _, p_adj, _, _ = multipletests(mw["p_value"], method="fdr_bh")
    mw["p_adj"] = p_adj
    mw["significant"] = mw["p_adj"] < 0.05
    n_sig = mw["significant"].sum()
    print(f"  {n_sig}/{len(mw)} significant after BH-FDR")
    save_csv(mw, "rq1_mann_whitney")

    # ── 3. Dose-response ──
    print("\n--- Dose-Response ---")
    bins = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5, df["total_activities_submitted"].max() + 1]
    labels = ["0", "1", "2-3", "4-5", "6-10", "11-20", "21+"]
    df["act_bin"] = pd.cut(df["total_activities_submitted"], bins=bins, labels=labels, right=False)

    bin_stats = df.groupby("act_bin", observed=True).agg(
        n=("dropout_label", "count"),
        completers=("dropout_label", lambda x: (x == 0).sum()),
    ).reset_index()
    bin_stats["completion_pct"] = bin_stats["completers"] / bin_stats["n"] * 100

    # Bin-level rho is DESCRIPTIVE only (n = number of bins, not people).
    rho, p_trend = sp_stats.spearmanr(range(len(bin_stats)), bin_stats["completion_pct"])
    print(f"  Bin-level (descriptive) Spearman rho = {rho:.3f}, p = {p_trend:.4f}")

    # Inferential trend test at the INDIVIDUAL level (n = all starters):
    # Spearman correlation between activity count and completion (1 - dropout).
    rho_ind, p_ind = sp_stats.spearmanr(
        df["total_activities_submitted"], 1 - df["dropout_label"]
    )
    print(f"  Individual-level Spearman rho = {rho_ind:.3f}, "
          f"p = {p_ind:.2e} (n = {len(df):,})")
    bin_stats.attrs["individual_spearman_rho"] = rho_ind
    trend_df = pd.DataFrame([{
        "level": "bins (descriptive)", "n": len(bin_stats),
        "spearman_rho": round(rho, 3), "p_value": p_trend,
    }, {
        "level": "individual", "n": len(df),
        "spearman_rho": round(rho_ind, 3), "p_value": p_ind,
    }])
    save_csv(trend_df, "rq1_dose_response_trend")
    save_csv(bin_stats, "rq1_dose_response")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bin_stats["act_bin"], bin_stats["completion_pct"], color=PALETTE["blue"], edgecolor="white")
    for bar, n in zip(bars, bin_stats["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={n:,}", ha="center", fontsize=8)
    ax.set_xlabel("Total Activities Submitted")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_ylim(0, 105)
    save_fig(fig, "fig_rq1_dose_response")
    plt.close(fig)

    # ── 4. Early writing signals ──
    print("\n--- Early Writing Signals ---")
    df["wrote_first_week"] = (df["activities_in_first_7d"] > 0).astype(int)
    ct_ew = pd.crosstab(df["wrote_first_week"], df["dropout_label"])
    ct_ew.index = ["No week-1 writing", "Wrote in week 1"]
    ct_ew.columns = ["Completer", "Dropout"]
    chi2_ew, p_ew, _, _, _ = chi2_or(ct_ew)
    print(f"  Wrote in first week: chi2 = {chi2_ew:.1f}, p = {p_ew:.2e}")
    save_csv(ct_ew, "rq1_early_writing")

    # ── 5. Logistic regression ──
    # NOTE: avg_vocab_richness and days_to_first_activity are undefined
    # for non-writers, so listwise deletion restricts this model to the
    # WRITER subsample — report it as such.
    print("\n--- Logistic Regression (writer subsample) ---")
    X_cols = ["activities_in_first_7d", "avg_vocab_richness", "days_to_first_activity", "n_logins"]
    reg_df = df[X_cols + ["dropout_label", "course_name"]].dropna(subset=X_cols)
    X = reg_df[X_cols].copy()
    X = pd.concat([X, pd.get_dummies(reg_df["course_name"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    y = reg_df["dropout_label"]

    model = sm.Logit(y, X).fit(disp=0)
    results = pd.DataFrame({
        "OR": np.exp(model.params),
        "CI_low": np.exp(model.conf_int()[0]),
        "CI_high": np.exp(model.conf_int()[1]),
        "p_value": model.pvalues,
    })

    for pred in X_cols:
        r = results.loc[pred]
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {pred:30s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")

    save_csv(results.loc[X_cols], "rq1_logistic_regression")

    print(f"\n  Pseudo R2 = {model.prsquared:.4f}, N = {len(y):,} (writers with "
          "complete predictors)")

    # ── 6. Sensitivity — one row per user (de-clusters repeat enrolments) ──
    print("\n--- Sensitivity: first enrolment per user ---")
    first_only = (
        df.sort_values("started", na_position="last")
        .drop_duplicates(subset=["user_id"], keep="first")
    )
    reg_u = first_only[X_cols + ["dropout_label", "course_name"]].dropna(subset=X_cols)
    Xu = reg_u[X_cols].copy()
    Xu = pd.concat([Xu, pd.get_dummies(reg_u["course_name"], drop_first=True, dtype=float)], axis=1)
    Xu = sm.add_constant(Xu)
    mu = sm.Logit(reg_u["dropout_label"], Xu).fit(disp=0)
    res_u = pd.DataFrame({
        "OR": np.exp(mu.params),
        "CI_low": np.exp(mu.conf_int()[0]),
        "CI_high": np.exp(mu.conf_int()[1]),
        "p_value": mu.pvalues,
    }).loc[X_cols]
    for pred in X_cols:
        r = res_u.loc[pred]
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {pred:30s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")
    print(f"  N (unique users) = {len(reg_u):,}")
    save_csv(res_u, "rq1_sensitivity_unique_users")


if __name__ == "__main__":
    run()
