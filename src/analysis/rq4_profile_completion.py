"""
RQ4 — Participant Profile (Bio + Interview) and Completion
==========================================================

Questions:
  - Are participants who fill in a bio more likely to complete?
  - Do bio length / interview length correlate with completion?
  - Do bio sentiment or topic content predict completion?
  - Does profile completeness add incremental signal over engagement
    (i.e. is it an independent marker, or just another proxy for being a
    keen user)?

Methods:
  - Chi-square: has_bio / has_interview vs completion
  - Mann-Whitney U with BH-FDR: bio + interview text-stat and NLP features
  - Spearman trend: dose-response across bio word-count bins
  - Logistic regression: profile-only model + engagement-adjusted model
  - Sensitivity: first-enrolment-per-user (profile is user-level, so the
    primary enrolment-level analysis treats multi-cohort users with
    inflated weight)

Privacy note: profile text is kept behind the facilitator-only boundary
on the platform. The features used here are aggregates (word count,
sentiment score, topic probabilities) computed once at ingestion; the
raw bio / interview text is not displayed in any output and is not
fed into the dropout classifier in the sibling `engagement_ml` repo.
This script is exploratory and is not consumed by any
facilitator-facing tool.

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/rq4_*.csv
         output/analysis/figures/fig_rq4_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from config import (
    load_data, mann_whitney_compare, chi2_or,
    save_csv, save_fig, PALETTE,
)

try:
    from src.utils import apply_publication_style
    apply_publication_style()
except Exception:
    pass


PROFILE_TEXTSTAT = [
    "has_bio", "bio_word_count",
    "has_interview", "n_interview_answers", "interview_word_count",
]
PROFILE_NLP = [
    "bio_sentiment",
    "bio_topic_hope", "bio_topic_anxiety", "bio_topic_gratitude",
    "bio_topic_struggle", "bio_topic_social",
    "interview_sentiment",
    "interview_topic_hope", "interview_topic_anxiety", "interview_topic_gratitude",
    "interview_topic_struggle", "interview_topic_social",
]
PROFILE_ALL = PROFILE_TEXTSTAT + PROFILE_NLP


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("RQ4: Participant Profile (Bio + Interview) and Completion")
    print("=" * 60)

    missing = [c for c in PROFILE_ALL if c not in df.columns]
    if missing:
        print(f"  Skipping RQ4: missing profile columns {missing}")
        return

    # Cast booleans to int so chi-square and logistic models accept them.
    df = df.copy()
    df["has_bio"] = df["has_bio"].astype(int)
    df["has_interview"] = df["has_interview"].astype(int)

    # ── 1. Profile completion vs Completion (chi-square) ────────────
    print("\n--- Has-Bio vs Completion ---")
    ct_bio = pd.crosstab(df["has_bio"], df["dropout_label"])
    ct_bio.index = ["No bio", "Has bio"]
    ct_bio.columns = ["Completer", "Dropout"]
    chi2, p, OR, ci_lo, ci_hi = chi2_or(ct_bio)
    bio_yes = ct_bio.loc["Has bio", "Completer"] / ct_bio.loc["Has bio"].sum() * 100
    bio_no = ct_bio.loc["No bio", "Completer"] / ct_bio.loc["No bio"].sum() * 100
    print(f"  Has-bio completion:    {bio_yes:.1f}%")
    print(f"  No-bio completion:     {bio_no:.1f}%")
    print(f"  Chi2 = {chi2:.1f}, p = {p:.2e}")
    print(f"  OR  = {OR:.2f} [{ci_lo:.2f}, {ci_hi:.2f}] (no-bio vs has-bio: dropout direction)")
    save_csv(ct_bio, "rq4_bio_chi2")

    print("\n--- Has-Interview vs Completion ---")
    ct_iv = pd.crosstab(df["has_interview"], df["dropout_label"])
    ct_iv.index = ["No interview", "Has interview"]
    ct_iv.columns = ["Completer", "Dropout"]
    chi2_iv, p_iv, OR_iv, ci_lo_iv, ci_hi_iv = chi2_or(ct_iv)
    iv_yes = ct_iv.loc["Has interview", "Completer"] / ct_iv.loc["Has interview"].sum() * 100
    iv_no = ct_iv.loc["No interview", "Completer"] / ct_iv.loc["No interview"].sum() * 100
    print(f"  Has-interview completion: {iv_yes:.1f}%")
    print(f"  No-interview completion:  {iv_no:.1f}%")
    print(f"  Chi2 = {chi2_iv:.1f}, p = {p_iv:.2e}")
    print(f"  OR  = {OR_iv:.2f} [{ci_lo_iv:.2f}, {ci_hi_iv:.2f}]")
    save_csv(ct_iv, "rq4_interview_chi2")

    # ── 2. Univariate comparisons across profile features ───────────
    print("\n--- Univariate Comparisons across Profile Features (BH-FDR) ---")
    mw = mann_whitney_compare(df, PROFILE_ALL)
    if len(mw):
        _, p_adj, _, _ = multipletests(mw["p_value"], method="fdr_bh")
        mw["p_adj"] = p_adj
        mw["significant"] = mw["p_adj"] < 0.05
        n_sig = mw["significant"].sum()
        print(f"  {n_sig}/{len(mw)} significant after BH-FDR")
        top = mw.sort_values("p_adj").head(5)
        for _, row in top.iterrows():
            mark = "*" if row["significant"] else " "
            print(
                f"   {mark} {row['feature']:30s} r={row['rank_biserial_r']:+.3f} "
                f"p_adj={row['p_adj']:.4f} (compl_med={row['compl_median']}, drop_med={row['drop_median']})"
            )
        save_csv(mw, "rq4_mann_whitney")
    else:
        print("  Skipped: no comparable rows")

    # ── 3. Bio word-count dose-response ─────────────────────────────
    print("\n--- Dose-Response (bio word count) ---")
    bio = df[df["has_bio"] == 1].copy()
    if len(bio) > 50:
        bins = [0, 5.5, 10.5, 20.5, 40.5, bio["bio_word_count"].max() + 1]
        labels = ["1-5", "6-10", "11-20", "21-40", "41+"]
        bio["bio_bin"] = pd.cut(
            bio["bio_word_count"], bins=bins, labels=labels, right=False
        )
        bin_stats = (
            bio.groupby("bio_bin", observed=True)
            .agg(
                n=("dropout_label", "count"),
                completers=("dropout_label", lambda x: (x == 0).sum()),
            )
            .reset_index()
        )
        bin_stats["completion_pct"] = bin_stats["completers"] / bin_stats["n"] * 100
        rho, p_trend = sp_stats.spearmanr(
            range(len(bin_stats)), bin_stats["completion_pct"]
        )
        print(f"  Spearman rho = {rho:.3f}, p = {p_trend:.4f}")
        save_csv(bin_stats, "rq4_dose_response")

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            bin_stats["bio_bin"].astype(str),
            bin_stats["completion_pct"],
            color=PALETTE["blue"],
            edgecolor="white",
        )
        for bar, n in zip(bars, bin_stats["n"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"n={n:,}",
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("Bio Word Count")
        ax.set_ylabel("Completion Rate (%)")
        ax.set_ylim(0, 105)
        save_fig(fig, "fig_rq4_bio_dose_response")
        plt.close(fig)
    else:
        print(f"  Skipped: only {len(bio)} bio rows")

    # ── 4. Logistic regression — profile-only ───────────────────────
    print("\n--- Logistic Regression A: profile-only (+ course) ---")
    profile_predictors = [
        "has_bio", "bio_word_count", "bio_sentiment",
        "bio_topic_hope", "bio_topic_anxiety",
        "has_interview", "n_interview_answers", "interview_sentiment",
    ]
    res_a, na = _fit_logit(df, profile_predictors)
    if res_a is not None:
        for pred in profile_predictors:
            if pred not in res_a.index:
                continue
            r = res_a.loc[pred]
            sig = "*" if r["p_value"] < 0.05 else ""
            print(
                f"  {pred:30s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
                f"p={r['p_value']:.4f} {sig}"
            )
        print(f"  N = {na:,}")
        save_csv(res_a.loc[profile_predictors], "rq4_logistic_profile_only")

    # ── 5. Logistic regression — profile + engagement controls ──────
    print("\n--- Logistic Regression B: profile + engagement controls ---")
    eng_controls = [
        "activities_in_first_7d",
        "n_logins",
        "days_to_first_activity",
    ]
    predictors_b = profile_predictors + eng_controls
    res_b, nb = _fit_logit(df, predictors_b)
    if res_b is not None:
        for pred in predictors_b:
            if pred not in res_b.index:
                continue
            r = res_b.loc[pred]
            sig = "*" if r["p_value"] < 0.05 else ""
            mark = "(profile)" if pred in profile_predictors else "(control)"
            print(
                f"  {pred:30s} {mark}: OR={r['OR']:.3f} "
                f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}"
            )
        print(f"  N = {nb:,}")
        save_csv(res_b.loc[predictors_b], "rq4_logistic_profile_adjusted")

    # ── 6. Sensitivity — one row per user ───────────────────────────
    print("\n--- Sensitivity: first enrolment per user (de-clusters) ---")
    first_only = (
        df.sort_values("started", na_position="last")
        .drop_duplicates(subset=["user_id"], keep="first")
    )
    ct_sens = pd.crosstab(first_only["has_bio"], first_only["dropout_label"])
    ct_sens.index = ["No bio", "Has bio"]
    ct_sens.columns = ["Completer", "Dropout"]
    chi2_s, p_s, OR_s, ci_lo_s, ci_hi_s = chi2_or(ct_sens)
    print(f"  N (unique users) = {len(first_only):,}")
    print(f"  Has-bio OR = {OR_s:.2f} [{ci_lo_s:.2f}, {ci_hi_s:.2f}], p = {p_s:.2e}")
    save_csv(ct_sens, "rq4_sensitivity_unique_users")


def _fit_logit(df: pd.DataFrame, predictors: list[str]):
    """Fit a logistic regression of dropout on `predictors` + course dummies.

    Returns (results_df, n) or (None, 0) on convergence / singularity failure.
    """
    cols = predictors + ["dropout_label", "course_name"]
    reg = df[cols].dropna(subset=predictors + ["dropout_label", "course_name"])
    if len(reg) < 50:
        print(f"  Skipped: only {len(reg)} usable rows")
        return None, 0

    X = reg[predictors].astype(float).copy()
    course_dummies = pd.get_dummies(
        reg["course_name"], drop_first=True, dtype=float
    )
    X = pd.concat([X, course_dummies], axis=1)
    X = sm.add_constant(X)
    y = reg["dropout_label"].astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:
        print(f"  Logit fit failed: {exc}")
        return None, len(reg)

    results = pd.DataFrame({
        "OR": np.exp(model.params),
        "CI_low": np.exp(model.conf_int()[0]),
        "CI_high": np.exp(model.conf_int()[1]),
        "p_value": model.pvalues,
    })
    return results, len(reg)


if __name__ == "__main__":
    run()
