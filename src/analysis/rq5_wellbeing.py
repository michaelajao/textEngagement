"""
RQ5 — Mental Wellbeing (SWEMWBS) and Writing Engagement
========================================================

SWEMWBS (Short Warwick–Edinburgh Mental Wellbeing Scale) is an *optional*
in-course survey. Only participants who answered it appear with non-null
``swemwbs_*`` columns; everyone else is NaN and is excluded *pairwise* from
each test below (a missing optional-survey score is not a zero score).
The full engagement cohort (RQ1–RQ4, clustering, survival) is unaffected.

Four user-selected framings, each filtering to its own non-NaN subset and
printing ``N=``, skipping gracefully when the usable sample is too small:

  (a) Writing engagement -> Pre->Post wellbeing change
      Dose-response of total_activities_submitted / activities_in_first_7d
      on swemwbs_change_pre_post: binned means + Spearman rho + OLS with
      course dummies.
  (b) Baseline wellbeing -> dropout
      chi2 + OR on swemwbs_baseline_depressed x dropout_label; baseline
      band (Low/Average/High) vs completion; nested logistic regression
      (A: baseline + course; B: + early-engagement controls) to test
      whether baseline wellbeing carries signal independent of engagement.
  (c) Completion -> wellbeing change
      Paired Wilcoxon Pre->Post overall and within completers / dropouts;
      Mann-Whitney on change for completers vs dropouts; meaningful-change
      (>=1 raw point) x dropout_label chi-square.
  (d) Wellbeing by engagement profile
      Joins the clustering profile assignments and describes baseline /
      change / depressed-% per profile + Kruskal-Wallis. Self-skips if
      cluster_assignments.csv is absent.

Methodological caveats (also stated in the manuscript Limitations):
  - SWEMWBS entries carry no per-entry session tag; timepoints are inferred
    from *occasion order* (1st = Pre, last = Post, nearest-midpoint = Mid),
    an explicit proxy for the embedded Session 1/3-4/6-8 schedule.
  - The optional survey induces selection (responders are self-selected),
    so all RQ5 estimates are associational and exploratory.

Privacy boundary: SWEMWBS scores are sensitive clinical data kept private
from facilitators on the platform. They are excluded from
``groups['all_features']`` and never enter clustering, the univariate
sweep, or the sibling facilitator-facing ``engagement_ml`` pipeline. This
script is the only consumer and uses aggregate scores only — no raw item
responses appear in any output.

Inputs:  output/features/user_level_features.csv
         output/analysis/tables/cluster_assignments.csv  (framing d, optional)
Outputs: output/analysis/tables/rq5_*.csv
         output/analysis/figures/fig_rq5_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as sp_stats

from config import (
    load_data, chi2_or, save_csv, save_fig, PALETTE, TABLE_DIR,
)

try:
    from src.utils import apply_publication_style
    apply_publication_style()
except Exception:
    pass


SWEMWBS_FEATURE_COLS = [
    "swemwbs_pre", "swemwbs_mid", "swemwbs_post",
    "swemwbs_pre_raw", "swemwbs_post_raw",
    "n_swemwbs_occasions",
    "swemwbs_change_pre_post", "swemwbs_change_pre_mid",
    "swemwbs_baseline_depressed", "swemwbs_band_pre",
    "swemwbs_meaningful_change", "swemwbs_course_sessions",
]
BAND_ORDER = ["Low", "Average", "High"]
MIN_N = 30  # below this a framing/sub-test is skipped


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("RQ5: Mental Wellbeing (SWEMWBS) and Writing Engagement")
    print("=" * 60)

    missing = [c for c in SWEMWBS_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Skipping RQ5: missing SWEMWBS columns {missing}")
        print("  (regenerate features: python src/features.py)")
        return

    df = df.copy()
    n_resp = df["swemwbs_pre"].notna().sum()
    print(f"\n  SWEMWBS responders: {n_resp:,} / {len(df):,} enrolments "
          f"({n_resp / len(df) * 100:.1f}%)")
    if n_resp < MIN_N:
        print(f"  Skipping RQ5: only {n_resp} responders (< {MIN_N})")
        return

    _framing_a(df)
    _framing_b(df)
    _framing_c(df)
    _framing_d(df)


# ─────────────────────────────────────────────────────────────────────
# (a) Writing engagement -> Pre->Post wellbeing change
# ─────────────────────────────────────────────────────────────────────
def _framing_a(df: pd.DataFrame) -> None:
    print("\n--- (a) Writing engagement -> Pre->Post wellbeing change ---")
    sub = df.dropna(subset=["swemwbs_change_pre_post"]).copy()
    print(f"  N (Pre+Post responders) = {len(sub):,}")
    if len(sub) < MIN_N:
        print("  Skipped: too few Pre+Post responders")
        return

    rows = []
    for eng in ["total_activities_submitted", "activities_in_first_7d"]:
        s = sub.dropna(subset=[eng])
        if len(s) < MIN_N:
            print(f"  {eng}: skipped (N={len(s)})")
            continue
        rho, p = sp_stats.spearmanr(s[eng], s["swemwbs_change_pre_post"])
        print(f"  Spearman({eng}, change) rho={rho:+.3f} p={p:.4f}  N={len(s):,}")
        rows.append({
            "engagement_metric": eng,
            "n": len(s),
            "spearman_rho": round(rho, 4),
            "p_value": p,
        })
    if rows:
        save_csv(pd.DataFrame(rows).set_index("engagement_metric"),
                 "rq5_dose_response")

    # Binned means of Pre->Post change across writing-volume bins
    s = sub.dropna(subset=["total_activities_submitted"]).copy()
    if len(s) >= MIN_N:
        max_act = s["total_activities_submitted"].max()
        bins = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5, max_act + 1]
        labels = ["0", "1", "2-3", "4-5", "6-10", "11-20", "21+"]
        s["act_bin"] = pd.cut(s["total_activities_submitted"], bins=bins,
                              labels=labels, right=False)
        bin_stats = (
            s.groupby("act_bin", observed=True)
            .agg(n=("swemwbs_change_pre_post", "count"),
                 mean_change=("swemwbs_change_pre_post", "mean"),
                 sd_change=("swemwbs_change_pre_post", "std"))
            .reset_index()
        )
        bin_stats["mean_change"] = bin_stats["mean_change"].round(3)
        bin_stats["sd_change"] = bin_stats["sd_change"].round(3)
        save_csv(bin_stats.set_index("act_bin"), "rq5_change_by_writing_bin")

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(bin_stats["act_bin"].astype(str),
                      bin_stats["mean_change"],
                      color=PALETTE["blue"], edgecolor="white")
        for bar, n in zip(bars, bin_stats["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.05 if bar.get_height() >= 0 else -0.15),
                    f"n={n:,}", ha="center", fontsize=8)
        ax.axhline(0, color=PALETTE["grey"], lw=0.8)
        ax.set_xlabel("Total Activities Submitted")
        ax.set_ylabel("Mean SWEMWBS Pre→Post change (metric)")
        save_fig(fig, "fig_rq5_change_by_writing")
        plt.close(fig)

    # OLS: change ~ engagement + course dummies
    ols_cols = ["total_activities_submitted", "activities_in_first_7d"]
    res, n = _fit_ols(sub, "swemwbs_change_pre_post", ols_cols)
    if res is not None:
        print(f"  OLS (change ~ engagement + course), N={n:,}:")
        for pred in ols_cols:
            if pred in res.index:
                r = res.loc[pred]
                sig = "*" if r["p_value"] < 0.05 else ""
                print(f"    {pred:28s} beta={r['coef']:+.4f} "
                      f"[{r['CI_low']:+.4f}, {r['CI_high']:+.4f}] "
                      f"p={r['p_value']:.4f} {sig}")
        save_csv(res.loc[[c for c in ols_cols if c in res.index]],
                 "rq5_change_ols")


# ─────────────────────────────────────────────────────────────────────
# (b) Baseline wellbeing -> dropout
# ─────────────────────────────────────────────────────────────────────
def _framing_b(df: pd.DataFrame) -> None:
    print("\n--- (b) Baseline wellbeing -> dropout ---")

    # b1. Baseline "probably depressed" x dropout
    sub = df.dropna(subset=["swemwbs_baseline_depressed", "dropout_label"]).copy()
    print(f"  N (baseline responders) = {len(sub):,}")
    if len(sub) >= MIN_N:
        sub["swemwbs_baseline_depressed"] = sub["swemwbs_baseline_depressed"].astype(int)
        ct = pd.crosstab(sub["swemwbs_baseline_depressed"], sub["dropout_label"])
        ct.index = ["Not depressed", "Prob. depressed"]
        ct.columns = ["Completer", "Dropout"]
        chi2, p, OR, lo, hi = chi2_or(ct)
        for label in ct.index:
            comp_pct = ct.loc[label, "Completer"] / ct.loc[label].sum() * 100
            print(f"  {label:16s} completion: {comp_pct:5.1f}%  (n={ct.loc[label].sum():,})")
        print(f"  Chi2={chi2:.1f}, p={p:.2e}, "
              f"OR={OR:.2f} [{lo:.2f}, {hi:.2f}] "
              f"(not-depressed vs depressed, dropout direction)")
        save_csv(ct, "rq5_baseline_dropout_chi2")
    else:
        print("  b1 skipped: too few baseline responders")

    # b2. Baseline band (Low/Average/High) vs completion
    sb = df.dropna(subset=["swemwbs_band_pre", "dropout_label"]).copy()
    if len(sb) >= MIN_N:
        sb["band"] = pd.Categorical(sb["swemwbs_band_pre"],
                                    categories=BAND_ORDER, ordered=True)
        band = (
            sb.groupby("band", observed=True)
            .agg(n=("dropout_label", "count"),
                 completers=("dropout_label", lambda x: (x == 0).sum()))
            .reset_index()
        )
        band["completion_pct"] = (band["completers"] / band["n"] * 100).round(1)
        ct_band = pd.crosstab(sb["band"], sb["dropout_label"])
        try:
            chi2_b, p_b, _, _ = sp_stats.chi2_contingency(ct_band)
            print(f"  Band vs completion: chi2={chi2_b:.1f}, p={p_b:.4f}")
        except ValueError:
            print("  Band vs completion: chi-square not computable")
        for _, r in band.iterrows():
            print(f"    {str(r['band']):8s} completion {r['completion_pct']:5.1f}%  (n={r['n']:,})")
        save_csv(band.set_index("band"), "rq5_band_completion")

        fig, ax = plt.subplots(figsize=(6, 4))
        colours = {"Low": PALETTE["red"], "Average": PALETTE["orange"],
                   "High": PALETTE["green"]}
        bars = ax.bar(band["band"].astype(str), band["completion_pct"],
                      color=[colours.get(b, PALETTE["blue"]) for b in band["band"]],
                      edgecolor="white")
        for bar, n in zip(bars, band["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={n:,}", ha="center", fontsize=8)
        ax.set_xlabel("Baseline SWEMWBS Wellbeing Band")
        ax.set_ylabel("Completion Rate (%)")
        ax.set_ylim(0, 105)
        save_fig(fig, "fig_rq5_band_completion")
        plt.close(fig)

    # b3. Nested logistic regression: does baseline wellbeing survive
    #     adjustment for early engagement?
    base = ["swemwbs_pre"]
    controls = ["activities_in_first_7d", "n_logins", "days_to_first_activity"]
    res_a, na = _fit_logit(df, base)
    res_b, nb = _fit_logit(df, base + controls)
    out = []
    if res_a is not None and "swemwbs_pre" in res_a.index:
        r = res_a.loc["swemwbs_pre"]
        print(f"  Logit A (baseline + course):       "
              f"swemwbs_pre OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
              f"p={r['p_value']:.4f}  N={na:,}")
        out.append(res_a.loc["swemwbs_pre"].rename("A_baseline_only"))
    if res_b is not None and "swemwbs_pre" in res_b.index:
        r = res_b.loc["swemwbs_pre"]
        print(f"  Logit B (+ early engagement):      "
              f"swemwbs_pre OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
              f"p={r['p_value']:.4f}  N={nb:,}")
        out.append(res_b.loc["swemwbs_pre"].rename("B_engagement_adjusted"))
    if out:
        save_csv(pd.concat(out, axis=1).T, "rq5_logistic_nested")


# ─────────────────────────────────────────────────────────────────────
# (c) Completion -> wellbeing change
# ─────────────────────────────────────────────────────────────────────
def _framing_c(df: pd.DataFrame) -> None:
    print("\n--- (c) Completion -> wellbeing change ---")
    sub = df.dropna(
        subset=["swemwbs_pre_raw", "swemwbs_post_raw", "dropout_label"]
    ).copy()
    print(f"  N (paired Pre+Post) = {len(sub):,}")

    rows = []
    if len(sub) >= MIN_N:
        # Paired Wilcoxon overall and by completion status
        for label, mask in [
            ("Overall", sub.index),
            ("Completers", sub.index[sub["dropout_label"] == 0]),
            ("Dropouts", sub.index[sub["dropout_label"] == 1]),
        ]:
            grp = sub.loc[mask]
            if len(grp) < MIN_N:
                print(f"  Wilcoxon {label}: skipped (N={len(grp)})")
                continue
            pre = grp["swemwbs_pre_raw"].to_numpy(float)
            post = grp["swemwbs_post_raw"].to_numpy(float)
            try:
                stat, p = sp_stats.wilcoxon(post, pre)
            except ValueError as exc:  # e.g. all differences zero
                print(f"  Wilcoxon {label}: not computable ({exc})")
                continue
            d_med = float(np.median(post - pre))
            print(f"  Wilcoxon {label:11s}: median Δraw={d_med:+.1f} "
                  f"W={stat:.0f} p={p:.4f}  N={len(grp):,}")
            rows.append({
                "group": label, "n": len(grp),
                "median_raw_change": round(d_med, 2),
                "wilcoxon_W": round(float(stat), 1),
                "p_value": p,
            })
    if rows:
        save_csv(pd.DataFrame(rows).set_index("group"),
                 "rq5_completion_change")

    # Mann-Whitney: metric Pre->Post change, completers vs dropouts
    mw = df.dropna(subset=["swemwbs_change_pre_post", "dropout_label"]).copy()
    comp = mw.loc[mw["dropout_label"] == 0, "swemwbs_change_pre_post"]
    drop = mw.loc[mw["dropout_label"] == 1, "swemwbs_change_pre_post"]
    if len(comp) >= MIN_N and len(drop) >= MIN_N:
        stat, p = sp_stats.mannwhitneyu(comp, drop, alternative="two-sided")
        r = 1 - (2 * stat) / (len(comp) * len(drop))
        print(f"  Mann-Whitney change (compl vs drop): "
              f"r={r:+.3f} p={p:.4f}  "
              f"(compl med={comp.median():+.2f} n={len(comp):,}; "
              f"drop med={drop.median():+.2f} n={len(drop):,})")
        save_csv(
            pd.DataFrame([{
                "compl_n": len(comp), "drop_n": len(drop),
                "compl_median_change": round(float(comp.median()), 3),
                "drop_median_change": round(float(drop.median()), 3),
                "rank_biserial_r": round(float(r), 3),
                "p_value": p,
            }]).set_index("compl_n"),
            "rq5_change_by_completion",
        )
    else:
        print(f"  Mann-Whitney change: skipped "
              f"(compl N={len(comp)}, drop N={len(drop)})")

    # Meaningful change (>=1 raw point) x completion
    mc = df.dropna(subset=["swemwbs_meaningful_change", "dropout_label"]).copy()
    if len(mc) >= MIN_N:
        mc["swemwbs_meaningful_change"] = mc["swemwbs_meaningful_change"].astype(int)
        ct = pd.crosstab(mc["swemwbs_meaningful_change"], mc["dropout_label"])
        ct.index = ["No meaningful gain", "Meaningful gain (≥1 pt)"]
        ct.columns = ["Completer", "Dropout"]
        chi2, p, OR, lo, hi = chi2_or(ct)
        print(f"  Meaningful-gain x completion: chi2={chi2:.1f}, p={p:.4f}, "
              f"OR={OR:.2f} [{lo:.2f}, {hi:.2f}]")
        save_csv(ct, "rq5_meaningful_chi2")
    else:
        print(f"  Meaningful-gain x completion: skipped (N={len(mc)})")


# ─────────────────────────────────────────────────────────────────────
# (d) Wellbeing by engagement profile
# ─────────────────────────────────────────────────────────────────────
def _framing_d(df: pd.DataFrame) -> None:
    print("\n--- (d) Wellbeing by engagement profile ---")
    ca_path = TABLE_DIR / "cluster_assignments.csv"
    if not ca_path.exists():
        print(f"  Skipped: {ca_path.name} not found "
              "(run clustering before rq5_wellbeing in run_all)")
        return

    ca = pd.read_csv(ca_path)
    keys = ["module_id", "user_id", "cohort_id"]
    if not all(k in ca.columns for k in keys) or "profile" not in ca.columns:
        print(f"  Skipped: {ca_path.name} missing key/profile columns")
        return

    merged = df.merge(ca[keys + ["profile"]], on=keys, how="inner")
    sub = merged.dropna(subset=["swemwbs_pre"]).copy()
    print(f"  N (responders with a cluster profile) = {len(sub):,}")
    if len(sub) < MIN_N:
        print("  Skipped: too few responders with a profile")
        return

    desc = (
        sub.groupby("profile")
        .agg(
            n=("swemwbs_pre", "count"),
            pre_mean=("swemwbs_pre", "mean"),
            pre_sd=("swemwbs_pre", "std"),
            change_n=("swemwbs_change_pre_post", "count"),
            change_mean=("swemwbs_change_pre_post", "mean"),
            depressed_pct=("swemwbs_baseline_depressed",
                           lambda x: x.dropna().mean() * 100),
        )
        .round(3)
        .reset_index()
    )
    print(desc.to_string(index=False))
    save_csv(desc.set_index("profile"), "rq5_by_profile")

    # Kruskal-Wallis across profiles
    for col, name in [("swemwbs_pre", "baseline"),
                      ("swemwbs_change_pre_post", "Pre→Post change")]:
        samples = [
            g[col].dropna().to_numpy(float)
            for _, g in sub.groupby("profile")
            if g[col].notna().sum() >= 5
        ]
        if len(samples) >= 2:
            h, p = sp_stats.kruskal(*samples)
            print(f"  Kruskal-Wallis ({name}) across profiles: "
                  f"H={h:.2f} p={p:.4f}")
        else:
            print(f"  Kruskal-Wallis ({name}): skipped (insufficient groups)")

    fig, ax = plt.subplots(figsize=(7, 4))
    order = desc.sort_values("pre_mean")["profile"].tolist()
    d = desc.set_index("profile").loc[order].reset_index()
    bars = ax.bar(d["profile"].astype(str), d["pre_mean"],
                  color=PALETTE["blue"], edgecolor="white")
    for bar, n in zip(bars, d["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"n={n:,}", ha="center", fontsize=8)
    ax.set_xlabel("Engagement Profile")
    ax.set_ylabel("Mean baseline SWEMWBS (metric)")
    save_fig(fig, "fig_rq5_by_profile")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────
def _fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]):
    """OLS of `outcome` on `predictors` + course dummies.

    Returns (results_df, n) or (None, 0) when too few rows / fit fails.
    """
    cols = predictors + [outcome, "course_name"]
    reg = df[cols].dropna(subset=predictors + [outcome, "course_name"])
    if len(reg) < MIN_N:
        return None, 0
    X = reg[predictors].astype(float).copy()
    X = pd.concat(
        [X, pd.get_dummies(reg["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X)
    y = reg[outcome].astype(float)
    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:
        print(f"  OLS fit failed: {exc}")
        return None, len(reg)
    ci = model.conf_int()
    return (
        pd.DataFrame({
            "coef": model.params,
            "CI_low": ci[0],
            "CI_high": ci[1],
            "p_value": model.pvalues,
        }),
        len(reg),
    )


def _fit_logit(df: pd.DataFrame, predictors: list[str]):
    """Logistic regression of dropout on `predictors` + course dummies.

    Returns (results_df, n) or (None, 0) on too-few-rows / fit failure.
    """
    cols = predictors + ["dropout_label", "course_name"]
    reg = df[cols].dropna(subset=predictors + ["dropout_label", "course_name"])
    if len(reg) < MIN_N:
        return None, 0
    X = reg[predictors].astype(float).copy()
    X = pd.concat(
        [X, pd.get_dummies(reg["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X)
    y = reg["dropout_label"].astype(float)
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:
        print(f"  Logit fit failed: {exc}")
        return None, len(reg)
    return (
        pd.DataFrame({
            "OR": np.exp(model.params),
            "CI_low": np.exp(model.conf_int()[0]),
            "CI_high": np.exp(model.conf_int()[1]),
            "p_value": model.pvalues,
        }),
        len(reg),
    )


if __name__ == "__main__":
    run()
