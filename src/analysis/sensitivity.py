"""
Sensitivity and robustness analyses (Multimedia Appendix 1).

Five checks that together test the main findings against the most obvious
threats to validity. They were previously five separate modules; they are
merged here because they share inputs, share the GEE specification (now in
config.fit_gee), and all feed a single appendix.

  1. E-values (VanderWeele & Ding 2017)   -> sensitivity_evalue.csv
  2. Cluster bootstrap CIs for the GEE    -> sensitivity_bootstrap.csv
  3. Incremental value of NLP features    -> sensitivity_nlp_value.csv
  4. Leave-one-module-out GEE replication -> sensitivity_lomo.csv
  5. Prospective day-7-only features      -> sensitivity_prospective_day7.csv

run() executes them in that order, which matters: the E-value check reads the
RQ1/RQ3 regression tables written earlier in the pipeline.

Inputs:  output/features/user_level_features.csv
         output/features/activity_level_features.csv
         output/analysis/tables/rq1_*.csv, rq3_logistic_regression.csv
         data/csv/facilitator_comments.csv, data/csv/discussions.csv
Outputs: output/analysis/tables/sensitivity_*.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from config import TABLE_DIR, fit_gee, load_data, save_csv
from gee_robustness import GEE_FEATS
from src.utils import parse_mixed_datetime

# ── 2. bootstrap ──
N_ITER = 2000
RANDOM_SEED = 42

# ── 4. leave-one-module-out ──
HEADLINE_FEATS = [
    "total_activities_submitted",
    "n_distinct_pages",
    "word_count_trend",
    "total_discussion_replies",
]

# ── 3. NLP incremental value ──
VOLUME_FEATS = [
    "activities_in_first_7d",
    "days_to_first_activity",
    "writing_span_days",
    "n_logins",
]
REGEX_FEATS = [
    "avg_vocab_richness",
    "avg_self_reference",
    "avg_future_orientation",
]
TRANSFORMER_FEATS = [
    "avg_sentiment",
    "sentiment_trend",
    "word_count_trend",
]

# ── 5. prospective day-7 ──
FEAT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "features"
CSV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "csv"
WINDOW_DAYS = 7
OBS_KEYS = ["module_id", "user_id", "cohort_id"]


# ══════════════════════════════════════════════════════════════════════
# 1. E-values for unmeasured confounding
# ══════════════════════════════════════════════════════════════════════
# For the two CONTINUOUS exposures, E-values are reported on the per-SD OR
# scale: a per-unit OR sits arbitrarily close to 1 when the unit is small
# relative to the spread, which makes the per-unit E-value uninformatively
# small by construction. For common outcomes, OR is converted to an
# approximate RR via RR = sqrt(OR) before applying the E-value formula.

def _evalue(rr: float) -> float:
    """E-value for a point estimate on the risk-ratio scale."""
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def _or_to_rr(odds_ratio: float) -> float:
    """Approximate RR from OR for common outcomes: RR ~ sqrt(OR)."""
    return np.sqrt(odds_ratio)


def _ci_closest_to_null(or_point: float, ci_low: float, ci_high: float) -> float:
    """Return the bound of the OR CI that sits closer to the null (1.0)."""
    return ci_low if or_point > 1.0 else ci_high


def _writer_or_from_chi2_table() -> tuple[float, float, float]:
    """Recover the 2x2 OR and Haldane-corrected 95% CI for writer vs non-writer."""
    ct = pd.read_csv(TABLE_DIR / "rq1_writer_chi2.csv", index_col=0)
    # Layout: rows = Non-writer / Writer; cols = Completer / Dropout
    a = ct.loc["Writer", "Completer"]
    b = ct.loc["Writer", "Dropout"]
    c = ct.loc["Non-writer", "Completer"]
    d = ct.loc["Non-writer", "Dropout"]
    # OR for completion given writer status (completer in numerator)
    vals = np.array([a, b, c, d], dtype=float) + 0.5  # Haldane correction
    or_point = (vals[0] * vals[3]) / (vals[1] * vals[2])
    se = np.sqrt(sum(1.0 / v for v in vals))
    ci_low = np.exp(np.log(or_point) - 1.96 * se)
    ci_high = np.exp(np.log(or_point) + 1.96 * se)
    return or_point, ci_low, ci_high


def run_evalue(data):
    df_feats, _, _ = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: E-values (VanderWeele & Ding 2017)")
    print("=" * 60)

    # SDs used to rescale per-unit ORs to per-SD ORs (computed on the
    # same samples the source regressions used: RQ1 = writers with
    # complete predictors, RQ3 = all starters).
    sd_act7 = float(
        df_feats.loc[
            df_feats["total_activities_submitted"] > 0, "activities_in_first_7d"
        ].std()
    )
    sd_replies = float(df_feats["total_discussion_replies"].std())

    rows = []

    or_w, ci_lo_w, ci_hi_w = _writer_or_from_chi2_table()
    rows.append({
        "finding": "Writer vs non-writer (chi-square)",
        "OR": or_w,
        "CI_low": ci_lo_w,
        "CI_high": ci_hi_w,
        "CI_closest_to_null": _ci_closest_to_null(or_w, ci_lo_w, ci_hi_w),
    })

    rq1 = pd.read_csv(TABLE_DIR / "rq1_logistic_regression.csv", index_col=0)
    r = rq1.loc["activities_in_first_7d"]
    or_sd = float(r["OR"]) ** sd_act7
    ci_lo_sd = float(r["CI_low"]) ** sd_act7
    ci_hi_sd = float(r["CI_high"]) ** sd_act7
    rows.append({
        "finding": f"Activities in first 7 days (RQ1 adj., per SD={sd_act7:.2f})",
        "OR": or_sd,
        "CI_low": ci_lo_sd,
        "CI_high": ci_hi_sd,
        "CI_closest_to_null": _ci_closest_to_null(or_sd, ci_lo_sd, ci_hi_sd),
    })

    rq3 = pd.read_csv(TABLE_DIR / "rq3_logistic_regression.csv", index_col=0)
    r = rq3.loc["total_discussion_replies"]
    or_sd = float(r["OR"]) ** sd_replies
    ci_lo_sd = float(r["CI_low"]) ** sd_replies
    ci_hi_sd = float(r["CI_high"]) ** sd_replies
    rows.append({
        "finding": f"Total discussion replies (RQ3 adj., per SD={sd_replies:.2f})",
        "OR": or_sd,
        "CI_low": ci_lo_sd,
        "CI_high": ci_hi_sd,
        "CI_closest_to_null": _ci_closest_to_null(or_sd, ci_lo_sd, ci_hi_sd),
    })

    for row in rows:
        rr_point = _or_to_rr(row["OR"])
        rr_ci = _or_to_rr(row["CI_closest_to_null"])
        row["evalue_point"] = round(_evalue(rr_point), 2)
        row["evalue_ci"] = round(_evalue(rr_ci), 2)
        row["OR"] = round(row["OR"], 3)
        row["CI_low"] = round(row["CI_low"], 3)
        row["CI_high"] = round(row["CI_high"], 3)
        row["CI_closest_to_null"] = round(row["CI_closest_to_null"], 3)

    out = pd.DataFrame(rows)
    print()
    print(out[["finding", "OR", "CI_closest_to_null", "evalue_point", "evalue_ci"]].to_string(index=False))
    save_csv(out, "sensitivity_evalue")


# ══════════════════════════════════════════════════════════════════════
# 2. Cluster bootstrap CIs for the GEE coefficients
# ══════════════════════════════════════════════════════════════════════
# The sandwich SE used by the main GEE is asymptotically justified in the
# number of clusters, not observations. With only 8 module-level clusters it
# may be anti-conservative (Li & Redden 2015 recommend 30+), so the GEE is
# refit on module-level resamples and percentile CIs reported.

def run_bootstrap(data):
    df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: Cluster Bootstrap CIs (GEE)")
    print("=" * 60)

    feats = [f for f in GEE_FEATS if f in writers.columns]
    gee_df = writers[["module_id", "dropout_label"] + feats].dropna().copy()
    modules = gee_df["module_id"].unique()
    n_clusters = len(modules)
    print(f"\nResampling {n_clusters} modules with replacement, {N_ITER:,} iterations")
    print(f"Sample: {len(gee_df):,} writers")

    # Pre-split by module for fast resampling
    by_module = {m: gee_df[gee_df["module_id"] == m] for m in modules}

    # Point estimates from the main GEE (for SE ratio reporting)
    main_result = fit_gee(gee_df, feats, "module_id")
    main_coefs = main_result.params.drop("const", errors="ignore")
    main_bse = main_result.bse.drop("const", errors="ignore")

    rng = np.random.default_rng(RANDOM_SEED)
    coef_samples = {f: [] for f in feats}
    n_converged = 0

    # Scope the warning suppression to the bootstrap loop only — a
    # module-level filterwarnings("ignore") would leak into every later
    # script in run_all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_ITER):
            draws = rng.choice(modules, size=n_clusters, replace=True)
            pieces = []
            for k, mod_id in enumerate(draws):
                piece = by_module[mod_id].copy()
                piece["_boot_cluster"] = k  # fresh cluster id for duplicates
                pieces.append(piece)
            boot_df = pd.concat(pieces, ignore_index=True)

            try:
                result = fit_gee(boot_df, feats, "_boot_cluster")
            except Exception:
                continue
            if not getattr(result, "converged", True):
                continue

            params = result.params.drop("const", errors="ignore")
            for f in feats:
                if f in params.index:
                    coef_samples[f].append(params[f])
            n_converged += 1

            if (i + 1) % 500 == 0:
                print(f"  iter {i + 1:,}/{N_ITER:,} (converged: {n_converged:,})")

    print(f"  converged: {n_converged:,}/{N_ITER:,}")
    if n_converged < 0.95 * N_ITER:
        print("  WARNING: <95% of bootstrap fits converged; percentile CIs "
              "may be biased by convergence selection.")

    rows = []
    for f in feats:
        samples = np.array(coef_samples[f])
        if len(samples) == 0:
            continue
        ci_low_coef = np.percentile(samples, 2.5)
        ci_high_coef = np.percentile(samples, 97.5)
        boot_se = float(np.std(samples, ddof=1))
        sandwich_se = float(main_bse.get(f, np.nan))
        rows.append({
            "feature": f,
            "coef": float(main_coefs.get(f, np.nan)),
            "OR": float(np.exp(main_coefs.get(f, np.nan))),
            "sandwich_se": sandwich_se,
            "bootstrap_se": boot_se,
            "se_ratio_boot_over_sandwich": (
                boot_se / sandwich_se if sandwich_se > 0 else np.nan
            ),
            "bootstrap_OR_CI_low": float(np.exp(ci_low_coef)),
            "bootstrap_OR_CI_high": float(np.exp(ci_high_coef)),
            "boot_ci_excludes_null": bool(
                ci_high_coef < 0 or ci_low_coef > 0
            ),
            "n_converged": n_converged,
        })

    out = pd.DataFrame(rows)
    print("\n--- Bootstrap Results (percentile 95% CI on OR scale) ---")
    for _, r in out.iterrows():
        marker = " *" if r["boot_ci_excludes_null"] else ""
        print(
            f"  {r['feature']:35s}: OR={r['OR']:.3f} "
            f"[{r['bootstrap_OR_CI_low']:.3f}, {r['bootstrap_OR_CI_high']:.3f}] "
            f"SE_boot/SE_sw={r['se_ratio_boot_over_sandwich']:.2f}{marker}"
        )

    save_csv(out, "sensitivity_bootstrap")


# ══════════════════════════════════════════════════════════════════════
# 3. Incremental value of NLP features over volume-only
# ══════════════════════════════════════════════════════════════════════
# Nested logistic models on the writer subsample:
#   M0 volume+timing -> M1 (+RegEx) -> M2 (+transformer). Compared by
#   Delta-AIC and likelihood-ratio tests.

def _fit_model(df: pd.DataFrame, feats: list[str]):
    """Fit logistic regression of dropout_label on feats + course dummies."""
    X = df[feats].copy()
    X = pd.concat(
        [X, pd.get_dummies(df["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X)
    y = df["dropout_label"]
    return sm.Logit(y, X).fit(disp=0)


def _lr_test(res_reduced, res_full) -> tuple[float, int, float]:
    """Likelihood-ratio chi-square test between nested models."""
    lr = 2 * (res_full.llf - res_reduced.llf)
    df_diff = int(res_full.df_model - res_reduced.df_model)
    p = float(sp_stats.chi2.sf(lr, df_diff))
    return float(lr), df_diff, p


def run_nlp_value(data):
    df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: NLP incremental value (Delta-AIC)")
    print("=" * 60)

    needed = VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS + ["dropout_label", "course_name"]
    sub = writers[needed].dropna().copy()
    print(f"\nSample: {len(sub):,} writers with complete NLP features")

    m0 = _fit_model(sub, VOLUME_FEATS)
    m1 = _fit_model(sub, VOLUME_FEATS + REGEX_FEATS)
    m2 = _fit_model(sub, VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS)

    lr_01_stat, lr_01_df, lr_01_p = _lr_test(m0, m1)
    lr_12_stat, lr_12_df, lr_12_p = _lr_test(m1, m2)
    lr_02_stat, lr_02_df, lr_02_p = _lr_test(m0, m2)

    rows = [
        {
            "model": "M0 (volume + timing)",
            "features_added": ", ".join(VOLUME_FEATS),
            "n_params": int(m0.df_model) + 1,
            "log_likelihood": float(m0.llf),
            "AIC": float(m0.aic),
            "pseudo_R2": float(m0.prsquared),
            "delta_AIC_vs_M0": 0.0,
            "LR_stat_vs_previous": np.nan,
            "LR_df": np.nan,
            "LR_p_value": np.nan,
        },
        {
            "model": "M1 (+ RegEx linguistic)",
            "features_added": ", ".join(REGEX_FEATS),
            "n_params": int(m1.df_model) + 1,
            "log_likelihood": float(m1.llf),
            "AIC": float(m1.aic),
            "pseudo_R2": float(m1.prsquared),
            "delta_AIC_vs_M0": float(m1.aic - m0.aic),
            "LR_stat_vs_previous": lr_01_stat,
            "LR_df": lr_01_df,
            "LR_p_value": lr_01_p,
        },
        {
            "model": "M2 (+ transformer NLP)",
            "features_added": ", ".join(TRANSFORMER_FEATS),
            "n_params": int(m2.df_model) + 1,
            "log_likelihood": float(m2.llf),
            "AIC": float(m2.aic),
            "pseudo_R2": float(m2.prsquared),
            "delta_AIC_vs_M0": float(m2.aic - m0.aic),
            "LR_stat_vs_previous": lr_12_stat,
            "LR_df": lr_12_df,
            "LR_p_value": lr_12_p,
        },
    ]

    out = pd.DataFrame(rows)
    print()
    print(out[[
        "model", "n_params", "AIC", "pseudo_R2",
        "delta_AIC_vs_M0", "LR_stat_vs_previous", "LR_df", "LR_p_value",
    ]].to_string(index=False))

    print()
    print(f"  LR test M0 -> M1 (+ RegEx):        chi2({lr_01_df}) = {lr_01_stat:.2f}, p = {lr_01_p:.4g}")
    print(f"  LR test M1 -> M2 (+ transformers): chi2({lr_12_df}) = {lr_12_stat:.2f}, p = {lr_12_p:.4g}")
    print(f"  LR test M0 -> M2 (all NLP):        chi2({lr_02_df}) = {lr_02_stat:.2f}, p = {lr_02_p:.4g}")

    save_csv(out, "sensitivity_nlp_value")


# ══════════════════════════════════════════════════════════════════════
# 4. Leave-one-module-out (LOMO) replication of the GEE
# ══════════════════════════════════════════════════════════════════════
# Refits the GEE once per module, holding that module out, and reports the OR
# and sandwich CI for each headline feature. A finding is LOMO-stable if its
# OR and sign survive every held-out fit.

def run_lomo(data):
    df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: Leave-one-module-out (LOMO) GEE")
    print("=" * 60)

    feats = [f for f in GEE_FEATS if f in writers.columns]
    gee_df = writers[["module_id", "dropout_label"] + feats].dropna().copy()
    modules = sorted(gee_df["module_id"].unique())
    print(f"\nHolding out each of {len(modules)} modules in turn; {len(feats)} features per fit.")

    rows = []
    # Scope the warning suppression to the refit loop only — a module-level
    # filterwarnings("ignore") would leak into every later script in run_all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for held_out in modules:
            sub = gee_df[gee_df["module_id"] != held_out].copy()
            try:
                result = fit_gee(sub, feats, "module_id")
            except Exception as exc:
                print(f"  Module {held_out}: fit failed ({exc})")
                continue
            params = result.params.drop("const", errors="ignore")
            conf = result.conf_int().drop("const", errors="ignore")
            for feat in HEADLINE_FEATS:
                if feat in params.index:
                    coef = float(params[feat])
                    ci_low = float(conf.loc[feat, 0])
                    ci_high = float(conf.loc[feat, 1])
                    rows.append({
                        "held_out_module": int(held_out),
                        "feature": feat,
                        "coef": coef,
                        "OR": float(np.exp(coef)),
                        "OR_CI_low": float(np.exp(ci_low)),
                        "OR_CI_high": float(np.exp(ci_high)),
                        "p_value": float(result.pvalues.get(feat, np.nan)),
                        "n_used": int(len(sub)),
                        "n_clusters": int(sub["module_id"].nunique()),
                    })

    out = pd.DataFrame(rows)
    save_csv(out, "sensitivity_lomo")

    print("\n--- LOMO OR range per headline feature ---")
    for feat in HEADLINE_FEATS:
        block = out[out["feature"] == feat]
        if block.empty:
            continue
        or_min = block["OR"].min()
        or_max = block["OR"].max()
        p_max = block["p_value"].max()
        ci_excludes_null = (
            ((block["OR_CI_high"] < 1) | (block["OR_CI_low"] > 1)).all()
        )
        marker = " *" if ci_excludes_null else ""
        print(
            f"  {feat:35s}: OR range [{or_min:.3f}, {or_max:.3f}], "
            f"max p = {p_max:.4f}{marker}"
        )


# ══════════════════════════════════════════════════════════════════════
# 5. Prospective day-7-only features
# ══════════════════════════════════════════════════════════════════════
# Rebuilds a strictly prospective feature matrix in which every predictor is
# observable by end of day 7, addressing the concern that many main-analysis
# features grow mechanically with retention.
#
# Deliberately EXCLUDED because they cannot be windowed from this export:
#   n_logins             whole-enrolment count; no per-login timestamps exist
#   page-visit features  per-page records carry only the LATEST visit
#                        timestamp, so a first-week filter is contaminated

def _window_agg(
    df: pd.DataFrame,
    join_keys: list[str],
    start_lookup: pd.DataFrame,
    ts_col: str,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Filter df rows to those whose ts_col is within [0, window_days) days of start."""
    merged = df.merge(start_lookup, on=join_keys, how="inner")
    merged["_delta_days"] = (merged[ts_col] - merged["started"]).dt.total_seconds() / 86400
    return merged[
        (merged["_delta_days"] >= 0) & (merged["_delta_days"] < window_days)
    ].copy()


def run_prospective_day7(data):
    df, writers, groups = data

    print("\n" + "=" * 60)
    print(f"Sensitivity Analysis: Prospective day-{WINDOW_DAYS} features only")
    print("=" * 60)

    base = df[OBS_KEYS + ["started", "dropout_label", "course_name"]].copy()
    base["started"] = parse_mixed_datetime(base["started"])
    start_lookup = base[OBS_KEYS + ["started"]].copy()

    # ── Activities in first 7 days (module, user, cohort keys) ─────
    act = pd.read_csv(FEAT_DIR / "activity_level_features.csv")
    act = act[act["type_name"] != "Emotions"]  # word-cloud is not writing
    act["recorded"] = parse_mixed_datetime(act["recorded"])
    act_window = _window_agg(act, OBS_KEYS, start_lookup, "recorded")
    act_agg = (
        act_window.groupby(OBS_KEYS, dropna=False)
        .agg(
            n_activities_first_7d=("activity_id", "count"),
            n_words_first_7d=("word_count", "sum"),
            mean_sentiment_first_7d=("compound_score", "mean"),
        )
        .reset_index()
    )

    # ── Facilitator comments in first 7 days ───────────────────────
    # Join via activity_id -> OBS_KEYS in activities table.
    fc = pd.read_csv(CSV_DIR / "facilitator_comments.csv")
    fc["recorded"] = parse_mixed_datetime(fc["recorded"])
    fc = fc[["activity_id", "recorded"]]  # drop fc's own module_id
    act_key_lookup = act[OBS_KEYS + ["activity_id"]].drop_duplicates()
    fc_keyed = fc.merge(act_key_lookup, on="activity_id", how="inner")
    fc_window = _window_agg(fc_keyed, OBS_KEYS, start_lookup, "recorded")
    fc_agg = (
        fc_window.groupby(OBS_KEYS, dropna=False)
        .size()
        .rename("n_comments_first_7d")
        .reset_index()
    )

    # ── Forum replies in first 7 days ──────────────────────────────
    # discussions.csv has module_id + user_id but no cohort_id. We join on the
    # two available keys; the activity-level data already disambiguates at the
    # user level within a module, so this is acceptable for a window count.
    disc = pd.read_csv(CSV_DIR / "discussions.csv")
    disc["recorded"] = parse_mixed_datetime(disc["recorded"])
    user_mod_lookup = base[["module_id", "user_id", "cohort_id", "started"]].drop_duplicates()
    disc_keyed = disc.merge(
        user_mod_lookup, on=["module_id", "user_id"], how="inner",
    )
    disc_keyed["_delta_days"] = (disc_keyed["recorded"] - disc_keyed["started"]).dt.total_seconds() / 86400
    disc_window = disc_keyed[
        (disc_keyed["_delta_days"] >= 0) & (disc_keyed["_delta_days"] < WINDOW_DAYS)
    ]
    disc_agg = (
        disc_window.groupby(OBS_KEYS, dropna=False)
        .size()
        .rename("n_forum_replies_first_7d")
        .reset_index()
    )

    # ── Assemble prospective feature table ─────────────────────────
    prosp = (
        base.merge(act_agg, on=OBS_KEYS, how="left")
            .merge(fc_agg, on=OBS_KEYS, how="left")
            .merge(disc_agg, on=OBS_KEYS, how="left")
    )
    for c in ["n_activities_first_7d", "n_words_first_7d",
              "n_comments_first_7d", "n_forum_replies_first_7d"]:
        prosp[c] = prosp[c].fillna(0)
    prosp["mean_sentiment_first_7d"] = prosp["mean_sentiment_first_7d"].fillna(0)
    prosp["wrote_in_first_week"] = (prosp["n_activities_first_7d"] > 0).astype(int)
    prosp["received_comment_first_7d"] = (prosp["n_comments_first_7d"] > 0).astype(int)
    prosp["posted_in_first_week"] = (prosp["n_forum_replies_first_7d"] > 0).astype(int)

    predictors = [
        "n_activities_first_7d",
        "wrote_in_first_week",
        "n_words_first_7d",
        "mean_sentiment_first_7d",
        "received_comment_first_7d",
        "posted_in_first_week",
    ]
    reg_df = prosp[predictors + ["dropout_label", "course_name"]].dropna()
    print(f"\nSample: {len(reg_df):,} starters (all features observable by day 7)")
    print(f"Predictors ({len(predictors)}): {', '.join(predictors)}")

    X = reg_df[predictors].copy()
    X = pd.concat(
        [X, pd.get_dummies(reg_df["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X)
    y = reg_df["dropout_label"]

    model = sm.Logit(y, X).fit(disp=0)
    summary = pd.DataFrame({
        "OR": np.exp(model.params),
        "CI_low": np.exp(model.conf_int()[0]),
        "CI_high": np.exp(model.conf_int()[1]),
        "p_value": model.pvalues,
    }).loc[predictors]

    print("\n--- Prospective (day-7) Logistic Regression ---")
    for feat in predictors:
        r = summary.loc[feat]
        sig = " *" if r["p_value"] < 0.05 else ""
        print(
            f"  {feat:32s}: OR={r['OR']:.3f} "
            f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
            f"p={r['p_value']:.4f}{sig}"
        )

    print(f"\n  Pseudo R2 = {model.prsquared:.4f}, N = {len(reg_df):,}")
    save_csv(summary, "sensitivity_prospective_day7")


# ══════════════════════════════════════════════════════════════════════

def run(data=None):
    """Run all five checks in the order the pipeline requires.

    run_evalue must come first: it reads the RQ1/RQ3 regression tables written
    earlier in run_all.
    """
    if data is None:
        data = load_data()

    run_evalue(data)
    run_bootstrap(data)
    run_nlp_value(data)
    run_lomo(data)
    run_prospective_day7(data)


if __name__ == "__main__":
    run()
