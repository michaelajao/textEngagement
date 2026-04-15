"""
Rebuild the full pipeline with reclassified population:
  1. Reclassify 36 active non-starters as starters
  2. Rebuild v2 features
  3. Rerun all analyses
  4. Save all figures and tables

Usage:
  cd textEngagement/src/analysis_v2
  python rebuild_all.py
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    compute_dropout_label, parse_mixed_datetime,
    assign_discussion_cohorts, apply_publication_style, PALETTE,
)

CSV_DIR = ROOT / "data" / "csv"
FEAT_DIR = ROOT / "output" / "features"
OUT_DIR = ROOT / "output" / "analysis_v2"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for d in [FEAT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

apply_publication_style()
warnings.filterwarnings("ignore")

OBS_KEYS = ["module_id", "user_id", "cohort_id"]
ORIGINALS = {"n_logins", "login_span_days", "n_bookmarks", "n_page_visits",
             "n_distinct_pages", "total_activities_submitted",
             "total_comments_received", "total_discussion_replies"}


def save_csv(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True)
    print(f"  Saved: {path.name}")


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD RAW DATA & RECLASSIFY                        ║
# ╚══════════════════════════════════════════════════════════════╝

def step1_load_and_reclassify():
    print("\n" + "=" * 60)
    print("STEP 1: Load Raw Data & Reclassify")
    print("=" * 60)

    users = pd.read_csv(CSV_DIR / "users.csv")
    activities = pd.read_csv(CSV_DIR / "activities.csv")
    fc_comments = pd.read_csv(CSV_DIR / "facilitator_comments.csv")
    discussions = pd.read_csv(CSV_DIR / "discussions.csv")
    page_visits = pd.read_csv(CSV_DIR / "page_visits.csv")

    for col in ["started", "finished"]:
        users[col] = parse_mixed_datetime(users[col])
    activities["recorded"] = parse_mixed_datetime(activities["recorded"])
    fc_comments["recorded"] = parse_mixed_datetime(fc_comments["recorded"])
    discussions["recorded"] = parse_mixed_datetime(discussions["recorded"])
    page_visits["latest"] = parse_mixed_datetime(page_visits["latest"])

    activities = activities.drop_duplicates(subset=["activity_id"], keep="last")
    fc_comments = fc_comments.drop_duplicates(subset=["comment_id"], keep="last")
    users = users.sort_values("started", na_position="first").drop_duplicates(subset=OBS_KEYS, keep="last")

    # Identify non-starters with behavioural evidence
    not_started = users[users["started"].isna()]
    ns_pairs = set(zip(not_started["module_id"], not_started["user_id"]))

    # Writers among non-starters
    ns_writers = activities[activities.apply(
        lambda r: (r["module_id"], r["user_id"]) in ns_pairs, axis=1
    )]
    writer_pairs = set(zip(ns_writers["module_id"], ns_writers["user_id"]))

    # Forum posters among non-starters
    ns_posters = discussions[discussions["user_id"].isin(not_started["user_id"])]
    poster_ids = set(ns_posters["user_id"])

    # Reclassify: impute start date from first activity or first forum post
    reclassified = 0
    for idx, row in users.iterrows():
        if pd.notna(row["started"]):
            continue
        mid, uid = row["module_id"], row["user_id"]
        is_active = (mid, uid) in writer_pairs or uid in poster_ids
        if not is_active:
            continue

        # Impute start date from earliest behavioural timestamp
        timestamps = []
        acts = ns_writers[(ns_writers["module_id"] == mid) & (ns_writers["user_id"] == uid)]
        if len(acts) > 0:
            timestamps.append(acts["recorded"].min())
        posts = ns_posters[ns_posters["user_id"] == uid]
        if len(posts) > 0:
            timestamps.append(posts["recorded"].min())
        if timestamps:
            users.at[idx, "started"] = min(timestamps)
            reclassified += 1

    users["dropout_label"] = compute_dropout_label(users)
    starters = users[users["started"].notna()].copy()

    n_total = len(users)
    n_started = len(starters)
    n_excluded = n_total - n_started

    print(f"  Total enrolment records: {n_total:,}")
    print(f"  Reclassified (active non-starters): {reclassified}")
    print(f"  Final starters: {n_started:,}")
    print(f"  Excluded (true non-initiators): {n_excluded:,}")
    print(f"  Completers: {(starters['dropout_label'] == 0).sum():,}")
    print(f"  Dropouts: {(starters['dropout_label'] == 1).sum():,}")

    # Save non-initiator characterisation
    excluded = users[users["started"].isna()]
    excl_summary = pd.DataFrame([{
        "total_excluded": len(excluded),
        "any_logins": (excluded["n_logins"] > 0).sum(),
        "any_page_visits": (excluded["n_page_visits"] > 0).sum(),
        "mean_logins": round(excluded["n_logins"].mean(), 1),
        "mean_page_visits": round(excluded["n_page_visits"].mean(), 1),
    }])
    save_csv(excl_summary, "excluded_non_initiators")

    return users, starters, activities, fc_comments, discussions, page_visits


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 2: BUILD v2 FEATURES                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def step2_build_features(starters, activities, fc_comments, discussions, page_visits):
    print("\n" + "=" * 60)
    print("STEP 2: Build v2 Features")
    print("=" * 60)

    base = starters[OBS_KEYS + ["course_name", "cohort_name", "started", "finished", "dropout_label"]].copy()

    # Load pre-computed activity-level NLP features
    act_df = pd.read_csv(FEAT_DIR / "activity_level_features.csv")
    act_df["recorded"] = parse_mixed_datetime(act_df["recorded"])

    # ── Platform features ──
    platform = starters[OBS_KEYS + ["n_logins", "login_span_days", "n_bookmarks",
                                     "n_page_visits", "n_distinct_pages"]].copy()

    def categorise_page(url):
        if pd.isna(url): return "other"
        u = str(url).lower()
        if "dashboard" in u: return "dashboard"
        if any(k in u for k in ["/activities", "/newsfeed", "/gratitude", "/goals"]): return "activity"
        if any(k in u for k in ["/session", "/journey", "/learning", "/welcome", "/further-resources"]): return "content"
        if any(k in u for k in ["/discussion", "/forum", "/topic"]): return "forum"
        if any(k in u for k in ["/profile", "/personal", "/notifications"]): return "profile"
        return "other"

    page_visits["page_category"] = page_visits["url"].apply(categorise_page)
    pv = page_visits.merge(base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="inner")
    pv["days_since_start"] = (pv["latest"] - pv["started"]).dt.total_seconds() / 86400

    pv_depth = pv.groupby(OBS_KEYS).agg(
        pv_mean_duration=("avg_duration", "mean"),
        pv_mean_hits_per_page=("hits", "mean"),
    ).reset_index()

    cat_hits = pv.groupby(OBS_KEYS + ["page_category"])["hits"].sum().unstack(fill_value=0)
    cat_pct = cat_hits.div(cat_hits.sum(axis=1), axis=0)
    cat_pct.columns = [f"pv_pct_{c}" for c in cat_pct.columns]
    cat_pct = cat_pct.reset_index()

    early_pv = pv[pv["days_since_start"].between(0, 6)]
    pv_early = early_pv.groupby(OBS_KEYS)["url"].nunique().rename("pv_pages_first_7d").reset_index()

    pv_all = pv_depth.merge(cat_pct, on=OBS_KEYS, how="left").merge(pv_early, on=OBS_KEYS, how="left")
    platform = platform.merge(pv_all, on=OBS_KEYS, how="left")
    for c in platform.columns:
        if c not in OBS_KEYS:
            platform[c] = pd.to_numeric(platform[c], errors="coerce").fillna(0)

    # ── Writing features ──
    def _linear_slope(values):
        n = len(values)
        if n < 2: return np.nan
        slope, _, _, _, _ = sp_stats.linregress(np.arange(n, dtype=float), values)
        return float(slope)

    def _shannon_entropy(counts):
        total = counts.sum()
        if total == 0: return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs)))

    g = act_df.groupby(OBS_KEYS, dropna=False)

    volume = g.agg(
        total_activities_submitted=("activity_id", "count"),
        first_activity=("recorded", "min"),
        last_activity=("recorded", "max"),
    ).reset_index()
    volume["writing_span_days"] = ((volume["last_activity"] - volume["first_activity"]).dt.total_seconds() / 86400).fillna(0)
    vol_start = volume.merge(base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left")
    volume["days_to_first_activity"] = (vol_start["first_activity"] - vol_start["started"]).dt.total_seconds() / 86400
    volume = volume.drop(columns=["first_activity", "last_activity"])

    acts_start = act_df.merge(base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left")
    acts_start["days_since_start"] = (acts_start["recorded"] - acts_start["started"]).dt.total_seconds() / 86400
    ew_7 = acts_start[acts_start["days_since_start"].between(0, 6)].groupby(OBS_KEYS)["activity_id"].count().rename("activities_in_first_7d").reset_index()

    quality = g.agg(avg_vocab_richness=("vocab_richness", "mean")).reset_index()

    def _vocab_evolution(grp):
        grp = grp.sort_values("recorded")
        if len(grp) < 4: return np.nan
        mid = len(grp) // 2
        return grp.iloc[mid:]["vocab_richness"].mean() - grp.iloc[:mid]["vocab_richness"].mean()

    vocab_evo = act_df.groupby(OBS_KEYS, dropna=False).apply(_vocab_evolution).rename("vocab_evolution").reset_index()

    linguistic = g.agg(
        avg_self_reference=("self_reference_ratio", "mean"),
        avg_future_orientation=("future_orientation", "mean"),
        avg_sentiment=("compound_score", "mean"),
    ).reset_index()

    TYPES = ["Gratitude", "Emotions"]
    def _type_features(grp):
        total = len(grp)
        counts = grp["type_name"].value_counts()
        pcts = {f"pct_{t.lower()}": counts.get(t, 0) / total if total > 0 else 0.0 for t in TYPES}
        all_counts = np.array([counts.get(t, 0) for t in ["Gratitude", "GoalSetting", "Emotions"]])
        pcts["activity_type_entropy"] = _shannon_entropy(all_counts)
        return pd.Series(pcts)

    type_div = act_df.groupby(OBS_KEYS, dropna=False).apply(_type_features).reset_index()

    def _trajectory(grp):
        grp = grp.sort_values("recorded")
        n = len(grp)
        wc_trend = _linear_slope(grp["word_count"].values) if n >= 2 else np.nan
        sent_trend = _linear_slope(grp["compound_score"].values) if n >= 2 else np.nan
        if n >= 2:
            gaps = grp["recorded"].diff().dt.total_seconds().dropna() / 86400
            gap_std = gaps.std()
            regularity = 1.0 / (1.0 + gap_std) if not np.isnan(gap_std) else np.nan
        else:
            regularity = np.nan
        return pd.Series({"word_count_trend": wc_trend, "sentiment_trend": sent_trend, "activity_regularity": regularity})

    trajectories = act_df.groupby(OBS_KEYS, dropna=False).apply(_trajectory).reset_index()

    writing = volume.copy()
    for df in [quality, vocab_evo, linguistic, type_div, trajectories, ew_7]:
        writing = writing.merge(df, on=OBS_KEYS, how="left")

    # ── Facilitator features ──
    fac = g.agg(total_comments_received=("num_comments", "sum")).reset_index()

    def _continued(grp):
        commented = grp[grp["has_comment"] == 1]
        if commented.empty: return np.nan
        return 1.0 if len(grp[grp["recorded"] > commented["recorded"].min()]) > 0 else 0.0

    cont = act_df.groupby(OBS_KEYS, dropna=False).apply(_continued).rename("continued_after_comment").reset_index()
    fac = fac.merge(cont, on=OBS_KEYS, how="left")

    pairs_df = pd.read_csv(FEAT_DIR / "comment_pairs.csv")
    if not pairs_df.empty:
        pairs_df["activity_recorded"] = parse_mixed_datetime(pairs_df["activity_recorded"])
        pairs_df["comment_recorded"] = parse_mixed_datetime(pairs_df["comment_recorded"])
        pairs_df["response_hours"] = ((pairs_df["comment_recorded"] - pairs_df["activity_recorded"]).dt.total_seconds() / 3600).clip(lower=0)
        act_cohort = act_df[["activity_id", "cohort_id"]].drop_duplicates()
        pairs_df = pairs_df.merge(act_cohort, on="activity_id", how="left")
        cp = pairs_df.groupby(OBS_KEYS, dropna=False).agg(
            avg_response_hours=("response_hours", "mean"),
            avg_comment_word_count=("comment_word_count", "mean"),
        ).reset_index()
        fac = fac.merge(cp, on=OBS_KEYS, how="left")

    # ── Forum features (computed from raw discussions table) ──
    dt = assign_discussion_cohorts(
        discussions,
        base[OBS_KEYS + ["cohort_name", "started"]].rename(columns={"cohort_name": "cohort_name"}),
    )
    dt["recorded"] = parse_mixed_datetime(dt["recorded"])
    dt = dt[dt["cohort_id"].notna()].copy()

    # Word count and basic aggregation
    dt["d_word_count"] = dt["comment"].fillna("").apply(lambda t: len(str(t).split()))

    dg = (
        dt.groupby(OBS_KEYS, dropna=False)
        .agg(
            total_discussion_replies=("reply_id", "count"),
            first_post=("recorded", "min"),
            last_post=("recorded", "max"),
        )
        .reset_index()
    )
    dg["forum_span_days"] = (
        (dg["last_post"] - dg["first_post"]).dt.total_seconds() / 86400
    ).fillna(0)

    # days_to_first_post
    dg_start = dg.merge(base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left")
    dg["days_to_first_post"] = (
        (dg_start["first_post"] - dg_start["started"]).dt.total_seconds() / 86400
    )
    dg = dg.drop(columns=["first_post", "last_post"])

    # Forum sentiment — use pre-computed v2 if available, otherwise set to NaN
    # (BERT sentiment requires GPU; we avoid re-running it here)
    v2_path = FEAT_DIR / "user_level_features_v2.csv"
    if v2_path.exists():
        v2_prev = pd.read_csv(v2_path)
        if "forum_sentiment_mean" in v2_prev.columns:
            sent_cols = v2_prev[OBS_KEYS + ["forum_sentiment_mean"]].drop_duplicates(subset=OBS_KEYS)
            dg = dg.merge(sent_cols, on=OBS_KEYS, how="left")
        else:
            dg["forum_sentiment_mean"] = np.nan
    else:
        dg["forum_sentiment_mean"] = np.nan

    forum = dg.copy()

    # ── Merge all ──
    features = base.copy()
    for name, df in [("platform", platform), ("writing", writing), ("facilitator", fac), ("forum", forum)]:
        features = features.merge(df, on=OBS_KEYS, how="left")

    fill_zero = ["total_activities_submitted", "total_comments_received", "total_discussion_replies",
                  "activities_in_first_7d", "n_logins", "n_bookmarks", "n_page_visits", "n_distinct_pages", "pv_pages_first_7d"]
    for c in fill_zero:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0)
    for c in features.columns:
        if "pv_pct_" in c:
            features[c] = features[c].fillna(0)

    META = OBS_KEYS + ["course_name", "cohort_name", "started", "finished", "dropout_label"]
    FEAT_COLS = [c for c in features.columns if c not in META]

    # Save
    features.to_csv(FEAT_DIR / "user_level_features_v2.csv", index=False)

    pf_cols = [c for c in platform.columns if c not in OBS_KEYS]
    wf_cols = [c for c in writing.columns if c not in OBS_KEYS]
    ff_cols = [c for c in fac.columns if c not in OBS_KEYS]
    sf_cols = [c for c in forum.columns if c not in OBS_KEYS]

    groups = {
        "platform": [c for c in pf_cols if c in FEAT_COLS],
        "writing": [c for c in wf_cols if c in FEAT_COLS],
        "facilitator": [c for c in ff_cols if c in FEAT_COLS],
        "forum": [c for c in sf_cols if c in FEAT_COLS],
        "all_features": FEAT_COLS,
        "originals": sorted(ORIGINALS),
        "meta": META,
    }
    with open(FEAT_DIR / "feature_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    print(f"  Features: {len(FEAT_COLS)} ({len(ORIGINALS)} original + {len(FEAT_COLS) - len(ORIGINALS)} engineered)")
    print(f"  Saved: user_level_features_v2.csv ({features.shape[0]:,} rows)")
    print(f"  Saved: feature_groups.json")

    return features, groups


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 3: RUN ALL ANALYSES                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def step3_analyses(features, groups):
    print("\n" + "=" * 60)
    print("STEP 3: Run All Analyses")
    print("=" * 60)

    META = groups["meta"]
    ALL_FEATS = groups["all_features"]
    df = features.copy()

    # Duration for survival
    df["duration_days"] = np.where(
        df["finished"].notna(),
        (df["finished"] - df["started"]).dt.total_seconds() / 86400,
        df["writing_span_days"].fillna(0) + df["days_to_first_activity"].fillna(1),
    )
    df["duration_days"] = df["duration_days"].clip(lower=1)
    df["is_writer"] = (df["total_activities_submitted"] > 0).astype(int)
    df["received_comment"] = (df["total_comments_received"] > 0).astype(int)
    df["is_poster"] = (df["total_discussion_replies"] > 0).astype(int)
    writers = df[df["is_writer"] == 1].copy()

    n_total = len(df)
    n_comp = (df["dropout_label"] == 0).sum()
    n_drop = (df["dropout_label"] == 1).sum()
    print(f"  Participants: {n_total:,} (completers: {n_comp:,}, dropouts: {n_drop:,})")
    print(f"  Writers: {len(writers):,}")

    # ── RQ1: Writer vs Non-Writer ──
    print("\n--- RQ1: Writer vs Non-Writer ---")
    ct = pd.crosstab(df["is_writer"], df["dropout_label"])
    ct.index = ["Non-writer", "Writer"]
    ct.columns = ["Completer", "Dropout"]
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(ct)
    vals = ct.values.flatten() + 0.5
    OR = (vals[0] * vals[3]) / (vals[1] * vals[2])
    wr_comp = ct.loc["Writer", "Completer"] / ct.loc["Writer"].sum() * 100
    nw_comp = ct.loc["Non-writer", "Completer"] / ct.loc["Non-writer"].sum() * 100
    print(f"  Writer: {wr_comp:.1f}% | Non-writer: {nw_comp:.1f}% | Chi2={chi2:.1f} | OR={OR:.2f}")
    save_csv(ct, "rq1_writer_chi2")

    # ── Univariate Mann-Whitney ──
    print("\n--- Univariate Comparisons ---")
    numeric = df[ALL_FEATS].select_dtypes(include=[np.number]).columns.tolist()
    comp = df[df["dropout_label"] == 0]
    drop = df[df["dropout_label"] == 1]
    mw_rows = []
    for col in numeric:
        c_vals, d_vals = comp[col].dropna(), drop[col].dropna()
        if len(c_vals) < 2 or len(d_vals) < 2: continue
        stat, pv = sp_stats.mannwhitneyu(c_vals, d_vals, alternative="two-sided")
        r = 1 - (2 * stat) / (len(c_vals) * len(d_vals))
        mw_rows.append({"feature": col, "type": "Original" if col in ORIGINALS else "Engineered",
                         "compl_median": round(c_vals.median(), 3), "drop_median": round(d_vals.median(), 3),
                         "rank_biserial_r": round(r, 3), "p_value": pv})
    mw_df = pd.DataFrame(mw_rows)
    _, p_adj, _, _ = multipletests(mw_df["p_value"], method="fdr_bh")
    mw_df["p_adj"] = p_adj
    mw_df["significant"] = mw_df["p_adj"] < 0.05
    mw_df = mw_df.sort_values("rank_biserial_r")
    print(f"  {mw_df['significant'].sum()}/{len(mw_df)} significant after BH-FDR")
    save_csv(mw_df, "rq1_mann_whitney")

    # ── Dose-response ──
    print("\n--- Dose-Response ---")
    bins = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5, df["total_activities_submitted"].max() + 1]
    labels = ["0", "1", "2-3", "4-5", "6-10", "11-20", "21+"]
    df["act_bin"] = pd.cut(df["total_activities_submitted"], bins=bins, labels=labels, right=False)
    bin_stats = df.groupby("act_bin", observed=True).agg(
        n=("dropout_label", "count"), completers=("dropout_label", lambda x: (x == 0).sum()),
    ).reset_index()
    bin_stats["completion_pct"] = bin_stats["completers"] / bin_stats["n"] * 100
    rho, p_trend = sp_stats.spearmanr(range(len(bin_stats)), bin_stats["completion_pct"])
    print(f"  Spearman rho = {rho:.3f}, p = {p_trend:.4f}")
    save_csv(bin_stats, "rq1_dose_response")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bin_stats["act_bin"], bin_stats["completion_pct"], color=PALETTE["blue"], edgecolor="white")
    for bar, n in zip(bars, bin_stats["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"n={n:,}", ha="center", fontsize=8)
    ax.set_xlabel("Total Activities Submitted"); ax.set_ylabel("Completion Rate (%)")
    ax.set_title(f"Dose-Response (Spearman rho={rho:.2f})"); ax.set_ylim(0, 105)
    save_fig(fig, "fig_rq1_dose_response")

    # ── RQ1 Regression ──
    print("\n--- RQ1 Logistic Regression ---")
    X_cols = ["activities_in_first_7d", "avg_vocab_richness", "days_to_first_activity", "n_logins"]
    reg = df[X_cols + ["dropout_label", "course_name"]].dropna(subset=X_cols)
    X = pd.concat([reg[X_cols], pd.get_dummies(reg["course_name"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    m = sm.Logit(reg["dropout_label"], X).fit(disp=0)
    r1 = pd.DataFrame({"OR": np.exp(m.params), "CI_low": np.exp(m.conf_int()[0]),
                         "CI_high": np.exp(m.conf_int()[1]), "p_value": m.pvalues})
    for p in X_cols:
        r = r1.loc[p]; sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {p:30s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f} {sig}")
    save_csv(r1.loc[X_cols], "rq1_logistic_regression")

    # ── RQ2: Comments ──
    print("\n--- RQ2: Facilitator Comments ---")
    ct2 = pd.crosstab(writers["received_comment"], writers["dropout_label"])
    ct2.index = ["No comments", "Received comments"]; ct2.columns = ["Completer", "Dropout"]
    chi2_2, p2, _, _ = chi2_contingency(ct2)
    v2 = ct2.values.flatten() + 0.5; OR2 = (v2[0] * v2[3]) / (v2[1] * v2[2])
    print(f"  Chi2={chi2_2:.1f}, OR={OR2:.2f}")
    save_csv(ct2, "rq2_comment_chi2")

    X2_cols = ["received_comment", "total_activities_submitted", "n_logins"]
    reg2 = writers[X2_cols + ["dropout_label", "course_name"]].dropna(subset=X2_cols)
    X2 = pd.concat([reg2[X2_cols], pd.get_dummies(reg2["course_name"], drop_first=True, dtype=float)], axis=1)
    X2 = sm.add_constant(X2)
    m2 = sm.Logit(reg2["dropout_label"], X2).fit(disp=0)
    r2 = pd.DataFrame({"OR": np.exp(m2.params), "CI_low": np.exp(m2.conf_int()[0]),
                         "CI_high": np.exp(m2.conf_int()[1]), "p_value": m2.pvalues})
    for p in X2_cols:
        r = r2.loc[p]; sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {p:35s}: OR={r['OR']:.3f} p={r['p_value']:.4f} {sig}")
    save_csv(r2.loc[X2_cols], "rq2_model1")

    # ── RQ3: Forum ──
    print("\n--- RQ3: Forum Participation ---")
    ct3 = pd.crosstab(df["is_poster"], df["dropout_label"])
    ct3.index = ["Non-poster", "Poster"]; ct3.columns = ["Completer", "Dropout"]
    chi2_3, p3, _, _ = chi2_contingency(ct3)
    poster_comp = ct3.loc["Poster", "Completer"] / ct3.loc["Poster"].sum() * 100
    print(f"  Poster completion: {poster_comp:.1f}%, Chi2={chi2_3:.1f}")
    save_csv(ct3, "rq3_forum_chi2")

    X3_cols = ["total_discussion_replies", "total_activities_submitted", "total_comments_received", "n_logins"]
    reg3 = df[X3_cols + ["dropout_label", "course_name"]].dropna(subset=X3_cols)
    X3 = pd.concat([reg3[X3_cols], pd.get_dummies(reg3["course_name"], drop_first=True, dtype=float)], axis=1)
    X3 = sm.add_constant(X3)
    m3 = sm.Logit(reg3["dropout_label"], X3).fit(disp=0)
    r3 = pd.DataFrame({"OR": np.exp(m3.params), "CI_low": np.exp(m3.conf_int()[0]),
                         "CI_high": np.exp(m3.conf_int()[1]), "p_value": m3.pvalues})
    for p in X3_cols:
        r = r3.loc[p]; sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {p:35s}: OR={r['OR']:.3f} p={r['p_value']:.4f} {sig}")
    save_csv(r3.loc[X3_cols], "rq3_logistic_regression")

    # ── Clustering (k=5, all features, all participants) ──
    print("\n--- Clustering (k=5, 37 features, all participants) ---")
    X_df = df[numeric].copy()
    for c in X_df.columns:
        if X_df[c].isna().any():
            X_df[c] = X_df[c].fillna(X_df[c].median() if X_df[c].median() != 0 else 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Silhouette
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lab = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, lab)
        print(f"  k={k}: silhouette={sil:.4f}")

    # Fit k=5
    km5 = KMeans(n_clusters=5, n_init=30, random_state=42)
    df["cluster"] = km5.fit_predict(X_scaled)
    engage_score = df.groupby("cluster")[["total_activities_submitted", "n_page_visits", "total_discussion_replies"]].mean().mean(axis=1)
    order = engage_score.sort_values().index.tolist()
    label_map = {order[i]: name for i, name in enumerate(["Disengaged", "Minimal", "Light", "Moderate", "Deep"])}
    df["profile"] = df["cluster"].map(label_map)

    summary_cols = ["total_activities_submitted", "n_logins", "n_distinct_pages", "n_bookmarks",
                     "total_comments_received", "total_discussion_replies", "pv_pages_first_7d"]
    summary_cols = [c for c in summary_cols if c in df.columns]
    profiles = df.groupby("profile").agg(
        n=("dropout_label", "count"),
        dropout_pct=("dropout_label", lambda x: round(x.mean() * 100, 1)),
        **{f"mean_{c}": (c, "mean") for c in summary_cols},
    ).round(1).sort_values("dropout_pct", ascending=False)
    print(profiles.to_string())
    save_csv(profiles, "cluster_profiles")
    save_csv(df[["module_id", "user_id", "cohort_id", "cluster", "profile"]], "cluster_assignments")

    # PCA + dropout bar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    color_cycle = [PALETTE["grey"], PALETTE["red"], PALETTE["orange"], PALETTE["blue"], PALETTE["green"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, profile in enumerate(profiles.index):
        mask = (df["profile"] == profile).values
        ax.scatter(X_pca[mask & (df["dropout_label"]==0).values, 0], X_pca[mask & (df["dropout_label"]==0).values, 1],
                   c=color_cycle[i], alpha=0.3, s=12, label=profile)
        ax.scatter(X_pca[mask & (df["dropout_label"]==1).values, 0], X_pca[mask & (df["dropout_label"]==1).values, 1],
                   c=color_cycle[i], alpha=0.8, s=20, marker="x")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Projection (x = dropouts)"); ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(profiles.index, profiles["dropout_pct"], color=color_cycle[:len(profiles)], edgecolor="white")
    for bar, n in zip(bars, profiles["n"].astype(int)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"n={n:,}", ha="center", fontsize=9)
    ax.set_ylabel("Dropout Rate (%)"); ax.set_title("Dropout by Profile"); ax.set_ylim(0, 80)
    ax.tick_params(axis="x", rotation=15)
    save_fig(fig, "fig_cluster_profiles")

    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, profile in enumerate(profiles.index):
        mask = (df["profile"] == profile).values
        ax.scatter(X_tsne[mask & (df["dropout_label"]==0).values, 0], X_tsne[mask & (df["dropout_label"]==0).values, 1],
                   c=color_cycle[i], alpha=0.3, s=12, label=profile)
        ax.scatter(X_tsne[mask & (df["dropout_label"]==1).values, 0], X_tsne[mask & (df["dropout_label"]==1).values, 1],
                   c=color_cycle[i], alpha=0.8, s=20, marker="x")
    ax.set_title("t-SNE Projection (x = dropouts)"); ax.legend(fontsize=8)
    save_fig(fig, "fig_cluster_tsne")

    # Heatmap
    norm = df.groupby("profile")[numeric].mean()
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
    norm = norm.loc[profiles.index]
    fig, ax = plt.subplots(figsize=(14, max(6, len(numeric) * 0.3)))
    sns.heatmap(norm.T, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1, annot_kws={"size": 7})
    ax.set_title("Normalised Feature Means by Profile (k=5)")
    save_fig(fig, "fig_cluster_heatmap")

    # ── Survival ──
    print("\n--- Survival Analysis ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comparisons = [
        ("is_writer", {0: "Non-writers", 1: "Writers"}, "Writers vs Non-Writers", df),
        ("received_comment", {0: "No comments", 1: "Received comments"}, "Comment Receipt (Writers)", writers),
        ("is_poster", {0: "Non-posters", 1: "Forum posters"}, "Forum Posters vs Non-Posters", df),
    ]
    lr_results = []
    for (gcol, labels, title, data_sub), ax in zip(comparisons, axes):
        kmf = KaplanMeierFitter()
        for g, color in zip(sorted(data_sub[gcol].unique()), [PALETTE["blue"], PALETTE["red"]]):
            sub = data_sub[data_sub[gcol] == g]
            kmf.fit(sub["duration_days"], event_observed=(sub["dropout_label"] == 1), label=labels[g])
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
        ax.set_xlabel("Days"); ax.set_ylabel("Retention"); ax.set_title(title); ax.set_ylim(0, 1.05); ax.legend(loc="lower left")
        sub0, sub1 = data_sub[data_sub[gcol] == 0], data_sub[data_sub[gcol] == 1]
        lr = logrank_test(sub0["duration_days"], sub1["duration_days"],
                          event_observed_A=(sub0["dropout_label"]==1), event_observed_B=(sub1["dropout_label"]==1))
        print(f"  {title}: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")
        lr_results.append({"comparison": title, "chi2": round(lr.test_statistic, 1), "p_value": lr.p_value})
    save_fig(fig, "fig_km_all")
    save_csv(pd.DataFrame(lr_results), "survival_logrank")

    # Individual KM figures
    for (gcol, labels, title, data_sub), fname in zip(comparisons,
        ["fig_km_writer_nonwriter", "fig_km_commented_vs_not", "fig_km_forum_poster"]):
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        kmf = KaplanMeierFitter()
        for g, color in zip(sorted(data_sub[gcol].unique()), [PALETTE["blue"], PALETTE["red"]]):
            sub = data_sub[data_sub[gcol] == g]
            kmf.fit(sub["duration_days"], event_observed=(sub["dropout_label"] == 1), label=labels[g])
            kmf.plot_survival_function(ax=ax2, ci_show=True, color=color)
        ax2.set_xlabel("Days"); ax2.set_ylabel("Retention"); ax2.set_title(title); ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower left")
        save_fig(fig2, fname)

    # ── GEE ──
    print("\n--- GEE Robustness ---")
    GEE_FEATS = ["total_activities_submitted", "avg_vocab_richness", "avg_sentiment",
                  "avg_future_orientation", "activity_type_entropy", "word_count_trend",
                  "total_comments_received", "total_discussion_replies",
                  "activities_in_first_7d", "n_distinct_pages"]
    GEE_FEATS = [f for f in GEE_FEATS if f in writers.columns]
    gee_df = writers[["module_id", "dropout_label"] + GEE_FEATS].dropna()
    scaler_gee = StandardScaler()
    X_gee = pd.DataFrame(scaler_gee.fit_transform(gee_df[GEE_FEATS]), columns=GEE_FEATS, index=gee_df.index)
    X_gee = sm.add_constant(X_gee)
    gee = sm.GEE(gee_df["dropout_label"], X_gee, groups=gee_df["module_id"],
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    gee_result = gee.fit()
    gee_summary = pd.DataFrame({
        "OR": np.exp(gee_result.params), "CI_low": np.exp(gee_result.conf_int().iloc[:, 0]),
        "CI_high": np.exp(gee_result.conf_int().iloc[:, 1]), "p_value": gee_result.pvalues,
    }).drop("const", errors="ignore")
    _, gee_padj, _, _ = multipletests(gee_summary["p_value"], method="fdr_bh")
    gee_summary["p_adj"] = gee_padj
    gee_summary["sig_fdr"] = gee_summary["p_adj"] < 0.05
    for f in GEE_FEATS:
        if f in gee_summary.index:
            r = gee_summary.loc[f]
            sig = " *" if r["sig_fdr"] else ""
            print(f"  {f:35s}: OR={r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] p={r['p_value']:.4f}{sig}")
    print(f"  Significant after BH-FDR: {gee_summary['sig_fdr'].sum()}/{len(gee_summary)}")
    save_csv(gee_summary, "gee_results")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  FULL PIPELINE REBUILD (with reclassification)")
    print("=" * 70)

    users, starters, activities, fc_comments, discussions, page_visits = step1_load_and_reclassify()
    features, groups = step2_build_features(starters, activities, fc_comments, discussions, page_visits)
    step3_analyses(features, groups)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"  Features: {FEAT_DIR / 'user_level_features_v2.csv'}")
    print(f"  Tables:   {TABLE_DIR}")
    print(f"  Figures:  {FIG_DIR}")
    print("=" * 70)
