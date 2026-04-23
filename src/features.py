"""
Build analytical feature tables from raw CSVs.

Reads the 5 CSVs produced by dataset.py and outputs 3 analytical tables:

  activity_level_features.csv  - one row per activity with NLP features
  comment_pairs.csv            - activity text paired with facilitator response
  user_level_features.csv      - one row per (module, cohort, user) with 37 analytical
                                 features (+ duration_days for survival) across 8 dimensions

Usage:
  python src/features.py
  python src/features.py --csv-dir data/csv --out-dir output/features
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_features import NLPFeatureExtractor, TOPIC_KEYS
from src.utils import compute_dropout_label, assign_discussion_cohorts, parse_mixed_datetime

warnings.filterwarnings("ignore", category=FutureWarning)

OBS_KEYS = ["module_id", "user_id", "cohort_id"]


# =====================================================================
# Data loading
# =====================================================================

def load_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw CSVs into DataFrames."""
    users = pd.read_csv(csv_dir / "users.csv")
    activities = pd.read_csv(csv_dir / "activities.csv")
    fc_comments = pd.read_csv(csv_dir / "facilitator_comments.csv")
    discussions = pd.read_csv(csv_dir / "discussions.csv")
    page_visits = pd.read_csv(csv_dir / "page_visits.csv")

    users["started"] = parse_mixed_datetime(users["started"])
    users["finished"] = parse_mixed_datetime(users["finished"])
    activities["recorded"] = parse_mixed_datetime(activities["recorded"])
    fc_comments["recorded"] = parse_mixed_datetime(fc_comments["recorded"])
    discussions["recorded"] = parse_mixed_datetime(discussions["recorded"])
    page_visits["latest"] = parse_mixed_datetime(page_visits["latest"])

    # Deduplicate
    activities = activities.drop_duplicates(subset=["activity_id"], keep="last")
    fc_comments = fc_comments.drop_duplicates(subset=["comment_id"], keep="last")
    users = (
        users.sort_values("started", na_position="first")
        .drop_duplicates(subset=OBS_KEYS, keep="last")
    )

    users["dropout_label"] = compute_dropout_label(users)

    return {
        "users": users,
        "activities": activities,
        "fc_comments": fc_comments,
        "discussions": discussions,
        "page_visits": page_visits,
    }


# =====================================================================
# Platform engagement features
# =====================================================================

_PAGE_KEYWORDS = {
    "dashboard": ["dashboard"],
    "activity":  ["/activities", "/newsfeed", "/gratitude", "/goals"],
    "content":   ["/session", "/journey", "/learning", "/welcome", "/further-resources"],
    "forum":     ["/discussion", "/forum", "/topic"],
    "profile":   ["/profile", "/personal", "/notifications"],
}


def _categorise_page(url: str) -> str:
    """Classify a page URL into a coarse content category."""
    if pd.isna(url):
        return "other"
    u = str(url).lower()
    for category, keys in _PAGE_KEYWORDS.items():
        if any(k in u for k in keys):
            return category
    return "other"


def build_platform_features(
    starters: pd.DataFrame, page_visits: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate platform engagement features (logins, bookmarks, page visits).

    Includes five originals from users.csv (n_logins, login_span_days,
    n_bookmarks, n_page_visits, n_distinct_pages) plus engineered
    depth, category-proportion, and first-week breadth features from
    page_visits.csv.
    """
    platform = starters[OBS_KEYS + [
        "n_logins", "login_span_days", "n_bookmarks",
        "n_page_visits", "n_distinct_pages",
    ]].copy()

    pv = page_visits.copy()
    pv["page_category"] = pv["url"].apply(_categorise_page)
    pv = pv.merge(
        starters[OBS_KEYS + ["started"]], on=OBS_KEYS, how="inner"
    )
    pv["started"] = parse_mixed_datetime(pv["started"])
    pv["days_since_start"] = (
        (pv["latest"] - pv["started"]).dt.total_seconds() / 86400
    )

    # Depth: mean duration per page and mean hits per page
    pv_depth = (
        pv.groupby(OBS_KEYS)
        .agg(
            pv_mean_duration=("avg_duration", "mean"),
            pv_mean_hits_per_page=("hits", "mean"),
        )
        .reset_index()
    )

    # Category proportions (share of total hits in each category)
    cat_hits = (
        pv.groupby(OBS_KEYS + ["page_category"])["hits"]
        .sum()
        .unstack(fill_value=0)
    )
    cat_pct = cat_hits.div(cat_hits.sum(axis=1).replace(0, np.nan), axis=0)
    cat_pct.columns = [f"pv_pct_{c}" for c in cat_pct.columns]
    cat_pct = cat_pct.reset_index()
    # pv_pct_other is redundant (1 minus the sum of the five reported categories)
    cat_pct = cat_pct.drop(columns=["pv_pct_other"], errors="ignore")

    # First-week breadth: distinct pages visited in days 0-6
    early = pv[pv["days_since_start"].between(0, 6)]
    pv_early = (
        early.groupby(OBS_KEYS)["url"]
        .nunique()
        .rename("pv_pages_first_7d")
        .reset_index()
    )

    platform = (
        platform.merge(pv_depth, on=OBS_KEYS, how="left")
        .merge(cat_pct, on=OBS_KEYS, how="left")
        .merge(pv_early, on=OBS_KEYS, how="left")
    )

    for c in platform.columns:
        if c in OBS_KEYS:
            continue
        platform[c] = pd.to_numeric(platform[c], errors="coerce").fillna(0)

    return platform


# =====================================================================
# Table A: activity-level features
# =====================================================================

def build_activity_level(
    tables: dict[str, pd.DataFrame], nlp: NLPFeatureExtractor
) -> pd.DataFrame:
    """Per-activity features: text metrics + BERT sentiment + zero-shot topics."""
    acts = tables["activities"].copy()
    acts = acts[acts["description"].notna() & (acts["description"].str.strip() != "")]
    acts = acts.reset_index(drop=True)

    print(f"  Computing text metrics for {len(acts):,} activities ...")

    # --- Regex-based features (fast) ---
    texts = acts["description"].fillna("").tolist()
    acts["word_count"] = [nlp.word_count(t) for t in texts]
    acts["unique_words"] = [nlp.unique_words(t) for t in texts]
    acts["sentence_count"] = [nlp.sentence_count(t) for t in texts]
    acts["avg_sentence_length"] = np.where(
        acts["sentence_count"] > 0,
        acts["word_count"] / acts["sentence_count"],
        0.0,
    )
    acts["vocab_richness"] = np.where(
        acts["word_count"] > 0,
        acts["unique_words"] / acts["word_count"],
        0.0,
    )
    acts["self_reference_ratio"] = [nlp.self_reference_ratio(t) for t in texts]
    acts["future_orientation"] = [nlp.future_orientation(t) for t in texts]

    # --- BERT sentiment (batched) ---
    print("  Running BERT sentiment analysis ...")
    acts["compound_score"] = nlp.batch_sentiment(texts, batch_size=32)

    # --- BERT zero-shot topics (batched) ---
    print("  Running zero-shot topic classification ...")
    topic_results = nlp.batch_zero_shot(texts, batch_size=8)
    for key in TOPIC_KEYS:
        acts[key] = [r.get(key, 0.0) for r in topic_results]

    # --- Comment info ---
    acts["num_comments"] = acts["num_comments"].fillna(0).astype(int)
    acts["has_comment"] = (acts["num_comments"] > 0).astype(int)

    # --- Join dropout label ---
    user_labels = (
        tables["users"][OBS_KEYS + ["dropout_label"]]
        .drop_duplicates()
    )
    acts = acts.merge(user_labels, on=OBS_KEYS, how="left")

    return acts


# =====================================================================
# Table B: comment pairs
# =====================================================================

def build_comment_pairs(
    tables: dict[str, pd.DataFrame],
    act_df: pd.DataFrame,
    nlp: NLPFeatureExtractor,
) -> pd.DataFrame:
    """Join each activity to its facilitator comment(s)."""
    act_sub = act_df[
        [
            "module_id", "cohort_id", "activity_id", "user_id", "type_name",
            "description", "recorded", "word_count", "compound_score",
            "dropout_label",
        ]
    ].copy()
    act_sub = act_sub.rename(columns={
        "recorded": "activity_recorded",
        "word_count": "activity_word_count",
        "compound_score": "activity_sentiment",
    })

    fc = tables["fc_comments"].copy()
    fc = fc.rename(columns={"recorded": "comment_recorded"})

    pairs = act_sub.merge(
        fc[["activity_id", "comment_id", "facilitator_user_id",
            "comment_text", "comment_recorded", "word_count"]],
        on="activity_id",
        how="inner",
    )
    pairs = pairs.rename(columns={"word_count": "comment_word_count"})

    # Timestamps
    pairs["activity_recorded"] = parse_mixed_datetime(pairs["activity_recorded"])
    pairs["comment_recorded"] = parse_mixed_datetime(pairs["comment_recorded"])

    # Response latency
    pairs["response_hours"] = (
        (pairs["comment_recorded"] - pairs["activity_recorded"])
        .dt.total_seconds()
        / 3600
    ).clip(lower=0)

    # BERT sentiment on facilitator comments
    print("  Running BERT sentiment on facilitator comments ...")
    comment_texts = pairs["comment_text"].fillna("").tolist()
    pairs["comment_sentiment"] = nlp.batch_sentiment(comment_texts, batch_size=32)

    return pairs


# =====================================================================
# Table C: user-level features (~55 features, 8 dimensions)
# =====================================================================

def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector (base-e)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _linear_slope(values: np.ndarray) -> float:
    """Slope of OLS fit of values against their index."""
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    slope, _, _, _, _ = sp_stats.linregress(x, values)
    return float(slope)


def build_user_level(
    tables: dict[str, pd.DataFrame],
    act_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    nlp: NLPFeatureExtractor,
) -> pd.DataFrame:
    """Aggregate per (module_id, cohort_id, user_id) across 8 dimensions."""
    users = tables["users"].copy()
    starters = users[users["started"].notna()].copy()
    starters["dropout_label"] = starters["dropout_label"].astype(float)

    base = starters[
        [
            "module_id", "module_name", "course_id", "course_name",
            "cohort_id", "cohort_name",
            "user_id", "started", "finished", "dropout_label",
        ]
    ].drop_duplicates(subset=OBS_KEYS, keep="last")

    # ── Dimension 0: Platform Engagement ─────────────────────────────
    platform_feats = build_platform_features(starters, tables["page_visits"])

    acts = act_df.copy()
    acts["recorded"] = parse_mixed_datetime(acts["recorded"])

    # ── Dimension 1: Writing Volume & Frequency ─────────────────────
    g = acts.groupby(OBS_KEYS, dropna=False)

    volume = g.agg(
        total_activities_submitted=("activity_id", "count"),
        first_activity=("recorded", "min"),
        last_activity=("recorded", "max"),
    ).reset_index()

    volume["writing_span_days"] = (
        (volume["last_activity"] - volume["first_activity"]).dt.total_seconds()
        / 86400
    ).fillna(0)

    # days_to_first_activity (needs started from base)
    vol_with_start = volume.merge(
        base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left"
    )
    vol_with_start["started"] = parse_mixed_datetime(vol_with_start["started"])
    volume["days_to_first_activity"] = (
        (vol_with_start["first_activity"] - vol_with_start["started"]).dt.total_seconds()
        / 86400
    )

    # ── Dimension 2: Writing Quality & Depth ────────────────────────
    quality = g.agg(
        avg_vocab_richness=("vocab_richness", "mean"),
    ).reset_index()

    # vocab_evolution: TTR in second half minus first half
    def _vocab_evolution(group: pd.DataFrame) -> float:
        group = group.sort_values("recorded")
        n = len(group)
        if n < 4:
            return np.nan
        mid = n // 2
        first_half = group.iloc[:mid]["vocab_richness"].mean()
        second_half = group.iloc[mid:]["vocab_richness"].mean()
        return second_half - first_half

    vocab_evo = (
        acts.groupby(OBS_KEYS, dropna=False)
        .apply(_vocab_evolution)
        .rename("vocab_evolution")
        .reset_index()
    )

    # ── Dimension 3: Linguistic Content (NLP) ───────────────────────
    linguistic = g.agg(
        avg_self_reference=("self_reference_ratio", "mean"),
        avg_future_orientation=("future_orientation", "mean"),
        avg_sentiment=("compound_score", "mean"),
    ).reset_index()

    # ── Dimension 4: Content Diversity ──────────────────────────────
    CANONICAL_TYPES = ["Gratitude", "GoalSetting", "Emotions"]
    # pct_goalsetting is not reported separately (r = 0.93 with avg_future_orientation);
    # GoalSetting counts still feed the activity_type_entropy calculation below.
    REPORTED_TYPES = ["Gratitude", "Emotions"]

    def _type_features(group: pd.DataFrame) -> pd.Series:
        total = len(group)
        counts = group["type_name"].value_counts()
        pcts = {
            f"pct_{t.lower()}": counts.get(t, 0) / total if total > 0 else 0.0
            for t in REPORTED_TYPES
        }
        type_counts = np.array([counts.get(t, 0) for t in CANONICAL_TYPES])
        pcts["activity_type_entropy"] = _shannon_entropy(type_counts)
        return pd.Series(pcts)

    type_div = (
        acts.groupby(OBS_KEYS, dropna=False)
        .apply(_type_features)
        .reset_index()
    )

    # ── Dimension 5: Engagement Trajectories ────────────────────────
    def _trajectory_features(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("recorded")
        n = len(group)

        wc_trend = _linear_slope(group["word_count"].values) if n >= 2 else np.nan
        sent_trend = (
            _linear_slope(group["compound_score"].values) if n >= 2 else np.nan
        )

        # Activity regularity (inverse of gap std dev)
        if n >= 2:
            gaps = group["recorded"].diff().dt.total_seconds().dropna() / 86400
            gap_std = gaps.std()
            regularity = 1.0 / (1.0 + gap_std) if not np.isnan(gap_std) else np.nan
        else:
            regularity = np.nan

        return pd.Series({
            "word_count_trend": wc_trend,
            "sentiment_trend": sent_trend,
            "activity_regularity": regularity,
        })

    print("  Computing engagement trajectories ...")
    trajectories = (
        acts.groupby(OBS_KEYS, dropna=False)
        .apply(_trajectory_features)
        .reset_index()
    )

    # ── Dimension 6: Facilitator Interaction ────────────────────────
    fac_user = g.agg(
        total_comments_received=("num_comments", "sum"),
    ).reset_index()

    # continued_after_comment
    def _continued_after_comment(group: pd.DataFrame) -> float:
        commented = group[group["has_comment"] == 1]
        if commented.empty:
            return np.nan
        first_comment_time = commented["recorded"].min()
        later = group[group["recorded"] > first_comment_time]
        return 1.0 if len(later) > 0 else 0.0

    cont = (
        acts.groupby(OBS_KEYS, dropna=False)
        .apply(_continued_after_comment)
        .rename("continued_after_comment")
        .reset_index()
    )
    fac_user = fac_user.merge(cont, on=OBS_KEYS, how="left")

    # Comment-level aggregates from pairs
    if not pairs_df.empty:
        cp = (
            pairs_df.groupby(OBS_KEYS, dropna=False)
            .agg(
                avg_response_hours=("response_hours", "mean"),
                avg_comment_word_count=("comment_word_count", "mean"),
            )
            .reset_index()
        )
        fac_user = fac_user.merge(cp, on=OBS_KEYS, how="left")

    # ── Dimension 7: Social/Peer Interaction (Forum) ────────────────
    dt = assign_discussion_cohorts(
        tables["discussions"],
        base[OBS_KEYS + ["cohort_name", "started"]],
    )
    dt["recorded"] = parse_mixed_datetime(dt["recorded"])
    dt = dt[dt["cohort_id"].notna()].copy()

    # Forum sentiment (batched)
    dt_texts = dt["comment"].fillna("").tolist()
    if dt_texts and nlp is not None:
        print("  Running BERT sentiment on discussion posts ...")
        dt["d_sentiment"] = nlp.batch_sentiment(dt_texts, batch_size=32)
    else:
        dt["d_sentiment"] = 0.0

    dg = (
        dt.groupby(OBS_KEYS, dropna=False)
        .agg(
            total_discussion_replies=("reply_id", "count"),
            first_post=("recorded", "min"),
            last_post=("recorded", "max"),
            forum_sentiment_mean=("d_sentiment", "mean"),
        )
        .reset_index()
    )
    dg["forum_span_days"] = (
        (dg["last_post"] - dg["first_post"]).dt.total_seconds() / 86400
    ).fillna(0)

    # days_to_first_post (needs started)
    dg_with_start = dg.merge(
        base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left"
    )
    dg_with_start["started"] = parse_mixed_datetime(dg_with_start["started"])
    dg["days_to_first_post"] = (
        (dg_with_start["first_post"] - dg_with_start["started"]).dt.total_seconds()
        / 86400
    )
    dg = dg.drop(columns=["first_post", "last_post"])

    # ── Dimension 8: Early Warning Signals ──────────────────────────
    acts_with_start = acts.merge(
        base[OBS_KEYS + ["started"]], on=OBS_KEYS, how="left"
    )
    acts_with_start["started"] = parse_mixed_datetime(acts_with_start["started"])
    acts_with_start["days_since_start"] = (
        (acts_with_start["recorded"] - acts_with_start["started"]).dt.total_seconds()
        / 86400
    )

    def _early_warning(window_days: int, suffix: str) -> pd.DataFrame:
        early = acts_with_start[
            acts_with_start["days_since_start"].between(0, window_days - 1)
        ]
        agg = (
            early.groupby(OBS_KEYS, dropna=False)
            .agg(
                **{f"activities_in_first_{suffix}": ("activity_id", "count")},
            )
            .reset_index()
        )
        return agg

    ew_7 = _early_warning(7, "7d")

    # ── Merge all dimensions onto base ──────────────────────────────
    result = base.copy()
    for df in [
        platform_feats, volume, quality, vocab_evo, linguistic, type_div,
        trajectories, fac_user, dg, ew_7,
    ]:
        result = result.merge(df, on=OBS_KEYS, how="left")

    # ── Temporal / survival features ────────────────────────────────
    result["started"] = parse_mixed_datetime(result["started"])
    result["finished"] = parse_mixed_datetime(result["finished"])
    result["first_activity"] = parse_mixed_datetime(result["first_activity"])
    result["last_activity"] = parse_mixed_datetime(result["last_activity"])

    result["duration_days"] = np.where(
        result["finished"].notna(),
        (result["finished"] - result["started"]).dt.total_seconds() / 86400,
        (result["last_activity"] - result["started"]).dt.total_seconds() / 86400,
    )
    result["duration_days"] = result["duration_days"].fillna(1).clip(lower=1)

    # (Binary flags wrote_anything, received_comment, posted_in_forum,
    #  wrote_in_first_week, wrote_in_first_two_weeks removed —
    #  redundant with their count counterparts > 0)

    # ── Drop internal-only columns ──────────────────────────────────
    result = result.drop(
        columns=["first_activity", "last_activity"], errors="ignore"
    )

    # ── Fill NaN for count/sum features ─────────────────────────────
    fill_zero_cols = [
        "total_activities_submitted",
        "total_comments_received", "total_discussion_replies",
        "activities_in_first_7d",
        "n_logins", "login_span_days",
        "n_bookmarks", "n_page_visits", "n_distinct_pages",
        "pv_pages_first_7d",
    ]
    for c in fill_zero_cols:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0)

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build analytical feature tables from raw CSVs."
    )
    parser.add_argument(
        "--csv-dir",
        default=str(PROJECT_ROOT / "data" / "csv"),
        help="Directory containing raw CSVs from dataset.py",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "output" / "features"),
        help="Directory for analytical table output",
    )
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip BERT models; reuse existing activity_level_features.csv "
             "and comment_pairs.csv, only rebuild user_level_features.csv",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSVs ...")
    tables = load_tables(csv_dir)

    print(f"  users:       {len(tables['users']):,}")
    print(f"  activities:  {len(tables['activities']):,}")
    print(f"  comments:    {len(tables['fc_comments']):,}")
    print(f"  discussions: {len(tables['discussions']):,}")
    print(f"  page_visits: {len(tables['page_visits']):,}")
    print()

    if args.skip_nlp:
        act_path = out_dir / "activity_level_features.csv"
        pairs_path = out_dir / "comment_pairs.csv"
        if not act_path.exists() or not pairs_path.exists():
            print("ERROR: --skip-nlp requires existing activity_level_features.csv "
                  "and comment_pairs.csv in out-dir.")
            return
        print(f"--skip-nlp: reusing {act_path.name} and {pairs_path.name}")
        act_df = pd.read_csv(act_path)
        act_df["recorded"] = parse_mixed_datetime(act_df["recorded"])
        pairs_df = pd.read_csv(pairs_path)
        # Still need NLP for forum-post sentiment (fast: ~2 min)
        print("  Loading NLP models (for forum sentiment only) ...")
        nlp = NLPFeatureExtractor()
        print()
    else:
        print("Initialising NLP models ...")
        nlp = NLPFeatureExtractor()
        print()

        # --- Table A: activity-level features ---
        print("Building activity-level features ...")
        act_df = build_activity_level(tables, nlp)
        act_path = out_dir / "activity_level_features.csv"
        act_df.to_csv(act_path, index=False)
        print(f"  -> {act_path.name}: {len(act_df):,} rows\n")

        # --- Table B: comment pairs ---
        print("Building comment pairs ...")
        pairs_df = build_comment_pairs(tables, act_df, nlp)
        pairs_path = out_dir / "comment_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False)
        print(f"  -> {pairs_path.name}: {len(pairs_df):,} rows\n")

    # --- Table C: user-level features ---
    print("Building user-level features ...")
    user_df = build_user_level(tables, act_df, pairs_df, nlp)
    user_path = out_dir / "user_level_features.csv"
    user_df.to_csv(user_path, index=False)
    print(f"  -> {user_path.name}: {len(user_df):,} rows\n")

    # --- Feature groups JSON (used by analysis/config.py) ---
    meta_cols = ["module_id", "module_name", "course_id", "course_name",
                 "cohort_id", "cohort_name", "user_id", "started", "finished",
                 "dropout_label"]
    feat_cols = [c for c in user_df.columns if c not in meta_cols]
    platform_cols = [c for c in feat_cols if c in {
        "n_logins", "login_span_days", "n_bookmarks",
        "n_page_visits", "n_distinct_pages",
        "pv_mean_duration", "pv_mean_hits_per_page",
        "pv_pages_first_7d",
    } or c.startswith("pv_pct_")]

    originals = {
        "n_logins", "login_span_days", "n_bookmarks",
        "n_page_visits", "n_distinct_pages",
        "total_activities_submitted",
        "total_comments_received",
        "total_discussion_replies",
    }

    groups = {
        "platform": platform_cols,
        "writing_volume": ["total_activities_submitted", "writing_span_days",
                           "days_to_first_activity"],
        "writing_quality": ["avg_vocab_richness", "vocab_evolution"],
        "linguistic": ["avg_self_reference", "avg_future_orientation", "avg_sentiment"],
        "content_diversity": ["pct_gratitude", "pct_emotions",
                              "activity_type_entropy"],
        "trajectories": ["word_count_trend", "sentiment_trend", "activity_regularity"],
        "facilitator": ["total_comments_received",
                        "continued_after_comment", "avg_response_hours",
                        "avg_comment_word_count"],
        "forum": ["total_discussion_replies", "forum_sentiment_mean",
                  "forum_span_days", "days_to_first_post"],
        "early_warning": ["activities_in_first_7d"],
        "survival": ["duration_days"],
        "originals": sorted(originals),
        # duration_days is used only in the Kaplan-Meier survival analysis; it
        # is excluded from all_features so it does not enter clustering or the
        # univariate Mann-Whitney tests that consume groups["all_features"].
        "all_features": [c for c in feat_cols if c != "duration_days"],
        "meta": meta_cols,
    }
    groups_path = out_dir / "feature_groups.json"
    with open(groups_path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"  -> {groups_path.name}: {len(groups)} groups\n")

    # --- Summary ---
    starters = user_df[user_df["dropout_label"].notna()]
    completers = starters[starters["dropout_label"] == 0]
    dropouts = starters[starters["dropout_label"] == 1]
    writers = starters[starters["total_activities_submitted"] > 0]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Starters:    {len(starters):,}")
    print(f"  Completers:  {len(completers):,} ({len(completers)/len(starters)*100:.1f}%)")
    print(f"  Dropouts:    {len(dropouts):,} ({len(dropouts)/len(starters)*100:.1f}%)")
    print(f"  Writers:     {len(writers):,} ({len(writers)/len(starters)*100:.1f}%)")
    print()
    print(f"  Courses: {starters['course_name'].nunique()}")
    print(f"  Modules: {starters['module_id'].nunique()}")
    print()
    print("  Feature columns:", user_df.shape[1])
    print()

    for label, name in [(0, "Completers"), (1, "Dropouts")]:
        sub = starters[starters["dropout_label"] == label]
        w = sub[sub["total_activities_submitted"] > 0]
        print(f"  {name}:")
        print(f"    {len(w)}/{len(sub)} ({len(w)/len(sub)*100:.0f}%) wrote at least once")
        print(f"    Mean activities: {sub['total_activities_submitted'].mean():.1f}")

    print()
    print(f"Saved: {act_path}")
    print(f"Saved: {pairs_path}")
    print(f"Saved: {user_path}")


if __name__ == "__main__":
    main()
