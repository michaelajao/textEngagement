"""
Build analytical feature tables from raw CSVs.

Reads the 5 CSVs produced by dataset.py and outputs 3 analytical tables:

  activity_level_features.csv  - one row per activity with NLP features
  comment_pairs.csv            - activity text paired with facilitator response
  user_level_features.csv      - one row per (module, user) with ~55 features
                                 across 8 engagement dimensions

Usage:
  python src/features.py
  python src/features.py --csv-dir data/csv --out-dir output/features
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_features import NLPFeatureExtractor, TOPIC_KEYS
from src.utils import compute_dropout_label

warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# Data loading
# =====================================================================

def load_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw CSVs into DataFrames."""
    users = pd.read_csv(csv_dir / "users.csv", parse_dates=["started", "finished"])
    activities = pd.read_csv(csv_dir / "activities.csv", parse_dates=["recorded"])
    fc_comments = pd.read_csv(
        csv_dir / "facilitator_comments.csv", parse_dates=["recorded"]
    )
    discussions = pd.read_csv(csv_dir / "discussions.csv", parse_dates=["recorded"])

    # Deduplicate
    activities = activities.drop_duplicates(subset=["activity_id"], keep="last")
    fc_comments = fc_comments.drop_duplicates(subset=["comment_id"], keep="last")
    users = (
        users.sort_values("started", na_position="first")
        .drop_duplicates(subset=["module_id", "user_id"], keep="last")
    )

    users["dropout_label"] = compute_dropout_label(users)

    return {
        "users": users,
        "activities": activities,
        "fc_comments": fc_comments,
        "discussions": discussions,
    }


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
        tables["users"][["module_id", "user_id", "dropout_label"]]
        .drop_duplicates()
    )
    acts = acts.merge(user_labels, on=["module_id", "user_id"], how="left")

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
            "module_id", "activity_id", "user_id", "type_name",
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
    pairs["activity_recorded"] = pd.to_datetime(
        pairs["activity_recorded"], errors="coerce"
    )
    pairs["comment_recorded"] = pd.to_datetime(
        pairs["comment_recorded"], errors="coerce"
    )

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
    """Aggregate per (module_id, user_id) across 8 engagement dimensions."""
    users = tables["users"].copy()
    starters = users[users["started"].notna()].copy()
    starters["dropout_label"] = starters["dropout_label"].astype(float)

    base = starters[
        [
            "module_id", "module_name", "course_id", "course_name",
            "cohort_id", "cohort_name",
            "user_id", "started", "finished", "dropout_label",
        ]
    ].drop_duplicates(subset=["module_id", "user_id"], keep="last")

    acts = act_df.copy()
    acts["recorded"] = pd.to_datetime(acts["recorded"], errors="coerce")

    # ── Dimension 1: Writing Volume & Frequency ─────────────────────
    g = acts.groupby(["module_id", "user_id"], dropna=False)

    volume = g.agg(
        total_activities_submitted=("activity_id", "count"),
        total_words_written=("word_count", "sum"),
        avg_description_length=("word_count", "mean"),
        max_description_length=("word_count", "max"),
        first_activity=("recorded", "min"),
        last_activity=("recorded", "max"),
    ).reset_index()

    volume["writing_span_days"] = (
        (volume["last_activity"] - volume["first_activity"]).dt.total_seconds()
        / 86400
    ).fillna(0)
    volume["writing_frequency"] = np.where(
        volume["writing_span_days"] > 0,
        volume["total_activities_submitted"] / volume["writing_span_days"],
        volume["total_activities_submitted"].astype(float),
    )

    # days_to_first_activity (needs started from base)
    vol_with_start = volume.merge(
        base[["module_id", "user_id", "started"]], on=["module_id", "user_id"], how="left"
    )
    vol_with_start["started"] = pd.to_datetime(vol_with_start["started"], errors="coerce")
    volume["days_to_first_activity"] = (
        (vol_with_start["first_activity"] - vol_with_start["started"]).dt.total_seconds()
        / 86400
    )

    # ── Dimension 2: Writing Quality & Depth ────────────────────────
    quality = g.agg(
        avg_vocab_richness=("vocab_richness", "mean"),
        avg_sentence_length=("avg_sentence_length", "mean"),
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
        acts.groupby(["module_id", "user_id"], dropna=False)
        .apply(_vocab_evolution, include_groups=False)
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

    def _type_features(group: pd.DataFrame) -> pd.Series:
        total = len(group)
        counts = group["type_name"].value_counts()
        pcts = {
            f"pct_{t.lower()}": counts.get(t, 0) / total if total > 0 else 0.0
            for t in CANONICAL_TYPES
        }
        type_counts = np.array([counts.get(t, 0) for t in CANONICAL_TYPES])
        pcts["activity_type_entropy"] = _shannon_entropy(type_counts)
        return pd.Series(pcts)

    type_div = (
        acts.groupby(["module_id", "user_id"], dropna=False)
        .apply(_type_features, include_groups=False)
        .reset_index()
    )

    # Topic entropy + sentiment variance/range
    def _diversity_features(group: pd.DataFrame) -> pd.Series:
        topic_cols_present = [c for c in TOPIC_KEYS if c in group.columns]
        if topic_cols_present:
            topic_means = group[topic_cols_present].mean()
            topic_entropy = _shannon_entropy(topic_means.values)
        else:
            topic_entropy = 0.0

        scores = group["compound_score"].dropna()
        return pd.Series({
            "topic_entropy": topic_entropy,
            "sentiment_variance": scores.var() if len(scores) > 1 else 0.0,
            "emotional_range": scores.max() - scores.min() if len(scores) > 1 else 0.0,
        })

    diversity = (
        acts.groupby(["module_id", "user_id"], dropna=False)
        .apply(_diversity_features, include_groups=False)
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
            longest_gap = gaps.max() if len(gaps) > 0 else 0.0
        else:
            regularity = np.nan
            longest_gap = np.nan

        return pd.Series({
            "word_count_trend": wc_trend,
            "sentiment_trend": sent_trend,
            "activity_regularity": regularity,
            "longest_gap_days": longest_gap,
        })

    print("  Computing engagement trajectories ...")
    trajectories = (
        acts.groupby(["module_id", "user_id"], dropna=False)
        .apply(_trajectory_features, include_groups=False)
        .reset_index()
    )

    # ── Dimension 6: Facilitator Interaction ────────────────────────
    fac_user = g.agg(
        total_comments_received=("num_comments", "sum"),
        activities_with_comment=("has_comment", "sum"),
    ).reset_index()
    fac_user["pct_activities_with_comments"] = np.where(
        fac_user["total_comments_received"] > 0,
        fac_user["activities_with_comment"]
        / (fac_user["total_comments_received"] + fac_user["activities_with_comment"]
           - fac_user["activities_with_comment"]),
        0.0,
    )
    # Correct: pct = activities_with_comment / total_activities
    total_per_user = g["activity_id"].count().rename("_total_acts").reset_index()
    fac_user = fac_user.merge(total_per_user, on=["module_id", "user_id"], how="left")
    fac_user["pct_activities_with_comments"] = np.where(
        fac_user["_total_acts"] > 0,
        fac_user["activities_with_comment"] / fac_user["_total_acts"] * 100,
        0.0,
    )
    fac_user = fac_user.drop(columns=["activities_with_comment", "_total_acts"])

    # continued_after_comment
    def _continued_after_comment(group: pd.DataFrame) -> float:
        commented = group[group["has_comment"] == 1]
        if commented.empty:
            return np.nan
        first_comment_time = commented["recorded"].min()
        later = group[group["recorded"] > first_comment_time]
        return 1.0 if len(later) > 0 else 0.0

    cont = (
        acts.groupby(["module_id", "user_id"], dropna=False)
        .apply(_continued_after_comment, include_groups=False)
        .rename("continued_after_comment")
        .reset_index()
    )
    fac_user = fac_user.merge(cont, on=["module_id", "user_id"], how="left")

    # Comment-level aggregates from pairs
    if not pairs_df.empty:
        cp = (
            pairs_df.groupby(["module_id", "user_id"], dropna=False)
            .agg(
                avg_response_hours=("response_hours", "mean"),
                avg_comment_word_count=("comment_word_count", "mean"),
            )
            .reset_index()
        )
        fac_user = fac_user.merge(cp, on=["module_id", "user_id"], how="left")

    # ── Dimension 7: Social/Peer Interaction (Forum) ────────────────
    dt = tables["discussions"].copy()
    dt["recorded"] = pd.to_datetime(dt["recorded"], errors="coerce")

    # Word count for discussion posts
    dt["d_word_count"] = dt["comment"].apply(
        lambda t: nlp.word_count(str(t)) if pd.notna(t) else 0
    )

    # Forum sentiment (batched)
    dt_texts = dt["comment"].fillna("").tolist()
    if dt_texts:
        print("  Running BERT sentiment on discussion posts ...")
        dt["d_sentiment"] = nlp.batch_sentiment(dt_texts, batch_size=32)
    else:
        dt["d_sentiment"] = 0.0

    dg = (
        dt.groupby(["module_id", "user_id"], dropna=False)
        .agg(
            total_discussion_replies=("reply_id", "count"),
            discussion_words_written=("d_word_count", "sum"),
            n_topics_participated=("topic_id", "nunique"),
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
        base[["module_id", "user_id", "started"]], on=["module_id", "user_id"], how="left"
    )
    dg_with_start["started"] = pd.to_datetime(dg_with_start["started"], errors="coerce")
    dg["days_to_first_post"] = (
        (dg_with_start["first_post"] - dg_with_start["started"]).dt.total_seconds()
        / 86400
    )
    dg = dg.drop(columns=["first_post", "last_post"])

    # ── Dimension 8: Early Warning Signals ──────────────────────────
    acts_with_start = acts.merge(
        base[["module_id", "user_id", "started"]], on=["module_id", "user_id"], how="left"
    )
    acts_with_start["started"] = pd.to_datetime(acts_with_start["started"], errors="coerce")
    acts_with_start["days_since_start"] = (
        (acts_with_start["recorded"] - acts_with_start["started"]).dt.total_seconds()
        / 86400
    )

    def _early_warning(window_days: int, suffix: str) -> pd.DataFrame:
        early = acts_with_start[
            acts_with_start["days_since_start"].between(0, window_days - 1)
        ]
        agg = (
            early.groupby(["module_id", "user_id"], dropna=False)
            .agg(
                **{f"activities_in_first_{suffix}": ("activity_id", "count")},
                **{f"words_in_first_{suffix}": ("word_count", "sum")},
            )
            .reset_index()
        )
        return agg

    ew_7 = _early_warning(7, "7d")
    ew_14 = _early_warning(14, "14d")

    # ── Merge all dimensions onto base ──────────────────────────────
    result = base.copy()
    for df in [
        volume, quality, vocab_evo, linguistic, type_div, diversity,
        trajectories, fac_user, dg, ew_7, ew_14,
    ]:
        result = result.merge(df, on=["module_id", "user_id"], how="left")

    # ── Temporal / survival features ────────────────────────────────
    result["started"] = pd.to_datetime(result["started"], errors="coerce")
    result["finished"] = pd.to_datetime(result["finished"], errors="coerce")
    result["first_activity"] = pd.to_datetime(result["first_activity"], errors="coerce")
    result["last_activity"] = pd.to_datetime(result["last_activity"], errors="coerce")

    result["duration_days"] = np.where(
        result["finished"].notna(),
        (result["finished"] - result["started"]).dt.total_seconds() / 86400,
        (result["last_activity"] - result["started"]).dt.total_seconds() / 86400,
    )
    result["duration_days"] = result["duration_days"].fillna(1).clip(lower=1)

    result["wrote_anything"] = (
        result["total_activities_submitted"].fillna(0) > 0
    ).astype(int)
    result["received_comment"] = (
        result["total_comments_received"].fillna(0) > 0
    ).astype(int)
    result["posted_in_forum"] = (
        result["total_discussion_replies"].fillna(0) > 0
    ).astype(int)

    # Early warning binaries
    result["wrote_in_first_week"] = (
        result["activities_in_first_7d"].fillna(0) > 0
    ).astype(int)
    result["wrote_in_first_two_weeks"] = (
        result["activities_in_first_14d"].fillna(0) > 0
    ).astype(int)

    # ── Drop internal-only columns ──────────────────────────────────
    result = result.drop(
        columns=["first_activity", "last_activity"], errors="ignore"
    )

    # ── Fill NaN for count/sum features ─────────────────────────────
    fill_zero_cols = [
        "total_activities_submitted", "total_words_written",
        "total_comments_received", "total_discussion_replies",
        "discussion_words_written",
        "activities_in_first_7d", "words_in_first_7d",
        "activities_in_first_14d", "words_in_first_14d",
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
    print()

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

    # --- Summary ---
    starters = user_df[user_df["dropout_label"].notna()]
    completers = starters[starters["dropout_label"] == 0]
    dropouts = starters[starters["dropout_label"] == 1]
    writers = starters[starters["wrote_anything"] == 1]

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
        w = sub[sub["wrote_anything"] == 1]
        print(f"  {name}:")
        print(f"    {len(w)}/{len(sub)} ({len(w)/len(sub)*100:.0f}%) wrote at least once")
        print(f"    Mean activities: {sub['total_activities_submitted'].mean():.1f}")
        print(f"    Mean words:      {sub['total_words_written'].mean():.0f}")

    print()
    print(f"Saved: {act_path}")
    print(f"Saved: {pairs_path}")
    print(f"Saved: {user_path}")


if __name__ == "__main__":
    main()
