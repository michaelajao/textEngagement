"""
Sensitivity analysis: prospective-only features measured at day 7.

Rebuilds a strictly prospective feature matrix in which every predictor can
be observed by end of day 7 from a participant's start timestamp, then refits
a logistic regression for programme completion. This addresses the cumulative-
feature concern raised in Section 2 (Methods) and acknowledged in the Results:
many main-analysis features (total activities submitted, forum replies, writing
span) mechanically grow with retention and cannot be used in an early-warning
prediction system.

Prospective features computed over days 0-6:
    n_activities_first_7d              count of activities submitted
    n_words_first_7d                   total words across those activities
    mean_sentiment_first_7d            mean compound sentiment score
    wrote_in_first_week                1 if any activity in days 0-6 else 0
    n_comments_first_7d                facilitator comments received in window
    received_comment_first_7d          binary version
    n_forum_replies_first_7d           forum replies posted in window
    posted_in_first_week               binary version
    pv_pages_first_7d                  distinct platform pages visited in window

All features are observable by day 7; the target (dropout_label) is observed
later, so this is a genuinely prospective specification.

Inputs:  output/features/activity_level_features.csv  (for NLP scores)
         output/features/user_level_features.csv      (for started timestamp,
                                                       pv_pages_first_7d,
                                                       course_name, outcome)
         data/csv/facilitator_comments.csv            (for comments in window)
         data/csv/discussions.csv                     (for forum posts in window)
Outputs: output/analysis/tables/sensitivity_prospective_day7.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import TABLE_DIR, load_data, save_csv
from src.utils import parse_mixed_datetime, assign_discussion_cohorts


FEAT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "features"
CSV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "csv"
WINDOW_DAYS = 7
OBS_KEYS = ["module_id", "user_id", "cohort_id"]


def _days_since(ts_col: pd.Series, start_col: pd.Series) -> pd.Series:
    """Vectorised elapsed-days computation."""
    return (ts_col - start_col).dt.total_seconds() / 86400


def _first_window_agg(
    df: pd.DataFrame,
    keys: list[str],
    start_lookup: pd.DataFrame,
    ts_col: str,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Return one row per key with events in [0, window_days) joined back."""
    merged = df.merge(start_lookup, on=keys, how="inner")
    merged["days_since_start"] = _days_since(merged[ts_col], merged["started"])
    in_window = merged[merged["days_since_start"].between(0, window_days - 1)]
    return in_window


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print(f"Sensitivity Analysis: Prospective day-{WINDOW_DAYS} features only")
    print("=" * 60)

    base = df[OBS_KEYS + ["started", "dropout_label", "course_name",
                          "n_logins", "pv_pages_first_7d"]].copy()
    base["started"] = parse_mixed_datetime(base["started"])
    start_lookup = base[OBS_KEYS + ["started"]].copy()

    # ── Activities in first 7 days ─────────────────────────────────
    act = pd.read_csv(FEAT_DIR / "activity_level_features.csv")
    act["recorded"] = parse_mixed_datetime(act["recorded"])
    act_window = _first_window_agg(act, OBS_KEYS, start_lookup, "recorded")
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
    fc = pd.read_csv(CSV_DIR / "facilitator_comments.csv")
    fc["recorded"] = parse_mixed_datetime(fc["recorded"])
    # Facilitator comments carry activity_id; join via activities to reach user keys.
    act_min = act[OBS_KEYS + ["activity_id"]].drop_duplicates()
    fc_keyed = fc.merge(act_min, on="activity_id", how="inner")
    fc_window = _first_window_agg(fc_keyed, OBS_KEYS, start_lookup, "recorded")
    fc_agg = (
        fc_window.groupby(OBS_KEYS, dropna=False)
        .size()
        .rename("n_comments_first_7d")
        .reset_index()
    )

    # ── Forum replies in first 7 days ──────────────────────────────
    disc = pd.read_csv(CSV_DIR / "discussions.csv")
    disc["recorded"] = parse_mixed_datetime(disc["recorded"])
    disc = assign_discussion_cohorts(
        disc,
        base[OBS_KEYS + ["started"]].rename(columns={"started": "started"})
             .merge(pd.read_csv(CSV_DIR / "users.csv")[OBS_KEYS + ["cohort_name"]],
                    on=OBS_KEYS, how="left"),
    )
    disc = disc[disc["cohort_id"].notna()].copy()
    disc_window = _first_window_agg(disc, OBS_KEYS, start_lookup, "recorded")
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
        "pv_pages_first_7d",
        "n_logins",
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


if __name__ == "__main__":
    run()
