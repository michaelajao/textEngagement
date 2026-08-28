"""Leakage diagnostic for the follow-on dropout-prediction study.

Completion is written by the platform when a participant reaches the final content
page, so any model predicting completion from whole-follow-up page-visit features is
partly predicting a page visit from page visits. This quantifies the inflation, and
shows what is left once the exposure is restricted to what is genuinely observable
before the outcome.

Run:  python -m src.analysis.leakage_check_for_ml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
FEAT_DIR = ROOT / "output" / "features"

d = pd.read_csv(FEAT_DIR / "user_level_features.csv", low_memory=False)
y = d["dropout_label"].astype(int)

sub_path = FEAT_DIR / "browsing_subsets.csv"
if sub_path.exists():
    sub = pd.read_csv(sub_path, low_memory=False)
    for c in ["module_id", "user_id", "cohort_id"]:
        d[c] = d[c].astype(str)
        sub[c] = sub[c].astype(str)
    d = d.merge(
        sub[["module_id", "user_id", "cohort_id", "session 1 only", "sessions 1-2 only"]],
        on=["module_id", "user_id", "cohort_id"], how="left",
    )
else:
    print("!! run sensitivity_browsing_outcome.py first to build browsing_subsets.csv")
    sys.exit(1)

d = d.fillna(0)

BLOCKS = {
    "A  whole-follow-up, everything (what a naive model would use)": [
        "n_distinct_pages", "n_page_visits", "n_logins", "n_bookmarks",
        "total_activities_submitted", "total_comments_received",
        "total_discussion_replies", "writing_span_days", "login_span_days",
        "forum_span_days", "activities_in_first_7d",
    ],
    "B  drop the span features (they end at the outcome by construction)": [
        "n_distinct_pages", "n_page_visits", "n_logins", "n_bookmarks",
        "total_activities_submitted", "total_comments_received",
        "total_discussion_replies", "activities_in_first_7d",
    ],
    "C  browsing restricted to sessions 1-2 (cannot contain the trigger)": [
        "sessions 1-2 only", "n_logins", "n_bookmarks",
        "total_activities_submitted", "total_comments_received",
        "total_discussion_replies", "activities_in_first_7d",
    ],
    "D  day-7 observable only (what a deployed model would actually see)": [
        "activities_in_first_7d", "days_to_first_activity",
    ],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("=" * 86)
print("DROPOUT PREDICTION — cross-validated AUC by how much of the outcome is in the features")
print(f"n = {len(d)},  dropout rate = {y.mean():.1%},  5-fold stratified CV, gradient boosting")
print("=" * 86)

results = {}
for name, cols in BLOCKS.items():
    cols = [c for c in cols if c in d.columns]
    X = d[cols].astype(float)
    auc = cross_val_score(
        GradientBoostingClassifier(random_state=0), X, y, cv=cv, scoring="roc_auc"
    )
    results[name] = auc.mean()
    print(f"  {name:<64} AUC {auc.mean():.3f}  (SD {auc.std():.3f})")

a = results["A  whole-follow-up, everything (what a naive model would use)"]
dd = results["D  day-7 observable only (what a deployed model would actually see)"]
print()
print(f"  Inflation from leakage, A over D: {a - dd:+.3f} AUC")
print()
print("  A is not a performance estimate. Span features end at the outcome, and")
print("  distinct pages contains the terminal page that writes the completion")
print("  timestamp. D is the honest ceiling for a day-7 early-warning model on")
print("  these features; anything between is a design choice that must be argued for.")
