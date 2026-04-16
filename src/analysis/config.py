"""
Shared configuration for all analysis scripts.

Loads the feature table, defines groups, sets up output directories,
and provides common helper functions used across all analyses.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import parse_mixed_datetime, apply_publication_style, PALETTE

FEAT_DIR = ROOT / "output" / "features"
OUT_DIR = ROOT / "output" / "analysis"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for d in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load data ──
def load_data():
    """Load feature table and return (df, writers, groups) tuple."""
    df = pd.read_csv(FEAT_DIR / "user_level_features.csv")
    df["started"] = parse_mixed_datetime(df["started"])
    df["finished"] = parse_mixed_datetime(df["finished"])

    with open(FEAT_DIR / "feature_groups.json") as f:
        groups = json.load(f)

    # duration_days is computed once in src/features.py and persisted to the
    # feature table; reuse it here to keep completion and dropout semantics
    # consistent across feature engineering and downstream survival analysis.
    df["duration_days"] = df["duration_days"].clip(lower=1)

    # Convenience binary flags
    df["is_writer"] = (df["total_activities_submitted"] > 0).astype(int)
    df["received_comment"] = (df["total_comments_received"] > 0).astype(int)
    df["is_poster"] = (df["total_discussion_replies"] > 0).astype(int)

    writers = df[df["is_writer"] == 1].copy()

    return df, writers, groups


# ── Helpers ──
OBS_KEYS = ["module_id", "user_id", "cohort_id"]
ORIGINALS = {"n_logins", "login_span_days", "n_bookmarks", "n_page_visits",
             "n_distinct_pages", "total_activities_submitted",
             "total_comments_received", "total_discussion_replies"}


def mann_whitney_compare(df, columns, group_col="dropout_label"):
    """Mann-Whitney U comparing completers (0) vs dropouts (1).

    Returns DataFrame with feature, medians, rank-biserial r, and p-value.
    """
    comp = df[df[group_col] == 0]
    drop = df[df[group_col] == 1]
    rows = []
    for col in columns:
        c_vals = comp[col].dropna()
        d_vals = drop[col].dropna()
        if len(c_vals) < 2 or len(d_vals) < 2:
            continue
        stat, p = sp_stats.mannwhitneyu(c_vals, d_vals, alternative="two-sided")
        n1, n2 = len(c_vals), len(d_vals)
        r = 1 - (2 * stat) / (n1 * n2)
        rows.append({
            "feature": col,
            "type": "Original" if col in ORIGINALS else "Engineered",
            "compl_median": round(c_vals.median(), 3),
            "drop_median": round(d_vals.median(), 3),
            "rank_biserial_r": round(r, 3),
            "p_value": p,
        })
    return pd.DataFrame(rows).sort_values("rank_biserial_r")


def chi2_or(ct):
    """Chi-square test + Haldane-corrected odds ratio from a 2x2 table."""
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(ct)
    vals = ct.values.flatten() + 0.5  # Haldane correction
    OR = (vals[0] * vals[3]) / (vals[1] * vals[2])
    se = np.sqrt(sum(1 / v for v in vals))
    ci_low = np.exp(np.log(OR) - 1.96 * se)
    ci_high = np.exp(np.log(OR) + 1.96 * se)
    return chi2, p, OR, ci_low, ci_high


def save_csv(df, name):
    """Save a DataFrame to the tables directory."""
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True)
    print(f"  Saved: {path.name}")


def save_fig(fig, name):
    """Save a figure to the figures directory."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path.name}")
