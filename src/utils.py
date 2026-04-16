"""
Shared utilities for the textual engagement analysis pipeline.

Provides:
- compute_dropout_label: binary outcome from users.finished
- apply_publication_style: matplotlib defaults for paper-ready figures
- Colour palettes for consistent visualisations
"""

from __future__ import annotations

import matplotlib
import pandas as pd


# ---------------------------------------------------------------------------
# Outcome variable
# ---------------------------------------------------------------------------

def compute_dropout_label(users_df: pd.DataFrame) -> pd.Series:
    """Return 1 for dropout (finished is null), 0 for completer, NaN if not started."""
    label = users_df["finished"].isna().astype(float)
    # Users who never started are not in the analysis
    label[users_df["started"].isna()] = float("nan")
    return label


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed-format ISO timestamps consistently.

    Platform exports contain a mix of timestamps with and without fractional
    seconds. Pandas' default vectorized parser may infer a single format from
    the first non-null value and silently coerce valid rows in the alternate
    format to ``NaT``. Using ``format='mixed'`` preserves those enrolments.
    """
    return pd.to_datetime(series, errors="coerce", format="mixed")


def assign_discussion_cohorts(
    discussions_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign module-level discussion posts to cohort enrolments.

    Discussion exports do not include ``cohort_id``. We therefore align each
    post to the cohort for the same ``(module_id, user_id)`` whose start date
    is the latest one at or before the post timestamp. If no start date
    precedes the post, we fall back to the nearest available cohort start.
    Rows without any started cohort remain unmatched.
    """
    if discussions_df.empty:
        result = discussions_df.copy()
        result["cohort_id"] = pd.Series(dtype="float64")
        result["cohort_name"] = pd.Series(dtype="object")
        return result

    cohort_map = users_df[
        ["module_id", "user_id", "cohort_id", "cohort_name", "started"]
    ].copy()
    cohort_map["started"] = parse_mixed_datetime(cohort_map["started"])
    cohort_map = cohort_map[cohort_map["started"].notna()].copy()

    result = discussions_df.copy()
    result["recorded"] = parse_mixed_datetime(result["recorded"])

    if cohort_map.empty:
        result["cohort_id"] = pd.Series(dtype="float64")
        result["cohort_name"] = pd.Series(dtype="object")
        return result

    matched = result.merge(
        cohort_map,
        on=["module_id", "user_id"],
        how="left",
    )
    matched["started_before_post"] = matched["started"] <= matched["recorded"]
    matched["abs_gap_seconds"] = (
        matched["recorded"] - matched["started"]
    ).abs().dt.total_seconds()
    matched = matched.sort_values(
        ["reply_id", "started_before_post", "abs_gap_seconds", "started"],
        ascending=[True, False, True, False],
        na_position="last",
    )
    matched = matched.drop_duplicates(subset=["reply_id"], keep="first")
    return matched


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

PALETTE = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#9E9E9E",
    "dark_blue": "#0D47A1",
    "light_blue": "#BBDEFB",
    "mid_blue": "#64B5F6",
    "deep_blue": "#1565C0",
}

OUTCOME_COLORS = {0: PALETTE["blue"], 1: PALETTE["red"]}
OUTCOME_LABELS = {0: "Completers", 1: "Dropouts"}

ENGAGEMENT_COLOURS = {
    "High": PALETTE["green"],
    "Medium": PALETTE["orange"],
    "Low": PALETTE["red"],
}


# ---------------------------------------------------------------------------
# Publication-ready matplotlib style
# ---------------------------------------------------------------------------

def apply_publication_style(two_col: bool = False) -> None:
    """Configure matplotlib for publication-quality figures.

    Parameters
    ----------
    two_col : bool
        If True, use a narrower default figure width suitable for
        two-column journal layouts (~3.5 in).  Otherwise use a
        single-column width (~7 in).
    """
    width = 3.5 if two_col else 7.0
    matplotlib.rcParams.update({
        "figure.figsize": (width, width * 0.7),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })
