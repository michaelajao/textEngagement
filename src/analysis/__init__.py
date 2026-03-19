"""
Layer 3 analysis package — shared scaffolding.

Centralises paths, data loading, and style imports so that
individual RQ scripts never hard-code directory locations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "output" / "features"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
CSV_DIR = PROJECT_ROOT / "data" / "csv"

# ---------------------------------------------------------------------------
# Re-exports from src.utils
# ---------------------------------------------------------------------------

from src.utils import (  # noqa: E402
    apply_publication_style,
    compute_dropout_label,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    PALETTE,
    ENGAGEMENT_COLOURS,
)

__all__ = [
    "PROJECT_ROOT",
    "FEATURES_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "CSV_DIR",
    "apply_publication_style",
    "compute_dropout_label",
    "OUTCOME_COLORS",
    "OUTCOME_LABELS",
    "PALETTE",
    "ENGAGEMENT_COLOURS",
    "load_analytical_tables",
    "ensure_output_dirs",
    "rank_biserial",
    "chi2_or_fisher",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_analytical_tables() -> dict[str, pd.DataFrame]:
    """Load the 3 Layer-2 analytical tables.

    Returns dict with keys: ``users``, ``activities``, ``pairs``.
    Drops rows where ``dropout_label`` is missing and casts to int.
    """
    users = pd.read_csv(FEATURES_DIR / "user_level_features.csv")
    users = users[users["dropout_label"].notna()].copy()
    users["dropout_label"] = users["dropout_label"].astype(int)

    activities = pd.read_csv(FEATURES_DIR / "activity_level_features.csv")
    activities["recorded"] = pd.to_datetime(
        activities["recorded"], errors="coerce"
    )
    activities = activities[activities["dropout_label"].notna()].copy()
    activities["dropout_label"] = activities["dropout_label"].astype(int)

    pairs = pd.read_csv(
        FEATURES_DIR / "comment_pairs.csv",
        parse_dates=["activity_recorded", "comment_recorded"],
    )
    pairs = pairs[pairs["dropout_label"].notna()].copy()
    pairs["dropout_label"] = pairs["dropout_label"].astype(int)

    return {"users": users, "activities": activities, "pairs": pairs}


# ---------------------------------------------------------------------------
# Shared statistical helpers
# ---------------------------------------------------------------------------


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from a Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)


def chi2_or_fisher(table: np.ndarray) -> dict:
    """Chi-square (or Fisher if expected < 5) on a 2x2 contingency table.

    Returns test name, statistic, p-value, odds ratio with 95% CI
    (Haldane +0.5 correction for robustness when any cell is zero).
    """
    chi2, p, _, expected = stats.chi2_contingency(table)
    if (expected < 5).any():
        _, p = stats.fisher_exact(table)
        test_used = "Fisher exact"
    else:
        test_used = "chi2"

    a, b, c, d = (
        table[0, 0] + 0.5,
        table[0, 1] + 0.5,
        table[1, 0] + 0.5,
        table[1, 1] + 0.5,
    )
    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "test": test_used,
        "chi2_stat": chi2 if test_used == "chi2" else np.nan,
        "p_value": p,
        "odds_ratio": odds_ratio,
        "OR_95CI_low": np.exp(log_or - 1.96 * se),
        "OR_95CI_high": np.exp(log_or + 1.96 * se),
    }
