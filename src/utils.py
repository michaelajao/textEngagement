"""
Shared utilities for the textual engagement analysis pipeline.

Provides:
- compute_dropout_label: binary outcome from users.finished
- apply_publication_style: matplotlib defaults for paper-ready figures
- Colour palettes for consistent visualisations
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
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
