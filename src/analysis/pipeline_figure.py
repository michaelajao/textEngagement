"""
Generate the analytical pipeline figure as a PNG.

Recovered from the earlier GitHub history and aligned with the current
analysis output directory.

Usage:
    python -m src.analysis.pipeline_figure
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, ANALYSIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

FIG_DIR = PROJECT_ROOT / "output" / "analysis" / "figures"
matplotlib.rcParams.update(
    {
        "figure.figsize": (7.0, 4.9),
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
    }
)


def ensure_output_dirs() -> None:
    """Create the output figures directory if needed."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    facecolor: str,
    title_fontsize: float = 10.5,
    subtitle_fontsize: float = 8.0,
) -> tuple[float, float, float, float]:
    """Draw a rounded box in axes coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.1,
        edgecolor="#444444",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.74,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + w / 2,
        y + h * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_fontsize,
        linespacing=1.25,
        transform=ax.transAxes,
    )
    return x, y, w, h


def _add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.0,
) -> None:
    """Draw a clean arrow between two points in axes coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color="#555555",
        shrinkA=8,
        shrinkB=8,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def build_pipeline_figure() -> Path:
    """Create and save the analytical framework figure."""
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    src_color = "#EAF3FF"
    feat_color = "#FFF1E6"
    agg_color = "#FFF8D9"
    analysis_color = "#EAF7EA"

    # Top row: four data sources. Each box is 0.225 wide with 0.008 horizontal
    # gap so two-line subtitles fit comfortably without spilling over.
    src_w, src_h = 0.225, 0.135
    src_y = 0.825
    src_x = [0.020, 0.265, 0.510, 0.755]

    browsing = _add_box(
        ax,
        src_x[0],
        src_y,
        src_w,
        src_h,
        "Platform Browsing",
        "Logins, bookmarks,\npage visits & depth",
        src_color,
    )
    writing = _add_box(
        ax,
        src_x[1],
        src_y,
        src_w,
        src_h,
        "Participant Writing",
        "Structured responses\n& free-text submissions",
        src_color,
    )
    comments = _add_box(
        ax,
        src_x[2],
        src_y,
        src_w,
        src_h,
        "Facilitator Comments",
        "Response timing\n& comment text",
        src_color,
    )
    forums = _add_box(
        ax,
        src_x[3],
        src_y,
        src_w,
        src_h,
        "Forum Participation",
        "Replies, topics\n& discussion sentiment",
        src_color,
    )

    # Middle row: feature engineering and NLP extraction.
    engineering = _add_box(
        ax,
        0.07,
        0.555,
        0.39,
        0.135,
        "Feature Engineering",
        "counts, spans, timing, browsing depth,\nforum breadth, early-warning measures",
        feat_color,
    )
    nlp = _add_box(
        ax,
        0.54,
        0.555,
        0.39,
        0.135,
        "Text NLP Extraction",
        "sentiment, zero-shot topics, self-reference,\nfuture orientation, vocabulary richness",
        feat_color,
    )

    # Aggregation row.
    tables = _add_box(
        ax,
        0.15,
        0.32,
        0.70,
        0.13,
        "Analytical Feature Tables",
        "activity-level, comment-pair,\nand user-level engagement measures",
        agg_color,
    )

    # Bottom row: three analysis pillars.
    ana_w, ana_h = 0.295, 0.13
    ana_y = 0.07
    association = _add_box(
        ax,
        0.020,
        ana_y,
        ana_w,
        ana_h,
        "Primary Analyses",
        "RQ1 writing, RQ2 comments,\nRQ3 forum, engagement funnel",
        analysis_color,
    )
    temporal = _add_box(
        ax,
        0.3525,
        ana_y,
        ana_w,
        ana_h,
        "Temporal & Retention",
        "trajectory plots, sentiment over time,\nKaplan-Meier survival curves",
        analysis_color,
    )
    profiling = _add_box(
        ax,
        0.685,
        ana_y,
        ana_w,
        ana_h,
        "Profiles & Robustness",
        "K-means clustering, PCA, t-SNE,\nheatmaps, GEE robustness",
        analysis_color,
    )

    # ---- Arrows: sources -> feature engineering / NLP --------------------
    # Browsing feeds Feature Engineering only (no text component).
    _add_arrow(
        ax,
        (browsing[0] + browsing[2] * 0.50, browsing[1]),
        (engineering[0] + engineering[2] * 0.15, engineering[1] + engineering[3]),
        rad=0.10,
    )
    # Writing -> both blocks.
    _add_arrow(
        ax,
        (writing[0] + writing[2] * 0.40, writing[1]),
        (engineering[0] + engineering[2] * 0.55, engineering[1] + engineering[3]),
        rad=0.05,
    )
    _add_arrow(
        ax,
        (writing[0] + writing[2] * 0.70, writing[1]),
        (nlp[0] + nlp[2] * 0.20, nlp[1] + nlp[3]),
        rad=-0.05,
    )
    # Comments -> both blocks.
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.30, comments[1]),
        (engineering[0] + engineering[2] * 0.85, engineering[1] + engineering[3]),
        rad=0.05,
    )
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.60, comments[1]),
        (nlp[0] + nlp[2] * 0.45, nlp[1] + nlp[3]),
        rad=-0.05,
    )
    # Forums -> NLP (single arrow, centred).
    _add_arrow(
        ax,
        (forums[0] + forums[2] * 0.50, forums[1]),
        (nlp[0] + nlp[2] * 0.82, nlp[1] + nlp[3]),
        rad=-0.04,
    )

    # ---- Feature blocks -> aggregation table ----------------------------
    _add_arrow(
        ax,
        (engineering[0] + engineering[2] * 0.55, engineering[1]),
        (tables[0] + tables[2] * 0.30, tables[1] + tables[3]),
        rad=-0.03,
    )
    _add_arrow(
        ax,
        (nlp[0] + nlp[2] * 0.45, nlp[1]),
        (tables[0] + tables[2] * 0.70, tables[1] + tables[3]),
        rad=0.03,
    )

    # ---- Aggregation table -> analyses ----------------------------------
    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.18, tables[1]),
        (association[0] + association[2] * 0.55, association[1] + association[3]),
        rad=0.05,
    )
    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.50, tables[1]),
        (temporal[0] + temporal[2] * 0.50, temporal[1] + temporal[3]),
        rad=0.0,
    )
    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.82, tables[1]),
        (profiling[0] + profiling[2] * 0.45, profiling[1] + profiling[3]),
        rad=-0.05,
    )

    output_path = FIG_DIR / "fig_pipeline_framework.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)
    print(f"  Saved: {output_path.name}")
    return output_path


def run(data: dict | None = None) -> None:
    """Match the analysis module interface used by run_all."""
    build_pipeline_figure()


def main() -> None:
    build_pipeline_figure()


if __name__ == "__main__":
    main()
