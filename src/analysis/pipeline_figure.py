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
) -> tuple[float, float, float, float]:
    """Draw a rounded box in axes coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#444444",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
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

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    src_color = "#EAF3FF"
    feat_color = "#FFF1E6"
    agg_color = "#FFF8D9"
    analysis_color = "#EAF7EA"

    browsing = _add_box(
        ax,
        0.03,
        0.84,
        0.20,
        0.10,
        "Platform Browsing",
        "users.csv and page_visits.csv\nlogins, bookmarks, page depth",
        src_color,
    )
    writing = _add_box(
        ax,
        0.27,
        0.84,
        0.20,
        0.10,
        "Participant Writing",
        "activities.csv\nstructured responses and text",
        src_color,
    )
    comments = _add_box(
        ax,
        0.51,
        0.84,
        0.20,
        0.10,
        "Facilitator Comments",
        "facilitator_comments.csv\nresponse timing and comment text",
        src_color,
    )
    forums = _add_box(
        ax,
        0.75,
        0.84,
        0.20,
        0.10,
        "Forum Participation",
        "discussions.csv\nreplies, topics, and sentiment",
        src_color,
    )

    engineering = _add_box(
        ax,
        0.11,
        0.57,
        0.34,
        0.12,
        "Feature Engineering",
        "counts, spans, timing, browsing depth,\nforum breadth, and early-warning measures",
        feat_color,
    )
    nlp = _add_box(
        ax,
        0.55,
        0.57,
        0.34,
        0.12,
        "Text NLP Extraction",
        "sentiment, zero-shot topics, self-reference,\nfuture orientation, and vocabulary richness",
        feat_color,
    )

    tables = _add_box(
        ax,
        0.15,
        0.34,
        0.70,
        0.13,
        "Analytical Feature Tables",
        "activity_level_features.csv, comment_pairs.csv,\nand user_level_features.csv",
        agg_color,
    )

    association = _add_box(
        ax,
        0.03,
        0.09,
        0.29,
        0.11,
        "Primary Analyses",
        "RQ1 writing, RQ2 comments,\nRQ3 forum, and engagement funnel",
        analysis_color,
    )
    temporal = _add_box(
        ax,
        0.355,
        0.09,
        0.29,
        0.11,
        "Temporal And Retention",
        "trajectory plots, sentiment over time,\nand Kaplan-Meier survival curves",
        analysis_color,
    )
    profiling = _add_box(
        ax,
        0.68,
        0.09,
        0.29,
        0.11,
        "Profiles And Robustness",
        "K-means clustering, PCA, t-SNE,\nheatmaps, and GEE robustness",
        analysis_color,
    )

    _add_arrow(
        ax,
        (browsing[0] + browsing[2] * 0.50, browsing[1]),
        (engineering[0] + engineering[2] * 0.22, engineering[1] + engineering[3]),
        rad=0.10,
    )
    _add_arrow(
        ax,
        (writing[0] + writing[2] * 0.35, writing[1]),
        (engineering[0] + engineering[2] * 0.48, engineering[1] + engineering[3]),
        rad=0.03,
    )
    _add_arrow(
        ax,
        (writing[0] + writing[2] * 0.65, writing[1]),
        (nlp[0] + nlp[2] * 0.18, nlp[1] + nlp[3]),
        rad=-0.06,
    )
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.45, comments[1]),
        (engineering[0] + engineering[2] * 0.72, engineering[1] + engineering[3]),
        rad=-0.06,
    )
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.55, comments[1]),
        (nlp[0] + nlp[2] * 0.40, nlp[1] + nlp[3]),
        rad=0.03,
    )
    _add_arrow(
        ax,
        (forums[0] + forums[2] * 0.45, forums[1]),
        (engineering[0] + engineering[2] * 0.92, engineering[1] + engineering[3]),
        rad=-0.10,
    )
    _add_arrow(
        ax,
        (forums[0] + forums[2] * 0.55, forums[1]),
        (nlp[0] + nlp[2] * 0.82, nlp[1] + nlp[3]),
        rad=-0.03,
    )
    _add_arrow(
        ax,
        (engineering[0] + engineering[2] * 0.58, engineering[1]),
        (tables[0] + tables[2] * 0.40, tables[1] + tables[3]),
        rad=-0.03,
    )
    _add_arrow(
        ax,
        (nlp[0] + nlp[2] * 0.42, nlp[1]),
        (tables[0] + tables[2] * 0.60, tables[1] + tables[3]),
        rad=0.03,
    )

    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.22, tables[1]),
        (association[0] + association[2] * 0.50, association[1] + association[3]),
        rad=0.03,
    )
    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.50, tables[1]),
        (temporal[0] + temporal[2] * 0.50, temporal[1] + temporal[3]),
        rad=0.0,
    )
    _add_arrow(
        ax,
        (tables[0] + tables[2] * 0.78, tables[1]),
        (profiling[0] + profiling[2] * 0.50, profiling[1] + profiling[3]),
        rad=-0.03,
    )

    output_path = FIG_DIR / "fig_pipeline_framework.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
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
