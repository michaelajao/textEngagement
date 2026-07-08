"""
Generate the analytical pipeline figure as a PNG.

Four-row layout:
  1. Data sources (four groups)
  2. Feature engineering stages (three parallel steps: direct / derived / NLP)
  3. Unified feature table
  4. Analysis pillars (four roles: primary / temporal / profiles / robustness)

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
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def ensure_output_dirs() -> None:
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
    subtitle_fontsize: float = 7.8,
    title_y_frac: float = 0.74,
    subtitle_y_frac: float = 0.32,
) -> tuple[float, float, float, float]:
    """Draw a rounded box in axes coordinates and return (x, y, w, h)."""
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
        y + h * title_y_frac,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + w / 2,
        y + h * subtitle_y_frac,
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
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.3,
        color="#555555",
        shrinkA=6,
        shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def build_pipeline_figure() -> Path:
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    src_color = "#EAF3FF"
    feat_color = "#FFF1E6"
    agg_color = "#FFF8D9"
    analysis_color = "#EAF7EA"

    # ── Row 1: four data sources ─────────────────────────────────────
    src_w, src_h = 0.225, 0.135
    src_y = 0.830
    src_x = [0.020, 0.265, 0.510, 0.755]

    browsing = _add_box(
        ax, src_x[0], src_y, src_w, src_h,
        "Platform Browsing",
        "Logins, bookmarks,\npage visits & depth",
        src_color,
    )
    writing = _add_box(
        ax, src_x[1], src_y, src_w, src_h,
        "Participant Writing",
        "Structured responses\n& free-text submissions",
        src_color,
    )
    comments = _add_box(
        ax, src_x[2], src_y, src_w, src_h,
        "Facilitator Comments",
        "Response timing\n& comment text",
        src_color,
    )
    forums = _add_box(
        ax, src_x[3], src_y, src_w, src_h,
        "Forum Participation",
        "Replies, topics\n& discussion sentiment",
        src_color,
    )

    # ── Row 2: three feature-engineering stages ──────────────────────
    feat_w, feat_h = 0.295, 0.155
    feat_y = 0.575
    feat_x = [0.020, 0.3525, 0.685]

    direct = _add_box(
        ax, feat_x[0], feat_y, feat_w, feat_h,
        "Direct Aggregation",
        "Counts, sums, spans from\nplatform log (logins, activities,\ncomments, forum replies)",
        feat_color,
    )
    derived = _add_box(
        ax, feat_x[1], feat_y, feat_w, feat_h,
        "Derived Features",
        "Windowed counts, ratios,\nproportions, trajectories, entropy,\nearly-warning measures",
        feat_color,
    )
    nlp = _add_box(
        ax, feat_x[2], feat_y, feat_w, feat_h,
        "NLP Inference",
        "Sentiment, zero-shot topics,\nlinguistic markers (applied to\nwriting, comment, forum text)",
        feat_color,
    )

    # ── Row 3: aggregation table ─────────────────────────────────────
    tables = _add_box(
        ax, 0.15, 0.345, 0.70, 0.13,
        "Analytical Feature Tables",
        "33 user-level engagement features\n(activity-level & comment-pair tables feed the user-level summary)",
        agg_color,
    )

    # ── Row 4: four analysis pillars ─────────────────────────────────
    ana_w, ana_h = 0.225, 0.175
    ana_y = 0.055
    ana_x = [0.010, 0.260, 0.510, 0.760]

    primary = _add_box(
        ax, ana_x[0], ana_y, ana_w, ana_h,
        "Primary Analyses",
        "Mann-Whitney + logistic\nregression: RQ1 writing,\nRQ2 comments, RQ3 forum,\nmilestone prevalence",
        analysis_color,
        subtitle_y_frac=0.36,
    )
    temporal = _add_box(
        ax, ana_x[1], ana_y, ana_w, ana_h,
        "Temporal & Retention",
        "Kaplan-Meier + log-rank\nsurvival; trajectory plots,\nsentiment over time,\ndisengagement detection",
        analysis_color,
        subtitle_y_frac=0.36,
    )
    profiles = _add_box(
        ax, ana_x[2], ana_y, ana_w, ana_h,
        "Engagement Profiles",
        "K-means clustering\n(all 33 features);\nPCA, t-SNE projections,\nfeature heatmap",
        analysis_color,
        subtitle_y_frac=0.36,
    )
    robustness = _add_box(
        ax, ana_x[3], ana_y, ana_w, ana_h,
        "Robustness & Sensitivity",
        "GEE with sandwich SEs;\ncluster bootstrap CIs;\nE-values; outcome-definition\nsensitivity checks",
        analysis_color,
        subtitle_y_frac=0.36,
    )

    # ── Arrows: sources → feature-engineering row ────────────────────
    # Each data source sends a single straight-down arrow to the top of the
    # feature-engineering band; the NLP subtitle makes it explicit that
    # platform data does not enter NLP, without needing tangled connectors.
    feat_top = feat_y + feat_h
    for src in (browsing, writing, comments, forums):
        src_cx = src[0] + src[2] / 2
        _add_arrow(
            ax,
            (src_cx, src[1]),
            (src_cx, feat_top),
            rad=0.0,
        )

    # ── Arrows: feature-engineering row → aggregation table ──────────
    table_top = tables[1] + tables[3]
    for feat in (direct, derived, nlp):
        feat_cx = feat[0] + feat[2] / 2
        _add_arrow(
            ax,
            (feat_cx, feat[1]),
            (feat_cx, table_top),
            rad=0.0,
        )

    # ── Arrows: aggregation table → four analysis pillars ────────────
    table_bx_left = tables[0] + tables[2] * 0.20
    table_bx_mid_left = tables[0] + tables[2] * 0.40
    table_bx_mid_right = tables[0] + tables[2] * 0.60
    table_bx_right = tables[0] + tables[2] * 0.80
    table_by = tables[1]

    _add_arrow(
        ax,
        (table_bx_left, table_by),
        (primary[0] + primary[2] / 2, primary[1] + primary[3]),
        rad=0.06,
    )
    _add_arrow(
        ax,
        (table_bx_mid_left, table_by),
        (temporal[0] + temporal[2] / 2, temporal[1] + temporal[3]),
        rad=0.02,
    )
    _add_arrow(
        ax,
        (table_bx_mid_right, table_by),
        (profiles[0] + profiles[2] / 2, profiles[1] + profiles[3]),
        rad=-0.02,
    )
    _add_arrow(
        ax,
        (table_bx_right, table_by),
        (robustness[0] + robustness[2] / 2, robustness[1] + robustness[3]),
        rad=-0.06,
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
    build_pipeline_figure()


def main() -> None:
    build_pipeline_figure()


if __name__ == "__main__":
    main()
