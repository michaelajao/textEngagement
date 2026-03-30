"""
Generate the analytical pipeline figure as a PNG.

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PALETTE, apply_publication_style

apply_publication_style()

FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def ensure_output_dirs() -> None:
    """Create the output figures directory if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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


def _center_top(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h


def _center_bottom(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y


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


def build_pipeline_figure() -> None:
    """Create and save the analytical framework figure."""
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    src_color = "#EAF3FF"
    feat_color = "#FFF1E6"
    agg_color = "#FFF8D9"
    analysis_color = "#EAF7EA"

    writing = _add_box(
        ax, 0.05, 0.82, 0.24, 0.11,
        "Participant Writing",
        "Structured activities\nand reflections",
        src_color,
    )
    comments = _add_box(
        ax, 0.38, 0.82, 0.24, 0.11,
        "Facilitator Comments",
        "Responses to participant\nwriting",
        src_color,
    )
    forums = _add_box(
        ax, 0.71, 0.82, 0.24, 0.11,
        "Discussion Forums",
        "Peer posts\nand replies",
        src_color,
    )

    nlp = _add_box(
        ax, 0.18, 0.57, 0.27, 0.11,
        "Transformer NLP",
        "Sentiment and\ntopic signals",
        feat_color,
    )
    linguistic = _add_box(
        ax, 0.55, 0.57, 0.27, 0.11,
        "Linguistic Markers",
        "Self-reference, future\norientation, vocabulary",
        feat_color,
    )

    measures = _add_box(
        ax, 0.27, 0.36, 0.46, 0.12,
        "User-Level Engagement Measures",
        "Writing, linguistic, facilitator,\nforum, and timing measures",
        agg_color,
    )

    univariate = _add_box(
        ax, 0.06, 0.12, 0.26, 0.11,
        "Association Tests",
        "Mann-Whitney and\nchi-square tests",
        analysis_color,
    )
    robustness = _add_box(
        ax, 0.37, 0.12, 0.26, 0.11,
        "Robustness Checks",
        "Adjusted models\nand GEE",
        analysis_color,
    )
    profiling = _add_box(
        ax, 0.68, 0.12, 0.26, 0.11,
        "Engagement Profiling",
        "Kaplan-Meier and\nK-means clustering",
        analysis_color,
    )

    # Use offset anchors and gentle curvature to avoid overlap in dense regions.
    _add_arrow(
        ax,
        (writing[0] + writing[2] * 0.68, writing[1]),
        (nlp[0] + nlp[2] * 0.28, nlp[1] + nlp[3]),
        rad=0.02,
    )
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.22, comments[1]),
        (nlp[0] + nlp[2] * 0.75, nlp[1] + nlp[3]),
        rad=-0.06,
    )
    _add_arrow(
        ax,
        (comments[0] + comments[2] * 0.78, comments[1]),
        (linguistic[0] + linguistic[2] * 0.25, linguistic[1] + linguistic[3]),
        rad=0.06,
    )
    _add_arrow(
        ax,
        (forums[0] + forums[2] * 0.32, forums[1]),
        (linguistic[0] + linguistic[2] * 0.72, linguistic[1] + linguistic[3]),
        rad=-0.02,
    )

    _add_arrow(
        ax,
        (nlp[0] + nlp[2] * 0.62, nlp[1]),
        (measures[0] + measures[2] * 0.42, measures[1] + measures[3]),
        rad=-0.04,
    )
    _add_arrow(
        ax,
        (linguistic[0] + linguistic[2] * 0.38, linguistic[1]),
        (measures[0] + measures[2] * 0.58, measures[1] + measures[3]),
        rad=0.04,
    )

    _add_arrow(
        ax,
        (measures[0] + measures[2] * 0.40, measures[1]),
        (univariate[0] + univariate[2] * 0.50, univariate[1] + univariate[3]),
        rad=0.02,
    )
    _add_arrow(
        ax,
        (measures[0] + measures[2] * 0.50, measures[1]),
        (robustness[0] + robustness[2] * 0.50, robustness[1] + robustness[3]),
        rad=0.0,
    )
    _add_arrow(
        ax,
        (measures[0] + measures[2] * 0.60, measures[1]),
        (profiling[0] + profiling[2] * 0.50, profiling[1] + profiling[3]),
        rad=-0.02,
    )

    fig.savefig(
        FIGURES_DIR / "fig_pipeline_framework.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def run(data: dict | None = None) -> None:
    """Match the analysis module interface used by run_all."""
    build_pipeline_figure()


def main() -> None:
    build_pipeline_figure()


if __name__ == "__main__":
    main()
