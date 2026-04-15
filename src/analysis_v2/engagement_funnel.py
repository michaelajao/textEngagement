"""
Engagement Funnel Analysis
============================

Traces the full participant journey from enrolment to completion,
measuring conversion rates and dropout at each stage.

Stages:
  1. Enrolled        — all enrolment records in the database
  2. Initiated       — started the programme (has start timestamp)
  3. Browsed         — visited at least one page (n_page_visits > 0)
  4. Wrote           — submitted at least one writing activity
  5. Received comment — got at least one facilitator comment
  6. Posted in forum — posted at least one discussion reply
  7. Completed       — platform recorded a finished timestamp

Outputs:
  tables/funnel_overall.csv, funnel_by_course.csv, funnel_dropout_by_stage.csv
  figures/fig_funnel_overall.png, fig_funnel_by_course.png, fig_funnel_dropout_by_stage.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import parse_mixed_datetime, apply_publication_style, PALETTE

CSV_DIR = ROOT / "data" / "csv"
FEAT_DIR = ROOT / "output" / "features"
FIG_DIR = ROOT / "output" / "analysis_v2" / "figures"
TABLE_DIR = ROOT / "output" / "analysis_v2" / "tables"

apply_publication_style()


def run():
    print("\n" + "=" * 60)
    print("Engagement Funnel Analysis")
    print("=" * 60)

    # Load raw data for total enrolment count
    users_raw = pd.read_csv(CSV_DIR / "users.csv")
    users_raw["started"] = parse_mixed_datetime(users_raw["started"])
    users_raw["finished"] = parse_mixed_datetime(users_raw["finished"])
    n_enrolled = len(users_raw)

    # Load v2 features (reclassified starters)
    df = pd.read_csv(FEAT_DIR / "user_level_features_v2.csv")
    df["started"] = parse_mixed_datetime(df["started"])
    df["finished"] = parse_mixed_datetime(df["finished"])

    # ── Build funnel stages ──
    n_initiated = len(df)
    n_browsed = (df["n_page_visits"] > 0).sum()
    n_wrote = (df["total_activities_submitted"] > 0).sum()
    n_commented = (df["total_comments_received"] > 0).sum()
    n_posted = (df["total_discussion_replies"] > 0).sum()
    n_completed = df["finished"].notna().sum()

    stages = pd.DataFrame([
        {"stage": "Enrolled", "n": n_enrolled},
        {"stage": "Initiated", "n": n_initiated},
        {"stage": "Browsed pages", "n": int(n_browsed)},
        {"stage": "Wrote", "n": int(n_wrote)},
        {"stage": "Received comment", "n": int(n_commented)},
        {"stage": "Posted in forum", "n": int(n_posted)},
        {"stage": "Completed", "n": int(n_completed)},
    ])
    stages["pct_of_enrolled"] = (stages["n"] / n_enrolled * 100).round(1)
    stages["pct_of_previous"] = [100.0] + [
        round(stages.iloc[i]["n"] / stages.iloc[i - 1]["n"] * 100, 1)
        if stages.iloc[i - 1]["n"] > 0 else 0.0
        for i in range(1, len(stages))
    ]

    print("\n--- Overall Funnel ---")
    for _, row in stages.iterrows():
        bar = "█" * int(row["pct_of_enrolled"] / 2)
        print(f"  {row['stage']:20s} {row['n']:>6,}  ({row['pct_of_enrolled']:5.1f}%)  {bar}")
    stages.to_csv(TABLE_DIR / "funnel_overall.csv", index=False)
    print(f"  Saved: funnel_overall.csv")

    # ── Funnel bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE["grey"], PALETTE["dark_blue"], PALETTE["blue"], PALETTE["green"],
              PALETTE["orange"], PALETTE["purple"], PALETTE["teal"]]
    bars = ax.barh(stages["stage"][::-1], stages["n"][::-1],
                   color=colors[::-1], edgecolor="white")
    for bar, row in zip(bars, stages.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + n_enrolled * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row.n:,} ({row.pct_of_enrolled:.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("Number of Participants")
    ax.set_title("Engagement Funnel: Enrolment to Completion")
    ax.set_xlim(0, n_enrolled * 1.25)
    fig.savefig(FIG_DIR / "fig_funnel_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_funnel_overall.png")

    # ── Dropout rate by stage reached ──
    # For each stage, what's the dropout rate among those who reached it?
    df["reached_browsed"] = (df["n_page_visits"] > 0).astype(int)
    df["reached_wrote"] = (df["total_activities_submitted"] > 0).astype(int)
    df["reached_commented"] = (df["total_comments_received"] > 0).astype(int)
    df["reached_forum"] = (df["total_discussion_replies"] > 0).astype(int)
    df["is_dropout"] = df["finished"].isna().astype(int)

    stage_dropout = []
    for stage, col in [("All starters", None),
                        ("Browsed", "reached_browsed"),
                        ("Wrote", "reached_wrote"),
                        ("Received comment", "reached_commented"),
                        ("Posted in forum", "reached_forum")]:
        if col is None:
            sub = df
        else:
            sub = df[df[col] == 1]
        n = len(sub)
        n_drop = sub["is_dropout"].sum()
        stage_dropout.append({
            "stage": stage,
            "n": n,
            "n_dropout": int(n_drop),
            "dropout_pct": round(n_drop / n * 100, 1) if n > 0 else 0,
        })

    sd_df = pd.DataFrame(stage_dropout)
    print("\n--- Dropout Rate by Stage Reached ---")
    for _, row in sd_df.iterrows():
        print(f"  {row['stage']:20s}: {row['dropout_pct']:5.1f}% dropout (n={row['n']:,})")
    sd_df.to_csv(TABLE_DIR / "funnel_dropout_by_stage.csv", index=False)
    print(f"  Saved: funnel_dropout_by_stage.csv")

    # Dropout by stage chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sd_df["stage"], sd_df["dropout_pct"],
                  color=[PALETTE["red"], PALETTE["orange"], PALETTE["blue"],
                         PALETTE["green"], PALETTE["purple"]],
                  edgecolor="white")
    for bar, n in zip(bars, sd_df["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n:,}", ha="center", fontsize=8)
    ax.set_ylabel("Dropout Rate (%)")
    ax.set_title("Dropout Rate by Furthest Engagement Stage Reached")
    ax.set_ylim(0, max(sd_df["dropout_pct"]) + 10)
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(FIG_DIR / "fig_funnel_dropout_by_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_funnel_dropout_by_stage.png")

    # ── Funnel by course ──
    print("\n--- Funnel by Course ---")
    course_rows = []
    for course in sorted(df["course_name"].unique()):
        sub = df[df["course_name"] == course]
        course_enrolled = len(users_raw[users_raw["course_name"] == course])
        course_rows.append({
            "course": course,
            "enrolled": course_enrolled,
            "initiated": len(sub),
            "browsed": int((sub["n_page_visits"] > 0).sum()),
            "wrote": int((sub["total_activities_submitted"] > 0).sum()),
            "commented": int((sub["total_comments_received"] > 0).sum()),
            "forum": int((sub["total_discussion_replies"] > 0).sum()),
            "completed": int(sub["finished"].notna().sum()),
        })

    course_df = pd.DataFrame(course_rows)
    # Convert to percentages of enrolled
    for col in ["initiated", "browsed", "wrote", "commented", "forum", "completed"]:
        course_df[f"{col}_pct"] = (course_df[col] / course_df["enrolled"] * 100).round(1)

    print(course_df[["course", "enrolled", "initiated", "browsed", "wrote",
                      "commented", "forum", "completed"]].to_string(index=False))
    course_df.to_csv(TABLE_DIR / "funnel_by_course.csv", index=False)
    print(f"  Saved: funnel_by_course.csv")

    # Small-multiple funnel by course
    courses = course_df.sort_values("enrolled", ascending=False)
    stage_cols = ["initiated_pct", "browsed_pct", "wrote_pct", "commented_pct", "forum_pct", "completed_pct"]
    stage_labels = ["Initiated", "Browsed", "Wrote", "Commented", "Forum", "Completed"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    for i, (_, row) in enumerate(courses.iterrows()):
        if i >= 7:
            break
        ax = axes[i]
        vals = [row[c] for c in stage_cols]
        bars = ax.barh(stage_labels[::-1], vals[::-1], color=PALETTE["blue"], edgecolor="white")
        ax.set_xlim(0, 105)
        ax.set_title(f"{row['course']}\n(n={row['enrolled']})", fontsize=9)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%", va="center", fontsize=7)
    # Hide unused subplot
    if len(courses) < 8:
        axes[7].set_visible(False)
    fig.suptitle("Engagement Funnel by Course (% of enrolled)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_funnel_by_course.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_funnel_by_course.png")


if __name__ == "__main__":
    run()
