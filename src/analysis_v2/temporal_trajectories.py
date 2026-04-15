"""
Temporal Engagement Trajectories
==================================

Tracks within-person engagement patterns over time to understand
WHEN dropouts disengage compared to completers.

Sub-analyses:
  a. Cumulative activity curves (completers vs dropouts)
  b. Weekly page visit trends
  c. Sentiment trajectory by activity index
  d. Engagement velocity (time to milestones)
  e. Disengagement detection (last active day, gap analysis)

Outputs:
  tables/trajectory_*.csv
  figures/fig_trajectory_*.png
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

OBS_KEYS = ["module_id", "user_id", "cohort_id"]
apply_publication_style()


def run():
    print("\n" + "=" * 60)
    print("Temporal Engagement Trajectories")
    print("=" * 60)

    # Load data
    df = pd.read_csv(FEAT_DIR / "user_level_features_v2.csv")
    df["started"] = parse_mixed_datetime(df["started"])
    df["finished"] = parse_mixed_datetime(df["finished"])
    df["is_writer"] = (df["total_activities_submitted"] > 0).astype(int)
    df["dropout_label"] = df["finished"].isna().astype(float)
    df.loc[df["started"].isna(), "dropout_label"] = np.nan

    act_df = pd.read_csv(FEAT_DIR / "activity_level_features.csv")
    act_df["recorded"] = parse_mixed_datetime(act_df["recorded"])

    discussions = pd.read_csv(CSV_DIR / "discussions.csv")
    discussions["recorded"] = parse_mixed_datetime(discussions["recorded"])

    page_visits = pd.read_csv(CSV_DIR / "page_visits.csv")
    page_visits["latest"] = parse_mixed_datetime(page_visits["latest"])

    # Join activities to user outcome (drop old dropout_label from activity CSV first)
    act_clean = act_df.drop(columns=["dropout_label"], errors="ignore")
    user_outcome = df[OBS_KEYS + ["dropout_label", "started"]].drop_duplicates(subset=OBS_KEYS)
    act = act_clean.merge(user_outcome, on=OBS_KEYS, how="inner")
    act["days_since_start"] = (act["recorded"] - act["started"]).dt.total_seconds() / 86400

    writers = df[df["is_writer"] == 1].copy()
    comp_keys = set(zip(df.loc[df["dropout_label"] == 0, "module_id"],
                         df.loc[df["dropout_label"] == 0, "user_id"]))

    # ════════════════════════════════════════════════════════════
    # 2a. Cumulative Activity Curves
    # ════════════════════════════════════════════════════════════
    print("\n--- 2a. Cumulative Activity Curves ---")

    max_days = 60
    day_range = np.arange(0, max_days + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]), (1, "Dropouts", PALETTE["red"])]:
        sub_users = df[df["dropout_label"] == g]
        sub_acts = act[act["dropout_label"] == g]
        # For each day, count mean cumulative activities per user
        cum_means = []
        for day in day_range:
            acts_by_day = sub_acts[sub_acts["days_since_start"] <= day]
            user_counts = acts_by_day.groupby(OBS_KEYS).size()
            # Include non-writers as 0
            mean_count = user_counts.sum() / len(sub_users) if len(sub_users) > 0 else 0
            cum_means.append(mean_count)
        ax.plot(day_range, cum_means, label=f"{label} (n={len(sub_users):,})",
                color=color, linewidth=2)

    ax.set_xlabel("Days Since Enrolment")
    ax.set_ylabel("Mean Cumulative Activities per Participant")
    ax.set_title("Cumulative Writing Activity: Completers vs Dropouts")
    ax.legend()
    ax.set_xlim(0, max_days)
    fig.savefig(FIG_DIR / "fig_trajectory_cumulative_activities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_trajectory_cumulative_activities.png")

    # ════════════════════════════════════════════════════════════
    # 2b. Weekly Page Visit Trends
    # ════════════════════════════════════════════════════════════
    print("\n--- 2b. Weekly Page Visit Trends ---")

    pv = page_visits.merge(df[OBS_KEYS + ["started", "dropout_label"]], on=OBS_KEYS, how="inner")
    pv["days_since_start"] = (pv["latest"] - pv["started"]).dt.total_seconds() / 86400
    pv["week"] = (pv["days_since_start"] // 7).astype(int)
    pv = pv[(pv["week"] >= 0) & (pv["week"] <= 8)]  # first 9 weeks

    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]), (1, "Dropouts", PALETTE["red"])]:
        n_users = (df["dropout_label"] == g).sum()
        sub = pv[pv["dropout_label"] == g]
        weekly = sub.groupby("week")["hits"].sum() / n_users
        ax.plot(weekly.index, weekly.values, "o-", label=f"{label} (n={n_users:,})",
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Week Since Enrolment")
    ax.set_ylabel("Mean Page Hits per Participant")
    ax.set_title("Weekly Page Visit Trends: Completers vs Dropouts")
    ax.legend()
    ax.set_xticks(range(9))
    fig.savefig(FIG_DIR / "fig_trajectory_weekly_page_visits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_trajectory_weekly_page_visits.png")

    # ════════════════════════════════════════════════════════════
    # 2c. Sentiment Trajectory
    # ════════════════════════════════════════════════════════════
    print("\n--- 2c. Sentiment Trajectory ---")

    # Add activity index per user (chronological order)
    act_sorted = act.sort_values(OBS_KEYS + ["recorded"])
    act_sorted["act_index"] = act_sorted.groupby(OBS_KEYS).cumcount() + 1

    max_index = 15
    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]), (1, "Dropouts", PALETTE["red"])]:
        sub = act_sorted[(act_sorted["dropout_label"] == g) & (act_sorted["act_index"] <= max_index)]
        means = sub.groupby("act_index")["compound_score"].agg(["mean", "sem"])
        ax.plot(means.index, means["mean"], "o-", label=label, color=color, linewidth=2, markersize=5)
        ax.fill_between(means.index,
                         means["mean"] - 1.96 * means["sem"],
                         means["mean"] + 1.96 * means["sem"],
                         alpha=0.2, color=color)

    ax.set_xlabel("Activity Index (chronological)")
    ax.set_ylabel("Mean Sentiment Score")
    ax.set_title("Sentiment Trajectory: First 15 Activities (95% CI)")
    ax.legend()
    ax.set_xlim(1, max_index)
    fig.savefig(FIG_DIR / "fig_trajectory_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_trajectory_sentiment.png")

    # ════════════════════════════════════════════════════════════
    # 2d. Engagement Velocity (time to milestones)
    # ════════════════════════════════════════════════════════════
    print("\n--- 2d. Engagement Velocity ---")

    milestones = {
        "1st activity": 1,
        "3rd activity": 3,
        "5th activity": 5,
        "10th activity": 10,
    }

    vel_rows = []
    for milestone_name, n_acts in milestones.items():
        for g, label in [(0, "Completers"), (1, "Dropouts")]:
            sub = act_sorted[(act_sorted["dropout_label"] == g) & (act_sorted["act_index"] == n_acts)]
            if len(sub) > 0:
                days = sub["days_since_start"]
                vel_rows.append({
                    "milestone": milestone_name,
                    "group": label,
                    "n_reached": len(sub),
                    "median_days": round(days.median(), 1),
                    "mean_days": round(days.mean(), 1),
                })

    vel_df = pd.DataFrame(vel_rows)
    print(vel_df.to_string(index=False))
    vel_df.to_csv(TABLE_DIR / "trajectory_velocity.csv", index=False)
    print(f"  Saved: trajectory_velocity.csv")

    # Velocity chart
    fig, ax = plt.subplots(figsize=(8, 4))
    for g, label, color in [("Completers", "Completers", PALETTE["blue"]),
                              ("Dropouts", "Dropouts", PALETTE["red"])]:
        sub = vel_df[vel_df["group"] == g]
        ax.plot(sub["milestone"], sub["median_days"], "o-", label=label, color=color,
                linewidth=2, markersize=8)
    ax.set_xlabel("Milestone")
    ax.set_ylabel("Median Days to Reach")
    ax.set_title("Engagement Velocity: Time to Milestones")
    ax.legend()
    fig.savefig(FIG_DIR / "fig_trajectory_engagement_velocity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_trajectory_engagement_velocity.png")

    # ════════════════════════════════════════════════════════════
    # 2e. Disengagement Detection
    # ════════════════════════════════════════════════════════════
    print("\n--- 2e. Disengagement Detection ---")

    # Last active day for dropouts (writers only)
    dropout_writers = writers[writers["dropout_label"] == 1].copy()
    if "writing_span_days" in dropout_writers.columns and "days_to_first_activity" in dropout_writers.columns:
        dropout_writers["last_active_day"] = (
            dropout_writers["days_to_first_activity"].fillna(0) +
            dropout_writers["writing_span_days"].fillna(0)
        )
    else:
        dropout_writers["last_active_day"] = 0

    # Also get last activity day from raw data
    dropout_act = act[act["dropout_label"] == 1]
    last_act_day = dropout_act.groupby(OBS_KEYS)["days_since_start"].max().reset_index()
    last_act_day.columns = OBS_KEYS + ["last_activity_day"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of last activity day for dropouts
    ax = axes[0]
    vals = last_act_day["last_activity_day"].dropna()
    vals = vals[vals >= 0]
    ax.hist(vals.clip(upper=60), bins=30, color=PALETTE["red"], edgecolor="white", alpha=0.7)
    ax.axvline(vals.median(), color="black", linestyle="--", label=f"Median: {vals.median():.0f} days")
    ax.set_xlabel("Last Activity Day (since enrolment)")
    ax.set_ylabel("Count")
    ax.set_title(f"When Dropouts Stop Writing (n={len(vals):,})")
    ax.legend()

    # Gap analysis: what fraction of dropouts had a gap > 7 days before their last activity?
    ax = axes[1]
    gap_analysis = []
    for threshold in [3, 7, 14, 21]:
        # Dropouts who had at least one gap > threshold days
        n_with_gap = 0
        total_dropout_writers = 0
        for keys, grp in dropout_act.groupby(OBS_KEYS):
            if len(grp) < 2:
                continue
            total_dropout_writers += 1
            grp = grp.sort_values("recorded")
            gaps = grp["recorded"].diff().dt.total_seconds().dropna() / 86400
            if (gaps > threshold).any():
                n_with_gap += 1
        pct = n_with_gap / total_dropout_writers * 100 if total_dropout_writers > 0 else 0
        gap_analysis.append({"gap_threshold_days": threshold, "n_with_gap": n_with_gap,
                              "total": total_dropout_writers, "pct": round(pct, 1)})
        print(f"  Gap > {threshold}d: {n_with_gap}/{total_dropout_writers} dropout writers ({pct:.1f}%)")

    gap_df = pd.DataFrame(gap_analysis)
    ax.bar([f">{d}d" for d in gap_df["gap_threshold_days"]], gap_df["pct"],
           color=PALETTE["orange"], edgecolor="white")
    ax.set_ylabel("% of Dropout Writers")
    ax.set_xlabel("Gap Threshold")
    ax.set_title("Dropout Writers with Large Gaps Before Last Activity")
    ax.set_ylim(0, 100)

    fig.savefig(FIG_DIR / "fig_trajectory_last_active_day.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_trajectory_last_active_day.png")
    gap_df.to_csv(TABLE_DIR / "trajectory_gap_analysis.csv", index=False)
    print(f"  Saved: trajectory_gap_analysis.csv")

    # ── Summary stats ──
    print("\n--- Summary ---")
    summary = {
        "dropout_writers_n": len(dropout_writers),
        "last_activity_day_median": round(vals.median(), 1) if len(vals) > 0 else None,
        "last_activity_day_mean": round(vals.mean(), 1) if len(vals) > 0 else None,
        "pct_with_gap_gt_7d": gap_df.loc[gap_df["gap_threshold_days"] == 7, "pct"].values[0] if len(gap_df) > 0 else None,
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "trajectory_summary.csv", index=False)
    print(f"  Saved: trajectory_summary.csv")


if __name__ == "__main__":
    run()
