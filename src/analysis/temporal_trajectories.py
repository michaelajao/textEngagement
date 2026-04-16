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

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import parse_mixed_datetime, PALETTE

from config import save_csv, save_fig

OBS_KEYS = ["module_id", "user_id", "cohort_id"]


def run(data=None):
    print("\n" + "=" * 60)
    print("Temporal Engagement Trajectories")
    print("=" * 60)

    ROOT = Path(__file__).resolve().parent.parent.parent
    CSV_DIR = ROOT / "data" / "csv"
    FEAT_DIR = ROOT / "output" / "features"

    if data is None:
        from config import load_data
        df, writers, _ = load_data()
    else:
        df, writers, _ = data

    df = df.copy()
    df["is_writer"] = (df["total_activities_submitted"] > 0).astype(int)

    act_df = pd.read_csv(FEAT_DIR / "activity_level_features.csv")
    act_df["recorded"] = parse_mixed_datetime(act_df["recorded"])

    page_visits = pd.read_csv(CSV_DIR / "page_visits.csv")
    page_visits["latest"] = parse_mixed_datetime(page_visits["latest"])

    act_clean = act_df.drop(columns=["dropout_label"], errors="ignore")
    user_outcome = df[OBS_KEYS + ["dropout_label", "started"]].drop_duplicates(subset=OBS_KEYS)
    act = act_clean.merge(user_outcome, on=OBS_KEYS, how="inner")
    act["started"] = parse_mixed_datetime(act["started"])
    act["days_since_start"] = (act["recorded"] - act["started"]).dt.total_seconds() / 86400

    # 2a. Cumulative activity curves
    print("\n--- 2a. Cumulative Activity Curves ---")
    max_days = 60
    day_range = np.arange(0, max_days + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]),
                            (1, "Dropouts", PALETTE["red"])]:
        sub_users = df[df["dropout_label"] == g]
        sub_acts = act[act["dropout_label"] == g]
        cum_means = []
        for day in day_range:
            acts_by_day = sub_acts[sub_acts["days_since_start"] <= day]
            user_counts = acts_by_day.groupby(OBS_KEYS).size()
            mean_count = user_counts.sum() / len(sub_users) if len(sub_users) > 0 else 0
            cum_means.append(mean_count)
        ax.plot(day_range, cum_means,
                label=f"{label} (n={len(sub_users):,})",
                color=color, linewidth=2)
    ax.set_xlabel("Days Since Enrolment")
    ax.set_ylabel("Mean Cumulative Activities per Participant")
    ax.set_title("Cumulative Writing Activity: Completers vs Dropouts")
    ax.legend()
    ax.set_xlim(0, max_days)
    save_fig(fig, "fig_trajectory_cumulative_activities")
    plt.close(fig)

    # 2b. Weekly page visit trends
    print("\n--- 2b. Weekly Page Visit Trends ---")
    pv = page_visits.merge(
        df[OBS_KEYS + ["started", "dropout_label"]], on=OBS_KEYS, how="inner"
    )
    pv["days_since_start"] = (pv["latest"] - pv["started"]).dt.total_seconds() / 86400
    pv["week"] = (pv["days_since_start"] // 7).astype("Int64")
    pv = pv[(pv["week"] >= 0) & (pv["week"] <= 8)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]),
                            (1, "Dropouts", PALETTE["red"])]:
        n_users = (df["dropout_label"] == g).sum()
        sub = pv[pv["dropout_label"] == g]
        weekly = sub.groupby("week")["hits"].sum() / n_users
        ax.plot(weekly.index, weekly.values, "o-",
                label=f"{label} (n={n_users:,})",
                color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Week Since Enrolment")
    ax.set_ylabel("Mean Page Hits per Participant")
    ax.set_title("Weekly Page Visit Trends: Completers vs Dropouts")
    ax.legend()
    ax.set_xticks(range(9))
    save_fig(fig, "fig_trajectory_weekly_page_visits")
    plt.close(fig)

    # 2c. Sentiment trajectory
    print("\n--- 2c. Sentiment Trajectory ---")
    act_sorted = act.sort_values(OBS_KEYS + ["recorded"]).copy()
    act_sorted["act_index"] = act_sorted.groupby(OBS_KEYS).cumcount() + 1
    max_index = 15
    fig, ax = plt.subplots(figsize=(10, 5))
    for g, label, color in [(0, "Completers", PALETTE["blue"]),
                            (1, "Dropouts", PALETTE["red"])]:
        sub = act_sorted[(act_sorted["dropout_label"] == g)
                         & (act_sorted["act_index"] <= max_index)]
        means = sub.groupby("act_index")["compound_score"].agg(["mean", "sem"])
        ax.plot(means.index, means["mean"], "o-",
                label=label, color=color, linewidth=2, markersize=5)
        ax.fill_between(means.index,
                        means["mean"] - 1.96 * means["sem"],
                        means["mean"] + 1.96 * means["sem"],
                        alpha=0.2, color=color)
    ax.set_xlabel("Activity Index (chronological)")
    ax.set_ylabel("Mean Sentiment Score")
    ax.set_title("Sentiment Trajectory: First 15 Activities (95% CI)")
    ax.legend()
    ax.set_xlim(1, max_index)
    save_fig(fig, "fig_trajectory_sentiment")
    plt.close(fig)

    # 2d. Engagement velocity
    print("\n--- 2d. Engagement Velocity ---")
    milestones = {"1st activity": 1, "3rd activity": 3, "5th activity": 5, "10th activity": 10}
    vel_rows = []
    for milestone_name, n_acts in milestones.items():
        for g, label in [(0, "Completers"), (1, "Dropouts")]:
            sub = act_sorted[(act_sorted["dropout_label"] == g)
                             & (act_sorted["act_index"] == n_acts)]
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
    save_csv(vel_df, "trajectory_velocity")

    fig, ax = plt.subplots(figsize=(8, 4))
    for g, label, color in [("Completers", "Completers", PALETTE["blue"]),
                            ("Dropouts", "Dropouts", PALETTE["red"])]:
        sub = vel_df[vel_df["group"] == g]
        ax.plot(sub["milestone"], sub["median_days"], "o-",
                label=label, color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Milestone")
    ax.set_ylabel("Median Days to Reach")
    ax.set_title("Engagement Velocity: Time to Milestones")
    ax.legend()
    save_fig(fig, "fig_trajectory_engagement_velocity")
    plt.close(fig)

    # 2e. Disengagement detection
    print("\n--- 2e. Disengagement Detection ---")
    dropout_writers = writers[writers["dropout_label"] == 1].copy()
    dropout_act = act[act["dropout_label"] == 1]
    last_act_day = dropout_act.groupby(OBS_KEYS)["days_since_start"].max().reset_index()
    last_act_day.columns = OBS_KEYS + ["last_activity_day"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    vals = last_act_day["last_activity_day"].dropna()
    vals = vals[vals >= 0]
    ax.hist(vals.clip(upper=60), bins=30, color=PALETTE["red"],
            edgecolor="white", alpha=0.7)
    ax.axvline(vals.median(), color="black", linestyle="--",
               label=f"Median: {vals.median():.0f} days")
    ax.set_xlabel("Last Activity Day (since enrolment)")
    ax.set_ylabel("Count")
    ax.set_title(f"When Dropouts Stop Writing (n={len(vals):,})")
    ax.legend()

    ax = axes[1]
    gap_analysis = []
    for threshold in [3, 7, 14, 21]:
        n_with_gap = 0
        total_dropout_writers = 0
        for keys, grp in dropout_act.groupby(OBS_KEYS):
            if len(grp) < 2:
                continue
            total_dropout_writers += 1
            grp_sorted = grp.sort_values("recorded")
            gaps = grp_sorted["recorded"].diff().dt.total_seconds().dropna() / 86400
            if (gaps > threshold).any():
                n_with_gap += 1
        pct = n_with_gap / total_dropout_writers * 100 if total_dropout_writers > 0 else 0
        gap_analysis.append({
            "gap_threshold_days": threshold,
            "n_with_gap": n_with_gap,
            "total": total_dropout_writers,
            "pct": round(pct, 1),
        })
        print(f"  Gap > {threshold}d: {n_with_gap}/{total_dropout_writers} "
              f"dropout writers ({pct:.1f}%)")

    gap_df = pd.DataFrame(gap_analysis)
    ax.bar([f">{d}d" for d in gap_df["gap_threshold_days"]], gap_df["pct"],
           color=PALETTE["orange"], edgecolor="white")
    ax.set_ylabel("% of Dropout Writers")
    ax.set_xlabel("Gap Threshold")
    ax.set_title("Dropout Writers with Large Gaps Before Last Activity")
    ax.set_ylim(0, 100)
    save_fig(fig, "fig_trajectory_last_active_day")
    plt.close(fig)
    save_csv(gap_df, "trajectory_gap_analysis")

    summary = {
        "dropout_writers_n": len(dropout_writers),
        "last_activity_day_median": round(vals.median(), 1) if len(vals) > 0 else None,
        "last_activity_day_mean": round(vals.mean(), 1) if len(vals) > 0 else None,
        "pct_with_gap_gt_7d": (
            gap_df.loc[gap_df["gap_threshold_days"] == 7, "pct"].values[0]
            if len(gap_df) > 0 else None
        ),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")
    save_csv(pd.DataFrame([summary]), "trajectory_summary")


if __name__ == "__main__":
    run()
