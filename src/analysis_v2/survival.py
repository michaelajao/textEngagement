"""
Survival Analysis — Kaplan-Meier & Log-Rank Tests
===================================================

Descriptive retention curves comparing engagement groups over time.

Groups compared:
  1. Writers vs non-writers
  2. Comment recipients vs non-recipients (writers only)
  3. Forum posters vs non-posters

Method:
  - Kaplan-Meier estimator per group
  - Log-rank test for each comparison
  - Duration = days from start to completion (censored) or last activity (event)

Note: These are descriptive, not causal. Grouping variables are defined
over follow-up, not at baseline.

Inputs:  output/features/user_level_features_v2.csv
Outputs: output/analysis_v2/tables/survival_*.csv
         output/analysis_v2/figures/fig_km_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from config import load_data, save_csv, save_fig, PALETTE


def _km_plot(data, group_col, labels, title, ax):
    """Fit and plot KM curves for two groups, return log-rank result."""
    kmf = KaplanMeierFitter()
    colors = [PALETTE["blue"], PALETTE["red"]]
    groups = sorted(data[group_col].unique())

    for g, color in zip(groups, colors):
        sub = data[data[group_col] == g]
        kmf.fit(sub["duration_days"], event_observed=(sub["dropout_label"] == 1),
                label=labels[g])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Days Since Enrolment")
    ax.set_ylabel("Retention Probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")

    sub0 = data[data[group_col] == groups[0]]
    sub1 = data[data[group_col] == groups[1]]
    lr = logrank_test(
        sub0["duration_days"], sub1["duration_days"],
        event_observed_A=(sub0["dropout_label"] == 1),
        event_observed_B=(sub1["dropout_label"] == 1),
    )
    return lr


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("Survival Analysis (Kaplan-Meier)")
    print("=" * 60)

    comparisons = [
        ("is_writer", {0: "Non-writers", 1: "Writers"}, "Writers vs Non-Writers", df),
        ("received_comment", {0: "No comments", 1: "Received comments"}, "Comment Receipt (Writers)", writers),
        ("is_poster", {0: "Non-posters", 1: "Forum posters"}, "Forum Posters vs Non-Posters", df),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    results = []

    for (group_col, labels, title, data_sub), ax in zip(comparisons, axes):
        lr = _km_plot(data_sub, group_col, labels, title, ax)
        print(f"\n  {title}")
        print(f"    Log-rank chi2 = {lr.test_statistic:.1f}, p = {lr.p_value:.2e}")
        results.append({
            "comparison": title,
            "log_rank_chi2": round(lr.test_statistic, 1),
            "p_value": lr.p_value,
        })

    save_fig(fig, "fig_km_all")
    plt.close(fig)

    # Also save individual KM figures
    for (group_col, labels, title, data_sub), fname in zip(
        comparisons,
        ["fig_km_writer_nonwriter", "fig_km_commented_vs_not", "fig_km_forum_poster"],
    ):
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        _km_plot(data_sub, group_col, labels, title, ax2)
        save_fig(fig2, fname)
        plt.close(fig2)

    lr_df = pd.DataFrame(results)
    save_csv(lr_df, "survival_logrank")


if __name__ == "__main__":
    run()
