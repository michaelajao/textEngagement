"""
Survival / Time-to-Event Analysis

Models time-to-dropout (days from enrolment to last activity/completion)
as a function of text-based engagement.

Analyses:
1. Kaplan-Meier curves: writers vs non-writers, commented vs uncommented,
   forum posters vs non-posters — with log-rank tests.
2. Cox proportional hazards regression with key writing/NLP features.

Event definition:
  - Event (dropout=1): participant started but never finished.
  - Censored (dropout=0): participant completed the programme.
  - Duration: days from ``started`` to ``finished`` (completers) or to
    ``last_activity`` (dropouts). Where neither is available, duration = 1.

Outputs
-------
Tables  survival_logrank_tests.csv, cox_regression_results.csv
Figs    fig_km_writer_nonwriter.pdf, fig_km_commented_vs_not.pdf,
        fig_km_forum_poster.pdf, fig_cox_forest.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from src.analysis import (
    FEATURES_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    apply_publication_style,
    ensure_output_dirs,
    OUTCOME_COLORS,
    PALETTE,
)

apply_publication_style()


# ── data ─────────────────────────────────────────────────────────────────────

def _prepare_survival_data(user: pd.DataFrame) -> pd.DataFrame:
    """Add ``duration_days`` for survival modelling."""
    df = user.copy()
    for col in ["started", "finished", "first_activity", "last_activity"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["duration_days"] = np.where(
        df["dropout_label"] == 0,
        (df["finished"] - df["started"]).dt.days,
        (df["last_activity"] - df["started"]).dt.days,
    )
    df["duration_days"] = df["duration_days"].fillna(1).clip(lower=1)
    return df


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def km_comparison(
    user: pd.DataFrame,
    group_col: str,
    group_labels: dict,
    title: str,
    fname: str,
) -> tuple[list[dict], float, float]:
    """KM curves + log-rank test for a binary grouping variable."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["orange"]]

    groups = sorted(user[group_col].dropna().unique())
    lr_data: list[dict] = []

    for i, g in enumerate(groups):
        sub = user[user[group_col] == g]
        kmf = KaplanMeierFitter()
        label_str = group_labels.get(g, str(g))
        kmf.fit(
            sub["duration_days"],
            event_observed=(sub["dropout_label"] == 1),
            label=f"{label_str} (n={len(sub)})",
        )
        kmf.plot_survival_function(
            ax=ax, color=colors[i % len(colors)], ci_show=True,
        )
        lr_data.append({
            "group": label_str,
            "n": len(sub),
            "n_events": int((sub["dropout_label"] == 1).sum()),
            "median_survival_days": kmf.median_survival_time_,
        })

    # Log-rank test (pairwise)
    sub0 = user[user[group_col] == groups[0]]
    sub1 = user[user[group_col] == groups[1]]
    lr = logrank_test(
        sub0["duration_days"], sub1["duration_days"],
        event_observed_A=(sub0["dropout_label"] == 1),
        event_observed_B=(sub1["dropout_label"] == 1),
    )
    p_val = lr.p_value
    stat = lr.test_statistic

    ax.set_title(f"{title}\nLog-rank: \u03c7\u00b2={stat:.2f}, p={p_val:.2e}")
    ax.set_xlabel("Days Since Enrolment")
    ax.set_ylabel("Survival (Programme Retention)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}  (log-rank p={p_val:.2e})")

    return lr_data, p_val, stat


# ── Cox PH ───────────────────────────────────────────────────────────────────

def run_cox(user: pd.DataFrame) -> tuple[pd.DataFrame, CoxPHFitter]:
    """Cox proportional hazards on key writing features.

    Excluded:
      - writing_span_days: tautological with duration for dropouts
      - continued_after_comment: quasi-leaky with survival time
    """
    cox_features = [
        "total_activities_submitted",
        "total_words_written",
        "avg_vocab_richness",
        "avg_self_reference",
        "avg_future_orientation",
        "total_comments_received",
        "pct_activities_with_comments",
        "total_discussion_replies",
    ]
    cox_features = [c for c in cox_features if c in user.columns]

    df = user[["duration_days", "dropout_label"] + cox_features].copy()
    df[cox_features] = df[cox_features].fillna(0)
    df["event"] = df["dropout_label"].astype(bool)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df[["duration_days", "event"] + cox_features],
        duration_col="duration_days",
        event_col="event",
    )
    cph.print_summary()

    print("\n  Checking proportional hazards assumption ...")
    cph.check_assumptions(
        df[["duration_days", "event"] + cox_features],
        p_value_threshold=0.05,
        show_plots=False,
    )

    summary = cph.summary.copy().reset_index()
    summary.columns = [str(c) for c in summary.columns]
    summary.rename(
        columns={
            "covariate": "feature",
            "exp(coef)": "HR",
            "exp(coef) lower 95%": "HR_CI_low",
            "exp(coef) upper 95%": "HR_CI_high",
            "p": "p_value",
        },
        inplace=True,
    )
    summary["significant_005"] = summary["p_value"] < 0.05
    return summary, cph


# ── figures ──────────────────────────────────────────────────────────────────

def fig_cox_forest(cox_summary: pd.DataFrame) -> None:
    if cox_summary.empty or "HR" not in cox_summary.columns:
        return

    df = cox_summary.sort_values("HR", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
    colors = [
        PALETTE["red"] if p < 0.05 else PALETTE["grey"]
        for p in df["p_value"]
    ]

    for i, row in enumerate(df.itertuples()):
        hr = max(row.HR, 0.001)
        ci_lo = max(row.HR_CI_low, 0.001)
        ci_hi = min(row.HR_CI_high, 200)
        ax.plot(
            [np.log(ci_lo), np.log(ci_hi)], [i, i],
            color=colors[i], lw=2, alpha=0.7,
        )
        ax.scatter([np.log(hr)], [i], color=colors[i], s=60, zorder=3)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("ln(Hazard Ratio)  [HR>1 = higher dropout risk]")
    ax.set_title("Cox Proportional Hazards\nRed = significant (p<0.05)")

    for i, row in enumerate(df.itertuples()):
        ax.text(
            ax.get_xlim()[1] + 0.02, i,
            f"HR={row.HR:.2f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cox_forest.pdf")
    plt.close(fig)
    print("  -> fig_cox_forest.pdf")


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    ensure_output_dirs()
    user = _prepare_survival_data(data["users"])

    print("Survival Analysis")
    print("=" * 50)
    print(
        f"  {len(user)} users, duration range: "
        f"{user['duration_days'].min():.0f}–{user['duration_days'].max():.0f} days"
    )

    logrank_rows: list[dict] = []

    # 1. KM: Writers vs Non-writers
    print("\nKM: Writers vs Non-writers ...")
    user["wrote_anything_bin"] = (
        user["total_activities_submitted"] > 0
    ).astype(int)
    lr1, p1, s1 = km_comparison(
        user, "wrote_anything_bin",
        {0: "Non-writer", 1: "Writer"},
        "Retention: Writers vs Non-Writers",
        "fig_km_writer_nonwriter.pdf",
    )
    logrank_rows.extend(lr1)
    logrank_rows.append({
        "comparison": "Writer vs Non-writer",
        "log_rank_chi2": s1, "p_value": p1,
    })

    # 2. KM: Commented vs not (writers only)
    print("\nKM: Commented vs not (writers only) ...")
    writers = user[user["wrote_anything_bin"] == 1].copy()
    writers["received_comment_bin"] = (
        writers["total_comments_received"] > 0
    ).astype(int)
    lr2, p2, s2 = km_comparison(
        writers, "received_comment_bin",
        {0: "No comment received", 1: "Received \u22651 comment"},
        "Retention: Comment Recipients vs Non-Recipients\n(Writers only)",
        "fig_km_commented_vs_not.pdf",
    )
    logrank_rows.append({
        "comparison": "Commented vs Uncommented (writers)",
        "log_rank_chi2": s2, "p_value": p2,
    })

    # 3. KM: Forum posters vs non-posters
    print("\nKM: Forum posters vs non-posters ...")
    user["posted_forum_bin"] = (
        user["total_discussion_replies"] > 0
    ).astype(int)
    lr3, p3, s3 = km_comparison(
        user, "posted_forum_bin",
        {0: "No forum posts", 1: "Posted in forum"},
        "Retention: Forum Posters vs Non-Posters",
        "fig_km_forum_poster.pdf",
    )
    logrank_rows.append({
        "comparison": "Forum poster vs Non-poster",
        "log_rank_chi2": s3, "p_value": p3,
    })

    pd.DataFrame(logrank_rows).to_csv(
        TABLES_DIR / "survival_logrank_tests.csv", index=False,
    )
    print("\n  Saved: survival_logrank_tests.csv")

    # 4. Cox PH
    print("\nFitting Cox proportional hazards model ...")
    cox_summary, _ = run_cox(user)
    cox_summary.to_csv(
        TABLES_DIR / "cox_regression_results.csv", index=False,
    )
    print("  Saved: cox_regression_results.csv")

    print("\nGenerating figures ...")
    fig_cox_forest(cox_summary)

    print("\nSurvival done.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
