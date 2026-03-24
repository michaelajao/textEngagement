"""
RQ2: Do facilitator comments drive continued engagement?

Analyses
--------
- Commented vs uncommented users (controlling for activity count)
- Before/after comment engagement (continuation rate)
- Comment quality (sentiment, length, latency) vs outcome
- Chi-square test for comment recipients vs non-recipients

Outputs
-------
Tables  rq2_commenter_chi2.csv, rq2_continuation.csv,
        rq2_comment_quality.csv, rq2_commented_stratified.csv
Figs    fig_rq2_engagement_funnel.pdf, fig_rq2_comment_latency.pdf,
        fig_rq2_comment_coverage.pdf, fig_rq2_sentiment_alignment.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis import (
    FIGURES_DIR,
    TABLES_DIR,
    apply_publication_style,
    ensure_output_dirs,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    PALETTE,
    chi2_or_fisher,
)

apply_publication_style()


# ── tests ────────────────────────────────────────────────────────────────────

def commenter_chi2(user: pd.DataFrame) -> dict:
    """Chi-square: comment recipients vs non-recipients (writers only)."""
    writers = user[user["total_activities_submitted"] > 0].copy()
    rec = writers[writers["total_comments_received"] > 0]
    norec = writers[writers["total_comments_received"] == 0]

    table = np.array([
        [(rec["dropout_label"] == 0).sum(),
         (rec["dropout_label"] == 1).sum()],
        [(norec["dropout_label"] == 0).sum(),
         (norec["dropout_label"] == 1).sum()],
    ])
    result = chi2_or_fisher(table)
    result.update({
        "commented_completion_rate": (rec["dropout_label"] == 0).mean(),
        "uncommented_completion_rate": (norec["dropout_label"] == 0).mean(),
        "n_commented": len(rec),
        "n_uncommented": len(norec),
    })
    return result


def commented_vs_not_stratified(user: pd.DataFrame) -> pd.DataFrame:
    """Dropout rate for commented vs uncommented, stratified by activity level."""
    writers = user[user["total_activities_submitted"] > 0].copy()
    writers["received_comment"] = (
        writers["total_comments_received"] > 0
    ).astype(int)
    writers["act_q"] = pd.qcut(
        writers["total_activities_submitted"], q=3,
        labels=["Low", "Medium", "High"], duplicates="drop",
    )
    strat = (
        writers.groupby(["act_q", "received_comment"])
        .agg(n=("dropout_label", "count"),
             dropout_rate=("dropout_label", "mean"))
        .reset_index()
    )
    strat["received_comment"] = strat["received_comment"].map(
        {0: "No", 1: "Yes"}
    )
    return strat


def continued_after_comment(user: pd.DataFrame) -> dict:
    """What fraction of commented users submitted another activity?"""
    commented = user[user["total_comments_received"] > 0].copy()
    if commented.empty:
        return {}
    has_cont = commented["continued_after_comment"].notna()
    continued = commented.loc[has_cont, "continued_after_comment"]
    return {
        "n_commented_users": len(commented),
        "n_with_continuation_data": int(has_cont.sum()),
        "continued_pct": continued.mean() * 100 if len(continued) > 0 else 0,
        "continued_completers": commented.loc[
            (commented["continued_after_comment"] == 1)
            & (commented["dropout_label"] == 0)
        ].shape[0],
        "continued_dropouts": commented.loc[
            (commented["continued_after_comment"] == 1)
            & (commented["dropout_label"] == 1)
        ].shape[0],
    }


def comment_quality_tests(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney on comment properties by outcome."""
    metrics = [
        ("comment_word_count", "Comment Word Count"),
        ("response_hours", "Response Latency (hours)"),
        ("comment_sentiment", "Comment Sentiment"),
    ]
    rows = []
    for col, label in metrics:
        if col not in pairs.columns:
            continue
        c = pairs[pairs["dropout_label"] == 0][col].dropna()
        d = pairs[pairs["dropout_label"] == 1][col].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        u, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        rows.append({
            "metric": label,
            "completers_median": c.median(),
            "dropouts_median": d.median(),
            "completers_mean": c.mean(),
            "dropouts_mean": d.mean(),
            "U_stat": u,
            "p_value": p,
        })
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_engagement_funnel(user: pd.DataFrame) -> None:
    starters = len(user)
    writers = len(user[user["total_activities_submitted"] > 0])
    commented = len(user[user["total_comments_received"] > 0])
    continued = len(user[user["continued_after_comment"] == 1])

    stages = [
        "All Starters", "Wrote ≥1 Activity",
        "Received Comment", "Wrote After Comment",
    ]
    counts = [starters, writers, commented, continued]
    colors = [PALETTE["light_blue"], PALETTE["mid_blue"],
              PALETTE["blue"], PALETTE["deep_blue"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(stages)), counts, color=colors, alpha=0.85)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / starters * 100
        ax.text(
            bar.get_width() + starters * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.0f}%)", va="center", fontsize=10,
        )
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("Number of Participants")
    ax.set_title("Engagement Funnel: Writing and Facilitator Comments")
    ax.invert_yaxis()
    fig.savefig(FIGURES_DIR / "fig_rq2_engagement_funnel.pdf")
    plt.close(fig)
    print("  -> fig_rq2_engagement_funnel.pdf")


def fig_comment_latency(pairs: pd.DataFrame) -> None:
    pairs_clip = pairs[pairs["response_hours"].between(0, 720)]
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        pairs_clip[pairs_clip["dropout_label"] == 0]["response_hours"].dropna(),
        pairs_clip[pairs_clip["dropout_label"] == 1]["response_hours"].dropna(),
    ]
    bp = ax.boxplot(
        data, labels=["Completers", "Dropouts"],
        patch_artist=True, showfliers=False,
    )
    bp["boxes"][0].set_facecolor(OUTCOME_COLORS[0])
    bp["boxes"][1].set_facecolor(OUTCOME_COLORS[1])
    for b in bp["boxes"]:
        b.set_alpha(0.6)
    ax.set_ylabel("Response Latency (hours)")
    ax.set_title("Facilitator Response Time by Outcome")
    fig.savefig(FIGURES_DIR / "fig_rq2_comment_latency.pdf")
    plt.close(fig)
    print("  -> fig_rq2_comment_latency.pdf")


def fig_comment_coverage(user: pd.DataFrame) -> None:
    writers = user[user["total_activities_submitted"] > 0].copy()
    writers["comment_pct_bin"] = pd.cut(
        writers["pct_activities_with_comments"],
        bins=[-1, 0, 25, 50, 75, 100],
        labels=["0%", "1-25%", "26-50%", "51-75%", "76-100%"],
    )
    g = (
        writers.groupby("comment_pct_bin", observed=True)
        .agg(
            n=("dropout_label", "count"),
            completion_rate=(
                "dropout_label", lambda x: (x == 0).mean() * 100
            ),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(g)), g["completion_rate"],
                  color="#2196F3", alpha=0.8)
    for i, row in g.iterrows():
        ax.text(i, row["completion_rate"] + 1.5, f"n={row['n']}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["comment_pct_bin"])
    ax.set_xlabel("% of Activities That Received a Comment")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Comment Coverage vs Completion Rate")
    ax.set_ylim(0, 105)
    fig.savefig(FIGURES_DIR / "fig_rq2_comment_coverage.pdf")
    plt.close(fig)
    print("  -> fig_rq2_comment_coverage.pdf")


def fig_sentiment_alignment(pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in [0, 1]:
        sub = pairs[pairs["dropout_label"] == label]
        ax.scatter(
            sub["activity_sentiment"], sub["comment_sentiment"],
            c=OUTCOME_COLORS[label], alpha=0.3, s=20,
            label=OUTCOME_LABELS[label],
        )
    ax.set_xlabel("Participant Activity Sentiment")
    ax.set_ylabel("Facilitator Comment Sentiment")
    ax.set_title("Sentiment Alignment: Activity vs Response")
    ax.legend()
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, ":", color="gray", alpha=0.5)
    fig.savefig(FIGURES_DIR / "fig_rq2_sentiment_alignment.pdf")
    plt.close(fig)
    print("  -> fig_rq2_sentiment_alignment.pdf")


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    user = data["users"]
    pairs = data["pairs"]
    ensure_output_dirs()

    print("RQ2: Facilitator Comments")
    print("=" * 50)

    print("\nChi-square: commented vs uncommented writers ...")
    chi2 = commenter_chi2(user)
    pd.DataFrame([chi2]).to_csv(
        TABLES_DIR / "rq2_commenter_chi2.csv", index=False
    )
    print(
        f"  p={chi2['p_value']:.2e}, OR={chi2['odds_ratio']:.2f} "
        f"[{chi2['OR_95CI_low']:.2f}, {chi2['OR_95CI_high']:.2f}]"
    )

    print("\nStratified by activity level ...")
    strat = commented_vs_not_stratified(user)
    strat.to_csv(TABLES_DIR / "rq2_commented_stratified.csv", index=False)
    print(strat.to_string(index=False))

    print("\nContinuation after comment ...")
    cont = continued_after_comment(user)
    pd.DataFrame([cont]).to_csv(
        TABLES_DIR / "rq2_continuation.csv", index=False
    )
    print(f"  {cont.get('continued_pct', 0):.1f}% continued writing")

    print("\nComment quality tests ...")
    cq = comment_quality_tests(pairs)
    if not cq.empty:
        cq.to_csv(TABLES_DIR / "rq2_comment_quality.csv", index=False)
        print(cq.round(4).to_string(index=False))

    # Logistic regression: nested models (H2a-c)
    print("\nRQ2 logistic regression (H2a-c) ...")
    from src.analysis import fit_logistic_regression

    writers = user[user["total_activities_submitted"] > 0].copy()
    writers["received_comment"] = (writers["total_comments_received"] > 0).astype(int)
    course_dummies = pd.get_dummies(writers["course_name"], prefix="course", drop_first=True)
    df_rq2 = pd.concat([writers, course_dummies], axis=1)
    control_cols = ["total_activities_submitted", "n_logins"] + [c for c in course_dummies.columns]

    # Model 1: comment receipt only
    m1 = fit_logistic_regression(
        df_rq2, outcome="dropout_label",
        predictors=["received_comment"],
        controls=control_cols,
        label="RQ2-M1: Comment receipt",
    )
    # Model 2: + comment quality
    m2 = fit_logistic_regression(
        df_rq2, outcome="dropout_label",
        predictors=["received_comment", "avg_response_hours", "avg_comment_word_count"],
        controls=control_cols,
        label="RQ2-M2: + Comment quality",
    )
    rq2_reg = pd.concat([m1, m2], ignore_index=True)
    if len(rq2_reg) > 0:
        rq2_reg.to_csv(TABLES_DIR / "rq2_logistic_regression.csv", index=False)
        print(rq2_reg[["model", "feature", "OR", "OR_CI_low", "OR_CI_high", "p_value"]].round(4).to_string(index=False))
    else:
        print("  Models failed to converge.")

    print("\nGenerating figures ...")
    fig_engagement_funnel(user)
    fig_comment_latency(pairs)
    fig_comment_coverage(user)
    fig_sentiment_alignment(pairs)

    print("\nRQ2 done.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
