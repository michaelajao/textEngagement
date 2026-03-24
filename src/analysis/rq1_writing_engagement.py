"""
RQ1: Does writing engagement — volume, depth, and early timing —
     predict programme completion?

Part A  Volume & depth tests (Mann-Whitney, chi-square, dose-response)
Part B  Early writing signals (first 7/14 days)
Part C  Temporal patterns (sentiment trajectory, topics, writing over time)

Outputs
-------
Tables  rq1_mann_whitney.csv, rq1_writer_chi2.csv, rq1_early_writing.csv
Figs    fig_rq1_dose_response.pdf, fig_rq1_boxplots.pdf,
        fig_rq1_sentiment_trajectory.pdf, fig_rq1_topic_prevalence.pdf,
        fig_rq1_early_writing.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
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
    rank_biserial,
    chi2_or_fisher,
)

apply_publication_style()


# ── Part A: volume & depth ──────────────────────────────────────────────────

def mann_whitney_tests(user: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests on 14 engagement metrics, BH-FDR corrected."""
    comp = user[user["dropout_label"] == 0]
    drop = user[user["dropout_label"] == 1]

    metrics = [
        ("total_activities_submitted", "Total activities submitted"),
        ("total_words_written", "Total words written"),
        ("avg_description_length", "Avg description word count"),
        ("writing_span_days", "Writing span (days)"),
        ("avg_vocab_richness", "Avg vocabulary richness"),
        ("avg_self_reference", "Avg self-reference ratio"),
        ("avg_future_orientation", "Avg future orientation"),
        ("avg_sentiment", "Avg sentiment"),
        ("pct_gratitude", "Pct Gratitude activities"),
        ("pct_goalsetting", "Pct GoalSetting activities"),
    ]

    rows = []
    for col, label in metrics:
        if col not in user.columns:
            continue
        c_vals = comp[col].dropna()
        d_vals = drop[col].dropna()
        if len(c_vals) < 2 or len(d_vals) < 2:
            continue
        u, p = stats.mannwhitneyu(c_vals, d_vals, alternative="two-sided")
        r = rank_biserial(u, len(c_vals), len(d_vals))
        rows.append({
            "metric": label,
            "completers_median": c_vals.median(),
            "dropouts_median": d_vals.median(),
            "completers_mean": c_vals.mean(),
            "dropouts_mean": d_vals.mean(),
            "U_statistic": u,
            "p_value": p,
            "rank_biserial_r": r,
        })

    df = pd.DataFrame(rows).sort_values("p_value")
    reject, p_adj, _, _ = multipletests(
        df["p_value"].values, alpha=0.05, method="fdr_bh"
    )
    df["p_value_adj_BH"] = p_adj
    df["significant_005_BH"] = reject
    return df


def writer_nonwriter_chi2(user: pd.DataFrame) -> dict:
    """Chi-square + OR for writer vs non-writer completion."""
    writers = user[user["total_activities_submitted"] > 0]
    nonwriters = user[user["total_activities_submitted"] == 0]

    table = np.array([
        [(writers["dropout_label"] == 0).sum(),
         (writers["dropout_label"] == 1).sum()],
        [(nonwriters["dropout_label"] == 0).sum(),
         (nonwriters["dropout_label"] == 1).sum()],
    ])
    result = chi2_or_fisher(table)
    result.update({
        "writers_completion_rate": (
            (writers["dropout_label"] == 0).mean()
        ),
        "nonwriters_completion_rate": (
            (nonwriters["dropout_label"] == 0).mean()
        ),
        "n_writers": len(writers),
        "n_nonwriters": len(nonwriters),
    })
    return result


# ── Part B: early writing signals ───────────────────────────────────────────

def early_writing_tests(user: pd.DataFrame) -> pd.DataFrame:
    """Chi-square and Mann-Whitney tests on early-engagement features."""
    rows = []

    # Chi-square: first-week / first-two-weeks writers vs non
    # Derive binary flags from activity counts
    user = user.copy()
    user["wrote_in_first_week"] = (user["activities_in_first_7d"].fillna(0) > 0).astype(int)
    user["wrote_in_first_two_weeks"] = (user["activities_in_first_14d"].fillna(0) > 0).astype(int)
    for col, label in [
        ("wrote_in_first_week", "Wrote in first week"),
        ("wrote_in_first_two_weeks", "Wrote in first two weeks"),
    ]:
        yes = user[user[col] == 1]
        no = user[user[col] == 0]
        table = np.array([
            [(yes["dropout_label"] == 0).sum(),
             (yes["dropout_label"] == 1).sum()],
            [(no["dropout_label"] == 0).sum(),
             (no["dropout_label"] == 1).sum()],
        ])
        res = chi2_or_fisher(table)
        res["metric"] = label
        res["yes_completion_rate"] = (yes["dropout_label"] == 0).mean()
        res["no_completion_rate"] = (no["dropout_label"] == 0).mean()
        res["n_yes"] = len(yes)
        res["n_no"] = len(no)
        rows.append(res)

    # Mann-Whitney on continuous early metrics
    comp = user[user["dropout_label"] == 0]
    drop = user[user["dropout_label"] == 1]
    for col, label in [
        ("activities_in_first_7d", "Activities in first 7 days"),
        ("activities_in_first_14d", "Activities in first 14 days"),
        ("days_to_first_activity", "Days to first activity"),
    ]:
        if col not in user.columns:
            continue
        c = comp[col].dropna()
        d = drop[col].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        u, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        rows.append({
            "metric": label,
            "test": "Mann-Whitney U",
            "completers_median": c.median(),
            "dropouts_median": d.median(),
            "U_statistic": u,
            "p_value": p,
            "rank_biserial_r": rank_biserial(u, len(c), len(d)),
        })

    return pd.DataFrame(rows)


# ── Part C: figures ─────────────────────────────────────────────────────────

def fig_dose_response(user: pd.DataFrame) -> None:
    writers = user[user["total_words_written"] > 0].copy()
    writers["word_bin"] = pd.cut(
        writers["total_words_written"],
        bins=[0, 10, 30, 60, 100, 200, 500, float("inf")],
        labels=["1-10", "11-30", "31-60", "61-100", "101-200", "201-500", "500+"],
    )
    nw = user[user["total_words_written"] == 0]
    nw_rate = (nw["dropout_label"] == 0).mean() * 100

    grouped = (
        writers.groupby("word_bin", observed=True)
        .agg(
            n=("dropout_label", "count"),
            completion_rate=("dropout_label", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    # Combine non-writers and writer bins into one sequence
    x_labels = ["None"] + list(grouped["word_bin"])
    x_pos = list(range(len(x_labels)))
    rates = [nw_rate] + list(grouped["completion_rate"])
    counts = [len(nw)] + list(grouped["n"])

    # Line plot with markers
    ax.plot(x_pos, rates, color="#2196F3", marker="o", markersize=8,
            linewidth=2, zorder=3)
    # Annotate sample sizes
    for i, (rate, n) in enumerate(zip(rates, counts)):
        ax.annotate(f"n={n}", (x_pos[i], rate),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8)
    # Shade the non-writer point differently
    ax.plot(0, nw_rate, marker="s", color="#9E9E9E", markersize=10, zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel("Total Words Written")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Dose-Response: Writing Volume vs Completion Rate")
    ax.set_ylim(0, 105)

    rho, p_trend = spearmanr(
        range(len(grouped)), grouped["completion_rate"]
    )
    ax.text(
        0.98, 0.04,
        f"Spearman $\\rho$ = {rho:.3f}, p = {p_trend:.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )
    print(f"  Dose-response trend: Spearman rho={rho:.3f}, p={p_trend:.4f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_rq1_dose_response.pdf")
    plt.close(fig)
    print("  -> fig_rq1_dose_response.pdf")


def fig_boxplots(user: pd.DataFrame) -> None:
    metrics = [
        ("total_activities_submitted", "Total Activities"),
        ("total_words_written", "Total Words"),
        ("avg_description_length", "Avg Word Count"),
        ("avg_vocab_richness", "Vocab Richness"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (col, title) in zip(axes, metrics):
        data = [
            user[user["dropout_label"] == 0][col].dropna(),
            user[user["dropout_label"] == 1][col].dropna(),
        ]
        bp = ax.boxplot(
            data,
            labels=["Completers", "Dropouts"],
            patch_artist=True,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor(OUTCOME_COLORS[0])
        bp["boxes"][1].set_facecolor(OUTCOME_COLORS[1])
        for b in bp["boxes"]:
            b.set_alpha(0.6)
        ax.set_title(title)
    fig.suptitle("Writing Volume: Completers vs Dropouts", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_rq1_boxplots.pdf")
    plt.close(fig)
    print("  -> fig_rq1_boxplots.pdf")


def fig_sentiment_trajectory(act: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in [0, 1]:
        sub = act[act["dropout_label"] == label].copy()
        sub = sub.sort_values(["user_id", "recorded"])
        sub["act_idx"] = sub.groupby(["module_id", "user_id"]).cumcount() + 1
        sub = sub[sub["act_idx"] <= 15]
        means = (
            sub.groupby("act_idx")["compound_score"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax.plot(
            means["act_idx"], means["mean"],
            color=OUTCOME_COLORS[label],
            label=OUTCOME_LABELS[label], marker="o", markersize=4,
        )
        ax.fill_between(
            means["act_idx"],
            means["mean"] - 1.96 * means["sem"],
            means["mean"] + 1.96 * means["sem"],
            alpha=0.15, color=OUTCOME_COLORS[label],
        )
    ax.set_xlabel("Activity Number (chronological)")
    ax.set_ylabel("Mean Compound Sentiment")
    ax.set_title("Sentiment Trajectory by Outcome")
    ax.legend()
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    fig.savefig(FIGURES_DIR / "fig_rq1_sentiment_trajectory.pdf")
    plt.close(fig)
    print("  -> fig_rq1_sentiment_trajectory.pdf")


def fig_topic_prevalence(act: pd.DataFrame) -> None:
    topics = [
        "topic_hope", "topic_anxiety", "topic_gratitude",
        "topic_struggle", "topic_social",
    ]
    labels = ["Hope", "Anxiety", "Gratitude", "Struggle", "Social"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(topics))
    w = 0.35
    for i, (label_val, name) in enumerate(OUTCOME_LABELS.items()):
        sub = act[act["dropout_label"] == label_val]
        means = [sub[t].mean() for t in topics]
        ax.bar(
            x + i * w - w / 2, means, w,
            label=name, color=OUTCOME_COLORS[label_val], alpha=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Topic Score")
    ax.set_title("Topic Prevalence by Outcome")
    ax.legend()
    fig.savefig(FIGURES_DIR / "fig_rq1_topic_prevalence.pdf")
    plt.close(fig)
    print("  -> fig_rq1_topic_prevalence.pdf")


def fig_early_writing(user: pd.DataFrame) -> None:
    """Completion rate by first-week writing status."""
    user = user.copy()
    user["wrote_in_first_week"] = (user["activities_in_first_7d"].fillna(0) > 0).astype(int)
    user["wrote_in_first_two_weeks"] = (user["activities_in_first_14d"].fillna(0) > 0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col, title) in zip(axes, [
        ("wrote_in_first_week", "Wrote in First Week"),
        ("wrote_in_first_two_weeks", "Wrote in First Two Weeks"),
    ]):
        g = (
            user.groupby(col)
            .agg(
                n=("dropout_label", "count"),
                rate=("dropout_label", lambda x: (x == 0).mean() * 100),
            )
            .reset_index()
        )
        labels_map = {0: "No", 1: "Yes"}
        colors = ["#9E9E9E", "#2196F3"]
        bars = ax.bar(
            [labels_map[v] for v in g[col]],
            g["rate"], color=colors, alpha=0.8,
        )
        for bar, row in zip(bars, g.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{row.rate:.1f}%\n(n={row.n})",
                ha="center", fontsize=10,
            )
        ax.set_ylabel("Completion Rate (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)

    fig.suptitle("Early Writing and Completion", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_rq1_early_writing.pdf")
    plt.close(fig)
    print("  -> fig_rq1_early_writing.pdf")


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    """Run all RQ1 analyses."""
    user = data["users"]
    act = data["activities"]
    ensure_output_dirs()

    print("RQ1: Writing Engagement")
    print("=" * 50)

    # Part A
    print("\nMann-Whitney U tests (BH-FDR) ...")
    mw = mann_whitney_tests(user)
    mw.to_csv(TABLES_DIR / "rq1_mann_whitney.csv", index=False)
    n_sig = mw["significant_005_BH"].sum()
    print(f"  {len(mw)} tests, {n_sig} significant after BH-FDR")
    print(
        mw[["metric", "completers_median", "dropouts_median",
            "p_value_adj_BH", "rank_biserial_r"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nWriter vs non-writer chi-square ...")
    chi2 = writer_nonwriter_chi2(user)
    pd.DataFrame([chi2]).to_csv(
        TABLES_DIR / "rq1_writer_chi2.csv", index=False
    )
    print(
        f"  p={chi2['p_value']:.2e}, OR={chi2['odds_ratio']:.2f} "
        f"[{chi2['OR_95CI_low']:.2f}, {chi2['OR_95CI_high']:.2f}]"
    )

    # Part B
    print("\nEarly writing tests ...")
    early = early_writing_tests(user)
    early.to_csv(TABLES_DIR / "rq1_early_writing.csv", index=False)
    print(early[["metric", "p_value"]].round(4).to_string(index=False))

    # Part C: Logistic regression (hypothesis testing)
    print("\nRQ1 logistic regression (H1a-c) ...")
    from src.analysis import fit_logistic_regression, TABLES_DIR as _TD

    course_dummies = pd.get_dummies(user["course_name"], prefix="course", drop_first=True)
    df_rq1 = pd.concat([user, course_dummies], axis=1)
    control_cols = ["n_logins"] + [c for c in course_dummies.columns]

    rq1_reg = fit_logistic_regression(
        df_rq1,
        outcome="dropout_label",
        predictors=["activities_in_first_7d", "avg_vocab_richness", "days_to_first_activity"],
        controls=control_cols,
        label="RQ1: Early writing",
    )
    if len(rq1_reg) > 0:
        rq1_reg.to_csv(TABLES_DIR / "rq1_logistic_regression.csv", index=False)
        print(rq1_reg[["feature", "OR", "OR_CI_low", "OR_CI_high", "p_value"]].round(4).to_string(index=False))
    else:
        print("  Model failed to converge.")

    # Figures
    print("\nGenerating figures ...")
    fig_dose_response(user)
    fig_boxplots(user)
    fig_sentiment_trajectory(act)
    fig_topic_prevalence(act)
    fig_early_writing(user)

    print("\nRQ1 done.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
