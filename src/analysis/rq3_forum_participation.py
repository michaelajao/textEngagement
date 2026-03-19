"""
RQ3: Does discussion forum participation predict completion?

Analyses
--------
- Participation rate: completers vs dropouts (chi-square)
- Volume: number of posts, word count (Mann-Whitney among posters)
- Timing: early posters (≤14 d) vs late/non-posters

NO BERT models are loaded — uses pre-aggregated forum features from
user_level_features.csv and NLPFeatureExtractor static methods only.

Outputs
-------
Tables  rq3_participation.csv, rq3_forum_volume.csv, rq3_forum_timing.csv
Figs    fig_rq3_participation.pdf, fig_rq3_early_posting.pdf
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
    rank_biserial,
)

apply_publication_style()


# ── tests ────────────────────────────────────────────────────────────────────

def participation_rates(user: pd.DataFrame) -> pd.DataFrame:
    """Forum participation rate by outcome (chi-square)."""
    rows = []
    for label in [0, 1]:
        sub = user[user["dropout_label"] == label]
        rows.append({
            "outcome": OUTCOME_LABELS[label],
            "n_users": len(sub),
            "n_posted": int(sub["posted_in_forum"].sum()),
            "participation_rate": sub["posted_in_forum"].mean() * 100,
        })
    rows.append({
        "outcome": "All",
        "n_users": len(user),
        "n_posted": int(user["posted_in_forum"].sum()),
        "participation_rate": user["posted_in_forum"].mean() * 100,
    })

    # Chi-square on poster vs non-poster × outcome
    table = np.array([
        [
            ((user["dropout_label"] == 0) & (user["posted_in_forum"] == 1)).sum(),
            ((user["dropout_label"] == 1) & (user["posted_in_forum"] == 1)).sum(),
        ],
        [
            ((user["dropout_label"] == 0) & (user["posted_in_forum"] == 0)).sum(),
            ((user["dropout_label"] == 1) & (user["posted_in_forum"] == 0)).sum(),
        ],
    ])
    chi2, p, _, _ = stats.chi2_contingency(table)
    df = pd.DataFrame(rows)
    df["chi2_stat"] = chi2
    df["chi2_p"] = p
    return df


def volume_tests(user: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney on forum posting volume among posters."""
    posters = user[user["posted_in_forum"] == 1].copy()
    comp = posters[posters["dropout_label"] == 0]
    drop = posters[posters["dropout_label"] == 1]

    metrics = [
        ("total_discussion_replies", "Total discussion replies"),
        ("discussion_words_written", "Total discussion words"),
        ("avg_forum_word_count", "Avg forum word count"),
        ("n_topics_participated", "Topics participated"),
        ("forum_sentiment_mean", "Forum sentiment (mean)"),
    ]
    rows = []
    for col, label in metrics:
        if col not in posters.columns:
            continue
        c = comp[col].dropna()
        d = drop[col].dropna()
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
            "rank_biserial_r": rank_biserial(u, len(c), len(d)),
        })
    return pd.DataFrame(rows)


def timing_tests(user: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney on forum timing features among posters."""
    posters = user[user["posted_in_forum"] == 1].copy()
    comp = posters[posters["dropout_label"] == 0]
    drop = posters[posters["dropout_label"] == 1]

    metrics = [
        ("days_to_first_post", "Days to first post"),
        ("forum_span_days", "Forum span (days)"),
    ]
    rows = []
    for col, label in metrics:
        if col not in posters.columns:
            continue
        c = comp[col].dropna()
        d = drop[col].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        u, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        rows.append({
            "metric": label,
            "completers_median": c.median(),
            "dropouts_median": d.median(),
            "U_stat": u,
            "p_value": p,
            "rank_biserial_r": rank_biserial(u, len(c), len(d)),
        })
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_participation(part: pd.DataFrame) -> None:
    sub = part[part["outcome"] != "All"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        sub["outcome"], sub["participation_rate"],
        color=[OUTCOME_COLORS[0], OUTCOME_COLORS[1]], alpha=0.8,
    )
    for bar, row in zip(bars, sub.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{row.participation_rate:.1f}%\n"
            f"(n={row.n_posted}/{row.n_users})",
            ha="center", fontsize=10,
        )
    ax.set_ylabel("Forum Participation Rate (%)")
    ax.set_title("Discussion Forum Participation by Outcome")
    ax.set_ylim(0, max(sub["participation_rate"]) * 1.3)
    fig.savefig(FIGURES_DIR / "fig_rq3_participation.pdf")
    plt.close(fig)
    print("  -> fig_rq3_participation.pdf")


def fig_early_posting(user: pd.DataFrame) -> None:
    """Completion rate by early vs late vs non-posters."""
    u = user.copy()
    u["posting_group"] = np.where(
        u["posted_in_forum"] == 0, "Non-poster",
        np.where(
            u["days_to_first_post"] <= 14,
            "Early poster (≤14d)",
            "Late poster (>14d)",
        ),
    )
    g = (
        u.groupby("posting_group")
        .agg(
            n=("dropout_label", "count"),
            completion_rate=(
                "dropout_label", lambda x: (x == 0).mean() * 100
            ),
        )
        .reset_index()
    )
    order = ["Non-poster", "Late poster (>14d)", "Early poster (≤14d)"]
    g["order"] = g["posting_group"].map({v: i for i, v in enumerate(order)})
    g = g.sort_values("order")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#9E9E9E", "#64B5F6", "#1565C0"]
    bars = ax.bar(range(len(g)), g["completion_rate"],
                  color=colors, alpha=0.85)
    for i, row in enumerate(g.itertuples()):
        ax.text(
            i, row.completion_rate + 1.5,
            f"{row.completion_rate:.1f}%\n(n={row.n})",
            ha="center", fontsize=10,
        )
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["posting_group"], rotation=15, ha="right")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Completion Rate by Forum Posting Timing")
    ax.set_ylim(0, 105)
    fig.savefig(FIGURES_DIR / "fig_rq3_early_posting.pdf")
    plt.close(fig)
    print("  -> fig_rq3_early_posting.pdf")


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    user = data["users"]
    ensure_output_dirs()

    print("RQ3: Forum Participation")
    print("=" * 50)

    print("\nParticipation rates ...")
    part = participation_rates(user)
    part.to_csv(TABLES_DIR / "rq3_participation.csv", index=False)
    print(part[["outcome", "n_users", "n_posted", "participation_rate"]]
          .to_string(index=False))

    print("\nVolume tests (posters only) ...")
    vol = volume_tests(user)
    if not vol.empty:
        vol.to_csv(TABLES_DIR / "rq3_forum_volume.csv", index=False)
        print(vol.round(4).to_string(index=False))
    else:
        print("  Insufficient data for volume tests.")

    print("\nTiming tests (posters only) ...")
    tim = timing_tests(user)
    if not tim.empty:
        tim.to_csv(TABLES_DIR / "rq3_forum_timing.csv", index=False)
        print(tim.round(4).to_string(index=False))
    else:
        print("  Insufficient data for timing tests.")

    print("\nGenerating figures ...")
    fig_participation(part)
    fig_early_posting(user)

    print("\nRQ3 done.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
