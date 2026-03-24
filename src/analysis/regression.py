"""
Nested logistic regression predicting dropout from text-based
engagement features.

Seven models with 19 deduplicated features (no pair r > 0.80):
  1. Volume   2. +Quality   3. +Linguistic   4. +Content
  5. +Trajectories   6. +Facilitator/Forum   7. +Early Warning

Each model reports pseudo-R², AIC, BIC, cross-validated AUC, and a
likelihood-ratio test against the previous model.

Outputs
-------
Tables  regression_model_comparison.csv, regression_coefficients.csv
Figs    fig_regression_roc.pdf, fig_regression_forest.pdf,
        fig_summary_4panel.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats as sp_stats
import statsmodels.api as sm
import warnings

from src.analysis import (
    FIGURES_DIR,
    TABLES_DIR,
    apply_publication_style,
    ensure_output_dirs,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    PALETTE,
)

apply_publication_style()


# ── feature groups ───────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Cleaned feature set (19 features, no pair with r > 0.80)
#
# Removed: max_description_length (r=0.85 with avg), avg_sentence_length
# (r=0.80 with avg_description), writing_span_days/longest_gap_days (r=0.93),
# total_comments_received (r=0.92 with total_activities), pct_goalsetting
# (r=0.92 with future_orientation), frequency_decay/continued_after_comment
# (r=0.85), discussion_words_written/n_topics_participated (r>0.80 with
# total_replies), wrote_in_first_week/two_weeks (binary versions of 7d/14d),
# words_in_first_7d/14d (r>0.77 with activities_in_7d/14d), pct_myhope
# (n=267), emotional_range (r=0.73 with sentiment_variance),
# avg_comment_word_count/avg_response_hours (not significant), forum_span_days
# (correlated with duration).
# ---------------------------------------------------------------------------

VOLUME = [
    "total_activities_submitted",
    "total_words_written",
    "days_to_first_activity",
]

QUALITY = [
    "avg_vocab_richness",
    "vocab_evolution",
]

LINGUISTIC = [
    "avg_self_reference",
    "avg_future_orientation",
    "avg_sentiment",
]

CONTENT = [
    "pct_gratitude",
    "activity_type_entropy",
]

TRAJECTORIES = [
    "word_count_trend",
    "sentiment_trend",
    "activity_regularity",
]

FACILITATOR_FORUM = [
    "pct_activities_with_comments",
    "total_discussion_replies",
    "forum_sentiment_mean",
]

EARLY_WARNING = [
    "activities_in_first_7d",
]

# Prospective-only features (measurable before most dropout occurs)
PROSPECTIVE = [
    "activities_in_first_7d",
]

MODEL_SPECS: list[tuple[str, list[str]]] = [
    ("1. Volume", VOLUME),
    ("2. +Quality", VOLUME + QUALITY),
    ("3. +Linguistic", VOLUME + QUALITY + LINGUISTIC),
    ("4. +Content", VOLUME + QUALITY + LINGUISTIC + CONTENT),
    ("5. +Trajectories", VOLUME + QUALITY + LINGUISTIC + CONTENT + TRAJECTORIES),
    ("6. +Facilitator/Forum", VOLUME + QUALITY + LINGUISTIC + CONTENT + TRAJECTORIES + FACILITATOR_FORUM),
    ("7. +Early Warning", VOLUME + QUALITY + LINGUISTIC + CONTENT + TRAJECTORIES + FACILITATOR_FORUM + EARLY_WARNING),
]

DIMENSION_MAP = {}
for feat in VOLUME:
    DIMENSION_MAP[feat] = "Volume"
for feat in QUALITY:
    DIMENSION_MAP[feat] = "Quality"
for feat in LINGUISTIC:
    DIMENSION_MAP[feat] = "Linguistic"
for feat in CONTENT:
    DIMENSION_MAP[feat] = "Content"
for feat in TRAJECTORIES:
    DIMENSION_MAP[feat] = "Trajectories"
for feat in FACILITATOR_FORUM:
    DIMENSION_MAP[feat] = "Facilitator/Forum"
for feat in EARLY_WARNING:
    DIMENSION_MAP[feat] = "Early Warning"


# ── modelling ────────────────────────────────────────────────────────────────

def _prepare_data(
    user: pd.DataFrame, features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to writers, select features, fill NaN with 0."""
    df = user[user["total_activities_submitted"] > 0].copy()
    cols = [c for c in features if c in df.columns]
    X = df[cols].fillna(0)
    y = df["dropout_label"]
    return X, y


def fit_statsmodels(
    X: pd.DataFrame, y: pd.Series,
) -> dict:
    """Fit logistic regression via statsmodels for CIs and LR test."""
    X_const = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.Logit(y, X_const).fit(disp=0, maxiter=200)
    conf = model.conf_int(alpha=0.05)
    results = pd.DataFrame({
        "feature": X.columns,
        "coef": model.params.iloc[1:].values,
        "OR": np.exp(model.params.iloc[1:].values),
        "OR_CI_low": np.exp(conf.iloc[1:, 0].values),
        "OR_CI_high": np.exp(conf.iloc[1:, 1].values),
        "p_value": model.pvalues.iloc[1:].values,
    })
    return {
        "model": model,
        "pseudo_r2": model.prsquared,
        "aic": model.aic,
        "bic": model.bic,
        "log_likelihood": model.llf,
        "n_obs": model.nobs,
        "results": results,
    }


def cv_auc(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
) -> tuple[float, float]:
    """5-fold stratified CV AUC."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X.fillna(0), y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def train_roc(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) fitted on full data (for plotting only)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])
    pipe.fit(X.fillna(0), y)
    probs = pipe.predict_proba(X.fillna(0))[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    return fpr, tpr, roc_auc_score(y, probs)


def run_nested_models(
    user: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all 8 nested models. Return comparison table + full coefficients."""
    comparison_rows = []
    all_coefs = []
    prev_ll = None
    prev_k = 0

    roc_data = {}

    for name, features in MODEL_SPECS:
        X, y = _prepare_data(user, features)
        n_features = X.shape[1]

        # statsmodels fit
        sm_fit = fit_statsmodels(X, y)

        # CV AUC
        auc_mean, auc_std = cv_auc(X, y)

        # LR test vs previous model
        lr_p = np.nan
        if prev_ll is not None:
            lr_stat = -2 * (prev_ll - sm_fit["log_likelihood"])
            lr_df = n_features - prev_k
            if lr_df > 0 and lr_stat > 0:
                lr_p = sp_stats.chi2.sf(lr_stat, lr_df)

        comparison_rows.append({
            "model": name,
            "n_features": n_features,
            "n_obs": sm_fit["n_obs"],
            "pseudo_r2": sm_fit["pseudo_r2"],
            "aic": sm_fit["aic"],
            "bic": sm_fit["bic"],
            "log_likelihood": sm_fit["log_likelihood"],
            "cv_auc_mean": auc_mean,
            "cv_auc_std": auc_std,
            "lr_test_p": lr_p,
        })

        prev_ll = sm_fit["log_likelihood"]
        prev_k = n_features

        # Coefficients from full model
        coef_df = sm_fit["results"].copy()
        coef_df["model"] = name
        coef_df["dimension"] = coef_df["feature"].map(DIMENSION_MAP)
        all_coefs.append(coef_df)

        # ROC data for plotting
        fpr, tpr, auc_train = train_roc(X, y)
        roc_data[name] = (fpr, tpr, auc_mean)

    comparison = pd.DataFrame(comparison_rows)
    coefficients = pd.concat(all_coefs, ignore_index=True)

    # Store ROC data on comparison df as attribute for figure generation
    comparison.attrs["roc_data"] = roc_data

    return comparison, coefficients


# ── figures ──────────────────────────────────────────────────────────────────

def fig_roc_curves(comparison: pd.DataFrame) -> None:
    roc_data = comparison.attrs.get("roc_data", {})
    if not roc_data:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(roc_data)))
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), cmap):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=1.5)
    ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Nested Logistic Regression Models")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES_DIR / "fig_regression_roc.pdf")
    plt.close(fig)
    print("  -> fig_regression_roc.pdf")


def fig_forest_plot(coefficients: pd.DataFrame) -> None:
    """Forest plot of ORs from the full (Model 8) fit."""
    full = coefficients[coefficients["model"] == "7. +Early Warning"].copy()
    sig = full[full["p_value"] < 0.05].sort_values("OR")
    if sig.empty:
        # If no features are significant at 0.05, show top 10 by p-value
        sig = full.nsmallest(10, "p_value").sort_values("OR")

    dim_colors = {
        "Volume": PALETTE["blue"],
        "Quality": PALETTE["green"],
        "Linguistic": PALETTE["purple"],
        "Content": PALETTE["orange"],
        "Trajectories": PALETTE["teal"],
        "Facilitator": PALETTE["red"],
        "Forum": PALETTE["dark_blue"],
        "Early Warning": PALETTE["deep_blue"],
    }

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.4)))
    y_pos = range(len(sig))
    for i, row in enumerate(sig.itertuples()):
        color = dim_colors.get(row.dimension, "gray")
        ax.plot(
            [row.OR_CI_low, row.OR_CI_high], [i, i],
            color=color, lw=2, solid_capstyle="round",
        )
        ax.plot(row.OR, i, "o", color=color, markersize=7)

    ax.axvline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sig["feature"])
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Forest Plot — Full Model Predictors")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_regression_forest.pdf")
    plt.close(fig)
    print("  -> fig_regression_forest.pdf")


def fig_summary_4panel(user: pd.DataFrame) -> None:
    """4-panel summary figure for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Writer vs non-writer completion rate
    ax = axes[0, 0]
    u = user.copy()
    u["group"] = np.where(
        u["total_activities_submitted"] > 0, "Wrote", "Never wrote"
    )
    g = u.groupby("group").agg(
        n=("dropout_label", "count"),
        rate=("dropout_label", lambda x: (x == 0).mean() * 100),
    ).reset_index()
    bars = ax.bar(g["group"], g["rate"],
                  color=[PALETTE["blue"], PALETTE["grey"]], alpha=0.8)
    for bar, row in zip(bars, g.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{row.rate:.1f}% (n={row.n})", ha="center", fontsize=9,
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("A. Writers vs Non-Writers")

    # Panel B: Dose-response
    ax = axes[0, 1]
    writers = user[user["total_words_written"] > 0].copy()
    writers["bin"] = pd.cut(
        writers["total_words_written"],
        bins=[0, 20, 60, 150, 500, float("inf")],
        labels=["1-20", "21-60", "61-150", "151-500", "500+"],
    )
    gb = (
        writers.groupby("bin", observed=True)
        .agg(
            n=("dropout_label", "count"),
            rate=("dropout_label", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
    )
    ax.bar(range(len(gb)), gb["rate"], color=PALETTE["blue"], alpha=0.8)
    for i, row in gb.iterrows():
        ax.text(i, row["rate"] + 1.5, f"n={row['n']}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(gb)))
    ax.set_xticklabels(gb["bin"], rotation=20, ha="right")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("B. Dose-Response: Words Written")
    ax.set_ylim(0, 105)

    # Panel C: Activity type distribution
    ax = axes[1, 0]
    type_cols = ["pct_gratitude", "pct_goalsetting", "pct_emotions"]
    type_labels = ["Gratitude", "GoalSetting", "Emotions"]
    w2 = user[user["total_activities_submitted"] > 0].copy()
    x = np.arange(len(type_cols))
    w = 0.35
    for i, (lv, name) in enumerate(OUTCOME_LABELS.items()):
        sub = w2[w2["dropout_label"] == lv]
        means = [sub[c].mean() * 100 for c in type_cols]
        ax.bar(x + i * w - w / 2, means, w,
               label=name, color=OUTCOME_COLORS[lv], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean % of Activities")
    ax.set_title("C. Activity Type Distribution")
    ax.legend(fontsize=9)

    # Panel D: Comment coverage vs completion
    ax = axes[1, 1]
    w3 = user[user["total_activities_submitted"] > 0].copy()
    w3["cbin"] = pd.cut(
        w3["pct_activities_with_comments"],
        bins=[-1, 0, 50, 100],
        labels=["0%", "1-50%", "51-100%"],
    )
    gc = (
        w3.groupby("cbin", observed=True)
        .agg(
            n=("dropout_label", "count"),
            rate=("dropout_label", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
    )
    ax.bar(range(len(gc)), gc["rate"], color=PALETTE["green"], alpha=0.8)
    for i, row in gc.iterrows():
        ax.text(i, row["rate"] + 1.5, f"n={row['n']}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(gc)))
    ax.set_xticklabels(gc["cbin"])
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("D. Comment Coverage vs Completion")
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Writing Engagement and Facilitator Comments: Key Findings",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_summary_4panel.pdf")
    plt.close(fig)
    print("  -> fig_summary_4panel.pdf")


# ── mixed-effects & course-controlled models ────────────────────────────────

ALL_FEATURES = (
    VOLUME + QUALITY + LINGUISTIC + CONTENT
    + TRAJECTORIES + FACILITATOR_FORUM + EARLY_WARNING
)


def run_mixed_effects(user: pd.DataFrame) -> pd.DataFrame:
    """GEE logistic regression with exchangeable correlation for module_id.

    Uses a reduced feature set (one key feature per dimension) to avoid
    convergence issues with only 10 clusters and 46 features.
    """
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    # Reduced set: one or two key features per dimension (avoids
    # collinearity and convergence issues with only 10 module clusters)
    reduced_features = [
        "total_activities_submitted",   # Volume
        "total_words_written",          # Volume
        "avg_vocab_richness",           # Quality
        "avg_sentiment",                # Linguistic
        "avg_future_orientation",       # Linguistic
        "activity_type_entropy",        # Content
        "word_count_trend",             # Trajectories
        "pct_activities_with_comments", # Facilitator
        "total_discussion_replies",     # Forum
        "activities_in_first_7d",       # Early Warning
    ]

    df = user[user["total_activities_submitted"] > 0].copy()
    features = [c for c in reduced_features if c in df.columns]
    df[features] = df[features].fillna(0)

    # Standardise features for comparable ORs
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # GEE with exchangeable correlation within modules — accounts for
    # within-module correlation without assuming a specific random-effects
    # distribution. Robust (sandwich) standard errors are the default.
    print(f"  Fitting GEE ({len(features)} features, module_id clusters) ...")
    df = df.sort_values("module_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gee = GEE(
            df["dropout_label"],
            sm.add_constant(df[features].astype(float)),
            groups=df["module_id"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        ).fit(maxiter=200)

    print(gee.summary())

    results = pd.DataFrame({
        "feature": features,
        "coef": gee.params.iloc[1:].values,
        "OR": np.exp(gee.params.iloc[1:].values),
        "robust_se": gee.bse.iloc[1:].values,
        "p_value": gee.pvalues.iloc[1:].values,
    })
    conf = gee.conf_int().iloc[1:]
    results["OR_CI_low"] = np.exp(conf.iloc[:, 0].values)
    results["OR_CI_high"] = np.exp(conf.iloc[:, 1].values)
    results["significant_005"] = results["p_value"] < 0.05
    results["dimension"] = results["feature"].map(DIMENSION_MAP)
    return results


def run_course_controlled(user: pd.DataFrame) -> pd.DataFrame:
    """Full logistic regression with course_name dummy variables as controls."""
    df = user[user["total_activities_submitted"] > 0].copy()
    features = [c for c in ALL_FEATURES if c in df.columns]
    df[features] = df[features].fillna(0)

    # Add course dummies (drop first to avoid collinearity)
    course_dummies = pd.get_dummies(df["course_name"], prefix="course", drop_first=True)
    X = pd.concat([df[features], course_dummies], axis=1)
    y = df["dropout_label"]

    X_const = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.Logit(y, X_const).fit(disp=0, maxiter=200)

    conf = model.conf_int(alpha=0.05)
    results = pd.DataFrame({
        "feature": X.columns,
        "coef": model.params.iloc[1:].values,
        "OR": np.exp(model.params.iloc[1:].values),
        "OR_CI_low": np.exp(conf.iloc[1:, 0].values),
        "OR_CI_high": np.exp(conf.iloc[1:, 1].values),
        "p_value": model.pvalues.iloc[1:].values,
    })
    results["significant_005"] = results["p_value"] < 0.05
    results["is_course_dummy"] = results["feature"].str.startswith("course_")
    results["dimension"] = results["feature"].map(DIMENSION_MAP).fillna("Course control")

    # CV AUC with course dummies
    auc_mean, auc_std = cv_auc(X, y)

    print(f"  Course-controlled model: pseudo-R²={model.prsquared:.4f}, "
          f"AIC={model.aic:.1f}, CV AUC={auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  n_features={len(features)} + {len(course_dummies.columns)} course dummies")

    return results, model.prsquared, model.aic, auc_mean


# ── entry point ─────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    user = data["users"]
    ensure_output_dirs()

    # Mixed-effects (GEE with module-level clustering)
    print("=" * 50)
    print("Mixed-Effects Model (GEE, module_id clusters)")
    print("=" * 50)
    me_results = run_mixed_effects(user)
    me_results.to_csv(
        TABLES_DIR / "regression_mixed_effects.csv", index=False,
    )
    n_sig = me_results["significant_005"].sum()
    print(f"\n  {n_sig}/{len(me_results)} features significant after "
          "accounting for module-level clustering")

    # Course-controlled model
    print("\n" + "=" * 50)
    print("Course-Controlled Logistic Regression")
    print("=" * 50)
    cc_results, cc_r2, cc_aic, cc_auc = run_course_controlled(user)
    cc_results.to_csv(
        TABLES_DIR / "regression_course_controlled.csv", index=False,
    )

    # VIF check on full model
    print("\n" + "=" * 50)
    print("VIF Check (full model features)")
    print("=" * 50)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df_vif = user[user["total_activities_submitted"] > 0].copy()
    vif_feats = [c for c in ALL_FEATURES if c in df_vif.columns]
    X_vif = df_vif[vif_feats].fillna(0).astype(float)
    X_vif = sm.add_constant(X_vif)
    vif_rows = []
    for i, feat in enumerate(vif_feats):
        vif_val = variance_inflation_factor(X_vif.values, i + 1)
        flag = " *** HIGH" if vif_val > 5 else ""
        vif_rows.append({"feature": feat, "VIF": round(vif_val, 2)})
        print(f"  {feat:40s} VIF={vif_val:.2f}{flag}")
    pd.DataFrame(vif_rows).to_csv(
        TABLES_DIR / "regression_vif.csv", index=False,
    )

    print("\nGEE and robustness checks done.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
