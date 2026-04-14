"""
Robustness regression models predicting dropout from text-based
engagement features.

Fits a GEE model (exchangeable correlation, module-level clusters)
and a course-controlled logistic regression, plus a VIF check on
the full feature set.

Outputs
-------
Tables  regression_mixed_effects.csv, regression_course_controlled.csv,
        regression_vif.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.analysis import (
    TABLES_DIR,
    apply_publication_style,
    ensure_output_dirs,
    load_analytical_tables,
)

apply_publication_style()


# ── feature groups ───────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Cleaned feature set (19 features, no pair with r > 0.80)
#
# Removed: max_description_length (r=0.85 with avg), avg_sentence_length
# (r=0.80 with avg_description),
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


ALL_FEATURES = (
    VOLUME + QUALITY + LINGUISTIC + CONTENT
    + TRAJECTORIES + FACILITATOR_FORUM + EARLY_WARNING
)


def run_mixed_effects(user: pd.DataFrame) -> pd.DataFrame:
    """GEE logistic regression with exchangeable correlation for module_id.

    Uses a reduced feature set (one key feature per dimension) to avoid
    convergence issues with only 10 clusters and 46 features.
    """
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
    data = load_analytical_tables()
    run(data)
