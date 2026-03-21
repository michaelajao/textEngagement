"""
Machine Learning Comparison: XGBoost vs Logistic Regression

Trains an XGBoost gradient boosting classifier on the same 17
deduplicated features used in the nested logistic regression, with
SHAP explanations for feature importance.

Purpose: test whether non-linear models capture structure missed by
logistic regression.  If XGBoost matches LR (AUC ~0.82), this confirms
interpretable models suffice.

Outputs
-------
Tables  ml_model_comparison.csv, ml_xgboost_best_params.csv,
        ml_shap_importance.csv, ml_prospective_comparison.csv
Figs    fig_ml_roc_comparison.pdf, fig_shap_summary.pdf,
        fig_shap_bar.pdf, fig_shap_waterfall_dropout.pdf,
        fig_shap_waterfall_completer.pdf,
        fig_shap_dependence_*.pdf
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, recall_score, precision_score,
)
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV,
)
from scipy import stats as sp_stats
from xgboost import XGBClassifier
import shap

from src.analysis import (
    FIGURES_DIR,
    TABLES_DIR,
    apply_publication_style,
    OUTCOME_COLORS,
)
from src.analysis.regression import (
    VOLUME, QUALITY, LINGUISTIC, CONTENT,
    TRAJECTORIES, FACILITATOR_FORUM, EARLY_WARNING,
    PROSPECTIVE,
)

apply_publication_style()

# ── All 17 deduplicated features ────────────────────────────────────────────

ALL_FEATURES = (
    VOLUME + QUALITY + LINGUISTIC + CONTENT
    + TRAJECTORIES + FACILITATOR_FORUM + EARLY_WARNING
)

FEATURE_LABELS = {
    "total_activities_submitted": "Total Activities",
    "total_words_written": "Total Words",
    "days_to_first_activity": "Days to First Activity",
    "avg_vocab_richness": "Vocabulary Richness",
    "vocab_evolution": "Vocabulary Evolution",
    "avg_self_reference": "Self-Reference",
    "avg_future_orientation": "Future Orientation",
    "avg_sentiment": "Avg Sentiment",
    "pct_gratitude": "% Gratitude",
    "activity_type_entropy": "Activity Type Entropy",
    "word_count_trend": "Word Count Trend",
    "sentiment_trend": "Sentiment Trend",
    "activity_regularity": "Activity Regularity",
    "pct_activities_with_comments": "% Activities with Comments",
    "total_discussion_replies": "Forum Replies",
    "forum_sentiment_mean": "Forum Sentiment",
    "activities_in_first_7d": "Activities in First 7 Days",
}

# ── Cross-validation setup (matches regression.py) ─────────────────────────

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Data preparation ───────────────────────────────────────────────────────

def _prepare_xy(
    user: pd.DataFrame, features: list[str], *, writers_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select features, fill NaN, return (X, y)."""
    df = user.copy()
    if writers_only:
        df = df[df["wrote_anything"] == 1]
    cols = [c for c in features if c in df.columns]
    X = df[cols].fillna(0)
    y = df["dropout_label"]
    return X, y


# ── XGBoost tuning ─────────────────────────────────────────────────────────

def tune_xgboost(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[XGBClassifier, dict]:
    """RandomizedSearchCV for XGBoost hyperparameters."""
    n_pos = (y == 0).sum()
    n_neg = (y == 1).sum()
    spw = n_pos / n_neg  # scale_pos_weight

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 3, 5],
    }

    xgb = XGBClassifier(
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=50,
        scoring="roc_auc",
        cv=CV,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ── Model comparison ───────────────────────────────────────────────────────

def cv_comparison(
    X: pd.DataFrame, y: pd.Series, xgb_model: XGBClassifier,
) -> pd.DataFrame:
    """Run both LR and XGBoost through the same 5-fold CV.

    Returns per-fold metrics for paired comparison.
    """
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])

    rows = []
    for fold, (train_idx, test_idx) in enumerate(CV.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Logistic Regression
        lr_pipe.fit(X_train, y_train)
        lr_probs = lr_pipe.predict_proba(X_test)[:, 1]
        lr_preds = lr_pipe.predict(X_test)

        # XGBoost
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_preds = xgb_model.predict(X_test)

        for name, probs, preds in [
            ("Logistic Regression", lr_probs, lr_preds),
            ("XGBoost", xgb_probs, xgb_preds),
        ]:
            rows.append({
                "model": name,
                "fold": fold + 1,
                "auc": roc_auc_score(y_test, probs),
                "sensitivity": recall_score(y_test, preds, pos_label=1),
                "specificity": recall_score(y_test, preds, pos_label=0),
                "precision": precision_score(y_test, preds, pos_label=1, zero_division=0),
                "f1": f1_score(y_test, preds, pos_label=1, zero_division=0),
            })

    df = pd.DataFrame(rows)

    # Paired t-test on per-fold AUC
    lr_aucs = df[df["model"] == "Logistic Regression"]["auc"].values
    xgb_aucs = df[df["model"] == "XGBoost"]["auc"].values
    _, p_val = sp_stats.ttest_rel(lr_aucs, xgb_aucs)

    print(f"  LR  mean AUC: {lr_aucs.mean():.4f} +/- {lr_aucs.std():.4f}")
    print(f"  XGB mean AUC: {xgb_aucs.mean():.4f} +/- {xgb_aucs.std():.4f}")
    print(f"  Paired t-test p = {p_val:.4f}")

    return df, p_val


# ── Prospective model ──────────────────────────────────────────────────────

def run_prospective(user: pd.DataFrame) -> pd.DataFrame:
    """Compare LR vs XGBoost on first-week features (all users)."""
    X, y = _prepare_xy(user, PROSPECTIVE, writers_only=False)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )

    rows = []
    for name, model in [("LR", lr_pipe), ("XGBoost", xgb)]:
        aucs = []
        for train_idx, test_idx in CV.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = model.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], probs))
        rows.append({
            "model": name,
            "features": "activities_in_first_7d",
            "n_users": len(X),
            "cv_auc_mean": np.mean(aucs),
            "cv_auc_std": np.std(aucs),
        })

    df = pd.DataFrame(rows)
    print("  Prospective comparison:")
    print(df.to_string(index=False))
    return df


# ── SHAP analysis ──────────────────────────────────────────────────────────

def compute_shap_values(
    model: XGBClassifier, X: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def shap_importance_table(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature, ranked."""
    imp = np.abs(shap_values).mean(axis=0)
    labels = [FEATURE_LABELS.get(c, c) for c in X.columns]
    df = pd.DataFrame({
        "feature": X.columns,
        "label": labels,
        "mean_abs_shap": imp,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


# ── Figures ────────────────────────────────────────────────────────────────

def fig_roc_comparison(
    X: pd.DataFrame, y: pd.Series, xgb_model: XGBClassifier,
) -> None:
    """Overlay ROC curves for LR and XGBoost."""
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])

    fig, ax = plt.subplots(figsize=(8, 7))
    models = [
        ("Logistic Regression", lr_pipe, "#2196F3"),
        ("XGBoost", xgb_model, "#F44336"),
    ]

    for name, model, color in models:
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Logistic Regression vs XGBoost")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ml_roc_comparison.pdf")
    plt.close(fig)
    print("  -> fig_ml_roc_comparison.pdf")


def fig_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """SHAP beeswarm summary plot."""
    X_labelled = X.rename(columns=FEATURE_LABELS)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_labelled, show=False)
    plt.title("SHAP Feature Importance (XGBoost)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_shap_summary.pdf")
    plt.close(fig)
    print("  -> fig_shap_summary.pdf")


def fig_shap_bar(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """Mean |SHAP| bar chart."""
    X_labelled = X.rename(columns=FEATURE_LABELS)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_labelled, plot_type="bar", show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_shap_bar.pdf")
    plt.close(fig)
    print("  -> fig_shap_bar.pdf")


def fig_shap_dependence(
    shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 3,
) -> None:
    """Dependence plots for top features by mean |SHAP|."""
    imp = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(imp)[::-1][:top_n]

    for idx in top_idx:
        feat = X.columns[idx]
        label = FEATURE_LABELS.get(feat, feat)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            idx, shap_values, X.values,
            feature_names=[FEATURE_LABELS.get(c, c) for c in X.columns],
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP Dependence: {label}")
        fig.tight_layout()
        fname = f"fig_shap_dependence_{feat}.pdf"
        fig.savefig(FIGURES_DIR / fname)
        plt.close(fig)
        print(f"  -> {fname}")


def fig_shap_waterfall(
    model: XGBClassifier, X: pd.DataFrame, y: pd.Series,
) -> None:
    """Waterfall plots for a representative dropout and completer."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Rename features for display
    explanation.feature_names = [
        FEATURE_LABELS.get(c, c) for c in X.columns
    ]

    probs = model.predict_proba(X)[:, 1]

    # Representative dropout: median predicted probability among true dropouts
    drop_mask = y == 1
    if drop_mask.any():
        drop_probs = probs[drop_mask]
        median_drop = np.argsort(np.abs(drop_probs - np.median(drop_probs)))[0]
        drop_global_idx = np.where(drop_mask)[0][median_drop]

        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation[drop_global_idx], show=False)
        plt.title("SHAP Waterfall: Representative Dropout", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_shap_waterfall_dropout.pdf")
        plt.close(fig)
        print("  -> fig_shap_waterfall_dropout.pdf")

    # Representative completer: median predicted probability among true completers
    comp_mask = y == 0
    if comp_mask.any():
        comp_probs = probs[comp_mask]
        median_comp = np.argsort(np.abs(comp_probs - np.median(comp_probs)))[0]
        comp_global_idx = np.where(comp_mask)[0][median_comp]

        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation[comp_global_idx], show=False)
        plt.title("SHAP Waterfall: Representative Completer", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_shap_waterfall_completer.pdf")
        plt.close(fig)
        print("  -> fig_shap_waterfall_completer.pdf")


# ── Entry point ────────────────────────────────────────────────────────────

def run(data: dict[str, pd.DataFrame]) -> None:
    """Run full ML comparison analysis."""
    print("=" * 50)
    print("ML COMPARISON: XGBoost vs Logistic Regression")
    print("=" * 50)

    user = data["users"]
    X, y = _prepare_xy(user, ALL_FEATURES, writers_only=True)
    print(f"\nDataset: {len(X)} writers, {(y == 1).sum()} dropouts, {len(ALL_FEATURES)} features")

    # 1. Tune XGBoost
    print("\n1. Tuning XGBoost (RandomizedSearchCV, 50 iterations)...")
    best_xgb, best_params = tune_xgboost(X, y)
    pd.DataFrame([best_params]).to_csv(
        TABLES_DIR / "ml_xgboost_best_params.csv", index=False
    )

    # 2. Cross-validated comparison
    print("\n2. 5-fold CV comparison...")
    comparison_df, paired_p = cv_comparison(X, y, best_xgb)
    summary = comparison_df.groupby("model").agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"),
    ).round(4).reset_index()
    summary["paired_ttest_p"] = paired_p
    summary.to_csv(TABLES_DIR / "ml_model_comparison.csv", index=False)
    print(f"  Saved: ml_model_comparison.csv")

    # 3. Prospective comparison
    print("\n3. Prospective model (first-week features)...")
    prosp = run_prospective(user)
    prosp.to_csv(TABLES_DIR / "ml_prospective_comparison.csv", index=False)

    # 4. Fit final XGBoost on all data for SHAP
    print("\n4. Computing SHAP values...")
    best_xgb.fit(X, y)
    shap_values = compute_shap_values(best_xgb, X)
    imp_df = shap_importance_table(shap_values, X)
    imp_df.to_csv(TABLES_DIR / "ml_shap_importance.csv", index=False)
    print("  Top 5 features by mean |SHAP|:")
    print(imp_df.head().to_string(index=False))

    # 5. Generate figures
    print("\n5. Generating figures...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig_roc_comparison(X, y, best_xgb)
        fig_shap_summary(shap_values, X)
        fig_shap_bar(shap_values, X)
        fig_shap_dependence(shap_values, X, top_n=3)
        fig_shap_waterfall(best_xgb, X, y)

    print("\nML comparison complete.")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables, ensure_output_dirs
    ensure_output_dirs()
    data = load_analytical_tables()
    run(data)
