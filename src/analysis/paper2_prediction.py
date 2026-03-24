"""
Prediction-focused analyses extracted from Paper 1 for Paper 2.

Contains:
  - Nested logistic regression with AUC (incremental feature dimensions)
  - XGBoost + SHAP (non-linear comparison, feature importance)
  - Prospective model (first-week early-warning AUC)
  - Cox proportional hazards (time-to-event prediction)

These were removed from Paper 1 (behavioural marker study) to be
incorporated into Paper 2 (ML prediction framework).

Usage:
    python -m src.analysis.paper2_prediction

Requires: output/features/user_level_features.csv
"""

from __future__ import annotations

# ==========================================================================
# The full implementations live in the existing analysis modules.
# This script re-exports them as a single entry point for Paper 2.
# ==========================================================================

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "output" / "features"


def load_data() -> pd.DataFrame:
    """Load user-level features."""
    path = FEATURES_DIR / "user_level_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/features.py --skip-nlp"
        )
    return pd.read_csv(path, parse_dates=["started", "finished"])


def run_nested_regression(user: pd.DataFrame) -> None:
    """Nested logistic regression with AUC — 7 models, incremental features."""
    from src.analysis.regression import run as regression_run
    data = {"users": user}
    regression_run(data)


def run_xgboost_shap(user: pd.DataFrame) -> None:
    """XGBoost vs LR comparison with SHAP feature importance."""
    from src.analysis.ml_models import run as ml_run
    data = {"users": user}
    ml_run(data)


def run_cox_ph(user: pd.DataFrame) -> None:
    """Cox proportional hazards with concordance index."""
    from src.analysis.survival import run as survival_run
    data = {"users": user}
    survival_run(data)


def main():
    print("=" * 60)
    print("PAPER 2: Prediction Analyses")
    print("=" * 60)

    user = load_data()
    print(f"Loaded {len(user)} users\n")

    print("1. Nested Logistic Regression")
    print("-" * 40)
    run_nested_regression(user)
    print()

    print("2. XGBoost + SHAP")
    print("-" * 40)
    run_xgboost_shap(user)
    print()

    print("3. Cox Proportional Hazards + Kaplan-Meier")
    print("-" * 40)
    run_cox_ph(user)
    print()

    print("=" * 60)
    print("All Paper 2 prediction analyses complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
