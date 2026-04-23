"""
Sensitivity analysis: incremental value of NLP features over volume-only.

Question addressed: do the transformer-derived and RegEx-based NLP features add
predictive information over a model that uses only writing-volume and timing
features? Three nested logistic-regression models are fit on the writer
subsample (n = 1,621):

  M0 (volume only):    activities_in_first_7d + days_to_first_activity +
                       writing_span_days + n_logins + course dummies
  M1 (+ RegEx):        M0 + avg_vocab_richness + avg_self_reference +
                       avg_future_orientation
  M2 (+ transformers): M1 + avg_sentiment + sentiment_trend + word_count_trend

Model comparison uses Delta-AIC and a likelihood-ratio chi-square test between
nested pairs. A meaningful Delta-AIC (> 10) indicates the added block carries
information beyond volume/timing.

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/sensitivity_nlp_value.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from config import load_data, save_csv


VOLUME_FEATS = [
    "activities_in_first_7d",
    "days_to_first_activity",
    "writing_span_days",
    "n_logins",
]
REGEX_FEATS = [
    "avg_vocab_richness",
    "avg_self_reference",
    "avg_future_orientation",
]
TRANSFORMER_FEATS = [
    "avg_sentiment",
    "sentiment_trend",
    "word_count_trend",
]


def _fit_model(df: pd.DataFrame, feats: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit logistic regression of dropout_label on feats + course dummies."""
    X = df[feats].copy()
    X = pd.concat(
        [X, pd.get_dummies(df["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X)
    y = df["dropout_label"]
    return sm.Logit(y, X).fit(disp=0)


def _lr_test(res_reduced, res_full) -> tuple[float, int, float]:
    """Likelihood-ratio chi-square test between nested models."""
    lr = 2 * (res_full.llf - res_reduced.llf)
    df_diff = int(res_full.df_model - res_reduced.df_model)
    p = float(sp_stats.chi2.sf(lr, df_diff))
    return float(lr), df_diff, p


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: NLP incremental value (Delta-AIC)")
    print("=" * 60)

    needed = VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS + ["dropout_label", "course_name"]
    sub = writers[needed].dropna().copy()
    print(f"\nSample: {len(sub):,} writers with complete NLP features")

    m0 = _fit_model(sub, VOLUME_FEATS)
    m1 = _fit_model(sub, VOLUME_FEATS + REGEX_FEATS)
    m2 = _fit_model(sub, VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS)

    lr_01_stat, lr_01_df, lr_01_p = _lr_test(m0, m1)
    lr_12_stat, lr_12_df, lr_12_p = _lr_test(m1, m2)
    lr_02_stat, lr_02_df, lr_02_p = _lr_test(m0, m2)

    rows = [
        {
            "model": "M0 (volume + timing)",
            "features_added": ", ".join(VOLUME_FEATS),
            "n_params": int(m0.df_model) + 1,
            "log_likelihood": float(m0.llf),
            "AIC": float(m0.aic),
            "pseudo_R2": float(m0.prsquared),
            "delta_AIC_vs_M0": 0.0,
            "LR_stat_vs_previous": np.nan,
            "LR_df": np.nan,
            "LR_p_value": np.nan,
        },
        {
            "model": "M1 (+ RegEx linguistic)",
            "features_added": ", ".join(REGEX_FEATS),
            "n_params": int(m1.df_model) + 1,
            "log_likelihood": float(m1.llf),
            "AIC": float(m1.aic),
            "pseudo_R2": float(m1.prsquared),
            "delta_AIC_vs_M0": float(m1.aic - m0.aic),
            "LR_stat_vs_previous": lr_01_stat,
            "LR_df": lr_01_df,
            "LR_p_value": lr_01_p,
        },
        {
            "model": "M2 (+ transformer NLP)",
            "features_added": ", ".join(TRANSFORMER_FEATS),
            "n_params": int(m2.df_model) + 1,
            "log_likelihood": float(m2.llf),
            "AIC": float(m2.aic),
            "pseudo_R2": float(m2.prsquared),
            "delta_AIC_vs_M0": float(m2.aic - m0.aic),
            "LR_stat_vs_previous": lr_12_stat,
            "LR_df": lr_12_df,
            "LR_p_value": lr_12_p,
        },
    ]

    out = pd.DataFrame(rows)
    print()
    print(out[[
        "model", "n_params", "AIC", "pseudo_R2",
        "delta_AIC_vs_M0", "LR_stat_vs_previous", "LR_df", "LR_p_value",
    ]].to_string(index=False))

    print()
    print(f"  LR test M0 -> M1 (+ RegEx):        chi2({lr_01_df}) = {lr_01_stat:.2f}, p = {lr_01_p:.4g}")
    print(f"  LR test M1 -> M2 (+ transformers): chi2({lr_12_df}) = {lr_12_stat:.2f}, p = {lr_12_p:.4g}")
    print(f"  LR test M0 -> M2 (all NLP):        chi2({lr_02_df}) = {lr_02_stat:.2f}, p = {lr_02_p:.4g}")

    save_csv(out, "sensitivity_nlp_value")


if __name__ == "__main__":
    run()
