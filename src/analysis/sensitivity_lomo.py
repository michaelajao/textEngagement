"""
Sensitivity analysis: Leave-one-module-out (LOMO) replication of the GEE.

Refits the main GEE 10 times, each time holding out one of the 10 modules, and
records the odds ratio and the sandwich 95% CI for each of the four features
flagged as significant in the full-sample analysis. A finding is considered
LOMO-stable if its OR and sign are preserved across all 10 held-out fits.

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/sensitivity_lomo.csv
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from config import load_data, save_csv

from gee_robustness import GEE_FEATS


HEADLINE_FEATS = [
    "total_activities_submitted",
    "n_distinct_pages",
    "word_count_trend",
    "total_discussion_replies",
]


def _fit_gee(df: pd.DataFrame, feats: list[str]):
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(df[feats]),
        columns=feats, index=df.index,
    )
    X = sm.add_constant(X)
    gee = sm.GEE(
        df["dropout_label"], X,
        groups=df["module_id"],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return gee.fit()


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: Leave-one-module-out (LOMO) GEE")
    print("=" * 60)

    feats = [f for f in GEE_FEATS if f in writers.columns]
    gee_df = writers[["module_id", "dropout_label"] + feats].dropna().copy()
    modules = sorted(gee_df["module_id"].unique())
    print(f"\nHolding out each of {len(modules)} modules in turn; {len(feats)} features per fit.")

    warnings.filterwarnings("ignore")

    rows = []
    for held_out in modules:
        sub = gee_df[gee_df["module_id"] != held_out].copy()
        try:
            result = _fit_gee(sub, feats)
        except Exception as exc:
            print(f"  Module {held_out}: fit failed ({exc})")
            continue
        params = result.params.drop("const", errors="ignore")
        conf = result.conf_int().drop("const", errors="ignore")
        for feat in HEADLINE_FEATS:
            if feat in params.index:
                coef = float(params[feat])
                ci_low = float(conf.loc[feat, 0])
                ci_high = float(conf.loc[feat, 1])
                rows.append({
                    "held_out_module": int(held_out),
                    "feature": feat,
                    "coef": coef,
                    "OR": float(np.exp(coef)),
                    "OR_CI_low": float(np.exp(ci_low)),
                    "OR_CI_high": float(np.exp(ci_high)),
                    "p_value": float(result.pvalues.get(feat, np.nan)),
                    "n_used": int(len(sub)),
                    "n_clusters": int(sub["module_id"].nunique()),
                })

    out = pd.DataFrame(rows)
    save_csv(out, "sensitivity_lomo")

    print("\n--- LOMO OR range per headline feature ---")
    for feat in HEADLINE_FEATS:
        block = out[out["feature"] == feat]
        if block.empty:
            continue
        or_min = block["OR"].min()
        or_max = block["OR"].max()
        p_max = block["p_value"].max()
        ci_excludes_null = (
            ((block["OR_CI_high"] < 1) | (block["OR_CI_low"] > 1)).all()
        )
        marker = " *" if ci_excludes_null else ""
        print(
            f"  {feat:35s}: OR range [{or_min:.3f}, {or_max:.3f}], "
            f"max p = {p_max:.4f}{marker}"
        )


if __name__ == "__main__":
    run()
