"""
Sensitivity analysis: Cluster bootstrap CIs for the GEE coefficients.

The sandwich (robust) standard error estimator used by the main GEE is
asymptotically justified in the number of clusters, not observations. With
only 10 module-level clusters the sandwich SE may be anti-conservative
(Li & Redden 2015 recommend 30+ clusters). This script validates the GEE
findings under a cluster-resampling bootstrap:

  1. Resample the 10 modules with replacement (10 draws per iteration).
  2. Reassemble a bootstrap dataset from the drawn modules, assigning each
     draw a fresh cluster ID so duplicated modules do not collapse.
  3. Refit the GEE on the bootstrap sample.
  4. Record the coefficients. Repeat n_iter times.
  5. Report percentile 95% CIs and sandwich-vs-bootstrap SE ratios.

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/sensitivity_bootstrap.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from config import load_data, save_csv

from gee_robustness import GEE_FEATS


N_ITER = 2000
RANDOM_SEED = 42


def _fit_gee(df: pd.DataFrame, feats: list[str], group_col: str):
    """Fit the same GEE specification as gee_robustness.py on the given frame."""
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(df[feats]),
        columns=feats, index=df.index,
    )
    X = sm.add_constant(X)
    gee = sm.GEE(
        df["dropout_label"], X,
        groups=df[group_col],
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = gee.fit()
    return result


def run(data=None):
    if data is None:
        df, writers, groups = load_data()
    else:
        df, writers, groups = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: Cluster Bootstrap CIs (GEE)")
    print("=" * 60)

    feats = [f for f in GEE_FEATS if f in writers.columns]
    gee_df = writers[["module_id", "dropout_label"] + feats].dropna().copy()
    modules = gee_df["module_id"].unique()
    n_clusters = len(modules)
    print(f"\nResampling {n_clusters} modules with replacement, {N_ITER:,} iterations")
    print(f"Sample: {len(gee_df):,} writers")

    # Pre-split by module for fast resampling
    by_module = {m: gee_df[gee_df["module_id"] == m] for m in modules}

    # Point estimates from the main GEE (for SE ratio reporting)
    main_result = _fit_gee(gee_df, feats, "module_id")
    main_coefs = main_result.params.drop("const", errors="ignore")
    main_bse = main_result.bse.drop("const", errors="ignore")

    # Bootstrap
    rng = np.random.default_rng(RANDOM_SEED)
    coef_samples = {f: [] for f in feats}
    n_converged = 0

    # Scope the warning suppression to the bootstrap loop only — a
    # module-level filterwarnings("ignore") would leak into every later
    # script in run_all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_ITER):
            draws = rng.choice(modules, size=n_clusters, replace=True)
            pieces = []
            for k, mod_id in enumerate(draws):
                piece = by_module[mod_id].copy()
                piece["_boot_cluster"] = k  # fresh cluster id for duplicates
                pieces.append(piece)
            boot_df = pd.concat(pieces, ignore_index=True)

            try:
                result = _fit_gee(boot_df, feats, "_boot_cluster")
            except Exception:
                continue
            if not getattr(result, "converged", True):
                continue

            params = result.params.drop("const", errors="ignore")
            for f in feats:
                if f in params.index:
                    coef_samples[f].append(params[f])
            n_converged += 1

            if (i + 1) % 500 == 0:
                print(f"  iter {i + 1:,}/{N_ITER:,} (converged: {n_converged:,})")

    print(f"  converged: {n_converged:,}/{N_ITER:,}")
    if n_converged < 0.95 * N_ITER:
        print("  WARNING: <95% of bootstrap fits converged; percentile CIs "
              "may be biased by convergence selection.")

    # Compile summary
    rows = []
    for f in feats:
        samples = np.array(coef_samples[f])
        if len(samples) == 0:
            continue
        ci_low_coef = np.percentile(samples, 2.5)
        ci_high_coef = np.percentile(samples, 97.5)
        boot_se = float(np.std(samples, ddof=1))
        sandwich_se = float(main_bse.get(f, np.nan))
        rows.append({
            "feature": f,
            "coef": float(main_coefs.get(f, np.nan)),
            "OR": float(np.exp(main_coefs.get(f, np.nan))),
            "sandwich_se": sandwich_se,
            "bootstrap_se": boot_se,
            "se_ratio_boot_over_sandwich": (
                boot_se / sandwich_se if sandwich_se > 0 else np.nan
            ),
            "bootstrap_OR_CI_low": float(np.exp(ci_low_coef)),
            "bootstrap_OR_CI_high": float(np.exp(ci_high_coef)),
            "boot_ci_excludes_null": bool(
                ci_high_coef < 0 or ci_low_coef > 0
            ),
            "n_converged": n_converged,
        })

    out = pd.DataFrame(rows)
    print("\n--- Bootstrap Results (percentile 95% CI on OR scale) ---")
    for _, r in out.iterrows():
        marker = " *" if r["boot_ci_excludes_null"] else ""
        print(
            f"  {r['feature']:35s}: OR={r['OR']:.3f} "
            f"[{r['bootstrap_OR_CI_low']:.3f}, {r['bootstrap_OR_CI_high']:.3f}] "
            f"SE_boot/SE_sw={r['se_ratio_boot_over_sandwich']:.2f}{marker}"
        )

    save_csv(out, "sensitivity_bootstrap")


if __name__ == "__main__":
    run()
