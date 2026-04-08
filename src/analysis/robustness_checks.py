"""
Additional robustness checks addressing three reviewer concerns:

1. E-values for key odds ratios (unmeasured confounding sensitivity)
2. Characterisation of the 1,001 excluded records (no start date)
3. Cluster bootstrap for GEE (small number of module clusters)

Outputs
-------
Tables  evalue_sensitivity.csv, excluded_records_comparison.csv,
        gee_bootstrap_ci.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

from src.analysis import (
    CSV_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)


# ── 1. E-values ─────────────────────────────────────────────────────────────


def _evalue(rr: float) -> float:
    """E-value for a risk ratio >= 1 (VanderWeele & Ding 2017)."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def _or_to_rr(or_val: float) -> float:
    """Approximate RR from OR using the square-root transformation."""
    return np.sqrt(or_val) if or_val >= 1 else 1 / np.sqrt(1 / or_val)


def run_evalues() -> pd.DataFrame:
    """Compute E-values for the key findings reported in the paper."""
    findings = [
        ("Writer vs non-writer (chi-square)", 11.71, 9.59, 14.29),
        ("Activities in first 7 days (RQ1 adj.)", 1 / 0.57, 1 / 0.68, 1 / 0.49),
        ("Facilitator comment receipt (unadj.)", 6.87, 5.02, 9.41),
        ("Forum per reply (RQ3 adj.)", 1 / 0.78, 1 / 0.89, 1 / 0.69),
        ("GEE: total activities per SD", 1 / 0.01, 1 / 0.02, 1 / 0.001),
    ]

    rows = []
    for name, or_val, ci_low, ci_high in findings:
        rr_point = _or_to_rr(or_val)
        rr_ci = _or_to_rr(ci_low)  # CI bound closest to null
        rows.append({
            "finding": name,
            "OR": round(or_val, 2),
            "CI_closest_to_null": round(ci_low, 2),
            "evalue_point": round(_evalue(rr_point), 2),
            "evalue_ci": round(_evalue(rr_ci), 2),
        })

    df = pd.DataFrame(rows)
    print("\n  E-values for key findings:")
    for _, r in df.iterrows():
        print(f"    {r['finding']:45s}  E = {r['evalue_point']:.1f}  "
              f"(CI: {r['evalue_ci']:.1f})")
    return df


# ── 2. Excluded records characterisation ─────────────────────────────────────


def run_excluded_characterisation() -> pd.DataFrame:
    """Compare the 1,001 excluded records with the 2,515 included starters."""
    users = pd.read_csv(CSV_DIR / "users.csv")
    activities = pd.read_csv(CSV_DIR / "activities.csv")
    discussions = pd.read_csv(CSV_DIR / "discussions.csv")

    included = users[users["started"].notna()].copy()
    excluded = users[users["started"].isna()].copy()
    print(f"\n  Included: {len(included)},  Excluded: {len(excluded)}")

    # Count activities and forum posts per user
    act_counts = activities.groupby("user_id").size()
    disc_counts = discussions.groupby("user_id").size()

    for df in (included, excluded):
        df["n_activities"] = df["user_id"].map(act_counts).fillna(0).astype(int)
        df["n_forum_posts"] = df["user_id"].map(disc_counts).fillna(0).astype(int)
        df["has_any_data"] = (
            (df["n_activities"] > 0)
            | (df["n_forum_posts"] > 0)
            | (df["n_page_visits"] > 0)
        )

    # Comparison metrics
    rows = []

    def _add(name, inc_val, exc_val, p=np.nan):
        rows.append({
            "metric": name,
            "included": round(inc_val, 3),
            "excluded": round(exc_val, 3),
            "p_value": p,
        })

    _add("n", len(included), len(excluded))
    _add("any_behavioural_data_pct",
         included["has_any_data"].mean() * 100,
         excluded["has_any_data"].mean() * 100)

    # Continuous comparisons with Mann-Whitney
    for col, label in [
        ("n_logins", "mean_logins"),
        ("n_page_visits", "mean_page_visits"),
        ("n_activities", "mean_activities"),
        ("n_forum_posts", "mean_forum_posts"),
    ]:
        inc = included[col].fillna(0)
        exc = excluded[col].fillna(0)
        _, p = mannwhitneyu(inc, exc, alternative="two-sided")
        _add(label, inc.mean(), exc.mean(), p)
        _add(f"pct_any_{col}",
             (inc > 0).mean() * 100,
             (exc > 0).mean() * 100)

    # Course distribution chi-square
    ct = pd.crosstab(
        users["started"].notna().map({True: "included", False: "excluded"}),
        users["course_name"],
    )
    chi2, p_course, _, _ = chi2_contingency(ct)
    _add("course_distribution_chi2", chi2, np.nan, p_course)

    df = pd.DataFrame(rows)

    n_active = int(excluded["has_any_data"].sum())
    print(f"  Excluded with any data: {n_active}/{len(excluded)} "
          f"({n_active / len(excluded) * 100:.1f}%)")
    print(f"  Course distribution chi2 = {chi2:.1f}, p = {p_course:.4f}")
    return df


# ── 3. Cluster bootstrap for GEE ────────────────────────────────────────────

GEE_FEATURES = [
    "total_activities_submitted",
    "total_words_written",
    "avg_vocab_richness",
    "avg_sentiment",
    "avg_future_orientation",
    "activity_type_entropy",
    "word_count_trend",
    "pct_activities_with_comments",
    "total_discussion_replies",
    "activities_in_first_7d",
]


def _fit_gee(df: pd.DataFrame, features: list[str], group_col: str):
    """Fit a single GEE and return coefficient array (excl. intercept)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GEE(
            df["dropout_label"],
            sm.add_constant(df[features].astype(float)),
            groups=df[group_col],
            family=Binomial(),
            cov_struct=Exchangeable(),
        ).fit(maxiter=200)
    return model.params.iloc[1:].values


def run_cluster_bootstrap(
    user: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap for GEE coefficients (Pan & Wall 2002)."""
    df = user[user["total_activities_submitted"] > 0].copy()
    features = [c for c in GEE_FEATURES if c in df.columns]
    df[features] = df[features].fillna(0)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    df = df.sort_values("module_id")

    modules = df["module_id"].unique()
    n_modules = len(modules)
    rng = np.random.default_rng(seed)

    # Original fit
    orig_coefs = _fit_gee(df, features, "module_id")

    # Original sandwich SEs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_model = GEE(
            df["dropout_label"],
            sm.add_constant(df[features].astype(float)),
            groups=df["module_id"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        ).fit(maxiter=200)
    sandwich_se = orig_model.bse.iloc[1:].values

    # Bootstrap
    print(f"\n  Running {n_boot} cluster-bootstrap iterations "
          f"({n_modules} modules) ...")
    boot_coefs = []
    for i in range(n_boot):
        boot_modules = rng.choice(modules, size=n_modules, replace=True)
        chunks = []
        for j, m in enumerate(boot_modules):
            chunk = df[df["module_id"] == m].copy()
            chunk["boot_group"] = j
            chunks.append(chunk)
        boot_df = pd.concat(chunks, ignore_index=True)

        try:
            coefs = _fit_gee(boot_df, features, "boot_group")
            if np.all(np.isfinite(coefs)):
                boot_coefs.append(coefs)
        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_boot} done")

    boot_matrix = np.array(boot_coefs)
    print(f"  Converged: {len(boot_coefs)}/{n_boot}")

    ci_low = np.percentile(boot_matrix, 2.5, axis=0)
    ci_high = np.percentile(boot_matrix, 97.5, axis=0)
    boot_se = np.std(boot_matrix, axis=0)

    results = pd.DataFrame({
        "feature": features,
        "coef": orig_coefs,
        "OR": np.exp(orig_coefs),
        "sandwich_se": sandwich_se,
        "bootstrap_se": boot_se,
        "bootstrap_OR_CI_low": np.exp(ci_low),
        "bootstrap_OR_CI_high": np.exp(ci_high),
        "boot_ci_excludes_null": ~((ci_low < 0) & (ci_high > 0)),
        "n_converged": len(boot_coefs),
    })

    print("\n  Sandwich SE vs Bootstrap SE:")
    for _, r in results.iterrows():
        ratio = r["bootstrap_se"] / r["sandwich_se"] if r["sandwich_se"] > 0 else np.nan
        tag = "WIDER" if ratio > 1.1 else ("narrower" if ratio < 0.9 else "similar")
        print(f"    {r['feature']:35s}  sand={r['sandwich_se']:.3f}  "
              f"boot={r['bootstrap_se']:.3f}  ratio={ratio:.2f} ({tag})")

    return results


# ── entry point ─────────────────────────────────────────────────────────────


def run(data: dict[str, pd.DataFrame]) -> None:
    ensure_output_dirs()

    print("=" * 60)
    print("1. E-VALUE SENSITIVITY ANALYSIS")
    print("=" * 60)
    evals = run_evalues()
    evals.to_csv(TABLES_DIR / "evalue_sensitivity.csv", index=False)

    print("\n" + "=" * 60)
    print("2. EXCLUDED RECORDS CHARACTERISATION")
    print("=" * 60)
    excluded = run_excluded_characterisation()
    excluded.to_csv(TABLES_DIR / "excluded_records_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("3. CLUSTER BOOTSTRAP FOR GEE")
    print("=" * 60)
    bootstrap = run_cluster_bootstrap(data["users"], n_boot=2000)
    bootstrap.to_csv(TABLES_DIR / "gee_bootstrap_ci.csv", index=False)

    print("\nRobustness checks complete.\n")


if __name__ == "__main__":
    from src.analysis import load_analytical_tables

    data = load_analytical_tables()
    run(data)
