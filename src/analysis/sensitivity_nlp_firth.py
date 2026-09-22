"""Incremental value of linguistic features: separation-aware refit.

Standalone check (NOT part of run_all). The nested M0 -> M1 -> M2 logistic
models in `sensitivity.py` (volume/timing -> + rule-based linguistic ->
+ transformer-derived) did not converge: the complete-case writer sample has
31 non-completion events and one course group with no events (quasi-complete
separation). Maximum-likelihood likelihood-ratio tests are therefore not
trustworthy. This script answers the same question two ways that do not
depend on ML convergence:

  1. Firth's penalised likelihood (Jeffreys prior) logistic regression with
     penalised likelihood-ratio tests (Heinze & Schemper 2002). Firth
     estimates are finite under separation.
  2. Out-of-sample discrimination: repeated stratified 5-fold cross-validated
     AUC for each nested model, using L2-regularised logistic regression
     (C=1) so every fold fits. Paired differences in fold AUCs are reported.

Both use exactly the 1002-writer complete-case sample and course dummies of
the original comparison.

Usage:
    python -m src.analysis.sensitivity_nlp_firth

Inputs:  output/features/user_level_features.csv
Outputs: output/analysis/tables/sensitivity_nlp_firth.csv
         output/analysis/tables/sensitivity_nlp_cv_auc.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from config import TABLE_DIR, load_data, save_csv  # noqa: E402
from sensitivity import REGEX_FEATS, TRANSFORMER_FEATS, VOLUME_FEATS  # noqa: E402
from src.utils import configure_stdout_utf8  # noqa: E402

N_REPEATS = 20
N_FOLDS = 5
SEED = 42


def firth_logit(X: np.ndarray, y: np.ndarray, free: np.ndarray | None = None,
                max_iter: int = 200, tol: float = 1e-8):
    """Firth-penalised logistic regression via Newton-Raphson.

    ``free`` is a boolean mask of coefficients to estimate; the others are
    held at zero while the Jeffreys penalty is still evaluated on the FULL
    design. This is the constrained fit required for the penalised
    likelihood-ratio test of Heinze & Schemper (2002): the reduced model is
    the full model with the tested coefficients fixed at zero, not a
    separately penalised smaller model.

    Returns (beta, penalised log-likelihood, standard errors).
    """
    n, p = X.shape
    if free is None:
        free = np.ones(p, dtype=bool)
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        info = (X.T * w) @ X
        info_inv = np.linalg.pinv(info)
        # hat-matrix diagonal from the full design
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], info_inv, X * np.sqrt(w)[:, None])
        score = X.T @ (y - mu + h * (0.5 - mu))
        step = np.zeros(p)
        step[free] = np.linalg.pinv(info[np.ix_(free, free)]) @ score[free]
        # step-halving for stability
        for _k in range(20):
            nb = beta + step
            if np.all(np.isfinite(nb)) and np.abs(X @ nb).max() < 30:
                break
            step /= 2
        beta = nb
        if np.abs(step).max() < tol:
            break
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta))
    w = mu * (1 - mu)
    info = (X.T * w) @ X
    sign, logdet = np.linalg.slogdet(info)
    pll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)) + 0.5 * logdet)
    se = np.sqrt(np.diag(np.linalg.pinv(info)))
    return beta, pll, se


def design(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, list[str]]:
    X = pd.concat(
        [df[feats], pd.get_dummies(df["course_name"], drop_first=True, dtype=float)],
        axis=1,
    ).astype(float)
    X.insert(0, "const", 1.0)
    return X.to_numpy(), list(X.columns)


def main() -> None:
    configure_stdout_utf8()
    df, writers, groups = load_data()
    needed = VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS + ["dropout_label", "course_name"]
    sub = writers[needed].dropna().copy()
    y = sub["dropout_label"].to_numpy(dtype=float)
    print(f"Sample: {len(sub):,} complete-case writers, {int(y.sum())} non-completion events")
    ev = sub.groupby("course_name")["dropout_label"].agg(["size", "sum"])
    print("Events by course group:\n" + ev.to_string())

    specs = [
        ("M0 (volume + timing)", VOLUME_FEATS),
        ("M1 (+ rule-based linguistic)", VOLUME_FEATS + REGEX_FEATS),
        ("M2 (+ transformer-derived)", VOLUME_FEATS + REGEX_FEATS + TRANSFORMER_FEATS),
    ]

    # ── 1. Firth penalised likelihood ────────────────────────────────
    print("\n" + "=" * 70)
    print("1. Firth penalised-likelihood nested comparison")
    print("=" * 70)
    # Full (M2) design, standardised so the Jeffreys penalty is scale-free.
    all_feats = specs[2][1]
    X, cols = design(sub, all_feats)
    cont = [i for i, c in enumerate(cols) if c in all_feats]
    X[:, cont] = (X[:, cont] - X[:, cont].mean(axis=0)) / X[:, cont].std(axis=0, ddof=0)
    fits = {}
    for name, feats in specs:
        free = np.array([c not in all_feats or c in feats for c in cols])
        beta, pll, se = firth_logit(X, y, free=free)
        fits[name] = (feats, beta, pll, se)
        print(f"\n{name}: penalised log-lik (full-design penalty, "
              f"{int((~free).sum())} coefficients fixed at 0) = {pll:.3f}")
        for j, c in enumerate(cols):
            if c in feats:
                orr = np.exp(beta[j])
                lo, hi = np.exp(beta[j] - 1.96 * se[j]), np.exp(beta[j] + 1.96 * se[j])
                print(f"    {c:26s} OR per SD = {orr:.3f} [{lo:.3f}, {hi:.3f}]")
    rows = []
    for a, b in [(specs[0][0], specs[1][0]), (specs[1][0], specs[2][0]), (specs[0][0], specs[2][0])]:
        lr = 2 * (fits[b][2] - fits[a][2])
        df_ = len(fits[b][0]) - len(fits[a][0])
        p = float(stats.chi2.sf(lr, df_))
        rows.append({"comparison": f"{b} vs {a}", "pLR_stat": lr, "df": df_, "p_value": p,
                     "n": len(sub), "events": int(y.sum())})
        print(f"\n{b} vs {a}: penalised LR = {lr:.2f} on {df_} df, p = {p:.3f}")
    save_csv(pd.DataFrame(rows), "sensitivity_nlp_firth")

    # ── 2. Repeated cross-validated AUC ──────────────────────────────
    print("\n" + "=" * 70)
    print(f"2. Repeated stratified {N_FOLDS}-fold CV AUC ({N_REPEATS} repeats, L2 logistic, C=1)")
    print("=" * 70)
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    aucs = {name: [] for name, _ in specs}
    designs = {name: design(sub, feats)[0][:, 1:] for name, feats in specs}  # drop const (sklearn adds it)
    for train, test in cv.split(designs[specs[0][0]], y):
        if y[test].sum() == 0:
            continue
        for name, _ in specs:
            X = designs[name]
            sc = StandardScaler().fit(X[train])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[train]), y[train])
            aucs[name].append(roc_auc_score(y[test], clf.predict_proba(sc.transform(X[test]))[:, 1]))
    out = []
    base = np.array(aucs[specs[0][0]])
    for name, _ in specs:
        a = np.array(aucs[name])
        diff = a - base
        out.append({"model": name, "cv_auc_mean": a.mean(), "cv_auc_sd": a.std(ddof=1),
                    "delta_auc_vs_M0_mean": diff.mean(),
                    "delta_auc_vs_M0_ci_low": np.percentile(diff, 2.5),
                    "delta_auc_vs_M0_ci_high": np.percentile(diff, 97.5),
                    "n_folds": len(a)})
        print(f"  {name:32s} AUC = {a.mean():.3f} (SD {a.std(ddof=1):.3f}); "
              f"Δ vs M0 = {diff.mean():+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]")
    save_csv(pd.DataFrame(out), "sensitivity_nlp_cv_auc")
    print("\nNote: the fold-difference percentile interval describes variability across resampled "
          "folds, not a formal confidence interval (folds overlap).")
    print(f"Tables written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
