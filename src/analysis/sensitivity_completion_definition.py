"""Completion definition check: platform timestamp vs a >=50%-of-sessions rule.

Standalone check (NOT part of run_all). Two questions raised in coauthor
review:

  1. What does the platform completion timestamp (`finished`) coincide with?
     We cannot read the platform trigger from the export, but we can test it
     empirically against the page-visit records (which hold one row per page
     with a cumulative hit count and the LATEST visit timestamp).
  2. Would the headline results change under the session-based definition
     used in the HOPE feasibility papers (completer = visited at least half of
     the scheduled sessions: 3/6 or 4/8)?

Session numbers are parsed from content URLs of the form
``/modules/session-N/...``; session 0 is the introductory session and is not
counted. The session-based outcome is built from page records, so it inherits
their incompleteness (157 eligible enrolments have no page records at all).

Usage:
    python -m src.analysis.sensitivity_completion_definition

Inputs:  output/features/user_level_features.csv, data/csv/page_visits.csv,
         output/features/activity_level_features.csv, data/csv/*.csv
Outputs: output/analysis/tables/sensitivity_completion_trigger.csv
         output/analysis/tables/sensitivity_completion_crosstab.csv
         output/analysis/tables/sensitivity_completion_day7_models.csv
         output/analysis/tables/sensitivity_completion_bivariate.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from config import TABLE_DIR, load_data, save_csv  # noqa: E402
from sensitivity import build_prospective_features  # noqa: E402
from src.utils import configure_stdout_utf8, parse_mixed_datetime  # noqa: E402

OBS = ["module_id", "user_id", "cohort_id"]
PREDICTORS = [
    "n_activities_first_7d",
    "wrote_in_first_week",
    "n_words_first_7d",
    "mean_sentiment_first_7d",
    "received_comment_first_7d",
    "posted_in_first_week",
]


def _fit(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    X = pd.concat(
        [df[PREDICTORS], pd.get_dummies(df["course_name"], drop_first=True, dtype=float)],
        axis=1,
    )
    X = sm.add_constant(X.astype(float))
    m = sm.Logit(df[outcome].astype(float), X).fit(disp=0)
    out = pd.DataFrame({
        "OR": np.exp(m.params),
        "CI_low": np.exp(m.conf_int()[0]),
        "CI_high": np.exp(m.conf_int()[1]),
        "p_value": m.pvalues,
    }).loc[PREDICTORS]
    out["n"] = len(df)
    out["events"] = int(df[outcome].sum())
    return out


def _or_ci(ct: pd.DataFrame) -> tuple[float, float, float]:
    """Haldane-Anscombe corrected odds ratio for a 2x2 table (rows: exposure)."""
    a, b = ct.iloc[1, 0] + 0.5, ct.iloc[1, 1] + 0.5
    c, d = ct.iloc[0, 0] + 0.5, ct.iloc[0, 1] + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, orr * np.exp(-1.96 * se), orr * np.exp(1.96 * se)


def main() -> None:
    configure_stdout_utf8()
    df, writers, groups = load_data()
    df_raw = df.copy()  # original key dtypes, needed by build_prospective_features
    for k in OBS:
        df[k] = df[k].astype(str)

    pv = pd.read_csv(ROOT / "data" / "csv" / "page_visits.csv", low_memory=False)
    for k in OBS:
        pv[k] = pv[k].astype(str)
    pv["url"] = pv["url"].astype(str)
    pv = pv.merge(df[OBS + ["dropout_label"]], on=OBS, how="inner")
    pv["latest"] = parse_mixed_datetime(pv["latest"])
    pv["session"] = pv["url"].str.extract(r"^/modules/session-(\d+)/", flags=re.I)[0].astype(float)
    pv["is_finished_page"] = pv["url"].str.contains(r"/finished\b", case=False)
    max_sess = pv.groupby("module_id")["session"].max()
    pv["max_session"] = pv["module_id"].map(max_sess)

    # ── 1. What does the completion timestamp coincide with? ──────────
    print("=" * 70)
    print("1. Completion timestamp vs page-visit records (completers only)")
    print("=" * 70)
    comp = df[df["dropout_label"] == 0].copy()
    noncomp = df[df["dropout_label"] == 1].copy()
    checks = {
        "any page": pv["url"].notna(),
        "any /finished page": pv["is_finished_page"],
        "any final-session page": pv["session"] == pv["max_session"],
        "final-session /finished page": pv["is_finished_page"] & (pv["session"] == pv["max_session"]),
    }
    rows = []
    for name, mask in checks.items():
        latest = pv[mask].groupby(OBS)["latest"].max().rename("t")
        x = comp.merge(latest, on=OBS, how="left")
        dt = (x["finished"] - x["t"]).dt.total_seconds()
        y = noncomp.merge(latest, on=OBS, how="left")
        rows.append({
            "page_set": name,
            "completers_with_visit": int(x["t"].notna().sum()),
            "completers_total": len(x),
            "abs_diff_le_60s": int((dt.abs() <= 60).sum()),
            "abs_diff_le_1h": int((dt.abs() <= 3600).sum()),
            "finished_gt_1d_before_latest_visit": int((dt < -86400).sum()),
            "finished_gt_1d_after_latest_visit": int((dt > 86400).sum()),
            "noncompleters_with_visit": int(y["t"].notna().sum()),
            "noncompleters_total": len(y),
        })
    trig = pd.DataFrame(rows)
    print(trig.to_string(index=False))
    save_csv(trig, "sensitivity_completion_trigger")

    # Is the timestamp administrative (cohort-level)?
    comp["fin_date"] = comp["finished"].dt.date
    modal_share = comp.groupby("cohort_id")["fin_date"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    sizes = comp.groupby("cohort_id").size()
    print(f"\nCohorts (>=5 completers) where >=50% of completers share one finished date: "
          f"{int(((modal_share >= 0.5) & (sizes >= 5)).sum())} of {int((sizes >= 5).sum())}")
    d = (comp["finished"] - comp["started"]).dt.days
    print(f"Completers finished within 1 day of start: {int((d < 1).sum())}; within 7 days: {int((d < 7).sum())}")

    # ── 2. Session-based outcome ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. >=50% of sessions visited (3/6 or 4/8; session 0 = intro, not counted)")
    print("=" * 70)
    sess = (pv[pv["session"] >= 1].groupby(OBS)["session"].nunique().rename("n_sessions_visited"))
    g = df.merge(sess, on=OBS, how="left")
    g["n_sessions_visited"] = g["n_sessions_visited"].fillna(0)
    g["scheduled_sessions"] = g["module_id"].map(max_sess)
    g["half_sessions"] = (g["n_sessions_visited"] >= np.ceil(g["scheduled_sessions"] / 2)).astype(int)
    g["dropout_sessions"] = 1 - g["half_sessions"]
    ct = pd.crosstab(g["half_sessions"], g["dropout_label"], margins=True)
    ct.index = ["<50% sessions", ">=50% sessions", "All"]
    ct.columns = ["Timestamp completer", "Timestamp non-completer", "All"]
    print(ct.to_string())
    agree = ((g["half_sessions"] == 1) == (g["dropout_label"] == 0)).mean() * 100
    print(f"Agreement between the two definitions: {agree:.1f}%")
    print(f"Session-based completion rate: {g['half_sessions'].mean()*100:.1f}% "
          f"(timestamp-based: {(1 - g['dropout_label']).mean()*100:.1f}%)")
    print("Sessions visited by timestamp status:")
    print(pd.crosstab(g["n_sessions_visited"].astype(int), g["dropout_label"]).to_string())
    save_csv(ct, "sensitivity_completion_crosstab")

    # ── 3. Headline bivariate comparisons under both definitions ──────
    print("\n" + "=" * 70)
    print("3. Writer / forum-poster completion under both definitions")
    print("=" * 70)
    brows = []
    for exp_name, exp in [("writer", g["total_activities_submitted"] > 0),
                          ("forum poster", g["total_discussion_replies"] > 0),
                          ("wrote in first week", g["activities_in_first_7d"] > 0)]:
        for out_name, out in [("timestamp", 1 - g["dropout_label"]), ("sessions>=50%", g["half_sessions"])]:
            t = pd.crosstab(exp.astype(int), out.astype(int))
            orr, lo, hi = _or_ci(t.iloc[:, ::-1])  # columns -> [completer, non-completer]
            brows.append({
                "exposure": exp_name, "outcome": out_name,
                "completion_exposed_pct": round(out[exp].mean() * 100, 1),
                "completion_unexposed_pct": round(out[~exp].mean() * 100, 1),
                "OR_completion": round(orr, 2), "CI_low": round(lo, 2), "CI_high": round(hi, 2),
            })
    biv = pd.DataFrame(brows)
    print(biv.to_string(index=False))
    save_csv(biv, "sensitivity_completion_bivariate")

    # ── 4. First-week model under both definitions ────────────────────
    print("\n" + "=" * 70)
    print("4. First-week logistic model under both outcome definitions")
    print("=" * 70)
    prosp = build_prospective_features(df_raw)
    for k in OBS:
        prosp[k] = prosp[k].astype(str)
    prosp = prosp.merge(g[OBS + ["dropout_sessions"]], on=OBS, how="left")
    reg = prosp.dropna(subset=PREDICTORS + ["dropout_label", "course_name"])
    res = []
    for label, outcome in [("timestamp", "dropout_label"), ("sessions>=50%", "dropout_sessions")]:
        r = _fit(reg, outcome)
        r.insert(0, "outcome", label)
        r.insert(1, "predictor", r.index)
        res.append(r)
        print(f"\n  outcome = non-completion by {label} (n={len(reg)}, events={int(reg[outcome].sum())})")
        for feat, row in r.iterrows():
            sig = " *" if row["p_value"] < 0.05 else ""
            print(f"    {feat:28s} OR={row['OR']:.3f} [{row['CI_low']:.3f}, {row['CI_high']:.3f}] p={row['p_value']:.4f}{sig}")
    save_csv(pd.concat(res).reset_index(drop=True), "sensitivity_completion_day7_models")
    print(f"\nTables written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
