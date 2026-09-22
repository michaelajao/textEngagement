"""How well do inferred SWEMWBS occasions (first / intermediate / last) match sessions?

Standalone check (NOT part of run_all). The raw export carries no session tag
on questionnaire results (`questionnaireResults` entries hold only userId,
cohortId, format, started, finished and the two scores), so the feature
layer labels occasions by order. Two independent pieces of evidence can test
that labelling:

  1. Elapsed time from enrolment start to each occasion, compared with the
     weekly release schedule (session k opens in week k). A first occasion in
     week 1 is consistent with the Session 1 survey; a last occasion in or
     after the final week is consistent with the final-session survey.
  2. The final-session page "test-picture-of-your-wellbeing" appears in the
     page-visit records for six of the eight programme variants with its
     latest visit timestamp. A survey `finished` time within one hour of that
     timestamp identifies that survey as the final-session one.

Usage:
    python -m src.analysis.sensitivity_swemwbs_timing

Inputs:  data/csv/swemwbs.csv, data/csv/page_visits.csv,
         output/features/user_level_features.csv
Outputs: output/analysis/tables/sensitivity_swemwbs_timing.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from config import TABLE_DIR, load_data, save_csv  # noqa: E402
from src.utils import configure_stdout_utf8, parse_mixed_datetime  # noqa: E402

OBS = ["module_id", "user_id", "cohort_id"]
MATCH_TOL_S = 3600


def occasions(sw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature-layer occasion rules: one per calendar day, first entry kept."""
    s = sw.copy()
    s["ts"] = s["finished"].fillna(s["started"])
    s = s.dropna(subset=["ts"]).sort_values("ts")
    s["_day"] = s["ts"].dt.normalize()
    occ = s.drop_duplicates(subset=OBS + ["_day"], keep="first").copy()
    occ["k"] = occ.groupby(OBS).cumcount()
    occ["n"] = occ.groupby(OBS)["ts"].transform("size")
    occ["label"] = "other"
    occ.loc[occ["k"] == 0, "label"] = "first"
    occ.loc[(occ["k"] == occ["n"] - 1) & (occ["n"] >= 2), "label"] = "last"
    # intermediate: nearest the temporal midpoint among interior occasions (n>=3)
    for keys, grp in occ[occ["n"] >= 3].groupby(OBS):
        t0, t1 = grp["ts"].iloc[0], grp["ts"].iloc[-1]
        mid = t0 + (t1 - t0) / 2
        interior = grp.iloc[1:-1]
        idx = (interior["ts"] - mid).abs().idxmin()
        occ.loc[idx, "label"] = "intermediate"
    return occ


def main() -> None:
    configure_stdout_utf8()
    df, _, _ = load_data()
    for k in OBS:
        df[k] = df[k].astype(str)
    sw = pd.read_csv(ROOT / "data" / "csv" / "swemwbs.csv", low_memory=False)
    for k in OBS:
        sw[k] = sw[k].astype(str)
    sw["started"] = parse_mixed_datetime(sw["started"])
    sw["finished"] = parse_mixed_datetime(sw["finished"])
    sw = sw.merge(df[OBS + ["started", "dropout_label", "module_name", "duration_days"]]
                  .rename(columns={"started": "enrol_start"}), on=OBS, how="inner")
    occ = occasions(sw)
    occ["days_since_start"] = (occ["ts"] - occ["enrol_start"]).dt.total_seconds() / 86400

    pv = pd.read_csv(ROOT / "data" / "csv" / "page_visits.csv", low_memory=False)
    for k in OBS:
        pv[k] = pv[k].astype(str)
    pv["url"] = pv["url"].astype(str)
    pv["latest"] = parse_mixed_datetime(pv["latest"])
    pv["session"] = pv["url"].str.extract(r"^/modules/session-(\d+)/", flags=re.I)[0].astype(float)
    n_sess = pv.groupby("module_id")["session"].max()
    occ["scheduled_sessions"] = occ["module_id"].map(n_sess)
    occ["final_week_start"] = (occ["scheduled_sessions"] - 1) * 7

    print("=" * 70)
    print("1. Elapsed days from enrolment start to each inferred occasion")
    print("=" * 70)
    rows = []
    for lab in ["first", "intermediate", "last"]:
        x = occ[occ["label"] == lab]
        d = x["days_since_start"]
        rows.append({
            "occasion": lab, "n": len(x),
            "median_days": d.median(), "q1": d.quantile(0.25), "q3": d.quantile(0.75),
            "pct_within_week1": (d < 7).mean() * 100,
            "pct_before_final_week": (d < x["final_week_start"]).mean() * 100,
            "pct_in_or_after_final_week": (d >= x["final_week_start"]).mean() * 100,
        })
        print(f"  {lab:12s} n={len(x):4d}  median {d.median():6.1f} d (IQR {d.quantile(.25):.1f}–{d.quantile(.75):.1f}); "
              f"in week 1: {(d < 7).mean()*100:5.1f}%; in/after final week: {(d >= x['final_week_start']).mean()*100:5.1f}%")
    timing = pd.DataFrame(rows)

    # last occasions among non-completers and among people who completed early
    last = occ[occ["label"] == "last"]
    nc = last[last["dropout_label"] == 1]
    print(f"\n  'last' occasions from non-completers: {len(nc)}; of these before the final week: "
          f"{int((nc['days_since_start'] < nc['final_week_start']).sum())}")
    print("  Caution: sessions are scheduled weekly but the records show some enrolments completing "
          "within days of starting, so calendar weeks are only a rough guide to session position.")

    # Position of the 'last' occasion relative to the enrolment's own completion timestamp
    lc = last[last["dropout_label"] == 0].merge(
        df[OBS + ["finished"]].rename(columns={"finished": "enrol_finished"}), on=OBS, how="left")
    rel = (lc["enrol_finished"] - lc["ts"]).dt.total_seconds() / 86400
    print(f"\n  'last' occasions from completers: {len(lc)}; taken within 1 day of the completion "
          f"timestamp: {int((rel.abs() <= 1).sum())} ({(rel.abs() <= 1).mean()*100:.1f}%); "
          f"more than 7 days before completion: {int((rel > 7).sum())} ({(rel > 7).mean()*100:.1f}%)")

    print("\n" + "=" * 70)
    print("2. Match to the final-session wellbeing page (latest visit within 1 h)")
    print("=" * 70)
    wp = pv[pv["url"].str.contains("test-picture-of-your-wellbeing", case=False)]
    print(f"  page rows: {len(wp)}; sessions in which it appears: {sorted(wp['session'].dropna().unique().astype(int))}; "
          f"variants: {wp['module_id'].nunique()} of 8")
    m = occ.merge(wp[OBS + ["latest", "session"]], on=OBS, how="left")
    m["dt"] = (m["ts"] - m["latest"]).dt.total_seconds().abs()
    m["matched"] = m["dt"] <= MATCH_TOL_S
    matched = m[m["matched"]]
    by_label = matched["label"].value_counts()
    print(f"  occasions matched to the final-session page: {len(matched)}")
    print("  inferred label of matched occasions:\n" + by_label.to_string())
    # for enrolments where a match exists, was the matched occasion the last one?
    share_last = (matched["label"] == "last").mean() * 100 if len(matched) else np.nan
    print(f"  share of matched occasions labelled 'last': {share_last:.1f}%")
    # first occasions matched to the final page = mislabelled baseline
    print(f"  matched occasions labelled 'first' (single-survey enrolments answering only at the end): "
          f"{int(((matched['label'] == 'first')).sum())}")
    timing["matched_to_final_page"] = timing["occasion"].map(by_label).fillna(0).astype(int)

    # Broader check: any final-session page (all eight variants)
    fs = pv[pv["session"] == pv["module_id"].map(n_sess)]
    m2 = occ.merge(fs[OBS + ["latest"]], on=OBS, how="left")
    m2["dt"] = (m2["ts"] - m2["latest"]).dt.total_seconds().abs()
    hit = m2[m2["dt"] <= MATCH_TOL_S].drop_duplicates(subset=OBS + ["ts"])
    print(f"\n  occasions within 1 h of the latest visit to ANY final-session page: {len(hit)}; "
          f"labelled 'last': {(hit['label'] == 'last').mean()*100:.1f}%; "
          f"share of all 'last' occasions so matched: {(hit['label'] == 'last').sum() / len(last)*100:.1f}%")
    save_csv(timing, "sensitivity_swemwbs_timing")
    print(f"\nTables written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
