"""Is the browsing->completion association an artefact of the outcome definition?

Completion is written when a participant reaches the final content page, so
`n_distinct_pages` partly contains the outcome. This rebuilds the browsing
exposure over page subsets that CANNOT contain the completion trigger and
re-tests the association.
"""
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
FEAT_DIR = ROOT / "output" / "features"
from src.dataset import parse_user_activity

OBS = ["module_id", "user_id", "cohort_id"]

d = pd.read_csv(FEAT_DIR / "user_level_features.csv", low_memory=False)
print(f"starters: {len(d)}   completers: {d['finished'].notna().sum()}")

_, _, pv_rows = parse_user_activity(str(ROOT / "data"), exclude_demo=False)
pv = pd.DataFrame(pv_rows)
pv["url"] = pv["url"].astype(str)
for k in OBS:
    pv[k] = pv[k].astype(str)
    d[k] = d[k].astype(str)

pv = pv.merge(d[OBS + ["dropout_label"]], on=OBS, how="inner")
print(f"page-visit rows matched to starters: {len(pv):,}")

pv["session"] = pv["url"].str.extract(r"/session[-/_]?(\d+)", flags=re.I).astype(float)
pv["is_finished_page"] = pv["url"].str.contains(r"/finished\b|/session-\d+/finished", case=False)
pv["is_module"] = pv["url"].str.contains("/modules/", case=False)

# final session per programme variant
maxsess = pv.groupby("module_id")["session"].max().rename("max_session")
pv = pv.merge(maxsess, on="module_id", how="left")
print("\nmax session by module:", dict(maxsess.astype('Int64')))
print(f"'finished' pages: {pv['is_finished_page'].sum():,} rows, "
      f"{pv.loc[pv['is_finished_page'],'url'].nunique()} distinct URLs")


def distinct(mask, name):
    s = (pv[mask].groupby(OBS)["url"].nunique().rename(name))
    return s


variants = {
    "all pages (as published)":        pv["url"].notna(),
    "excl. every /finished page":      ~pv["is_finished_page"],
    "excl. the entire final session":  pv["session"] < pv["max_session"],
    "sessions 1-2 only":               pv["session"] <= 2,
    "session 1 only":                  pv["session"] == 1,
}

feat = d[OBS + ["dropout_label", "course_name", "n_distinct_pages"]].copy()
for name, mask in variants.items():
    feat = feat.merge(distinct(mask, name), on=OBS, how="left")
feat[list(variants)] = feat[list(variants)].fillna(0)

print("\n" + "=" * 92)
print("MEDIAN DISTINCT PAGES, and association with dropout")
print("=" * 92)
print(f"{'exposure':<32}{'compl':>8}{'drop':>8}{'rank-biserial r':>18}{'  logistic OR (course-adj, per SD)':>34}")
print("-" * 92)

comp = feat["dropout_label"] == 0
for name in variants:
    x = feat[name]
    a, b = x[~comp], x[comp]                      # dropouts, completers
    U = mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * U.statistic / (len(a) * len(b))

    z = (x - x.mean()) / x.std()
    X = pd.concat([z.rename("exp"),
                   pd.get_dummies(feat["course_name"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    m = sm.Logit(feat["dropout_label"].astype(float), X.astype(float)).fit(disp=0)
    orr = np.exp(m.params["exp"])
    lo, hi = np.exp(m.conf_int().loc["exp"])
    star = "" if m.pvalues["exp"] < .05 else "  NULL"
    print(f"{name:<32}{b.median():>8.0f}{a.median():>8.0f}{r:>18.3f}"
          f"{f'  {orr:.3f} [{lo:.3f}, {hi:.3f}]':>26}{star}")

print()
print("=" * 92)
print("HOW MUCH OF THE OUTCOME IS IN THE EXPOSURE?")
print("=" * 92)
fin = pv[pv["is_finished_page"]].groupby(OBS)["url"].nunique().rename("n_finished")
chk = d[OBS + ["dropout_label"]].merge(fin, on=OBS, how="left").fillna({"n_finished": 0})
print(f"  completers who visited >=1 '/finished' page: "
      f"{(chk.loc[chk.dropout_label==0,'n_finished']>0).mean()*100:.1f}%")
print(f"  dropouts   who visited >=1 '/finished' page: "
      f"{(chk.loc[chk.dropout_label==1,'n_finished']>0).mean()*100:.1f}%")

last = pv[pv["session"] == pv["max_session"]].groupby(OBS)["url"].nunique().rename("n_final_sess")
chk2 = d[OBS + ["dropout_label"]].merge(last, on=OBS, how="left").fillna({"n_final_sess": 0})
print(f"  completers who visited >=1 final-session page: "
      f"{(chk2.loc[chk2.dropout_label==0,'n_final_sess']>0).mean()*100:.1f}%")
print(f"  dropouts   who visited >=1 final-session page: "
      f"{(chk2.loc[chk2.dropout_label==1,'n_final_sess']>0).mean()*100:.1f}%")

feat.to_csv(FEAT_DIR / "browsing_subsets.csv", index=False)
print("\nwrote output/features/browsing_subsets.csv")
