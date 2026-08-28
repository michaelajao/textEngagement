import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
FEAT_DIR = ROOT / "output" / "features"
"""How sensitive is the H4 forum finding to specification of the engagement controls?

Table 7 in the manuscript reports discussion replies OR = 0.73 [0.64, 0.84], p < .001,
and concludes forum participation is associated with completion "rather than merely
correlating with general engagement tendency". That claim rests on the covariates
absorbing general engagement. This grid varies only how the covariates are specified.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

d = pd.read_csv(
    FEAT_DIR / "user_level_features.csv",
    low_memory=False,
)

TARGET = "total_discussion_replies"


def fit(df, xcols, label):
    reg = df[xcols + ["dropout_label", "course_name"]].dropna(subset=xcols)
    X = reg[xcols].copy()
    X = pd.concat([X, pd.get_dummies(reg["course_name"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    m = sm.Logit(reg["dropout_label"].astype(float), X.astype(float)).fit(disp=0)
    ci = m.conf_int()
    o, lo, hi = np.exp(m.params[TARGET]), np.exp(ci.loc[TARGET, 0]), np.exp(ci.loc[TARGET, 1])
    sig = "  <-- NULL" if (lo < 1 < hi) else ""
    print(f"  {label:<52} OR={o:5.3f} [{lo:5.3f}, {hi:5.3f}]  p={m.pvalues[TARGET]:.4f}{sig}")


print("=" * 96)
print("H4 SENSITIVITY:  outcome = dropout;  target = total_discussion_replies (per reply)")
print("All models adjust for course. n = 2270 starters unless noted.")
print("=" * 96)

base = ["total_activities_submitted", "total_comments_received", "n_logins"]

print("\n-- As published --")
fit(d, [TARGET] + base, "raw logins  (manuscript Table 7)")

print("\n-- Same model, logins given a standard skew treatment --")
for name, tf in [("log1p(logins)", lambda s: np.log1p(s)),
                 ("sqrt(logins)", np.sqrt),
                 ("logins winsorised at p99 (60)", lambda s: s.clip(upper=s.quantile(.99))),
                 ("logins winsorised at p95 (21)", lambda s: s.clip(upper=s.quantile(.95)))]:
    dd = d.copy()
    dd["n_logins"] = tf(dd["n_logins"])
    fit(dd, [TARGET] + base, name)

print("\n-- Drop the single 2051-login outlier, keep everything else as published --")
fit(d[d["n_logins"] < 1000], [TARGET] + base, "raw logins, n = 2269")

print("\n-- Add browsing depth as an additional general-engagement control --")
dd = d.copy()
fit(dd, [TARGET] + base + ["n_distinct_pages"], "raw logins + distinct pages")
dd["n_logins"] = np.log1p(dd["n_logins"])
dd["n_distinct_pages"] = np.log1p(dd["n_distinct_pages"])
fit(dd, [TARGET] + base + ["n_distinct_pages"], "log logins + log distinct pages")

print("\n-- Target itself log-transformed (replies are skewed too) --")
dd = d.copy()
dd[TARGET] = np.log1p(dd[TARGET])
fit(dd, [TARGET] + base, "log1p(replies), raw logins")
dd["n_logins"] = np.log1p(dd["n_logins"])
fit(dd, [TARGET] + base, "log1p(replies), log1p(logins)")

print("\n-- Binary exposure: posted at all vs not --")
dd = d.copy()
dd["posted_any"] = (dd["total_discussion_replies"] > 0).astype(float)
TARGET = "posted_any"
fit(dd, ["posted_any"] + base, "posted at all, raw logins")
dd2 = dd.copy()
dd2["n_logins"] = np.log1p(dd2["n_logins"])
fit(dd2, ["posted_any"] + base, "posted at all, log1p(logins)")

print()
print("=" * 96)
print("WHY: how much general-engagement variance does each login spec absorb?")
print("=" * 96)
lg = d["n_logins"]
print(f"  n_logins  skew = {lg.skew():.1f}   median = {lg.median():.0f}   max = {lg.max():.0f}")
print(f"  log1p(n_logins) skew = {np.log1p(lg).skew():.2f}")
print(f"  Spearman(logins, replies)          = {lg.corr(d[ 'total_discussion_replies'], method='spearman'):.3f}")
print(f"  Spearman(logins, distinct pages)   = {lg.corr(d['n_distinct_pages'], method='spearman'):.3f}")
print(f"  Pearson (logins, distinct pages)   = {lg.corr(d['n_distinct_pages']):.3f}")
