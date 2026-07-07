"""
Sensitivity analysis: E-values for unmeasured confounding.

Computes E-values following VanderWeele & Ding (2017, Ann Intern Med) for the
three focal associations cited in the Discussion section of the paper:

  1. Writer vs non-writer (chi-square OR from the writer/completer 2x2 table)
  2. Activities in first 7 days (RQ1 adjusted logistic regression)
  3. Total discussion replies (RQ3 adjusted logistic regression)

For the two CONTINUOUS exposures (2 and 3), E-values are reported on the
per-standard-deviation OR scale rather than the per-unit scale: a per-unit
OR is arbitrarily close to 1 when the unit is small relative to the spread
of the exposure, which makes the per-unit E-value uninformatively small by
construction. OR_per_SD = OR_per_unit ** SD (exact rescaling of the log
odds coefficient and its CI).

For common outcomes (dropout prevalence ~31%), odds ratios are converted to
approximate risk ratios via RR = sqrt(OR) before applying the E-value formula:

  E = RR + sqrt(RR * (RR - 1))     for RR >= 1
  RR <- 1/RR; E as above           for RR < 1

The CI bound closest to the null (1.0) is used to compute the E-value for the
confidence interval. This represents the magnitude of unmeasured confounding
(on the risk-ratio scale, for both exposure-confounder and confounder-outcome
associations) required to fully explain away the observed association.

Inputs:  output/analysis/tables/rq1_writer_chi2.csv
         output/analysis/tables/rq1_logistic_regression.csv
         output/analysis/tables/rq3_logistic_regression.csv
Outputs: output/analysis/tables/sensitivity_evalue.csv
"""

import numpy as np
import pandas as pd

from config import TABLE_DIR, save_csv


def _evalue(rr: float) -> float:
    """E-value for a point estimate on the risk-ratio scale."""
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def _or_to_rr(odds_ratio: float) -> float:
    """Approximate RR from OR for common outcomes: RR ~ sqrt(OR)."""
    return np.sqrt(odds_ratio)


def _ci_closest_to_null(or_point: float, ci_low: float, ci_high: float) -> float:
    """Return the bound of the OR CI that sits closer to the null (1.0)."""
    return ci_low if or_point > 1.0 else ci_high


def _writer_or_from_chi2_table() -> tuple[float, float, float]:
    """Recover the 2x2 OR and Haldane-corrected 95% CI for writer vs non-writer."""
    ct = pd.read_csv(TABLE_DIR / "rq1_writer_chi2.csv", index_col=0)
    # Layout: rows = Non-writer / Writer; cols = Completer / Dropout
    a = ct.loc["Writer", "Completer"]
    b = ct.loc["Writer", "Dropout"]
    c = ct.loc["Non-writer", "Completer"]
    d = ct.loc["Non-writer", "Dropout"]
    # OR for completion given writer status (completer in numerator)
    vals = np.array([a, b, c, d], dtype=float) + 0.5  # Haldane correction
    or_point = (vals[0] * vals[3]) / (vals[1] * vals[2])
    se = np.sqrt(sum(1.0 / v for v in vals))
    ci_low = np.exp(np.log(or_point) - 1.96 * se)
    ci_high = np.exp(np.log(or_point) + 1.96 * se)
    return or_point, ci_low, ci_high


def run(data=None):
    if data is None:
        from config import load_data
        data = load_data()
    df_feats, _, _ = data

    print("\n" + "=" * 60)
    print("Sensitivity Analysis: E-values (VanderWeele & Ding 2017)")
    print("=" * 60)

    # SDs used to rescale per-unit ORs to per-SD ORs (computed on the
    # same samples the source regressions used: RQ1 = writers with
    # complete predictors, RQ3 = all starters).
    sd_act7 = float(
        df_feats.loc[
            df_feats["total_activities_submitted"] > 0, "activities_in_first_7d"
        ].std()
    )
    sd_replies = float(df_feats["total_discussion_replies"].std())

    rows = []

    # ── 1. Writer vs non-writer ────────────────────────────────
    or_w, ci_lo_w, ci_hi_w = _writer_or_from_chi2_table()
    rows.append({
        "finding": "Writer vs non-writer (chi-square)",
        "OR": or_w,
        "CI_low": ci_lo_w,
        "CI_high": ci_hi_w,
        "CI_closest_to_null": _ci_closest_to_null(or_w, ci_lo_w, ci_hi_w),
    })

    # ── 2. Activities in first 7 days (RQ1), per-SD scale ──────
    rq1 = pd.read_csv(TABLE_DIR / "rq1_logistic_regression.csv", index_col=0)
    r = rq1.loc["activities_in_first_7d"]
    or_sd = float(r["OR"]) ** sd_act7
    ci_lo_sd = float(r["CI_low"]) ** sd_act7
    ci_hi_sd = float(r["CI_high"]) ** sd_act7
    rows.append({
        "finding": f"Activities in first 7 days (RQ1 adj., per SD={sd_act7:.2f})",
        "OR": or_sd,
        "CI_low": ci_lo_sd,
        "CI_high": ci_hi_sd,
        "CI_closest_to_null": _ci_closest_to_null(or_sd, ci_lo_sd, ci_hi_sd),
    })

    # ── 3. Total discussion replies (RQ3), per-SD scale ────────
    rq3 = pd.read_csv(TABLE_DIR / "rq3_logistic_regression.csv", index_col=0)
    r = rq3.loc["total_discussion_replies"]
    or_sd = float(r["OR"]) ** sd_replies
    ci_lo_sd = float(r["CI_low"]) ** sd_replies
    ci_hi_sd = float(r["CI_high"]) ** sd_replies
    rows.append({
        "finding": f"Total discussion replies (RQ3 adj., per SD={sd_replies:.2f})",
        "OR": or_sd,
        "CI_low": ci_lo_sd,
        "CI_high": ci_hi_sd,
        "CI_closest_to_null": _ci_closest_to_null(or_sd, ci_lo_sd, ci_hi_sd),
    })

    # ── E-value computation ────────────────────────────────────
    for row in rows:
        rr_point = _or_to_rr(row["OR"])
        rr_ci = _or_to_rr(row["CI_closest_to_null"])
        row["evalue_point"] = round(_evalue(rr_point), 2)
        row["evalue_ci"] = round(_evalue(rr_ci), 2)
        row["OR"] = round(row["OR"], 3)
        row["CI_low"] = round(row["CI_low"], 3)
        row["CI_high"] = round(row["CI_high"], 3)
        row["CI_closest_to_null"] = round(row["CI_closest_to_null"], 3)

    df = pd.DataFrame(rows)
    print()
    print(df[["finding", "OR", "CI_closest_to_null", "evalue_point", "evalue_ci"]].to_string(index=False))
    save_csv(df, "sensitivity_evalue")


if __name__ == "__main__":
    run()
