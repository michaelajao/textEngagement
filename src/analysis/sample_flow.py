"""
Sample flow / exclusion accounting for the STROBE-style participant flow.

Recomputes, from the current data build, every number in the manuscript's
Study Population paragraph:

  1. Total enrolments in the export
  2. Non-initiators (no platform start timestamp), broken into:
       - no observable event of any kind
       - browsed pages only ("lurkers")
       - behavioural evidence of engagement (forum post or writing activity)
  3. Initiators (start timestamp present)
  4. Administratively censored initiators (started < ELIGIBILITY_WINDOW_DAYS
     before the export horizon) — excluded by the eligibility criterion
  5. Platform accounts (facilitator/admin logins with no submissions)
  6. Final analytic sample, completers, dropouts

Outputs: output/analysis/tables/sample_flow.csv
"""

from pathlib import Path

import pandas as pd

from config import save_csv
from src.utils import parse_mixed_datetime
from src.features import ELIGIBILITY_WINDOW_DAYS, STAFF_LOGIN_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = ROOT / "data" / "csv"


def run(data=None):
    print("\n" + "=" * 60)
    print("Sample Flow / Exclusion Accounting")
    print("=" * 60)

    users = pd.read_csv(CSV_DIR / "users.csv")
    users["started"] = parse_mixed_datetime(users["started"])
    users["finished"] = parse_mixed_datetime(users["finished"])
    activities = pd.read_csv(CSV_DIR / "activities.csv")
    activities["recorded"] = parse_mixed_datetime(activities["recorded"])
    # Submissions of every type, retained for the platform-account rule below,
    # which asks whether an account ever submitted anything at all.
    all_activities = activities
    # Keep the "wrote" milestone consistent with the feature table: the
    # Emotions word-cloud is not participant-authored writing.
    activities = activities[activities["type_name"] != "Emotions"]
    discussions = pd.read_csv(CSV_DIR / "discussions.csv")
    page_visits = pd.read_csv(CSV_DIR / "page_visits.csv")
    page_visits["latest"] = parse_mixed_datetime(page_visits["latest"])

    n_total = len(users)

    # Event evidence per enrolment / user
    wrote_keys = set(map(tuple, activities[["module_id", "user_id"]]
                         .drop_duplicates().values))
    forum_keys = set(map(tuple, discussions[["module_id", "user_id"]]
                         .drop_duplicates().values))
    browse_keys = set(map(tuple, page_visits[["module_id", "user_id"]]
                          .drop_duplicates().values))

    mu = list(zip(users["module_id"], users["user_id"]))
    users["_wrote"] = [k in wrote_keys for k in mu]
    users["_forum"] = [k in forum_keys for k in mu]
    users["_browsed"] = [k in browse_keys for k in mu]
    users["_logged_in"] = users["n_logins"].fillna(0) > 0

    non_init = users[users["started"].isna()]
    n_non_init = len(non_init)
    behav = non_init["_wrote"] | non_init["_forum"]
    n_behavioural = int(behav.sum())
    lurker = ~behav & (non_init["_browsed"] | non_init["_logged_in"])
    n_lurkers = int(lurker.sum())
    n_no_event = n_non_init - n_behavioural - n_lurkers

    initiators = users[users["started"].notna()]
    n_init = len(initiators)

    export_ts = max(
        users["finished"].max(),
        activities["recorded"].max(),
        page_visits["latest"].max(),
    )
    cutoff = export_ts - pd.Timedelta(days=ELIGIBILITY_WINDOW_DAYS)
    censored = initiators[initiators["started"] >= cutoff]
    n_censored = len(censored)
    n_censored_nofinish = int(censored["finished"].isna().sum())

    eligible = initiators[initiators["started"] < cutoff]

    # Same platform-account rule as features.build_user_level, so this flow
    # table and the feature table report the same analytic sample: an enrolment
    # with many logins and no submitted activity of any type.
    keys = ["module_id", "user_id", "cohort_id"]
    act_counts = (
        all_activities.groupby(keys).size().rename("_n_acts").reset_index()
    )
    eligible = eligible.merge(act_counts, on=keys, how="left")
    eligible["_n_acts"] = eligible["_n_acts"].fillna(0)
    is_platform_account = (
        (eligible["n_logins"] >= STAFF_LOGIN_THRESHOLD) & (eligible["_n_acts"] == 0)
    )
    n_staff = int(is_platform_account.sum())

    analytic = eligible[~is_platform_account]
    n_analytic = len(analytic)
    n_completers = int(analytic["finished"].notna().sum())
    n_dropouts = n_analytic - n_completers

    rows = [
        {"step": "Total enrolments", "n": n_total},
        {"step": "Non-initiators (no start timestamp)", "n": n_non_init},
        {"step": "  - no observable event", "n": n_no_event},
        {"step": "  - browsed/logged in only (lurkers)", "n": n_lurkers},
        {"step": "  - behavioural evidence (wrote or forum)", "n": n_behavioural},
        {"step": "Initiators", "n": n_init},
        {"step": f"Administratively censored (<{ELIGIBILITY_WINDOW_DAYS}d before export)",
         "n": n_censored},
        {"step": "  - of which lacked completion timestamp", "n": n_censored_nofinish},
        {"step": f"Platform accounts (>={STAFF_LOGIN_THRESHOLD} logins, no activity)",
         "n": n_staff},
        {"step": "Analytic sample", "n": n_analytic},
        {"step": "  - completers", "n": n_completers},
        {"step": "  - dropouts", "n": n_dropouts},
    ]
    flow = pd.DataFrame(rows)
    flow["pct_of_total"] = (flow["n"] / n_total * 100).round(1)
    print(f"\n  Export horizon: {export_ts:%Y-%m-%d}; cutoff: {cutoff:%Y-%m-%d}")
    for _, r in flow.iterrows():
        print(f"  {r['step']:55s} {r['n']:>6,} ({r['pct_of_total']:5.1f}%)")
    print(f"\n  Completion rate (analytic sample): "
          f"{n_completers / n_analytic * 100:.1f}%")
    save_csv(flow, "sample_flow")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
