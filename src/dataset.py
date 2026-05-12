"""
Convert H4C platform JSON exports into consolidated flat CSV files.

Reads three JSON files (UserActivity, FacilitatorComments, DiscussionTopics)
and produces 5 CSVs:

  users.csv                 - one row per (module, cohort, user) with enrollment, outcome,
                              cohort info, and aggregated login/bookmark counts
  activities.csv            - unified activities (merged from UserActivity +
                              FacilitatorComments) with word_count and fc_only flag
  facilitator_comments.csv  - facilitator comment text with word_count
  discussions.csv           - discussion replies with topic metadata and word_count
  page_visits.csv           - page engagement (hits, avg_duration)

Usage:
  python src/dataset.py                          # defaults
  python src/dataset.py -i data -o data/csv
  python src/dataset.py --exclude-demo           # filter demo/test cohorts
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path


DEMO_PATTERN = re.compile(r"(DEMO|TEST|PPIE|GLITCH|^PHOTO\s)", re.IGNORECASE)


def _word_count(text: str | None) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def write_csv(
    output_dir: str, filename: str, rows: list[dict], fieldnames: list[str]
) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {filename}: {len(rows):,} rows")


def find_json_file(input_dir: str, candidates: list[str]) -> str:
    """Return the first existing file from a list of candidate filenames."""
    for name in candidates:
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"None of {candidates} found in {input_dir}")


# ---------------------------------------------------------------------------
# UserActivity parser
# ---------------------------------------------------------------------------

def parse_user_activity(input_dir: str, exclude_demo: bool = False):
    """Parse UserActivity JSON.

    Returns users, activities, page_visits.
    Logins and bookmarks are aggregated directly into user rows.
    """
    path = find_json_file(input_dir, [
        "UserActivity (2).txt",
        "UserActivity (1).txt",
        "UserActivity.txt",
    ])
    print(f"Processing {os.path.basename(path)} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    users_rows: list[dict] = []
    ua_activities: list[dict] = []
    page_visits_rows: list[dict] = []

    for module in data.get("modules", []):
        mod_id = module.get("id")
        mod_name = module.get("name")
        course_id = module.get("course", {}).get("id")
        course_name = module.get("course", {}).get("name")

        for cohort in module.get("cohorts", []):
            cohort_id = cohort.get("id")
            cohort_name = cohort.get("name", "").strip()

            if exclude_demo and DEMO_PATTERN.search(cohort_name):
                continue

            for user in cohort.get("users", []):
                uid = user.get("userId")

                # Aggregate logins
                logins = user.get("logins", [])
                n_logins = len(logins)
                login_span_days = 0.0
                if n_logins >= 2:
                    dates = sorted(
                        lg.get("signedIn", "") for lg in logins if lg.get("signedIn")
                    )
                    if len(dates) >= 2:
                        first = datetime.fromisoformat(dates[0].rstrip("Z"))
                        last = datetime.fromisoformat(dates[-1].rstrip("Z"))
                        login_span_days = (last - first).total_seconds() / 86400

                # Aggregate bookmarks
                bookmarks = user.get("bookmarks", [])
                n_bookmarks = len(bookmarks)

                # Aggregate page visits
                pvs = user.get("pageVisits", [])
                n_page_visits = sum(pv.get("hits", 0) for pv in pvs)
                n_distinct_pages = len({pv.get("url", "") for pv in pvs})

                finished = user.get("finished")
                users_rows.append({
                    "module_id": mod_id,
                    "module_name": mod_name,
                    "course_id": course_id,
                    "course_name": course_name,
                    "cohort_id": cohort_id,
                    "cohort_name": cohort_name,
                    "user_id": uid,
                    "started": user.get("started"),
                    "finished": finished,
                    "is_completer": finished is not None,
                    "n_logins": n_logins,
                    "login_span_days": round(login_span_days, 2),
                    "n_bookmarks": n_bookmarks,
                    "n_page_visits": n_page_visits,
                    "n_distinct_pages": n_distinct_pages,
                })

                for act in user.get("activities", []):
                    desc = act.get("description", "")
                    ua_activities.append({
                        "module_id": mod_id,
                        "module_name": mod_name,
                        "course_id": course_id,
                        "course_name": course_name,
                        "activity_id": act.get("id"),
                        "user_id": act.get("userId") or uid,
                        "cohort_id": act.get("cohortId") or cohort_id,
                        "type_name": act.get("typeName"),
                        "description": desc,
                        "recorded": act.get("recorded"),
                        "word_count": _word_count(desc),
                        "has_facilitator_comments": bool(act.get("facilitatorComments")),
                    })

                for pv in pvs:
                    page_visits_rows.append({
                        "module_id": mod_id,
                        "module_name": mod_name,
                        "user_id": pv.get("userId") or uid,
                        "cohort_id": pv.get("cohortId") or cohort_id,
                        "url": pv.get("url"),
                        "page_title": pv.get("pageTitle"),
                        "hits": pv.get("hits"),
                        "avg_duration": pv.get("avgDuration"),
                        "latest": pv.get("latest"),
                    })

    del data
    return users_rows, ua_activities, page_visits_rows


# ---------------------------------------------------------------------------
# FacilitatorComments parser
# ---------------------------------------------------------------------------

def parse_facilitator_comments(input_dir: str, exclude_demo: bool = False):
    """Parse FacilitatorComments JSON into fc_activities and comments."""
    path = find_json_file(input_dir, [
        "FacilitatorComments.txt",
        "FacilitatorComments (1).txt",
    ])
    print(f"Processing {os.path.basename(path)} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fc_activities: list[dict] = []
    comments_rows: list[dict] = []

    for module in data.get("modules", []):
        mod_id = module.get("id")
        mod_name = module.get("name")
        course_id = module.get("course", {}).get("id")
        course_name = module.get("course", {}).get("name")

        cohort_lookup = {
            c["id"]: c.get("name", "").strip()
            for c in module.get("cohorts", [])
            if "id" in c
        }

        for act in module.get("userActivities", []):
            act_id = act.get("id")
            cohort_id = act.get("cohortId")
            cohort_name = cohort_lookup.get(cohort_id, "")
            if exclude_demo and DEMO_PATTERN.search(cohort_name):
                continue
            fc_comments_list = act.get("facilitatorComments") or []
            desc = act.get("description", "")

            fc_activities.append({
                "module_id": mod_id,
                "module_name": mod_name,
                "course_id": course_id,
                "course_name": course_name,
                "activity_id": act_id,
                "user_id": act.get("userId"),
                "cohort_id": cohort_id,
                "cohort_name": cohort_name,
                "type_name": act.get("typeName"),
                "description": desc,
                "recorded": act.get("recorded"),
                "word_count": _word_count(desc),
                "num_comments": len(fc_comments_list),
                "has_facilitator_comments": bool(fc_comments_list),
            })

            for comment in fc_comments_list:
                ctxt = comment.get("comment", "")
                comments_rows.append({
                    "module_id": mod_id,
                    "module_name": mod_name,
                    "activity_id": act_id,
                    "comment_id": comment.get("id"),
                    "facilitator_user_id": comment.get("userId"),
                    "comment_text": ctxt,
                    "recorded": comment.get("recorded"),
                    "word_count": _word_count(ctxt),
                })

    del data
    return fc_activities, comments_rows


# ---------------------------------------------------------------------------
# Merge activities
# ---------------------------------------------------------------------------

def merge_activities(
    ua_activities: list[dict], fc_activities: list[dict]
) -> list[dict]:
    """Full outer join of UserActivity and FacilitatorComments activities.

    Adds fc_only flag for activities only found in FacilitatorComments.
    """
    fc_lookup = {row["activity_id"]: row for row in fc_activities}

    seen_ids: set[int] = set()
    merged: list[dict] = []
    for row in ua_activities:
        aid = row["activity_id"]
        seen_ids.add(aid)
        fc_info = fc_lookup.get(aid, {})
        merged.append({
            **row,
            "cohort_name": fc_info.get("cohort_name", ""),
            "num_comments": fc_info.get("num_comments", 0),
            "fc_only": False,
        })

    fc_only_count = 0
    for row in fc_activities:
        aid = row["activity_id"]
        if aid not in seen_ids:
            merged.append({**row, "fc_only": True})
            fc_only_count += 1

    if fc_only_count:
        print(f"  (added {fc_only_count} activities found only in FacilitatorComments)")

    return merged


# ---------------------------------------------------------------------------
# DiscussionTopics parser
# ---------------------------------------------------------------------------

def parse_discussion_topics(input_dir: str):
    """Parse DiscussionTopics JSON into discussion replies with topic metadata."""
    path = find_json_file(input_dir, [
        "DiscussionTopics.txt",
        "DiscussionTopics (1).txt",
    ])
    print(f"Processing {os.path.basename(path)} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    discussions_rows: list[dict] = []

    for module in data.get("modules", []):
        mod_id = module.get("id")
        mod_name = module.get("name")
        course_id = module.get("course", {}).get("id")
        course_name = module.get("course", {}).get("name")

        for topic in module.get("topics", []):
            topic_id = topic.get("id")
            page_title = topic.get("pageTitle")

            for reply in topic.get("replies", []):
                ctxt = reply.get("comment", "")
                discussions_rows.append({
                    "module_id": mod_id,
                    "module_name": mod_name,
                    "course_id": course_id,
                    "course_name": course_name,
                    "topic_id": topic_id,
                    "page_title": page_title,
                    "reply_id": reply.get("id"),
                    "user_id": reply.get("userId"),
                    "comment": ctxt,
                    "recorded": reply.get("recorded"),
                    "word_count": _word_count(ctxt),
                })

    del data
    return discussions_rows


# ---------------------------------------------------------------------------
# UserProfile parser
# ---------------------------------------------------------------------------

INTERVIEW_SEPARATOR = " ||| "


def parse_user_profile(input_dir: str):
    """Parse UserProfile JSON into one row per user.

    UserProfile export shape: { "modules": [ { "userProfiles": [ {
        "userId": int, "bio": str|null,
        "interview": { "items": [{ "question": str, "answer": str }] } | null
    } ] } ] }

    Profile is user-level (not enrolment-specific). The same userId may appear
    across multiple modules; we keep the first non-empty bio/interview seen.
    """
    try:
        path = find_json_file(input_dir, [
            "UserProfile (1).txt",
            "UserProfile.txt",
        ])
    except FileNotFoundError:
        print("UserProfile file not found - skipping profile extraction.")
        return []

    print(f"Processing {os.path.basename(path)} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_user: dict[int, dict] = {}
    for module in data.get("modules", []):
        for prof in module.get("userProfiles") or []:
            uid = prof.get("userId")
            if uid is None:
                continue
            bio_raw = prof.get("bio")
            bio = bio_raw.strip() if isinstance(bio_raw, str) else ""
            interview = prof.get("interview") or {}
            items = interview.get("items") or []
            qa_pairs = [
                (it.get("question", "").strip(), it.get("answer", "").strip())
                for it in items
                if isinstance(it, dict)
            ]
            qa_pairs = [(q, a) for q, a in qa_pairs if q or a]
            n_answers = len(qa_pairs)
            interview_concat = INTERVIEW_SEPARATOR.join(
                f"{q} {a}".strip() for q, a in qa_pairs
            )

            existing = by_user.get(uid)
            if existing is None:
                by_user[uid] = {
                    "user_id": uid,
                    "has_bio": bool(bio),
                    "bio": bio,
                    "bio_word_count": _word_count(bio),
                    "has_interview": n_answers > 0,
                    "n_interview_answers": n_answers,
                    "interview_text": interview_concat,
                    "interview_word_count": _word_count(interview_concat),
                }
            else:
                # Prefer the first record that actually has content; only fill
                # in missing fields from later records.
                if not existing["has_bio"] and bio:
                    existing["has_bio"] = True
                    existing["bio"] = bio
                    existing["bio_word_count"] = _word_count(bio)
                if not existing["has_interview"] and n_answers > 0:
                    existing["has_interview"] = True
                    existing["n_interview_answers"] = n_answers
                    existing["interview_text"] = interview_concat
                    existing["interview_word_count"] = _word_count(interview_concat)

    del data
    return list(by_user.values())


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

USERS_FIELDS = [
    "module_id", "module_name", "course_id", "course_name",
    "cohort_id", "cohort_name",
    "user_id", "started", "finished", "is_completer",
    "n_logins", "login_span_days", "n_bookmarks",
    "n_page_visits", "n_distinct_pages",
]

ACTIVITIES_FIELDS = [
    "module_id", "module_name", "course_id", "course_name",
    "activity_id", "user_id", "cohort_id", "cohort_name",
    "type_name", "description", "recorded", "word_count",
    "num_comments", "has_facilitator_comments", "fc_only",
]

COMMENTS_FIELDS = [
    "module_id", "module_name", "activity_id", "comment_id",
    "facilitator_user_id", "comment_text", "recorded", "word_count",
]

DISCUSSIONS_FIELDS = [
    "module_id", "module_name", "course_id", "course_name",
    "topic_id", "page_title", "reply_id", "user_id",
    "comment", "recorded", "word_count",
]

PAGE_VISITS_FIELDS = [
    "module_id", "module_name", "user_id", "cohort_id",
    "url", "page_title", "hits", "avg_duration", "latest",
]

USER_PROFILES_FIELDS = [
    "user_id", "has_bio", "bio", "bio_word_count",
    "has_interview", "n_interview_answers",
    "interview_text", "interview_word_count",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert H4C JSON exports to 5 consolidated CSVs."
    )
    script_dir = Path(__file__).resolve().parent.parent
    default_input = str(script_dir / "data")
    default_output = str(script_dir / "data" / "csv")

    parser.add_argument(
        "-i", "--input", default=default_input,
        help=f"Directory containing JSON exports (default: {default_input})",
    )
    parser.add_argument(
        "-o", "--output", default=default_output,
        help=f"Directory for CSV output (default: {default_output})",
    )
    parser.add_argument(
        "--exclude-demo", action="store_true", default=False,
        help="Exclude demo/test cohorts (names containing DEMO, TEST, or PPIE)",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if args.exclude_demo:
        print("Filtering: excluding demo/test cohorts")
    print()

    # --- 1. UserActivity ---
    users, ua_acts, page_visits = parse_user_activity(
        input_dir, exclude_demo=args.exclude_demo
    )
    write_csv(output_dir, "users.csv", users, USERS_FIELDS)
    write_csv(output_dir, "page_visits.csv", page_visits, PAGE_VISITS_FIELDS)
    print()

    # --- 2. FacilitatorComments ---
    fc_acts, comments = parse_facilitator_comments(
        input_dir, exclude_demo=args.exclude_demo
    )
    write_csv(output_dir, "facilitator_comments.csv", comments, COMMENTS_FIELDS)
    print()

    # --- 3. Merge activities ---
    print("Merging activities (UserActivity + FacilitatorComments) ...")
    activities = merge_activities(ua_acts, fc_acts)
    write_csv(output_dir, "activities.csv", activities, ACTIVITIES_FIELDS)

    ua_count = len(ua_acts)
    fc_count = len(fc_acts)
    merged_count = len(activities)
    print(f"  (ua: {ua_count:,} + fc: {fc_count:,} -> merged: {merged_count:,})")
    print()

    del ua_acts, fc_acts

    # --- 4. DiscussionTopics ---
    discussions = parse_discussion_topics(input_dir)
    write_csv(output_dir, "discussions.csv", discussions, DISCUSSIONS_FIELDS)
    print()

    # --- 5. UserProfile (bio + interview) ---
    profiles = parse_user_profile(input_dir)
    if profiles:
        write_csv(output_dir, "user_profiles.csv", profiles, USER_PROFILES_FIELDS)
        print()

    # --- Summary ---
    print("=" * 50)
    n_csvs = 5 + (1 if profiles else 0)
    print(f"Done! {n_csvs} CSVs saved to:", output_dir)
    print(f"  users:                {len(users):>8,}")
    print(f"  activities:           {len(activities):>8,}")
    print(f"  facilitator_comments: {len(comments):>8,}")
    print(f"  discussions:          {len(discussions):>8,}")
    print(f"  page_visits:          {len(page_visits):>8,}")
    if profiles:
        print(f"  user_profiles:        {len(profiles):>8,}")


if __name__ == "__main__":
    main()
