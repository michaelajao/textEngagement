"""
Run all Layer 3 analysis scripts.

Usage:
    python -m src.analysis.run_all
"""

from __future__ import annotations

from src.analysis import (
    apply_publication_style,
    ensure_output_dirs,
    load_analytical_tables,
)
from src.analysis import rq1_writing_engagement
from src.analysis import rq2_facilitator_comments
from src.analysis import rq3_forum_participation
from src.analysis import regression
from src.analysis import survival
from src.analysis import clustering
from src.analysis import ml_models


def main() -> None:
    apply_publication_style()
    ensure_output_dirs()

    print("Loading analytical tables ...")
    data = load_analytical_tables()
    print(
        f"  users: {len(data['users']):,}  |  "
        f"activities: {len(data['activities']):,}  |  "
        f"pairs: {len(data['pairs']):,}\n"
    )

    rq1_writing_engagement.run(data)
    rq2_facilitator_comments.run(data)
    rq3_forum_participation.run(data)
    regression.run(data)
    survival.run(data)
    clustering.run(data)
    ml_models.run(data)

    print("=" * 50)
    print("All analyses complete.")


if __name__ == "__main__":
    main()
