"""
Run all analyses in sequence.

Usage:
  cd textEngagement
  python -m src.analysis.run_all

Or from the analysis directory:
  python run_all.py

Outputs all tables to output/analysis/tables/
Outputs all figures to output/analysis/figures/
"""

import sys
import time
from pathlib import Path

# Ensure imports work from both locations
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_data, OUT_DIR


def main():
    print("=" * 70)
    print("  ANALYSIS — Full Pipeline")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    t0 = time.time()

    # Load data once, share across all scripts
    print("\nLoading features...")
    data = load_data()
    df, writers, groups = data
    print(f"  {len(df):,} participants, {len(writers):,} writers, {len(groups['all_features'])} features")

    # Run each analysis
    import sample_flow
    sample_flow.run(data)

    import rq1_writing
    rq1_writing.run(data)

    import rq2_comments
    rq2_comments.run(data)

    import rq3_forum
    rq3_forum.run(data)

    import rq4_profile_completion
    rq4_profile_completion.run(data)

    import pipeline_figure
    pipeline_figure.run(data)

    import engagement_funnel
    engagement_funnel.run(data)

    import temporal_trajectories
    temporal_trajectories.run(data)

    import clustering
    clustering.run(data)

    # rq5_wellbeing reads cluster_assignments.csv (framing d), so it must
    # run after clustering.
    import rq5_wellbeing
    rq5_wellbeing.run(data)

    import survival
    survival.run(data)

    import gee_robustness
    gee_robustness.run(data)

    import sensitivity_evalue
    sensitivity_evalue.run(data)

    import sensitivity_bootstrap
    sensitivity_bootstrap.run(data)

    import sensitivity_nlp_value
    sensitivity_nlp_value.run(data)

    import sensitivity_lomo
    sensitivity_lomo.run(data)

    import sensitivity_prospective_day7
    sensitivity_prospective_day7.run(data)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  All analyses complete in {elapsed:.1f}s")
    print(f"  Tables: {OUT_DIR / 'tables'}")
    print(f"  Figures: {OUT_DIR / 'figures'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
