"""
Run all v2 analyses in sequence.

Usage:
  cd textEngagement
  python -m src.analysis_v2.run_all

Or from the analysis_v2 directory:
  python run_all.py

Outputs all tables to output/analysis_v2/tables/
Outputs all figures to output/analysis_v2/figures/
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
    print("  ANALYSIS v2 — Full Pipeline")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    t0 = time.time()

    # Load data once, share across all scripts
    print("\nLoading v2 features...")
    data = load_data()
    df, writers, groups = data
    print(f"  {len(df):,} participants, {len(writers):,} writers, {len(groups['all_features'])} features")

    # Run each analysis
    import rq1_writing
    rq1_writing.run(data)

    import rq2_comments
    rq2_comments.run(data)

    import rq3_forum
    rq3_forum.run(data)

    import clustering
    clustering.run(data)

    import survival
    survival.run(data)

    import gee_robustness
    gee_robustness.run(data)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  All analyses complete in {elapsed:.1f}s")
    print(f"  Tables: {OUT_DIR / 'tables'}")
    print(f"  Figures: {OUT_DIR / 'figures'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
