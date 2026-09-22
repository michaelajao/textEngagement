# Behavioural Engagement Signals and Retention in a Digital Self-Management Programme

Analysis code for the retrospective cohort study *Behavioral Engagement Signals
Associated With Retention in a Digital Self-Management Program*. The study asks
whether participant writing, platform browsing, facilitator comments and forum
participation are associated with programme completion in the Hope Programme, a
web-based self-management intervention for people living with long-term
conditions, and whether those signals are present within the first week.

## Reproducing the analysis

The pipeline runs in three layers; each step writes to `output/`.

```bash
pip install -r requirements.txt
python src/dataset.py --exclude-demo     # 1. platform JSON exports -> flat CSVs
python src/features.py                   # 2. CSVs -> 32 user-level engagement features
python -m src.analysis.run_all           # 3. statistical analyses, figures and tables
```

`src/features.py --skip-nlp` reuses previously computed NLP features. Transformer
inference (RoBERTa sentiment, BART zero-shot topics) uses a GPU if one is available.
Core dependencies: pandas, numpy, scipy, statsmodels, scikit-learn, lifelines,
transformers, torch, matplotlib.

## Data access

The platform data are **not included** in this repository: the exports contain
verbatim participant writing, forum posts, profile text and wellbeing scores. The
data are held by Hope for the Community CIC under a data-sharing agreement with
Coventry University. To reproduce the analysis, place the platform JSON exports in
`data/` and run the three steps above; everything else is regenerated.

## Repository layout

```text
src/
├── dataset.py                        # layer 1: JSON -> CSV
├── features.py                       # layer 2: CSV -> feature tables
├── nlp_features.py                   # RoBERTa sentiment, BART zero-shot topics, RegEx markers
├── utils.py                          # shared utilities and figure style
└── analysis/
    ├── config.py                     # feature loading, GEE spec, output paths, shared helpers
    ├── run_all.py                    # runs the twelve modules below in order
    ├── sample_flow.py                # analytic sample derivation and exclusion accounting
    ├── rq1_writing.py                # writing behaviours and completion
    ├── rq2_comments.py               # facilitator comments
    ├── rq3_forum.py                  # forum participation
    ├── rq4_profile_completion.py     # profile completion
    ├── rq5_wellbeing.py              # SWEMWBS (exploratory)
    ├── engagement_funnel.py          # engagement milestone prevalence
    ├── temporal_trajectories.py      # within-person engagement over time
    ├── clustering.py                 # K-means engagement profiles
    ├── survival.py                   # Kaplan-Meier and log-rank
    ├── gee_robustness.py             # GEE clustered by programme variant
    ├── sensitivity.py                # Multimedia Appendix 1: E-values, cluster bootstrap,
    │                                 #   NLP incremental value, leave-one-module-out, day-7 refit
    │
    │                                 # standalone checks, run individually:
    ├── sensitivity_browsing_outcome.py  # is browsing->completion an outcome-definition artefact?
    ├── sensitivity_forum_spec.py        # H4 forum finding vs. engagement-control specification
    ├── sensitivity_completion_definition.py  # completion timestamp vs >=50%-of-sessions rule
    ├── sensitivity_nlp_firth.py         # Firth refit + CV AUC for the linguistic-feature comparison
    ├── sensitivity_swemwbs_timing.py    # validates inferred SWEMWBS occasions against page visits
    └── leakage_check_for_ml.py          # leakage diagnostic for the follow-on prediction study
data/                                 # platform exports (local only)
output/                               # generated features, figures and tables (local only)
```

`run_all.py` runs the twelve pipeline modules above it in dependency order;
`config.py` is imported by all of them. The six standalone checks are not part
of that pipeline and are run on their own, for example
`python -m src.analysis.leakage_check_for_ml`.

## Analyses

- Univariate comparisons (Mann–Whitney U, χ²) with Benjamini–Hochberg correction
- Logistic regressions for writing, facilitator comments and forum participation,
  adjusted for course
- Generalised estimating equations clustered by programme variant, with cluster
  bootstrap and E-values
- A prospective model restricted to behaviour observable in days 0–7
- Kaplan–Meier retention curves
- Engagement milestone prevalence and temporal trajectories
- K-means engagement profiling
- Exploratory analysis of SWEMWBS wellbeing change

The observation unit is one enrolment (participant × programme variant × start date).
