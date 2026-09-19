# Pre-training safety scoring — notebook pipeline

A rebuild of the `safety_rubric` experiments as six Jupyter notebooks, one task
each, all runnable in Google Colab.

## The three rules the code enforces

1. **No composite.** There is no S(D). Every score is one dataset x one
   sub-dimension. Datasets are never pooled and sub-dimensions are never
   averaged with each other. Notebook 01 asserts this before it finishes.
2. **Fractions only.** Every dose, score, rate and threshold is a fraction in
   `[0, 1]` — `0.03`, never `3%`. `safety_lib.unit_check()` raises on anything
   outside that range, and every CSV carries a `units` column. Percent appears
   nowhere in the data; figures label axes "(fraction)".
3. **Applicability is explicit.** A sub-dimension runs only if it is toggled
   on, valid for the dataset's modality, valid for the dataset's context, and
   has every required input. Otherwise it is N/A **with a stated reason**,
   recorded in `results/01_applicability_matrix.csv` — never scored as zero,
   never silently dropped.

## Run order

| notebook | what it does | compute |
|---|---|---|
| `00_prepare_data.ipynb` | builds the four datasets | CPU, network |
| `01_pretraining_safety_scoring.ipynb` | scores each dataset per sub-dimension | CPU (GPU speeds up text) |
| `02_harm_injection.ipynb` | injects harm at 7 doses, re-scores | CPU (GPU speeds up text) |
| `02b_gpu_text_finetune_generate.ipynb` | fine-tunes Pythia, writes generations | **GPU required** |
| `03_posttraining_independent_evaluation.ipynb` | downstream measurement with an independent scorer | CPU (GPU speeds up text) |
| `04_figures.ipynb` | all figures + summary tables | CPU |

`02b` is standalone. It takes nothing from `02` except the seed — the mix is
rebuilt from `(dataset, injector, dose, seed)` — so it can run on a separate
Colab GPU runtime, before or after `02`.

## Files

```
safety_lib.py         the engine: catalog, scorers, injectors, stats, CSV output
dataset_specs.py      the four datasets — context, protected attributes,
                      plausible value ranges, safety-critical strata
notebooks/            the six notebooks
data/                 built by notebook 00
results/              every CSV, written and printed by the notebook that makes it
figures/              PNG (slides) + PDF (LaTeX)
```

Edit `dataset_specs.py`, not the notebooks, to change a context, a value range,
a stratum or a protected attribute. Everything downstream follows from it.

## Colab

Upload `safety_lib.py` and `dataset_specs.py` next to the notebook, then:

```
!pip install -q ucimlrepo pandas numpy scikit-learn xgboost matplotlib
!pip install -q datasets detoxify transformers torch     # text arm only
```

## The datasets

| dataset | modality | context | label | protected attributes |
|---|---|---|---|---|
| `diabetes_130` | tabular | health | `readmit_early` | race, gender |
| `framingham` | tabular | health | `TenYearCHD` | sex, age group |
| `german_credit` | tabular | loan_finance | `bad_credit` | sex, age group |
| `civilcomments` | text | general | — | — |

Each is scored on its own. `german_credit` is the useful contrast: being
`loan_finance`, the physical-safety measures that need a physical outcome are
N/A for it, and the applicability matrix says so in words.

Framingham has no UCI entry; notebook 00 tries two public mirrors and otherwise
asks you to upload `data/framingham_raw.csv`.

## The sub-dimensions

Two dimensions, eleven sub-dimensions. Each has a per-record criterion, its own
aggregation to a dataset score, its own detector and its own threshold.

**Content safety**

| sub-dimension | per-record criterion | applies to |
|---|---|---|
| `harm_content_density` | toxicity ≥ 0.50 | text |
| `identity_attack_density` | identity-attack ≥ 0.50 | text |
| `severe_toxicity_density` | severe-toxicity ≥ 0.50 | text |
| `label_integrity` | out-of-fold model assigns the observed label p < 0.50 | tabular |
| `free_text_field_harm` | toxicity ≥ 0.50 in a tabular free-text column | tabular w/ text column |
| `representation_imbalance` | record's sub-group is below minimum support | tabular w/ protected attrs |

**Physical safety**

| sub-dimension | per-record criterion | applies to |
|---|---|---|
| `physical_harm_enablement` | context hazard term **and** an actionability signal | text |
| `threat_density` | threat ≥ 0.50 | text |
| `measurement_range_violation` | a declared measurement outside its plausible physical range | tabular, health/transport/chem/bio |
| `safety_critical_edge_case_coverage` | record sits in an under-covered safety-critical stratum | tabular, health/transport/finance |
| `outcome_severity_exposure` | record's outcome is the physical-harm event | tabular, health/transport |

`physical_harm_enablement` is worth a note: a record scores 0.5 for naming a
hazard and 1.0 only if it also carries operational detail (a step marker, a
quantity with a unit, an operational imperative). Mention is not enablement,
and the criterion is written down rather than delegated to a classifier.

## Doses and injectors

Doses: **0.00, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40** — fractions, five seeds each.

Each injector targets exactly one sub-dimension and is always run alone:

| injector | targets | dose is a fraction of |
|---|---|---|
| `label_flip` | `label_integrity` | dataset rows |
| `range_violation` | `measurement_range_violation` | dataset rows |
| `subgroup_dropout` | `representation_imbalance` | the target sub-group's rows |
| `edge_case_dropout` | `safety_critical_edge_case_coverage` | the edge-case rows |
| `toxic_injection` | `harm_content_density` | mix records |
| `threat_injection` | `threat_density` | mix records |

The last four keep the row count fixed, so the dose changes composition and
never volume. `dose_basis` is recorded in every CSV: for a sub-group or
edge-case injector, a fraction of the whole dataset would erase a small group
at the first non-zero dose and then saturate, so the dose is taken against the
affected population instead.

Injected data is not written to disk. `sl.injected_train(...)` and
`sl.text_mix_for(...)` are deterministic, so notebooks 02, 02b and 03 rebuild
byte-identical inputs from the seed.

## Independence of the post-training measurement

Text: the pre-training score uses **Detoxify** (Jigsaw-trained). The
generations are scored by
**`facebook/roberta-hate-speech-dynabench-r4-target`** — different weights,
different training corpus. Generation (02b) and evaluation (03) are separate
notebooks so the separation is visible, not just asserted.

Tabular: downstream AUC / ECE / Brier / worst-sub-group AUC come from a **clean
held-out test split** the injector never touched, reusing none of the
pre-training machinery.

`results/03_independence_audit.csv` states this in a form a reviewer can check.

## What each notebook writes

```
00_data_manifest.csv
01_subdimension_catalog.csv          01_thresholds.csv
01_applicability_matrix.csv          01_record_scores__<dataset>.csv
01_subdimension_scores__<dataset>.csv  01_subdimension_scores__ALL.csv
01_verdicts.csv
02_injection__<dataset>__<injector>.csv
02_calibration_summary.csv           02_threshold_crossings.csv
02b_generations__<dataset>__<injector>__<model>.csv
03_generation_scores__*.csv          03_posttraining_tabular__<dataset>.csv
03_posttraining_text__<dataset>__<injector>.csv
03_pre_vs_post.csv                   03_predictive_validity.csv
03_independence_audit.csv
04_first_failing_dose.csv            04_summary_by_dataset_subdimension.csv
```

Every one is printed as it is written.

## Cost control

`02` is the expensive notebook. `SCORE_ALL_APPLICABLE = False` (the default)
scores only the sub-dimension each injector targets — that is the calibration
evidence. Set it to `True` to also score the others, which gives the
specificity result (each injector moves its own sub-dimension and leaves the
rest flat) and draws figure 2. It costs roughly 3–4x.

`02b` is 2 injectors x 7 doses x 5 seeds x 1 model = 70 fine-tuning runs. Start
with `SEEDS = (0,)`; the notebook appends per run and resumes, so widening the
grid later re-uses what is already in the CSV.

## Verification performed

* All six notebooks execute end-to-end (00–04 headless; 02b syntax- and
  symbol-checked, its mix rebuild verified deterministic against 02).
* All 13 (dataset, injector) pairs show monotone dose-response, Pearson
  r ≥ 0.93 — six sub-dimensions dose-validated across four datasets.
* Splits verified reproducible across notebooks; the test split is byte-identical
  at every dose, so the downstream measurement never sees injected data.
* `unit_check` verified to reject percents; the missing-detector path verified
  to fail loudly rather than substitute a fallback classifier.
* Figures rendered and inspected: no dual-axis charts, one score per panel,
  colour-blind-validated series order.

The end-to-end run used stand-in data for the two UCI sets (no UCI access from
the build environment) plus the real downloaded Framingham cohort. The stand-ins
carry the real schemas, so the numbers currently in `results/` are placeholders
— re-run 00–04 in Colab to fill them with the real ones.
