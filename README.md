# Safety Risk Assessment: An Application-Conditional Rubric for Data Quality, Safety, and Rights Impact

> **Paper status:** under double-blind review at ICTAI 2026.
> Author block is anonymized in the paper PDF; this repository contains the
> framework implementation, experimental code, and replication scripts for the
> five-dataset empirical validation reported in the paper.

Existing AI risk governance frameworks (NIST AI RMF, OMB M-25-21, EU AI Act)
classify *systems* by risk tier, but the dominant sources of downstream
risk — representation gaps, poisoning susceptibility, consent provenance,
factual decay — originate in *training data*. This work proposes a
measurable, application-conditional, lifecycle-aware rubric that assesses
datasets *before* they enter model training, across three integrated
components:

1. Eight quality dimensions adapted from Pipino et al. (2002), with weights
   conditioned on application context (data warehouse, classical ML,
   LLM/DL training).
2. Safety-impact and rights-impact sub-rubrics, each decomposed into
   sub-dimensions operationalized through automated proxies.
3. Re-assessment triggers that govern when stored quality scores expire.

The framework is validated on five public datasets spanning the risk
space, training multiple model families and testing four pre-registered
hypotheses linking pre-training rubric scores to downstream accuracy,
fairness, and factuality outcomes.

---

## Repository layout

```
.
├── README.md                          ← you are here
├── paper/
│   └── pre_training_risk_assessment.pdf   (anonymized submission PDF)
│
├── data_risk_rubric/                  ← framework Python package
│   ├── README.md                      (per-package usage docs)
│   ├── pyproject.toml
│   └── src/data_risk_rubric/
│       ├── dimensions/                (8 quality dimensions, Table I)
│       ├── safety/                    (6 safety sub-dimensions, Table II)
│       ├── rights/                    (5 rights sub-dimensions, Table III)
│       ├── composite.py               (Q, S, R scoring, Section IV-E)
│       └── triggers.py                (5 re-assessment triggers, Table IV)
│
└── data_risk_experiments/             ← validation pipeline
    ├── README.md                      (per-package usage docs)
    ├── pyproject.toml
    ├── src/data_risk_experiments/
    │   ├── datasets/                  (per-dataset loaders + slicers)
    │   ├── models/                    (tabular and text model wrappers)
    │   ├── slicing.py                 (10-slice stratification per dataset)
    │   ├── analysis.py                (H1, H2, H3, H4 statistical tests)
    │   └── plots.py                   (Figures 2-5 generation)
    ├── scripts/
    │   ├── 01_score_datasets.py       (Stage 1: rubric scoring)
    │   ├── 02_train_tabular.py        (Stage 2: tabular model training)
    │   ├── 03_compute_metrics.py      (Stage 2: metrics aggregation)
    │   ├── 04_analyze.py              (H1/H2/H3/H4 with bootstrap CIs)
    │   ├── 05_pull_stage3_results.py  (Stage 3: pull Colab outputs)
    │   ├── 06_make_figures.py         (regenerate paper figures)
    │   └── prepare_stage3_uploads.py  (Stage 3: package for Drive upload)
    ├── notebooks/
    │   └── colab_stage3_h3.ipynb      (Stage 3: text-model training)
    └── tests/
        └── test_stage1.py             (regression tests for Stage 1)
```

The two sub-packages have their own READMEs with deeper usage docs. This
top-level README explains how they fit together and provides the
reproduction path for the paper's empirical claims.

---

## Quickstart: reproduce Stage 1 and Stage 2 (about 1 hour, laptop-only)

This reproduces the rubric scoring (Stage 1) and tabular hypothesis tests
H1, H2, H4 (Stage 2). Stage 3 (H3 with text models) requires a GPU and is
documented separately below.

```bash
# 1. Clone and set up
git clone <repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e data_risk_rubric
pip install -e data_risk_experiments

# 2. Run the pipeline
cd data_risk_experiments
python scripts/01_score_datasets.py       # ~5 min: produces rubric scores
python scripts/02_train_tabular.py        # ~30 min: trains tabular models
python scripts/03_compute_metrics.py      # ~1 min: aggregates outputs
python scripts/04_analyze.py              # ~5 min: bootstrap H1/H2/H4 tests
python scripts/06_make_figures.py         # ~30 sec: regenerates figures

# 3. Inspect results
ls results/analysis/                      # summary.json + h1/h2/h4 JSONs
ls results/figures/                       # Figures 2, 3, 4, 6 (PNG + PDF)
```

Expected output: Figures 2, 3, 4, 6 matching the paper, plus a
`summary.json` containing the headline statistics (Pearson r, 95% bootstrap
CIs, n per dataset) for H1, H2, and H4.

---

## Stage 3: H3 reproduction (Colab GPU required)

H3 fine-tunes Pythia-160M and GPT-2 small on ten CivilComments
toxicity-stratified slices and evaluates on TruthfulQA. The 100-run grid
takes 12-17 hours of L4 GPU time and is structured to run on Google Colab.

**Three phases:** laptop prep → Colab training → laptop analysis.

### Phase 1: laptop prep (one-time)

```bash
# Install extra deps
pip install pyarrow huggingface_hub datasets

# Set HuggingFace token (required for TruthfulQA download)
# Get a read token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here     # Windows: $env:HF_TOKEN = "hf_..."

# Build the upload package
cd data_risk_experiments
python scripts/prepare_stage3_uploads.py
```

This produces `./stage3_uploads/` containing 10 CivilComments slice
parquet files, `truthfulqa_mc.parquet`, and `config.json` (~50 MB total).
Upload this folder's contents to Google Drive at
`MyDrive/data_risk_stage3/inputs/`.

### Phase 2: Colab training

Open `notebooks/colab_stage3_h3.ipynb` in Google Colab. Set the runtime to
**L4 GPU** (`Runtime → Change runtime type`). Run cells in order; cell 5
is the long-running training loop.

The notebook is **disconnect-safe**: each completed (slice, model, seed)
run writes its result JSON to Drive immediately, and re-running cell 5
detects which combinations are already done and skips them. Plan for
multiple sessions over 2-3 days, leaving the browser tab open during each.

### Phase 3: laptop analysis

After all 100 runs complete and Drive sync finishes:

```bash
python scripts/05_pull_stage3_results.py \
    --drive-results-dir "path/to/synced/MyDrive/data_risk_stage3/results"
python scripts/04_analyze.py              # now includes H3
python scripts/06_make_figures.py         # now includes Figure 5 (H3)
```

H3 results land in `results/analysis/h3_results.json` alongside the other
hypotheses.

---

## Datasets

All five datasets are publicly available; no credentials required beyond
a HuggingFace token for CivilComments and TruthfulQA.

| Dataset | Source | Loaded via |
|---|---|---|
| Folktables ACSIncome | Ding et al., NeurIPS 2021 | `folktables` Python package |
| German Credit (UCI 144) | UCI ML Repository | `ucimlrepo` package |
| Diabetes 130-US Hospitals (UCI 296) | Strack et al., 2014 | `ucimlrepo` package |
| CivilComments | Borkan et al., WWW 2019 | HuggingFace `google/civil_comments` |
| UCI Wine Quality (UCI 186) | Cortez et al., 2009 | `ucimlrepo` package |

TruthfulQA (Lin et al., ACL 2022) is used as the H3 evaluation benchmark,
loaded from HuggingFace `truthfulqa/truthful_qa`.

Datasets are not redistributed in this repo. The loaders fetch them on
first use and cache to `data_cache/` (gitignored).

---

## Expected results

Reproducing the pipeline should yield the following headline statistics.
Bootstrap CIs may vary by ±0.01 across runs due to bootstrap sampling
randomness.

**H1 (composite quality vs accuracy)** — confirmed where slicing
varies Q meaningfully:

- German Credit: `r = +0.937, 95% CI [+0.808, +0.993], n = 8` ✓ excludes zero
- Folktables: `r = +0.758, 95% CI [-0.139, +0.951], n = 10` directional
- Wine Quality (negative control): `r = -0.293, CI crosses zero` correctly null
- Diabetes 130: underpowered by design (slicing varies rights/safety, not Q)

**H2 (rights composite vs fairness gaps)** — cross-dataset replication:

- Folktables `R` vs worst-subgroup error: `r = +0.66, CI [+0.25, +0.95]` ✓
- Diabetes 130 `R` vs worst-subgroup error: `r = +0.93, CI [+0.46, +1.00]` ✓
- Both dem_gap correlations also CI-exclude zero
- EOD and DPD not predicted in either dataset (reported as honest negative)

**H3 (safety composite vs TruthfulQA factuality)** — directional but
underpowered:

- All 8 sub-tests (2 predictors × 2 metrics × 2 models) point in the
  predicted negative direction, `r` in [-0.24, -0.19]
- No individual CI excludes zero at n = 10 slices

**H4 (conditional vs uniform weighting)** — empirical equivalence:

- `|ΔR²| < 0.02` across all four testable datasets
- Mixed signs, no consistent advantage
- Conditional weighting retained on theoretical grounds; future work
  identified for data-driven weight calibration

---

## Reproducibility notes

- **Seeds:** all stochastic steps use seeds from `config.GLOBAL["base_seed"]`
  (default 42). Stage 3 uses five seeds per (slice, model) combination:
  42, 43, 44, 45, 46.
- **Python:** developed on Python 3.14 (Windows). Tested on Linux Python
  3.11 in CI. Other 3.10+ versions likely fine.
- **Versions:** see `pyproject.toml` in each sub-package for pinned
  dependencies. The Stage 3 Colab notebook pins
  `transformers==4.44.0`, `datasets==2.20.0`, `accelerate==0.33.0` for
  consistency with the paper's reported numbers.
- **Hashing:** Stage 1 writes a content hash of each loaded dataset to
  `results/rubric_scores/<dataset>_rubric_scores.json` so that result
  drift caused by upstream dataset changes is detectable.
- **Negative results:** the analysis pipeline reports null and negative
  findings with the same prominence as positive findings (no
  significance-filtering). The paper's H3 underpowered result and H4
  empirical equivalence are intentional reporting choices.

---

## Citation

The paper is under double-blind review. A BibTeX entry will be added here
once the venue confirms acceptance and the paper is de-anonymized.

```bibtex
@misc{prerelease_ictai2026,
  title  = {Pre-Training Risk Assessment: An Application-Conditional Rubric
            for Data Quality, Safety, and Rights Impact},
  note   = {Under double-blind review at ICTAI 2026},
  year   = {2026}
}
```

---

## License

Code in this repository is released under the **MIT License** (a permissive
open-source software license). See `LICENSE` in each sub-package for the
full license text.

Datasets are *not* redistributed in this repository; each dataset's
loader fetches from its original source under that source's own license.
Users are responsible for complying with the license of each dataset they
use:

- Folktables ACSIncome: derived from U.S. Census ACS Public Use Microdata
  (public domain); the task definitions and loader are released under MIT.
- UCI datasets (German Credit, Diabetes 130, Wine Quality): released by
  UCI Machine Learning Repository under their standard terms (free for
  academic and research use with attribution).
- CivilComments: released by Google under CC0 1.0 (public domain
  dedication).
- TruthfulQA: released by the authors under Apache 2.0.
