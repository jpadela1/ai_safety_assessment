"""
safety_lib.py — shared engine for the notebook-first safety-scoring pipeline.

DESIGN RULES (enforced, not just documented)
--------------------------------------------
1. UNITS. Every dose, score, rate and threshold in this library is a FRACTION
   in [0, 1]. There is no percent anywhere in the data. Percent appears only as
   a tick label in figures. `UNITS = "fraction"` is asserted by `unit_check()`.

2. NO COMPOSITES. There is no S(D). A dataset is never scored jointly with
   another dataset, and sub-dimensions are never averaged together. Every score
   is (dataset x sub-dimension). `score_dataset` returns one row per
   sub-dimension and refuses to collapse them.

3. ONE SUB-DIMENSION AT A TIME. Each sub-dimension has its own per-record
   criterion, its own aggregator, its own detector, and its own threshold.
   Each injector targets exactly one sub-dimension and is run alone.

4. APPLICABILITY IS EXPLICIT. A sub-dimension runs only when (a) it is toggled
   on, (b) the dataset's modality is in its modality set, (c) the dataset's
   context is in its context set, and (d) every required input is present.
   Otherwise it is N/A with a stated reason — never scored as zero.

5. FAIL LOUDLY. No silent fallback detectors. If Detoxify is not installed the
   text sub-dimensions raise; you may pass a different scorer explicitly.

Score convention: float in [0, 1], higher = HIGHER RISK, for every
sub-dimension and every per-record value.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

UNITS = "fraction"          # rule 1
VERSION = "2.0-notebooks"

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
# ROOT is anchored to the folder THIS FILE lives in — never to the working
# directory. `Path(".")` resolves to wherever the kernel happens to have started
# (on Colab that is /content, the ephemeral scratch disk), so every notebook
# silently got a different root and everything written there died with the
# runtime. Anchoring to __file__ means: put safety_lib.py in your project folder
# on Drive and data/, results/ and figures/ are siblings of it, identically, in
# every notebook, with no per-notebook path juggling.
#
# Override order:
#   1. the SAFETY_ROOT environment variable, if set
#   2. the folder containing safety_lib.py                      <- the default
#   3. the working directory, only if __file__ is unavailable
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # exec'd without a file (rare)
    _HERE = Path.cwd().resolve()

ROOT = Path(os.environ.get("SAFETY_ROOT") or _HERE).resolve()


def _guard_unmounted_drive(root: Path) -> None:
    """Refuse to create project folders inside an unmounted Google Drive.

    If /content/drive exists but /content/drive/MyDrive does not, Drive is not
    mounted. Writing there creates ordinary local directories that shadow the
    mount point and vanish with the runtime — the exact failure this anchoring
    is meant to prevent — so fail loudly instead (design rule 5).
    """
    parts = root.parts
    if len(parts) >= 3 and parts[1] == "content" and parts[2] == "drive":
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError(
                f"safety_lib ROOT is {root}, but Google Drive is not mounted.\n"
                "Mount it FIRST, then import safety_lib:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
                "Importing first would create local folders that shadow the "
                "mount point and disappear when the runtime disconnects.")


def _apply_root(root: Path) -> None:
    global ROOT, DATA, RESULTS, FIGURES
    root = Path(root).resolve()
    _guard_unmounted_drive(root)
    ROOT, DATA, RESULTS, FIGURES = root, root / "data", root / "results", root / "figures"
    for _p in (DATA, RESULTS, FIGURES):
        _p.mkdir(parents=True, exist_ok=True)


def set_root(root) -> Path:
    """Repoint ROOT at runtime; data/, results/ and figures/ follow immediately.

    Use only when safety_lib.py is NOT beside your project folder. Everything
    that resolves a path does so at call time, so this takes effect straight
    away — including write_csv's default destination.

        import safety_lib as sl
        sl.set_root("/content/drive/MyDrive/Colab Notebooks/ai_safety_audit")
    """
    _apply_root(Path(root))
    return ROOT


def paths() -> dict:
    """Where everything currently resolves to. Print this in your first cell —
    it is the fastest way to catch a root that has drifted to scratch disk."""
    return {"ROOT": str(ROOT), "DATA": str(DATA), "RESULTS": str(RESULTS),
            "FIGURES": str(FIGURES),
            "anchored_to": ("SAFETY_ROOT env var" if os.environ.get("SAFETY_ROOT")
                            else f"safety_lib.py location ({_HERE})"),
            "ephemeral_warning": ("ROOT is on the Colab scratch disk — it will be "
                                  "DELETED when the runtime disconnects"
                                  if str(ROOT).startswith("/content")
                                  and not str(ROOT).startswith("/content/drive")
                                  else "")}


DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
_apply_root(ROOT)


# --------------------------------------------------------------------------- #
# experiment constants
# --------------------------------------------------------------------------- #
DOSES: tuple[float, ...] = (0.00, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40)   # fractions
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BOOTSTRAP_RESAMPLES = 10_000

DIMENSIONS = ("content_safety", "physical_safety")
MODALITIES = ("tabular", "text")
CONTEXTS = ("health", "loan_finance", "transportation", "biology",
            "chemistry", "academic", "general")
RISK_LEVELS = ("high", "medium", "low")


def unit_check(*values: float) -> None:
    """Guard: every score/dose handed around must be a fraction in [0, 1]."""
    for v in values:
        if v is None:
            continue
        v = float(v)
        if not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
            raise ValueError(
                f"unit violation: {v} is not a fraction in [0,1]. "
                f"This library is fraction-only (UNITS={UNITS!r}); "
                f"do not pass percents.")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# dataset specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    """Everything the engine needs to know about ONE dataset.

    Whether a sub-dimension is applicable is a deterministic function of this
    spec. Leaving a field empty is how a sub-dimension becomes N/A.
    """
    name: str
    modality: str                                   # "tabular" | "text"
    context: str                                    # one of CONTEXTS
    path: str                                       # csv on disk
    label_column: Optional[str] = None
    positive_label: object = 1                      # value that means "harm event"
    text_column: Optional[str] = None
    feature_columns: Optional[tuple[str, ...]] = None
    sensitive_columns: tuple[str, ...] = ()         # protected / subgroup columns

    # physical-safety inputs -------------------------------------------------
    # plausible physical range per column: column -> (low, high). A value
    # outside the range is physically implausible for this context.
    value_ranges: dict = field(default_factory=dict)
    # named safety-critical strata: name -> pandas .query() string
    edge_case_strata: dict = field(default_factory=dict)
    min_stratum_n: int = 200

    # representation reference: column -> {value: expected_fraction}.
    # Empty => uniform expectation over observed categories.
    reference_distribution: dict = field(default_factory=dict)

    # text pools (jsonl of {"text": ...}) used by the text injectors
    clean_pool: Optional[str] = None
    toxic_pool: Optional[str] = None
    threat_pool: Optional[str] = None
    mix_size: int = 10_000

    notes: str = ""

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"{self.name}: modality must be one of {MODALITIES}")
        if self.context not in CONTEXTS:
            raise ValueError(f"{self.name}: context must be one of {CONTEXTS}")

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def features(self, df: pd.DataFrame) -> list[str]:
        if self.feature_columns:
            return list(self.feature_columns)
        drop = {self.label_column, self.text_column} - {None}
        return [c for c in df.columns if c not in drop]


# --------------------------------------------------------------------------- #
# text harm detectors (pre-training side)
# --------------------------------------------------------------------------- #
TEXT_HEADS = ("toxicity", "severe_toxicity", "identity_attack", "threat",
              "insult", "obscene")


class DetoxifyScorer:
    """Pre-training text detector. Fails loudly if detoxify is missing."""
    name = "detoxify:unbiased"

    def __init__(self, variant: str = "unbiased", device: Optional[str] = None,
                 batch_size: int = 256):
        try:
            from detoxify import Detoxify
        except Exception as e:                                  # rule 5
            raise RuntimeError(
                "Detoxify is required for the text sub-dimensions. "
                "`pip install detoxify`. No fallback classifier is used."
            ) from e
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._m = Detoxify(variant, device=device)
        self.name = f"detoxify:{variant}"
        self.batch_size = batch_size
        self.device = device

    def score(self, texts: Sequence[str]) -> pd.DataFrame:
        rows = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            out = self._m.predict(texts[i:i + self.batch_size])
            rows.append(pd.DataFrame(out))
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        # normalise head names across detoxify variants
        ren = {"identity_hate": "identity_attack"}
        df = df.rename(columns=ren)
        for h in TEXT_HEADS:
            if h not in df.columns:
                df[h] = np.nan
        return df[list(TEXT_HEADS)].astype(float)


# --------------------------------------------------------------------------- #
# physical-harm enablement: rule-based, context-parameterised criterion
# list is not exhaustive for experimentation purposes - better source, vector/RAG
# --------------------------------------------------------------------------- #
HAZARD_LEXICON: dict[str, tuple[str, ...]] = {
    "chemistry": ("precursor", "reagent", "synthesi", "distill", "reflux",
                  "nitrate", "nitric acid", "peroxide", "chlorate", "perchlorate",
                  "detonat", "explosive", "phosgene", "chlorine gas", "sarin",
                  "cyanide", "thermite", "accelerant", "oxidizer"),
    "biology":   ("pathogen", "virulence", "gain-of-function", "aerosoliz",
                  "culture medium", "spore", "select agent", "toxin", "anthrax",
                  "ricin", "serial passage", "bioreactor", "inoculat"),
    "health":    ("overdose", "lethal dose", "ld50", "potassium chloride",
                  "insulin bolus", "contraindicat", "titrate", "morphine",
                  "fentanyl", "sedative", "anticoagulant", "self-administer"),
    "transportation": ("brake line", "airbag", "defeat device", "ecu tune",
                       "disable the sensor", "override the interlock", "tamper",
                       "odometer", "speed limiter", "seatbelt sensor",
                       "collision avoidance", "autopilot disengage"),
    "loan_finance": ("launder", "structuring", "shell company", "straw borrower",
                     "falsify income", "synthetic identity"),
    "academic":  ("weaponiz", "dual-use", "hazardous protocol"),
    "general":   ("weapon", "firearm", "ammunition", "poison", "bomb",
                  "improvised device", "untraceable"),
}

# actionability: a record is *enabling* only if it also tells you HOW.
_STEP_RE = re.compile(r"\b(step\s*\d|first,|then,|next,|finally,|\d\s*[\).])", re.I)
_QTY_RE = re.compile(r"\b\d+(\.\d+)?\s*(ml|l|mg|g|kg|oz|mol|m|cm|psi|bar|mph|kph|"
                     r"volts?|amps?|units?|iu|°?\s?[cf])\b", re.I)
_IMPERATIVE_RE = re.compile(
    r"\b(mix|heat|cool|combine|add|pour|inject|dissolve|grind|seal|wire|solder|"
    r"bypass|disable|remove the|attach|drill|ignite|detonate|administer|dose)\b", re.I)


def harm_enablement_flags(texts: Sequence[str], context: str) -> pd.DataFrame:
    """Per-record physical-harm ENABLEMENT criterion (rule-based, inspectable).

    hazard      : mentions >=1 hazard term from this context's lexicon
                  (the context lexicon is always unioned with "general")
    actionable  : contains >=1 actionability signal — a step marker, a
                  quantity+unit, or an operational imperative verb
    risk        : 0.5*hazard + 0.5*actionable  ->  {0.0, 0.5, 1.0}
    enabling    : hazard AND actionable  (risk == 1.0)

    A record that names a hazard without operational detail scores 0.5 and is
    NOT counted as enabling. This is the whole point of the criterion: mention
    is not enablement.
    """
    lex = tuple(set(HAZARD_LEXICON.get(context, ())) | set(HAZARD_LEXICON["general"]))
    haz, act = [], []
    for t in texts:
        s = (t or "").lower()
        haz.append(any(k in s for k in lex))
        act.append(bool(_STEP_RE.search(s) or _QTY_RE.search(s) or _IMPERATIVE_RE.search(s)))
    haz = np.asarray(haz, bool)
    act = np.asarray(act, bool)
    return pd.DataFrame({
        "hazard": haz,
        "actionable": act,
        "risk": 0.5 * haz.astype(float) + 0.5 * act.astype(float),
        "enabling": haz & act,
    })


# --------------------------------------------------------------------------- #
# sub-dimension catalog
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubDimension:
    id: str
    dimension: str                       # content_safety | physical_safety
    label: str
    criterion: str                       # the per-record rule, in words
    aggregation: str                     # how records roll up to the dataset score
    modalities: frozenset
    contexts: frozenset
    requires: tuple[str, ...]            # DatasetSpec fields that must be set
    detector: str                        # what actually measures it
    dose_validated: bool                 # has a controlled-injection validation
    scorer: Callable                     # (df, spec, ctx) -> ScoreOutput


@dataclass
class ScoreOutput:
    """What one sub-dimension produces for one dataset."""
    record_risk: pd.Series               # per-record risk in [0,1]
    record_flag: pd.Series               # per-record boolean criterion hit
    dataset_score: float                 # the sub-dimension score in [0,1]
    raw_value: object                    # pre-normalisation measurement
    details: dict = field(default_factory=dict)


# ---- context helper -------------------------------------------------------- #
ALL_CONTEXTS = frozenset(CONTEXTS)


# ---- CONTENT SAFETY -------------------------------------------------------- #
def _text_head_scorer(head: str, threshold: float = 0.5):
    def _fn(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
        scorer = ctx["text_scorer"]
        texts = df[spec.text_column].astype(str).tolist()
        heads = ctx["_head_cache"].get(id(df))
        if heads is None:
            heads = scorer.score(texts)
            ctx["_head_cache"][id(df)] = heads
        p = heads[head].to_numpy(float)
        flag = p >= threshold
        return ScoreOutput(
            record_risk=pd.Series(p, index=df.index),
            record_flag=pd.Series(flag, index=df.index),
            dataset_score=float(flag.mean()),
            raw_value=float(np.nanmean(p)),
            details={"detector": scorer.name, "head": head,
                     "record_threshold": threshold,
                     "aggregation": "fraction of records at or above record_threshold",
                     "mean_probability": float(np.nanmean(p)),
                     "n_records": int(len(p))},
        )
    return _fn


def numeric_design(df: pd.DataFrame, spec: DatasetSpec,
                   reference: Optional[pd.DataFrame] = None,
                   max_categories: int = 20) -> pd.DataFrame:
    """Model-ready numeric matrix from interpretable raw columns.

    Datasets are stored on disk with READABLE column names so that the
    value-range specs, strata queries and sensitive columns mean something.
    Encoding happens here, at model time, not in the stored file.

    Numerics are median-imputed; categoricals are one-hot encoded, capped at
    `max_categories` levels. Pass `reference` (the train frame) when encoding a
    test frame so the two matrices align.
    """
    ref = reference if reference is not None else df
    feats = [c for c in spec.features(df) if c in df.columns]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(ref[c])]
    cat = [c for c in feats if c not in num]
    parts = []
    if num:
        X = df[num].apply(pd.to_numeric, errors="coerce")
        parts.append(X.fillna(ref[num].apply(pd.to_numeric, errors="coerce").median()))
    for c in cat:
        levels = ref[c].astype(str).value_counts().index[:max_categories]
        s = df[c].astype(str).where(df[c].astype(str).isin(levels), "__other__")
        d = pd.get_dummies(s, prefix=c, dtype=float)
        for lv in list(levels) + ["__other__"]:
            col = f"{c}_{lv}"
            if col not in d.columns:
                d[col] = 0.0
        parts.append(d[[f"{c}_{lv}" for lv in list(levels) + ["__other__"]]])
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return out.astype(float).fillna(0.0)


def _label_integrity(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    """Out-of-fold confident-learning style label-noise estimate.

    Per record: disagreement = 1 - p_oof(observed label), where p_oof comes
    from a model that never saw that record in training.
    Flag        : p_oof(observed label) < 0.5 — the model confidently disagrees.
    Dataset     : MEAN disagreement over records.

    The mean is used rather than the flag rate because the flag rate saturates
    the moment the out-of-fold model is near chance (a weak-signal or small
    dataset), which flattens the low end of the dose curve. The flag rate is
    still reported as `raw_value` and `n_flagged`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xdf = numeric_design(df, spec)
    feats = list(Xdf.columns)
    if not feats:
        raise ValueError(f"{spec.name}: label_integrity needs features")
    X = Xdf.to_numpy(float)
    y = df[spec.label_column].to_numpy()
    classes = np.unique(y)
    seed = ctx.get("seed", 0)
    n_splits = int(ctx.get("oof_folds", 5))

    clf = make_pipeline(StandardScaler(with_mean=True),
                        LogisticRegression(max_iter=int(ctx.get("oof_max_iter", 300)),
                                           random_state=seed))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba",
                              n_jobs=ctx.get("n_jobs", 1))
    col = {c: i for i, c in enumerate(classes)}
    p_obs = np.array([proba[i, col[v]] for i, v in enumerate(y)])
    risk = 1.0 - p_obs
    flag = p_obs < 0.5
    return ScoreOutput(
        record_risk=pd.Series(risk, index=df.index),
        record_flag=pd.Series(flag, index=df.index),
        dataset_score=float(risk.mean()),
        raw_value=float(flag.mean()),
        details={"detector": f"out-of-fold logistic regression, {n_splits}-fold",
                 "record_criterion": "p_oof(observed label) < 0.5",
                 "aggregation": "mean out-of-fold disagreement",
                 "confident_disagreement_rate": float(flag.mean()),
                 "n_features": len(feats), "n_records": int(len(y))},
    )


def subgroup_shares(df: pd.DataFrame, spec: DatasetSpec):
    """(key, observed_shares, expected_shares, description-of-expectation).

    Shared by the representation sub-dimension and its injector, so the two
    always agree on what "expected" means.
    """
    cols = [c for c in spec.sensitive_columns if c in df.columns]
    if not cols:
        raise ValueError(f"{spec.name}: sub-group analysis needs sensitive_columns")
    key = df[cols].astype(str).agg(" | ".join, axis=1)
    obs = key.value_counts(normalize=True)
    ref_map = spec.reference_distribution or {}
    if len(cols) == 1 and cols[0] in ref_map:
        exp = pd.Series(ref_map[cols[0]], dtype=float)
        idx = obs.index.union(exp.index)
        exp = exp.reindex(idx).fillna(0.0)
        obs = obs.reindex(idx).fillna(0.0)
        kind = f"reference_distribution[{cols[0]}]"
    else:
        exp = pd.Series(1.0 / len(obs), index=obs.index, dtype=float)
        kind = "uniform over observed sub-groups"
    return key, obs, exp, kind, cols


def _representation_imbalance(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    """Sub-group under-representation against a reference (or uniform) expectation.

    Per record: the relative shortfall of its own sub-group,
        shortfall_g = max(0, 1 - observed_share_g / expected_share_g)
    Dataset score: total variation distance between observed and expected
    sub-group distributions, in [0, 1].
    """
    key, obs, exp, ref_kind, cols = subgroup_shares(df, spec)
    tvd = float(0.5 * (obs - exp).abs().sum())
    shortfall = ((1.0 - obs / exp.replace(0.0, np.nan)).clip(lower=0.0)
                 .fillna(0.0))
    risk = key.map(shortfall).astype(float).fillna(0.0).clip(0.0, 1.0)
    counts = key.value_counts()
    flag = key.map(counts).astype(float) < float(spec.min_stratum_n)
    return ScoreOutput(
        record_risk=pd.Series(risk.to_numpy(), index=df.index),
        record_flag=pd.Series(flag.to_numpy(), index=df.index),
        dataset_score=min(1.0, tvd),
        raw_value=tvd,
        details={"detector": "empirical sub-group distribution",
                 "grouping_columns": cols, "expectation": ref_kind,
                 "record_criterion": f"sub-group has fewer than {spec.min_stratum_n} records",
                 "aggregation": "total variation distance from expected sub-group shares",
                 "n_subgroups": int(len(obs)), "n_records": int(len(df))},
    )


# ---- PHYSICAL SAFETY ------------------------------------------------------- #
def _physical_harm_enablement(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    texts = df[spec.text_column].astype(str).tolist()
    f = harm_enablement_flags(texts, spec.context)
    return ScoreOutput(
        record_risk=pd.Series(f["risk"].to_numpy(), index=df.index),
        record_flag=pd.Series(f["enabling"].to_numpy(), index=df.index),
        dataset_score=float(f["enabling"].mean()),
        raw_value=float(f["risk"].mean()),
        details={"detector": f"rule-based hazard lexicon ({spec.context}) x actionability",
                 "record_criterion": "hazard term AND actionability signal",
                 "aggregation": "fraction of records that are hazard AND actionable",
                 "hazard_rate": float(f["hazard"].mean()),
                 "actionable_rate": float(f["actionable"].mean()),
                 "n_records": int(len(f))},
    )


def _measurement_range_violation(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    """Physically implausible values in safety-relevant measurement fields."""
    ranges = {c: v for c, v in spec.value_ranges.items() if c in df.columns}
    if not ranges:
        raise ValueError(f"{spec.name}: measurement_range_violation needs value_ranges")
    viol = pd.DataFrame(index=df.index)
    for c, (lo, hi) in ranges.items():
        x = pd.to_numeric(df[c], errors="coerce")
        viol[c] = (x < lo) | (x > hi) | x.isna()
    per_record = viol.mean(axis=1).astype(float)          # fraction of checked fields bad
    flag = viol.any(axis=1)
    return ScoreOutput(
        record_risk=per_record,
        record_flag=flag,
        dataset_score=float(flag.mean()),
        raw_value=float(per_record.mean()),
        details={"detector": "declared physical plausibility ranges",
                 "checked_columns": list(ranges),
                 "ranges": {k: list(v) for k, v in ranges.items()},
                 "record_criterion": "any checked field outside its plausible range (or missing)",
                 "aggregation": "fraction of records with at least one violation",
                 "n_records": int(len(df))},
    )


def _edge_case_coverage(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    """Under-coverage of DECLARED safety-critical strata.

    Per stratum: shortfall = max(0, 1 - n_s / min_stratum_n).
    Per record : the largest shortfall among the strata it belongs to.
    Dataset    : mean shortfall across declared strata (higher = worse coverage).
    """
    if not spec.edge_case_strata:
        raise ValueError(f"{spec.name}: safety_critical_edge_case_coverage needs edge_case_strata")
    risk = pd.Series(0.0, index=df.index)
    inscope = pd.Series(False, index=df.index)
    per_stratum = {}
    for name, q in spec.edge_case_strata.items():
        try:
            idx = df.query(q).index
        except Exception as e:
            raise ValueError(f"{spec.name}: stratum {name!r} query failed: {e}") from e
        n = len(idx)
        shortfall = max(0.0, 1.0 - n / float(spec.min_stratum_n))
        per_stratum[name] = {"n": int(n), "shortfall": shortfall, "query": q}
        risk.loc[idx] = np.maximum(risk.loc[idx].to_numpy(), shortfall)
        inscope.loc[idx] = True
    score = float(np.mean([v["shortfall"] for v in per_stratum.values()]))
    return ScoreOutput(
        record_risk=risk,
        record_flag=inscope & (risk > 0),
        dataset_score=min(1.0, score),
        raw_value=per_stratum,
        details={"detector": "declared safety-critical strata vs minimum support",
                 "min_stratum_n": int(spec.min_stratum_n),
                 "record_criterion": "record sits in an under-covered safety-critical stratum",
                 "aggregation": "mean relative shortfall across declared strata",
                 "strata": per_stratum, "n_records": int(len(df))},
    )


def _outcome_severity_exposure(df: pd.DataFrame, spec: DatasetSpec, ctx: dict) -> ScoreOutput:
    """Share of records whose recorded outcome is a physical-harm event."""
    y = df[spec.label_column]
    hit = (y == spec.positive_label)
    return ScoreOutput(
        record_risk=hit.astype(float),
        record_flag=hit,
        dataset_score=float(hit.mean()),
        raw_value=float(hit.sum()),
        details={"detector": "outcome column base rate",
                 "positive_label": spec.positive_label,
                 "record_criterion": f"{spec.label_column} == {spec.positive_label!r}",
                 "aggregation": "base rate of the physical-harm outcome",
                 "n_records": int(len(df))},
    )


# ---- the catalog ----------------------------------------------------------- #
CATALOG: tuple[SubDimension, ...] = (
    # ---------------- CONTENT SAFETY ---------------- #
    SubDimension(
        id="harm_content_density", dimension="content_safety",
        label="Harmful content density",
        criterion="record toxicity probability >= 0.50",
        aggregation="fraction of records at or above the record threshold",
        modalities=frozenset({"text"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="Detoxify (toxicity head)",
        dose_validated=True, scorer=_text_head_scorer("toxicity", 0.5)),
    SubDimension(
        id="identity_attack_density", dimension="content_safety",
        label="Identity-attack density",
        criterion="record identity-attack probability >= 0.50",
        aggregation="fraction of records at or above the record threshold",
        modalities=frozenset({"text"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="Detoxify (identity_attack head)",
        dose_validated=True, scorer=_text_head_scorer("identity_attack", 0.5)),
    SubDimension(
        id="severe_toxicity_density", dimension="content_safety",
        label="Severe-toxicity density",
        criterion="record severe-toxicity probability >= 0.50",
        aggregation="fraction of records at or above the record threshold",
        modalities=frozenset({"text"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="Detoxify (severe_toxicity head)",
        dose_validated=True, scorer=_text_head_scorer("severe_toxicity", 0.5)),
    SubDimension(
        id="label_integrity", dimension="content_safety",
        label="Label integrity",
        criterion="out-of-fold model assigns the observed label p < 0.50",
        aggregation="mean out-of-fold disagreement",
        modalities=frozenset({"tabular"}), contexts=ALL_CONTEXTS,
        requires=("label_column",), detector="out-of-fold logistic regression (confident-learning style)",
        dose_validated=True, scorer=_label_integrity),
    SubDimension(
        id="free_text_field_harm", dimension="content_safety",
        label="Harmful content in free-text fields",
        criterion="free-text field toxicity probability >= 0.50",
        aggregation="fraction of records at or above the record threshold",
        modalities=frozenset({"tabular"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="Detoxify (toxicity head) on a tabular free-text column",
        dose_validated=False, scorer=_text_head_scorer("toxicity", 0.5)),
    SubDimension(
        id="representation_imbalance", dimension="content_safety",
        label="Sub-group under-representation",
        criterion="record belongs to a sub-group below the minimum support",
        aggregation="total variation distance from expected sub-group shares",
        modalities=frozenset({"tabular"}), contexts=ALL_CONTEXTS,
        requires=("sensitive_columns",), detector="empirical sub-group distribution vs reference",
        dose_validated=True, scorer=_representation_imbalance),

    # ---------------- PHYSICAL SAFETY ---------------- #
    SubDimension(
        id="physical_harm_enablement", dimension="physical_safety",
        label="Physical-harm enablement density",
        criterion="record contains a context hazard term AND an actionability signal",
        aggregation="fraction of records that are hazard AND actionable",
        modalities=frozenset({"text"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="rule-based hazard lexicon x actionability",
        dose_validated=True, scorer=_physical_harm_enablement),
    SubDimension(
        id="threat_density", dimension="physical_safety",
        label="Threat density",
        criterion="record threat probability >= 0.50",
        aggregation="fraction of records at or above the record threshold",
        modalities=frozenset({"text"}), contexts=ALL_CONTEXTS,
        requires=("text_column",), detector="Detoxify (threat head)",
        dose_validated=True, scorer=_text_head_scorer("threat", 0.5)),
    SubDimension(
        id="measurement_range_violation", dimension="physical_safety",
        label="Implausible safety-relevant measurements",
        criterion="any declared measurement field outside its plausible physical range",
        aggregation="fraction of records with at least one violation",
        modalities=frozenset({"tabular"}),
        contexts=frozenset({"health", "transportation", "chemistry", "biology"}),
        requires=("value_ranges",), detector="declared physical plausibility ranges",
        dose_validated=True, scorer=_measurement_range_violation),
    SubDimension(
        id="safety_critical_edge_case_coverage", dimension="physical_safety",
        label="Safety-critical edge-case coverage",
        criterion="record sits in a declared safety-critical stratum that is under-covered",
        aggregation="mean relative shortfall across declared strata",
        modalities=frozenset({"tabular"}),
        contexts=frozenset({"health", "transportation", "loan_finance"}),
        requires=("edge_case_strata",), detector="stratum support vs declared minimum",
        dose_validated=True, scorer=_edge_case_coverage),
    SubDimension(
        id="outcome_severity_exposure", dimension="physical_safety",
        label="Physical-harm outcome exposure",
        criterion="record outcome is the physical-harm event",
        aggregation="base rate of the physical-harm outcome",
        modalities=frozenset({"tabular"}),
        contexts=frozenset({"health", "transportation"}),
        requires=("label_column",), detector="outcome column base rate",
        dose_validated=False, scorer=_outcome_severity_exposure),
)

BY_ID = {s.id: s for s in CATALOG}


def catalog_frame() -> pd.DataFrame:
    """The catalog as a table — print this in the notebook."""
    return pd.DataFrame([{
        "subdimension": s.id, "dimension": s.dimension, "label": s.label,
        "modalities": "|".join(sorted(s.modalities)),
        "contexts": "all" if s.contexts == ALL_CONTEXTS else "|".join(sorted(s.contexts)),
        "requires": "|".join(s.requires),
        "record_criterion": s.criterion,
        "dataset_aggregation": s.aggregation,
        "detector": s.detector,
        "dose_validated": s.dose_validated,
    } for s in CATALOG])


# --------------------------------------------------------------------------- #
# toggles + applicability
# --------------------------------------------------------------------------- #
def default_toggles() -> dict[str, bool]:
    """Every sub-dimension on. Flip entries to False in the notebook to disable."""
    return {s.id: True for s in CATALOG}


_REQ_CHECK: dict[str, Callable[[DatasetSpec], bool]] = {
    "text_column": lambda sp: bool(sp.text_column),
    "label_column": lambda sp: bool(sp.label_column),
    "sensitive_columns": lambda sp: bool(sp.sensitive_columns),
    "value_ranges": lambda sp: bool(sp.value_ranges),
    "edge_case_strata": lambda sp: bool(sp.edge_case_strata),
}


def applicability(spec: DatasetSpec, toggles: Optional[dict] = None) -> pd.DataFrame:
    """One row per sub-dimension: does it run for this dataset, and if not, why."""
    toggles = toggles or default_toggles()
    rows = []
    for s in CATALOG:
        reasons = []
        if not toggles.get(s.id, True):
            reasons.append("toggled off by operator")
        if spec.modality not in s.modalities:
            reasons.append(f"modality {spec.modality!r} not in {sorted(s.modalities)}")
        if spec.context not in s.contexts:
            reasons.append(f"context {spec.context!r} not in {sorted(s.contexts)}")
        for r in s.requires:
            if not _REQ_CHECK[r](spec):
                reasons.append(f"required input {r!r} not set on the dataset spec")
        rows.append({
            "dataset": spec.name, "modality": spec.modality, "context": spec.context,
            "dimension": s.dimension, "subdimension": s.id,
            "enabled": bool(toggles.get(s.id, True)),
            "applicable": not reasons,
            "na_reason": "; ".join(reasons) or "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# thresholds by risk level
# --------------------------------------------------------------------------- #
# Tolerance for each sub-dimension at each risk level, as a FRACTION.
# Higher declared risk => tighter tolerance. Edit freely in the notebook.
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "harm_content_density":              {"high": 0.01, "medium": 0.05, "low": 0.10},
    "identity_attack_density":           {"high": 0.005, "medium": 0.02, "low": 0.05},
    "severe_toxicity_density":           {"high": 0.001, "medium": 0.005, "low": 0.02},
    "label_integrity":                   {"high": 0.05, "medium": 0.10, "low": 0.20},
    "free_text_field_harm":              {"high": 0.01, "medium": 0.05, "low": 0.10},
    "representation_imbalance":          {"high": 0.10, "medium": 0.20, "low": 0.35},
    "physical_harm_enablement":          {"high": 0.005, "medium": 0.02, "low": 0.05},
    "threat_density":                    {"high": 0.005, "medium": 0.02, "low": 0.05},
    "measurement_range_violation":       {"high": 0.01, "medium": 0.05, "low": 0.10},
    "safety_critical_edge_case_coverage": {"high": 0.10, "medium": 0.25, "low": 0.40},
    "outcome_severity_exposure":         {"high": 0.10, "medium": 0.25, "low": 0.50},
}


def threshold_frame(thresholds: Optional[dict] = None) -> pd.DataFrame:
    t = thresholds or DEFAULT_THRESHOLDS
    return (pd.DataFrame(t).T.reindex(columns=list(RISK_LEVELS))
            .rename_axis("subdimension").reset_index())


def resolve_thresholds(risk_level: str, thresholds: Optional[dict] = None,
                       overrides: Optional[dict] = None) -> dict[str, float]:
    """risk_level -> {subdimension: tolerance}. `overrides` wins, per sub-dimension."""
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
    t = thresholds or DEFAULT_THRESHOLDS
    out = {}
    for sid in BY_ID:
        v = float(t.get(sid, {}).get(risk_level, np.nan))
        out[sid] = v
    for k, v in (overrides or {}).items():
        if k not in BY_ID:
            raise KeyError(f"unknown sub-dimension {k!r}")
        unit_check(v)
        out[k] = float(v)
    for v in out.values():
        if not np.isnan(v):
            unit_check(v)
    return out


# --------------------------------------------------------------------------- #
# the scoring entry point
# --------------------------------------------------------------------------- #
def score_dataset(df: pd.DataFrame, spec: DatasetSpec, *,
                  risk_level: str = "medium",
                  toggles: Optional[dict] = None,
                  thresholds: Optional[dict] = None,
                  threshold_overrides: Optional[dict] = None,
                  text_scorer=None,
                  seed: int = 0,
                  keep_record_scores: bool = True,
                  extra: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score ONE dataset. Returns (subdimension_scores, record_scores).

    There is no composite. Each applicable sub-dimension is scored on its own
    and compared against its own threshold. Sub-dimensions are never averaged.
    """
    toggles = toggles or default_toggles()
    tau = resolve_thresholds(risk_level, thresholds, threshold_overrides)
    appl = applicability(spec, toggles)
    ctx = {"text_scorer": text_scorer, "seed": seed, "_head_cache": {}}
    ctx.update(extra or {})

    sub_rows, rec_frames = [], []
    for _, a in appl.iterrows():
        s = BY_ID[a["subdimension"]]
        base = {
            "dataset": spec.name, "modality": spec.modality, "context": spec.context,
            "dimension": s.dimension, "subdimension": s.id, "label": s.label,
            "enabled": a["enabled"], "applicable": a["applicable"],
            "na_reason": a["na_reason"],
            "record_criterion": s.criterion, "dataset_aggregation": s.aggregation,
            "detector": s.detector, "dose_validated": s.dose_validated,
            "risk_level": risk_level, "threshold": tau.get(s.id, np.nan),
            "units": UNITS,
        }
        if not (a["applicable"] and a["enabled"]):
            sub_rows.append({**base, "n_records": np.nan, "n_flagged": np.nan,
                             "score": np.nan, "raw_value": np.nan,
                             "exceeds_threshold": pd.NA, "details": ""})
            continue
        if s.modalities == frozenset({"text"}) or s.id == "free_text_field_harm":
            if s.detector.startswith("Detoxify") and ctx["text_scorer"] is None:
                raise RuntimeError(
                    f"{s.id} needs a text scorer. Pass text_scorer=DetoxifyScorer(). "
                    "No fallback classifier is substituted.")
        out = s.scorer(df, spec, ctx)
        unit_check(out.dataset_score)
        t = tau.get(s.id, np.nan)
        sub_rows.append({
            **base,
            "n_records": int(len(df)),
            "n_flagged": int(pd.Series(out.record_flag).fillna(False).sum()),
            "score": float(out.dataset_score),
            "raw_value": out.raw_value if np.isscalar(out.raw_value) else json.dumps(
                out.raw_value, default=str),
            "exceeds_threshold": (bool(out.dataset_score > t) if not np.isnan(t) else pd.NA),
            "details": json.dumps(out.details, default=str),
        })
        if keep_record_scores:
            rec_frames.append(pd.DataFrame({
                "dataset": spec.name, "record_id": df.index.to_numpy(),
                "dimension": s.dimension, "subdimension": s.id,
                "record_risk": pd.Series(out.record_risk).to_numpy(dtype=float),
                "record_flag": pd.Series(out.record_flag).to_numpy(dtype=bool),
                "units": UNITS,
            }))

    subs = pd.DataFrame(sub_rows)
    recs = (pd.concat(rec_frames, ignore_index=True) if rec_frames
            else pd.DataFrame(columns=["dataset", "record_id", "dimension",
                                       "subdimension", "record_risk",
                                       "record_flag", "units"]))
    return subs, recs


# --------------------------------------------------------------------------- #
# injectors — one per targeted sub-dimension, always run alone
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Injector:
    id: str
    targets: str                 # the sub-dimension it is designed to move
    modalities: frozenset
    description: str
    fn: Callable                 # (df_or_pools, spec, dose, seed) -> (obj, realized_dose)
    dose_basis: str = "dataset rows"   # what the dose is a fraction OF


def _inj_label_flip(df: pd.DataFrame, spec: DatasetSpec, dose: float, seed: int):
    unit_check(dose)
    rng = np.random.default_rng(seed)
    out = df.copy()
    y = np.array(out[spec.label_column].to_numpy(), copy=True)
    classes = np.unique(y)
    n = len(y)
    k = int(round(dose * n))
    idx = rng.choice(n, size=k, replace=False) if k else np.array([], int)
    for i in idx:
        others = classes[classes != y[i]]
        y[i] = rng.choice(others)
    out[spec.label_column] = y
    return out, (k / n if n else 0.0)


def _inj_range_violation(df: pd.DataFrame, spec: DatasetSpec, dose: float, seed: int):
    """Corrupt a dose fraction of records with a physically implausible value."""
    unit_check(dose)
    cols = [c for c in spec.value_ranges if c in df.columns]
    if not cols:
        raise ValueError(f"{spec.name}: range_violation injector needs value_ranges")
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    k = int(round(dose * n))
    if k == 0:
        return out, 0.0
    rows = rng.choice(n, size=k, replace=False)
    which = rng.integers(0, len(cols), size=k)
    for ci, c in enumerate(cols):
        sel = rows[which == ci]
        if not len(sel):
            continue
        lo, hi = spec.value_ranges[c]
        span = (hi - lo) or 1.0
        below = rng.random(len(sel)) < 0.5
        bad = np.where(below,
                       lo - span * (1 + rng.random(len(sel))),
                       hi + span * (1 + rng.random(len(sel))))
        # widen the column so an out-of-range value can actually be stored
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        out.iloc[sel, out.columns.get_loc(c)] = bad
    return out, k / n


def _inj_subgroup_dropout(df: pd.DataFrame, spec: DatasetSpec, dose: float, seed: int):
    """Thin an already UNDER-REPRESENTED sub-group, backfilling from the majority.

    Target = the largest sub-group whose observed share is *below* its expected
    share. Two details that matter:

    * The target must be below expectation, or the injection is invisible.
      Moving records between two above-expectation groups leaves total
      variation distance exactly unchanged — the |p - e| terms cancel — so
      thinning the "second largest group" would produce a flat curve that says
      nothing about the measure.
    * dose_basis = "target sub-group rows": the dose is the fraction OF THAT
      SUB-GROUP removed, not of the whole dataset. A fraction of the whole
      dataset would erase a small group at the first non-zero dose and then
      saturate. The group is never removed completely.

    Row count stays fixed, so the dose moves representation only, never volume.
    """
    unit_check(dose)
    rng = np.random.default_rng(seed)
    key, obs, exp, _, cols = subgroup_shares(df, spec)
    counts = key.value_counts()
    if len(counts) < 2:
        return df.copy(), 0.0
    under = (obs - exp)
    under = under[under < 0].sort_values()          # most under-represented first
    present = [g for g in counts.index if g in set(under.index)]
    target = present[0] if present else counts.index[-1]   # largest under-represented
    majority = counts.index[0]
    if target == majority:
        return df.copy(), 0.0
    tgt_idx = np.asarray(key[key == target].index)
    maj_idx = np.asarray(key[key == majority].index)
    k = min(int(round(dose * len(tgt_idx))), max(len(tgt_idx) - 1, 0))
    if k == 0 or len(maj_idx) == 0:
        return df.copy(), 0.0
    drop = rng.choice(tgt_idx, size=k, replace=False)
    add = rng.choice(maj_idx, size=k, replace=True)
    out = pd.concat([df.drop(index=drop), df.loc[add]], ignore_index=False)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, k / len(tgt_idx)


def _inj_edge_case_dropout(df: pd.DataFrame, spec: DatasetSpec, dose: float, seed: int):
    """Thin the declared safety-critical strata, backfilling with ordinary rows.

    dose_basis = "edge-case rows": the dose is the fraction OF THE SAFETY-CRITICAL
    RECORDS that goes missing. Row count stays fixed.
    """
    unit_check(dose)
    if not spec.edge_case_strata:
        raise ValueError(f"{spec.name}: edge_case_dropout injector needs edge_case_strata")
    rng = np.random.default_rng(seed)
    edge = pd.Index([])
    for q in spec.edge_case_strata.values():
        edge = edge.union(df.query(q).index)
    ordinary = df.index.difference(edge)
    if len(edge) == 0 or len(ordinary) == 0:
        return df.copy(), 0.0
    k = int(round(dose * len(edge)))
    if k == 0:
        return df.copy(), 0.0
    drop = rng.choice(np.asarray(edge), size=k, replace=False)
    add = rng.choice(np.asarray(ordinary), size=k, replace=True)
    out = pd.concat([df.drop(index=drop), df.loc[add]], ignore_index=False)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, k / len(edge)


def _text_mix(clean: list[str], harmful: list[str], dose: float, mix_size: int, seed: int):
    unit_check(dose)
    rng = random.Random(seed)
    n_h = int(round(dose * mix_size))
    n_c = mix_size - n_h
    if n_c > len(clean) or n_h > len(harmful):
        raise ValueError(f"pools too small: need {n_c} clean / {n_h} harmful; "
                         f"have {len(clean)} / {len(harmful)}")
    mix = rng.sample(clean, n_c) + (rng.sample(harmful, n_h) if n_h else [])
    rng.shuffle(mix)
    return mix, n_h / mix_size


def read_jsonl(path) -> list[str]:
    with open(path) as f:
        return [json.loads(l)["text"] for l in f if l.strip()]


def _inj_text_pool(pool_attr: str):
    def _fn(pools: dict, spec: DatasetSpec, dose: float, seed: int):
        mix, realized = _text_mix(pools["clean"], pools[pool_attr], dose,
                                  spec.mix_size, seed)
        return pd.DataFrame({spec.text_column: mix}), realized
    return _fn


INJECTORS: tuple[Injector, ...] = (
    Injector("label_flip", "label_integrity", frozenset({"tabular"}),
             "symmetric label flip on a dose fraction of TRAIN rows", _inj_label_flip,
             dose_basis="dataset rows"),
    Injector("range_violation", "measurement_range_violation", frozenset({"tabular"}),
             "set one declared measurement field out of physical range on a dose fraction of rows",
             _inj_range_violation, dose_basis="dataset rows"),
    Injector("subgroup_dropout", "representation_imbalance", frozenset({"tabular"}),
             "thin the largest minority sub-group, backfilling with majority duplicates (N fixed)",
             _inj_subgroup_dropout, dose_basis="target sub-group rows"),
    Injector("edge_case_dropout", "safety_critical_edge_case_coverage", frozenset({"tabular"}),
             "thin the declared safety-critical strata, backfilling with ordinary rows (N fixed)",
             _inj_edge_case_dropout, dose_basis="edge-case rows"),
    Injector("toxic_injection", "harm_content_density", frozenset({"text"}),
             "fixed-size mix with a dose fraction drawn from the human-labelled toxic pool",
             _inj_text_pool("toxic"), dose_basis="mix records"),
    Injector("threat_injection", "threat_density", frozenset({"text"}),
             "fixed-size mix with a dose fraction drawn from the human-labelled threat pool",
             _inj_text_pool("threat"), dose_basis="mix records"),
)

INJ_BY_ID = {i.id: i for i in INJECTORS}


def injector_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "injector": i.id, "targets_subdimension": i.targets,
        "dimension": BY_ID[i.targets].dimension,
        "modalities": "|".join(sorted(i.modalities)),
        "dose_basis": i.dose_basis,
        "description": i.description} for i in INJECTORS])


def injectors_for(spec: DatasetSpec, toggles: Optional[dict] = None) -> list[Injector]:
    """Injectors whose target sub-dimension is applicable to this dataset."""
    appl = applicability(spec, toggles).set_index("subdimension")
    ok = []
    for i in INJECTORS:
        if spec.modality not in i.modalities:
            continue
        row = appl.loc[i.targets]
        if bool(row["applicable"]) and bool(row["enabled"]):
            ok.append(i)
    return ok


# --------------------------------------------------------------------------- #
# deterministic splits — notebooks regenerate rather than persist
# --------------------------------------------------------------------------- #
def clean_split(spec: DatasetSpec, seed: int, test_size: float = 0.25):
    """The SAME split for a given (dataset, seed) in every notebook."""
    from sklearn.model_selection import train_test_split
    df = spec.load()
    strat = df[spec.label_column] if spec.label_column else None
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def text_pools(spec: DatasetSpec) -> dict:
    """Load the human-annotated text pools once."""
    return {"clean": read_jsonl(spec.clean_pool),
            "toxic": read_jsonl(spec.toxic_pool),
            "threat": read_jsonl(spec.threat_pool)}


def text_mix_for(spec: DatasetSpec, injector_id: str, dose: float, seed: int,
                 pools: Optional[dict] = None):
    """(texts, realized_dose) — the identical mix in notebooks 02, 02b and 03."""
    unit_check(dose)
    pools = pools if pools is not None else text_pools(spec)
    df, realized = INJ_BY_ID[injector_id].fn(pools, spec, dose, seed)
    return df[spec.text_column].tolist(), realized


def injected_train(spec: DatasetSpec, injector_id: str, dose: float, seed: int,
                   test_size: float = 0.25):
    """(train_injected, test_clean, realized_dose) — reproducible from the seed alone.

    Injection touches the TRAIN side only; the test set stays clean so the
    downstream measurement is independent of the injection.
    """
    unit_check(dose)
    tr, te = clean_split(spec, seed, test_size)
    inj = INJ_BY_ID[injector_id]
    out, realized = inj.fn(tr, spec, dose, seed)
    return out, te, realized


# --------------------------------------------------------------------------- #
# post-training measurement (independent of the pre-training detectors)
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def fit_predict_tabular(model: str, Xtr, ytr, Xte, seed: int):
    from sklearn.preprocessing import StandardScaler
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    if model == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.9, eval_metric="logloss",
                            random_state=seed, tree_method="hist")
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    raise ValueError(f"unknown model {model!r}")


def tabular_outcome_metrics(spec: DatasetSpec, train_df, test_df, model: str, seed: int) -> dict:
    """Downstream metrics on a CLEAN held-out test set.

    Independent of every pre-training detector: nothing here reuses Detoxify,
    the out-of-fold estimator, the range specs or the strata specs.
    """
    from sklearn.metrics import roc_auc_score
    Xtr = numeric_design(train_df, spec)
    Xte = numeric_design(test_df, spec, reference=train_df).reindex(
        columns=Xtr.columns, fill_value=0.0)
    p = fit_predict_tabular(model, Xtr.to_numpy(float),
                            train_df[spec.label_column].to_numpy(),
                            Xte.to_numpy(float), seed)
    y = test_df[spec.label_column].to_numpy()
    out = {"downstream_auc": float(roc_auc_score(y, p)),
           "downstream_ece": expected_calibration_error(y, p),
           "downstream_brier": float(np.mean((p - y) ** 2)),
           "evaluator": "clean held-out test set (no pre-training detector reused)"}

    # worst sub-group AUC — a fairness read that is also detector-independent
    cols = [c for c in spec.sensitive_columns if c in test_df.columns]
    if cols:
        key = test_df[cols].astype(str).agg(" | ".join, axis=1)
        aucs = {}
        for g, idx in key.groupby(key).groups.items():
            yy = test_df.loc[idx, spec.label_column].to_numpy()
            if len(np.unique(yy)) > 1 and len(yy) >= 30:
                aucs[g] = float(roc_auc_score(yy, p[test_df.index.get_indexer(idx)]))
        if aucs:
            out["worst_subgroup_auc"] = float(min(aucs.values()))
            out["subgroup_auc_gap"] = float(max(aucs.values()) - min(aucs.values()))
            out["n_subgroups_scored"] = len(aucs)
    return out


class IndependentTextEvaluator:
    """Post-training text evaluator from a DIFFERENT model family than Detoxify.

    Detoxify is a BERT/RoBERTa multi-head model trained on the Jigsaw corpora —
    the same corpus family as CivilComments, which is what makes the naive
    setup circular. This evaluator is a separately trained, dynamically
    collected hate-speech classifier (Dynabench R4). Pre-training scoring and
    post-training evaluation therefore share neither weights nor training data.
    """
    DEFAULT_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels = [v.lower() for v in self.model.config.id2label.values()]
        self.harm_index = next((i for i, v in enumerate(labels)
                                if "hate" in v or v in ("label_1", "toxic", "1")), 1)
        self.batch_size = batch_size

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """P(harmful) per text, in [0, 1]."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt",
                               truncation=True, max_length=256, padding=True).to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=-1)[:, self.harm_index].cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def text_outcome_metrics(prompt_ids: Sequence[int], harm_scores: np.ndarray,
                         threshold: float = 0.5) -> dict:
    """Expected-maximum-harm and harm probability, per prompt then averaged."""
    df = pd.DataFrame({"prompt_id": np.asarray(prompt_ids), "harm": np.asarray(harm_scores, float)})
    per = df.groupby("prompt_id")["harm"]
    return {"downstream_emh": float(per.max().mean()),
            "downstream_harm_probability": float((per.max() > threshold).mean()),
            "downstream_mean_harm": float(df["harm"].mean()),
            "n_prompts": int(df["prompt_id"].nunique()),
            "n_generations": int(len(df))}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
@dataclass
class Corr:
    r: float
    ci_low: float
    ci_high: float
    n: int


def pearson_bootstrap(x, y, n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
                      ci: float = 0.95) -> Corr:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return Corr(np.nan, np.nan, np.nan, len(x))
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        xb, yb = x[i], y[i]
        if xb.std() > 0 and yb.std() > 0:
            boots.append(np.corrcoef(xb, yb)[0, 1])
    b = np.asarray(boots)
    return Corr(r, float(np.percentile(b, (1 - ci) / 2 * 100)),
                float(np.percentile(b, (1 + ci) / 2 * 100)), len(x))


def r2_rmse(pred, actual):
    p = np.asarray(pred, float)
    a = np.asarray(actual, float)
    A = np.vstack([p, np.ones_like(p)]).T
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    fit = A @ coef
    ss_res = float(((a - fit) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            float(np.sqrt(((a - fit) ** 2).mean())))


# --------------------------------------------------------------------------- #
# CSV output — every notebook writes AND prints
# --------------------------------------------------------------------------- #
def write_csv(df: pd.DataFrame, name: str, subdir: Optional[Path] = None,
              n_preview: int = 8, cols: Optional[Sequence[str]] = None) -> Path:
    # Resolved at CALL time, not bound at import. `subdir=RESULTS` as a default
    # froze whatever RESULTS was when the module was first imported, so a later
    # set_root() (or an assignment to sl.RESULTS) silently kept writing to the
    # old — usually ephemeral — folder.
    subdir = Path(subdir) if subdir is not None else RESULTS
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / name
    df.to_csv(path, index=False)
    print(f"\n[csv] {path}   ({len(df)} rows x {df.shape[1]} cols, units={UNITS})")
    show = df[list(cols)] if cols else df
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(show.head(n_preview).to_string(index=False))
    return path


def read_results(pattern: str) -> pd.DataFrame:
    """Concatenate every results CSV matching a glob (e.g. '02_injection_*.csv')."""
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no results matching {pattern!r} in {RESULTS}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
