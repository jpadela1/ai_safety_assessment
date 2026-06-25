"""
safety_rubric — pre-training SAFETY risk assessment with automated proxies.

Safety-only refactor of data_risk_rubric: the quality and rights axes are
removed so the released library matches the paper's single safety claim. The
six safety sub-dimensions of Table I are scored by automated proxies; the
composite S(D) is the unweighted mean of the APPLICABLE sub-dimensions, with
N/A sub-dimensions removed from the denominator (never scored zero).

Convention: for every safety proxy, HIGHER score = HIGHER risk.

Validated proxies (Sec. VI of the paper):
  - harm_content_density   (text)    via Detoxify
  - label_integrity        (tabular) via cross-validated confident-learning
The remaining proxies are specified and runnable but not empirically validated.

    from safety_rubric import assess_safety, SafetyConfig
    result, proxies = assess_safety(df, metadata, SafetyConfig(text_column="comment_text"))
    print(result.safety, result.passes_threshold(s_max=0.2))
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Iterable, Callable

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core result type
# ----------------------------------------------------------------------
@dataclass
class ProxyResult:
    """Structured result returned by every proxy. score in [0,1], higher = riskier."""
    name: str
    score: float
    raw_value: object = None
    applicable: bool = True
    details: dict = field(default_factory=dict)


def _clamp01(x: float) -> float:
    return 0.0 if (x is None or math.isnan(x)) else max(0.0, min(1.0, x))


# ======================================================================
# Group A — Content-origin (CO) proxies
# ======================================================================
def poisoning_susceptibility(metadata: dict) -> ProxyResult:
    """Susceptibility of the source to malicious modification before/during ingestion."""
    risk = 1.0
    if metadata.get("write_access_controls"):    risk -= 0.4
    if metadata.get("cryptographic_provenance"): risk -= 0.3
    source_count = int(metadata.get("source_count", 1))
    diversity_bonus = min(0.3, (source_count - 1) * 0.075)   # saturates at >=5 sources
    risk -= diversity_bonus
    return ProxyResult("poisoning_susceptibility", _clamp01(risk), raw_value=risk,
                       details={"group": "content_origin", "source_count": source_count,
                                "diversity_bonus": diversity_bonus})


def adversarial_provenance_risk(metadata: dict) -> ProxyResult:
    """Risk that adversarial actors contributed without detection."""
    ugc = float(metadata.get("ugc_fraction", 0.0))
    anon = bool(metadata.get("anonymous_contributions", False))
    breadth = metadata.get("scraping_breadth", "narrow")
    breadth_score = {"narrow": 0.0, "medium": 0.3, "broad": 0.7, "web_scale": 1.0}.get(breadth, 0.5)
    risk = 0.5 * ugc + 0.3 * breadth_score + 0.2 * (1.0 if anon else 0.0)
    return ProxyResult("adversarial_provenance_risk", _clamp01(risk), raw_value=risk,
                       details={"group": "content_origin", "ugc_fraction": ugc,
                                "scraping_breadth": breadth, "anonymous": anon})


def factual_decay_rate(metadata: dict, domain_half_life_days: float = 365.0) -> ProxyResult:
    """Rate at which facts become outdated; score = 1 - exp(-age/half_life)."""
    date_str = metadata.get("data_collection_end") or metadata.get("last_updated")
    if not date_str:
        return ProxyResult("factual_decay_rate", 0.5, applicable=False,
                           details={"group": "content_origin", "reason": "no date in metadata"})
    try:
        d = datetime.fromisoformat(date_str)
        d = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    except ValueError:
        return ProxyResult("factual_decay_rate", 1.0, raw_value=date_str,
                           details={"group": "content_origin", "reason": f"unparseable: {date_str}"})
    age_days = max(0.0, (datetime.now(timezone.utc) - d).total_seconds() / 86400.0)
    decay = 1.0 - math.exp(-age_days / max(1.0, domain_half_life_days))
    return ProxyResult("factual_decay_rate", _clamp01(decay), raw_value=age_days,
                       details={"group": "content_origin", "age_days": age_days,
                                "half_life_days": domain_half_life_days})


def harm_content_density(texts: Optional[Iterable[str]] = None, sample_size: int = 1000,
                         toxicity_threshold: float = 0.7, classifier: str = "detoxify",
                         pii_regexes: Optional[dict] = None) -> ProxyResult:
    """Density of content harms (toxicity/hate/harassment/PII) a model might reproduce. [VALIDATED, text]"""
    if texts is None:
        return ProxyResult("harm_content_density", 0.0, applicable=False,
                           details={"group": "content_origin", "reason": "no text column"})
    text_list = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if not text_list:
        return ProxyResult("harm_content_density", 0.0, applicable=False,
                           details={"group": "content_origin", "reason": "text column empty"})
    rng = np.random.default_rng(seed=42)
    n = min(sample_size, len(text_list))
    sampled = [text_list[i] for i in rng.choice(len(text_list), size=n, replace=False)]

    toxic_fraction, used = 0.0, None
    try:
        if classifier == "detoxify":
            from detoxify import Detoxify                      # lazy import
            preds = Detoxify("original").predict(sampled)
            cats = [np.asarray(v) for k, v in preds.items()
                    if k in ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")]
            if cats:
                toxic_fraction = float((np.maximum.reduce(cats) >= toxicity_threshold).mean())
            used = "detoxify"
    except ImportError:
        rough = re.compile(r"\b(hate|kill|stupid|idiot)\b", re.IGNORECASE)   # low-fidelity fallback
        toxic_fraction = sum(1 for t in sampled if rough.search(t)) / max(1, len(sampled))
        used = "lexicon_fallback"

    if pii_regexes is None:
        pii_regexes = {"email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
                       "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b")}
    pii_fraction = sum(1 for t in sampled if any(rx.search(t) for rx in pii_regexes.values())) / max(1, len(sampled))

    score = max(toxic_fraction, pii_fraction)
    return ProxyResult("harm_content_density", _clamp01(score),
                       raw_value={"toxic_fraction": toxic_fraction, "pii_fraction": pii_fraction},
                       details={"group": "content_origin", "classifier_used": used, "n_sampled": n})


def label_integrity(df: Optional[pd.DataFrame] = None, label_column: Optional[str] = None,
                    feature_columns: Optional[list] = None, n_splits: int = 5, seed: int = 0) -> ProxyResult:
    """Corrupted-label fraction via cross-validated confident-learning. [VALIDATED, tabular]

    A mislabeled record in a consequential decision dataset is an integrity
    failure, not a mere accuracy loss. Score = mean(1 - P(assigned label)) from
    a model's out-of-fold predictions; HIGHER = more corruption = HIGHER risk.
    """
    if df is None or label_column is None or label_column not in getattr(df, "columns", []):
        return ProxyResult("label_integrity", 0.0, applicable=False,
                           details={"group": "content_origin", "reason": "no label column"})
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    y = pd.factorize(df[label_column])[0]
    feats = feature_columns or [c for c in df.columns if c != label_column]
    X = df[feats].select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if X.shape[1] == 0:
        return ProxyResult("label_integrity", 0.0, applicable=False,
                           details={"group": "content_origin", "reason": "no numeric features"})
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    proba = cross_val_predict(pipe, X, y, method="predict_proba",
                              cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    score = float(1.0 - proba[np.arange(len(y)), y].mean())
    return ProxyResult("label_integrity", _clamp01(score), raw_value=score,
                       details={"group": "content_origin", "n_features": X.shape[1], "n_rows": len(y)})


# ======================================================================
# Group B — Physical-safety (PS) proxies  (specified; not validated)
# ======================================================================
def physical_harm_enablement_density(texts: Optional[Iterable[str]] = None, sample_size: int = 1000,
                                     dual_use_terms: Optional[list] = None) -> ProxyResult:
    """Density of content giving uplift toward physical harm. Flags for REVIEW, not exclusion."""
    if texts is None:
        return ProxyResult("physical_harm_enablement_density", 0.0, applicable=False,
                           details={"group": "physical_safety", "reason": "no text column"})
    text_list = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if not text_list:
        return ProxyResult("physical_harm_enablement_density", 0.0, applicable=False,
                           details={"group": "physical_safety", "reason": "text column empty"})
    # Placeholder category-level taxonomy (detection terms only, no operational content).
    terms = dual_use_terms or ["precursor", "synthesis route", "attack plan", "mass casualty"]
    pat = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    rng = np.random.default_rng(seed=42)
    n = min(sample_size, len(text_list))
    sampled = [text_list[i] for i in rng.choice(len(text_list), size=n, replace=False)]
    frac = sum(1 for t in sampled if pat.search(t)) / max(1, len(sampled))
    return ProxyResult("physical_harm_enablement_density", _clamp01(frac), raw_value=frac,
                       details={"group": "physical_safety", "n_sampled": n,
                                "note": "flag for elevated review + access control, not auto-exclude"})


def safety_critical_edge_case_coverage(present_probes: Optional[Iterable[str]] = None,
                                       required_probes: Optional[list] = None,
                                       physical_process_coupled: bool = False) -> ProxyResult:
    """Coverage of rare-but-consequential edge cases. N/A unless physical-process-coupled."""
    if not physical_process_coupled or not required_probes:
        return ProxyResult("safety_critical_edge_case_coverage", 0.0, applicable=False,
                           details={"group": "physical_safety",
                                    "reason": "purely informational application (N/A)"})
    present = set(present_probes or [])
    coverage = sum(1 for p in required_probes if p in present) / max(1, len(required_probes))
    return ProxyResult("safety_critical_edge_case_coverage", _clamp01(1.0 - coverage),  # gaps = risk
                       raw_value=coverage,
                       details={"group": "physical_safety", "coverage": coverage,
                                "n_required": len(required_probes)})


# ======================================================================
# Composite + entry point
# ======================================================================
@dataclass
class SafetyConfig:
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    feature_columns: Optional[list] = None
    domain_half_life_days: float = 365.0
    physical_process_coupled: bool = False
    required_edge_probes: Optional[list] = None
    present_edge_probes: Optional[list] = None
    harm_sample_size: int = 1000
    toxicity_threshold: float = 0.7
    classifier: str = "detoxify"


@dataclass
class SafetyResult:
    safety: Optional[float]                     # S(D) in [0,1], higher = riskier; None if no sub-dim applies
    breakdown: list = field(default_factory=list)      # (name, score|None, applicable)
    excluded_for_na: list = field(default_factory=list)

    def passes_threshold(self, s_max: float = 0.2) -> bool:
        # An axis with no applicable sub-dimension passes vacuously; check breakdown before trusting it.
        return (self.safety is None) or (self.safety <= s_max)

    def to_dict(self) -> dict:
        return {"safety": self.safety, "breakdown": self.breakdown,
                "excluded_for_na": self.excluded_for_na}


def compute_safety_composite(results: list[ProxyResult]) -> SafetyResult:
    applicable = [r.score for r in results if r.applicable]
    breakdown = [(r.name, r.score if r.applicable else None, r.applicable) for r in results]
    excluded = [r.name for r in results if not r.applicable]
    s = (sum(applicable) / len(applicable)) if applicable else None   # None != 0.0 (N/A vs assessed-safe)
    return SafetyResult(safety=s, breakdown=breakdown, excluded_for_na=excluded)


def assess_safety(df: pd.DataFrame, metadata: dict, config: SafetyConfig,
                  extra_proxies: Optional[list[Callable[[], ProxyResult]]] = None
                  ) -> tuple[SafetyResult, list[ProxyResult]]:
    """Run the applicable safety proxies and return (SafetyResult, per-proxy results).

    `extra_proxies` is the pluggable extension point: pass zero-arg callables
    (e.g. a custom image-content proxy) that return a ProxyResult, and they are
    folded into the composite with no change to scoring logic.
    """
    texts = df[config.text_column].tolist() if (config.text_column and config.text_column in df.columns) else None
    results = [
        poisoning_susceptibility(metadata),
        adversarial_provenance_risk(metadata),
        factual_decay_rate(metadata, config.domain_half_life_days),
        harm_content_density(texts, config.harm_sample_size, config.toxicity_threshold, config.classifier),
        label_integrity(df, config.label_column, config.feature_columns),
        physical_harm_enablement_density(texts, config.harm_sample_size),
        safety_critical_edge_case_coverage(config.present_edge_probes, config.required_edge_probes,
                                           config.physical_process_coupled),
    ]
    for p in (extra_proxies or []):
        results.append(p())
    return compute_safety_composite(results), results


__version__ = "0.4.0-safety"
__all__ = ["assess_safety", "SafetyConfig", "SafetyResult", "ProxyResult",
           "compute_safety_composite", "poisoning_susceptibility", "adversarial_provenance_risk",
           "factual_decay_rate", "harm_content_density", "label_integrity",
           "physical_harm_enablement_density", "safety_critical_edge_case_coverage"]
