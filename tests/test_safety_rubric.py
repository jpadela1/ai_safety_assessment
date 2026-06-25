"""Smoke tests for safety_rubric. Run: pytest tests/"""
import numpy as np, pandas as pd
from safety_rubric import (assess_safety, SafetyConfig, SafetyResult,
                           label_integrity, harm_content_density, compute_safety_composite)


def _toy_tabular(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = (df["x1"] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return df


def test_assess_returns_safety_result():
    df = _toy_tabular()
    result, proxies = assess_safety(df, {"source_count": 3}, SafetyConfig(label_column="y"))
    assert isinstance(result, SafetyResult)
    assert result.safety is None or 0.0 <= result.safety <= 1.0


def test_label_integrity_rises_with_noise():
    """A cleaner label set should score lower integrity-risk than a noisier one."""
    df = _toy_tabular(seed=1)
    clean = label_integrity(df, "y").score
    noisy = df.copy()
    flip = np.random.default_rng(2).random(len(noisy)) < 0.4
    noisy.loc[flip, "y"] = 1 - noisy.loc[flip, "y"]
    assert label_integrity(noisy, "y").score > clean


def test_na_logic_excludes_from_denominator():
    """Non-applicable proxies must not be averaged in."""
    df = _toy_tabular()
    result, _ = assess_safety(df, {}, SafetyConfig(label_column="y"))
    excluded = set(result.excluded_for_na)
    assert "harm_content_density" in excluded            # no text column -> N/A
    assert "label_integrity" not in excluded             # has label -> applicable


def test_harm_content_density_na_without_text():
    r = harm_content_density(None)
    assert r.applicable is False


def test_composite_none_when_nothing_applicable():
    res = compute_safety_composite([])
    assert res.safety is None
    assert res.passes_threshold(0.2) is True             # vacuous pass
