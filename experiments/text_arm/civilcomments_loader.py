"""
CivilComments loader, sliced by toxicity-label proportion.

Pulls CivilComments from HuggingFace `datasets` (or local cache). The
slicing strategy intentionally varies harm-content density — the focal
sub-dimension for the H3 mechanism in Section IV-C. Signing in to HuggingFace
to get a token may be required.

Requires:  pip install datasets

The full CivilComments train set is ~1.8M rows; we sample down to a
manageable size per slice (default 20k rows).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Cap the in-memory load during development. CivilComments is ~1.8M rows and
# several GB as a pandas frame; 300k rows still leaves ~24k toxic examples,
# plenty for slices up to 0.5 toxic at 20k rows. Set to None for the final run.
DEV_CAP = 300_000

def _load_full(cache_dir: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load CivilComments. "
            "Install with: pip install datasets"
        ) from e

    cache = Path(cache_dir) / "civilcomments"
    ds = load_dataset(
        "google/civil_comments",   # <-- namespaced path; bare 'civil_comments' no longer resolves
        split="train",
        cache_dir=str(cache),
    )
    df = ds.to_pandas()
    if DEV_CAP is not None and len(df) > DEV_CAP:
        df = df.sample(n=DEV_CAP, random_state=0).reset_index(drop=True)
    return df


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return slices that vary in toxicity-label proportion.

    slicing_spec keys used:
      strategy             : 'toxicity_stratified'
      n_slices             : number of slices
      toxicity_proportions : list of target toxicity proportions
                             (len == n_slices)
      rows_per_slice       : total rows per slice
    """
    if slicing_spec["strategy"] != "toxicity_stratified":
        raise ValueError(f"CivilComments loader only supports "
                         f"toxicity_stratified; got {slicing_spec['strategy']}")

    df = _load_full(cache_dir)
    # The 'toxicity' column is a float [0,1]; binarize at 0.5 to match
    # standard CivilComments evaluation convention.
    df["toxicity_label"] = (df["toxicity"] >= 0.5).astype(int)

    toxic_pool = df[df["toxicity_label"] == 1]
    safe_pool = df[df["toxicity_label"] == 0]

    rng = np.random.default_rng(seed)
    n = slicing_spec["rows_per_slice"]
    props = slicing_spec["toxicity_proportions"]
    n_seeds = slicing_spec.get("n_seeds", 1)  # NEW
    slice_seeds = np.random.default_rng(seed).integers(0, 1_000_000, size=n_seeds)

    slices = []
    for i, p_tox in enumerate(props):
        n_tox, n_safe = int(round(n * p_tox)), n - int(round(n * p_tox))
        for s in slice_seeds:  # NEW: replicate each dose
            rng = np.random.default_rng(int(s))
            sl_tox = toxic_pool.sample(n=n_tox, replace=n_tox > len(toxic_pool), random_state=int(rng.integers(1e9)))
            sl_safe = safe_pool.sample(n=n_safe, replace=n_safe > len(safe_pool), random_state=int(rng.integers(1e9)))
            sl = pd.concat([sl_tox, sl_safe]).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            slices.append((f"civilcomments_p{int(p_tox * 100):03d}_seed{int(s) % 100000:05d}", sl))
    return slices
