"""Assemble cached per-trip candidate tables into model-ready frames."""
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import FEATURES, FEATURES_V2

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
CACHE_V2 = ROOT / "cache_v2"
FEATURE_SETS = {"v1": FEATURES, "v2": FEATURES_V2}


def load_partition(part: str, trips=None, cache=None):
    """Return (cands, obs) where
    cands: one row per (trip, obs, arc) with FEATURES and y (1 = reference arc)
    obs:   one row per (trip, obs) with lab arc, reliability and miss flag.
    """
    frames, obs_frames = [], []
    cache = cache or CACHE
    files = sorted(cache.glob(f"feat_{part}_*.pkl"))
    for f in files:
        tid = f.stem.split("_")[-1]
        if trips is not None and tid not in trips:
            continue
        df = pd.read_pickle(f)
        lab = np.load(cache / f"lab_{part}_{tid}.npz")
        n = len(lab["lab"])
        df["y"] = (df.arc.to_numpy() == lab["lab"][df.obs.to_numpy()]).astype(np.int8)
        df["reliable"] = lab["reliable"][df.obs.to_numpy()]
        frames.append(df)
        o = pd.DataFrame({"trip": tid, "obs": np.arange(n), "lab": lab["lab"],
                          "reliable": lab["reliable"], "lab_dist": lab["dist"]})
        hit = df.groupby("obs").y.max().reindex(range(n), fill_value=0).to_numpy()
        o["in_cands"] = hit.astype(bool)
        o["has_cands"] = np.isin(np.arange(n), df.obs.unique())
        obs_frames.append(o)
    return pd.concat(frames, ignore_index=True), pd.concat(obs_frames, ignore_index=True)


def X_of(df, feats=None):
    return df[feats or FEATURES].to_numpy(dtype=np.float32)
