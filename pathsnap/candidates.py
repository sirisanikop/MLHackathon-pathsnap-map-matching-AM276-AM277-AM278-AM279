"""Candidate road-arc generation and per-candidate feature extraction.

Uses ONLY track + road network (never the reference route).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import point_segment

RADIUS_M = 80.0     # search radius around each GPS fix
MAX_K = 12          # keep at most K nearest arcs (two-way roads give 2 arcs each)
SAMPLE_STEP = 10.0  # densification step for the spatial index (m)

FEATURES = [
    "d", "d_rank", "d_minus_min", "d_ratio", "n_cand", "n_cand_30",
    "cos_head", "abs_head", "speed", "cos_head_x_speed",
    "t_proj", "t_outside", "arc_len", "log_arc_len",
    "d_prev1", "d_next1", "d_win5", "d_win15", "cos_head_win",
    "conn_prev", "conn_next", "same_prev", "same_next",
    "head_outdeg", "tail_indeg", "twoway", "gap_prev", "gap_next",
]


# ---------------------------------------------------------------------------
# v2 candidate/feature configuration (second tuning round; v1 above is kept
# unchanged so the six-model benchmark stays reproducible)
V2 = dict(radius=80.0, max_k=24, k_samples=192)
FEATURES_V2 = FEATURES + [
    "d_prev5", "d_next5", "d_prev15", "d_next15",            # asymmetric temporal context
    "d_win5_gap", "d_win15_gap", "d_prev5_gap", "d_next5_gap",  # context distance relative to best candidate
    "d_win5_rank", "cos_win_gap",                              # rank / gap of context evidence within the fix
    "cos_head_w20", "prog8", "disp20",                        # long-baseline direction (robust at stops)
    "t_raw", "t_prev5", "t_next5",                            # along-arc position of the fix and its neighbours
    "win_nearest_share", "win_cand_share",                    # how often neighbours agree on this arc
]


class ArcIndex:
    """KD-tree over densified arc samples (restricted to the track bbox)."""

    def __init__(self, trip, bbox_pad=RADIUS_M + 200):
        a, b = trip.arc_geometry()
        self.a, self.b = a, b
        lo = trip.xy.min(0) - bbox_pad
        hi = trip.xy.max(0) + bbox_pad
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        keep = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                        (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        if len(keep) == 0:          # the track is nowhere near the supplied road network
            self.arc_rep, self.tree = np.zeros(0, dtype=np.int64), None
            return
        L = np.linalg.norm(b[keep] - a[keep], axis=1)
        ns = np.maximum(1, np.ceil(L / SAMPLE_STEP).astype(int)) + 1
        arc_rep = np.repeat(keep, ns)
        frac = np.concatenate([np.linspace(0, 1, k) for k in ns])
        pts = a[arc_rep] + frac[:, None] * (b[arc_rep] - a[arc_rep])
        self.arc_rep = arc_rep
        self.tree = cKDTree(pts)

    def query(self, xy, radius=RADIUS_M, k_samples=96):
        if self.tree is None or len(xy) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        k_samples = min(k_samples, len(self.arc_rep))
        dd, ii = self.tree.query(xy, k=k_samples, distance_upper_bound=radius + SAMPLE_STEP,
                                 workers=2)
        obs = np.repeat(np.arange(len(xy)), k_samples)
        ii = np.asarray(ii).reshape(len(xy), -1).ravel()
        ok = ii < len(self.arc_rep)
        df = pd.DataFrame({"obs": obs[ok], "arc": self.arc_rep[ii[ok]]}).drop_duplicates()
        return df["obs"].to_numpy(), df["arc"].to_numpy()


def _gps_heading(xy, t, half=2):
    n = len(xy)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    v = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.linalg.norm(v, axis=1) / dt
    return v, speed


def build_candidates(trip, radius=RADIUS_M, max_k=MAX_K, index=None, k_samples=96, v2=False):
    """Return a DataFrame with one row per (observation, candidate arc) and the
    feature columns in FEATURES. Original trip id and arc id are preserved."""
    xy, t = trip.xy, trip.track[:, 2]
    n = len(xy)
    idx = index or ArcIndex(trip)
    a, b = idx.a, idx.b
    obs, arc = idx.query(xy, radius, k_samples=k_samples)
    d, tc, traw = point_segment(xy[obs, 0], xy[obs, 1], a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1])
    df = pd.DataFrame({"obs": obs, "arc": arc, "d": d, "t_proj": tc,
                       "t_outside": np.abs(traw - tc)})
    df = df[df.d <= radius]
    if len(df) == 0:                # no candidate arc for any fix: empty, correctly typed table
        cols = ["trip", "obs", "arc"] + (FEATURES_V2 if v2 else FEATURES)
        out = pd.DataFrame({c: np.zeros(0, dtype=(object if c == "trip" else np.int64 if c in ("obs", "arc") else np.float32))
                            for c in cols})
        return out
    df = df.sort_values(["obs", "d"], kind="mergesort")
    df["d_rank"] = df.groupby("obs").cumcount()
    df = df[df.d_rank < max_k].reset_index(drop=True)
    obs, arc = df.obs.to_numpy(), df.arc.to_numpy()

    g = df.groupby("obs")["d"]
    dmin = g.transform("min").to_numpy()
    df["d_minus_min"] = df.d - dmin
    df["d_ratio"] = df.d / (dmin + 1.0)
    df["n_cand"] = g.transform("size").to_numpy()
    df["n_cand_30"] = df.assign(c=(df.d <= 30)).groupby("obs")["c"].transform("sum").to_numpy()

    # heading agreement (GPS central-difference heading vs directed arc bearing)
    v, speed = _gps_heading(xy, t, 2)
    av = b[arc] - a[arc]
    alen = np.linalg.norm(av, axis=1)
    gv = v[obs]
    gn = np.linalg.norm(gv, axis=1)
    cosh = np.where((gn > 1e-6) & (alen > 1e-6),
                    (gv * av).sum(1) / np.maximum(gn * alen, 1e-9), 0.0)
    df["cos_head"] = cosh
    df["abs_head"] = np.arccos(np.clip(cosh, -1, 1))
    df["speed"] = speed[obs]
    df["cos_head_x_speed"] = cosh * np.minimum(speed[obs], 30) / 30
    v8, _ = _gps_heading(xy, t, 8)
    g8 = v8[obs]
    g8n = np.linalg.norm(g8, axis=1)
    df["cos_head_win"] = np.where((g8n > 1e-6) & (alen > 1e-6),
                                  (g8 * av).sum(1) / np.maximum(g8n * alen, 1e-9), 0.0)
    df["arc_len"] = alen
    df["log_arc_len"] = np.log1p(alen)

    # temporal context: distances of neighbouring fixes to the same arc
    def dist_to(off):
        j = np.clip(obs + off, 0, n - 1)
        dd, _, _ = point_segment(xy[j, 0], xy[j, 1], a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1])
        return dd
    df["d_prev1"] = dist_to(-1)
    df["d_next1"] = dist_to(1)
    df["d_win5"] = np.mean([dist_to(o) for o in (-5, -3, -1, 0, 1, 3, 5)], axis=0)
    df["d_win15"] = np.mean([dist_to(o) for o in (-15, -10, -5, 0, 5, 10, 15)], axis=0)

    # topology relative to neighbours' nearest arc (route-free)
    nearest = np.full(n, -1)
    first = df.d_rank.to_numpy() == 0
    nearest[obs[first]] = arc[first]
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]

    def conn(off):
        j = np.clip(obs + off, 0, n - 1)
        na = nearest[j]
        valid = na >= 0
        nas = np.where(valid, na, 0)
        same = valid & (nas == arc)
        touch = valid & ((frm[arc] == to[nas]) | (to[arc] == frm[nas]) |
                         (frm[arc] == frm[nas]) | (to[arc] == to[nas]))
        return same.astype(np.int8), (same | touch).astype(np.int8)
    df["same_prev"], df["conn_prev"] = conn(-1)
    df["same_next"], df["conn_next"] = conn(1)

    outdeg = np.bincount(frm, minlength=len(trip.nodes))
    indeg = np.bincount(to, minlength=len(trip.nodes))
    df["head_outdeg"] = outdeg[to[arc]]
    df["tail_indeg"] = indeg[frm[arc]]
    key = frm.astype(np.int64) * (len(trip.nodes) + 1) + to
    rkey = to.astype(np.int64) * (len(trip.nodes) + 1) + frm
    df["twoway"] = np.isin(rkey[arc], key).astype(np.int8)
    dtp = np.diff(t, prepend=t[0])
    dtn = np.diff(t, append=t[-1])
    df["gap_prev"] = dtp[obs]
    df["gap_next"] = dtn[obs]
    if v2:
        def dist_t(off):
            j = np.clip(obs + off, 0, n - 1)
            dd, _, traw = point_segment(xy[j, 0], xy[j, 1], a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1])
            return dd, traw
        df["d_prev5"] = np.mean([dist_to(o) for o in (-5, -4, -3, -2, -1)], axis=0)
        df["d_next5"] = np.mean([dist_to(o) for o in (1, 2, 3, 4, 5)], axis=0)
        df["d_prev15"] = np.mean([dist_to(o) for o in (-15, -12, -9, -6, -3)], axis=0)
        df["d_next15"] = np.mean([dist_to(o) for o in (3, 6, 9, 12, 15)], axis=0)
        grp = df.groupby("obs")
        for c in ["d_win5", "d_win15", "d_prev5", "d_next5"]:
            df[c + "_gap"] = df[c] - grp[c].transform("min")
        df["d_win5_rank"] = grp["d_win5"].rank(method="first").astype(np.float32) - 1
        df["cos_win_gap"] = grp["cos_head_win"].transform("max") - df["cos_head_win"]
        v20, _ = _gps_heading(xy, t, 20)
        g20 = v20[obs]
        g20n = np.linalg.norm(g20, axis=1)
        df["cos_head_w20"] = np.where((g20n > 1e-6) & (alen > 1e-6), (g20 * av).sum(1) / np.maximum(g20n * alen, 1e-9), 0.0)
        df["disp20"] = np.minimum(g20n, 2000.0)
        df["prog8"] = np.where(alen > 1e-6, (g8 * av).sum(1) / np.maximum(alen, 1e-9), 0.0).clip(-500, 500)
        _, traw0 = dist_t(0)
        df["t_raw"] = np.clip(traw0, -3, 4)
        df["t_prev5"] = np.clip(dist_t(-5)[1], -6, 7)
        df["t_next5"] = np.clip(dist_t(5)[1], -6, 7)
        offs = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
        df["win_nearest_share"] = np.mean([nearest[np.clip(obs + o, 0, n - 1)] == arc for o in offs], axis=0)
        A_ = len(trip.arcs) + 1
        keys = np.sort(obs.astype(np.int64) * A_ + arc)
        def member(off):
            j = np.clip(obs + off, 0, n - 1).astype(np.int64) * A_ + arc
            pos = np.clip(np.searchsorted(keys, j), 0, len(keys) - 1)
            return keys[pos] == j
        df["win_cand_share"] = np.mean([member(o) for o in (-10, -5, -3, -1, 1, 3, 5, 10)], axis=0)
    df.insert(0, "trip", trip.trip_id)
    df["d_rank"] = df["d_rank"].astype(np.int16)
    for c in (FEATURES_V2 if v2 else FEATURES):
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df
