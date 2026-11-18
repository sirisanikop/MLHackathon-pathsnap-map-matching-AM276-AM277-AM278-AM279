"""Planar segment geometry (metres, local projection)."""
import numpy as np


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points p to segments a-b (all arrays broadcastable).
    Returns (distance, t) with t the clipped projection fraction in [0,1]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.where(L2 > 1e-9, ((px - ax) * vx + (py - ay) * vy) / np.maximum(L2, 1e-9), 0.0)
    tc = np.clip(t, 0.0, 1.0)
    qx, qy = ax + tc * vx, ay + tc * vy
    return np.hypot(px - qx, py - qy), tc, t


def bearing(dx, dy):
    """Planar bearing in radians (atan2 of dy, dx)."""
    return np.arctan2(dy, dx)


def monotone_align(xy, a, b, extra_cost=None):
    """Assign each point to one segment of an ORDERED segment sequence (a[j]->b[j]) such that the segment
    index never decreases along the track and the summed point-segment distance is minimal (dynamic
    programme, O(n*m)). Returns (segment index per point, distance per point).

    Purely geometric: used both to derive labels from a reference route (labels.py) and to re-assign
    fixes to the arcs of a PREDICTED route (hmm.realign_to_route)."""
    n, m = len(xy), len(a)
    back = np.empty((n, m), dtype=np.int32)
    dist_rows = np.empty((n, m), dtype=np.float32)
    ar = np.arange(m)
    prev = None
    for i in range(n):
        d, _, _ = point_segment(xy[i, 0], xy[i, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        dist_rows[i] = d
        if extra_cost is not None:      # e.g. a direction penalty; reported distance stays geometric
            d = d + extra_cost(i)
        if prev is None:
            cur = d.astype(np.float64)
            back[i] = ar
        else:
            cm = np.minimum.accumulate(prev)
            back[i] = np.maximum.accumulate(np.where(prev == cm, ar, 0))
            cur = d + cm
        prev = cur
    j = int(np.argmin(prev))
    idx = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        idx[i] = j
        j = back[i, j]
    return idx, dist_rows[np.arange(n), idx]
