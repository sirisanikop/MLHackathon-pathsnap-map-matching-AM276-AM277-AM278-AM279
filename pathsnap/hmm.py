"""HMM / Viterbi route reconstruction over candidate road arcs.

Hidden state at observation i : one candidate arc c (with its projection
                                position t_c along the arc).
Emission  log P(z_i | c)      : w_e * log(p_norm(c)), p_norm = classifier
                                probability renormalised over the candidate set.
Transition log P(c' | c)      : -|route_dist(c, c') - gc_dist(z_i, z_i+1)| / beta
                                 - speed penalty if route_dist/dt > V_MAX
                                (Newson & Krumm-style; route distance from the
                                projection on c to the projection on c' through
                                the directed road graph).
Unreachable pairs (no path within the search limit) get -inf. If every
transition between i and i+1 is -inf, or the GPS gap is too long
(dt > GAP_S or gc > GAP_M), the chain is broken and a new Viterbi segment
starts (disconnected road segments / dropouts). Observations with no
candidate inside the search radius receive no prediction (-1) and also break
the chain.

The decoder never reads the reference route.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geometry import monotone_align

NEG = -1e18


@dataclass
class HMMParams:
    beta: float = 20.0          # transition scale (m)
    w_emit: float = 1.0         # emission weight
    gap_s: float = 60.0         # time gap that breaks the chain (s)
    gap_m: float = 1500.0       # distance gap that breaks the chain (m)
    v_max: float = 60.0         # m/s, plausibility limit
    back_tol_m: float = 15.0    # tolerated backward motion along the same arc
    back_pen_m: float = 60.0    # extra cost for larger backward motion on same arc
    sp_limit_m: float = 3000.0  # Dijkstra search limit
    top_m: int = 0              # keep only the classifier's top-M candidates per fix as states (0 = all)
    uturn_pen: float = 0.0      # extra log-penalty for switching to the opposite-direction arc
    realign_max_m: float = 80.0  # re-alignment keeps the Viterbi arc beyond this distance
    bridge: bool = False        # link across fixes that have no candidates (short dropouts) instead of breaking


class RoadGraph:
    """Directed graph restricted to arcs near the track (for fast Dijkstra)."""

    def __init__(self, trip, cand_arcs, pad_m=2000.0):
        a, b = trip.arc_geometry()
        self.arcs = trip.arcs
        self.alen = np.linalg.norm(b - a, axis=1)
        lo = trip.xy.min(0) - pad_m
        hi = trip.xy.max(0) + pad_m
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        keep = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                        (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        keep = np.union1d(keep, cand_arcs)
        nodes = np.unique(trip.arcs[keep].ravel())
        self.node_idx = -np.ones(len(trip.nodes), dtype=np.int64)
        self.node_idx[nodes] = np.arange(len(nodes))
        self.nodes = nodes
        u = self.node_idx[trip.arcs[keep, 0]]
        v = self.node_idx[trip.arcs[keep, 1]]
        w = np.maximum(self.alen[keep], 0.01)
        # keep the shortest parallel arc for each (u,v)
        order = np.lexsort((w, v, u))
        u, v, w, ka = u[order], v[order], w[order], keep[order]
        first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
        u, v, w, ka = u[first], v[first], w[first], ka[first]
        self.G = csr_matrix((w, (u, v)), shape=(len(nodes), len(nodes)))
        self.uv2arc = dict(zip(zip(u.tolist(), v.tolist()), ka.tolist()))

    def pair_distances(self, src_nodes, dst_nodes, limit):
        """Shortest-path distances for paired (src,dst) original node ids."""
        s = self.node_idx[src_nodes]
        d = self.node_idx[dst_nodes]
        out = np.full(len(s), np.inf)
        same = s == d
        out[same] = 0.0
        todo = np.where(~same)[0]
        if len(todo) == 0:
            return out
        us = np.unique(s[todo])
        CH = 256
        srow = np.searchsorted(us, s[todo])
        for c0 in range(0, len(us), CH):
            chunk = us[c0:c0 + CH]
            D = dijkstra(self.G, directed=True, indices=chunk, limit=limit)
            m = (srow >= c0) & (srow < c0 + CH)
            sel = todo[m]
            out[sel] = D[srow[m] - c0, d[sel]]
        return out

    def path_arcs(self, a_from, a_to, limit=5000.0):
        """Arc ids strictly between arc a_from and arc a_to along the shortest path."""
        key = (int(a_from), int(a_to))
        cache = self.__dict__.setdefault("_fill_cache", {})
        if key not in cache:
            cache[key] = self._path_arcs(a_from, a_to, limit)
        return cache[key]

    def _path_arcs(self, a_from, a_to, limit=5000.0):
        s = self.node_idx[self.arcs[a_from, 1]]
        t = self.node_idx[self.arcs[a_to, 0]]
        if s == t:
            return []
        D, P = dijkstra(self.G, directed=True, indices=[s], limit=limit, return_predecessors=True)
        if not np.isfinite(D[0, t]):
            return None
        path = [t]
        while path[-1] != s:
            path.append(P[0, path[-1]])
        path = path[::-1]
        return [self.uv2arc[(int(path[k]), int(path[k + 1]))] for k in range(len(path) - 1)]


class Prep:
    """Parameter-independent part of decoding for one trip: states, emissions (unweighted log
    probabilities), consecutive-fix links and the shortest-path distance of every candidate pair.
    Computed once; viterbi() can then be run cheaply for many decoder parameter settings."""
    PREP_GAP_S = 300.0   # links are prepared up to this gap; viterbi() applies params.gap_s (<= this)


def prepare(trip, cands, params: HMMParams = HMMParams(), score_col="pn", graph=None):
    n = trip.n_obs
    xy, t = trip.xy, trip.track[:, 2]
    c = cands.sort_values(["obs", score_col], ascending=[True, False])
    if params.top_m and params.top_m > 0:
        c = c[c.groupby("obs").cumcount() < params.top_m]
    P = Prep()
    P.n, P.trip = n, trip
    P.obs = obs = c.obs.to_numpy()
    P.arc = arc = c.arc.to_numpy()
    P.tp = c.t_proj.to_numpy().astype(float)
    P.logq = np.log(np.clip(c[score_col].to_numpy(), 1e-6, 1.0))
    P.starts = starts = np.searchsorted(obs, np.arange(n))
    P.ends = ends = np.searchsorted(obs, np.arange(n), side="right")
    P.g = g = graph or RoadGraph(trip, np.unique(arc))
    # active fixes = fixes that have at least one candidate; step k links act[k] -> act[k+1]
    P.act = act = np.where(ends > starts)[0]
    P.dt = dt = np.diff(t[act]) if len(act) > 1 else np.zeros(0)
    P.gc = gc = np.linalg.norm(np.diff(xy[act], axis=0), axis=1) if len(act) > 1 else np.zeros(0)
    adjacent = np.diff(act) == 1 if len(act) > 1 else np.zeros(0, bool)
    gap_s = max(params.gap_s, Prep.PREP_GAP_S)
    # without bridging a fix without candidates breaks the chain (only adjacent fixes are linked);
    # with bridging the neighbours of a short candidate dropout are linked directly
    P.link = link = (dt <= gap_s) & (gc <= params.gap_m) & (adjacent | bool(params.bridge))
    pi_src, pi_dst, P.pair_slices = [], [], {}
    off = 0
    for k in np.where(link)[0]:
        A = arc[starts[act[k]]:ends[act[k]]]
        B = arc[starts[act[k + 1]]:ends[act[k + 1]]]
        pi_src.append(np.repeat(trip.arcs[A, 1], len(B)))
        pi_dst.append(np.tile(trip.arcs[B, 0], len(A)))
        P.pair_slices[k] = (off, off + len(A) * len(B))
        off += len(A) * len(B)
    P.SP = g.pair_distances(np.concatenate(pi_src), np.concatenate(pi_dst), params.sp_limit_m) if off else np.zeros(0)
    return P


def viterbi(P, params: HMMParams = HMMParams()):
    """Viterbi over prepared states. Returns (arc per observation, -1 = none; chain id per observation)."""
    n, trip, arc, tp, starts, ends, g, act = P.n, P.trip, P.arc, P.tp, P.starts, P.ends, P.g, P.act
    L, gc, dt, SP = g.alen, P.gc, P.dt, P.SP
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
    logp = params.w_emit * P.logq
    link = P.link & (dt <= params.gap_s)
    pred = -np.ones(n, dtype=np.int64)
    seg = -np.ones(n, dtype=np.int64)
    seg_id = 0
    delta = None
    back = {}
    k_start = None

    def backtrack(k_end, delta_end):
        j = int(np.argmax(delta_end))
        k = k_end
        while True:
            i = act[k]
            pred[i] = arc[starts[i] + j]
            seg[i] = seg_id
            if k == k_start:
                break
            j = back[k][j]
            k -= 1

    for k in range(len(act)):
        i = act[k]
        e = logp[starts[i]:ends[i]]
        if delta is None:
            delta, k_start, back = e.copy(), k, {}
            continue
        if not link[k - 1]:
            backtrack(k - 1, delta)
            seg_id += 1
            delta, k_start, back = e.copy(), k, {}
            continue
        ip = act[k - 1]
        A = arc[starts[ip]:ends[ip]]
        B = arc[starts[i]:ends[i]]
        ta = tp[starts[ip]:ends[ip]]
        tb = tp[starts[i]:ends[i]]
        s0, s1 = P.pair_slices[k - 1]
        sp = SP[s0:s1].reshape(len(A), len(B))
        r = (1 - ta)[:, None] * L[A][:, None] + sp + (tb * L[B])[None, :]
        same = A[:, None] == B[None, :]
        along = (tb[None, :] - ta[:, None]) * L[A][:, None]
        r_same = np.where(along >= -params.back_tol_m, np.abs(along), np.abs(along) + params.back_pen_m)
        r = np.where(same, r_same, r)
        tr = -np.abs(r - gc[k - 1]) / params.beta
        speed = r / max(dt[k - 1], 1e-3)
        tr = tr - np.maximum(0.0, speed - params.v_max) / 5.0
        if params.uturn_pen:
            rev = (frm[A][:, None] == to[B][None, :]) & (to[A][:, None] == frm[B][None, :])
            tr = tr - params.uturn_pen * rev
        tr[~np.isfinite(r)] = NEG
        cand = delta[:, None] + tr
        best = cand.argmax(0)
        if np.all(cand.max(0) <= NEG / 2):  # disconnected: break the chain
            backtrack(k - 1, delta)
            seg_id += 1
            delta, k_start, back = e.copy(), k, {}
            continue
        back[k] = best
        delta = cand[best, np.arange(len(B))] + e
    if delta is not None:
        backtrack(len(act) - 1, delta)
    return pred, seg


def decode(trip, cands, params: HMMParams = HMMParams(), score_col="pn", graph=None):
    """cands: DataFrame (obs, arc, t_proj, <score_col>) for ONE trip.
    Returns (pred_arc per observation (-1 = none), segment id per observation, graph)."""
    P = prepare(trip, cands, params, score_col, graph)
    pred, seg = viterbi(P, params)
    return pred, seg, P.g


def reconstruct_route(pred, seg, graph, fill=True):
    """Collapse per-observation arcs into a route; optionally fill the gaps
    between consecutive distinct arcs with shortest paths (within segments)."""
    route = []
    prev, prev_seg = -1, -1
    n_fill_fail = 0
    for a, s in zip(pred, seg):
        if a < 0:
            continue
        if a == prev:
            continue
        if fill and prev >= 0 and s == prev_seg and graph.arcs[prev, 1] != graph.arcs[a, 0]:
            mid = graph.path_arcs(prev, a)
            if mid is None:
                n_fill_fail += 1
            elif len(mid) <= 60:
                route.extend(mid)
        route.append(int(a))
        prev, prev_seg = a, s
    return np.array(route, dtype=np.int64), n_fill_fail


def realign_to_route(trip, pred, route, max_dist_m=80.0):
    """Decoder post-step (reference-free). Viterbi decides WHICH arcs are driven; this step decides
    which fix belongs to which arc of that predicted route, using a monotone (order-preserving)
    nearest-arc assignment along the route. It removes arc-boundary jitter of the per-fix states and
    assigns fixes to short arcs that were only inserted by shortest-path filling.
    Fixes without a Viterbi state stay unmatched (-1); a fix keeps its Viterbi arc if the re-assigned
    arc would be farther than max_dist_m (safety for wrongly filled stretches)."""
    out = pred.copy()
    m = np.where(pred >= 0)[0]
    if len(m) == 0 or len(route) == 0:
        return out
    a = trip.node_xy[trip.arcs[route, 0]]
    b = trip.node_xy[trip.arcs[route, 1]]
    idx, dist = monotone_align(trip.xy[m], a, b)
    ok = dist <= max_dist_m
    out[m[ok]] = route[idx[ok]]
    return out
