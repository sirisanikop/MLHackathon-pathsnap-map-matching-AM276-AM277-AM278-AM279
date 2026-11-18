"""Server-side evaluation step (laptop 1 only).

Runs AFTER prediction is complete. It is the only API code that reads the
private reference route. Nothing computed here is fed back into the
prediction, and reference geometry/labels are never returned to the UI --
only aggregate scores.
"""
import math

import numpy as np
from sklearn.metrics import f1_score

from pathsnap.data import load_route
from pathsnap.labels import OFFROUTE_M, align_route
from pathsnap.metrics import collapse, continuity, route_overlap


def _num(x):
    """JSON-safe float: undefined values (NaN/inf) become null instead of breaking the response."""
    x = float(x)
    return x if math.isfinite(x) else None


def evaluate_prediction(trip, cls_pred, hmm_pred, final_pred, route, graph=None, partition=None, data_root=None):
    ref = load_route(trip.trip_id, partition, data_root)
    if ref is None or len(ref) == 0:
        return {"available": False, "reason": "no private reference route on server for this trip"}
    if trip.n_obs == 0:
        return {"available": False, "reason": "the trip has no GPS observations"}
    if ref.min() < 0 or ref.max() >= len(trip.arcs):
        return {"available": False, "reason": "reference route contains arc ids outside this trip's road network"}
    a, b = trip.arc_geometry()
    alen = np.linalg.norm(b - a, axis=1)
    lab, _, ldist = align_route(trip.xy, trip.node_xy, trip.arcs, ref)
    rel = ldist <= OFFROUTE_M
    if not rel.any():
        return {"available": False, "n_observations": int(len(lab)), "n_scored_observations": 0,
                "reason": f"no GPS observation lies within {OFFROUTE_M:.0f} m of the reference route"}
    y = lab[rel]
    ro = route_overlap(route, ref, alen)
    ro_c = route_overlap(collapse(cls_pred), ref, alen)
    stages = {}
    for name, p in [("classifier_only", cls_pred), ("hmm", hmm_pred), ("final", final_pred)]:
        stages[name] = {
            "observation_top1_accuracy": _num((p[rel] == y).mean()),
            "observation_macro_f1": _num(f1_score(y, p[rel], average="macro", zero_division=0)),
            "route_continuity": _num(continuity(p, trip.arcs)),
            "observation_top1_accuracy_all_fixes": _num((p == lab).mean()),
        }
    return {
        "available": True,
        "n_observations": int(len(lab)),
        "n_scored_observations": int(rel.sum()),
        "scoring_note": f"fixes farther than {OFFROUTE_M:.0f} m from the aligned reference route are not scored",
        "stages": stages,
        "observation_top1_accuracy": stages["final"]["observation_top1_accuracy"],
        "route_precision": _num(ro["route_precision"]),
        "route_recall": _num(ro["route_recall"]),
        "route_f1": _num(ro["route_f1"]),
        "route_f1_classifier_only": _num(ro_c["route_f1"]),
        "reference_route_length_km": _num(alen[ref].sum() / 1000),
    }
