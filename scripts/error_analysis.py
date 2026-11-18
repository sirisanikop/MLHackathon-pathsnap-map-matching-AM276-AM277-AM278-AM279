"""Residual-error breakdown on the 10 student-test trips, and a label-noise check.

For every scored fix where a prediction differs from the label, classify the error:
  adjacent_on_route : predicted arc is the previous/next arc of the reference route (arc-boundary disagreement)
  reverse_direction : predicted arc is the opposite direction of a reference arc
  other_on_route    : another arc of the reference route
  off_route         : arc not on the reference route
Label-noise check: share of moving fixes (speed > 3 m/s) whose LABEL arc points against the GPS
direction of travel (cos < -0.5) -- labels come from a geometry-only alignment and cannot be verified.
Also verifies that predictions are identical when the .route files are physically absent.
usage: python scripts/error_analysis.py [round1]
output: results/error_breakdown.json (final system) or results/error_breakdown_round1.json
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT, list_trips, load_route  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

ROUND1 = len(sys.argv) > 1 and sys.argv[1] == "round1"     # "round1": analyse the round-1 system instead
P = Predictor(config=ROOT / "configs/round1_config.json") if ROUND1 else Predictor()
OUT = ROOT / ("results/error_breakdown_round1.json" if ROUND1 else "results/error_breakdown.json")
tmp = Path(tempfile.mkdtemp(prefix="pathsnap_noroute_"))
cats = ["adjacent_on_route", "reverse_direction", "other_on_route", "off_route", "no_prediction"]
count = {s: dict.fromkeys(cats, 0) for s in ["cls_pred", "hmm_pred", "final_pred"]}
n_scored, identical, per_trip = 0, True, []
against_all, moving_all, rev_err_moving, rev_err_moving_label_against = 0, 0, 0, 0
for t in list_trips("student_test"):
    d = tmp / "noroute" / t
    d.mkdir(parents=True)
    for e in ["track", "nodes", "arcs"]:
        shutil.copy(DEFAULT_DATA_ROOT / "student_test" / t / f"{t}.{e}", d / f"{t}.{e}")
    a = P.predict_trip(t, "student_test")
    b = P.predict_trip(t, "noroute", tmp)
    identical &= all(np.array_equal(a[k], b[k]) for k in ["cls_pred", "hmm_pred", "final_pred", "route"])
    lab = np.load(ROOT / "cache" / f"lab_student_test_{t}.npz")
    y, rel, ridx = lab["lab"], lab["reliable"], lab["ridx"]
    ref = load_route(t, "student_test")
    tr = a["trip"]
    arcs = tr.arcs
    refset = set(ref.tolist())
    rev = {(int(arcs[r, 1]), int(arcs[r, 0])) for r in ref}
    n_scored += int(rel.sum())
    for s in count:
        p = a[s]
        for i in np.where(rel & (p != y))[0]:
            if p[i] < 0:
                k = "no_prediction"
            elif p[i] in (ref[max(ridx[i] - 1, 0)], ref[min(ridx[i] + 1, len(ref) - 1)]):
                k = "adjacent_on_route"
            elif (int(arcs[p[i], 0]), int(arcs[p[i], 1])) in rev:
                k = "reverse_direction"
            elif p[i] in refset:
                k = "other_on_route"
            else:
                k = "off_route"
            count[s][k] += 1
    n = tr.n_obs
    i0, i1 = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    v = tr.xy[i1] - tr.xy[i0]
    sp = np.linalg.norm(v, axis=1) / np.maximum(tr.track[i1, 2] - tr.track[i0, 2], 1e-6)
    av = tr.node_xy[arcs[y, 1]] - tr.node_xy[arcs[y, 0]]
    cos = (v * av).sum(1) / np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(av, axis=1), 1e-9)
    mv = rel & (sp > 3)
    f = a["final_pred"]
    fs = np.maximum(f, 0)
    is_rev = (f >= 0) & (arcs[fs, 0] == arcs[y, 1]) & (arcs[fs, 1] == arcs[y, 0]) & mv
    against_all += int((cos[mv] < -0.5).sum())
    moving_all += int(mv.sum())
    rev_err_moving += int(is_rev.sum())
    rev_err_moving_label_against += int((cos[is_rev] < -0.5).sum())
    per_trip.append(dict(trip=t, final_top1=float((f == y)[rel].mean()),
                         label_against_motion_share=float((cos[mv] < -0.5).mean()) if mv.sum() else 0.0))
shutil.rmtree(tmp, ignore_errors=True)
out = {"n_scored_fixes": n_scored, "predictions_identical_without_route_files": bool(identical),
       "errors": {s: {"n_errors": int(sum(c.values())), "error_rate": sum(c.values()) / n_scored, **c}
                  for s, c in count.items()},
       "label_noise": {"moving_scored_fixes": moving_all, "label_against_motion": against_all,
                       "share": against_all / moving_all,
                       "final_reverse_direction_errors_while_moving": rev_err_moving,
                       "of_which_label_against_motion": rev_err_moving_label_against},
       "per_trip": per_trip}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_trip"}, indent=1))
