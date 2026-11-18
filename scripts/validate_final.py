"""Validation of the final system (run after tune_decoder.py).

1. Trip-level bootstrap (10,000 resamples of trips) for pooled final top-1 of the round-1 system and the
   final system, and for their paired difference, on the 80 training trips (out-of-fold) and on the 10
   student-test trips.
2. Label-protocol sensitivity on student-test: both systems are re-scored against direction-aware labels
   (same monotone alignment plus a 40 m penalty when a fix moving faster than 3 m/s would be assigned to an
   arc pointing against its direction of travel). Evaluation only; predictions are unchanged.
3. Independent recomputation of the final student-test top-1 straight from the pipeline output, and the
   no-.route leakage check for the final system.
output: results/validation_final.json
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT, list_trips, load_route  # noqa: E402
from pathsnap.geometry import monotone_align  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

RES = ROOT / "results"
rng = np.random.default_rng(42)
B = 10_000


def boot(ok, n, ok2=None):
    """Percentile bootstrap over trips for pooled accuracy (and paired difference)."""
    ok, n = np.asarray(ok, float), np.asarray(n, float)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    acc = ok[idx].sum(1) / n[idx].sum(1)
    out = {"point": float(ok.sum() / n.sum()), "ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]}
    if ok2 is not None:
        ok2 = np.asarray(ok2, float)
        d = ok2[idx].sum(1) / n[idx].sum(1) - acc
        out = {"point": float((ok2.sum() - ok.sum()) / n.sum()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
               "prob_improvement": float((d > 0).mean())}
    return out


out = {"bootstrap_resamples": B}
for part, f1, f2 in [("train_oof", "realign_train_oof.csv", "final_train_oof.csv"), ("student_test", "realign_test.csv", "final_test.csv")]:
    a = pd.read_csv(RES / f1, dtype={"trip": str}).set_index("trip")
    b = pd.read_csv(RES / f2, dtype={"trip": str}).set_index("trip").loc[a.index]
    assert (a.n_reliable.to_numpy() == b.n_rel.to_numpy()).all()
    n = a.n_reliable.to_numpy()
    ok1 = np.round(a.realign_top1.to_numpy() * n)
    ok2 = b.n_ok.to_numpy()
    out[part] = {"n_trips": int(len(n)), "n_fixes": int(n.sum()), "round1_final_top1": boot(ok1, n),
                 "final_top1": boot(ok2, n), "difference": boot(ok1, n, ok2),
                 "trips_better": int((ok2 > ok1).sum()), "trips_worse": int((ok2 < ok1).sum()),
                 "trips_equal": int((ok2 == ok1).sum()),
                 "worst_trip_final": {"trip": b.final_top1.idxmin(), "top1": float(b.final_top1.min())}}
    print(part, json.dumps(out[part]), flush=True)

# ---- label sensitivity + independent recomputation + leakage check on student-test
P1 = Predictor(config=ROOT / "configs/round1_config.json")
P2 = Predictor()
tmp = Path(tempfile.mkdtemp(prefix="pathsnap_noroute_"))
tot = {k: 0 for k in ["n_geo", "n_dir", "r1_geo", "r1_dir", "f_geo", "f_dir", "labels_changed"]}
identical = True
for t in list_trips("student_test"):
    d = tmp / "noroute" / t
    d.mkdir(parents=True)
    for e in ["track", "nodes", "arcs"]:
        shutil.copy(DEFAULT_DATA_ROOT / "student_test" / t / f"{t}.{e}", d / f"{t}.{e}")
    a1 = P1.predict_trip(t, "student_test")
    a2 = P2.predict_trip(t, "student_test")
    b2 = P2.predict_trip(t, "noroute", tmp)
    identical &= all(np.array_equal(a2[k], b2[k]) for k in ["cls_pred", "hmm_pred", "final_pred", "route"])
    tr = a2["trip"]
    ref = load_route(t, "student_test")
    A, Bn = tr.node_xy[tr.arcs[ref, 0]], tr.node_xy[tr.arcs[ref, 1]]
    av = Bn - A
    avn = np.maximum(np.linalg.norm(av, axis=1), 1e-9)
    n = tr.n_obs
    i0, i1 = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    v = tr.xy[i1] - tr.xy[i0]
    sp = np.linalg.norm(v, axis=1) / np.maximum(tr.track[i1, 2] - tr.track[i0, 2], 1e-6)

    def extra(i):
        if sp[i] <= 3:
            return 0.0
        cos = (av @ v[i]) / (avn * max(np.linalg.norm(v[i]), 1e-9))
        return 40.0 * (cos < -0.5)
    ig, dg = monotone_align(tr.xy, A, Bn)
    idd, dd = monotone_align(tr.xy, A, Bn, extra)
    yg, yd, rg, rd = ref[ig], ref[idd], dg <= 60, dd <= 60
    tot["n_geo"] += int(rg.sum()); tot["n_dir"] += int(rd.sum())
    tot["labels_changed"] += int((yg != yd).sum())
    tot["r1_geo"] += int((a1["final_pred"] == yg)[rg].sum()); tot["r1_dir"] += int((a1["final_pred"] == yd)[rd].sum())
    tot["f_geo"] += int((a2["final_pred"] == yg)[rg].sum()); tot["f_dir"] += int((a2["final_pred"] == yd)[rd].sum())
shutil.rmtree(tmp, ignore_errors=True)
out["label_sensitivity_student_test"] = {
    "labels_changed_by_direction_term": tot["labels_changed"], "scored_fixes_geometry_only": tot["n_geo"],
    "scored_fixes_direction_aware": tot["n_dir"],
    "round1_final_top1": {"geometry_only_labels": tot["r1_geo"] / tot["n_geo"], "direction_aware_labels": tot["r1_dir"] / tot["n_dir"]},
    "final_top1": {"geometry_only_labels": tot["f_geo"] / tot["n_geo"], "direction_aware_labels": tot["f_dir"] / tot["n_dir"]}}
out["independent_recomputation"] = {"final_top1_from_pipeline": tot["f_geo"] / tot["n_geo"],
                                    "final_top1_from_tuning_csv": out["student_test"]["final_top1"]["point"],
                                    "match": bool(abs(tot["f_geo"] / tot["n_geo"] - out["student_test"]["final_top1"]["point"]) < 1e-9)}
out["predictions_identical_without_route_files"] = bool(identical)
json.dump(out, open(RES / "validation_final.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["label_sensitivity_student_test", "independent_recomputation", "predictions_identical_without_route_files"]}, indent=1))
