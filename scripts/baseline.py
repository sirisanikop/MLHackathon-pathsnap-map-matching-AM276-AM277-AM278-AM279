"""Geometric (non-learned) baseline: nearest segment.

Score = exp(-0.5 (d / sigma)^2), sigma = 10 m (a typical GPS error scale);
top-1 = nearest arc. The score is renormalised per observation so the same
observation metrics (and the HMM decoder) can be applied. Not counted as one
of the trained models. Same CV folds / strides / student-test protocol.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402
from pathsnap.dataset import load_partition  # noqa: E402
from pathsnap.metrics import (candidate_metrics, continuity, normalise_scores,  # noqa: E402
                              observation_metrics, top1)

RES = ROOT / "results"
SIGMA = 10.0


def run(c, o):
    c = c.copy()
    # tiny rank tie-break so exact ties (two-way arcs) resolve deterministically, like "nearest"
    c["p"] = np.exp(-0.5 * (c.d / SIGMA) ** 2) * (1 - 1e-6 * c.d_rank)
    c["pn"] = normalise_scores(c)
    pred = top1(c)
    om, pt = observation_metrics(c, o, pred)
    return c, pred, {**om, **candidate_metrics(c)}, pt


tr_c, tr_o = load_partition("train")
folds = json.load(open(RES / "folds.json"))
tr_c["fold"] = tr_c.trip.map(folds)
tr_o["fold"] = tr_o.trip.map(folds)
rows = []
for k in range(5):
    vc = tr_c[(tr_c.fold == k) & (tr_c.obs % 3 == 0)]
    vo = tr_o[(tr_o.fold == k) & (tr_o.obs % 3 == 0)]
    _, _, m, _ = run(vc, vo)
    rows.append({"fold": k, **m})
cv = pd.DataFrame(rows)
te_c, te_o = load_partition("student_test")
t0 = time.time()
c2, pred, met, pt = run(te_c, te_o)
inf = time.time() - t0
conts = []
for tid, g in pred.groupby("trip"):
    arcs = np.loadtxt(DEFAULT_DATA_ROOT / "student_test" / tid / f"{tid}.arcs", dtype=np.int64)
    conts.append(continuity(g.sort_values("obs").pred.to_numpy(), arcs))
    pt.loc[pt.trip == tid, "continuity"] = conts[-1]
met["route_continuity"] = float(np.mean(conts))
met["infer_time_s"] = inf
out = {"model": "baseline", "config": {"family": "Geometric heuristic (not trained)", "sigma_m": SIGMA,
                                        "rule": "nearest arc by point-segment distance"},
       "cv_mean": cv.drop(columns="fold").mean().to_dict(), "cv_std": cv.drop(columns="fold").std().to_dict(),
       "student_test": met}
json.dump(out, open(RES / "baseline_nearest.json", "w"), indent=1, default=float)
pt.to_csv(RES / "test_pertrip_baseline.csv", index=False)
c2[["trip", "obs", "arc", "d", "y", "reliable", "p", "pn"]].to_pickle(RES / "test_scores_baseline.pkl")
print("CV top1 %.4f macroF1 %.4f | test top1 %.4f macroF1 %.4f cont %.4f" % (
    out["cv_mean"]["obs_top1"], out["cv_mean"]["obs_macro_f1_trip"], met["obs_top1"], met["obs_macro_f1_trip"],
    met["route_continuity"]))
