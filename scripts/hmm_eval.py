"""HMM/Viterbi experiments.

1. Out-of-fold (5 trip-grouped folds) probabilities of the selected model on
   ALL observations of the 80 training trips.
2. Decoder parameter grid (beta, w_emit) chosen on training-trip OOF scores
   only (student-test is never used to choose decoder parameters).
3. Classifier-only vs classifier+HMM on the 10 student-test trips, for every
   trained model (same candidates, same labels, same metrics).

usage: python scripts/hmm_eval.py <selected_model>
outputs: results/hmm_grid.csv, results/hmm_ablation_models.csv,
         results/hmm_test_pertrip.csv, results/hmm_summary.json
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_route, load_trip  # noqa: E402
from pathsnap.dataset import X_of, load_partition  # noqa: E402
from pathsnap.hmm import HMMParams, decode, reconstruct_route  # noqa: E402
from pathsnap.metrics import collapse, continuity, normalise_scores, route_overlap  # noqa: E402
from pathsnap.models import TRAIN_STRIDE, fit_model, make_model  # noqa: E402

RES = ROOT / "results"
SEL = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
CACHE = ROOT / "cache"


def trip_eval(tid, part, c, params, fill=True):
    """Return dict of classifier-only and HMM metrics for one trip."""
    tr = load_trip(tid, part, with_route=False)
    f = pd.read_pickle(CACHE / f"feat_{part}_{tid}.pkl")[["obs", "arc", "t_proj"]]
    c = c.merge(f, on=["obs", "arc"])
    lab = np.load(CACHE / f"lab_{part}_{tid}.npz")
    rel, y = lab["reliable"], lab["lab"]
    n = tr.n_obs
    cls = np.full(n, -1, dtype=np.int64)
    b = c.loc[c.groupby("obs").pn.idxmax()]
    cls[b.obs.to_numpy()] = b.arc.to_numpy()
    t0 = time.time()
    pred, seg, g = decode(tr, c, params)
    t_dec = time.time() - t0
    route, nfail = reconstruct_route(pred, seg, g, fill=fill)
    ref = load_route(tid, part)   # evaluation step only
    rc = route_overlap(collapse(cls), ref, g.alen)
    rh = route_overlap(route, ref, g.alen)
    return dict(trip=tid, n_obs=n, n_reliable=int(rel.sum()),
                cls_top1=float((cls == y)[rel].mean()), hmm_top1=float((pred == y)[rel].mean()),
                cls_continuity=continuity(cls, tr.arcs), hmm_continuity=continuity(pred, tr.arcs),
                cls_route_precision=rc["route_precision"], cls_route_recall=rc["route_recall"],
                cls_route_f1=rc["route_f1"], hmm_route_precision=rh["route_precision"],
                hmm_route_recall=rh["route_recall"], hmm_route_f1=rh["route_f1"],
                hmm_segments=int(seg.max() + 1), fill_failures=nfail,
                unmatched_obs=int((pred < 0).sum()), decode_time_s=t_dec)


def oof_scores(model_name):
    f = RES / f"oof_scores_{model_name}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    tr_c, _ = load_partition("train")
    folds = json.load(open(RES / "folds.json"))
    tr_c["fold"] = tr_c.trip.map(folds)
    parts = []
    for k in range(5):
        trn = tr_c[(tr_c.fold != k) & tr_c.reliable & (tr_c.obs % TRAIN_STRIDE[model_name] == 0)]
        m = make_model(model_name, n_jobs=4)
        fit_model(m, X_of(trn), trn.y.to_numpy())
        v = tr_c[tr_c.fold == k][["trip", "obs", "arc", "y", "reliable"]].copy()
        v["p"] = m.predict_proba(X_of(tr_c[tr_c.fold == k]))[:, 1]
        parts.append(v)
        print("oof fold", k, flush=True)
    oof = pd.concat(parts, ignore_index=True)
    oof["pn"] = normalise_scores(oof)
    oof.to_pickle(f)
    return oof


if __name__ == "__main__":
    t_all = time.time()
    oof = oof_scores(SEL)
    # ---- decoder grid on training OOF scores ----
    # initial grid {5..80} x {0.5,1,2}; its optimum lay on the boundary (beta=5, w=0.5), so the grid was
    # extended towards smaller values. Already-evaluated settings are reused from results/hmm_grid.csv.
    betas, ws = [2, 3.5, 5, 10, 20, 40, 80], [0.25, 0.35, 0.5, 1.0, 2.0]
    prev = pd.read_csv(RES / "hmm_grid.csv") if (RES / "hmm_grid.csv").exists() else pd.DataFrame()
    grid = prev.to_dict("records")
    done = {(float(r["beta"]), float(r["w_emit"])) for r in grid}
    trips = sorted(oof.trip.unique())
    groups = {t: g for t, g in oof.groupby("trip")}
    for beta, w in itertools.product(betas, ws):
        if (float(beta), float(w)) in done:
            continue
        p = HMMParams(beta=beta, w_emit=w)
        rows = Parallel(n_jobs=4)(delayed(trip_eval)(t, "train", groups[t], p, False) for t in trips)
        d = pd.DataFrame(rows)
        wt = d.n_reliable
        grid.append(dict(beta=beta, w_emit=w, hmm_top1=np.average(d.hmm_top1, weights=wt),
                         cls_top1=np.average(d.cls_top1, weights=wt),
                         hmm_continuity=d.hmm_continuity.mean(), hmm_route_f1_nofill=d.hmm_route_f1.mean()))
        print("grid", grid[-1], flush=True)
    G = pd.DataFrame(grid)
    G.to_csv(RES / "hmm_grid.csv", index=False)
    best = G.sort_values(["hmm_top1", "hmm_route_f1_nofill"], ascending=False).iloc[0]
    bp = HMMParams(beta=float(best.beta), w_emit=float(best.w_emit))
    print("best decoder", bp, flush=True)
    # per-trip effect on the training trips (OOF scores) with the selected decoder -- shows where HMM hurts
    rows = Parallel(n_jobs=4)(delayed(trip_eval)(t, "train", groups[t], bp, True) for t in trips)
    pd.DataFrame(rows).to_csv(RES / "hmm_train_oof_pertrip.csv", index=False)

    # ---- classifier-only vs HMM on student-test for every model ----
    abl, per = [], []
    for f in sorted(RES.glob("test_scores_*.pkl")):
        name = f.stem.replace("test_scores_", "")
        sc = pd.read_pickle(f)
        rows = Parallel(n_jobs=4)(delayed(trip_eval)(t, "student_test", g, bp, True)
                                  for t, g in sc.groupby("trip"))
        d = pd.DataFrame(rows)
        d.insert(0, "model", name)
        per.append(d)
        wt = d.n_reliable
        abl.append(dict(model=name, cls_top1=np.average(d.cls_top1, weights=wt),
                        hmm_top1=np.average(d.hmm_top1, weights=wt),
                        cls_continuity=d.cls_continuity.mean(), hmm_continuity=d.hmm_continuity.mean(),
                        cls_route_f1=d.cls_route_f1.mean(), hmm_route_f1=d.hmm_route_f1.mean(),
                        cls_route_precision=d.cls_route_precision.mean(), hmm_route_precision=d.hmm_route_precision.mean(),
                        cls_route_recall=d.cls_route_recall.mean(), hmm_route_recall=d.hmm_route_recall.mean(),
                        trips_hmm_hurt_top1=int((d.hmm_top1 < d.cls_top1).sum()),
                        trips_hmm_hurt_route_f1=int((d.hmm_route_f1 < d.cls_route_f1).sum()),
                        mean_decode_time_s=d.decode_time_s.mean()))
        print("ablation", abl[-1], flush=True)
    pd.DataFrame(abl).to_csv(RES / "hmm_ablation_models.csv", index=False)
    pd.concat(per).to_csv(RES / "hmm_test_pertrip.csv", index=False)
    json.dump({"selected_model": SEL, "best_params": bp.__dict__, "grid_selected_on": "training OOF (80 trips)",
               "wall_time_s": time.time() - t_all}, open(RES / "hmm_summary.json", "w"), indent=1)
