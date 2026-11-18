"""Decoder post-step experiment: Viterbi states vs Viterbi + monotone re-alignment to the PREDICTED route.

Protocol: identical candidates, classifier scores (tuned LightGBM), decoder parameters and labels.
Selection evidence = 80 training trips with out-of-fold scores; student-test is reported afterwards.
outputs: results/realign_train_oof.csv, results/realign_test.csv, results/realign_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_route, load_trip  # noqa: E402
from pathsnap.hmm import HMMParams, decode, realign_to_route, reconstruct_route  # noqa: E402
from pathsnap.metrics import continuity, route_overlap  # noqa: E402

RES, CACHE = ROOT / "results", ROOT / "cache"
CFG = json.load(open(ROOT / "configs/round1_config.json"))   # round-1 system (v1 features, tuned LightGBM)
HP = HMMParams(**{k: v for k, v in CFG["hmm_params"].items() if k in ("beta", "w_emit")})


def one(tid, part, c):
    tr = load_trip(tid, part, with_route=False)
    f = pd.read_pickle(CACHE / f"feat_{part}_{tid}.pkl")[["obs", "arc", "t_proj"]]
    c = c.merge(f, on=["obs", "arc"])
    lab = np.load(CACHE / f"lab_{part}_{tid}.npz")
    y, rel = lab["lab"], lab["reliable"]
    cls = np.full(tr.n_obs, -1, dtype=np.int64)
    b = c.loc[c.groupby("obs").pn.idxmax()]
    cls[b.obs.to_numpy()] = b.arc.to_numpy()
    hmm, seg, g = decode(tr, c, HP)
    route, _ = reconstruct_route(hmm, seg, g)
    re = realign_to_route(tr, hmm, route)
    ref = load_route(tid, part)  # evaluation only
    out = dict(trip=tid, n_obs=tr.n_obs, n_reliable=int(rel.sum()))
    for name, p in [("cls", cls), ("hmm", hmm), ("realign", re)]:
        out[f"{name}_top1"] = float((p == y)[rel].mean())
        out[f"{name}_macro_f1"] = float(f1_score(y[rel], p[rel], average="macro", zero_division=0))
        out[f"{name}_continuity"] = continuity(p, tr.arcs)
    out["route_f1"] = route_overlap(route, ref, g.alen)["route_f1"]
    return out


def summarise(d):
    w = d.n_reliable
    s = {}
    for name in ["cls", "hmm", "realign"]:
        s[name] = dict(top1=float(np.average(d[f"{name}_top1"], weights=w)), macro_f1=float(d[f"{name}_macro_f1"].mean()),
                       continuity=float(d[f"{name}_continuity"].mean()))
    s["trips_realign_worse_top1"] = int((d.realign_top1 < d.hmm_top1 - 1e-12).sum())
    s["trips_realign_better_top1"] = int((d.realign_top1 > d.hmm_top1 + 1e-12).sum())
    s["route_f1"] = float(d.route_f1.mean())
    return s


if __name__ == "__main__":
    out = {"model": CFG["model"], "hmm_params": CFG["hmm_params"]}
    for part, f in [("train", f"oof_scores_{CFG['model']}.pkl"), ("student_test", f"test_scores_{CFG['model']}.pkl")]:
        sc = pd.read_pickle(RES / f)
        rows = Parallel(n_jobs=4)(delayed(one)(t, part, g) for t, g in sc.groupby("trip"))
        d = pd.DataFrame(rows)
        d.to_csv(RES / ("realign_train_oof.csv" if part == "train" else "realign_test.csv"), index=False)
        out[part] = summarise(d)
        print(part, json.dumps(out[part], indent=None), flush=True)
    json.dump(out, open(RES / "realign_summary.json", "w"), indent=1)
