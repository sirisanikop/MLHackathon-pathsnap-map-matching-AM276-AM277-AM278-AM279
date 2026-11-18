"""Train + evaluate one model: 5-fold trip-grouped CV on the 80 training trips,
then refit on all training trips and score the 10 student-test trips.

usage: python scripts/train_models.py <model> [--jobs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402
from pathsnap.dataset import CACHE, CACHE_V2, FEATURE_SETS, X_of as _X_of, load_partition as _load_partition  # noqa: E402
from pathsnap.metrics import (candidate_metrics, continuity, normalise_scores,  # noqa: E402
                              observation_metrics, top1)
from pathsnap.models import SEED, TRAIN_STRIDE, fit_model, get_config, make_model  # noqa: E402

RES = ROOT / "results"
N_FOLDS = 5
# --fs v2 : same protocol on the round-2 candidate/feature table (cache_v2, 46 features); outputs get the
#           suffix "_v2feat" and models are not saved (benchmark only)
FS = "v2" if "--fs" in sys.argv and sys.argv[sys.argv.index("--fs") + 1] == "v2" else "v1"
SUF = "_v2feat" if FS == "v2" else ""


def X_of(df):
    return _X_of(df, FEATURE_SETS[FS])


def load_partition(part):
    return _load_partition(part, cache=CACHE_V2 if FS == "v2" else CACHE)


def get_folds(trips):
    f = RES / "folds.json"
    if f.exists():
        return json.load(open(f))
    rng = np.random.default_rng(SEED)
    perm = list(rng.permutation(sorted(trips)))
    folds = {t: int(i % N_FOLDS) for i, t in enumerate(perm)}
    json.dump(folds, open(f, "w"), indent=1)
    return folds


def train_rows(c, stride):
    return c[(c.reliable) & (c.obs % stride == 0)]


def score(model, cands):
    t0 = time.time()
    p = model.predict_proba(X_of(cands))[:, 1]
    return p, time.time() - t0


def evaluate(cands, obs, p):
    cands = cands.copy()
    cands["p"] = p
    cands["pn"] = normalise_scores(cands)
    pred = top1(cands)
    om, pt = observation_metrics(cands, obs, pred)
    cm = candidate_metrics(cands)
    return cands, pred, {**om, **cm}, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fs", default="v1")
    args = ap.parse_args()
    name = args.model
    stride = TRAIN_STRIDE[name]
    out = {"model": name, "config": get_config(name), "seed": SEED, "train_stride": stride,
           "cv_eval_stride": 3}
    tr_c, tr_o = load_partition("train")
    folds = get_folds(tr_c.trip.unique())
    tr_c["fold"] = tr_c.trip.map(folds)
    tr_o["fold"] = tr_o.trip.map(folds)

    # ---------------- trip-grouped cross-validation ----------------
    fold_rows, hist = [], []
    for k in range(N_FOLDS):
        trn = train_rows(tr_c[tr_c.fold != k], stride)
        val_c = tr_c[(tr_c.fold == k) & (tr_c.obs % 3 == 0)]
        val_o = tr_o[(tr_o.fold == k) & (tr_o.obs % 3 == 0)]
        m = make_model(name, n_jobs=args.jobs)
        vrel = val_c[val_c.reliable]
        fit_t = fit_model(m, X_of(trn), trn.y.to_numpy(), X_of(vrel), vrel.y.to_numpy())
        p, inf_t = score(m, val_c)
        _, _, met, _ = evaluate(val_c, val_o, p)
        met.update(fold=k, fit_time_s=fit_t, infer_time_s=inf_t, n_train_rows=len(trn),
                   n_val_trips=val_o.trip.nunique())
        # in-sample (training) metrics on the same fold's training trips (overfitting evidence)
        trn_c = tr_c[(tr_c.fold != k) & (tr_c.obs % 3 == 0)]
        trn_o = tr_o[(tr_o.fold != k) & (tr_o.obs % 3 == 0)]
        if name != "knn":  # KNN in-sample = memorised; computed below on a subsample
            ptr, _ = score(m, trn_c)
            _, _, mtr, _ = evaluate(trn_c, trn_o, ptr)
            met.update(train_obs_top1=mtr["obs_top1"], train_obs_macro_f1=mtr["obs_macro_f1_trip"],
                       train_cand_logloss=mtr["cand_logloss"])
        else:
            sub_t = trn_c.trip.unique()[:12]
            ptr, _ = score(m, trn_c[trn_c.trip.isin(sub_t)])
            _, _, mtr, _ = evaluate(trn_c[trn_c.trip.isin(sub_t)], trn_o[trn_o.trip.isin(sub_t)], ptr)
            met.update(train_obs_top1=mtr["obs_top1"], train_obs_macro_f1=mtr["obs_macro_f1_trip"],
                       train_cand_logloss=mtr["cand_logloss"])
        if getattr(m, "history", None):
            for h in m.history:
                hist.append({"fold": k, **h})
        fold_rows.append(met)
        print(name, "fold", k, {kk: round(v, 4) for kk, v in met.items() if isinstance(v, float)}, flush=True)
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(RES / f"cv_{name}{SUF}.csv", index=False)
    if hist:
        pd.DataFrame(hist).to_csv(RES / f"history_{name}{SUF}.csv", index=False)
    out["cv_mean"] = cv.drop(columns=["fold"]).mean().to_dict()
    out["cv_std"] = cv.drop(columns=["fold"]).std().to_dict()

    # ---------------- refit on all 80 training trips ----------------
    trn = train_rows(tr_c, stride)
    m = make_model(name, n_jobs=args.jobs)
    fit_t = fit_model(m, X_of(trn), trn.y.to_numpy())
    if FS == "v1":
        joblib.dump(m, ROOT / "models" / f"{name}.joblib", compress=3)
    out["final_fit_time_s"] = fit_t
    out["final_train_rows"] = len(trn)

    # ---------------- student-test evaluation (all observations) ----------------
    te_c, te_o = load_partition("student_test")
    p, inf_t = score(m, te_c)
    te_c2, pred, met, pt = evaluate(te_c, te_o, p)
    conts = []
    for tid, g in pred.groupby("trip"):
        arcs = np.loadtxt(DEFAULT_DATA_ROOT / "student_test" / tid / f"{tid}.arcs", dtype=np.int64)
        seq = g.sort_values("obs").pred.to_numpy()
        conts.append(continuity(seq, arcs))
        pt.loc[pt.trip == tid, "continuity"] = conts[-1]
    met["route_continuity"] = float(np.mean(conts))
    met["infer_time_s"] = inf_t
    met["infer_ms_per_obs"] = 1000 * inf_t / te_o.shape[0]
    out["student_test"] = met
    pt.to_csv(RES / f"test_pertrip_{name}{SUF}.csv", index=False)
    if FS == "v1":
        te_c2[["trip", "obs", "arc", "d", "y", "reliable", "p", "pn"]].to_pickle(RES / f"test_scores_{name}.pkl")
    out["feature_set"] = FS
    json.dump(out, open(RES / f"model_{name}{SUF}.json", "w"), indent=1, default=float)
    print(name, "TEST", {k: round(v, 4) for k, v in met.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
