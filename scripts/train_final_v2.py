"""Train and evaluate the round-2 scorer (v2 candidates + v2 features) with a given configuration.

usage: python scripts/train_final_v2.py <name> <objective> <train_stride> [params.json key]
  name       e.g. lgbm_v2_untuned / lgbm_v2
  params     "base" (round-1 tuned LightGBM params) or "optuna" (results/optuna_v2_best.json)

Same protocol as scripts/train_models.py: 5 trip-grouped folds (results/folds.json) on the 80 training
trips, CV metrics on every 3rd validation fix, refit on all training trips, one evaluation on the 10
student-test trips (all fixes). Additionally stores out-of-fold scores for ALL training fixes (decoder
tuning) and, for ranking objectives, a softmax temperature fitted on those OOF scores (training only).
outputs: results/model_<name>.json, cv_<name>.csv, oof_scores_<name>.pkl, test_scores_<name>.pkl,
         test_pertrip_<name>.csv, models/<name>.joblib
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tune_v2 as T  # noqa: E402
from pathsnap.candidates import FEATURES_V2  # noqa: E402
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402
from pathsnap.dataset import CACHE_V2, load_partition  # noqa: E402
from pathsnap.metrics import candidate_metrics, continuity, observation_metrics, top1  # noqa: E402
from pathsnap.models import SEED, CandidateScorer  # noqa: E402

NAME, OBJ, STRIDE = sys.argv[1], sys.argv[2], int(sys.argv[3])
PSRC = sys.argv[4] if len(sys.argv) > 4 else "base"
PARAMS = dict(T.BASE) if PSRC == "base" else dict(json.load(open(T.RES / "optuna_v2_best.json"))["promoted_params" if PSRC == "promoted" else "best_params"])
KIND = "binary" if OBJ == "binary" else "rank"
c, o = T.DATA["v2"]


def obs_ll(v, Tt):
    pn = T.softmax_group(v, v["p"].to_numpy(), Tt)
    m = (v.y.to_numpy() == 1) & v.reliable.to_numpy()
    return float(-np.log(np.clip(pn[m], 1e-6, 1)).mean())


rows, oof = [], []
for k in range(5):
    trn = c[(c.fold != k) & c.reliable & (c.obs % STRIDE == 0)]
    val = c[c.fold == k]
    t0 = time.time()
    v, m = T.fit_predict(OBJ, PARAMS, trn, val, FEATURES_V2)
    fit_t = time.time() - t0
    oof.append(v[["trip", "obs", "arc", "y", "reliable", "p", "pn", "fold"]])
    vs, vo = v[v.obs % 3 == 0], o[(o.fold == k) & (o.obs % 3 == 0)]
    om, _ = observation_metrics(vs, vo, top1(vs))
    # in-sample metrics on the training trips of this fold (over-fitting evidence)
    trc = c[(c.fold != k) & (c.obs % 3 == 0)].copy()
    sc = CandidateScorer(m, FEATURES_V2, KIND)
    trc["p"], trc["pn"] = sc.score_candidates(trc)
    omt, _ = observation_metrics(trc, o[(o.fold != k) & (o.obs % 3 == 0)], top1(trc))
    om.update(fold=k, fit_time_s=fit_t, n_train_rows=len(trn), train_obs_top1=omt["obs_top1"],
              train_obs_macro_f1=omt["obs_macro_f1_trip"])
    rows.append(om)
    print(NAME, "fold", k, {kk: round(float(vv), 4) for kk, vv in om.items() if kk in ("obs_top1", "obs_macro_f1_trip", "obs_logloss", "train_obs_top1")}, flush=True)
cv = pd.DataFrame(rows)
oof = pd.concat(oof, ignore_index=True)
Tt = 1.0
if KIND == "rank":   # temperature on OOF scores (training trips only)
    Tt = float(minimize_scalar(lambda x: obs_ll(oof, x), bounds=(0.2, 5.0), method="bounded").x)
    oof["pn"] = T.softmax_group(oof, oof["p"].to_numpy(), Tt)
    cv["obs_logloss_T1"] = cv["obs_logloss"]
    cv["obs_logloss"] = [obs_ll(oof[(oof.fold == k) & (oof.obs % 3 == 0)], Tt) for k in range(5)]
oof.drop(columns="fold").to_pickle(T.RES / f"oof_scores_{NAME}.pkl")
cv.to_csv(T.RES / f"cv_{NAME}.csv", index=False)

trn = c[c.reliable & (c.obs % STRIDE == 0)]
te_c, te_o = load_partition("student_test", cache=CACHE_V2)
t0 = time.time()
_, m = T.fit_predict(OBJ, PARAMS, trn, te_c.head(100), FEATURES_V2)
fit_t = time.time() - t0
scorer = CandidateScorer(m, FEATURES_V2, KIND, Tt, PARAMS)
joblib.dump(scorer, ROOT / "models" / f"{NAME}.joblib", compress=3)
t0 = time.time()
te_c["p"], te_c["pn"] = scorer.score_candidates(te_c)
inf_t = time.time() - t0
pred = top1(te_c)
met, pt = observation_metrics(te_c, te_o, pred)
if KIND == "binary":
    met.update(candidate_metrics(te_c))
conts = []
for tid, g in pred.groupby("trip"):
    arcs = np.loadtxt(DEFAULT_DATA_ROOT / "student_test" / tid / f"{tid}.arcs", dtype=np.int64)
    conts.append(continuity(g.sort_values("obs").pred.to_numpy(), arcs))
    pt.loc[pt.trip == tid, "continuity"] = conts[-1]
met.update(route_continuity=float(np.mean(conts)), infer_time_s=inf_t)
pt.to_csv(T.RES / f"test_pertrip_{NAME}.csv", index=False)
te_c[["trip", "obs", "arc", "d", "y", "reliable", "p", "pn"]].to_pickle(T.RES / f"test_scores_{NAME}.pkl")
num = cv.drop(columns=["fold"]).select_dtypes("number")
out = {"model": NAME, "config": {"family": "Boosting (LightGBM)", "objective": OBJ, "params_source": PSRC, **PARAMS},
       "feature_set": "v2", "n_features": len(FEATURES_V2), "candidates": "v2 (80 m, K=24)", "seed": SEED,
       "train_stride": STRIDE, "cv_eval_stride": 3, "softmax_temperature": Tt,
       "cv_mean": num.mean().to_dict(), "cv_std": num.std().to_dict(), "final_fit_time_s": fit_t,
       "final_train_rows": int(len(trn)), "student_test": met}
json.dump(out, open(T.RES / f"model_{NAME}.json", "w"), indent=1, default=float)
print(NAME, "CV top1 %.4f±%.4f mF1 %.4f LL %.4f train-top1 %.4f | TEST top1 %.4f mF1 %.4f LL %.4f cont %.4f | T=%.3f" % (
    cv.obs_top1.mean(), cv.obs_top1.std(), cv.obs_macro_f1_trip.mean(), cv.obs_logloss.mean(), cv.train_obs_top1.mean(),
    met["obs_top1"], met["obs_macro_f1_trip"], met["obs_logloss"], met["route_continuity"], Tt), flush=True)
