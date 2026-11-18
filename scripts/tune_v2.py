"""Second tuning round for the selected family (LightGBM): candidate set v2, feature set v2,
training objective and training-set size, compared on the SAME 5 trip-grouped folds of the 80
training trips as every earlier experiment. Student-test is not touched here.

usage: python scripts/tune_v2.py            -> results/tune_v2_variants.csv
Metric definitions are those of pathsnap/metrics.py (observation top-1 counts candidate misses as errors).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import FEATURES, FEATURES_V2  # noqa: E402
from pathsnap.dataset import CACHE, CACHE_V2, load_partition  # noqa: E402
from pathsnap.metrics import normalise_scores, observation_metrics, top1  # noqa: E402
from pathsnap.models import SEED, get_config  # noqa: E402

RES = ROOT / "results"
JOBS = 4
BASE = {k: v for k, v in get_config("lgbm_tuned").items() if k not in ("family", "loss")}


def softmax_group(c, score, T=1.0):
    z = score / T
    z = z - pd.Series(z).groupby([c.trip.to_numpy(), c.obs.to_numpy()]).transform("max").to_numpy()
    e = np.exp(z)
    return e / pd.Series(e).groupby([c.trip.to_numpy(), c.obs.to_numpy()]).transform("sum").to_numpy()


def fit_predict(objective, params, trn, val, feats):
    Xt, yt = trn[feats].to_numpy(np.float32), trn.y.to_numpy()
    Xv = val[feats].to_numpy(np.float32)
    if objective == "binary":
        m = lgb.LGBMClassifier(objective="binary", random_state=SEED, n_jobs=JOBS, verbose=-1, **params)
        m.fit(Xt, yt)
        v = val.copy()
        v["p"] = m.predict_proba(Xv)[:, 1]
        v["pn"] = normalise_scores(v)
        return v, m
    grp = trn.groupby(["trip", "obs"], sort=False).size().to_numpy()
    params = dict(params)
    trunc = params.pop("lambdarank_truncation_level", 5)
    m = lgb.LGBMRanker(objective=objective, random_state=SEED, n_jobs=JOBS, verbose=-1,
                       lambdarank_truncation_level=trunc, **params)
    m.fit(Xt, yt, group=grp)
    v = val.copy()
    v["p"] = m.predict(Xv)
    v["pn"] = softmax_group(v, v["p"].to_numpy())
    return v, m


def cv(name, cache, feats, objective="binary", stride=3, params=None, keep_oof=False, quiet=False):
    params = params or BASE
    c, o = DATA[cache]
    rows, t0, oof = [], time.time(), []
    for k in range(5):
        trn = c[(c.fold != k) & c.reliable & (c.obs % stride == 0)]
        val = c[(c.fold == k) & (c.obs % 3 == 0)]
        vo = o[(o.fold == k) & (o.obs % 3 == 0)]
        v, _ = fit_predict(objective, params, trn, val, feats)
        om, _ = observation_metrics(v, vo, top1(v))
        rows.append(om)
        if keep_oof:
            oof.append(v)
    d = pd.DataFrame(rows)
    out = dict(variant=name, candidates=cache, n_features=len(feats), objective=objective, train_stride=stride,
               cv_top1=d.obs_top1.mean(), cv_top1_std=d.obs_top1.std(), cv_macro_f1=d.obs_macro_f1_trip.mean(),
               cv_obs_logloss=d.obs_logloss.mean(), cv_miss_rate=d.miss_rate.mean(), wall_s=time.time() - t0,
               folds_top1=[round(x, 4) for x in d.obs_top1])
    if not quiet:
        print(json.dumps(out), flush=True)
    return out


folds = json.load(open(RES / "folds.json"))
DATA = {}
for key, cache in [("v1", CACHE), ("v2", CACHE_V2)]:
    c, o = load_partition("train", cache=cache)
    c["fold"] = c.trip.map(folds)
    o["fold"] = o.trip.map(folds)
    DATA[key] = (c, o)

if __name__ == "__main__":
    R = [
        cv("A0 reference: v1 candidates, v1 features, binary", "v1", FEATURES),
        cv("A1 v2 candidates (K=24), v1 features, binary", "v2", FEATURES),
        cv("B  v2 candidates, v2 features, binary", "v2", FEATURES_V2),
        cv("C1 v2 features, lambdarank", "v2", FEATURES_V2, "lambdarank"),
        cv("C2 v2 features, rank_xendcg (listwise softmax)", "v2", FEATURES_V2, "rank_xendcg"),
    ]
    best = max(R[2:], key=lambda r: r["cv_top1"])
    R.append(cv(f"D  {best['objective']}, v2 features, all fixes (stride 1)", "v2", FEATURES_V2, best["objective"], stride=1))
    pd.DataFrame(R).to_csv(RES / "tune_v2_variants.csv", index=False)
    print(pd.DataFrame(R)[["variant", "cv_top1", "cv_top1_std", "cv_macro_f1", "cv_obs_logloss", "cv_miss_rate", "wall_s"]].round(4).to_string())
