"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model
(LightGBM), using the same 5 trip-grouped folds on the 80 training trips.

Objective (maximised): mean CV observation-level top-1 accuracy (validation
observations with stride 3, reliable labels only) -- the primary selection
metric. Each trial also logs CV macro-F1 and candidate log loss.

To keep each trial affordable, training rows inside the search use stride 6
(every 6th fix of each training trip); the refit of the best configuration
uses the standard stride-3 pipeline (scripts/train_models.py lgbm_tuned).

outputs: results/optuna_trials.csv, results/optuna_best.json, results/optuna_study.pkl
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.dataset import X_of, load_partition  # noqa: E402
from pathsnap.metrics import normalise_scores, observation_metrics, top1, candidate_metrics  # noqa: E402
from pathsnap.models import CONFIGS, SEED, LGBMWrap  # noqa: E402

RES = ROOT / "results"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 2700
SEARCH_STRIDE = 6
JOBS = 4

SPACE = {
    "n_estimators": "int [100, 800] step 50",
    "learning_rate": "log-uniform [0.01, 0.2]",
    "num_leaves": "int [15, 255] (log)",
    "min_child_samples": "int [10, 400] (log)",
    "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.4, 1.0]",
    "reg_lambda": "log-uniform [1e-3, 10]",
    "reg_alpha": "log-uniform [1e-3, 10]",
}

tr_c, tr_o = load_partition("train")
folds = json.load(open(RES / "folds.json"))
tr_c["fold"] = tr_c.trip.map(folds)
tr_o["fold"] = tr_o.trip.map(folds)
SPLITS = []
for k in range(5):
    trn = tr_c[(tr_c.fold != k) & tr_c.reliable & (tr_c.obs % SEARCH_STRIDE == 0)]
    vc = tr_c[(tr_c.fold == k) & (tr_c.obs % 3 == 0)].copy()
    vo = tr_o[(tr_o.fold == k) & (tr_o.obs % 3 == 0)]
    SPLITS.append((X_of(trn), trn.y.to_numpy(), X_of(vc), vc, vo))


def cv_score(params):
    accs, f1s, lls, fit_t = [], [], [], 0.0
    for Xt, yt, Xv, vc, vo in SPLITS:
        m = LGBMWrap(n_jobs=JOBS, **params)
        t0 = time.time()
        m.m.fit(Xt, yt)
        fit_t += time.time() - t0
        vc = vc.copy()
        vc["p"] = m.predict_proba(Xv)[:, 1]
        vc["pn"] = normalise_scores(vc)
        om, _ = observation_metrics(vc, vo, top1(vc))
        cm = candidate_metrics(vc)
        accs.append(om["obs_top1"])
        f1s.append(om["obs_macro_f1_trip"])
        lls.append(cm["cand_logloss"])
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.mean(lls)), fit_t


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    )
    acc, sd, f1, ll, ft = cv_score(params)
    trial.set_user_attr("cv_top1_std", sd)
    trial.set_user_attr("cv_macro_f1", f1)
    trial.set_user_attr("cv_cand_logloss", ll)
    trial.set_user_attr("fit_time_s", ft)
    print(f"trial {trial.number}: top1={acc:.4f} f1={f1:.4f} ll={ll:.4f} t={ft:.0f}s {params}", flush=True)
    return acc


if __name__ == "__main__":
    t0 = time.time()
    # baseline (untuned) configuration evaluated under the identical search protocol
    base = {k: v for k, v in CONFIGS["lgbm"].items() if k not in ("family", "loss")}
    b_acc, b_sd, b_f1, b_ll, b_ft = cv_score(base)
    print(f"untuned (search protocol): top1={b_acc:.4f} f1={b_f1:.4f} ll={b_ll:.4f}", flush=True)
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_lgbm_tpe")
    study.enqueue_trial({k: base[k] for k in ["n_estimators", "learning_rate", "num_leaves", "min_child_samples",
                                              "subsample", "colsample_bytree"]} | {"reg_lambda": 1e-3, "reg_alpha": 1e-3})
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    df = study.trials_dataframe()
    df.to_csv(RES / "optuna_trials.csv", index=False)
    joblib.dump(study, RES / "optuna_study.pkl")
    best = dict(study.best_params)
    best["subsample_freq"] = 1
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "best_trial": study.best_trial.number,
               "n_trials": len(study.trials), "sampler": "TPE (multivariate, seed 42, 8 startup trials)",
               "objective": "mean 5-fold trip-grouped CV observation top-1 accuracy",
               "search_space": SPACE, "search_train_stride": SEARCH_STRIDE,
               "untuned_same_protocol": {"cv_top1": b_acc, "cv_top1_std": b_sd, "cv_macro_f1": b_f1,
                                         "cv_cand_logloss": b_ll},
               "wall_time_s": time.time() - t0}, open(RES / "optuna_best.json", "w"), indent=1)
    print("BEST", study.best_value, best, "wall", time.time() - t0)
