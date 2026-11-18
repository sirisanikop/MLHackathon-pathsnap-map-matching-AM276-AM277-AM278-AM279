"""Bayesian optimisation round 2 (Optuna TPE) for the round-2 scorer: v2 candidates + v2 features,
objective family chosen in scripts/tune_v2.py. Same 5 trip-grouped folds, training trips only.

A first attempt with n_estimators up to 900 and num_leaves up to 255 on a stride-6 subsample was stopped after
2 trials because single trials took > 3 minutes; the space below (smaller capacity range, stride-9 search
subsample) keeps a trial at roughly half a minute so that TPE gets a meaningful number of trials.

usage: python scripts/bayes_opt_v2.py <objective> [n_trials] [timeout_s]
outputs: results/optuna_v2_trials.csv, results/optuna_v2_best.json, results/optuna_v2_study.pkl
"""
import json
import sys
import time
from pathlib import Path

import joblib
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tune_v2 as T  # noqa: E402  (loads the cached training tables once)
from pathsnap.candidates import FEATURES_V2  # noqa: E402

OBJ = sys.argv[1] if len(sys.argv) > 1 else "lambdarank"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
SEARCH_STRIDE = 9
SPACE = {"n_estimators": "int [100, 600] step 50", "learning_rate": "log-uniform [0.01, 0.2]",
         "num_leaves": "int [15, 127] (log)", "min_child_samples": "int [10, 400] (log)",
         "subsample": "uniform [0.5, 1.0]", "colsample_bytree": "uniform [0.3, 1.0]",
         "reg_lambda": "log-uniform [1e-3, 10]", "reg_alpha": "log-uniform [1e-3, 10]"}
if OBJ == "lambdarank":
    SPACE["lambdarank_truncation_level"] = "int [1, 12]"


def objective(trial):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True))
    if OBJ == "lambdarank":
        p["lambdarank_truncation_level"] = trial.suggest_int("lambdarank_truncation_level", 1, 12)
    r = T.cv(f"trial {trial.number}", "v2", FEATURES_V2, OBJ, stride=SEARCH_STRIDE, params=p, quiet=True)
    for k in ("cv_top1_std", "cv_macro_f1", "cv_obs_logloss", "wall_s"):
        trial.set_user_attr(k, r[k])
    print(f"trial {trial.number}: top1={r['cv_top1']:.4f} f1={r['cv_macro_f1']:.4f} ll={r['cv_obs_logloss']:.4f} t={r['wall_s']:.0f}s", flush=True)
    return r["cv_top1"]


if __name__ == "__main__":
    t0 = time.time()
    base = dict(T.BASE)
    b = T.cv("untuned (round-1 best params) under search protocol", "v2", FEATURES_V2, OBJ, stride=SEARCH_STRIDE, params=base, quiet=True)
    print(f"untuned: top1={b['cv_top1']:.4f} f1={b['cv_macro_f1']:.4f}", flush=True)
    study = optuna.create_study(direction="maximize", study_name="pathsnap_scorer_v2_tpe",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10, multivariate=True))
    enq = {k: base[k] for k in ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
                                "colsample_bytree", "reg_lambda", "reg_alpha"]}
    if OBJ == "lambdarank":
        enq["lambdarank_truncation_level"] = 5
    study.enqueue_trial(enq)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    study.trials_dataframe().to_csv(T.RES / "optuna_v2_trials.csv", index=False)
    joblib.dump(study, T.RES / "optuna_v2_study.pkl")
    best = dict(study.best_params)
    best["subsample_freq"] = 1
    json.dump({"objective_family": OBJ, "best_params": best, "best_cv_top1": study.best_value,
               "best_trial": study.best_trial.number, "n_trials": len(study.trials),
               "sampler": "TPE (multivariate, seed 42, 10 startup trials)",
               "objective": "mean 5-fold trip-grouped CV observation top-1 (training trips only)",
               "search_space": SPACE, "search_train_stride": SEARCH_STRIDE,
               "untuned_same_protocol": {"params": base, "cv_top1": b["cv_top1"], "cv_macro_f1": b["cv_macro_f1"]},
               "wall_time_s": time.time() - t0}, open(T.RES / "optuna_v2_best.json", "w"), indent=1)
    print("BEST", study.best_value, best, "wall", time.time() - t0, flush=True)
