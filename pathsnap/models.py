"""Model zoo: six candidate scorers from different algorithm families.

Every model is a binary classifier over (observation, candidate-arc) rows:
y=1 if the candidate is the reference arc. Probabilities are renormalised
within each observation's candidate set (metrics.normalise_scores).
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# training-row subsampling: every k-th observation of each training trip
TRAIN_STRIDE = {"logreg": 3, "gnb": 3, "knn": 9, "rf": 3, "lgbm": 3, "mlp": 3, "lgbm_tuned": 3}

CONFIGS = {
    "logreg": dict(family="Linear / probabilistic", C=1.0, solver="lbfgs", max_iter=1000,
                   scaler="StandardScaler", loss="binary cross-entropy + L2"),
    "gnb": dict(family="Probabilistic (generative)", var_smoothing=1e-9,
                loss="Gaussian class-conditional maximum likelihood"),
    "knn": dict(family="Instance-based", n_neighbors=50, weights="distance", metric="euclidean",
                scaler="StandardScaler", loss="none (lazy learner; vote share as probability)"),
    "rf": dict(family="Ensemble (bagging)", n_estimators=200, max_depth=20, min_samples_leaf=3,
               max_features="sqrt", loss="Gini impurity"),
    "lgbm": dict(family="Boosting", n_estimators=400, learning_rate=0.05, num_leaves=63,
                 min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_lambda=0.0, loss="binary log loss"),
    "mlp": dict(family="Neural network", hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", learning_rate_init=1e-3, alpha=1e-4, batch_size=1024, epochs=20,
                scaler="StandardScaler", loss="binary cross-entropy + L2"),
}

NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "KNN (distance-weighted)",
         "rf": "Random Forest", "lgbm": "LightGBM (GBDT)", "mlp": "MLP (64-32)",
         "lgbm_tuned": "LightGBM (Bayesian-tuned)"}


def get_config(name):
    if name == "lgbm_tuned":
        import json
        from pathlib import Path
        best = json.load(open(Path(__file__).resolve().parents[1] / "results" / "optuna_best.json"))
        return dict(family="Boosting", loss="binary log loss", **best["best_params"])
    return dict(CONFIGS[name])


class EpochMLP:
    """sklearn MLP trained epoch-by-epoch with partial_fit so that genuine
    per-epoch train/validation log-loss histories can be recorded."""

    def __init__(self, n_jobs=1, **cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.net = MLPClassifier(hidden_layer_sizes=cfg["hidden_layer_sizes"], activation=cfg["activation"],
                                 solver="adam", learning_rate_init=cfg["learning_rate_init"],
                                 alpha=cfg["alpha"], batch_size=cfg["batch_size"], random_state=SEED)
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(SEED)
        Xs = self.scaler.fit_transform(X)
        Xv = self.scaler.transform(X_val) if X_val is not None else None
        mon = rng.choice(len(Xs), size=min(100_000, len(Xs)), replace=False)
        for ep in range(self.cfg["epochs"]):
            perm = rng.permutation(len(Xs))
            self.net.partial_fit(Xs[perm], y[perm], classes=np.array([0, 1]))
            rec = {"epoch": ep + 1,
                   "train_logloss": log_loss(y[mon], self.net.predict_proba(Xs[mon])[:, 1], labels=[0, 1])}
            if Xv is not None:
                rec["val_logloss"] = log_loss(y_val, self.net.predict_proba(Xv)[:, 1], labels=[0, 1])
            self.history.append(rec)
        return self

    def predict_proba(self, X):
        return self.net.predict_proba(self.scaler.transform(X))


class LGBMWrap:
    def __init__(self, n_jobs=4, **cfg):
        import lightgbm as lgb
        p = {k: v for k, v in cfg.items() if k not in ("family", "loss")}
        self.m = lgb.LGBMClassifier(objective="binary", random_state=SEED, n_jobs=n_jobs, verbose=-1, **p)
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        es = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
        names = ["train"] + (["val"] if X_val is not None else [])
        # monitor train on a subsample for speed
        if len(X) > 150_000:
            rng = np.random.default_rng(SEED)
            s = rng.choice(len(X), 150_000, replace=False)
            es[0] = (X[s], y[s])
        self.m.fit(X, y, eval_set=es, eval_names=names, eval_metric="binary_logloss")
        r = self.m.evals_result_
        self.history = [{"epoch": i + 1, "train_logloss": r["train"]["binary_logloss"][i],
                         **({"val_logloss": r["val"]["binary_logloss"][i]} if "val" in r else {})}
                        for i in range(len(r["train"]["binary_logloss"]))]
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make_model(name, n_jobs=4, cfg=None):
    c = get_config(name) if cfg is None else cfg
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=c["C"], max_iter=c["max_iter"],
                                                                  solver=c["solver"], random_state=SEED))
    if name == "gnb":
        return GaussianNB(var_smoothing=c["var_smoothing"])
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=c["n_neighbors"],
                                                                    weights=c["weights"], n_jobs=n_jobs))
    if name == "rf":
        return RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                                      min_samples_leaf=c["min_samples_leaf"], max_features=c["max_features"],
                                      n_jobs=n_jobs, random_state=SEED)
    if name in ("lgbm", "lgbm_tuned"):
        return LGBMWrap(n_jobs=n_jobs, **c)
    if name == "mlp":
        return EpochMLP(**c)
    raise KeyError(name)


def fit_model(model, X, y, X_val=None, y_val=None):
    t0 = time.time()
    if isinstance(model, (EpochMLP, LGBMWrap)):
        model.fit(X, y, X_val, y_val)
    else:
        model.fit(X, y)
    return time.time() - t0


class CandidateScorer:
    """Final (round-2) scorer: LightGBM model + feature list + how raw outputs become a distribution over
    a fix's candidates. kind='binary': predict_proba renormalised per fix; kind='rank': softmax of ranking
    scores per fix with temperature T (T fitted on training out-of-fold scores only)."""

    def __init__(self, model, feats, kind, T=1.0, params=None):
        self.model, self.feats, self.kind, self.T, self.params = model, list(feats), kind, float(T), params or {}

    def raw(self, cands):
        X = cands[self.feats].to_numpy(np.float32)
        return self.model.predict_proba(X)[:, 1] if self.kind == "binary" else self.model.predict(X)

    def score_candidates(self, cands):
        """Return (raw score, normalised probability within each fix) for a candidate table."""
        import pandas as pd
        raw = self.raw(cands)
        key = [cands.trip.to_numpy(), cands.obs.to_numpy()]
        if self.kind == "binary":
            s = pd.Series(np.clip(raw, 1e-6, 1.0))
            return raw, (s / s.groupby(key).transform("sum")).to_numpy()
        z = pd.Series(raw / self.T)
        e = np.exp(z - z.groupby(key).transform("max"))
        return raw, (e / e.groupby(key).transform("sum")).to_numpy()
