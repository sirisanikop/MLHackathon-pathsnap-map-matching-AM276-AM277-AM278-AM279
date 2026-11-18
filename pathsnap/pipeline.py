"""End-to-end, route-free prediction pipeline (raw trip files -> matched route).

Used by the FastAPI server. Nothing here imports labels.py or reads .route.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .candidates import V2, build_candidates
from .data import load_trip
from .dataset import X_of
from .hmm import HMMParams, decode, realign_to_route, reconstruct_route
from .metrics import normalise_scores

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, model_dir=ROOT / "models", config=None):
        cfg = json.load(open(config or Path(model_dir) / "final_config.json"))
        self.cfg = cfg
        self.model = joblib.load(Path(model_dir) / f"{cfg['model']}.joblib")
        self.hmm = HMMParams(**cfg["hmm_params"])
        self.realign = bool(cfg.get("realign", True))
        self.v2 = cfg.get("feature_set", "v1") == "v2"

    def score(self, cands):
        cands = cands.copy()
        if not len(cands):
            cands["p"], cands["pn"] = [], []
        elif hasattr(self.model, "score_candidates"):      # round-2 scorer (feature list + normalisation inside)
            cands["p"], cands["pn"] = self.model.score_candidates(cands)
        else:
            cands["p"] = self.model.predict_proba(X_of(cands))[:, 1]
            cands["pn"] = normalise_scores(cands)
        return cands

    def predict_trip(self, trip_id, partition, data_root=None, use_hmm=True):
        timings = {}
        t0 = time.time()
        trip = load_trip(trip_id, partition, data_root, with_route=False)
        timings["load_s"] = time.time() - t0
        t0 = time.time()
        if trip.n_obs == 0:
            empty = np.zeros(0, dtype=np.int64)
            timings["total_s"] = timings["load_s"]
            return dict(trip=trip, cands=None, cls_pred=empty, hmm_pred=empty, final_pred=empty, segments=empty,
                        route=empty, graph=None, fill_failures=0, timings=timings)
        cands = build_candidates(trip, v2=True, **V2) if self.v2 else build_candidates(trip)
        timings["candidates_features_s"] = time.time() - t0
        t0 = time.time()
        cands = self.score(cands)
        timings["classifier_s"] = time.time() - t0
        n = trip.n_obs
        cls = np.full(n, -1, dtype=np.int64)
        if len(cands):
            best = cands.loc[cands.groupby("obs").pn.idxmax()]
            cls[best.obs.to_numpy()] = best.arc.to_numpy()
        out = dict(trip=trip, cands=cands, cls_pred=cls, timings=timings)
        if use_hmm and len(cands) == 0:      # no candidate arc for any fix: nothing can be matched
            none = np.full(n, -1, dtype=np.int64)
            out.update(hmm_pred=none, final_pred=none.copy(), segments=none.copy(), route=np.zeros(0, dtype=np.int64),
                       graph=None, fill_failures=0)
        elif use_hmm:
            t0 = time.time()
            pred, seg, g = decode(trip, cands, self.hmm)
            route, nfail = reconstruct_route(pred, seg, g)
            timings["hmm_s"] = time.time() - t0
            t0 = time.time()
            final = realign_to_route(trip, pred, route, self.hmm.realign_max_m) if self.realign else pred
            timings["realign_s"] = time.time() - t0
            out.update(hmm_pred=pred, final_pred=final, segments=seg, route=route, graph=g, fill_failures=nfail)
        timings["total_s"] = sum(v for k, v in timings.items() if k != "total_s")
        return out
