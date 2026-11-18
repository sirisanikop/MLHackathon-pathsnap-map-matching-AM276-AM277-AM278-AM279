"""PathSnap FastAPI backend (Laptop 1).

Run:
    PATHSNAP_DATA="/path/to/data_root" PATHSNAP_PARTITION=student_test \
    uvicorn api.server:app --host 0.0.0.0 --port 8000

PATHSNAP_DATA       folder that contains the partition folder(s)
PATHSNAP_PARTITION  sub-folder holding the trips offered to the evaluator UI
                    (e.g. the ten reserved evaluator trips). Trip IDs are read
                    from the folder at request time -- nothing is hardcoded.

Endpoints
    GET  /health
    GET  /trips              -> trip IDs available for evaluation
    POST /predict {trip_id}  -> predicted route + server-side evaluation scores
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.evaluation import evaluate_prediction  # noqa: E402
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT))
PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")

(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / "api_requests.log")):
    h.setFormatter(_fmt)
    log.addHandler(h)

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor()
log.info("model loaded: %s | hmm=%s | realign=%s | partition=%s",
         predictor.cfg["model"], predictor.cfg["hmm_params"], predictor.realign, PARTITION)


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    d = DATA_ROOT / PARTITION
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d in %.0f ms", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg.get("display_name", predictor.cfg["model"]), "partition": PARTITION}


@app.get("/trips")
def trips():
    t = available_trips()
    return {"partition": PARTITION, "count": len(t), "trips": t}


def _latlon(trip, xy_lonlat):
    return [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in xy_lonlat]


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in available_trips():
        raise HTTPException(404, f"unknown trip id {tid!r}")
    log.info("PREDICT request trip=%s", tid)
    # ---- 1. prediction (route-free) ----
    out = predictor.predict_trip(tid, PARTITION, DATA_ROOT, use_hmm=True)
    trip, route, hmm_pred, final_pred = out["trip"], out["route"], out["hmm_pred"], out["final_pred"]
    nodes = trip.nodes
    route_lines = [[[float(nodes[trip.arcs[a, 0], 1]), float(nodes[trip.arcs[a, 0], 0])],
                    [float(nodes[trip.arcs[a, 1], 1]), float(nodes[trip.arcs[a, 1], 0])]] for a in route]
    step = max(1, trip.n_obs // 1500)
    gps = _latlon(trip, trip.track[::step, :2])
    # ---- 2. separate server-side evaluation against the private reference ----
    t0 = time.time()
    ev = evaluate_prediction(trip, out["cls_pred"], hmm_pred, final_pred, route, out["graph"], PARTITION, DATA_ROOT)
    out["timings"]["evaluation_s"] = time.time() - t0
    st = ev.get("stages", {})
    acc = lambda k: f"{st[k]['observation_top1_accuracy']:.4f}" if k in st and st[k]["observation_top1_accuracy"] is not None else "n/a"  # noqa: E731
    log.info("PREDICT done trip=%s obs=%d route_arcs=%d acc_final=%s acc_hmm=%s acc_cls=%s total=%.2fs",
             tid, trip.n_obs, len(route), acc("final"), acc("hmm"), acc("classifier_only"), out["timings"]["total_s"])
    return {
        "trip_id": tid,
        "model": predictor.cfg.get("display_name", predictor.cfg["model"]),
        "n_observations": trip.n_obs,
        "n_matched_observations": int((final_pred >= 0).sum()),
        "n_route_arcs": int(len(route)),
        "n_hmm_segments": int(out["segments"].max() + 1) if len(out["segments"]) else 0,
        "predicted_route_arc_ids": [int(a) for a in route],
        "predicted_route_lines": route_lines,
        "gps_points": gps,
        "gps_display_stride": step,
        "timings_s": {k: round(v, 3) for k, v in out["timings"].items()},
        "evaluation": ev,
    }
