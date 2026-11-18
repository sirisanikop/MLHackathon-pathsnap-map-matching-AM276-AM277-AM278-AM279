"""Evaluation-day pre-flight check (run on laptop 1 BEFORE the evaluator arrives).

    PATHSNAP_DATA="/path/to/data_root" PATHSNAP_PARTITION=<folder with the reserved trips> python scripts/preflight.py

For the configured partition it checks, per trip: required files, shapes and ID ranges, coordinate
plausibility, then runs the complete deployed pipeline (and the server-side evaluation if a .route file is
present). It prints one line per trip and a final verdict, the LAN address to give to laptop 2, and exits
with a non-zero status if any trip cannot be served. It reads reference routes only in the evaluation step,
exactly like the API.
"""
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.evaluation import evaluate_prediction  # noqa: E402
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

DATA = Path(os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT))
PART = os.environ.get("PATHSNAP_PARTITION", "student_test")


def check_files(d, tid):
    """Return (list of problems, has_route)."""
    probs = []
    for ext in ("track", "nodes", "arcs"):
        f = d / f"{tid}.{ext}"
        if not f.exists():
            probs.append(f"missing {f.name}")
        elif f.stat().st_size == 0:
            probs.append(f"empty {f.name}")
    if probs:
        return probs, False
    try:
        tr = np.loadtxt(d / f"{tid}.track", ndmin=2)
        nd = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
        ar = np.loadtxt(d / f"{tid}.arcs", ndmin=2)
    except Exception as e:  # noqa: BLE001
        return [f"cannot parse files: {e}"], False
    if tr.shape[1] != 3:
        probs.append(f".track has {tr.shape[1]} columns (expected lon lat t)")
    if nd.shape[1] != 2:
        probs.append(f".nodes has {nd.shape[1]} columns (expected lon lat)")
    if ar.shape[1] != 2:
        probs.append(f".arcs has {ar.shape[1]} columns (expected from to)")
    if probs:
        return probs, False
    if np.isnan(tr).any() or np.isnan(nd).any():
        probs.append("NaN values in track/nodes")
    if (np.abs(tr[:, 0]) > 180).any() or (np.abs(tr[:, 1]) > 90).any():
        probs.append("track coordinates out of range (column order lon, lat?)")
    if ar.min() < 0 or ar.max() >= len(nd):
        probs.append("arc refers to a node id outside the node list")
    if (np.diff(tr[:, 2]) <= 0).any():
        probs.append("warning: timestamps not strictly increasing (handled, but check the file)")
    rf = d / f"{tid}.route"
    has_route = rf.exists() and rf.stat().st_size > 0
    if has_route:
        rt = np.loadtxt(rf, ndmin=1)
        if rt.min() < 0 or rt.max() >= len(ar):
            probs.append("route refers to an arc id outside the arc list (evaluation will be unavailable)")
    return probs, has_route


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))      # no packet is sent; just selects the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return None


def main():
    pdir = DATA / PART
    print(f"data root : {DATA}\npartition : {PART}")
    if not pdir.exists():
        print(f"FAIL: folder {pdir} does not exist. Set PATHSNAP_DATA / PATHSNAP_PARTITION.")
        return 2
    trips = sorted(p.name for p in pdir.iterdir() if p.is_dir())
    print(f"trips     : {len(trips)} found" + ("" if len(trips) == 10 else "  (note: the evaluation uses 10 reserved trips)"))
    if not trips:
        print("FAIL: no trip folders found.")
        return 2
    t0 = time.time()
    P = Predictor()
    print(f"model     : {P.cfg.get('display_name', P.cfg['model'])} loaded in {time.time() - t0:.1f} s; decoder {P.cfg['hmm_params']}\n")
    print(f"{'trip':>10} {'fixes':>6} {'matched':>7} {'route arcs':>10} {'seconds':>7}  {'evaluation':<12} {'top-1':>6} {'route F1':>8}  notes")
    bad = 0
    for tid in trips:
        probs, has_route = check_files(pdir / tid, tid)
        hard = [p for p in probs if not p.startswith("warning") and "evaluation will be unavailable" not in p]
        if hard:
            bad += 1
            print(f"{tid:>10} {'-':>6} {'-':>7} {'-':>10} {'-':>7}  {'-':<12} {'-':>6} {'-':>8}  FAIL: " + "; ".join(probs))
            continue
        try:
            t1 = time.time()
            out = P.predict_trip(tid, PART, DATA)
            ev = evaluate_prediction(out["trip"], out["cls_pred"], out["hmm_pred"], out["final_pred"], out["route"], out["graph"], PART, DATA)
            el = time.time() - t1
            acc = ev["stages"]["final"]["observation_top1_accuracy"] if ev.get("available") else None
            notes = "; ".join(probs)
            if not ev.get("available"):
                notes = (notes + "; " if notes else "") + ev.get("reason", "")
            n, m = out["trip"].n_obs, int((out["final_pred"] >= 0).sum())
            if n and m == 0:
                notes = (notes + "; " if notes else "") + "no fix could be matched (track far from the road network?)"
            rf1 = ev.get("route_f1") if ev.get("available") else None
            acc_s = f"{acc:.3f}" if acc is not None else "-"
            rf1_s = f"{rf1:.3f}" if rf1 is not None else "-"
            status = "available" if ev.get("available") else "unavailable"
            print(f"{tid:>10} {n:>6} {m:>7} {len(out['route']):>10} {el:>7.2f}  {status:<12} {acc_s:>6} {rf1_s:>8}  {notes}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"{tid:>10}  FAIL: pipeline raised {type(e).__name__}: {e}")
    ip = lan_ip()
    print("\nstart the API   : uvicorn api.server:app --host 0.0.0.0 --port 8000   (same PATHSNAP_DATA / PATHSNAP_PARTITION)")
    if ip:
        print(f"laptop 2 opens  : http://localhost:8080/?api=http://{ip}:8000   (after: cd ui && python -m http.server 8080)")
        print(f"test from laptop 2 first: curl http://{ip}:8000/trips   (if it times out, allow Python through the firewall on laptop 1)")
    print(f"\nPRE-FLIGHT {'PASSED' if bad == 0 else 'FAILED'}: {len(trips) - bad} of {len(trips)} trips can be served.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
