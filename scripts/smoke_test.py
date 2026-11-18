"""End-to-end smoke test in fresh processes.

1. starts the API (uvicorn subprocess) on a temporary data root that contains
   two real student-test trips and one synthetic trip whose middle fixes are
   moved 5 km away (-> no candidates, chain must break, no crash);
2. calls /health, /trips, /predict for each trip;
3. checks response shape, ID validity, route connectivity, absence of
   reference data in the response, and that evaluation is present.
Writes results/smoke_test.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402

PORT = 8765
TRIPS = ["00000009", "00000077"]


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pathsnap_smoke_"))
    part = tmp / "smoke_partition"
    for t in TRIPS:
        shutil.copytree(DEFAULT_DATA_ROOT / "student_test" / t, part / t)
    # synthetic trip with a dropout: fixes 100-199 moved ~5 km north (no road candidates)
    src = DEFAULT_DATA_ROOT / "student_test" / "00000052"
    dst = part / "99999999"
    dst.mkdir(parents=True)
    for ext in ["nodes", "arcs", "route"]:
        shutil.copy(src / f"00000052.{ext}", dst / f"99999999.{ext}")
    tr = np.loadtxt(src / "00000052.track")
    tr[100:200, 1] += 0.045
    np.savetxt(dst / "99999999.track", tr, fmt="%.6f")

    env = dict(os.environ, PATHSNAP_DATA=str(tmp), PATHSNAP_PARTITION="smoke_partition")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app", "--port", str(PORT)],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    report = {"checks": {}}
    try:
        for _ in range(120):
            try:
                call("/health")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        h = call("/health")
        trips = call("/trips")
        report["health"] = h
        report["checks"]["trips_listed"] = trips["trips"] == sorted(TRIPS + ["99999999"])
        for t in trips["trips"]:
            t0 = time.time()
            r = call("/predict", {"trip_id": t})
            el = time.time() - t0
            arcs = np.loadtxt(part / t / f"{t}.arcs", dtype=np.int64)
            n_obs = len(np.loadtxt(part / t / f"{t}.track"))
            route = np.array(r["predicted_route_arc_ids"])
            conn = float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if len(route) > 1 else float("nan")
            keys = set(r.keys()) | set(r["evaluation"].keys()) | set(r["evaluation"]["stages"]["final"].keys())
            report[t] = {
                "seconds": round(el, 2), "n_observations": r["n_observations"],
                "n_matched": r["n_matched_observations"], "route_arcs": len(route),
                "route_ids_valid": bool(((route >= 0) & (route < len(arcs))).all()),
                "route_consecutive_connected": conn, "hmm_segments": r["n_hmm_segments"],
                "obs_count_matches_file": r["n_observations"] == n_obs,
                "no_reference_fields": not any(k in keys for k in ("route", "reference_route", "labels", "lab")),
                "evaluation_available": r["evaluation"].get("available"),
                "acc_final": r["evaluation"]["stages"]["final"]["observation_top1_accuracy"],
                "acc_hmm": r["evaluation"]["stages"]["hmm"]["observation_top1_accuracy"],
                "acc_cls": r["evaluation"]["stages"]["classifier_only"]["observation_top1_accuracy"],
            }
            print(t, report[t], flush=True)
        report["checks"]["missing_candidates_handled"] = (report["99999999"]["n_matched"] < report["99999999"]["n_observations"]
                                                          and report["99999999"]["hmm_segments"] >= 2)
        report["checks"]["all_ids_valid"] = all(report[t]["route_ids_valid"] for t in trips["trips"])
        report["checks"]["no_reference_leak"] = all(report[t]["no_reference_fields"] for t in trips["trips"])
        try:
            call("/predict", {"trip_id": "does_not_exist"})
            report["checks"]["unknown_trip_404"] = False
        except urllib.error.HTTPError as e:
            report["checks"]["unknown_trip_404"] = e.code == 404
    finally:
        proc.terminate()
        out, _ = proc.communicate(timeout=20)
        report["server_log_tail"] = out.splitlines()[-12:]
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(report, open(ROOT / "results/smoke_test.json", "w"), indent=1, default=str)
    print(json.dumps(report["checks"], indent=1))
    ok = all(report["checks"].values())
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
