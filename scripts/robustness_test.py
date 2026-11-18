"""Edge-case robustness test of the deployed pipeline and of the server-side evaluation.

Builds awkward synthetic trips from one real student-test trip, serves them from a fresh API process and
requests each one over HTTP: every request must return 200 with well-formed JSON (no exception, no NaN).
Nothing here is used for tuning.
output: results/robustness_test.json
"""
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT  # noqa: E402

SRC = DEFAULT_DATA_ROOT / "student_test" / "00000052"
tmp = Path(tempfile.mkdtemp(prefix="pathsnap_robust_"))
part = tmp / "edge"
track = np.loadtxt(SRC / "00000052.track")
rng = np.random.default_rng(0)


def make(name, tr, route=True, nodes=None, arcs=None, fmt="%.6f"):
    d = part / name
    d.mkdir(parents=True)
    for ext in ["nodes", "arcs"]:
        shutil.copy(SRC / f"00000052.{ext}", d / f"{name}.{ext}")
    if route:
        shutil.copy(SRC / "00000052.route", d / f"{name}.route")
    np.savetxt(d / f"{name}.track", np.atleast_2d(tr), fmt=fmt)


far = track.copy(); far[:, 1] += 1.0                       # whole track ~110 km away: no candidates at all
gap = track.copy(); gap[130:, 2] += 5000                   # one huge time gap
dup = np.repeat(track[:80], 3, axis=0); dup[:, 2] = np.arange(len(dup)) + 1   # stationary / repeated positions
jit = track.copy(); jit[:, :2] += rng.normal(0, 3e-4, size=(len(track), 2))   # ~30 m noise
coarse = track[::15].copy()                                # 15 s sampling
rev = track[::-1].copy(); rev[:, 2] = track[:, 2]          # driven backwards relative to the reference
unsorted = track.copy(); unsorted[50:60] = unsorted[50:60][::-1]             # locally non-monotone timestamps
sameT = track.copy(); sameT[100:110, 2] = sameT[100, 2]    # duplicate timestamps
cases = {
    "e01_single_fix": (track[:1], True), "e02_two_fixes": (track[:2], True), "e03_three_fixes": (track[:3], True),
    "e04_no_candidates": (far, True), "e05_huge_time_gap": (gap, True), "e06_stationary": (dup, True),
    "e07_heavy_noise": (jit, True), "e08_coarse_15s": (coarse, True), "e09_reversed": (rev, True),
    "e10_unsorted_time": (unsorted, True), "e11_duplicate_time": (sameT, True), "e12_no_route_file": (track, False),
    "e13_integer_format": (np.round(track, 5), True),
}
for k, (tr, route) in cases.items():
    make(k, tr, route)
# empty route file
make("e14_empty_route", track, route=False)
(part / "e14_empty_route" / "e14_empty_route.route").write_text("")

make("e15_empty_track", np.zeros((0, 3)))
(part / "e15_empty_track" / "e15_empty_track.track").write_text("")
make("e16_bad_route_ids", track, route=False)
(part / "e16_bad_route_ids" / "e16_bad_route_ids.route").write_text("5\n999999999\n7\n")

# ---- run every case through the REAL HTTP server (fresh uvicorn process), so JSON serialisation is tested too
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

PORT = 8766


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r_:
        return json.loads(r_.read())


env = dict(os.environ, PATHSNAP_DATA=str(tmp), PATHSNAP_PARTITION="edge")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app", "--port", str(PORT)], cwd=ROOT, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
res = {}
try:
    for _ in range(120):
        try:
            call("/health")
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    names = call("/trips")["trips"]
    assert len(names) == len(list(part.iterdir())), "every case must be listed"
    for name in names:
        r = {"ok": False}
        try:
            t0 = time.time()
            d = call("/predict", {"trip_id": name})
            ev = d["evaluation"]
            n = d["n_observations"]
            assert d["n_matched_observations"] <= n and len(d["predicted_route_lines"]) == d["n_route_arcs"]
            r.update(ok=True, n_obs=n, matched=d["n_matched_observations"], route_arcs=d["n_route_arcs"],
                     evaluation_available=bool(ev.get("available")), reason=ev.get("reason"),
                     top1_final=(ev["stages"]["final"]["observation_top1_accuracy"] if ev.get("available") else None),
                     continuity_final=(ev["stages"]["final"]["route_continuity"] if ev.get("available") else None),
                     seconds=round(time.time() - t0, 3))
        except urllib.error.HTTPError as e:
            r.update(error=f"HTTP {e.code}: {e.read()[:200]!r}")
        except Exception as e:  # noqa: BLE001
            r.update(error=f"{type(e).__name__}: {e}")
        res[name] = r
        print(name, r, flush=True)
finally:
    proc.terminate()
    log, _ = proc.communicate(timeout=20)
    shutil.rmtree(tmp, ignore_errors=True)
bad = [k for k, v in res.items() if not v["ok"]]
if bad:
    print("\n".join(ln for ln in log.splitlines() if "Error" in ln or "Traceback" in ln or "line " in ln)[-1500:])
res["all_ok"] = not bad
json.dump(res, open(ROOT / "results/robustness_test.json", "w"), indent=1, default=str)
print("ALL OK" if not bad else "FAILURES: " + ", ".join(bad))
sys.exit(0 if not bad else 1)
