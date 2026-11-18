"""Exploratory data analysis over train + student_test partitions.

Outputs: results/eda_trips.csv, results/eda_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import DEFAULT_DATA_ROOT, list_trips, load_trip  # noqa: E402

rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(part):
        d = DEFAULT_DATA_ROOT / part / tid
        files = {e: (d / f"{tid}.{e}") for e in ["track", "nodes", "arcs", "route"]}
        tr = load_trip(tid, part)
        tk = tr.track
        t = tk[:, 2]
        dt = np.diff(t)
        step = np.linalg.norm(np.diff(tr.xy, axis=0), axis=1)
        nn, na = len(tr.nodes), len(tr.arcs)
        A = coo_matrix((np.ones(na), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(nn, nn))
        n_wcc, _ = connected_components(A, directed=True, connection="weak")
        n_scc, lab = connected_components(A, directed=True, connection="strong")
        deg = np.bincount(tr.arcs.ravel(), minlength=nn)
        r = tr.route
        # route validity: consecutive arcs share node (head(prev) == tail(next))
        cont = tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0]
        arcset = set(map(tuple, tr.arcs.tolist()))
        n_twoway = sum((b, a) in arcset for a, b in arcset)
        alen = np.linalg.norm(tr.node_xy[tr.arcs[:, 1]] - tr.node_xy[tr.arcs[:, 0]], axis=1)
        route_len = alen[r].sum()
        rows.append(dict(
            partition=part, trip=tid,
            n_obs=len(tk), n_nodes=nn, n_arcs=na, route_len_arcs=len(r),
            size_track_kb=files["track"].stat().st_size / 1024,
            nan_track=int(np.isnan(tk).sum()), nan_nodes=int(np.isnan(tr.nodes).sum()),
            lon_min=tk[:, 0].min(), lon_max=tk[:, 0].max(),
            lat_min=tk[:, 1].min(), lat_max=tk[:, 1].max(),
            t_start=t[0], t_end=t[-1], duration_s=t[-1] - t[0],
            ts_monotonic=bool((dt > 0).all()), n_dup_ts=int((dt == 0).sum()),
            frac_dt1=float(np.mean(np.isclose(dt, 1.0))), max_dt=float(dt.max()),
            n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt30s=int((dt > 30).sum()),
            median_step_m=float(np.median(step)), max_step_m=float(step.max()),
            max_speed_ms=float(np.max(step / np.maximum(dt, 1e-6))),
            track_km=float(step.sum() / 1000), route_km=float(route_len / 1000),
            weak_components=int(n_wcc), strong_components=int(n_scc),
            largest_scc_frac=float(np.bincount(lab).max() / nn),
            isolated_nodes=int((deg == 0).sum()), frac_twoway_arcs=n_twoway / na,
            route_ids_valid=bool((r >= 0).all() and (r < na).all()),
            route_continuity=float(cont.mean()) if len(cont) else 1.0,
            route_repeats=int(len(r) - len(np.unique(r))),
            arc_len_median_m=float(np.median(alen)), zero_len_arcs=int((alen < 0.01).sum()),
            self_loops=int((tr.arcs[:, 0] == tr.arcs[:, 1]).sum()),
        ))
        print(part, tid, len(tk), nn, na, len(r), flush=True)

df = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / "results/eda_trips.csv", index=False)
summ = {}
for part, g in df.groupby("partition"):
    summ[part] = {c: {"min": float(g[c].min()), "median": float(g[c].median()),
                      "max": float(g[c].max()), "sum": float(g[c].sum())}
                  for c in g.columns if c not in ("trip","partition") and g[c].dtype.kind in "biuf"}
    summ[part]["n_trips"] = len(g)
json.dump(summ, open(ROOT / "results/eda_summary.json", "w"), indent=1)
pd.set_option("display.width", 250, "display.max_columns", 60)
print(df.describe().T.to_string())
