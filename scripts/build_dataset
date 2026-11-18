"""Build candidate/feature tables for all train + student_test trips.

cache/feat_<partition>_<trip>.pkl   route-free candidate features
cache/lab_<partition>_<trip>.npz    labels derived from .route (evaluation/training only)
results/candidate_stats.csv         per-trip candidate recall / miss rate
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import V2, build_candidates  # noqa: E402
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.labels import observation_labels  # noqa: E402

IS_V2 = "--v2" in sys.argv
CACHE = ROOT / ("cache_v2" if IS_V2 else "cache")
CACHE.mkdir(exist_ok=True)


def one(tid, part):
    t0 = time.time()
    tr = load_trip(tid, part, with_route=False)
    feat = build_candidates(tr, v2=True, **V2) if IS_V2 else build_candidates(tr)   # route never loaded here
    t1 = time.time()
    tr_lab = load_trip(tid, part, with_route=True)
    lab, ridx, ld, rel = observation_labels(tr_lab)
    t2 = time.time()
    feat.to_pickle(CACHE / f"feat_{part}_{tid}.pkl")
    np.savez(CACHE / f"lab_{part}_{tid}.npz", lab=lab, ridx=ridx, dist=ld, reliable=rel)
    s = set(zip(feat.obs.to_numpy(), feat.arc.to_numpy()))
    inset = np.array([(i, lab[i]) in s for i in range(tr.n_obs)])
    has_c = np.zeros(tr.n_obs, bool)
    has_c[feat.obs.unique()] = True
    nn = feat[feat.d_rank == 0].set_index("obs").arc.reindex(range(tr.n_obs)).to_numpy()
    return dict(partition=part, trip=tid, n_obs=tr.n_obs, n_rows=len(feat),
                cands_per_obs=len(feat) / tr.n_obs, frac_reliable=rel.mean(),
                n_offroute=int((~rel).sum()), frac_no_candidates=1 - has_c.mean(),
                candidate_recall=inset[rel].mean(), candidate_miss_rate=1 - inset[rel].mean(),
                nearest_top1=(nn[rel] == lab[rel]).mean(),
                feat_time_s=t1 - t0, label_time_s=t2 - t1)


if __name__ == "__main__":
    jobs = [(t, p) for p in ["train", "student_test"] for t in list_trips(p)]
    out = Parallel(n_jobs=4, verbose=5)(delayed(one)(t, p) for t, p in jobs)
    df = pd.DataFrame(out)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / ("results/candidate_stats_v2.csv" if IS_V2 else "results/candidate_stats.csv"), index=False)
    for p, g in df.groupby("partition"):
        w = g.n_obs * g.frac_reliable
        print(p, "obs", g.n_obs.sum(), "rows", g.n_rows.sum(), "offroute", g.n_offroute.sum(),
              "miss_rate(weighted) %.4f" % np.average(g.candidate_miss_rate, weights=w),
              "nearest_top1(weighted) %.4f" % np.average(g.nearest_top1, weights=w),
              "feat_time %.0fs" % g.feat_time_s.sum())
