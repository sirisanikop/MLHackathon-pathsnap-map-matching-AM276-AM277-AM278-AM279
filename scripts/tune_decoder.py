"""Bayesian (Optuna TPE) tuning of the decoder on TRAINING trips with out-of-fold scorer scores.

Design (second version; the first, pooled-objective run is kept in results/*_pooled.*):
  * objective  : mean PER-TRIP final top-1 (Viterbi + route re-alignment) -- the evaluator scores one trip
                 at a time, and a pooled objective is dominated by a few very long trips;
  * tuning set : the 64 training trips of CV folds 0-3; the 16 trips of fold 4 are held out;
  * adoption   : the tuned setting replaces the round-1 setting only if, on the HELD-OUT training trips,
                 it improves mean per-trip top-1 and does not lower mean per-trip macro-F1.
Student-test is evaluated once at the end and is not part of the rule.

Speed: shortest-path distances are parameter-independent, so each trip is prepared once per
top-M setting (cached on disk) and Viterbi is then re-run for every trial.

usage: python scripts/tune_decoder.py <model_name> [n_trials]
outputs: results/decoder_trials.csv, results/decoder_best.json,
         results/final_train_oof.csv, results/final_test.csv, results/final_summary.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_route, load_trip  # noqa: E402
from pathsnap.dataset import CACHE_V2  # noqa: E402
from pathsnap.hmm import HMMParams, prepare, realign_to_route, reconstruct_route, viterbi  # noqa: E402
from pathsnap.metrics import continuity, route_overlap  # noqa: E402

RES = ROOT / "results"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "lgbm_v2"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
BATCH = 8
PREP = ROOT / "cache_v2" / "prep"
PREP.mkdir(parents=True, exist_ok=True)
TOP_M = [6, 8, 12, 24]


def get_prep(tid, part, c, top_m, bridge=False):
    f = PREP / f"{MODEL}_{part}_{tid}_m{top_m}{'_b' if bridge else ''}_v2.pkl"
    if f.exists():
        return joblib.load(f)
    tr = load_trip(tid, part, with_route=False)
    feat = pd.read_pickle(CACHE_V2 / f"feat_{part}_{tid}.pkl")[["obs", "arc", "t_proj"]]
    P = prepare(tr, c.merge(feat, on=["obs", "arc"]), HMMParams(top_m=top_m, bridge=bridge))
    joblib.dump(P, f, compress=1)
    return P


def eval_trip(tid, part, c, configs, full=False):
    """Evaluate all decoder configs on one trip. Labels are used only to score."""
    lab = np.load(CACHE_V2 / f"lab_{part}_{tid}.npz")
    y, rel = lab["lab"], lab["reliable"]
    out, preps = [], {}
    for cfg in configs:
        hp = HMMParams(**cfg)
        if (hp.top_m, hp.bridge) not in preps:
            preps[(hp.top_m, hp.bridge)] = get_prep(tid, part, c, hp.top_m, hp.bridge)
        P = preps[(hp.top_m, hp.bridge)]
        t0 = time.time()
        vit, seg = viterbi(P, hp)
        route, _ = reconstruct_route(vit, seg, P.g)
        fin = realign_to_route(P.trip, vit, route, hp.realign_max_m)
        dec_t = time.time() - t0
        r = dict(trip=tid, n_rel=int(rel.sum()), n_ok=int((fin == y)[rel].sum()), n_ok_vit=int((vit == y)[rel].sum()),
                 f1=float(f1_score(y[rel], fin[rel], average="macro", zero_division=0)))
        if full:
            cls = np.full(P.n, -1, dtype=np.int64)
            b = c.loc[c.groupby("obs").pn.idxmax()]
            cls[b.obs.to_numpy()] = b.arc.to_numpy()
            ref = load_route(tid, part)
            ro = route_overlap(route, ref, P.g.alen)
            r.update(n_obs=P.n, cls_top1=float((cls == y)[rel].mean()), hmm_top1=float((vit == y)[rel].mean()),
                     final_top1=float((fin == y)[rel].mean()),
                     cls_macro_f1=float(f1_score(y[rel], cls[rel], average="macro", zero_division=0)),
                     hmm_macro_f1=float(f1_score(y[rel], vit[rel], average="macro", zero_division=0)),
                     final_macro_f1=float(f1_score(y[rel], fin[rel], average="macro", zero_division=0)),
                     cls_continuity=continuity(cls, P.trip.arcs), hmm_continuity=continuity(vit, P.trip.arcs),
                     final_continuity=continuity(fin, P.trip.arcs), route_precision=ro["route_precision"],
                     route_recall=ro["route_recall"], route_f1=ro["route_f1"], chains=int(seg.max() + 1),
                     decode_time_s=dec_t)
        out.append(r)
    return out


def run(part, groups, configs, full=False):
    res = Parallel(n_jobs=4)(delayed(eval_trip)(t, part, g, configs, full) for t, g in groups.items())
    return [[r[i] for r in res] for i in range(len(configs))]


def suggest(trial):
    return dict(beta=trial.suggest_float("beta", 2.0, 80.0, log=True),
                w_emit=trial.suggest_float("w_emit", 0.05, 2.0, log=True),
                back_tol_m=trial.suggest_float("back_tol_m", 3.0, 40.0),
                back_pen_m=trial.suggest_float("back_pen_m", 0.0, 200.0),
                uturn_pen=trial.suggest_float("uturn_pen", 0.0, 8.0),
                gap_s=trial.suggest_float("gap_s", 10.0, 300.0, log=True),
                top_m=trial.suggest_categorical("top_m", TOP_M),
                realign_max_m=trial.suggest_float("realign_max_m", 30.0, 150.0))


def summarise(rows):
    d = pd.DataFrame(rows)
    w = d.n_rel
    s = {k: dict(top1=float(np.average(d[f"{k}_top1"], weights=w)), macro_f1=float(d[f"{k}_macro_f1"].mean()),
                 continuity=float(d[f"{k}_continuity"].mean())) for k in ["cls", "hmm", "final"]}
    s.update(route_f1=float(d.route_f1.mean()), route_precision=float(d.route_precision.mean()),
             route_recall=float(d.route_recall.mean()), mean_decode_s=float(d.decode_time_s.mean()),
             max_decode_s=float(d.decode_time_s.max()))
    return s, d


def tm(rows, trips=None, key="top1"):
    """mean per-trip final top-1 (or macro-F1) over a set of trips"""
    v = [(r["n_ok"] / r["n_rel"] if key == "top1" else r["f1"]) for r in rows if trips is None or r["trip"] in trips]
    return float(np.mean(v))


if __name__ == "__main__":
    t_all = time.time()
    oof = pd.read_pickle(RES / f"oof_scores_{MODEL}.pkl")
    tr_groups = {t: g for t, g in oof.groupby("trip")}
    folds = json.load(open(RES / "folds.json"))
    TUNE = {t for t in tr_groups if folds[t] != 4}
    HOLD = {t for t in tr_groups if folds[t] == 4}
    default = dict(beta=20.0, w_emit=0.25, back_tol_m=15.0, back_pen_m=60.0, uturn_pen=0.0, gap_s=60.0, top_m=24,
                   realign_max_m=80.0)   # the round-1 decoder setting
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=16, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_decoder_tpe")
    study.enqueue_trial(default)
    n_done = 0
    while n_done < N_TRIALS:
        trials = [study.ask() for _ in range(min(BATCH, N_TRIALS - n_done))]
        cfgs = [suggest(t) for t in trials]
        res = run("train", tr_groups, cfgs)
        for t, cfg, rows in zip(trials, cfgs, res):
            t.set_user_attr("tune_trip_mean_macro_f1", tm(rows, TUNE, "f1"))
            t.set_user_attr("heldout_trip_mean_top1", tm(rows, HOLD))          # logged only, never optimised
            t.set_user_attr("pooled_top1_all80", sum(r["n_ok"] for r in rows) / sum(r["n_rel"] for r in rows))
            study.tell(t, tm(rows, TUNE))
        n_done += len(trials)
        print(f"{n_done} trials, best tuning-trip mean top-1 {study.best_value:.4f} | {time.time() - t_all:.0f}s", flush=True)
    study.trials_dataframe().to_csv(RES / "decoder_trials.csv", index=False)
    best = dict(study.best_params)

    # ---- round-1 setting vs tuned: tuning trips, held-out training trips, all 80, student-test ----
    te = pd.read_pickle(RES / f"test_scores_{MODEL}.pkl")
    te_groups = {t: g for t, g in te.groupby("trip")}
    out = {"model": MODEL, "decoder_tuned": best, "decoder_round1": default}
    per = {}
    for part, groups in [("train", tr_groups), ("student_test", te_groups)]:
        res = run(part, groups, [default, best], full=True)
        for name, rows in zip(["round1_decoder", "tuned_decoder"], res):
            s_, d_ = summarise(rows)
            s_["trip_mean_top1"] = float(d_.final_top1.mean())
            out.setdefault(part, {})[name] = s_
            per[(part, name)] = d_
            if part == "train":
                for sub, trips in [("tuning_trips", TUNE), ("heldout_trips", HOLD)]:
                    g_ = d_[d_.trip.isin(trips)]
                    out.setdefault(sub, {})[name] = dict(n_trips=int(len(g_)), trip_mean_top1=float(g_.final_top1.mean()),
                                                         trip_mean_macro_f1=float(g_.final_macro_f1.mean()),
                                                         pooled_top1=float(g_.n_ok.sum() / g_.n_rel.sum()),
                                                         continuity=float(g_.final_continuity.mean()))
    h0, h1 = out["heldout_trips"]["round1_decoder"], out["heldout_trips"]["tuned_decoder"]
    adopted = bool(h1["trip_mean_top1"] > h0["trip_mean_top1"] and h1["trip_mean_macro_f1"] >= h0["trip_mean_macro_f1"])
    sel = "tuned_decoder" if adopted else "round1_decoder"
    out["adopted_tuned_decoder"] = adopted
    out["selected_decoder"] = sel
    out["selected_params"] = best if adopted else default
    for part in ["train", "student_test"]:
        out[part]["final_decoder"] = out[part][sel]
        per[(part, sel)].to_csv(RES / ("final_train_oof.csv" if part == "train" else "final_test.csv"), index=False)
    json.dump({"model": MODEL, "best_params": best, "best_tuning_trip_mean_top1": study.best_value,
               "round1_setting": default, "round1_tuning_trip_mean_top1": study.trials[0].value,
               "n_trials": len(study.trials), "adopted": adopted, "selected_params": out["selected_params"],
               "adoption_rule": "tuned replaces round-1 only if on the 16 held-out training trips (fold 4) mean per-trip "
                                "final top-1 improves and mean per-trip macro-F1 does not decrease",
               "sampler": "Optuna TPE (multivariate, seed 42, 16 startup trials), batches of 8",
               "objective": "mean per-trip final top-1 on the 64 training trips of folds 0-3 (OOF scorer scores)",
               "wall_time_s": time.time() - t_all}, open(RES / "decoder_best.json", "w"), indent=1)
    json.dump(out, open(RES / "final_summary.json", "w"), indent=1)
    for k in ["tuning_trips", "heldout_trips"]:
        print(k, json.dumps({n: {a: round(b, 4) for a, b in v.items()} for n, v in out[k].items()}), flush=True)
    for part in ["train", "student_test"]:
        print(part, json.dumps({n: dict(top1=round(v["final"]["top1"], 4), trip_mean_top1=round(v["trip_mean_top1"], 4),
                                        macro_f1=round(v["final"]["macro_f1"], 4), cont=round(v["final"]["continuity"], 4),
                                        route_f1=round(v["route_f1"], 4)) for n, v in out[part].items() if n != "final_decoder"}), flush=True)
    print("BEST", best, "| ADOPTED:", adopted, flush=True)
