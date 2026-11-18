"""Low-dimensional decoder check for the round-2 scorer: 4 x 4 grid over (beta, w_emit) with all other
decoder parameters at the round-1 setting. Same design as tune_decoder.py: selection by mean per-trip final
top-1 on the 64 tuning trips (folds 0-3), adoption only if the 16 held-out training trips (fold 4) improve in
mean per-trip top-1 without losing macro-F1. Updates results/final_summary.json with the outcome.

usage: python scripts/decoder_grid_v2.py <model_name>     output: results/decoder_grid_v2.csv
"""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tune_decoder as TD  # noqa: E402

RES = TD.RES
oof = pd.read_pickle(RES / f"oof_scores_{TD.MODEL}.pkl")
tr_groups = {t: g for t, g in oof.groupby("trip")}
folds = json.load(open(RES / "folds.json"))
TUNE = {t for t in tr_groups if folds[t] != 4}
HOLD = {t for t in tr_groups if folds[t] == 4}
fs = json.load(open(RES / "final_summary.json"))
default = fs["decoder_round1"]
cfgs = [dict(default, beta=float(b), w_emit=float(w)) for b, w in itertools.product([5, 10, 20, 40], [0.1, 0.25, 0.5, 1.0])]
res = TD.run("train", tr_groups, cfgs)
rows = [dict(beta=c["beta"], w_emit=c["w_emit"], tune_trip_mean_top1=TD.tm(r, TUNE), tune_trip_mean_macro_f1=TD.tm(r, TUNE, "f1"),
             heldout_trip_mean_top1=TD.tm(r, HOLD), heldout_trip_mean_macro_f1=TD.tm(r, HOLD, "f1"),
             pooled_top1_all80=sum(x["n_ok"] for x in r) / sum(x["n_rel"] for x in r)) for c, r in zip(cfgs, res)]
G = pd.DataFrame(rows)
G.to_csv(RES / "decoder_grid_v2.csv", index=False)
print(G.round(4).to_string())
b = G.sort_values("tune_trip_mean_top1", ascending=False).iloc[0]
d0 = G[(G.beta == default["beta"]) & (G.w_emit == default["w_emit"])].iloc[0]
is_default = bool(b.beta == d0.beta and b.w_emit == d0.w_emit)
adopted = bool((not is_default) and b.heldout_trip_mean_top1 > d0.heldout_trip_mean_top1
               and b.heldout_trip_mean_macro_f1 >= d0.heldout_trip_mean_macro_f1)
fs["grid2"] = {"best_on_tuning_trips": {"beta": float(b.beta), "w_emit": float(b.w_emit)}, "best_is_round1_setting": is_default,
               "tuning": {"round1": float(d0.tune_trip_mean_top1), "best": float(b.tune_trip_mean_top1)},
               "heldout": {"round1": float(d0.heldout_trip_mean_top1), "best": float(b.heldout_trip_mean_top1),
                           "round1_macro_f1": float(d0.heldout_trip_mean_macro_f1), "best_macro_f1": float(b.heldout_trip_mean_macro_f1)},
               "adopted": adopted}
if adopted:
    best = dict(default, beta=float(b.beta), w_emit=float(b.w_emit))
    te = pd.read_pickle(RES / f"test_scores_{TD.MODEL}.pkl")
    for part, groups in [("train", tr_groups), ("student_test", {t: g for t, g in te.groupby("trip")})]:
        s_, d_ = TD.summarise(TD.run(part, groups, [best], full=True)[0])
        s_["trip_mean_top1"] = float(d_.final_top1.mean())
        fs[part]["grid2_decoder"] = s_
        fs[part]["final_decoder"] = s_
        d_.to_csv(RES / ("final_train_oof.csv" if part == "train" else "final_test.csv"), index=False)
    fs["selected_decoder"], fs["selected_params"] = "grid2_decoder", best
json.dump(fs, open(RES / "final_summary.json", "w"), indent=1)
print("best on tuning trips:", dict(beta=b.beta, w=b.w_emit), "| is round-1 setting:", is_default, "| ADOPTED:", adopted)
print("final decoder:", fs["selected_decoder"], fs["selected_params"])
for part in ["train", "student_test"]:
    f = fs[part]["final_decoder"]
    print(part, "final top1 %.4f macroF1 %.4f cont %.4f routeF1 %.4f" % (f["final"]["top1"], f["final"]["macro_f1"], f["final"]["continuity"], f["route_f1"]))
