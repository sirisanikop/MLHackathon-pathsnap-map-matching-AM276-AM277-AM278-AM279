"""Decoder option test: bridge candidate dropouts (link the neighbours of fixes that have no candidate arc
when the time/distance gap is within the usual limits) instead of breaking the Viterbi chain there.

Scorer and all decoder parameters are the deployed ones; only `bridge` differs.
Adoption rule (fixed before the run): adopt if, on the 16 held-out training trips (fold 4), mean per-trip final
top-1 and macro-F1 do not decrease, and mean per-trip final top-1 increases on the 64 tuning trips (folds 0-3).
Student-test is reported but not used by the rule.

usage: python scripts/bridge_eval.py <model_name>      output: results/bridge_eval.json (+ per-trip CSVs)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tune_decoder as TD  # noqa: E402

RES = TD.RES
fs = json.load(open(RES / "final_summary.json"))
base = dict(fs["decoder_round1"], bridge=False)
brid = dict(base, bridge=True)
folds = json.load(open(RES / "folds.json"))
oof = pd.read_pickle(RES / f"oof_scores_{TD.MODEL}.pkl")
te = pd.read_pickle(RES / f"test_scores_{TD.MODEL}.pkl")
out, per = {"model": TD.MODEL, "decoder": base}, {}
for part, groups in [("train", {t: g for t, g in oof.groupby("trip")}), ("student_test", {t: g for t, g in te.groupby("trip")})]:
    res = TD.run(part, groups, [base, brid], full=True)
    for name, rows in zip(["no_bridge", "bridge"], res):
        s_, d_ = TD.summarise(rows)
        s_["trip_mean_top1"] = float(d_.final_top1.mean())
        s_["total_chains"] = int(d_.chains.sum())
        out.setdefault(part, {})[name] = s_
        per[(part, name)] = d_
        if part == "train":
            for sub, sel in [("tuning_trips", [t for t in d_.trip if folds[t] != 4]), ("heldout_trips", [t for t in d_.trip if folds[t] == 4])]:
                g_ = d_[d_.trip.isin(sel)]
                out.setdefault(sub, {})[name] = dict(n_trips=int(len(g_)), trip_mean_top1=float(g_.final_top1.mean()),
                                                     trip_mean_macro_f1=float(g_.final_macro_f1.mean()),
                                                     pooled_top1=float(g_.n_ok.sum() / g_.n_rel.sum()))
h0, h1, t0, t1 = out["heldout_trips"]["no_bridge"], out["heldout_trips"]["bridge"], out["tuning_trips"]["no_bridge"], out["tuning_trips"]["bridge"]
out["adopted"] = bool(h1["trip_mean_top1"] >= h0["trip_mean_top1"] and h1["trip_mean_macro_f1"] >= h0["trip_mean_macro_f1"]
                      and t1["trip_mean_top1"] > t0["trip_mean_top1"])
a, b = per[("train", "no_bridge")].set_index("trip"), per[("train", "bridge")].set_index("trip")
ch = (b.final_top1 - a.final_top1)
out["train_trips_better"], out["train_trips_worse"] = int((ch > 1e-12).sum()), int((ch < -1e-12).sum())
out["largest_changes"] = {t: [float(a.final_top1[t]), float(b.final_top1[t]), int(a.chains[t]), int(b.chains[t])]
                          for t in ch.abs().sort_values(ascending=False).head(8).index}
for (part, name), d_ in per.items():
    d_.to_csv(RES / f"bridge_{part}_{name}.csv", index=False)
json.dump(out, open(RES / "bridge_eval.json", "w"), indent=1)
for k in ["tuning_trips", "heldout_trips"]:
    print(k, {n: {x: round(y, 4) for x, y in v.items()} for n, v in out[k].items()})
for part in ["train", "student_test"]:
    print(part, {n: dict(top1=round(v["final"]["top1"], 4), trip_mean=round(v["trip_mean_top1"], 4), macro_f1=round(v["final"]["macro_f1"], 4),
                         cont=round(v["final"]["continuity"], 4), route_f1=round(v["route_f1"], 4), chains=v["total_chains"]) for n, v in out[part].items()})
print("better/worse train trips:", out["train_trips_better"], out["train_trips_worse"], "| largest changes [before, after, chains before, chains after]:")
for t, v in out["largest_changes"].items():
    print("  ", t, [round(x, 4) if isinstance(x, float) else x for x in v])
print("ADOPTED:", out["adopted"])
