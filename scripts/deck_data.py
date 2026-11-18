"""Export the numbers used by the pitch deck (deck/build_deck.js) from results/."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
J = lambda p: json.load(open(RES / p))  # noqa: E731
MODELS = ["logreg", "gnb", "knn", "rf", "lgbm", "mlp"]
LAB = {"logreg": "LogReg", "gnb": "Gauss NB", "knn": "KNN", "rf": "Rand. Forest", "lgbm": "LightGBM", "mlp": "MLP",
       "lgbm_tuned": "LightGBM tuned"}
M = {m: J(f"model_{m}.json") for m in MODELS + ["lgbm_tuned"]}
eda = pd.read_csv(RES / "eda_trips.csv", dtype={"trip": str})
hab = pd.read_csv(RES / "hmm_ablation_models.csv").set_index("model")
hs = J("hmm_summary.json")
opt = J("optuna_best.json")
tv = pd.read_csv(RES / "tuned_vs_untuned.csv").set_index("model")
base = J("baseline_nearest.json")
smoke = J("smoke_test.json")
q = pd.read_csv(RES / "eda_quality_table.csv").set_index("check")
fi = pd.read_csv(RES / "feature_importance_lgbm_tuned.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
trials = pd.read_csv(RES / "optuna_trials.csv")
trials = trials[trials.state == "COMPLETE"]
sel = hs["selected_model"]
pt = pd.read_csv(RES / "hmm_test_pertrip.csv", dtype={"trip": str})
pt = pt[pt.model == sel]
team = json.load(open(ROOT / "configs/team.json")) if (ROOT / "configs/team.json").exists() else {}

re_ = J("realign_summary.json")
err = J("error_breakdown.json")
api = pd.read_csv(RES / "api_all_trips.csv", dtype={"trip": str})
rt = pd.read_csv(RES / "realign_test.csv", dtype={"trip": str})
fs = J("final_summary.json")
choice = J("final_choice.json")
m2 = J(f"model_{choice['selected']}.json")
val = J("validation_final.json")
var = pd.read_csv(RES / "tune_v2_variants.csv")
ft = pd.read_csv(RES / "final_test.csv", dtype={"trip": str})
r1t = rt.set_index("trip")
d = {
    "r2": {"variants": var[["variant", "cv_top1", "cv_macro_f1", "objective", "n_features"]].to_dict("records"),
           "scorer": {"cv_top1": m2["cv_mean"]["obs_top1"], "cv_f1": m2["cv_mean"]["obs_macro_f1_trip"],
                      "train_top1": m2["cv_mean"]["train_obs_top1"], "test_top1": m2["student_test"]["obs_top1"],
                      "test_f1": m2["student_test"]["obs_macro_f1_trip"], "test_ll": m2["student_test"]["obs_logloss"],
                      "n_features": m2["n_features"]},
           "r1_scorer": {"cv_top1": M["lgbm_tuned"]["cv_mean"]["obs_top1"], "cv_f1": M["lgbm_tuned"]["cv_mean"]["obs_macro_f1_trip"],
                         "test_top1": M["lgbm_tuned"]["student_test"]["obs_top1"], "test_f1": M["lgbm_tuned"]["student_test"]["obs_macro_f1_trip"],
                         "test_ll": M["lgbm_tuned"]["student_test"]["obs_logloss"]},
           "optuna2": {"n_trials": J("optuna_v2_best.json")["n_trials"], "selected": choice["selected"], "reason": choice["reason"]},
           "decoder": {"n_trials": J("decoder_best.json")["n_trials"], "best": J("decoder_best.json")["best_params"]},
           "test": fs["student_test"]["final_decoder"], "train": fs["train"]["final_decoder"],
           "decoder_adopted": fs["adopted_tuned_decoder"], "heldout": fs["heldout_trips"],
           "val": val,
           "six_v2": {m: {"cv_top1": J(f"model_{m}_v2feat.json")["cv_mean"]["obs_top1"], "cv_top1_v1": M[m]["cv_mean"]["obs_top1"]} for m in MODELS},
           "pertrip": [{"trip": r.trip, "cls": r.cls_top1, "hmm": r.hmm_top1, "final": r.final_top1,
                        "r1_final": float(r1t.loc[r.trip, "realign_top1"])} for r in ft.itertuples()]},
    "team": team,
    "final": {"test": re_["student_test"], "train": re_["train"], "errors": err["errors"], "label_noise": err["label_noise"],
              "identical_without_route": err["predictions_identical_without_route_files"],
              "api_max_s": float(api.seconds.max()), "api_ok": int((api.status == "ok").sum()),
              "pertrip": rt[["trip", "cls_top1", "hmm_top1", "realign_top1"]].to_dict("records")},
    "data": {"train_trips": int((eda.partition == "train").sum()), "test_trips": int((eda.partition == "student_test").sum()),
             "train_fixes": int(eda[eda.partition == "train"].n_obs.sum()),
             "test_fixes": int(eda[eda.partition == "student_test"].n_obs.sum()),
             "arcs_min": int(eda.n_arcs.min()), "arcs_max": int(eda.n_arcs.max()),
             "frac_dt1": float(eda.frac_dt1.median()), "max_gap": float(eda.max_dt.max()),
             "twoway": float(eda.frac_twoway_arcs.median()),
             "offroute_train": int(q.loc["observations flagged off-route (>60 m)", "train"]),
             "miss_test": float(q.loc["candidate miss rate (reliable obs)", "student_test"]),
             "miss_train": float(q.loc["candidate miss rate (reliable obs)", "train"])},
    "baseline": {"test_top1": base["student_test"]["obs_top1"], "cv_top1": base["cv_mean"]["obs_top1"],
                 "test_cont": base["student_test"]["route_continuity"]},
    "models": [{"key": m, "label": LAB[m], "family": M[m]["config"]["family"],
                "cv_f1": M[m]["cv_mean"]["obs_macro_f1_trip"], "cv_f1_std": M[m]["cv_std"]["obs_macro_f1_trip"],
                "cv_top1": M[m]["cv_mean"]["obs_top1"], "train_top1": M[m]["cv_mean"]["train_obs_top1"],
                "test_top1": M[m]["student_test"]["obs_top1"], "test_f1": M[m]["student_test"]["obs_macro_f1_trip"],
                "test_ll": M[m]["student_test"]["obs_logloss"], "cont": M[m]["student_test"]["route_continuity"],
                "infer_s": M[m]["student_test"]["infer_time_s"], "fit_s": M[m]["final_fit_time_s"]}
               for m in MODELS],
    "optuna": {"n_trials": int(len(trials)), "best": opt["best_params"], "best_cv": opt["best_cv_top1"],
               "untuned_cv_search": opt["untuned_same_protocol"]["cv_top1"], "wall_min": opt["wall_time_s"] / 60,
               "min": float(trials.value.min()), "max": float(trials.value.max()),
               "tv": {k: tv.loc[k].to_dict() for k in tv.index}},
    "hmm": {"sel": sel, "params": hs["best_params"],
            "rows": {k: hab.loc[k].to_dict() for k in hab.index},
            "pertrip": pt[["trip", "cls_top1", "hmm_top1", "cls_route_f1", "hmm_route_f1"]].to_dict("records")},
    "features": [[k, float(v)] for k, v in fi.head(8).items()],
    "smoke": {k: smoke[k] for k in ["00000009", "00000077"]} | {"checks": smoke["checks"]},
}
(ROOT / "deck").mkdir(exist_ok=True)
json.dump(d, open(ROOT / "deck/deck_data.json", "w"), indent=1, default=float)
print("deck data written")
