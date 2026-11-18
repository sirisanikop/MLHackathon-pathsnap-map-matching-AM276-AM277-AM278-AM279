"""Write models/final_config.json and the final per-trip result files from the recorded decisions:
results/final_choice.json (scorer), results/final_summary.json (decoder setting after the tuning study) and
results/bridge_eval.json (dropout bridging). No tuning happens here.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ch, fs, br = (json.load(open(RES / f)) for f in ("final_choice.json", "final_summary.json", "bridge_eval.json"))
hp = {k: v for k, v in fs["selected_params"].items() if k != "bridge"}
hp["bridge"] = bool(br["adopted"])
if br["adopted"]:
    for part, f in [("train", "final_train_oof.csv"), ("student_test", "final_test.csv")]:
        fs[part]["final_decoder"] = br[part]["bridge"]
        shutil.copy(RES / f"bridge_{part}_bridge.csv", RES / f)
fs["bridge_adopted"] = bool(br["adopted"])
fs["final_params"] = hp
json.dump(fs, open(RES / "final_summary.json", "w"), indent=1)
cfg = {"model": ch["selected"], "display_name": "LightGBM LambdaRank (v2 features)", "feature_set": "v2", "hmm_params": hp, "realign": True,
       "selection": {"scorer": "round-2 scorer chosen by 5-fold trip-grouped CV (results/tune_v2_variants.csv, final_choice.json); "
                               "hyperparameters = round-1 Bayesian-tuned values, kept after the round-2 search tied",
                     "decoder": "round-1 (beta, w) setting kept: Bayesian decoder tuning and a 2-parameter grid failed the held-out-"
                                "training-trips rule (results/decoder_best.json, final_summary.json); dropout bridging adopted "
                                "under the same kind of rule (results/bridge_eval.json)",
                     "realign": "selected on training OOF in round 1 (results/realign_summary.json)"}}
json.dump(cfg, open(ROOT / "models/final_config.json", "w"), indent=1)
f = fs["student_test"]["final_decoder"], fs["train"]["final_decoder"]
print("final config:", cfg["model"], hp)
print("student-test final top1 %.4f macroF1 %.4f | train OOF final top1 %.4f macroF1 %.4f" % (
    f[0]["final"]["top1"], f[0]["final"]["macro_f1"], f[1]["final"]["top1"], f[1]["final"]["macro_f1"]))
