"""Pre-declared selection rule between the untuned and the Bayesian-tuned round-2 scorer:
higher 5-fold trip-grouped CV top-1 under the standard protocol wins; if the difference is below
0.001 (far inside fold noise) the configuration with the smaller train-vs-validation gap is taken.
Student-test numbers are NOT used. output: results/final_choice.json
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
u, t = (json.load(open(RES / f"model_{m}.json")) for m in ("lgbm_v2_untuned", "lgbm_v2"))
cu, ct = u["cv_mean"], t["cv_mean"]
gu, gt = cu["train_obs_top1"] - cu["obs_top1"], ct["train_obs_top1"] - ct["obs_top1"]
d = ct["obs_top1"] - cu["obs_top1"]
if abs(d) < 0.001:
    pick, why = ("lgbm_v2", "CV top-1 tie (<0.001); tuned has the smaller train-validation gap") if gt < gu else \
                ("lgbm_v2_untuned", "CV top-1 tie (<0.001); untuned has the smaller train-validation gap")
else:
    pick, why = ("lgbm_v2", "higher CV top-1") if d > 0 else ("lgbm_v2_untuned", "higher CV top-1")
out = {"selected": pick, "reason": why, "cv_top1": {"untuned": cu["obs_top1"], "tuned": ct["obs_top1"]},
       "cv_macro_f1": {"untuned": cu["obs_macro_f1_trip"], "tuned": ct["obs_macro_f1_trip"]},
       "train_val_gap": {"untuned": gu, "tuned": gt}}
json.dump(out, open(RES / "final_choice.json", "w"), indent=1)
print(json.dumps(out, indent=1))
