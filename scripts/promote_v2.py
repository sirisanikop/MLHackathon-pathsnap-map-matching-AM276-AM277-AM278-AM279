"""Multi-fidelity promotion step for the round-2 Bayesian search: the search scored trials on a stride-9
training subsample; here the top-5 trials are re-scored with the standard protocol (every 3rd fix, same 5
folds). The best full-fidelity configuration is written to results/optuna_v2_best.json["promoted_params"].
output: results/optuna_v2_promotion.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tune_v2 as T  # noqa: E402
from pathsnap.candidates import FEATURES_V2  # noqa: E402

best = json.load(open(T.RES / "optuna_v2_best.json"))
tr = pd.read_csv(T.RES / "optuna_v2_trials.csv")
tr = tr[tr.state == "COMPLETE"].sort_values("value", ascending=False).head(5)
rows = []
for _, r in tr.iterrows():
    p = {c.replace("params_", ""): r[c] for c in tr.columns if c.startswith("params_")}
    for k in ("n_estimators", "num_leaves", "min_child_samples", "lambdarank_truncation_level"):
        p[k] = int(p[k])
    p["subsample_freq"] = 1
    res = T.cv(f"trial {int(r.number)}", "v2", FEATURES_V2, best["objective_family"], stride=3, params=p, quiet=True)
    rows.append(dict(trial=int(r.number), search_cv_top1=r.value, full_cv_top1=res["cv_top1"], full_cv_top1_std=res["cv_top1_std"],
                     full_cv_macro_f1=res["cv_macro_f1"], full_cv_logloss=res["cv_obs_logloss"], params=json.dumps(p)))
    print(rows[-1]["trial"], round(rows[-1]["search_cv_top1"], 4), round(rows[-1]["full_cv_top1"], 4), round(rows[-1]["full_cv_macro_f1"], 4), flush=True)
d = pd.DataFrame(rows)
d.to_csv(T.RES / "optuna_v2_promotion.csv", index=False)
b = d.sort_values("full_cv_top1", ascending=False).iloc[0]
best["promoted_params"] = json.loads(b.params)
best["promoted_trial"] = int(b.trial)
best["promoted_full_cv_top1"] = float(b.full_cv_top1)
json.dump(best, open(T.RES / "optuna_v2_best.json", "w"), indent=1)
print("promoted trial", int(b.trial), "full-protocol CV top-1", round(b.full_cv_top1, 4))
