"""Model comparison figures + consolidated tables from saved results.

outputs: figures/cmp_*.png, figures/hist_*.png, figures/cm_*.png, figures/optuna_*.png,
         figures/hmm_*.png, results/comparison_table.csv, results/tuned_vs_untuned.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap import plotstyle as ps  # noqa: E402
from pathsnap.models import NAMES  # noqa: E402

ps.apply()
plt = ps.plt
RES, FIG = ROOT / "results", ROOT / "figures"
ORDER = [m for m in ["logreg", "gnb", "knn", "rf", "lgbm", "mlp", "lgbm_tuned"] if (RES / f"model_{m}.json").exists()]
BASE = [m for m in ORDER if m != "lgbm_tuned"]
M = {m: json.load(open(RES / f"model_{m}.json")) for m in ORDER}
SHORT = {"logreg": "LogReg", "gnb": "GaussNB", "knn": "KNN", "rf": "RandForest", "lgbm": "LightGBM",
         "mlp": "MLP", "lgbm_tuned": "LightGBM\ntuned"}


def cfg_str(c):
    skip = {"family", "loss", "scaler"}
    return ", ".join(f"{k}={v}" for k, v in c.items() if k not in skip)


# ---------------- consolidated table ----------------
rows = []
for m in ORDER:
    d = M[m]
    cv, sd, te = d["cv_mean"], d["cv_std"], d["student_test"]
    rows.append(dict(model=m, name=NAMES[m], family=d["config"]["family"], config=cfg_str(d["config"]),
                     train_stride=d["train_stride"], final_train_rows=d["final_train_rows"],
                     cv_macro_f1=cv["obs_macro_f1_trip"], cv_macro_f1_std=sd["obs_macro_f1_trip"],
                     cv_top1=cv["obs_top1"], cv_top1_std=sd["obs_top1"],
                     cv_train_top1=cv.get("train_obs_top1"), cv_cand_logloss=cv["cand_logloss"],
                     test_top1=te["obs_top1"], test_macro_f1=te["obs_macro_f1_trip"],
                     test_weighted_f1=te["obs_weighted_f1_trip"],
                     test_obs_logloss=te["obs_logloss"], test_cand_logloss=te["cand_logloss"],
                     test_cand_precision=te["cand_precision_pos"], test_cand_recall=te["cand_recall_pos"],
                     test_cand_f1=te["cand_f1_pos"], test_cand_macro_f1=te["cand_macro_f1"],
                     test_continuity=te["route_continuity"], miss_rate=te["miss_rate"],
                     fit_time_s=d["final_fit_time_s"], infer_time_s=te["infer_time_s"],
                     infer_us_per_obs=1e6 * te["infer_time_s"] / 19160))
T = pd.DataFrame(rows)
hab = RES / "hmm_ablation_models.csv"
if hab.exists():
    H = pd.read_csv(hab)
    T = T.merge(H[["model", "hmm_top1", "hmm_continuity", "cls_route_f1", "hmm_route_f1"]], on="model", how="left")
T.to_csv(RES / "comparison_table.csv", index=False)
print(T[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_obs_logloss", "test_continuity",
         "infer_time_s"]].round(4).to_string())

x = np.arange(len(BASE))
B = T.set_index("model").loc[BASE]
labels = [SHORT[m] for m in BASE]

# CV macro F1 with fold std
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.bar(x, B.cv_macro_f1, yerr=B.cv_macro_f1_std, color=ps.INK, width=0.6, capsize=4,
              error_kw=dict(ecolor=ps.MID, lw=1.2))
ps.label_bars(ax, bars, pad=12)
ax.set_xticks(x, labels)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("macro-F1 (per-trip, averaged)")
ax.set_title("5-fold trip-grouped CV observation macro-F1 (train, 80 trips; bars = fold std)")
fig.savefig(FIG / "cmp_cv_macro_f1.png")
plt.close(fig)

# student-test top-1 and macro F1
fig, ax = plt.subplots(figsize=(7.5, 3.4))
w = 0.38
b1 = ax.bar(x - w / 2, B.test_top1, w, color=ps.INK, label="top-1 accuracy")
b2 = ax.bar(x + w / 2, B.test_macro_f1, w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="macro-F1")
ps.label_bars(ax, b1, size=7)
ps.label_bars(ax, b2, size=7)
ax.set_xticks(x, labels)
ax.set_ylim(0.6, 1.0)
ax.set_title("Student-test (10 trips) observation-level, classifier only")
ax.legend(loc="upper left", ncol=2)
fig.savefig(FIG / "cmp_test_top1_f1.png")
plt.close(fig)

# log loss
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
b = axes[0].bar(x, B.test_obs_logloss, color=ps.INK, width=0.6)
ps.label_bars(axes[0], b, size=7)
axes[0].set_xticks(x, labels, fontsize=8)
axes[0].set_title("Observation log loss (student-test)")
axes[0].set_ylabel("-mean log p(reference arc)")
b = axes[1].bar(x, B.test_cand_logloss, color=ps.MID, width=0.6)
ps.label_bars(axes[1], b, size=7)
axes[1].set_xticks(x, labels, fontsize=8)
axes[1].set_title("Candidate binary log loss (student-test)")
fig.tight_layout()
fig.savefig(FIG / "cmp_logloss.png")
plt.close(fig)

# overfitting: train vs CV top-1
fig, ax = plt.subplots(figsize=(7.5, 3.3))
b1 = ax.bar(x - w / 2, B.cv_train_top1, w, color=ps.LIGHT, edgecolor=ps.MID, label="in-sample (training folds)")
b2 = ax.bar(x + w / 2, B.cv_top1, w, color=ps.INK, label="CV validation folds")
ps.label_bars(ax, b1, size=7)
ps.label_bars(ax, b2, size=7)
ax.set_xticks(x, labels)
ax.set_ylim(0.6, 1.02)
ax.set_title("Over/under-fitting check: observation top-1, train vs validation (CV)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.savefig(FIG / "cmp_train_vs_cv.png")
plt.close(fig)

# timing
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
b = axes[0].bar(x, B.fit_time_s, color=ps.INK, width=0.6)
ps.label_bars(axes[0], b, fmt="{:.1f}", size=7)
axes[0].set_xticks(x, labels, fontsize=8)
axes[0].set_yscale("log")
axes[0].set_title("Final fit time (s, log)")
b = axes[1].bar(x, B.infer_time_s, color=ps.MID, width=0.6)
ps.label_bars(axes[1], b, fmt="{:.2f}", size=7)
axes[1].set_xticks(x, labels, fontsize=8)
axes[1].set_yscale("log")
axes[1].set_title("Scoring time for all student-test candidates (s, log)")
fig.tight_layout()
fig.savefig(FIG / "cmp_time.png")
plt.close(fig)

# per-trip heat table (student-test top-1)
P = pd.concat([pd.read_csv(RES / f"test_pertrip_{m}.csv", dtype={"trip": str}).assign(model=SHORT[m].replace("\n", " "))
               for m in BASE])
piv = P.pivot(index="model", columns="trip", values="top1").loc[[SHORT[m] for m in BASE]]
fig, ax = plt.subplots(figsize=(10, 3.2))
im = ax.imshow(piv.values, cmap="Greys", vmin=0.6, vmax=1.0, aspect="auto")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > 0.82 else "black")
ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=7, rotation=30)
ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
ax.grid(False)
ax.set_title("Per-trip student-test top-1 accuracy (classifier only)")
fig.colorbar(im, ax=ax, fraction=0.025)
fig.savefig(FIG / "cmp_pertrip_heat.png")
plt.close(fig)
piv.to_csv(RES / "test_pertrip_top1_matrix.csv")

# confusion matrices (candidate level, binary, threshold 0.5) on student-test
fig, axes = plt.subplots(2, 3, figsize=(8.5, 6))
axes = axes.ravel()
for ax, m in zip(axes, BASE):
    s = pd.read_pickle(RES / f"test_scores_{m}.pkl")
    s = s[s.reliable]
    yp = (s.p >= 0.5).astype(int)
    cm = np.array([[((s.y == a) & (yp == b)).sum() for b in (0, 1)] for a in (0, 1)])
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.3f}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["other", "ref"], fontsize=7)
    ax.set_yticks([0, 1], ["other", "ref"], fontsize=7)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_title(SHORT[m].replace("\n", " "), fontsize=9)
    ax.grid(False)
axes[0].set_ylabel("true", fontsize=8)
axes[3].set_ylabel("true", fontsize=8)
fig.suptitle("Candidate-level confusion matrices (student-test, threshold 0.5, row-normalised)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "cm_candidates.png")
plt.close(fig)

# epoch-wise histories (only models that produce them)
for m in ["mlp", "lgbm"]:
    f = RES / f"history_{m}.csv"
    if not f.exists():
        continue
    h = pd.read_csv(f)
    g = h.groupby("epoch")[["train_logloss", "val_logloss"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    e = g.index.to_numpy()
    for col, ls, lab in [("train_logloss", "-", "training"), ("val_logloss", "--", "validation fold")]:
        mu, sdv = g[(col, "mean")], g[(col, "std")]
        ax.plot(e, mu, ls, color=ps.INK, label=lab)
        ax.fill_between(e, mu - sdv, mu + sdv, color=ps.LIGHT, alpha=0.5, lw=0)
    ax.set_xlabel("epoch" if m == "mlp" else "boosting round")
    ax.set_ylabel("binary log loss (candidate)")
    ax.set_title(f"{NAMES[m]}: training vs validation loss (mean ± std over 5 CV folds)")
    ax.legend()
    fig.savefig(FIG / f"hist_{m}.png")
    plt.close(fig)

# ---------------- Bayesian optimisation ----------------
ob = RES / "optuna_best.json"
if ob.exists():
    best = json.load(open(ob))
    tr = pd.read_csv(RES / "optuna_trials.csv")
    tr = tr[tr.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(tr.number, tr.value, s=18, color=ps.MID, label="trial CV top-1")
    ax.plot(tr.number, tr.value.cummax(), color=ps.INK, label="best so far")
    ax.axhline(best["untuned_same_protocol"]["cv_top1"], color=ps.INK, ls=":", lw=1.2, label="untuned config")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV observation top-1")
    ax.set_title(f"Optuna TPE search on LightGBM ({len(tr)} trials, 5-fold trip-grouped CV)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "optuna_history.png")
    plt.close(fig)
    try:
        import optuna
        import joblib
        study = joblib.load(RES / "optuna_study.pkl")
        imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
        fig, ax = plt.subplots(figsize=(6, 3))
        k = list(imp.keys())[::-1]
        ax.barh(k, [imp[i] for i in k], color=ps.INK)
        ax.set_title("Hyperparameter importance (fANOVA over completed trials)")
        fig.savefig(FIG / "optuna_importance.png")
        plt.close(fig)
        pd.Series(imp).to_csv(RES / "optuna_importance.csv")
    except Exception as e:  # noqa: BLE001
        print("importance failed", e)
    if "lgbm_tuned" in M:
        rows = []
        for m in ["lgbm", "lgbm_tuned"]:
            d = M[m]
            rows.append(dict(model=m, cv_macro_f1=d["cv_mean"]["obs_macro_f1_trip"], cv_top1=d["cv_mean"]["obs_top1"],
                             test_top1=d["student_test"]["obs_top1"], test_macro_f1=d["student_test"]["obs_macro_f1_trip"],
                             test_obs_logloss=d["student_test"]["obs_logloss"],
                             test_cand_logloss=d["student_test"]["cand_logloss"],
                             test_continuity=d["student_test"]["route_continuity"],
                             infer_time_s=d["student_test"]["infer_time_s"], fit_time_s=d["final_fit_time_s"]))
        TV = pd.DataFrame(rows)
        TV.to_csv(RES / "tuned_vs_untuned.csv", index=False)
        print(TV.round(4).to_string())
        fig, ax = plt.subplots(figsize=(7, 3.2))
        mets = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1"]
        xx = np.arange(len(mets))
        b1 = ax.bar(xx - w / 2, TV.iloc[0][mets].astype(float), w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="untuned")
        b2 = ax.bar(xx + w / 2, TV.iloc[1][mets].astype(float), w, color=ps.INK, label="Bayesian-tuned")
        ps.label_bars(ax, b1, fmt="{:.4f}", size=7)
        ps.label_bars(ax, b2, fmt="{:.4f}", size=7)
        ax.set_xticks(xx, ["CV macro-F1", "CV top-1", "test top-1", "test macro-F1"])
        ax.set_ylim(0.8, 0.95)
        ax.set_title("LightGBM untuned vs tuned (same folds, data and metrics)")
        ax.legend(loc="upper left", ncol=2)
        fig.savefig(FIG / "optuna_tuned_vs_untuned.png")
        plt.close(fig)

# ---------------- HMM ----------------
if hab.exists():
    H = pd.read_csv(hab).set_index("model")
    HM = [m for m in ORDER if m in H.index]
    xx = np.arange(len(HM))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (a, b, t) in zip(axes, [("cls_top1", "hmm_top1", "observation top-1"),
                                    ("cls_continuity", "hmm_continuity", "route continuity"),
                                    ("cls_route_f1", "hmm_route_f1", "route F1 (length-weighted)")]):
        b1 = ax.bar(xx - w / 2, H.loc[HM, a], w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="classifier only")
        b2 = ax.bar(xx + w / 2, H.loc[HM, b], w, color=ps.INK, label="classifier + HMM")
        ax.set_xticks(xx, [SHORT[m] for m in HM], fontsize=7)
        lo = min(H.loc[HM, a].min(), H.loc[HM, b].min())
        ax.set_ylim(max(0, lo - 0.05), 1.0)
        ax.set_title(f"Student-test {t}")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "hmm_ablation_models.png")
    plt.close(fig)
    pt = pd.read_csv(RES / "hmm_test_pertrip.csv", dtype={"trip": str})
    sel = json.load(open(RES / "hmm_summary.json"))["selected_model"]
    q = pt[pt.model == sel]
    xt = np.arange(len(q))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, (a, b, t) in zip(axes, [("cls_top1", "hmm_top1", "observation top-1"), ("cls_route_f1", "hmm_route_f1", "route F1")]):
        b1 = ax.bar(xt - w / 2, q[a], w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="classifier only")
        b2 = ax.bar(xt + w / 2, q[b], w, color=ps.INK, label="classifier + HMM")
        ax.set_xticks(xt, q.trip, fontsize=7, rotation=30)
        ax.set_ylim(max(0, min(q[a].min(), q[b].min()) - 0.05), 1.0)
        ax.set_title(f"Per-trip {t} ({NAMES[sel]}, student-test)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "hmm_pertrip.png")
    plt.close(fig)
    G = pd.read_csv(RES / "hmm_grid.csv")
    pv = G.pivot(index="w_emit", columns="beta", values="hmm_top1")
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.imshow(pv.values, cmap="Greys", aspect="auto", vmin=pv.values.min() - 0.01, vmax=pv.values.max())
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.4f}", ha="center", va="center", fontsize=8,
                    color="white" if pv.values[i, j] > pv.values.mean() else "black")
    ax.set_xticks(range(pv.shape[1]), [f"β={b:g}" for b in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"w={x:g}" for x in pv.index])
    ax.grid(False)
    ax.set_title("Decoder grid: OOF top-1 on 80 training trips")
    fig.savefig(FIG / "hmm_grid.png")
    plt.close(fig)
print("figures written")

# ---------------- feature importance (gain) of the LightGBM models ----------------
import joblib  # noqa: E402
from pathsnap.candidates import FEATURES  # noqa: E402

for m in ["lgbm", "lgbm_tuned"]:
    f = ROOT / "models" / f"{m}.joblib"
    if not f.exists():
        continue
    mdl = joblib.load(f)
    gain = mdl.m.booster_.feature_importance("gain")
    s = pd.Series(gain / gain.sum(), index=FEATURES).sort_values()
    s.to_csv(RES / f"feature_importance_{m}.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(s.index, s.values, color=ps.INK)
    ax.set_xlabel("share of total split gain")
    ax.set_title(f"{NAMES[m]}: feature importance (gain)")
    ax.tick_params(axis="y", labelsize=8)
    fig.savefig(FIG / f"feat_importance_{m}.png")
    plt.close(fig)
print("feature importance written")

# ---------------- round 2: variant ladder, second Bayesian search, decoder tuning, final ladder ----------------
if (RES / "final_summary.json").exists():
    V = pd.read_csv(RES / "tune_v2_variants.csv")
    lab = ["A0  v1 cand.\nv1 feat.\nbinary", "A1  K=24\nv1 feat.\nbinary", "B  K=24\nv2 feat.\nbinary", "C1  v2 feat.\nlambdarank",
           "C2  v2 feat.\nrank_xendcg", "D  lambdarank\nall fixes"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    xx = np.arange(len(V))
    b1 = ax.bar(xx - w / 2, V.cv_top1, w, yerr=V.cv_top1_std, color=ps.INK, capsize=3, error_kw=dict(ecolor=ps.MID, lw=1), label="CV top-1")
    b2 = ax.bar(xx + w / 2, V.cv_macro_f1, w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="CV macro-F1")
    for b_, v_ in zip(b1, V.cv_top1):
        ax.annotate(f"{v_:.4f}", (b_.get_x() + b_.get_width() / 2, 0.842), ha="center", fontsize=7, color="white", rotation=90, va="bottom")
    ps.label_bars(ax, b2, fmt="{:.3f}", size=7)
    ax.set_xticks(xx, lab[:len(V)], fontsize=7)
    ax.set_ylim(0.84, 0.96)
    ax.set_title("Round 2: candidate set, features and objective (same 5 trip-grouped folds, 80 training trips)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.savefig(FIG / "v2_variants.png")
    plt.close(fig)

    o2 = json.load(open(RES / "optuna_v2_best.json"))
    t2 = pd.read_csv(RES / "optuna_v2_trials.csv")
    t2 = t2[t2.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(t2.number, t2.value, s=18, color=ps.MID, label="trial CV top-1")
    ax.plot(t2.number, t2.value.cummax(), color=ps.INK, label="best so far")
    ax.axhline(o2["untuned_same_protocol"]["cv_top1"], color=ps.INK, ls=":", lw=1.2, label="untuned config")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV observation top-1")
    ax.set_title(f"Round-2 Optuna TPE search, LambdaRank + v2 features ({len(t2)} trials)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "optuna_v2_history.png")
    plt.close(fig)

    dt_ = pd.read_csv(RES / "decoder_trials.csv")
    dt_ = dt_[dt_.state == "COMPLETE"]
    db = json.load(open(RES / "decoder_best.json"))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(dt_.number, dt_.value, s=18, color=ps.MID, label="trial: mean per-trip final top-1")
    ax.plot(dt_.number, dt_.value.cummax(), color=ps.INK, label="best so far")
    ax.axhline(db["round1_tuning_trip_mean_top1"], color=ps.INK, ls=":", lw=1.2, label="round-1 decoder setting")
    ax.set_ylim(max(dt_.value.min(), db["round1_tuning_trip_mean_top1"] - 0.02), dt_.value.max() + 0.002)
    ax.set_xlabel("trial")
    ax.set_ylabel("mean per-trip final top-1, 64 tuning trips")
    ax.set_title(f"Bayesian decoder tuning (Optuna TPE, {len(dt_)} trials)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "decoder_tuning_history.png")
    plt.close(fig)

    fs = json.load(open(RES / "final_summary.json"))
    r1 = json.load(open(RES / "realign_summary.json"))
    base = json.load(open(RES / "baseline_nearest.json"))
    stages = ["nearest\nsegment", "R1 scorer\n(v1 feat.)", "R1 + HMM", "R1 final\n(+ re-align)", "R2 scorer\n(v2 feat., rank)", "R2 + HMM", "R2 final"]
    te = [base["student_test"]["obs_top1"], r1["student_test"]["cls"]["top1"], r1["student_test"]["hmm"]["top1"], r1["student_test"]["realign"]["top1"],
          fs["student_test"]["final_decoder"]["cls"]["top1"], fs["student_test"]["final_decoder"]["hmm"]["top1"], fs["student_test"]["final_decoder"]["final"]["top1"]]
    trn = [base["cv_mean"]["obs_top1"], r1["train"]["cls"]["top1"], r1["train"]["hmm"]["top1"], r1["train"]["realign"]["top1"],
           fs["train"]["final_decoder"]["cls"]["top1"], fs["train"]["final_decoder"]["hmm"]["top1"], fs["train"]["final_decoder"]["final"]["top1"]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xx = np.arange(len(stages))
    b1 = ax.bar(xx - w / 2, trn, w, color=ps.LIGHT, edgecolor=ps.MID, hatch="///", label="80 training trips (CV / out-of-fold)")
    b2 = ax.bar(xx + w / 2, te, w, color=ps.INK, label="10 student-test trips")
    ps.label_bars(ax, b1, size=7)
    ps.label_bars(ax, b2, size=7)
    ax.set_xticks(xx, stages, fontsize=8)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("observation top-1")
    ax.set_title("Accuracy ladder: every stage on the same trips and labels")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.savefig(FIG / "final_ladder.png")
    plt.close(fig)
    print("round-2 figures written")
