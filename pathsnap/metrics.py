"""Metric definitions (units stated per function).

Arc IDs are trip-specific, so every observation-level macro average is
computed WITHIN a trip (classes = arcs occurring as reference or prediction
in that trip) and then averaged over trips (each trip weighted equally).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support

EPS = 1e-6


def normalise_scores(cands: pd.DataFrame, score_col="p"):
    """Convert per-candidate classifier probabilities into a distribution over
    each observation's candidate set (unit: candidate)."""
    s = cands[score_col].clip(EPS, 1.0)
    tot = s.groupby([cands.trip, cands.obs]).transform("sum")
    return s / tot


def top1(cands: pd.DataFrame, score_col="pn"):
    """Per-observation argmax candidate -> DataFrame(trip, obs, pred, pmax)."""
    idx = cands.groupby(["trip", "obs"])[score_col].idxmax()
    t = cands.loc[idx, ["trip", "obs", "arc", score_col]]
    return t.rename(columns={"arc": "pred", score_col: "pmax"}).reset_index(drop=True)


def candidate_metrics(cands, prob_col="p", rel_only=True):
    """Unit: candidate (binary: is this the reference arc?)."""
    c = cands[cands.reliable] if rel_only else cands
    y = c.y.to_numpy()
    p = c[prob_col].clip(EPS, 1 - EPS).to_numpy()
    yp = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average=None, labels=[0, 1], zero_division=0)
    return dict(cand_precision_pos=pr[1], cand_recall_pos=rc[1], cand_f1_pos=f1[1],
                cand_macro_f1=f1.mean(), cand_logloss=log_loss(y, p, labels=[0, 1]))


def observation_metrics(cands, obs, pred, pn_col="pn"):
    """Unit: observation (GPS fix). Only reliable observations are scored.
    Observations without candidates or whose reference arc is missing from the
    candidate set count as errors for accuracy/F1.
    pred: DataFrame(trip, obs, pred)."""
    o = obs[obs.reliable].merge(pred[["trip", "obs", "pred"]], on=["trip", "obs"], how="left")
    o["pred"] = o["pred"].fillna(-1).astype(np.int64)
    o["correct"] = o.pred == o.lab
    per_trip = []
    for tid, g in o.groupby("trip"):
        per_trip.append(dict(trip=tid, n=len(g), top1=g.correct.mean(),
                             macro_f1=f1_score(g.lab, g.pred, average="macro", zero_division=0),
                             weighted_f1=f1_score(g.lab, g.pred, average="weighted", zero_division=0)))
    pt = pd.DataFrame(per_trip)
    # observation log loss over the normalised candidate distribution
    c = cands[(cands.reliable) & (cands.y == 1)]
    ll = float(-np.log(c[pn_col].clip(EPS, 1)).mean()) if len(c) else np.nan
    return dict(obs_top1=o.correct.mean(), obs_macro_f1_trip=pt.macro_f1.mean(),
                obs_weighted_f1_trip=pt.weighted_f1.mean(),
                obs_top1_trip_mean=pt.top1.mean(), obs_logloss=ll,
                miss_rate=1 - obs[obs.reliable].in_cands.mean(), n_obs=len(o)), pt


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray):
    """Unit: consecutive observation pair. Fraction of pairs (i,i+1) whose
    predicted arcs are identical or directly connected head->tail."""
    p = pred_arcs[pred_arcs >= 0]
    if len(p) < 2:
        return np.nan
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_overlap(pred_route_arcs, ref_route, arc_len):
    """Unit: route (trip). Length-weighted overlap between the predicted arc set
    and the reference arc set: precision, recall, F1 (in metres)."""
    P = set(int(x) for x in pred_route_arcs if x >= 0)
    R = set(int(x) for x in ref_route)
    inter = sum(arc_len[a] for a in P & R)
    lp = sum(arc_len[a] for a in P)
    lr = sum(arc_len[a] for a in R)
    prec = inter / lp if lp > 0 else 0.0
    rec = inter / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(route_precision=prec, route_recall=rec, route_f1=f1)


def collapse(seq):
    seq = np.asarray(seq)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return seq
    keep = np.r_[True, seq[1:] != seq[:-1]]
    return seq[keep]
