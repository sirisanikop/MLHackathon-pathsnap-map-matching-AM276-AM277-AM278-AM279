"""Folium maps from the actual pipeline outputs.

usage: python scripts/make_maps.py <partition> <trip_id> [<trip_id> ...]
       python scripts/make_maps.py --eda <partition> <trip_id>

Layers: road network, raw GPS fixes, reference route (dev partitions only),
classifier-only prediction, classifier+HMM/Viterbi route, wrongly matched fixes.
Output: maps/<partition>_<trip>.html  (and maps/eda_<partition>_<trip>.html)
"""
import json
import sys
from pathlib import Path

import folium
import numpy as np
from branca.element import Element

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_route, load_trip  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import collapse, continuity, route_overlap  # noqa: E402

MAPS = ROOT / "maps"
MAPS.mkdir(exist_ok=True)


def arcs_to_lines(trip, arc_ids):
    n = trip.nodes
    return [[(n[trip.arcs[a, 0], 1], n[trip.arcs[a, 0], 0]), (n[trip.arcs[a, 1], 1], n[trip.arcs[a, 1], 0])]
            for a in arc_ids]


def legend(items, title):
    rows = "".join(f'<div style="margin:2px 0"><span style="display:inline-block;width:26px;height:0;'
                   f'border-top:{w}px {s} {c};vertical-align:middle;margin-right:6px"></span>{t}</div>'
                   for t, c, w, s in items)
    return Element(f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;'
                   f'padding:10px 12px;border:1px solid #999;border-radius:4px;font:12px Helvetica,Arial;'
                   f'box-shadow:0 1px 4px rgba(0,0,0,.2)"><b>{title}</b>{rows}</div>')


def base_map(trip, max_arcs=8000):
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    a, b = trip.arc_geometry()
    lo, hi = trip.xy.min(0) - 120, trip.xy.max(0) + 120
    keep = np.where(((np.minimum(a, b) >= lo) & (np.maximum(a, b) <= hi)).all(1))[0]
    if len(keep) > max_arcs:  # keep the arcs closest to the track for file size
        from scipy.spatial import cKDTree
        mid = (a[keep] + b[keep]) / 2
        d, _ = cKDTree(trip.xy).query(mid)
        keep = keep[np.argsort(d)[:max_arcs]]
    fg = folium.FeatureGroup(name=f"Road network ({len(keep)} arcs near track)", show=True)
    folium.PolyLine(arcs_to_lines(trip, keep), color="#9e9e9e", weight=1, opacity=0.7).add_to(fg)
    fg.add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m


def gps_layer(trip, m, max_pts=2500):
    step = max(1, trip.n_obs // max_pts)
    fg = folium.FeatureGroup(name=f"Raw GPS fixes ({len(range(0, trip.n_obs, step))} of {trip.n_obs} shown)")
    for i in range(0, trip.n_obs, step):
        folium.CircleMarker((trip.track[i, 1], trip.track[i, 0]), radius=2.5, color="#1565c0", weight=0,
                            fill=True, fill_opacity=0.75,
                            tooltip=f"fix {i}, t={trip.track[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker((trip.track[0, 1], trip.track[0, 0]), tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((trip.track[-1, 1], trip.track[-1, 0]), tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)


def eda_map(part, tid):
    trip = load_trip(tid, part, with_route=True)
    m = base_map(trip)
    fg = folium.FeatureGroup(name=f"Reference route ({len(trip.route)} arcs)")
    folium.PolyLine(arcs_to_lines(trip, trip.route), color="#2e7d32", weight=6, opacity=0.6).add_to(fg)
    fg.add_to(m)
    gps_layer(trip, m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(legend([("Raw GPS fixes", "#1565c0", 4, "dotted"),
                                        ("Reference route", "#2e7d32", 6, "solid"),
                                        ("Road network", "#9e9e9e", 2, "solid")],
                                       f"EDA - {part} trip {tid}"))
    out = MAPS / f"eda_{part}_{tid}.html"
    m.save(out)
    return out


def prediction_map(part, tid, predictor, detail=False):
    res = predictor.predict_trip(tid, part, use_hmm=True)
    trip, cls, hmm, route = res["trip"], res["cls_pred"], res["final_pred"], res["route"]
    vit = res["hmm_pred"]
    m = base_map(trip)
    ref = load_route(tid, part)   # displayed for development trips only
    stats = {}
    if len(ref):
        lab, _, ld = align_route(trip.xy, trip.node_xy, trip.arcs, ref)
        rel = ld <= 60
        stats = dict(cls_top1=float((cls == lab)[rel].mean()), viterbi_top1=float((vit == lab)[rel].mean()),
                     hmm_top1=float((hmm == lab)[rel].mean()),
                     cls_cont=continuity(cls, trip.arcs), hmm_cont=continuity(hmm, trip.arcs),
                     hmm_route_f1=route_overlap(route, ref, res["graph"].alen)["route_f1"],
                     cls_route_f1=route_overlap(collapse(cls), ref, res["graph"].alen)["route_f1"])
        fg = folium.FeatureGroup(name="Reference route (dev data only)")
        folium.PolyLine(arcs_to_lines(trip, ref), color="#2e7d32", weight=8, opacity=0.45).add_to(fg)
        fg.add_to(m)
    gps_layer(trip, m)
    fg = folium.FeatureGroup(name="Classifier-only prediction (per fix)", show=detail)
    folium.PolyLine(arcs_to_lines(trip, collapse(cls)), color="#ef6c00", weight=3, opacity=0.9,
                    dash_array="6,6").add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route: classifier + HMM/Viterbi")
    folium.PolyLine(arcs_to_lines(trip, route), color="#c62828", weight=3, opacity=0.95).add_to(fg)
    fg.add_to(m)
    if len(ref):
        wrong = np.where(rel & (hmm != lab))[0]
        fg = folium.FeatureGroup(name=f"Fixes matched to a non-reference arc ({len(wrong)})", show=detail)
        for i in wrong[:: max(1, len(wrong) // 800)]:
            folium.CircleMarker((trip.track[i, 1], trip.track[i, 0]), radius=4, color="#000", weight=1.5,
                                fill=False, tooltip=f"fix {i}: pred arc {hmm[i]}, ref arc {lab[i]}").add_to(fg)
        fg.add_to(m)
    if detail and len(ref):
        # zoom to the 120-fix window with the most classifier-only errors (auto-selected, documented)
        err = (rel & (cls != lab)).astype(int)
        win = 120
        cs = np.convolve(err, np.ones(win, int), mode="valid")
        i0 = int(np.argmax(cs))
        seg_ll = trip.track[i0:i0 + win]
        m.fit_bounds([[seg_ll[:, 1].min(), seg_ll[:, 0].min()], [seg_ll[:, 1].max(), seg_ll[:, 0].max()]])
        stats["detail_window"] = [i0, i0 + win]
    folium.LayerControl(collapsed=False).add_to(m)
    txt = (f"<br>top-1: classifier {stats['cls_top1']:.3f} &rarr; HMM {stats['viterbi_top1']:.3f} &rarr; final {stats['hmm_top1']:.3f}"
           f"<br>route F1: classifier {stats['cls_route_f1']:.3f} &rarr; HMM {stats['hmm_route_f1']:.3f}") if stats else ""
    m.get_root().html.add_child(legend([("Raw GPS fixes", "#1565c0", 4, "dotted"),
                                        ("Reference route", "#2e7d32", 8, "solid"),
                                        ("Classifier-only", "#ef6c00", 3, "dashed"),
                                        ("Classifier + HMM route", "#c62828", 3, "solid"),
                                        ("Road network", "#9e9e9e", 2, "solid")],
                                       f"{part} trip {tid} ({predictor.cfg['model']}){txt}"))
    out = MAPS / f"{part}_{tid}{'_detail' if detail else ''}.html"
    m.save(out)
    json.dump(stats, open(MAPS / f"{part}_{tid}{'_detail' if detail else ''}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--eda":
        print(eda_map(args[1], args[2]))
    elif args[0] == "--detail":
        from pathsnap.pipeline import Predictor
        print(prediction_map(args[1], args[2], Predictor(), detail=True))
    else:
        from pathsnap.pipeline import Predictor
        p = Predictor()
        for tid in args[1:]:
            print(prediction_map(args[0], tid, p))
