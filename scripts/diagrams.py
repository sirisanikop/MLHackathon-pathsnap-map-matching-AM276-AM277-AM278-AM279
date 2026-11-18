"""Pipeline + deployment diagrams (monochrome). Output: figures/diagram_*.png"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap import plotstyle as ps  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ps.apply()
plt = ps.plt


def box(ax, x, y, w, h, title, sub="", dark=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=ps.INK if dark else "white", ec=ps.INK, lw=1.4))
    c = "white" if dark else ps.INK
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=c)
    if sub:
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=7.0, color=c if dark else ps.MID)


def arrow(ax, x0, y0, x1, y1, txt=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=ps.INK, lw=1.3))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, txt, ha="center", va="bottom", fontsize=7.2, color=ps.INK)


# pipeline
fig, ax = plt.subplots(figsize=(12, 3.1))
ax.set_xlim(0, 12.2); ax.set_ylim(0, 3.1); ax.axis("off")
steps = [("Raw trip files", ".track .nodes .arcs"), ("Projection\n+ QA", "local metres, gaps"),
         ("Candidates", "KD-tree,\n80 m, K<=24"), ("46 features", "distance, heading,\ncontext, topology"),
         ("Scorer", "LightGBM LambdaRank\np(arc | fix)"), ("HMM / Viterbi", "graph distances,\ngaps, breaks"),
         ("Route", "arc sequence +\nshortest-path fill")]
w, h, gap = 1.5, 1.35, 0.24
for i, (t, s) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    box(ax, x, 1.4, w, h, t, s, dark=(i in (4, 5)))
    if i:
        arrow(ax, x - gap + 0.02, 1.4 + h / 2, x - 0.02, 1.4 + h / 2)
ax.text(0.15, 0.95, "Training only:  .route  ->  monotone DP alignment of fixes to the reference arc sequence  ->  per-fix label "
        "(y = 1 for the aligned arc)  ->  5-fold trip-grouped CV, Optuna TPE, decoder grid + held-out checks", fontsize=8.5, color=ps.INK)
ax.text(0.15, 0.45, "Evaluation only (server side):  same alignment on the private reference -> top-1, macro-F1, continuity, route P/R/F1 "
        "(never fed back to the predictor)", fontsize=8.5, color=ps.INK)
fig.savefig(ROOT / "figures/diagram_pipeline.png")
plt.close(fig)

# deployment
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.2, 0.25), 5.2, 2.8, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F4F4F4", ec=ps.MID))
ax.text(0.4, 2.8, "Laptop 1  -  FastAPI (uvicorn, 0.0.0.0:8000)", fontsize=9.5, fontweight="bold")
box(ax, 0.45, 1.45, 1.5, 1.0, "Predictor", "features + model\n+ HMM", dark=True)
box(ax, 2.1, 1.45, 1.5, 1.0, "Trip store", "PATHSNAP_DATA /\nPARTITION")
box(ax, 3.75, 1.45, 1.45, 1.0, "Evaluator", "private .route\n(after predict)")
box(ax, 0.45, 0.45, 4.75, 0.65, "Request log  (console + logs/api_requests.log)")
ax.add_patch(FancyBboxPatch((7.3, 0.25), 2.55, 2.8, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F4F4F4", ec=ps.MID))
ax.text(7.45, 2.8, "Laptop 2  -  UI", fontsize=9.5, fontweight="bold")
box(ax, 7.5, 1.45, 2.15, 1.0, "Browser UI", "trip list, map,\nscores")
arrow(ax, 7.25, 2.15, 5.45, 2.15, "GET /trips\nPOST /predict {trip_id}")
arrow(ax, 5.45, 1.55, 7.25, 1.55, "route geometry\n+ scores")
fig.savefig(ROOT / "figures/diagram_deployment.png")
plt.close(fig)
print("ok")
