"""Shared monochrome matplotlib style (prints well, works in a B/W deck)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#111111"
MID = "#666666"
LIGHT = "#BBBBBB"
GRID = "#E6E6E6"
SERIES = ["#111111", "#8C8C8C", "#C8C8C8", "#4D4D4D"]
HATCH = ["", "///", "...", "xx"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": MID, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MID, "ytick.color": MID, "legend.frameon": False,
        "lines.linewidth": 2,
    })


def label_bars(ax, bars, fmt="{:.3f}", pad=3, size=8):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom", fontsize=size, color=INK)
