"""Raw-data loading and geometry helpers for PathSnap.

File formats (verified in scripts/eda.py):
  .track : lon lat t           (whitespace separated, no header)
  .nodes : lon lat             (row index = node id, 0-based)
  .arcs  : from_node to_node   (row index = arc id, 0-based, directed)
  .route : arc_id              (chronological reference arcs)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)

DEFAULT_DATA_ROOT = Path(os.environ.get(
    "PATHSNAP_DATA", Path(__file__).resolve().parents[2] / "Student folder"))


def _read(path: Path, dtype=float, ncol=None) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncol or 1), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray          # (n,3) lon, lat, t
    nodes: np.ndarray          # (N,2) lon, lat
    arcs: np.ndarray           # (A,2) from, to
    route: np.ndarray | None   # (R,) arc ids -- None when withheld
    # projected (local equirectangular, metres) -- filled by project()
    lat0: float = 0.0
    lon0: float = 0.0
    xy: np.ndarray = field(default=None, repr=False)        # track xy
    node_xy: np.ndarray = field(default=None, repr=False)   # node xy

    @property
    def n_obs(self):
        return len(self.track)

    def project(self):
        """Local equirectangular projection around the trip centroid.

        Over the ~10-20 km extent of a trip the scale error of this projection
        is far below GPS noise (<0.1 %), and it lets us use fast planar
        segment-projection maths in metres.
        """
        allpts = np.vstack([self.track[:, :2], self.nodes]) if len(self.track) else self.nodes
        self.lon0 = float(np.mean(allpts[:, 0]))
        self.lat0 = float(np.mean(allpts[:, 1]))
        self.xy = lonlat_to_xy(self.track[:, 0], self.track[:, 1], self.lon0, self.lat0)
        self.node_xy = lonlat_to_xy(self.nodes[:, 0], self.nodes[:, 1], self.lon0, self.lat0)
        return self

    def arc_geometry(self):
        a = self.node_xy[self.arcs[:, 0]]
        b = self.node_xy[self.arcs[:, 1]]
        return a, b


def lonlat_to_xy(lon, lat, lon0, lat0):
    k = np.pi / 180.0
    x = (np.asarray(lon) - lon0) * k * EARTH_R * np.cos(lat0 * k)
    y = (np.asarray(lat) - lat0) * k * EARTH_R
    return np.column_stack([x, y])


def haversine(lon1, lat1, lon2, lat2):
    k = np.pi / 180.0
    dlat = (lat2 - lat1) * k
    dlon = (lon2 - lon1) * k
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * k) * np.cos(lat2 * k) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    root = Path(root or DEFAULT_DATA_ROOT)
    d = root / partition
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def load_trip(trip_id: str, partition: str, root: Path | None = None,
              with_route: bool = True) -> Trip:
    """Load one trip. with_route=False guarantees the reference route is never
    read (used by the prediction path)."""
    root = Path(root or DEFAULT_DATA_ROOT)
    d = root / partition / trip_id
    track = _read(d / f"{trip_id}.track", float, 3)
    nodes = _read(d / f"{trip_id}.nodes", float, 2)
    arcs = _read(d / f"{trip_id}.arcs", np.int64, 2)
    route = None
    if with_route and (d / f"{trip_id}.route").exists():
        route = _read(d / f"{trip_id}.route", np.int64, 1).ravel()
    t = Trip(trip_id, partition, track, nodes, arcs, route)
    return t.project()


def load_route(trip_id: str, partition: str, root: Path | None = None) -> np.ndarray:
    """Reference route only -- used exclusively by label/evaluation code."""
    root = Path(root or DEFAULT_DATA_ROOT)
    p = root / partition / trip_id / f"{trip_id}.route"
    return _read(p, np.int64, 1).ravel()
