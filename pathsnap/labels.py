"""Observation labels derived from the reference route.

The .route file lists arcs in travel order but has no per-observation
alignment. We derive one label per GPS observation with a *monotone
route-constrained alignment* (a DTW-style dynamic programme):

    choose j_1 <= j_2 <= ... <= j_n (indices into the route sequence)
    minimising sum_i dist(obs_i, route_arc[j_i])

Monotonicity respects the chronological order of the route (so loops and
repeated arcs are handled correctly) and arcs may be skipped (short arcs
traversed between two 1 Hz fixes). Observations whose aligned distance
exceeds OFFROUTE_M are flagged unreliable and excluded from training and
from observation-level metrics (their count is reported).

This module is ONLY used by training-label construction and the evaluation
step. The prediction pipeline (features.py / hmm.py / api predict) never
imports it.
"""
import numpy as np

from .geometry import monotone_align

OFFROUTE_M = 60.0


def align_route(xy, node_xy, arcs, route):
    a = node_xy[arcs[route, 0]]
    b = node_xy[arcs[route, 1]]
    idx, lab_dist = monotone_align(xy, a, b)
    return route[idx], idx, lab_dist


def observation_labels(trip):
    lab_arc, ridx, lab_dist = align_route(trip.xy, trip.node_xy, trip.arcs, trip.route)
    reliable = lab_dist <= OFFROUTE_M
    return lab_arc, ridx, lab_dist, reliable
