"""Unit tests for spatial candidate road segment generation."""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import load_trip
from src.features.candidates import generate_candidates, build_road_spatial_index


def test_build_road_spatial_index():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lon": [8.59, 8.60, 8.61],
        "lat": [49.84, 49.85, 49.86]
    })
    arcs = pd.DataFrame({
        "arc_id": [0, 1],
        "u": [0, 1],
        "v": [1, 2]
    })
    tree, lines = build_road_spatial_index(nodes, arcs)
    assert len(lines) == 2
    assert tree is not None


def test_generate_candidates_real_trip():
    trip = load_trip("00000001", partition="train")
    track = trip["track"].head(50)  # fast subset
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    cands = generate_candidates(track, nodes, arcs, config={"radius_m": 45.0})

    assert isinstance(cands, pd.DataFrame)
    assert set(["gps_index", "candidate_arc_id"]).issubset(cands.columns)
    assert len(cands) >= len(track)

    # Every GPS observation must have at least one candidate (0% miss rate guarantee)
    covered_gps = set(cands["gps_index"].unique())
    assert covered_gps == set(range(len(track)))
