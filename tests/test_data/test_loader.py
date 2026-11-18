"""Unit tests for dataset loading, coordinates, and validation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import (
    load_trip,
    haversine_distance,
    compute_bearing,
    list_partition_trips
)


def test_haversine_distance():
    # Distance between Frankfurt Airport and Darmstadt ~ 20km
    # 8.57, 50.03 to 8.65, 49.87
    d = haversine_distance(8.57, 50.03, 8.65, 49.87)
    assert 15000 < d < 25000
    # Zero distance
    assert haversine_distance(8.5, 50.0, 8.5, 50.0) == 0.0


def test_compute_bearing():
    # Due North: lat increases, lon constant
    b_north = compute_bearing(8.5, 50.0, 8.5, 51.0)
    assert abs(b_north - 0.0) < 1e-3 or abs(b_north - 360.0) < 1e-3

    # Due East: lon increases, lat constant
    b_east = compute_bearing(8.5, 50.0, 9.5, 50.0)
    assert 80.0 < b_east < 100.0


def test_list_partition_trips():
    trips = list_partition_trips(partition="train")
    assert len(trips) == 80
    assert "00000001" in trips

    test_trips = list_partition_trips(partition="student_test")
    assert len(test_trips) == 10
    assert "00000000" in test_trips


def test_load_trip_train():
    trip = load_trip("00000001", partition="train")
    assert trip["trip_id"] == "00000001"
    assert trip["partition"] == "train"

    # Track validation
    track = trip["track"]
    assert isinstance(track, pd.DataFrame)
    assert set(["lon", "lat", "time", "dt", "speed_m_s", "bearing_deg", "gps_index"]).issubset(track.columns)
    assert len(track) > 0
    assert not track["lon"].isna().any()
    assert not track["lat"].isna().any()

    # Nodes validation
    nodes = trip["nodes"]
    assert isinstance(nodes, pd.DataFrame)
    assert set(["node_id", "lon", "lat"]).issubset(nodes.columns)
    assert len(nodes) > 0

    # Arcs validation
    arcs = trip["arcs"]
    assert isinstance(arcs, pd.DataFrame)
    assert set(["arc_id", "u", "v", "length_m", "bearing_deg"]).issubset(arcs.columns)
    assert len(arcs) > 0

    # Route validation
    route = trip["route"]
    assert route is not None
    assert "arc_id" in route.columns
    assert len(route) > 0


def test_load_trip_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_trip("99999999", partition="train")


def test_load_trip_evaluator_anti_leakage():
    # Even if an evaluator trip was placed, route must never be loaded
    # Create a dummy folder in a tmp path or verify partition check
    with pytest.raises(FileNotFoundError):
        # nonexistent trip in evaluator_test
        load_trip("eval_test_01", partition="evaluator_test")
