"""Unit tests for exploratory data analysis functions."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import load_trip
from src.data.exploration import summarize_trip, summarize_partition


def test_summarize_trip():
    trip_data = load_trip("00000001", partition="train")
    # Take a slice for speed
    trip_data["track"] = trip_data["track"].head(30).copy()
    trip_data["arcs"] = trip_data["arcs"].head(50).copy()

    summary = summarize_trip(trip_data)

    assert summary["trip_id"] == "00000001"
    assert summary["gps_points"] == 30
    assert summary["dt_mean"] > 0
    assert "total_dist_km" in summary
    assert "duration_min" in summary
    assert "graph_components" in summary
