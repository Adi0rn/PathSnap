"""Unit tests for feature extraction and training dataset construction."""

import pytest
import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.features.candidates import generate_candidates
from src.features.engineering import extract_candidate_features
from src.features.dataset import build_trip_features, FEATURE_COLUMNS


def test_extract_candidate_features():
    trip = load_trip("00000001", partition="train")
    track = trip["track"].head(30)
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    cands = generate_candidates(track, nodes, arcs)
    feats = extract_candidate_features(track, cands, nodes, arcs, trip_id="00000001")

    assert not feats.empty
    assert "trip_id" in feats.columns
    assert set(FEATURE_COLUMNS).issubset(feats.columns)

    # Ensure no NaN or infinite values
    assert not feats[FEATURE_COLUMNS].isna().any().any()
    assert np.all(np.isfinite(feats[FEATURE_COLUMNS].values))

    # Geometrical checks
    assert (feats["dist_to_arc"] >= 0.0).all()
    assert (feats["projection_fraction"] >= 0.0).all()
    assert (feats["projection_fraction"] <= 1.0).all()
    assert (feats["cos_heading_diff"] >= -1.0).all()
    assert (feats["cos_heading_diff"] <= 1.0).all()
    assert (feats["dist_rank"] >= 1.0).all()


def test_build_trip_features_with_labels():
    feats = build_trip_features("00000001", partition="train")
    assert "label" in feats.columns
    # Check that both positive and negative labels are present
    counts = feats["label"].value_counts()
    assert 1 in counts.index
    assert 0 in counts.index
    assert counts[1] > 0
    assert counts[0] > counts[1]
