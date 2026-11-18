"""Unit tests for HMM and Viterbi route reconstruction."""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import joblib

from src.data.loader import load_trip
from src.features.candidates import generate_candidates
from src.features.engineering import extract_candidate_features
from src.models.baselines import predict_candidates
from src.reconstruction.viterbi import (
    build_road_graph,
    compute_emission_log_prob,
    compute_transition_log_prob,
    decode_route,
    compare_classifier_vs_hmm
)


def test_build_road_graph():
    arcs = pd.DataFrame({
        "arc_id": [0, 1, 2],
        "u": [10, 20, 20],
        "v": [20, 30, 40],
        "length_m": [100.0, 150.0, 200.0]
    })
    G, edge_to_arc = build_road_graph(arcs)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G[10][20]["weight"] == 100.0
    assert edge_to_arc[(10, 20)] == 0


def test_compute_emission_log_prob():
    # Closer distance should have higher log probability
    p_close = compute_emission_log_prob(model_prob=0.8, dist_to_arc=5.0, cos_heading_diff=1.0)
    p_far = compute_emission_log_prob(model_prob=0.8, dist_to_arc=50.0, cos_heading_diff=1.0)
    assert p_close > p_far

    # Higher model probability should increase score
    p_high_prob = compute_emission_log_prob(model_prob=0.9, dist_to_arc=10.0, cos_heading_diff=1.0)
    p_low_prob = compute_emission_log_prob(model_prob=0.1, dist_to_arc=10.0, cos_heading_diff=1.0)
    assert p_high_prob > p_low_prob


def test_decode_route_on_trip():
    trip_data = load_trip("00000001", partition="train")
    track = trip_data["track"].head(20).copy()
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]

    model = joblib.load("models/tuned_model.joblib")
    cands = generate_candidates(track, nodes, arcs)
    feats = extract_candidate_features(track, cands, nodes, arcs, trip_id="00000001")
    preds = predict_candidates(feats, model)

    G, _ = build_road_graph(arcs)
    result = decode_route(track, preds, G, arcs=arcs)

    assert "pointwise_arcs" in result
    assert "full_route_arcs" in result
    assert len(result["pointwise_arcs"]) == len(track)
    assert len(result["full_route_arcs"]) >= 1


def test_compare_classifier_vs_hmm():
    trip_data = load_trip("00000001", partition="train")
    trip_data["track"] = trip_data["track"].head(25).copy()

    model = joblib.load("models/tuned_model.joblib")
    res = compare_classifier_vs_hmm(trip_data, model)

    assert "classifier_top1_accuracy" in res
    assert "hmm_decoded_accuracy" in res
    assert "classifier_continuity" in res
    assert "hmm_continuity" in res
    assert 0.0 <= res["hmm_decoded_accuracy"] <= 1.0
    assert 0.0 <= res["hmm_continuity"] <= 1.0
