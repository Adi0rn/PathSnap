"""End-to-end pipeline integration test.

Validates complete system flow:
Raw Trip -> Loader -> Candidates -> Features -> Trained Model -> HMM Viterbi -> Reconstructed Route
"""

import pytest
import joblib
from pathlib import Path

from src.data.loader import load_trip
from src.features.candidates import generate_candidates
from src.features.engineering import extract_candidate_features
from src.models.baselines import predict_candidates
from src.reconstruction.viterbi import build_road_graph, decode_route


def test_full_end_to_end_pipeline():
    # 1. Load raw trip
    trip_id = "00000052"
    trip_data = load_trip(trip_id, partition="student_test")
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    assert len(track) > 0
    assert len(nodes) > 0
    assert len(arcs) > 0

    # 2. Candidate generation
    candidates = generate_candidates(track, nodes, arcs, config={"radius_m": 45.0})
    assert len(candidates) >= len(track)
    assert set(candidates["gps_index"].unique()) == set(range(len(track)))

    # 3. Feature extraction
    features = extract_candidate_features(track, candidates, nodes, arcs, trip_id=trip_id)
    assert len(features) == len(candidates)
    assert not features.isna().any().any()

    # 4. Model scoring
    model_path = Path("models/tuned_model.joblib")
    if not model_path.exists():
        model_path = Path("models/RandomForest.joblib")
    model = joblib.load(model_path)

    predictions = predict_candidates(features, model)
    assert "probability" in predictions.columns
    assert "top1_predicted" in predictions.columns

    # 5. HMM Viterbi route decoding
    G, _ = build_road_graph(arcs)
    decoded = decode_route(track, predictions, G, arcs=arcs)

    assert "pointwise_arcs" in decoded
    assert "full_route_arcs" in decoded
    assert len(decoded["pointwise_arcs"]) == len(track)
    assert len(decoded["full_route_arcs"]) >= 1

    # 6. Verify sequential continuity
    u_map = dict(zip(arcs["arc_id"], arcs["u"]))
    v_map = dict(zip(arcs["arc_id"], arcs["v"]))
    p_arcs = decoded["pointwise_arcs"]

    continuous_count = 0
    for i in range(1, len(p_arcs)):
        a1, a2 = p_arcs[i - 1], p_arcs[i]
        if a1 == a2 or v_map.get(a1) == u_map.get(a2):
            continuous_count += 1

    continuity = continuous_count / max(1, len(p_arcs) - 1)
    # High continuity threshold
    assert continuity >= 0.85
