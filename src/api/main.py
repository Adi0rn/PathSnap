"""FastAPI backend application for PathSnap.

Serves health checks, trip inventory, and full map-matching prediction endpoints.
Supports the Two-Laptop distributed evaluation architecture.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.loader import load_trip, list_partition_trips
from src.features.candidates import generate_candidates
from src.features.engineering import extract_candidate_features
from src.models.baselines import predict_candidates
from src.reconstruction.viterbi import build_road_graph, decode_route
from src.features.dataset import map_gps_to_route_arcs


app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction Backend",
    version="1.0.0"
)

# Enable CORS for cross-laptop and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and graph cache
MODEL_CACHE = {}
GRAPH_CACHE = {}
DATA_ROOT = Path("data/raw/Student folder")


def get_model():
    """Retrieve or load the tuned model artifact."""
    if "model" not in MODEL_CACHE:
        tuned_path = Path("models/tuned_model.joblib")
        fallback_path = Path("models/RandomForest.joblib")

        if tuned_path.exists():
            MODEL_CACHE["model"] = joblib.load(tuned_path)
            MODEL_CACHE["name"] = "RandomForest_BayesianTuned"
        elif fallback_path.exists():
            MODEL_CACHE["model"] = joblib.load(fallback_path)
            MODEL_CACHE["name"] = "RandomForest_Untuned"
        else:
            raise RuntimeError("No trained model found in models/ directory.")
    return MODEL_CACHE["model"], MODEL_CACHE["name"]


class HealthResponse(BaseModel):
    status: str
    implementation: str
    model_loaded: bool
    model_name: Optional[str]
    timestamp: str


class TripsResponse(BaseModel):
    partition: str
    trips: List[str]
    count: int


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint reporting API and model readiness."""
    model_loaded = False
    model_name = None
    try:
        _, name = get_model()
        model_loaded = True
        model_name = name
    except Exception:
        pass

    return {
        "status": "healthy",
        "implementation": "complete",
        "model_loaded": model_loaded,
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }


@app.get("/trips", response_model=TripsResponse)
def list_trips(partition: str = "student_test"):
    """List all available trips in the specified partition (default: student_test)."""
    trips = list_partition_trips(data_root=DATA_ROOT, partition=partition)
    return {
        "partition": partition,
        "trips": trips,
        "count": len(trips)
    }


@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str, partition: str = "student_test"):
    """Execute complete map-matching and route reconstruction pipeline for a trip.

    Loads the trip observations and road network, generates spatial candidates,
    scores candidates with the trained ML classifier, decodes the optimal continuous route
    with Viterbi dynamic programming, and calculates server-side evaluation metrics.
    """
    trip_dir = DATA_ROOT / partition / trip_id
    if not trip_dir.exists():
        # Resilient auto-routing: check if trip exists in alternate partition
        alt_partition = "student_test" if partition == "train" else "train"
        if (DATA_ROOT / alt_partition / trip_id).exists():
            partition = alt_partition
            trip_dir = DATA_ROOT / partition / trip_id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trip '{trip_id}' not found in partition '{partition}' or '{alt_partition}'."
            )

    t0 = time.time()
    try:
        # 1. Load raw data
        trip_data = load_trip(trip_id, partition=partition, data_root=DATA_ROOT)
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        route = trip_data["route"]

        # 2. Candidate generation & Feature extraction
        model, model_name = get_model()
        candidates = generate_candidates(track, nodes, arcs)
        features = extract_candidate_features(track, candidates, nodes, arcs, trip_id=trip_id)

        # 3. Model scoring
        predictions = predict_candidates(features, model)

        # 4. HMM Viterbi decoding
        cache_key = f"{partition}_{trip_id}"
        if cache_key not in GRAPH_CACHE:
            GRAPH_CACHE[cache_key], _ = build_road_graph(arcs)
        road_graph = GRAPH_CACHE[cache_key]

        decoded = decode_route(track, predictions, road_graph, arcs=arcs)
        pointwise_arcs = decoded["pointwise_arcs"]
        full_route_arcs = decoded["full_route_arcs"]

        # 5. Calculate Server-Side Metrics if route is permitted
        obs_accuracy = None
        if route is not None:
            gps_to_true = map_gps_to_route_arcs(track, nodes, arcs, route)
            ground_truth_seq = [gps_to_true[i] for i in range(len(track))]
            hits = sum(1 for p, t in zip(pointwise_arcs, ground_truth_seq) if p == t)
            obs_accuracy = float(hits / len(track))

        # Route continuity
        u_map = dict(zip(arcs["arc_id"], arcs["u"]))
        v_map = dict(zip(arcs["arc_id"], arcs["v"]))
        cont_steps = 0
        for i in range(1, len(pointwise_arcs)):
            a1, a2 = pointwise_arcs[i - 1], pointwise_arcs[i]
            if a1 == a2 or v_map.get(a1) == u_map.get(a2):
                cont_steps += 1
        continuity = float(cont_steps / max(1, len(pointwise_arcs) - 1))

        # Prepare geometry for UI map rendering
        node_lons = nodes["lon"].values
        node_lats = nodes["lat"].values
        u_vals = arcs["u"].values
        v_vals = arcs["v"].values

        route_coords = []
        for aid in full_route_arcs:
            u, v = u_vals[aid], v_vals[aid]
            route_coords.append([[float(node_lats[u]), float(node_lons[u])],
                                 [float(node_lats[v]), float(node_lons[v])]])

        ground_truth_coords = []
        if route is not None:
            for aid in route["arc_id"].values:
                u, v = u_vals[aid], v_vals[aid]
                ground_truth_coords.append([[float(node_lats[u]), float(node_lons[u])],
                                            [float(node_lats[v]), float(node_lons[v])]])

        gps_points = [
            {"lat": float(lat), "lon": float(lon), "time": float(t), "speed_kmh": float(s * 3.6)}
            for lat, lon, t, s in zip(track["lat"], track["lon"], track["time"], track["speed_m_s"])
        ]

        duration_min = float((track["time"].iloc[-1] - track["time"].iloc[0]) / 60.0) if len(track) > 1 else 0.0
        elapsed_s = time.time() - t0

        return {
            "trip_id": trip_id,
            "partition": partition,
            "status": "success",
            "model_used": model_name,
            "inference_duration_seconds": float(elapsed_s),
            "n_observations": len(track),
            "n_reconstructed_arcs": len(full_route_arcs),
            "metrics": {
                "observation_accuracy": obs_accuracy,
                "route_continuity": continuity,
                "duration_minutes": duration_min
            },
            "gps_points": gps_points,
            "full_route_arcs": [int(a) for a in full_route_arcs],
            "route_segments_coords": route_coords,
            "ground_truth_coords": ground_truth_coords
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )
