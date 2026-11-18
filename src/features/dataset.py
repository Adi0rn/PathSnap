"""Training dataset builder module.

Constructs feature matrices and ground-truth binary classification labels
from permitted training reference routes without data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

from src.data.loader import load_trip, list_partition_trips
from src.features.candidates import generate_candidates
from src.features.engineering import extract_candidate_features


FEATURE_COLUMNS = [
    "dist_to_arc",
    "projection_fraction",
    "dist_to_start_node",
    "dist_to_end_node",
    "arc_length",
    "heading_diff_deg",
    "cos_heading_diff",
    "sin_heading_diff",
    "gps_speed",
    "gps_delta_t",
    "dist_rank",
    "dist_ratio_to_nearest",
    "normalized_dist",
    "speed_length_ratio"
]


def map_gps_to_route_arcs(track: pd.DataFrame,
                          nodes: pd.DataFrame,
                          arcs: pd.DataFrame,
                          route: pd.DataFrame) -> Dict[int, int]:
    """Map each GPS observation to its closest reference route arc.

    Parameters
    ----------
    track : pd.DataFrame
        GPS trajectory with 'lon', 'lat'.
    nodes : pd.DataFrame
        Nodes with 'lon', 'lat'.
    arcs : pd.DataFrame
        Arcs with 'u', 'v'.
    route : pd.DataFrame
        Ground-truth reference route containing 'arc_id'.

    Returns
    -------
    dict: {gps_index: true_arc_id}
    """
    route_arcs = route["arc_id"].values
    unique_route_arcs = np.unique(route_arcs)

    u_idx = arcs["u"].values
    v_idx = arcs["v"].values
    node_lons = nodes["lon"].values
    node_lats = nodes["lat"].values

    route_lines = {
        aid: LineString([
            (node_lons[u_idx[aid]], node_lats[u_idx[aid]]),
            (node_lons[v_idx[aid]], node_lats[v_idx[aid]])
        ])
        for aid in unique_route_arcs
    }

    gps_to_arc = {}
    pts = [Point(lon, lat) for lon, lat in zip(track["lon"].values, track["lat"].values)]

    for i, pt in enumerate(pts):
        best_aid = min(unique_route_arcs, key=lambda aid: route_lines[aid].distance(pt))
        gps_to_arc[i] = int(best_aid)

    return gps_to_arc


def build_trip_features(trip_id: str,
                        partition: str = "train",
                        data_root: Union[str, Path] = "data/raw/Student folder") -> pd.DataFrame:
    """Extract candidate features and optional label for a single trip."""
    trip_data = load_trip(trip_id, partition=partition, data_root=data_root)
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    candidates = generate_candidates(track, nodes, arcs)
    features = extract_candidate_features(track, candidates, nodes, arcs, trip_id=trip_id)

    # Label only if reference route is provided and permitted
    if route is not None:
        gps_to_true_arc = map_gps_to_route_arcs(track, nodes, arcs, route)
        features["label"] = (features["candidate_arc_id"] == features["gps_index"].map(gps_to_true_arc)).astype(int)
    else:
        features["label"] = np.nan

    return features


def build_training_dataset(trip_ids: Optional[List[str]] = None,
                           data_root: Union[str, Path] = "data/raw/Student folder",
                           partition: str = "train",
                           max_trips: Optional[int] = None) -> pd.DataFrame:
    """Build complete training dataset across specified trips in the permitted partition.

    Parameters
    ----------
    trip_ids : list of str, optional
        List of trip IDs. If None, all trips in the partition are used.
    data_root : str or Path
        Dataset root path.
    partition : str
        Partition name ('train' or 'student_test').
    max_trips : int, optional
        Maximum trips to process (useful for rapid dev/testing).

    Returns
    -------
    pd.DataFrame:
        Consolidated candidate feature matrix with 'label'.
    """
    if partition == "evaluator_test":
        raise PermissionError("Evaluator testing partition must not be used for model training or feature building.")

    if trip_ids is None:
        trip_ids = list_partition_trips(data_root=data_root, partition=partition)

    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    dfs = []
    for tid in trip_ids:
        df_trip = build_trip_features(tid, partition=partition, data_root=data_root)
        dfs.append(df_trip)

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, groups) from candidate DataFrame for scikit-learn pipelines.

    Returns
    -------
    tuple: (X [features], y [labels], groups [trip_ids])
    """
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values.astype(int) if "label" in df.columns else None
    groups = df["trip_id"].values if "trip_id" in df.columns else None
    return X, y, groups
