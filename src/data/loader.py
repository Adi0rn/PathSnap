"""Dataset loading and validation module.

Handles reading raw .track, .nodes, .arcs, and .route files.
Implements coordinate calculations and strict anti-leakage policies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute geodesic distance in meters between two WGS84 coordinate pairs using Haversine formula."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_METERS * c)


def compute_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute forward azimuth / bearing in degrees [0, 360) from point 1 to point 2."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return float(bearing)


def list_partition_trips(data_root: Union[str, Path] = "data/raw/Student folder", partition: str = "train") -> List[str]:
    """List all available trip IDs within a given partition directory."""
    part_dir = Path(data_root) / partition
    if not part_dir.exists():
        return []
    return sorted([p.name for p in part_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])


def load_trip(trip_id: str,
              partition: str = "train",
              data_root: Union[str, Path] = "data/raw/Student folder") -> Dict[str, Union[str, pd.DataFrame, Optional[pd.DataFrame]]]:
    """Load a trip's track, nodes, arcs, and reference route (only if permitted).

    Parameters
    ----------
    trip_id : str
        The trip identifier (e.g. '00000001').
    partition : str
        Partition name: 'train', 'student_test', or 'evaluator_test'.
    data_root : str or Path
        Root directory containing dataset partition subfolders.

    Returns
    -------
    dict:
        {
            'trip_id': trip_id,
            'partition': partition,
            'track': pd.DataFrame [lon, lat, time, dt, speed_m_s, bearing_deg],
            'nodes': pd.DataFrame [node_id, lon, lat],
            'arcs': pd.DataFrame [arc_id, u, v, length_m, bearing_deg],
            'route': pd.DataFrame [arc_id, step] or None
        }
    """
    if partition == "evaluator_test":
        # Anti-leakage rule: evaluator ground truth must never be loaded into student pipeline
        allow_route = False
    elif partition in ("train", "student_test"):
        allow_route = True
    else:
        raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'student_test', or 'evaluator_test'.")

    trip_dir = Path(data_root) / partition / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    for req_file in (track_file, nodes_file, arcs_file):
        if not req_file.exists():
            raise FileNotFoundError(f"Required trip file missing: {req_file}")
        if req_file.stat().st_size == 0:
            raise ValueError(f"Required trip file is empty: {req_file}")

    # 1. Load Track
    track = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "time"], dtype=float)
    if track.empty:
        raise ValueError(f"Empty track file for trip {trip_id}")

    # Add temporal & kinematic features to track
    dt = np.diff(track["time"].values, prepend=track["time"].values[0])
    # Avoid zero division
    dt_safe = np.where(dt > 0, dt, 1.0)
    dt[0] = 1.0

    # Pairwise distances between consecutive GPS observations
    lons = track["lon"].values
    lats = track["lat"].values
    n_pts = len(track)
    step_dists = np.zeros(n_pts, dtype=float)
    step_bearings = np.zeros(n_pts, dtype=float)

    for i in range(1, n_pts):
        step_dists[i] = haversine_distance(lons[i - 1], lats[i - 1], lons[i], lats[i])
        step_bearings[i] = compute_bearing(lons[i - 1], lats[i - 1], lons[i], lats[i])

    if n_pts > 1:
        step_bearings[0] = step_bearings[1]

    speed = step_dists / dt_safe
    track["dt"] = dt
    track["speed_m_s"] = speed
    track["bearing_deg"] = step_bearings
    track["gps_index"] = np.arange(n_pts, dtype=int)

    # 2. Load Nodes
    nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"], dtype=float)
    nodes["node_id"] = np.arange(len(nodes), dtype=int)

    # 3. Load Arcs
    arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=int)
    arcs["arc_id"] = np.arange(len(arcs), dtype=int)

    # Calculate arc physical length & bearing
    u_idx = arcs["u"].values
    v_idx = arcs["v"].values
    u_lons = nodes["lon"].values[u_idx]
    u_lats = nodes["lat"].values[u_idx]
    v_lons = nodes["lon"].values[v_idx]
    v_lats = nodes["lat"].values[v_idx]

    # Vectorized Haversine for arcs
    phi1, phi2 = np.radians(u_lats), np.radians(v_lats)
    dphi = np.radians(v_lats - u_lats)
    dlam = np.radians(v_lons - u_lons)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    arc_lengths = EARTH_RADIUS_METERS * c

    # Vectorized Bearing for arcs
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    arc_bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

    arcs["length_m"] = arc_lengths
    arcs["bearing_deg"] = arc_bearings

    # 4. Load Route (only if permitted and present)
    route = None
    if allow_route and route_file.exists() and route_file.stat().st_size > 0:
        route = pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"], dtype=int)
        route["step"] = np.arange(len(route), dtype=int)

    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
