"""Candidate feature engineering module.

Calculates geometric, kinematic, and spatial ranking features for each candidate arc
associated with a GPS observation.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


METERS_PER_DEGREE_LAT = 111320.0


def extract_candidate_features(track: pd.DataFrame,
                               candidates: pd.DataFrame,
                               nodes: pd.DataFrame,
                               arcs: pd.DataFrame,
                               trip_id: Optional[str] = None) -> pd.DataFrame:
    """Extract machine learning features for all candidate pairs.

    Parameters
    ----------
    track : pd.DataFrame
        GPS points with 'gps_index', 'lon', 'lat', 'time', 'dt', 'speed_m_s', 'bearing_deg'.
    candidates : pd.DataFrame
        Candidate pairs with 'gps_index', 'candidate_arc_id'.
    nodes : pd.DataFrame
        Road network nodes with 'node_id', 'lon', 'lat'.
    arcs : pd.DataFrame
        Road network arcs with 'arc_id', 'u', 'v', 'length_m', 'bearing_deg'.
    trip_id : str, optional
        Trip identifier to include in resulting DataFrame.

    Returns
    -------
    pd.DataFrame:
        Table of features with columns:
        ['trip_id', 'gps_index', 'candidate_arc_id', 'dist_to_arc', 'projection_fraction',
         'dist_to_start_node', 'dist_to_end_node', 'arc_length', 'heading_diff_deg',
         'cos_heading_diff', 'sin_heading_diff', 'gps_speed', 'gps_delta_t',
         'dist_rank', 'dist_ratio_to_nearest', 'normalized_dist', 'speed_length_ratio']
    """
    if candidates.empty:
        return pd.DataFrame()

    gps_idx = candidates["gps_index"].values
    arc_idx = candidates["candidate_arc_id"].values

    # Local metric coordinates conversion
    mean_lat = float(track["lat"].mean())
    lat_scale = METERS_PER_DEGREE_LAT
    lon_scale = METERS_PER_DEGREE_LAT * np.cos(np.radians(mean_lat))

    # GPS metrics
    p_x = (track["lon"].values[gps_idx] - track["lon"].values[0]) * lon_scale
    p_y = (track["lat"].values[gps_idx] - track["lat"].values[0]) * lat_scale
    p_speed = track["speed_m_s"].values[gps_idx]
    p_dt = track["dt"].values[gps_idx]
    p_bearing = track["bearing_deg"].values[gps_idx]

    # Arc metrics
    u_idx = arcs["u"].values[arc_idx]
    v_idx = arcs["v"].values[arc_idx]
    a_len = arcs["length_m"].values[arc_idx]
    a_bearing = arcs["bearing_deg"].values[arc_idx]

    u_x = (nodes["lon"].values[u_idx] - track["lon"].values[0]) * lon_scale
    u_y = (nodes["lat"].values[u_idx] - track["lat"].values[0]) * lat_scale
    v_x = (nodes["lon"].values[v_idx] - track["lon"].values[0]) * lon_scale
    v_y = (nodes["lat"].values[v_idx] - track["lat"].values[0]) * lat_scale

    # Vector AB = V - U
    ab_x = v_x - u_x
    ab_y = v_y - u_y
    ab_sq = ab_x ** 2 + ab_y ** 2

    # Vector AP = P - U
    ap_x = p_x - u_x
    ap_y = p_y - u_y

    # Distance to start and end nodes
    dist_start = np.hypot(ap_x, ap_y)
    dist_end = np.hypot(p_x - v_x, p_y - v_y)

    # Projection fraction t along segment
    ab_sq_safe = np.where(ab_sq > 1e-6, ab_sq, 1.0)
    t = (ap_x * ab_x + ap_y * ab_y) / ab_sq_safe
    t_clamped = np.clip(t, 0.0, 1.0)

    # Closest point Q on segment
    q_x = u_x + t_clamped * ab_x
    q_y = u_y + t_clamped * ab_y
    dist_to_arc = np.hypot(p_x - q_x, p_y - q_y)

    # Heading difference
    diff_deg = np.abs(p_bearing - a_bearing) % 360.0
    diff_deg = np.where(diff_deg > 180.0, 360.0 - diff_deg, diff_deg)
    diff_rad = np.radians(diff_deg)
    cos_head = np.cos(diff_rad)
    sin_head = np.abs(np.sin(diff_rad))

    # Build dataframe
    df = pd.DataFrame({
        "gps_index": gps_idx,
        "candidate_arc_id": arc_idx,
        "dist_to_arc": dist_to_arc,
        "projection_fraction": t_clamped,
        "dist_to_start_node": dist_start,
        "dist_to_end_node": dist_end,
        "arc_length": a_len,
        "heading_diff_deg": diff_deg,
        "cos_heading_diff": cos_head,
        "sin_heading_diff": sin_head,
        "gps_speed": p_speed,
        "gps_delta_t": p_dt,
        "speed_length_ratio": (p_speed * p_dt) / np.maximum(a_len, 1.0),
        "normalized_dist": dist_to_arc / 50.0
    })

    if trip_id is not None:
        df.insert(0, "trip_id", str(trip_id))

    # Rank features per GPS point
    df["dist_rank"] = df.groupby("gps_index")["dist_to_arc"].rank(method="min", ascending=True)
    min_dists = df.groupby("gps_index")["dist_to_arc"].transform("min")
    df["dist_ratio_to_nearest"] = df["dist_to_arc"] / (min_dists + 1e-3)

    return df
