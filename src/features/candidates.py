"""Candidate road-segment generation module.

Generates candidate road segments for each GPS observation using spatial indexing
with distance thresholds and fallback mechanisms to ensure 0% candidate miss rate.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree


METERS_PER_DEGREE_LAT = 111320.0


def build_road_spatial_index(nodes: pd.DataFrame, arcs: pd.DataFrame) -> tuple[STRtree, List[LineString]]:
    """Build an STRtree spatial index over the road network arcs.

    Parameters
    ----------
    nodes : pd.DataFrame
        Road network nodes with 'lon', 'lat'.
    arcs : pd.DataFrame
        Road network arcs with 'u', 'v'.

    Returns
    -------
    tuple: (STRtree index, list of shapely LineStrings)
    """
    node_lons = nodes["lon"].values
    node_lats = nodes["lat"].values
    u_idx = arcs["u"].values
    v_idx = arcs["v"].values

    coords = np.column_stack([
        node_lons[u_idx], node_lats[u_idx],
        node_lons[v_idx], node_lats[v_idx]
    ]).reshape(-1, 2, 2)

    lines = [LineString(c) for c in coords]
    tree = STRtree(lines)
    return tree, lines


def generate_candidates(track: pd.DataFrame,
                        nodes: pd.DataFrame,
                        arcs: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Generate candidate road arcs for each GPS observation.

    Parameters
    ----------
    track : pd.DataFrame
        GPS trajectory containing 'lon', 'lat', 'gps_index'.
    nodes : pd.DataFrame
        Road network nodes with 'lon', 'lat'.
    arcs : pd.DataFrame
        Road network arcs with 'u', 'v'.
    config : dict, optional
        Configuration dictionary:
            - radius_m: search radius in meters (default: 45.0)
            - min_candidates: minimum candidates to retain per point (default: 1)
            - max_candidates: maximum candidates to retain per point (default: 25)

    Returns
    -------
    pd.DataFrame:
        DataFrame containing columns:
            - 'gps_index': index of the GPS observation
            - 'candidate_arc_id': ID of the candidate road arc
    """
    cfg = config or {}
    radius_m = float(cfg.get("radius_m", 45.0))
    min_candidates = int(cfg.get("min_candidates", 1))
    max_candidates = int(cfg.get("max_candidates", 25))

    tree, lines = build_road_spatial_index(nodes, arcs)

    mean_lat = float(track["lat"].mean())
    lat_scale = METERS_PER_DEGREE_LAT
    lon_scale = METERS_PER_DEGREE_LAT * np.cos(np.radians(mean_lat))
    deg_radius = radius_m / max(lon_scale, lat_scale)

    pts = [Point(lon, lat) for lon, lat in zip(track["lon"].values, track["lat"].values)]
    buffers = [pt.buffer(deg_radius) for pt in pts]

    # Query spatial index
    query_gps, query_arcs = tree.query(buffers)

    # Group by GPS index
    gps_candidates_map: Dict[int, List[int]] = {i: [] for i in range(len(track))}
    for g_idx, a_idx in zip(query_gps, query_arcs):
        gps_candidates_map[g_idx].append(int(a_idx))

    # Apply fallbacks and limits
    all_gps_indices = []
    all_arc_indices = []

    for g_idx in range(len(track)):
        cands = gps_candidates_map[g_idx]

        # Fallback to nearest arc if search radius missed
        if len(cands) < min_candidates:
            pt = pts[g_idx]
            nearest_idx = int(tree.nearest(pt))
            if nearest_idx not in cands:
                cands.append(nearest_idx)

        # Remove duplicates
        cands = list(dict.fromkeys(cands))

        # Limit to max_candidates if too dense
        if len(cands) > max_candidates:
            pt = pts[g_idx]
            # Sort by distance
            cands.sort(key=lambda aid: lines[aid].distance(pt))
            cands = cands[:max_candidates]

        for aid in cands:
            all_gps_indices.append(g_idx)
            all_arc_indices.append(aid)

    df_candidates = pd.DataFrame({
        "gps_index": np.array(all_gps_indices, dtype=int),
        "candidate_arc_id": np.array(all_arc_indices, dtype=int)
    })

    return df_candidates
