"""Dataset exploratory data analysis (EDA) module.

Generates summary statistics, anomaly detection reports, distributions,
and interactive Folium visualizations for training and testing partitions.
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import folium

from src.data.loader import load_trip, list_partition_trips


def summarize_trip(trip_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate detailed exploratory summary for a single loaded trip."""
    track = trip_dict["track"]
    nodes = trip_dict["nodes"]
    arcs = trip_dict["arcs"]
    route = trip_dict["route"]

    dt = track["dt"].values[1:] if len(track) > 1 else np.array([1.0])
    speeds = track["speed_m_s"].values
    dists = np.diff(track["time"].values, prepend=track["time"].values[0]) * speeds

    # Graph connectivity
    G = nx.DiGraph()
    G.add_nodes_from(nodes["node_id"].values)
    G.add_edges_from(zip(arcs["u"].values, arcs["v"].values))

    undirected_G = G.to_undirected()
    n_components = nx.number_connected_components(undirected_G)
    isolated_nodes = len(list(nx.isolates(G)))

    total_dist_km = float(np.sum(dists) / 1000.0)
    duration_min = float((track["time"].iloc[-1] - track["time"].iloc[0]) / 60.0) if len(track) > 1 else 0.0

    return {
        "trip_id": trip_dict["trip_id"],
        "partition": trip_dict["partition"],
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_arcs": len(route) if route is not None else 0,
        "total_dist_km": total_dist_km,
        "duration_min": duration_min,
        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_max": float(np.max(dt)),
        "dt_1s_pct": float(np.mean(dt == 1.0) * 100.0),
        "dt_gaps_gt_5s": int(np.sum(dt > 5.0)),
        "speed_max_kmh": float(np.max(speeds) * 3.6),
        "speed_mean_kmh": float(np.mean(speeds) * 3.6),
        "lon_min": float(track["lon"].min()),
        "lon_max": float(track["lon"].max()),
        "lat_min": float(track["lat"].min()),
        "lat_max": float(track["lat"].max()),
        "graph_components": n_components,
        "isolated_nodes": isolated_nodes,
    }


def summarize_partition(data_root: str = "data/raw/Student folder",
                        partition: str = "train",
                        max_trips: int = None) -> pd.DataFrame:
    """Summarize all trips in a given partition and return a summary DataFrame."""
    trip_ids = list_partition_trips(data_root=data_root, partition=partition)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    records = []
    for tid in trip_ids:
        trip_data = load_trip(tid, partition=partition, data_root=data_root)
        records.append(summarize_trip(trip_data))

    return pd.DataFrame(records)


def create_folium_map(trip_dict: Dict[str, Any],
                      output_html: str | Path,
                      max_graph_edges: int = 1500) -> folium.Map:
    """Create an interactive Folium map showing GPS observations, road network, and ground truth route."""
    track = trip_dict["track"]
    nodes = trip_dict["nodes"]
    arcs = trip_dict["arcs"]
    route = trip_dict["route"]

    mean_lat = float(track["lat"].mean())
    mean_lon = float(track["lon"].mean())

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14, tiles="CartoDB positron")

    # 1. Road network geometry (subset around track bounds to keep map fast and responsive)
    pad = 0.005
    lon_min, lon_max = track["lon"].min() - pad, track["lon"].max() + pad
    lat_min, lat_max = track["lat"].min() - pad, track["lat"].max() + pad

    node_lons = nodes["lon"].values
    node_lats = nodes["lat"].values
    u_idx = arcs["u"].values
    v_idx = arcs["v"].values

    in_bbox = ((node_lons[u_idx] >= lon_min) & (node_lons[u_idx] <= lon_max) &
               (node_lats[u_idx] >= lat_min) & (node_lats[u_idx] <= lat_max))
    candidate_arc_indices = np.where(in_bbox)[0]
    if len(candidate_arc_indices) > max_graph_edges:
        candidate_arc_indices = candidate_arc_indices[:max_graph_edges]

    road_layer = folium.FeatureGroup(name="Road Network (Grey)", show=True)
    for a_id in candidate_arc_indices:
        u, v = u_idx[a_id], v_idx[a_id]
        coords = [[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]]
        folium.PolyLine(coords, color="#95a5a6", weight=1.5, opacity=0.6).add_to(road_layer)
    road_layer.add_to(m)

    # 2. Ground-truth route if available
    if route is not None:
        route_layer = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        route_coords = []
        for a_id in route["arc_id"]:
            u, v = u_idx[a_id], v_idx[a_id]
            route_coords.append([node_lats[u], node_lons[u]])
            route_coords.append([node_lats[v], node_lons[v]])
        folium.PolyLine(route_coords, color="#2ecc71", weight=4, opacity=0.85, tooltip="Ground Truth Route").add_to(route_layer)
        route_layer.add_to(m)

    # 3. GPS observations
    gps_layer = folium.FeatureGroup(name="GPS Observations (Blue Dots)", show=True)
    for idx, row in track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.7,
            popup=f"GPS #{idx}<br>Time: {row['time']:.1f}s<br>Speed: {row['speed_m_s']*3.6:.1f} km/h"
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    folium.LayerControl().add_to(m)

    out_p = Path(output_html)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_p))
    return m


def run_full_eda(data_root: str = "data/raw/Student folder",
                 output_dir: str = "results/eda") -> Dict[str, Any]:
    """Execute complete EDA pipeline across train and test partitions, save plots and summary tables."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Running EDA on training partition (80 trips)...")
    train_summary = summarize_partition(data_root=data_root, partition="train")
    train_summary.to_csv(out_path / "train_trips_inventory.csv", index=False)

    print("Running EDA on student_test partition (10 trips)...")
    test_summary = summarize_partition(data_root=data_root, partition="student_test")
    test_summary.to_csv(out_path / "test_trips_inventory.csv", index=False)

    # Statistical Overview
    summary_stats = {
        "train_trips": len(train_summary),
        "test_trips": len(test_summary),
        "total_train_gps_points": int(train_summary["gps_points"].sum()),
        "total_test_gps_points": int(test_summary["gps_points"].sum()),
        "avg_points_per_trip": float(train_summary["gps_points"].mean()),
        "dt_mean_all_trips": float(train_summary["dt_mean"].mean()),
        "pct_1s_sampling": float(train_summary["dt_1s_pct"].mean()),
        "avg_trip_length_km": float(train_summary["total_dist_km"].mean()),
        "avg_trip_duration_min": float(train_summary["duration_min"].mean()),
        "max_speed_kmh": float(train_summary["speed_max_kmh"].max()),
    }

    with open(out_path / "eda_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Plot 1: GPS Sampling Intervals
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(train_summary["dt_mean"], bins=20, kde=True, color="#3498db")
    plt.title("Mean Sampling Interval per Trip (s)")
    plt.xlabel("Seconds")

    plt.subplot(1, 2, 2)
    sns.histplot(train_summary["dt_1s_pct"], bins=20, kde=True, color="#2ecc71")
    plt.title("Percentage of 1s Steps per Trip (%)")
    plt.xlabel("Percent exactly 1.0s")
    plt.tight_layout()
    plt.savefig(out_path / "sampling_intervals.png", dpi=150)
    plt.close()

    # Plot 2: Trajectory Length and Duration
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_summary[["total_dist_km"]], color="#e74c3c")
    plt.title("Trip Distance Distribution (km)")

    plt.subplot(1, 2, 2)
    sns.histplot(train_summary["duration_min"], bins=20, kde=True, color="#9b59b6")
    plt.title("Trip Duration Distribution (min)")
    plt.xlabel("Duration in minutes")
    plt.tight_layout()
    plt.savefig(out_path / "trajectory_distributions.png", dpi=150)
    plt.close()

    # Folium Interactive Map for a representative training trip
    sample_trip = load_trip("00000001", partition="train", data_root=data_root)
    create_folium_map(sample_trip, out_path / "folium_trip_00000001.html")
    print("EDA completed successfully. Outputs written to:", out_path)

    return summary_stats
