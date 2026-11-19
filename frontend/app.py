"""Streamlit frontend application for PathSnap.

High-performance, futuristic dashboard for trajectory map-matching and route reconstruction.
Visualizes noisy GPS observations against reconstructed road routes on interactive Folium maps,
providing real-time observation accuracy, route continuity, and pipeline telemetry.
Supports both distributed Two-Laptop evaluation and standalone engine execution.
"""

import sys
import os
from pathlib import Path

# Robustly ensure repo root is in sys.path so 'src' imports never fail
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import requests
import folium
import pandas as pd
import json
import time

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    import streamlit.components.v1 as components
    HAS_ST_FOLIUM = False

# Page configuration - Centered, Full Width, No Sidebar
st.set_page_config(
    page_title="PathSnap",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Styling: High-Voltage Titanium & Electric Cyan Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800;900&display=swap');

    /* Global Typography & Deep Obsidian Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -10%, rgba(0, 240, 255, 0.08) 0%, rgba(10, 14, 23, 0) 60%),
                    linear-gradient(180deg, #07090e 0%, #0c1017 100%) !important;
        color: #f8fafc;
    }

    /* Completely Remove Sidebar & Collapsed Controls */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main .block-container {
        max-width: 1280px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
    }

    /* Centered Powerful Brand Logo */
    .brand-hero {
        text-align: center;
        padding: 0.2rem 0 1.2rem 0;
        margin-bottom: 1.4rem;
    }
    .brand-title {
        display: inline-flex;
        align-items: baseline;
        justify-content: center;
        gap: 4px;
        line-height: 1;
    }
    .brand-path {
        font-size: 4.0rem;
        font-weight: 900;
        letter-spacing: -2px;
        color: #ffffff !important;
        text-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
    }
    .brand-snap {
        font-size: 4.0rem;
        font-weight: 900;
        letter-spacing: -2px;
        color: #00f0ff !important;
        background: linear-gradient(135deg, #00f0ff 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 30px rgba(0, 240, 255, 0.7));
    }
    .brand-subline {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    /* Centered Mission Control Command Deck */
    .control-deck-container {
        background: linear-gradient(145deg, rgba(22, 30, 46, 0.8) 0%, rgba(12, 17, 28, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 240, 255, 0.22);
        border-radius: 16px;
        padding: 22px 28px 18px 28px;
        margin: 0 auto 26px auto;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.65), 0 0 25px rgba(0, 240, 255, 0.08);
    }
    .control-deck-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .control-deck-title {
        color: #00f0ff;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .status-local {
        background: rgba(0, 240, 255, 0.12);
        border: 1px solid rgba(0, 240, 255, 0.35);
        color: #38bdf8;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 8px currentColor;
    }

    /* Electric Glowing Primary Action Button */
    div.stButton > button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #00f0ff 100%) !important;
        color: #07090e !important;
        font-weight: 900 !important;
        font-size: 1.0rem !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 10px !important;
        height: 44px !important;
        box-shadow: 0 4px 18px rgba(0, 240, 255, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 28px rgba(0, 240, 255, 0.7) !important;
        color: #000000 !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.99) !important;
    }

    /* Glowing Mission Control Stat Cards */
    .stat-card {
        background: linear-gradient(145deg, rgba(22, 30, 46, 0.75) 0%, rgba(12, 17, 28, 0.9) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
        margin-bottom: 12px;
        min-height: 112px;
    }
    .stat-card:hover {
        transform: translateY(-3px);
    }
    .stat-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .stat-icon {
        font-size: 1.15rem;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .stat-value {
        font-size: 2.15rem;
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: -0.5px;
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    .stat-help {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Welcome Hero Section */
    .hero-box {
        background: linear-gradient(135deg, rgba(22, 30, 46, 0.4) 0%, rgba(12, 17, 28, 0.7) 100%);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-badge-tag {
        display: inline-block;
        color: #00f0ff;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 10px 0;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 850px;
        margin-bottom: 22px;
    }
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .pipeline-card {
        background: rgba(12, 17, 28, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 10px;
        padding: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .pipeline-card:hover {
        transform: translateY(-2px);
        border-color: #00f0ff;
    }
    .pipe-icon {
        font-size: 1.35rem;
        margin-bottom: 6px;
    }
    .pipe-title {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pipe-stat {
        color: #00f0ff;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .pipe-desc {
        color: #64748b;
        font-size: 0.72rem;
        line-height: 1.3;
    }

    /* Legend Banner */
    .legend-banner {
        background: linear-gradient(135deg, #161e2e 0%, #0c111c 100%);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 700;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom: 2px solid #00f0ff !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Main Brand Header - Centered, Clean, Electric Cyan & Titanium White
st.markdown("""
<div class="brand-hero">
    <div class="brand-title">
        <span class="brand-path">Path</span><span class="brand-snap">Snap</span>
    </div>
    <div class="brand-subline">DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION</div>
</div>
""", unsafe_allow_html=True)

# Backend Connectivity Check
api_base_url = "http://127.0.0.1:8000"
backend_online = False
backend_model_name = "Standalone Local Engine"
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=2)
    if health_resp.status_code == 200:
        backend_online = True
        backend_model_name = health_resp.json().get("model_name", "Tuned Random Forest")
except Exception:
    backend_online = False

def get_trips_for_partition(part: str) -> list:
    """Dynamically fetch trip IDs for the selected partition."""
    if backend_online:
        try:
            trips_resp = requests.get(f"{api_base_url}/trips?partition={part}", timeout=2)
            if trips_resp.status_code == 200:
                trips = trips_resp.json().get("trips", [])
                if trips:
                    return trips
        except Exception:
            pass
    try:
        from src.data.loader import list_partition_trips
        return list_partition_trips(partition=part)
    except Exception:
        if part == "student_test":
            return ["00000000", "00000007", "00000009", "00000040", "00000052", "00000056", "00000065", "00000076", "00000077", "00000091"]
        return ["00000001", "00000002", "00000003", "00000004", "00000005", "00000006", "00000008"]


def execute_reconstruction(trip_id_to_run, part):
    """Run reconstruction pipeline either via backend or local engine fallback."""
    with st.spinner(f"Reconstructing route for trip {trip_id_to_run}..."):
        res = None
        # Attempt 1: Via FastAPI Backend
        if backend_online:
            try:
                resp = requests.post(f"{api_base_url}/predict/{trip_id_to_run}?partition={part}", timeout=90)
                if resp.status_code == 200:
                    res = resp.json()
                else:
                    st.warning(f"Backend returned {resp.status_code}. Using local engine.")
            except Exception as e:
                st.warning(f"Backend network call failed: {e}. Using local engine.")

        # Attempt 2: Direct Local Engine Fallback (Guaranteed!)
        if res is None:
            try:
                from src.api.main import predict_trip
                res = predict_trip(trip_id_to_run, partition=part)
            except Exception as e:
                st.error(f"Inference failed: {str(e)}")

        if res:
            st.session_state["result"] = res
            st.session_state["selected_trip"] = trip_id_to_run


def render_metric_card(col, icon: str, label: str, value: str, help_text: str = "", theme: str = "cyan"):
    """Render a high-contrast glowing dark card with bold typography."""
    themes = {
        "emerald": {"border": "#10b981", "glow": "rgba(16, 185, 129, 0.25)"},
        "cyan": {"border": "#00f0ff", "glow": "rgba(0, 240, 255, 0.25)"},
        "blue": {"border": "#0284c7", "glow": "rgba(2, 132, 199, 0.25)"},
        "amber": {"border": "#f59e0b", "glow": "rgba(245, 158, 11, 0.25)"},
        "slate": {"border": "#94a3b8", "glow": "rgba(148, 163, 184, 0.25)"},
    }
    t = themes.get(theme, themes["cyan"])
    col.markdown(f"""
    <div class="stat-card" style="border-left: 4px solid {t['border']}; box-shadow: 0 4px 20px {t['glow']};">
        <div class="stat-header">
            <span class="stat-icon">{icon}</span>
            <span class="stat-label">{label}</span>
        </div>
        <div class="stat-value">{value}</div>
        {f'<div class="stat-help">{help_text}</div>' if help_text else ''}
    </div>
    """, unsafe_allow_html=True)


# 2. Centered Trip Selection & Control Deck (Replaces Sidebar)
status_class = "status-online" if backend_online else "status-local"
status_label = f"Backend Online ({backend_model_name})" if backend_online else "Standalone Engine Ready"

st.markdown(f"""
<div class="control-deck-container">
    <div class="control-deck-top">
        <div class="control-deck-title">⚡ RECONSTRUCTION MISSION CONTROL</div>
        <div class="status-badge {status_class}">
            <span class="status-dot"></span>{status_label}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Centered Interactive Inputs: Partition first so Trip Options update dynamically
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([3, 4, 3])
with ctrl_c1:
    partition = st.selectbox("Partition", options=["student_test", "train"], index=0)

trip_options = get_trips_for_partition(partition)

with ctrl_c2:
    selected_trip = st.selectbox(f"Select Trip ({len(trip_options)} available)", options=trip_options, index=0)
with ctrl_c3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Map Reconstruction", type="primary", use_container_width=True)

if run_btn:
    execute_reconstruction(selected_trip, partition)

# 3. Render Results Dashboard OR Mission Control Landing Page
if "result" in st.session_state:
    result = st.session_state["result"]
    metrics = result.get("metrics", {})
    gps_points = result.get("gps_points", [])
    route_segments = result.get("route_segments_coords", [])
    ground_truth = result.get("ground_truth_coords", [])
    full_route_arcs = result.get("full_route_arcs", [])
    trip_id_display = result.get("trip_id", selected_trip)

    # Top Metric Telemetry Banner
    col1, col2, col3, col4, col5 = st.columns(5)
    acc = metrics.get("observation_accuracy")
    acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
    cont = metrics.get("🔗 Route Continuity", metrics.get("route_continuity"))
    cont_str = f"{cont*100:.2f}%" if cont is not None else "N/A"
    n_obs = f"{result.get('n_observations', len(gps_points)):,}"
    n_arcs = f"{result.get('n_reconstructed_arcs', len(full_route_arcs)):,}"
    inf_time = f"{result.get('inference_duration_seconds', 0.0):.2f}s"

    render_metric_card(col1, "🎯", "Accuracy", acc_str, "Matched to ground truth", theme="emerald")
    render_metric_card(col2, "🔗", "Continuity", cont_str, "Valid road transitions", theme="cyan")
    render_metric_card(col3, "📍", "GPS Points", n_obs, "Points processed", theme="slate")
    render_metric_card(col4, "🛣️", "Road Arcs", n_arcs, "Reconstructed segments", theme="amber")
    render_metric_card(col5, "⚡", "Runtime", inf_time, "Total pipeline latency", theme="blue")

    tab_map, tab_table, tab_details = st.tabs(["🗺️ Interactive Map View", "📊 Observation Table", "ℹ️ System Details"])

    with tab_map:
        if gps_points:
            lats = [p["lat"] for p in gps_points]
            lons = [p["lon"] for p in gps_points]
            mean_lat = sum(lats) / len(lats)
            mean_lon = sum(lons) / len(lons)

            # Folium Map with OpenStreetMap
            m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14, tiles="OpenStreetMap")

            # 1. Ground Truth Route Layer (Green Polyline)
            if ground_truth:
                gt_layer = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
                for seg in ground_truth:
                    folium.PolyLine(seg, color="#27ae60", weight=4, opacity=0.85, tooltip="Ground Truth Route").add_to(gt_layer)
                gt_layer.add_to(m)

            # 2. HMM Reconstructed Route Layer (Purple Polyline)
            route_layer = folium.FeatureGroup(name="HMM Reconstructed Route (Purple)", show=True)
            for seg in route_segments:
                folium.PolyLine(seg, color="#8e44ad", weight=5, opacity=0.9, tooltip="Reconstructed Route").add_to(route_layer)
            route_layer.add_to(m)

            # 3. Raw Noisy GPS Points Layer (Blue Dots)
            gps_layer = folium.FeatureGroup(name="Noisy GPS Observations (Blue Dots)", show=True)
            step_stride = max(1, len(gps_points) // 300)
            for idx in range(0, len(gps_points), step_stride):
                pt = gps_points[idx]
                folium.CircleMarker(
                    location=[pt["lat"], pt["lon"]],
                    radius=3.5,
                    color="#0284c7",
                    fill=True,
                    fill_color="#00f0ff",
                    fill_opacity=0.75,
                    popup=f"GPS #{idx}<br>Time: {pt['time']:.1f}s<br>Speed: {pt['speed_kmh']:.1f} km/h"
                ).add_to(gps_layer)
            gps_layer.add_to(m)

            # 4. Start and Finish Markers
            folium.Marker(
                [lats[0], lons[0]],
                popup="🚩 Route Start",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(m)
            folium.Marker(
                [lats[-1], lons[-1]],
                popup="🏁 Route Finish",
                icon=folium.Icon(color="red", icon="flag", prefix="fa")
            ).add_to(m)

            # Automatically fit bounds so the entire trip is in view
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
            folium.LayerControl().add_to(m)

            # Render map using streamlit-folium or components
            if HAS_ST_FOLIUM:
                st_folium(m, use_container_width=True, height=580)
            else:
                import streamlit.components.v1 as components
                components.html(m._repr_html_(), height=580)

            st.markdown("""
            <div class="legend-banner">
                <strong>💡 Map Telemetry:</strong> &nbsp;&nbsp;
                <span style="color:#4ade80; font-weight:700;">🟢 Green Polyline</span> = Ground Truth Route &nbsp;|&nbsp; 
                <span style="color:#c084fc; font-weight:700;">🟣 Purple Polyline</span> = HMM Reconstructed Route &nbsp;|&nbsp; 
                <span style="color:#00f0ff; font-weight:700;">🔵 Cyan Dots</span> = Raw GPS Telemetry
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No GPS observations available to display.")

    with tab_table:
        st.subheader("Raw GPS Observations")
        df_gps = pd.DataFrame(gps_points)
        st.dataframe(df_gps, use_container_width=True, height=280)

        st.subheader(f"Reconstructed Arc Sequence ({len(full_route_arcs)} road segments)")
        st.write("Chronological sequence of matched road network segment IDs:")
        st.code(", ".join(str(a) for a in full_route_arcs[:150]) + ("..." if len(full_route_arcs) > 150 else ""))

    with tab_details:
        st.subheader("Trip & Inference Metadata")
        st.json({
            "trip_id": result.get("trip_id"),
            "partition": result.get("partition"),
            "model_used": result.get("model_used"),
            "inference_duration_seconds": result.get("inference_duration_seconds"),
            "n_observations": result.get("n_observations"),
            "n_reconstructed_arcs": result.get("n_reconstructed_arcs"),
            "metrics": metrics
        })

else:
    # Mission Control Landing View when no trip is currently active
    st.markdown("""
    <div class="hero-box">
        <div class="hero-badge-tag">AUTONOMOUS TRAJECTORY RECONSTRUCTION</div>
        <h2 class="hero-title">Distributed Map-Matching & Route Recovery</h2>
        <p class="hero-desc">
            Transform high-noise, sparse GPS telemetry into physical, topology-consistent road routes using 
            geodesic STRtree spatial indexing, 14 kinematic features, Bayesian-tuned Random Forest classification, 
            and log-space Hidden Markov Model (HMM) Viterbi dynamic programming.
        </p>
        <div class="pipeline-grid">
            <div class="pipeline-card">
                <div class="pipe-icon">📍</div>
                <div class="pipe-title">Spatial Indexing</div>
                <div class="pipe-stat">100% Recall</div>
                <div class="pipe-desc">45m geodesic STRtree buffer with nearest candidate fallback</div>
            </div>
            <div class="pipeline-card">
                <div class="pipe-icon">🧠</div>
                <div class="pipe-title">ML Classifier</div>
                <div class="pipe-stat">0.887 F1 Score</div>
                <div class="pipe-desc">Bayesian-optimized Random Forest trained without data leakage</div>
            </div>
            <div class="pipeline-card">
                <div class="pipe-icon">🕸️</div>
                <div class="pipe-title">Graph Topology</div>
                <div class="pipe-stat">1,348 Arcs</div>
                <div class="pipe-desc">NetworkX directed graph with geodesic Dijkstra transitions</div>
            </div>
            <div class="pipeline-card">
                <div class="pipe-icon">⚡</div>
                <div class="pipe-title">HMM Viterbi</div>
                <div class="pipe-stat">97.8% Accuracy</div>
                <div class="pipe-desc">Log-space dynamic programming with adaptive gap penalties</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Launch Evaluation Trips")
    st.write("Click any trip below to run the end-to-end map-matching pipeline instantly:")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("🚗 Launch Trip 00000000", use_container_width=True):
            execute_reconstruction("00000000", "student_test")
            st.rerun()
    with q2:
        if st.button("🏎️ Launch Trip 00000007", use_container_width=True):
            execute_reconstruction("00000007", "student_test")
            st.rerun()
    with q3:
        if st.button("🚙 Launch Trip 00000052", use_container_width=True):
            execute_reconstruction("00000052", "student_test")
            st.rerun()
    with q4:
        if st.button("🚕 Launch Trip 00000091", use_container_width=True):
            execute_reconstruction("00000091", "student_test")
            st.rerun()
