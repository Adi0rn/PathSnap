"""HMM and Viterbi route decoding module.

Implements sequential Hidden Markov Model map-matching using:
- Emission probabilities derived from ML candidate scores and spatial distances
- Transition probabilities derived from network shortest paths and GPS displacement
- Viterbi forward dynamic programming in log-space and backward path reconstruction
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
import networkx as nx

from src.data.loader import haversine_distance


def build_road_graph(arcs: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[Tuple[int, int], int]]:
    """Build directed road graph from arcs DataFrame with length weights and arc_id mapping."""
    G = nx.DiGraph()
    u_vals = arcs["u"].values
    v_vals = arcs["v"].values
    len_vals = arcs["length_m"].values
    arc_ids = arcs["arc_id"].values

    edge_to_arc = {}
    for u, v, length, aid in zip(u_vals, v_vals, len_vals, arc_ids):
        u_int, v_int, aid_int = int(u), int(v), int(aid)
        G.add_edge(u_int, v_int, weight=float(length), arc_id=aid_int)
        edge_to_arc[(u_int, v_int)] = aid_int

    return G, edge_to_arc


def compute_emission_log_prob(model_prob: float,
                              dist_to_arc: float,
                              cos_heading_diff: float,
                              sigma_z: float = 12.0,
                              alpha: float = 1.2,
                              gamma: float = 1.0) -> float:
    """Compute log emission probability for a candidate road segment given a GPS observation.

    Parameters
    ----------
    model_prob : float
        Predicted probability of candidate being the true segment [0, 1].
    dist_to_arc : float
        Perpendicular distance from GPS point to candidate arc in meters.
    cos_heading_diff : float
        Cosine of angular heading difference in [-1, 1].
    sigma_z : float
        Standard deviation of GPS spatial error in meters.
    alpha : float
        Weight on ML model probability.
    gamma : float
        Weight on directional alignment.

    Returns
    -------
    float: log emission probability.
    """
    # Gaussian spatial emission
    spatial_log = -0.5 * (dist_to_arc / max(sigma_z, 1.0)) ** 2
    # Machine learning classifier emission
    prob_safe = max(float(model_prob), 1e-6)
    ml_log = alpha * math.log(prob_safe)
    # Directional alignment emission
    heading_log = gamma * float(cos_heading_diff)

    return float(spatial_log + ml_log + heading_log)


def compute_transition_log_prob(u_prev: int,
                                v_prev: int,
                                u_curr: int,
                                v_curr: int,
                                arc_prev: int,
                                arc_curr: int,
                                len_prev: float,
                                len_curr: float,
                                t_prev: float,
                                t_curr: float,
                                d_gps: float,
                                dt_gps: float,
                                road_graph: nx.DiGraph,
                                shortest_path_cache: Dict[Tuple[int, int], float],
                                beta: float = 15.0) -> float:
    """Compute log transition probability between two candidate road segments."""
    # Scale beta dynamically for large GPS time gaps
    beta_effective = beta * max(1.0, dt_gps / 2.0)

    # Case 1: Same road segment
    if arc_prev == arc_curr:
        delta_proj = max(0.0, t_curr - t_prev)
        d_route = delta_proj * len_prev
        diff = abs(d_route - d_gps)
        return float(-diff / beta_effective)

    # Case 2: Directly connected consecutive segments
    if v_prev == u_curr:
        d_route = (1.0 - t_prev) * len_prev + t_curr * len_curr
        diff = abs(d_route - d_gps)
        return float(-diff / beta_effective)

    # Case 3: Graph shortest path connection
    cache_key = (v_prev, u_curr)
    if cache_key in shortest_path_cache:
        path_len = shortest_path_cache[cache_key]
    else:
        # Search with distance cutoff to prevent slow global search
        cutoff = max(150.0, d_gps * 2.5 + 50.0)
        try:
            path_len = nx.shortest_path_length(road_graph, v_prev, u_curr, weight="weight")
            if path_len > cutoff:
                path_len = float("inf")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_len = float("inf")
        shortest_path_cache[cache_key] = path_len

    if math.isinf(path_len):
        # Disconnected transition penalty
        return -20.0

    d_route = (1.0 - t_prev) * len_prev + path_len + t_curr * len_curr
    diff = abs(d_route - d_gps)
    return float(-diff / beta_effective)


def decode_route(track: pd.DataFrame,
                 candidate_predictions: pd.DataFrame,
                 road_graph: nx.DiGraph,
                 arcs: Optional[pd.DataFrame] = None,
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute Viterbi trellis dynamic programming to decode optimal continuous route.

    Parameters
    ----------
    track : pd.DataFrame
        GPS trajectory with 'gps_index', 'lon', 'lat', 'time'.
    candidate_predictions : pd.DataFrame
        Candidate predictions with 'gps_index', 'candidate_arc_id', 'probability',
        'dist_to_arc', 'cos_heading_diff', 'projection_fraction'.
    road_graph : nx.DiGraph
        Road network directed graph.
    arcs : pd.DataFrame, optional
        Road arcs metadata containing 'u', 'v', 'length_m'.
    config : dict, optional
        Tuning parameters: sigma_z, beta, alpha, gamma.

    Returns
    -------
    dict:
        {
            'pointwise_arcs': list of decoded arc_ids for each GPS observation,
            'full_route_arcs': complete reconstructed sequence of traversed road arcs,
            'log_likelihood': total optimal path log score
        }
    """
    cfg = config or {}
    sigma_z = float(cfg.get("sigma_z", 10.0))
    beta = float(cfg.get("beta", 12.0))
    alpha = float(cfg.get("alpha", 1.2))
    gamma = float(cfg.get("gamma", 1.0))

    # Pre-index arc details
    arc_dict = {}
    if arcs is not None:
        u_vals = arcs["u"].values
        v_vals = arcs["v"].values
        l_vals = arcs["length_m"].values
        for aid, u, v, l in zip(arcs["arc_id"].values, u_vals, v_vals, l_vals):
            arc_dict[int(aid)] = (int(u), int(v), float(l))

    # Group candidates by GPS index
    T = len(track)
    candidates_by_step: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(T)}

    for _, row in candidate_predictions.iterrows():
        g_idx = int(row["gps_index"])
        if g_idx < T:
            aid = int(row["candidate_arc_id"])
            if aid in arc_dict:
                u, v, length = arc_dict[aid]
            elif road_graph.has_node(aid):
                u, v, length = aid, aid, 50.0
            else:
                u, v, length = 0, 0, 50.0

            candidates_by_step[g_idx].append({
                "arc_id": aid,
                "u": u,
                "v": v,
                "length": length,
                "prob": float(row.get("probability", row.get("score", 0.5))),
                "dist": float(row.get("dist_to_arc", 10.0)),
                "cos_diff": float(row.get("cos_heading_diff", 1.0)),
                "proj": float(row.get("projection_fraction", 0.5))
            })

    # Cache for shortest path queries
    path_cache: Dict[Tuple[int, int], float] = {}

    # Precompute GPS displacement between consecutive steps
    track_lons = track["lon"].values
    track_lats = track["lat"].values
    track_times = track["time"].values
    d_gps_steps = np.zeros(T, dtype=float)
    dt_gps_steps = np.ones(T, dtype=float)

    for t in range(1, T):
        d_gps_steps[t] = haversine_distance(track_lons[t - 1], track_lats[t - 1],
                                            track_lons[t], track_lats[t])
        dt_gps_steps[t] = max(0.1, track_times[t] - track_times[t - 1])

    # 1. Initialize Trellis for t = 0
    cands_0 = candidates_by_step[0]
    if not cands_0:
        # Safe fallback if no candidate
        cands_0 = [{"arc_id": 0, "u": 0, "v": 0, "length": 50.0, "prob": 0.5, "dist": 20.0, "cos_diff": 1.0, "proj": 0.5}]

    V = [{c["arc_id"]: compute_emission_log_prob(c["prob"], c["dist"], c["cos_diff"], sigma_z, alpha, gamma)
          for c in cands_0}]
    backpointers: List[Dict[int, int]] = []

    # 2. Forward Viterbi Trellis Dynamic Programming
    for t in range(1, T):
        cands_curr = candidates_by_step[t]
        cands_prev = candidates_by_step[t - 1]
        V_prev = V[t - 1]

        if not cands_curr:
            cands_curr = cands_prev

        d_gps = d_gps_steps[t]
        dt_gps = dt_gps_steps[t]

        V_curr = {}
        bp_curr = {}

        for c_curr in cands_curr:
            aid_curr = c_curr["arc_id"]
            emit_log = compute_emission_log_prob(c_curr["prob"], c_curr["dist"], c_curr["cos_diff"], sigma_z, alpha, gamma)

            best_val = -float("inf")
            best_prev_aid = cands_prev[0]["arc_id"]

            for c_prev in cands_prev:
                aid_prev = c_prev["arc_id"]
                if aid_prev not in V_prev:
                    continue

                trans_log = compute_transition_log_prob(
                    c_prev["u"], c_prev["v"], c_curr["u"], c_curr["v"],
                    aid_prev, aid_curr, c_prev["length"], c_curr["length"],
                    c_prev["proj"], c_curr["proj"],
                    d_gps, dt_gps, road_graph, path_cache, beta
                )

                score = V_prev[aid_prev] + trans_log + emit_log
                if score > best_val:
                    best_val = score
                    best_prev_aid = aid_prev

            V_curr[aid_curr] = best_val
            bp_curr[aid_curr] = best_prev_aid

        V.append(V_curr)
        backpointers.append(bp_curr)

    # 3. Backtracking to recover optimal path
    last_step = V[-1]
    best_last_arc = max(last_step.keys(), key=lambda aid: last_step[aid])
    total_log_likelihood = float(last_step[best_last_arc])

    pointwise_arcs = [best_last_arc]
    curr_arc = best_last_arc
    for t in range(T - 2, -1, -1):
        prev_arc = backpointers[t][curr_arc]
        pointwise_arcs.append(prev_arc)
        curr_arc = prev_arc

    pointwise_arcs.reverse()

    # 4. Construct contiguous full route
    full_route_arcs = []
    for i in range(len(pointwise_arcs)):
        aid = pointwise_arcs[i]
        if not full_route_arcs or full_route_arcs[-1] != aid:
            full_route_arcs.append(aid)

    return {
        "pointwise_arcs": pointwise_arcs,
        "full_route_arcs": full_route_arcs,
        "log_likelihood": total_log_likelihood
    }


def compare_classifier_vs_hmm(trip_data: Dict[str, Any],
                              model: Any,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run classifier-only vs classifier+HMM decoding on a single trip and return comparison metrics."""
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    from src.features.candidates import generate_candidates
    from src.features.engineering import extract_candidate_features
    from src.models.baselines import predict_candidates
    from src.features.dataset import map_gps_to_route_arcs

    # 1. Candidate features & Model Predictions
    cands = generate_candidates(track, nodes, arcs)
    feats = extract_candidate_features(track, cands, nodes, arcs, trip_id=trip_data["trip_id"])
    preds_df = predict_candidates(feats, model)

    # Ground truth mapping if available
    ground_truth_arcs = None
    if route is not None:
        gps_to_true = map_gps_to_route_arcs(track, nodes, arcs, route)
        ground_truth_arcs = [gps_to_true[i] for i in range(len(track))]

    # Classifier-only Top-1
    top1_df = preds_df[preds_df["top1_predicted"] == 1].sort_values("gps_index")
    classifier_top1_arcs = top1_df["candidate_arc_id"].values.tolist()

    # 2. HMM Viterbi Decoding
    road_graph, _ = build_road_graph(arcs)
    decoded = decode_route(track, preds_df, road_graph, arcs=arcs, config=config)
    hmm_arcs = decoded["pointwise_arcs"]

    # Calculate Accuracies if ground truth is present
    clf_accuracy = None
    hmm_accuracy = None
    clf_continuity = 0.0
    hmm_continuity = 0.0

    T = len(track)
    if ground_truth_arcs is not None:
        clf_hits = sum(1 for p, t in zip(classifier_top1_arcs, ground_truth_arcs) if p == t)
        hmm_hits = sum(1 for p, t in zip(hmm_arcs, ground_truth_arcs) if p == t)
        clf_accuracy = float(clf_hits / T)
        hmm_accuracy = float(hmm_hits / T)

    # Continuity calculation: consecutive arcs same or topologically connected
    u_map = dict(zip(arcs["arc_id"], arcs["u"]))
    v_map = dict(zip(arcs["arc_id"], arcs["v"]))

    def calc_cont(arc_seq):
        if len(arc_seq) < 2:
            return 1.0
        cont = 0
        for i in range(1, len(arc_seq)):
            a_prev, a_curr = arc_seq[i - 1], arc_seq[i]
            if a_prev == a_curr or v_map.get(a_prev) == u_map.get(a_curr):
                cont += 1
        return float(cont / (len(arc_seq) - 1))

    clf_continuity = calc_cont(classifier_top1_arcs)
    hmm_continuity = calc_cont(hmm_arcs)

    return {
        "trip_id": trip_data["trip_id"],
        "n_observations": T,
        "classifier_top1_accuracy": clf_accuracy,
        "hmm_decoded_accuracy": hmm_accuracy,
        "classifier_continuity": clf_continuity,
        "hmm_continuity": hmm_continuity,
        "pointwise_arcs": hmm_arcs,
        "full_reconstructed_route": decoded["full_route_arcs"],
        "log_likelihood": decoded["log_likelihood"]
    }
