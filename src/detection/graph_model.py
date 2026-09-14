"""
graph_model.py

Spatio-Temporal Graph Neural Network (GNN) baseline architecture for attack detection.
Combines 1D temporal convolutions/linear encoders over sliding window frames with
spatial graph convolutions over the physical C-Town WDS network topology matrix.
Sensor-to-graph node mapping is an explicit input/configuration (never fabricated).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatioTemporalGNN(nn.Module):
    """Spatio-Temporal GNN PyTorch architecture for WDS Cyber-Attack Detection.

    Learns normal hydraulic dynamics by forecasting next-step sensor states (B, F)
    from historical sliding window feature tensors (B, W, F).

    Args:
        in_features: Number of sensor feature channels per time step (F)
        window_size: Length of sliding window frame (W)
        hidden_dim: Number of hidden units in temporal and spatial layers
        num_nodes: Number of graph nodes in network topology
        sensor_to_node_map: Optional dictionary mapping sensor column index -> node ID
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        window_size: int = 12,
        hidden_dim: int = 64,
        num_nodes: Optional[int] = None,
        sensor_to_node_map: Optional[Dict[int, str]] = None,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes if num_nodes is not None else in_features
        self.sensor_to_node_map = sensor_to_node_map
        self.forecast_horizon = forecast_horizon

        # Temporal Encoder: Conv1d over sliding window frames
        self.temporal_encoder = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        # Spatial Graph Convolution Layer: Linear transformation for neighbor aggregation
        self.spatial_fc = nn.Linear(hidden_dim, hidden_dim)

        # Predictive Head: Forecasts next-step sensor features (B, forecast_horizon * in_features)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_features * forecast_horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for batch of sliding window feature tensors.

        Args:
            x: Input tensor of shape (B, W, F), where B=batch_size, W=window_size, F=in_features.
            adj_matrix: Optional normalized graph adjacency matrix tensor of shape (F, F).

        Returns:
            Predicted sensor features tensor of shape (B, F) if forecast_horizon=1 or (B, H, F) if H>1.
        """
        B, W, num_feats = x.shape

        # 1. Spatial aggregation across graph neighbors if adjacency matrix is provided
        if adj_matrix is not None and adj_matrix.shape[0] == num_feats:
            # Multiply (B, W, F) @ (F, F) -> (B, W, F)
            x_spatial = torch.matmul(x, adj_matrix)
        else:
            x_spatial = x

        # 2. Temporal Encoder: Conv1d over time dimension W for each feature channel
        # Permute to (B, F, W)
        x_temp = x_spatial.permute(0, 2, 1)
        h_temp = F.relu(self.temporal_encoder(x_temp))  # (B, hidden_dim, W)

        # 3. Spatial dense layer over temporal features
        h_perm = h_temp.permute(0, 2, 1)  # (B, W, hidden_dim)
        h_out = F.relu(self.spatial_fc(h_perm))  # (B, W, hidden_dim)

        # 4. Flatten (B, W * hidden_dim) for forecasting head
        h_flat = h_out.reshape(B, -1)
        preds = self.predictor(h_flat)  # (B, forecast_horizon * F)
        if self.forecast_horizon > 1:
            preds = preds.reshape(B, self.forecast_horizon, num_feats)

        return preds


def resolve_sensor_locations(
    g: nx.Graph,
    sensor_cols: List[str],
) -> Dict[str, Set[str]]:
    """Map BATADAL sensor column names to sets of C-Town physical graph nodes.

    - P_<junction> -> {junction_node_id}
    - L_<tank>     -> {tank_node_id}
    - F_<link_id>  -> {node1, node2} (link edge endpoints)
    - S_<link_id>  -> {node1, node2} (link edge endpoints)

    Args:
        g: NetworkX graph representing C-Town topology.
        sensor_cols: Ordered list of sensor column names.

    Returns:
        Dict mapping sensor_name -> set of C-Town node IDs.

    Raises:
        ValueError: If any sensor column cannot be mapped to the C-Town graph.
    """
    sensor_nodes: Dict[str, Set[str]] = {}
    unresolved: List[str] = []

    # Map node sensors (P_, L_)
    for col in sensor_cols:
        if col.startswith("L_") or col.startswith("P_"):
            comp_id = col[2:]
            if g.has_node(comp_id):
                sensor_nodes[col] = {comp_id}
            else:
                unresolved.append(col)

    # Map link sensors (F_, S_) from edge metadata
    for u, v, d in g.edges(data=True):
        link_id = d.get("id")
        if link_id:
            f_name = f"F_{link_id}"
            s_name = f"S_{link_id}"
            if f_name in sensor_cols:
                sensor_nodes[f_name] = {str(u), str(v)}
            if s_name in sensor_cols:
                sensor_nodes[s_name] = {str(u), str(v)}

    # Check for any unmapped sensor
    for col in sensor_cols:
        if col not in sensor_nodes:
            unresolved.append(col)

    if unresolved:
        raise ValueError(
            f"Failed to map sensors to C-Town graph topology. Unresolved sensors: {unresolved}"
        )

    return sensor_nodes


def build_adjacency_matrix_from_graph(
    g: nx.Graph,
    sensor_cols: List[str],
    max_hops: int = 3,
    sensor_to_node_map: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Build a normalized 43x43 sensor adjacency matrix using topological graph distance.

    Maps all sensors to C-Town network components, computes shortest-path hop distances,
    and applies <=3-hop distance weighting with symmetric degree normalization D^{-1/2} A D^{-1/2}.

    Args:
        g: NetworkX graph representing hydraulic network topology.
        sensor_cols: Ordered list of sensor feature column names.
        max_hops: Maximum shortest-path hop distance for topological connection (default 3).
        sensor_to_node_map: Unused legacy override parameter.

    Returns:
        PyTorch float tensor of shape (F, F) containing normalized adjacency matrix.
    """
    F_count = len(sensor_cols)
    adj = np.eye(F_count, dtype=np.float32)

    sensor_nodes = resolve_sensor_locations(g, sensor_cols)
    spl = dict(nx.all_pairs_shortest_path_length(g))

    for i, s1 in enumerate(sensor_cols):
        nodes1 = sensor_nodes[s1]
        for j, s2 in enumerate(sensor_cols):
            if i == j:
                continue
            nodes2 = sensor_nodes[s2]
            # Minimum shortest path distance between any component node of s1 and s2
            dists = [spl[n1][n2] for n1 in nodes1 for n2 in nodes2 if n2 in spl.get(n1, {})]
            if dists:
                min_dist = min(dists)
                if min_dist <= max_hops:
                    adj[i, j] = 1.0 / (1.0 + min_dist)

    nonzero_count = int(np.count_nonzero(adj))
    if nonzero_count <= F_count:
        raise ValueError("Sensor adjacency matrix collapsed to identity matrix. Check sensor topology mapping.")

    # Symmetric degree normalization: D^{-1/2} A D^{-1/2}
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = np.power(deg[mask], -0.5)
    D = np.diag(deg_inv_sqrt)
    norm_adj = D @ adj @ D

    logger.info(
        "Constructed C-Town 43-sensor topology adjacency matrix: shape %s, non-zero entries=%d",
        norm_adj.shape,
        nonzero_count,
    )
    return torch.from_numpy(norm_adj.astype(np.float32))


