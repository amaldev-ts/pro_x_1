"""
visualize_topology.py

Renders the C-Town network topology (colored by DMA zone) to a PNG using
matplotlib + the real node coordinates from CTOWN.INP's [COORDINATES]
section. Falls back to a spring layout for any node missing coordinates.

Status: IMPLEMENTED.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.network.zoning import assign_zones

logger = logging.getLogger(__name__)

NODE_TYPE_COLORS = {"junction": None, "tank": "black", "reservoir": "blue"}
NODE_TYPE_MARKERS = {"junction": "o", "tank": "s", "reservoir": "^"}
NODE_TYPE_SIZES = {"junction": 12, "tank": 90, "reservoir": 140}


def plot_topology(inp_path: str, out_path: str, zoning_method: str = "dma") -> None:
    """Parse, build graph, zone, and render the topology to out_path."""
    net = load_ctown_network(inp_path)
    g = build_network_graph(net)
    zones = assign_zones(g, method=zoning_method)

    pos = {n: (d["x"], d["y"]) for n, d in g.nodes(data=True) if "x" in d and "y" in d}
    missing = [n for n in g.nodes() if n not in pos]
    if missing:
        logger.warning("%d nodes missing coordinates; using spring layout for those", len(missing))
        fallback_pos = nx.spring_layout(g.subgraph(missing), seed=42)
        pos.update(fallback_pos)

    zone_ids = sorted({z for z in zones.values() if z != -1})
    cmap = plt.get_cmap("tab10")
    zone_colors = {z: cmap(i % 10) for i, z in enumerate(zone_ids)}
    zone_colors[-1] = "lightgray"

    fig, ax = plt.subplots(figsize=(14, 12))

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="gray", width=0.6, alpha=0.6)

    for node_type, marker in NODE_TYPE_MARKERS.items():
        nodes = [n for n, d in g.nodes(data=True) if d.get("type") == node_type]
        if not nodes:
            continue
        colors = [zone_colors[zones.get(n, -1)] for n in nodes]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_color=colors,
            node_shape=marker,
            node_size=[NODE_TYPE_SIZES[node_type]] * len(nodes),
            ax=ax,
            edgecolors="black" if node_type != "junction" else "none",
            linewidths=0.8,
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=zone_colors[z], markersize=10, label=f"Zone {z}")
        for z in zone_ids
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="black", markersize=10, label="Tank"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor="black", markersize=10, label="Reservoir"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(f"C-Town Network Topology (zoning={zoning_method})")
    ax.set_axis_off()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved topology plot to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/CTOWN.INP"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/ctown_topology.png"
    plot_topology(inp, out)
