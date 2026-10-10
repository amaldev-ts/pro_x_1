"""
export_outputs.py

Runs the Phase 1 network pipeline (parse -> graph -> zone) and writes the
required Phase 1 output artifacts to data/processed/:

- network_nodes.csv          (node metadata incl. assigned zone)
- network_edges.csv          (edge metadata)
- adjacency.json             (adjacency list)
- component_sensor_map.json  (PLACEHOLDER interface, see graph_builder.py)
- network_summary.json       (component counts)

Status: IMPLEMENTED.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import (
    build_network_graph,
    component_to_sensor_mapping_interface,
    edge_metadata_table,
    node_metadata_table,
)
from src.network.zoning import assign_zones

logger = logging.getLogger(__name__)


def run(inp_path: str, out_dir: str, zoning_method: str = "dma") -> dict:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    net = load_ctown_network(inp_path)
    g = build_network_graph(net)
    zones = assign_zones(g, method=zoning_method)
    for n in g.nodes():
        g.nodes[n]["zone"] = zones.get(n, -1)

    nodes = pd.DataFrame(node_metadata_table(g))
    edges = pd.DataFrame(edge_metadata_table(g))
    nodes.to_csv(out_dir_p / "network_nodes.csv", index=False)
    edges.to_csv(out_dir_p / "network_edges.csv", index=False)

    adjacency = {n: list(g.neighbors(n)) for n in g.nodes()}
    with open(out_dir_p / "adjacency.json", "w") as f:
        json.dump(adjacency, f, indent=2)

    sensor_map = component_to_sensor_mapping_interface(g)
    with open(out_dir_p / "component_sensor_map.json", "w") as f:
        json.dump(sensor_map, f, indent=2)

    summary = net.summary()
    summary["n_zones"] = len({z for z in zones.values() if z != -1})
    summary["zoning_method"] = zoning_method
    with open(out_dir_p / "network_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote Phase 1 network outputs to %s: %s", out_dir_p, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/CTOWN.INP"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    print(run(inp, out))
