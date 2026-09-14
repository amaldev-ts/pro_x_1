"""
graph_builder.py

Builds a NetworkX graph representation of the parsed C-Town network:
nodes = junctions/tanks/reservoirs, edges = pipes/pumps/valves.

Status: IMPLEMENTED.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from src.network.ctown_loader import CTownNetwork

logger = logging.getLogger(__name__)

NODE_TYPES = {"junction", "tank", "reservoir"}
EDGE_TYPES = {"pipe", "pump", "valve"}


def build_network_graph(net: CTownNetwork) -> nx.Graph:
    """Build an undirected NetworkX graph from a parsed CTownNetwork.

    Node attributes: type ('junction'/'tank'/'reservoir'), plus hydraulic
    properties available for that component type (elevation, demand,
    tank levels, etc.), plus x/y coordinates when available.

    Edge attributes: type ('pipe'/'pump'/'valve'), id, and available
    hydraulic properties (length, diameter, roughness for pipes;
    pump_type for pumps; valve_type/setting for valves).

    Args:
        net: A CTownNetwork produced by ctown_loader.load_ctown_network.

    Returns:
        A networkx.Graph with hydraulic connectivity and metadata.
    """
    g = nx.Graph()
    coords = net.coordinates.set_index("id") if len(net.coordinates) else None

    def _coord(node_id: str) -> dict[str, Any]:
        if coords is not None and node_id in coords.index:
            return {"x": float(coords.loc[node_id, "x"]), "y": float(coords.loc[node_id, "y"])}
        return {}

    for _, row in net.junctions.iterrows():
        g.add_node(
            row["id"],
            type="junction",
            elevation=row["elevation"],
            base_demand=row["base_demand"],
            pattern=row["pattern"],
            **_coord(row["id"]),
        )

    for _, row in net.tanks.iterrows():
        g.add_node(
            row["id"],
            type="tank",
            elevation=row["elevation"],
            init_level=row["init_level"],
            min_level=row["min_level"],
            max_level=row["max_level"],
            diameter=row["diameter"],
            **_coord(row["id"]),
        )

    for _, row in net.reservoirs.iterrows():
        g.add_node(row["id"], type="reservoir", base_head=row["base_head"], **_coord(row["id"]))

    for _, row in net.pipes.iterrows():
        g.add_edge(
            row["node1"],
            row["node2"],
            id=row["id"],
            type="pipe",
            length=row["length"],
            diameter=row["diameter"],
            roughness=row["roughness"],
        )

    for _, row in net.pumps.iterrows():
        g.add_edge(row["node1"], row["node2"], id=row["id"], type="pump", pump_type=row["pump_type"])

    for _, row in net.valves.iterrows():
        g.add_edge(
            row["node1"],
            row["node2"],
            id=row["id"],
            type="valve",
            valve_type=row["valve_type"],
            setting=row["setting"],
            diameter=row["diameter"],
        )

    logger.info(
        "Built graph: %d nodes, %d edges (%d connected components)",
        g.number_of_nodes(),
        g.number_of_edges(),
        nx.number_connected_components(g),
    )
    return g


def node_metadata_table(g: nx.Graph):
    """Return node metadata as a list of dicts (for CSV/JSON export)."""
    return [dict(id=n, **d) for n, d in g.nodes(data=True)]


def edge_metadata_table(g: nx.Graph):
    """Return edge metadata as a list of dicts (for CSV/JSON export)."""
    return [dict(node1=u, node2=v, **d) for u, v, d in g.edges(data=True)]


def component_to_sensor_mapping_interface(g: nx.Graph) -> dict[str, list[str]]:
    """Placeholder interface: component id -> list of sensor ids.

    STATUS: PLACEHOLDER. The supplied project documents do not specify
    which physical sensors exist in the BATADAL/C-Town deployment or how
    they map to components. This function currently maps every node to
    itself (assumes one virtual "sensor" per monitored node) so downstream
    modules have a consistent interface to call. Replace with the real
    sensor deployment map when available.
    """
    return {n: [n] for n, d in g.nodes(data=True) if d.get("type") in {"tank", "junction"}}
