"""
zoning.py

Operational zoning interface.

STATUS: PROTOTYPE APPROXIMATION.
The exact LightEdge-IDS zoning scheme is not specified in the supplied
project documents (extended abstract / next-steps roadmap). Those documents
only say "Leiden community detection OR the zoning scheme used in
LightEdge-IDS" -- the roadmap does not pin down LightEdge-IDS's actual
algorithm, parameters, or resulting zone count.

This module implements a swappable ZoningStrategy interface and a default
Louvain-based community-detection implementation (modularity maximization,
the same family of method as Leiden). Louvain is used instead of Leiden here
because it has no non-Python native-extension build dependency
(python-igraph/leidenalg can be substituted later without changing the
interface -- see LeidenZoning below, which raises ImportError with
instructions if leidenalg is not installed).

Do not treat the resulting zone assignments as the true LightEdge-IDS zones.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class ZoningStrategy(ABC):
    """Interface for an operational zoning method. Swap implementations
    freely -- everything downstream only depends on `assign_zones`."""

    @abstractmethod
    def assign_zones(self, g: nx.Graph) -> dict[str, int]:
        """Return a mapping of node id -> zone id (int)."""
        raise NotImplementedError


class LouvainZoning(ZoningStrategy):
    """Prototype zoning via Louvain modularity-maximization community
    detection (python-louvain / community package)."""

    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state

    def assign_zones(self, g: nx.Graph) -> dict[str, int]:
        try:
            import community as community_louvain  # python-louvain
        except ImportError as e:
            raise ImportError(
                "python-louvain is required for LouvainZoning. "
                "Install with: pip install python-louvain"
            ) from e

        partition = community_louvain.best_partition(
            g, resolution=self.resolution, random_state=self.random_state
        )
        n_zones = len(set(partition.values()))
        logger.info("LouvainZoning: assigned %d nodes to %d zones", len(partition), n_zones)
        return partition


class LeidenZoning(ZoningStrategy):
    """Alternative zoning via Leiden algorithm (igraph/leidenalg).

    Not used by default because leidenalg/python-igraph require a native
    build toolchain that may not be available in every environment. Swap
    this in for LouvainZoning in the pipeline config if your environment
    has these packages installed -- the rest of the codebase is unaffected
    since both implement ZoningStrategy.
    """

    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state

    def assign_zones(self, g: nx.Graph) -> dict[str, int]:
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise ImportError(
                "python-igraph and leidenalg are required for LeidenZoning. "
                "Install with: pip install python-igraph leidenalg"
            ) from e

        nodes = list(g.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in g.edges()]
        h = ig.Graph(n=len(nodes), edges=edges)
        partition = leidenalg.find_partition(
            h,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )
        result = {nodes[i]: partition.membership[i] for i in range(len(nodes))}
        logger.info(
            "LeidenZoning: assigned %d nodes to %d zones", len(result), len(set(result.values()))
        )
        return result


class DMAZoning(ZoningStrategy):
    """Zoning derived from the DMA (District Metered Area) demand patterns
    already present in the CTOWN.INP file itself.

    STATUS: THIS IS REAL DATA, NOT AN APPROXIMATION.
    Each junction in CTOWN.INP is assigned a demand pattern named
    'DMA<k>_pat' (k=1..5). These DMA labels are part of the standard
    C-Town/BATADAL benchmark network and represent genuine, physically
    meaningful operational zones (District Metered Areas) -- not a
    graph-community heuristic. Where a zoning ground truth is needed,
    prefer this over LouvainZoning/LeidenZoning.

    Nodes without a DMA-tagged demand pattern (tanks, reservoirs, and a
    handful of zero-demand junctions -- 54 of 388 junctions in CTOWN.INP)
    are assigned to the zone of their nearest DMA-tagged graph neighbor
    (BFS), and left as zone -1 if no such neighbor is reachable.
    """

    def assign_zones(self, g: nx.Graph) -> dict[str, int]:
        import re

        zone_of: dict[str, int] = {}
        for n, data in g.nodes(data=True):
            pattern = data.get("pattern")
            if pattern:
                match = re.match(r"DMA(\d+)_pat", str(pattern))
                if match:
                    zone_of[n] = int(match.group(1))

        unassigned = [n for n in g.nodes() if n not in zone_of]
        # Propagate zone labels to unassigned nodes via BFS over labeled neighbors.
        changed = True
        while unassigned and changed:
            changed = False
            still_unassigned = []
            for n in unassigned:
                neighbor_zones = [zone_of[m] for m in g.neighbors(n) if m in zone_of]
                if neighbor_zones:
                    zone_of[n] = max(set(neighbor_zones), key=neighbor_zones.count)
                    changed = True
                else:
                    still_unassigned.append(n)
            unassigned = still_unassigned

        for n in unassigned:
            zone_of[n] = -1  # no DMA-labeled node reachable (isolated component)

        logger.info(
            "DMAZoning: %d/%d nodes matched to a DMA zone directly, %d unresolved",
            len(g.nodes()) - len(unassigned),
            g.number_of_nodes(),
            len(unassigned),
        )
        return zone_of


def assign_zones(g: nx.Graph, method: str = "dma", **kwargs: Any) -> dict[str, int]:
    """Convenience dispatcher. method in {'dma', 'louvain', 'leiden'}.

    Default is 'dma' -- the real DMA zoning present in CTOWN.INP -- since it
    is genuine data rather than a heuristic approximation. Use 'louvain' or
    'leiden' only if you specifically need a graph-community-based zoning
    (e.g. to benchmark against DMA zoning, or for networks without DMA tags).
    """
    strategies = {"dma": DMAZoning, "louvain": LouvainZoning, "leiden": LeidenZoning}
    if method not in strategies:
        raise ValueError(f"Unknown zoning method '{method}'. Choose from {list(strategies)}.")
    strategy = strategies[method](**kwargs)
    return strategy.assign_zones(g)
