import sys
import re
import pandas as pd
import networkx as nx
from pathlib import Path
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.detection.graph_model import resolve_sensor_locations

def main():
    net = load_ctown_network("data/raw/CTOWN.INP")
    g = build_network_graph(net)
    
    df = pd.read_csv("data/raw/BATADAL_dataset03.csv")
    sensor_cols = [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    sensor_nodes = resolve_sensor_locations(g, sensor_cols)
    
    print("=== JUNCTION DEMAND PATTERNS IN CTOWN.INP ===")
    j_dma = {}
    for _, row in net.junctions.iterrows():
        pat = str(row["pattern"])
        m = re.match(r"DMA(\d+)_pat", pat)
        if m:
            j_dma[row["id"]] = int(m.group(1))
            
    print(f"Junctions with explicit DMA pattern: {len(j_dma)} / {len(net.junctions)}")
    dma_counts = {}
    for j, d in j_dma.items():
        dma_counts[d] = dma_counts.get(d, 0) + 1
    print("Explicit DMA junction counts:", sorted(dma_counts.items()))

    # Now check our 12 junction sensors P_J...
    print("\n=== 12 JUNCTION SENSORS ATTRIBUTED TO DMAs ===")
    for col in sensor_cols:
        if col.startswith("P_J"):
            node_id = col[2:]
            dma = j_dma.get(node_id, "NONE")
            print(f"{col} (Node {node_id}): DMA {dma}")

    # Check tanks L_T...
    print("\n=== 7 TANKS ATTRIBUTED TO DMAs (nearest explicit DMA junction) ===")
    spl = dict(nx.all_pairs_shortest_path_length(g))
    for col in sensor_cols:
        if col.startswith("L_T"):
            t_node = col[2:]
            # find shortest distance to any explicit DMA junction
            dists = []
            for j_node, dma in j_dma.items():
                if j_node in spl.get(t_node, {}):
                    dists.append((spl[t_node][j_node], dma, j_node))
            dists.sort()
            print(f"{col} (Node {t_node}): nearest DMA is {dists[0][1]} via junction {dists[0][2]} (dist={dists[0][0]})")

    # Check pumps/valves F_..., S_...
    print("\n=== LINK SENSORS ATTRIBUTED TO DMAs (endpoints explicit DMAs) ===")
    for col in sensor_cols:
        if col.startswith("F_") or col.startswith("S_"):
            nodes = sensor_nodes[col]
            dmas = [j_dma.get(n) for n in nodes if n in j_dma]
            print(f"{col} (Nodes {nodes}): explicit DMAs = {dmas}")

if __name__ == "__main__":
    main()
