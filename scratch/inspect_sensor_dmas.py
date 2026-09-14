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
    
    j_dma = {}
    for _, row in net.junctions.iterrows():
        pat = str(row["pattern"])
        m = re.match(r"DMA(\d+)_pat", pat)
        if m:
            j_dma[row["id"]] = int(m.group(1))

    spl = dict(nx.all_pairs_shortest_path_length(g))
    
    sensor_dma = {}
    dma_sensors = {}
    
    for s_name, nodes in sensor_nodes.items():
        # find min distance to each DMA
        dma_dists = {}
        for sn in nodes:
            for j_node, dma in j_dma.items():
                if j_node in spl.get(sn, {}):
                    d = spl[sn][j_node]
                    if dma not in dma_dists or d < dma_dists[dma]:
                        dma_dists[dma] = d
        # pick DMA with minimum shortest path distance
        best_dma = min(dma_dists.keys(), key=lambda d: dma_dists[d])
        min_dist = dma_dists[best_dma]
        sensor_dma[s_name] = (best_dma, min_dist)
        dma_sensors.setdefault(best_dma, []).append(s_name)
        
    print("=== SENSOR TO NEAREST HYDRAULIC DMA ===")
    for dma in sorted(dma_sensors.keys()):
        sensors = dma_sensors[dma]
        print(f"\nRegion {dma} (DMA {dma}) - {len(sensors)} sensors:")
        for s in sensors:
            print(f"  {s}: dist={sensor_dma[s][1]}")

if __name__ == "__main__":
    main()
