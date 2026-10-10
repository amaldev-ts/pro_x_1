import sys
import pandas as pd
import networkx as nx
from pathlib import Path
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.network.zoning import assign_zones
from src.detection.graph_model import resolve_sensor_locations, build_adjacency_matrix_from_graph

def main():
    inp_path = Path("data/raw/CTOWN.INP")
    df = pd.read_csv("data/raw/BATADAL_dataset03.csv")
    sensor_cols = [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    
    net = load_ctown_network(inp_path)
    g = build_network_graph(net)
    sensor_nodes = resolve_sensor_locations(g, sensor_cols)
    
    print("=== SENSOR TYPES & LOCATIONS ===")
    tanks = [s for s in sensor_cols if s.startswith("L_")]
    junctions = [s for s in sensor_cols if s.startswith("P_")]
    link_f = [s for s in sensor_cols if s.startswith("F_")]
    link_s = [s for s in sensor_cols if s.startswith("S_")]
    print(f"Tanks ({len(tanks)}): {tanks}")
    print(f"Junctions ({len(junctions)}): {junctions}")
    print(f"Flow sensors ({len(link_f)}): {link_f}")
    print(f"Status sensors ({len(link_s)}): {link_s}")

    # Build sensor-level graph where edge exists if shortest path <= 3 hops
    adj_tensor = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)
    adj_mat = adj_tensor.numpy()
    
    sensor_g = nx.Graph()
    for i, s in enumerate(sensor_cols):
        sensor_g.add_node(s)
    for i, s1 in enumerate(sensor_cols):
        for j, s2 in enumerate(sensor_cols):
            if i < j and adj_mat[i, j] > 0:
                sensor_g.add_edge(s1, s2)
                
    print(f"\nSensor Graph (<=3 hops): {sensor_g.number_of_nodes()} nodes, {sensor_g.number_of_edges()} edges")
    
    # Louvain community detection on sensor graph
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(sensor_g, random_state=42)
        communities = {}
        for s, cid in partition.items():
            communities.setdefault(cid, []).append(s)
        print("\n=== LOUVAIN COMMUNITIES ON SENSOR GRAPH ===")
        for cid, sensors in sorted(communities.items()):
            print(f"Region {cid + 1} ({len(sensors)} sensors): {sensors}")
    except Exception as e:
        print(f"Louvain failed: {e}")

    # Check tanks and their hydraulic sub-networks / neighborhoods
    print("\n=== TANK-CENTERED REGIONS (1-hop / 2-hop sensors from tanks) ===")
    spl = dict(nx.all_pairs_shortest_path_length(g))
    for t_sensor in tanks:
        t_node = list(sensor_nodes[t_sensor])[0]
        near_sensors = []
        for s_name, s_nodes in sensor_nodes.items():
            dists = [spl[t_node][sn] for sn in s_nodes if sn in spl.get(t_node, {})]
            if dists and min(dists) <= 5:
                near_sensors.append((s_name, min(dists)))
        near_sensors.sort(key=lambda x: x[1])
        print(f"Tank {t_sensor} (Node {t_node}) neighborhood <=5 hops: {[s[0] for s in near_sensors]}")

if __name__ == "__main__":
    main()
