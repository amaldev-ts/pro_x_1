import sys
import pandas as pd
from pathlib import Path
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.network.zoning import assign_zones
from src.detection.graph_model import resolve_sensor_locations

def main():
    inp_path = Path("data/raw/CTOWN.INP")
    df = pd.read_csv("data/raw/BATADAL_dataset03.csv")
    
    # Exclude timestamp and label columns if present
    sensor_cols = [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    print(f"Total sensors: {len(sensor_cols)}")

    net = load_ctown_network(inp_path)
    g = build_network_graph(net)
    
    # Map sensors to graph nodes
    sensor_nodes = resolve_sensor_locations(g, sensor_cols)
    
    # Assign zones to all graph nodes
    dma_zones = assign_zones(g, method="dma")
    
    # Assign sensors to regions (DMAs)
    sensor_to_dma = {}
    dma_to_sensors = {}
    
    for s_name, nodes in sensor_nodes.items():
        # Get zones of nodes corresponding to this sensor
        zones = set(dma_zones[n] for n in nodes if n in dma_zones)
        # If link sensor spans across DMAs, track all or primary zone
        sensor_to_dma[s_name] = sorted(list(zones))
        for z in zones:
            dma_to_sensors.setdefault(z, []).append(s_name)
            
    print("\n--- DMA Zone to Sensors ---")
    for z in sorted(dma_to_sensors.keys()):
        print(f"DMA {z}: {len(dma_to_sensors[z])} sensors -> {dma_to_sensors[z]}")
        
    print("\n--- Sensors spanning multiple DMAs ---")
    for s_name, zones in sensor_to_dma.items():
        if len(zones) > 1:
            print(f"{s_name}: DMAs {zones}")

if __name__ == "__main__":
    main()
