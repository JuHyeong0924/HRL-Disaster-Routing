import json
import networkx as nx
import numpy as np
import math
import argparse
import sys
import os

# Ensure we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.envs.disaster_map import DisasterMap

def create_grid_zones(node_file, net_file, out_prefix):
    dm = DisasterMap(node_file, net_file)
    G = dm.graph
    
    nodes = list(G.nodes(data=True))
    V = len(nodes)
    N = max(2, math.ceil(math.sqrt(V / 16.0))) # roughly N*N zones, 16 nodes per zone on avg
    
    xs = [data.get('x', 0) for n, data in nodes]
    ys = [data.get('y', 0) for n, data in nodes]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    eps_x = (max_x - min_x) * 1e-6
    eps_y = (max_y - min_y) * 1e-6
    max_x += eps_x
    max_y += eps_y
    
    dx = (max_x - min_x) / N
    dy = (max_y - min_y) / N
    
    grid_cells = {}
    for n, data in nodes:
        x, y = data.get('x', 0), data.get('y', 0)
        gx = int((x - min_x) / dx)
        gy = int((y - min_y) / dy)
        gx = min(max(gx, 0), N-1)
        gy = min(max(gy, 0), N-1)
        cell = (gx, gy)
        if cell not in grid_cells:
            grid_cells[cell] = []
        grid_cells[cell].append(n)
        
    cell_to_zone = {}
    zone_idx = 0
    for cell in sorted(grid_cells.keys()):
        cell_to_zone[cell] = zone_idx
        zone_idx += 1
        
    node_to_zone = {}
    for cell, ns in grid_cells.items():
        z = cell_to_zone[cell]
        for n in ns:
            node_to_zone[n] = z
            
    with open(f"{out_prefix}_node_to_zone.json", "w") as f:
        json.dump(node_to_zone, f, indent=4)
        
    zone_edges = set()
    for u, v in G.edges():
        zu = node_to_zone[u]
        zv = node_to_zone[v]
        if zu != zv:
            zone_edges.add((zu, zv))
            zone_edges.add((zv, zu))
            
    zone_adj = {z: [] for z in range(zone_idx)}
    for zu, zv in zone_edges:
        zone_adj[zu].append(zv)
        
    for z in range(zone_idx):
        zone_adj[z] = sorted(list(set(zone_adj[z])))
        
    with open(f"{out_prefix}_zone_graph.json", "w") as f:
        json.dump({"k": zone_idx, "zone_adjacency": zone_adj}, f, indent=4)
        
    print(f"Grid partitioning complete: N={N}, K={zone_idx}")

def create_macro_zones(node_file, net_file, micro_json, out_prefix):
    """
    Groups Micro-Zones into Macro-Zones using their centroids.
    """
    dm = DisasterMap(node_file, net_file)
    G = dm.graph
    
    with open(micro_json, 'r') as f:
        node_to_micro = json.load(f)
        
    # Calculate Micro-Zone Centroids
    micro_coords = {}
    for n_str, micro_z in node_to_micro.items():
        n = int(n_str) if n_str.isdigit() else n_str
        if n in G.nodes:
            x = G.nodes[n].get('x', 0)
            y = G.nodes[n].get('y', 0)
            if micro_z not in micro_coords:
                micro_coords[micro_z] = []
            micro_coords[micro_z].append((x, y))
            
    micro_centroids = {}
    for micro_z, coords in micro_coords.items():
        avg_x = sum(c[0] for c in coords) / len(coords)
        avg_y = sum(c[1] for c in coords) / len(coords)
        micro_centroids[micro_z] = (avg_x, avg_y)
        
    K_micro = max(micro_centroids.keys()) + 1
    N = max(2, math.ceil(math.sqrt(K_micro / 10.0))) # approx 10 micro-zones per macro-zone
    
    xs = [c[0] for c in micro_centroids.values()]
    ys = [c[1] for c in micro_centroids.values()]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    eps_x = (max_x - min_x) * 1e-6
    eps_y = (max_y - min_y) * 1e-6
    max_x += eps_x
    max_y += eps_y
    
    dx = (max_x - min_x) / N
    dy = (max_y - min_y) / N
    
    grid_cells = {}
    for micro_z, (x, y) in micro_centroids.items():
        gx = int((x - min_x) / dx)
        gy = int((y - min_y) / dy)
        gx = min(max(gx, 0), N-1)
        gy = min(max(gy, 0), N-1)
        cell = (gx, gy)
        if cell not in grid_cells:
            grid_cells[cell] = []
        grid_cells[cell].append(micro_z)
        
    cell_to_macro = {}
    macro_idx = 0
    for cell in sorted(grid_cells.keys()):
        cell_to_macro[cell] = macro_idx
        macro_idx += 1
        
    micro_to_macro = {}
    for cell, mzs in grid_cells.items():
        mz = cell_to_macro[cell]
        for z in mzs:
            micro_to_macro[z] = mz
            
    # Also save direct node_to_macro for convenience
    node_to_macro = {}
    for n_str, micro_z in node_to_micro.items():
        node_to_macro[n_str] = micro_to_macro[micro_z]
            
    with open(f"{out_prefix}_micro_to_macro.json", "w") as f:
        json.dump(micro_to_macro, f, indent=4)
        
    with open(f"{out_prefix}_node_to_zone.json", "w") as f: # Save as node_to_zone for compatibility with ManagerEnv
        json.dump(node_to_macro, f, indent=4)
        
    # Build Macro-Zone Graph
    macro_edges = set()
    for u, v in G.edges():
        if str(u) in node_to_macro and str(v) in node_to_macro:
            mu = node_to_macro[str(u)]
            mv = node_to_macro[str(v)]
            if mu != mv:
                macro_edges.add((mu, mv))
                macro_edges.add((mv, mu))
            
    macro_adj = {z: [] for z in range(macro_idx)}
    for mu, mv in macro_edges:
        macro_adj[mu].append(mv)
        
    for z in range(macro_idx):
        macro_adj[z] = sorted(list(set(macro_adj[z])))
        
    with open(f"{out_prefix}_zone_graph.json", "w") as f:
        json.dump({"k": macro_idx, "zone_adjacency": macro_adj}, f, indent=4)
        
    print(f"Macro partitioning complete: Micro K={K_micro}, Macro K={macro_idx}")

if __name__ == '__main__':
    if len(sys.argv) == 4:
        create_grid_zones(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == '--macro':
        # Usage: python grid_partitioner.py --macro node.tntp net.tntp micro.json out_prefix
        create_macro_zones(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        create_grid_zones("data/Anaheim_node.tntp", "data/Anaheim_net.tntp", "data/grid_Anaheim")
