
import os
import networkx as nx
import torch
from torch_geometric.data import Data

class GraphLoader:
    def __init__(self, node_file, net_file):
        """
        Loads TNTP format maps (Anaheim, SiouxFalls) and converts to NetworkX/PyG.
        """
        self.node_file = node_file
        self.net_file = net_file
        self.graph = nx.Graph() # Undirected, using MAPPED (0-based) IDs
        self.pos = {} 
        self.node_mapping = {} # Orig ID -> 0-based Index
        self.inv_node_mapping = {} # 0-based Index -> Orig ID
        
        self._load_network()
        
    def _load_network(self):
        # 1. Load Nodes
        with open(self.node_file, 'r') as f:
            lines = f.readlines()
            
        start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('Node'):
                start_line = i
                break
        
        temp_nodes = []
        for line in lines[start_line+1:]:
            parts = line.strip().replace(';', '').split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                temp_nodes.append((node_id, x, y))
        
        # Create Mapping
        temp_nodes.sort(key=lambda x: x[0])
        for idx, (nid, x, y) in enumerate(temp_nodes):
            self.node_mapping[nid] = idx
            self.inv_node_mapping[idx] = nid
            
            # Add to NX with MAPPED ID
            self.graph.add_node(idx, x=x, y=y)
            self.pos[idx] = (x, y)
            
        # 2. Load Edges
        with open(self.net_file, 'r') as f:
            lines = f.readlines()
            
        start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('~'):
                start_line = i + 1
                break
                
        for line in lines[start_line:]:
            parts = line.strip().replace(';', '').split()
            if len(parts) >= 3:
                u_orig = int(parts[0])
                v_orig = int(parts[1])
                
                if u_orig not in self.node_mapping or v_orig not in self.node_mapping:
                    continue
                    
                u = self.node_mapping[u_orig]
                v = self.node_mapping[v_orig]
                
                capacity = float(parts[2]) if len(parts) > 2 else 0.0
                length = float(parts[3]) if len(parts) > 3 else 0.0
                fft = float(parts[4]) if len(parts) > 4 else length
                
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v, weight=length, length=length, travel_time=fft, capacity=capacity)
                    
    def get_pyg_data(self):
        """
        Returns torch_geometric.data.Data object (Static Graph)
        """
        num_nodes = len(self.node_mapping)
        x = torch.zeros(num_nodes, 2)
        
        for i in range(num_nodes):
            x[i, 0] = self.graph.nodes[i]['x']
            x[i, 1] = self.graph.nodes[i]['y']
        
        # [Robustness] Normalize Coordinates to [0, 1] for Cross-Map Generalization
        if num_nodes > 0:
            x_min = x.min(dim=0)[0] # [2]
            x_max = x.max(dim=0)[0] # [2]
            scale = x_max - x_min
            scale[scale == 0] = 1.0 # Prevent div by zero
            x = (x - x_min) / scale
            
        edges = []
        edge_attrs = []
        
        for u, v, data in self.graph.edges(data=True):
            edges.append([u, v])
            edges.append([v, u])
            
            # Edge Feature: [length, damage, expected_time, energy_cost, is_closed, has_building, is_danger, capacity, speed]
            # Phase 1 (정상 도로): 모두 0, length, cap, speed만 값 존재
            l = data['length']
            tt = data.get('travel_time', l)  # travel_time이 없으면 length 사용
            cap = data.get('capacity', 0.0)
            speed = l / max(tt, 1e-6)  # 속도 = 거리 / 시간
            feat = [l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cap, speed]  # 9D: 정상 상태
            edge_attrs.append(feat)
            edge_attrs.append(feat)  # 양방향 동일
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_nx_graph(self):
        return self.graph
