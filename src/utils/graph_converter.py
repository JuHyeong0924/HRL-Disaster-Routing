import torch
from torch_geometric.data import Data
import networkx as nx

# [✅ 핵심] 로봇 클래스 가져오기 (계산 위임용)
from src.agents.robot import UGV

class GraphConverter:
    def __init__(self, map_core):
        self.map_core = map_core
        self.node_mapping = {node: i for i, node in enumerate(map_core.graph.nodes())}
        
        # 맵 크기 정보 (정규화용)
        if hasattr(map_core, 'bounds') and len(map_core.bounds) == 4:
            self.min_x, self.min_y, self.max_x, self.max_y = map_core.bounds
        else:
            # Fallback (Legacy)
            self.min_x, self.min_y = 0.0, 0.0
            self.max_x, self.max_y = 1.0, 1.0

        # [✅ 핵심] 계산기 역할을 할 '더미 로봇' 생성
        # 실제 시뮬레이션용이 아니라, 스펙(Spec) 참조 및 비용 계산용입니다.
        default_config = {'max_battery': 100.0, 'speed': 40.0, 'consumption_rate': 1.0}
        self.dummy_ugv = UGV(robot_id="calculator_ugv", start_node=0, config=default_config)
        # self.dummy_uav = UAV(...) # 나중에 추가

    def _calculate_agent_cost(self, edge_data, agent_type='UGV'):
        """
        로봇 클래스의 메서드를 호출하여 비용 계산 (로직 일원화)
        """
        length = edge_data.get('length', 1.0)
        status = edge_data.get('status', 'Normal')
        damage = edge_data.get('damage', 0.0)
        
        if agent_type == 'UGV':
            # UGV 클래스에게 계산을 시킴 (하드코딩 제거)
            # -> UGV.predict_edge_cost() 호출
            return self.dummy_ugv.predict_edge_cost(length, status, damage)
            
        elif agent_type == 'UAV':
            # 나중에 UAV 클래스가 생기면 교체
            # 드론은 도로 파괴 영향 적음
            return length / 60.0, length * 1.2
        
        else:
            return length / 40.0, length

    def networkx_to_pyg(self, current_node, target_node, agent_type='UGV'):
        """
        NetworkX 그래프를 PyTorch Geometric 데이터로 변환 (Vectorized)
        Args:
            current_node: 현재 위치 인덱스 (int or Tensor)
            target_node: 목표 위치 인덱스 (int or Tensor)
        """
        G = self.map_core.graph
        
        # 1. Node Features (Bulk Extraction)
        # Nodes are assumed to be 0..N-1 mapped by node_mapping if strictly ordered,
        # but self.node_mapping maps RealID -> Index.
        # We need to ensure we iterate in Index order.
        
        # In __init__, node_mapping = {node: i ...}. 
        # So we should iterate over nodes in the order of their indices.
        # Extract features in order.
        
        # Pre-calculate or cache these is better, but here we vectorize the extraction.
        # G.nodes(data=True) order is consistent with mapping if mapping derived from it.
        
        # Using lists is faster than repeated dict access in loop
        nodes_data = list(G.nodes(data=True)) # [(id, {attr}), ...]
        num_nodes = len(nodes_data)
        
        # Extract X, Y
        # Default 0.0 if missing
        xs = [d.get('x', 0.0) for _, d in nodes_data]
        ys = [d.get('y', 0.0) for _, d in nodes_data]
        
        # Tensor conversion
        x_tensor = torch.tensor(xs, dtype=torch.float)
        y_tensor = torch.tensor(ys, dtype=torch.float)
        
        # Normalize
        denom_x = (self.max_x - self.min_x) + 1e-6
        denom_y = (self.max_y - self.min_y) + 1e-6
        
        pos_x = (x_tensor - self.min_x) / denom_x
        pos_y = (y_tensor - self.min_y) / denom_y
        
        # One-hot encoding for Robot/Target
        is_robot = torch.zeros(num_nodes, dtype=torch.float)
        is_target = torch.zeros(num_nodes, dtype=torch.float)
        
        if isinstance(current_node, torch.Tensor):
            current_node = current_node.item()
        if isinstance(target_node, torch.Tensor):
            target_node = target_node.item()
            
        is_robot[current_node] = 1.0
        is_target[target_node] = 1.0
        
        # [Modified] Stack: [N, 6] -> x, y, robot, target, visit_count, network_dist
        
        # 5. Visit Count (New Feature)
        visit_count = torch.zeros(num_nodes, dtype=torch.float)
        visit_count[current_node] = 1.0 # Start with 1 visit
        
        # 6. Network Distance (APSP/Dijkstra) (New Feature)
        # Calculate distance from *each node* to *target_node*
        target_id_real = list(G.nodes())[target_node] # Get real ID from index
        try:
            # CPU Dijkstra (Single Target)
            # returns {node_id: dist}
            # weight='length'
            path_lengths = nx.single_source_dijkstra_path_length(G, target_id_real, weight='length')
            
            # Map back to tensor
            dists = []
            max_dist_val = 1.0
            
            # First pass to find max for normalization (or use map bounds/heuristic)
            # Just use max found value or 1.0
            max_found = 0.0
            
            # We need to iterate in index order to match tensor rows
            nodes_list = list(G.nodes())
            
            for n_id in nodes_list:
                d = path_lengths.get(n_id, float('inf'))
                if d != float('inf') and d > max_found:
                    max_found = d
                    
            if max_found == 0: max_found = 1.0
            if max_found == 0: max_found = 1.0
            
            for n_id in nodes_list:
                d = path_lengths.get(n_id, float('inf'))
                if d == float('inf'):
                    dists.append(1.0) # Unreachable = max normalized dist? or 1.0
                else:
                    dists.append(d / max_found)
                    
            network_dist = torch.tensor(dists, dtype=torch.float)
            
        except Exception:
            # Fallback if error (e.g. disconnected target?)
            network_dist = torch.ones(num_nodes, dtype=torch.float)

        x = torch.stack([pos_x, pos_y, is_robot, is_target, visit_count, network_dist], dim=1)
        
        # 2. Edge Features
        # Edge iteration
        edges_data = list(G.edges(data=True))
        
        # [Fix] Bidirectional Edge Handling for Physics Engine
        # Physics engine expects 76 edges (38 undirected * 2)
        # But G.edges() only gives 38.
        # We manually expand to bidirectional if undirected.
        
        final_u_list = []
        final_v_list = []
        final_edge_props = [] # Stores 'd' dict
        
        # Check directionality
        is_directed = G.is_directed()
        
        for u, v, d in edges_data:
            # Add u->v
            final_u_list.append(self.node_mapping[u])
            final_v_list.append(self.node_mapping[v])
            final_edge_props.append(d)
            
            if not is_directed:
                # Add v->u (Reverse)
                final_u_list.append(self.node_mapping[v])
                final_v_list.append(self.node_mapping[u])
                final_edge_props.append(d) # Share same props (undirected)
        
        edge_index = torch.tensor([final_u_list, final_v_list], dtype=torch.long)
        
        # Extract Attributes from final_edge_props
        lengths = [d.get('length', 1.0) for d in final_edge_props]
        damages = [d.get('damage', 0.0) for d in final_edge_props]
        is_closed = [1.0 if d.get('status') == 'Closed' else 0.0 for d in final_edge_props]
        is_danger = [1.0 if d.get('status') == 'Danger' else 0.0 for d in final_edge_props]
        has_building = [1.0 if d.get('has_building') else 0.0 for d in final_edge_props]
        
        # Predictions (Agent Cost)
        costs = [self._calculate_agent_cost(d, agent_type) for d in final_edge_props]
        
        # [Fix] Clip Infinite values to prevent NaN in Neural Network
        # Time and Energy can be Inf for Closed roads. Clip to large finite value (e.g. 100.0)
        # Normal Time ~0.025h, Energy ~0.1%. So 100.0 is sufficiently large (4000x normal).
        exp_times = [100.0 if c[0] == float('inf') else c[0] for c in costs]
        energy_costs = [100.0 if c[1] == float('inf') else c[1] for c in costs]
        
        # Stack Edge Attrs
        # [length, damage, expected_time, energy_cost, is_closed, has_building, is_danger, capacity, speed]
        capacities = [d.get('capacity', 0.0) for d in final_edge_props]
        speeds = [d.get('speed', 0.0) for d in final_edge_props]
        
        edge_attr = torch.tensor([
            lengths, damages, exp_times, energy_costs, is_closed, has_building, is_danger, capacities, speeds
        ], dtype=torch.float).t() # [E, 9]
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
