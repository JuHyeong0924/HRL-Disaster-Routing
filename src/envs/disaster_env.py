import gymnasium as gym
import numpy as np
import networkx as nx
import random
import torch
from gymnasium import spaces
from typing import Dict, List, Optional
from torch_geometric.data import Batch

from src.envs.disaster_map import DisasterMap
from src.utils.graph_converter import GraphConverter

class DisasterEnv:
    def __init__(self, node_file: str, net_file: str, reward_config: Optional[Dict] = None, device: Optional[str] = None, verbose: bool = True, enable_disaster: bool = False):
        # 1. 물리 환경 (Map) 및 변환기 (Converter) 설정
        self.map_core = DisasterMap(node_file, net_file)
        self.converter = GraphConverter(self.map_core)
        self.enable_disaster = enable_disaster
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.verbose = verbose
        
        # Reward Configuration (Rescaled 1/10 for Stability)
        default_config = {
            'arrival_bonus_base': 10.0,     # [Rescaled] 100 -> 10
            'arrival_bonus_alpha': 0.08,    # [Rescaled] 0.8 -> 0.08
            'edge_penalty_closed': 10.0,    # [Rescaled] 100 -> 10 (Hard Constraint)
            'edge_penalty_danger': 6.0,     # [Modified] 50 -> 6 (Strong Avoidance: > 50% of Arrival)
            'edge_penalty_caution': 0.2,    # [Rescaled] 2.0 -> 0.2
            'edge_penalty_normal': 0.0,     
            'exp_penalty_factor': 0.01,     # [Rescaled] 0.1 -> 0.01
            'exp_penalty_base': 1.1,        
            'loop_penalty_base': 0.05,      # [Rescaled] 0.5 -> 0.05
            'wait_penalty': 0.002,          # [Rescaled] 0.02 -> 0.002
            'fail_penalty': 1.0             # [Rescaled] 5.0 -> 1.0 (Battery Depletion)
        }
        if reward_config:
            default_config.update(reward_config)
        self.reward_config = default_config
        
        # 2. 정적 데이터 로드
        self.num_nodes = len(self.map_core.graph.nodes())
        
        # 3. 연결성 행렬 및 최단 거리 매트릭스 (APSP) 미리 계산
        self.adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool).to(self.device)
        
        # Node mapping
        self.node_mapping = {node: i for i, node in enumerate(self.map_core.graph.nodes())}
        self.idx_to_node = {v: k for k, v in self.node_mapping.items()}
        
        for u, v in self.map_core.graph.edges():
            u_idx, v_idx = self.node_mapping[u], self.node_mapping[v]
            self.adj_matrix[u_idx, v_idx] = True
            self.adj_matrix[v_idx, u_idx] = True 
            
        # [New] All-Pairs Shortest Path (APSP) for Curriculum
        # Use caching to avoid recalculating on large maps like Golden Coast
        import os
        map_name = os.path.basename(node_file).replace('_node.tntp', '').replace('.tntp', '')
        cache_dir = os.path.join(os.path.dirname(node_file), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{map_name}_apsp.pt")

        if os.path.exists(cache_path):
            if self.verbose: print(f"🌍 Loaded Pre-calculated Network Shortest Paths (APSP) from cache ({cache_path})...")
            cache_data = torch.load(cache_path, map_location=self.device, weights_only=True)
            self.apsp_matrix = cache_data['apsp_matrix']
            self.hop_matrix = cache_data['hop_matrix']
            self.hop_next_hop_matrix = cache_data['hop_next_hop_matrix']
            self.weighted_next_hop_matrix = cache_data['weighted_next_hop_matrix']
        else:
            if self.verbose: print(f"🌍 Pre-calculating Network Shortest Paths (APSP) for map: {map_name}...")
            # Calc length based shortest path
            length_apsp = dict(nx.all_pairs_dijkstra_path_length(self.map_core.graph, weight='length'))
            self.apsp_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float, device=self.device)
            
            for u_id, lengths in length_apsp.items():
                u_idx = self.node_mapping[u_id]
                for v_id, dist_val in lengths.items():
                    v_idx = self.node_mapping[v_id]
                    self.apsp_matrix[u_idx, v_idx] = dist_val

            # Manager의 후보 마스킹은 "3~10 hop" 기준으로 동작하므로
            # 길이 기반 APSP와 별도로 unweighted hop-distance 행렬을 유지한다.
            hop_apsp = dict(nx.all_pairs_shortest_path_length(self.map_core.graph))
            self.hop_matrix = torch.full((self.num_nodes, self.num_nodes), float('inf'), dtype=torch.float, device=self.device)
            for u_id, lengths in hop_apsp.items():
                u_idx = self.node_mapping[u_id]
                for v_id, hop_val in lengths.items():
                    v_idx = self.node_mapping[v_id]
                    self.hop_matrix[u_idx, v_idx] = float(hop_val)

            hop_paths = dict(nx.all_pairs_shortest_path(self.map_core.graph))
            self.hop_next_hop_matrix = torch.full(
                (self.num_nodes, self.num_nodes),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for src_id, paths in hop_paths.items():
                src_idx = self.node_mapping[src_id]
                for dst_id, path in paths.items():
                    dst_idx = self.node_mapping[dst_id]
                    if len(path) >= 2:
                        self.hop_next_hop_matrix[src_idx, dst_idx] = self.node_mapping[path[1]]
                    else:
                        self.hop_next_hop_matrix[src_idx, dst_idx] = src_idx

            weighted_paths = dict(nx.all_pairs_dijkstra_path(self.map_core.graph, weight='weight'))
            self.weighted_next_hop_matrix = torch.full(
                (self.num_nodes, self.num_nodes),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for src_id, paths in weighted_paths.items():
                src_idx = self.node_mapping[src_id]
                for dst_id, path in paths.items():
                    dst_idx = self.node_mapping[dst_id]
                    if len(path) >= 2:
                        self.weighted_next_hop_matrix[src_idx, dst_idx] = self.node_mapping[path[1]]
                    else:
                        self.weighted_next_hop_matrix[src_idx, dst_idx] = src_idx
            
            # Save to cache
            torch.save({
                'apsp_matrix': self.apsp_matrix,
                'hop_matrix': self.hop_matrix,
                'hop_next_hop_matrix': self.hop_next_hop_matrix,
                'weighted_next_hop_matrix': self.weighted_next_hop_matrix
            }, cache_path)
            if self.verbose: print(f"💾 Saved APSP matrices to cache: {cache_path}")
        
        self.max_dist = self.apsp_matrix.max().item()
        if self.verbose: print(f"✅ APSP Matrix Ready. Max Network Distance: {self.max_dist:.2f} km")

        # 위치 정보 텐서화 (NumNodes, 2)
        self.pos_tensor = torch.zeros((self.num_nodes, 2), dtype=torch.float, device=self.device)
        for i in range(self.num_nodes):
            node_id = self.idx_to_node[i]
            x = float(self.map_core.graph.nodes[node_id].get('x', 0))
            y = float(self.map_core.graph.nodes[node_id].get('y', 0))
            self.pos_tensor[i, 0] = x
            self.pos_tensor[i, 1] = y

        # Base Data 미리 계산 (Reset시 복사해서 사용)
        self.base_data = self.converter.networkx_to_pyg(0, 0)
        self.base_x = self.base_data.x[:, :2].to(self.device) # (N, 2)
        self.base_edge_index = self.base_data.edge_index.to(self.device)
        self.base_edge_attr = self.base_data.edge_attr.to(self.device)
        
        # Physics Matrices (Lazy Init) - Batch Size Dependent
        self.edge_time_matrix = None
        self.edge_energy_matrix = None
        self.edge_cost_matrix = None

        # Curriculum State
        self.curriculum_ratio = 0.0 # 0.0 (Easy/Full Battery) -> 1.0 (Hard/Tight Battery)

        # 커리큘럼 초기화
        self._init_physics_engine()

    def reconstruct_weighted_shortest_path_indices(self, start_idx: int, goal_idx: int) -> List[int]:
        if start_idx < 0 or goal_idx < 0 or start_idx >= self.num_nodes or goal_idx >= self.num_nodes:
            return []
        if start_idx == goal_idx:
            return [int(start_idx)]

        path = [int(start_idx)]
        current = int(start_idx)
        visited = {current}

        for _ in range(self.num_nodes):
            next_idx = int(self.weighted_next_hop_matrix[current, int(goal_idx)].item())
            if next_idx < 0:
                return []
            path.append(next_idx)
            if next_idx == int(goal_idx):
                return path
            if next_idx in visited:
                return []
            visited.add(next_idx)
            current = next_idx

        return []

    def reconstruct_hop_shortest_path_indices(self, start_idx: int, goal_idx: int) -> List[int]:
        if start_idx < 0 or goal_idx < 0 or start_idx >= self.num_nodes or goal_idx >= self.num_nodes:
            return []
        if start_idx == goal_idx:
            return [int(start_idx)]

        path = [int(start_idx)]
        current = int(start_idx)
        visited = {current}

        for _ in range(self.num_nodes):
            next_idx = int(self.hop_next_hop_matrix[current, int(goal_idx)].item())
            if next_idx < 0:
                return []
            path.append(next_idx)
            if next_idx == int(goal_idx):
                return path
            if next_idx in visited:
                return []
            visited.add(next_idx)
            current = next_idx

        return []
        
    def set_curriculum_ratio(self, ratio: float):
        """Called by Trainer to update difficulty"""
        self.curriculum_ratio = max(0.0, min(1.0, ratio))
    

    def reset(self, batch_size=1, sync_problem=False):
        """
        Hybrid Reset with RoboCue-X Physics & Curriculum
        sync_problem: If True, all batch items share the SAME Start/Goal (For POMO).
        """
        self.batch_size = batch_size
        
        # [Physics Constants]
        BASE_SPEED = 40.0 # km/h
        # ... (Constants kept same)
        MAX_BATTERY_J = 10.0 * 3.6e6 # 36 MJ
        
        if batch_size == 1:
             # (Keep existing batch=1 logic exactly as is, just wrapped)
             # ... 
             pass  # Actually I should allow replace_file_content to keep the batch=1 block if I match carefully.
             # The tool works by replacing lines. I need to be careful not to delete the batch=1 block.
             # But I am replacing the method signature line 100.
             # I need to re-write the batch=1 block or just the start.
             
             # Wait, replace_file_content replaces a chunk.
             # I should target lines 100 to 110 first to change signature.
             pass

        # Let's apply change to Signature first.
        # Actually I can replace the Start/Target generation block (Lines 290-316) based on sync_problem.
        
        # STEP 1: Update Signature
        # STEP 2: Update S/G Logic

        
        if batch_size == 1:
            # [Seismic Sequence Generation]
            # Unified "Shock" Model (No distinction between Main/After)
            self.seismic_schedule = {} 
            
            if self.enable_disaster:
                num_events = random.randint(3, 5)
                # t=0 is mandatory, others random
                future_steps = sorted(random.sample(range(5, 100), num_events - 1))
                all_steps = [0] + future_steps
                
                # Intensity Mixture Strategy
                intensity_pool = []
                # Ensure at least 1 Strong event for challenge
                intensity_pool.append((7.0, 8.5)) # Stronger
                for _ in range(num_events - 1):
                    if random.random() < 0.5:
                        intensity_pool.append((6.0, 7.5)) # Medium
                    else:
                        intensity_pool.append((5.5, 6.5)) # Weak
                
                random.shuffle(intensity_pool)
                
                for t, (i_min, i_max) in zip(all_steps, intensity_pool):
                     self.seismic_schedule[t] = {
                        "type": "shock",
                        "intensity_min": i_min,
                        "intensity_max": i_max,
                        "num_centers": random.randint(2, 4) if i_max > 6.0 else random.randint(1, 3)
                    }
            
            # Trigger First Shock (t=0) via Schedule
            if self.enable_disaster and 0 in self.seismic_schedule:
                params = self.seismic_schedule[0]
                pga = self._generate_ground_motion(1, 
                                                   intensity_min=params['intensity_min'], 
                                                   intensity_max=params['intensity_max'],
                                                   num_centers=params['num_centers'])
                self.pga = pga
                
                initial_states = torch.zeros((1, self.num_physical_edges), dtype=torch.long, device=self.device)
                if not hasattr(self, 'damage_states_theoretical'):
                    self.damage_states_theoretical = torch.zeros((1, self.num_physical_edges), dtype=torch.long, device=self.device)
                self.damage_states_theoretical.zero_()
                self.damage_states = self._apply_fragility(initial_states, pga)
                
                # [Cascade]
                self.damage_states = self._apply_cascading_damage(self.damage_states)
            else:
                self.pga = None
                self.damage_states = torch.zeros((1, self.num_physical_edges), dtype=torch.long, device=self.device)
            
            self.step_count = 0
            
            # 2. Map Core Update (Legacy Sync)
            # We map damage_states back to self.map_core for compatibility if needed, 
            # but mainly we need self.pyg_data to be updated via _update_edge_attributes.
            
            # --- Connectivity Check (CPU) ---
            # To do connectivity check, we need to know which edges are closed.
            # Convert damage_states to CPU list
            states_cpu = self.damage_states[0].cpu().numpy()
            
            # Physical Edge to (u, v) mapping?
            # self.map_core.graph.edges() is source of truth.
            # self.num_physical_edges corresponds to these edges in order.
            
            edges_list = list(self.map_core.graph.edges(data=True))
            valid_edges = []
            
            for i, (u, v, d) in enumerate(edges_list):
                 state = states_cpu[i]
                 if state == 2: # Collapsed
                     d['status'] = 'Closed'
                 elif state == 1: # Damaged
                     d['status'] = 'Danger'
                 else:
                     d['status'] = 'Normal'
                 
                 if d['status'] != 'Closed':
                     valid_edges.append((u, v))
            
            G_valid = nx.Graph()
            G_valid.nodes(data=True)
            G_valid.add_nodes_from(self.map_core.graph.nodes())
            G_valid.add_edges_from(valid_edges)
            
            comps = list(nx.connected_components(G_valid))
            valid_comps = [list(c) for c in comps if len(c) >= 2]
            if not valid_comps: valid_comps = [list(self.map_core.graph.nodes())]
            
            # Sample Start/Target
            comp = random.choice(valid_comps)
            comp_indices = [self.node_mapping[n] for n in comp]
            s, t = random.choice(comp_indices), random.choice(comp_indices)
            while s == t: t = random.choice(comp_indices)
            
            self.current_node = torch.tensor([s], dtype=torch.long, device=self.device)
            self.target_node = torch.tensor([t], dtype=torch.long, device=self.device)
            
            # --- Graph Data Sync ---
            agent_data = self.converter.networkx_to_pyg(s, t).to(self.device)
            self.pyg_data = Batch.from_data_list([agent_data])
            self.pyg_data.num_graphs = 1
            
            # --- Physics Matrix Init (Batch=1) ---
            # Must initialize BEFORE _update_edge_attributes
            self.edge_time_matrix = torch.zeros((1, self.num_nodes, self.num_nodes), device=self.device)
            self.edge_energy_matrix = torch.zeros((1, self.num_nodes, self.num_nodes), device=self.device)
            self.edge_cost_matrix = torch.zeros((1, self.num_nodes, self.num_nodes), device=self.device)
            
            # Update PyG Attributes using Physics Engine
            self._update_edge_attributes(1, self.damage_states)

            # SOC Init (Full for Test)
            self.initial_battery = torch.tensor([MAX_BATTERY_J], device=self.device)
            self.current_soc = torch.ones((1, 1), device=self.device) # Ratio 0~1
            
            # [New] Dynamic Feature Init (Visualization Mode)
            # Extracted from pyg_data produced by GraphConverter
            # x shape: [N, 6] -> x, y, robot, target, visit, net_dist
            self.visit_count = self.pyg_data.x[:, 4].unsqueeze(0) # [1, N]
            
            # Network Dist: Already normalized in converter
            self.network_dist_to_target = self.pyg_data.x[:, 5].unsqueeze(0) # [1, N]

        else:
            # [Training Mode] GPU Vectorized
            num_edges = self.base_edge_index.size(1)
            
            # [Seismic Sequence Generation]
            # Unified "Shock" Model
            self.seismic_schedule = {} 
            
            if self.enable_disaster:
                num_events = random.randint(3, 5)
                future_steps = sorted(random.sample(range(5, 100), num_events - 1))
                all_steps = [0] + future_steps
                
                intensity_pool = []
                intensity_pool.append((7.0, 8.5)) # Stronger
                for _ in range(num_events - 1):
                    if random.random() < 0.5:
                        intensity_pool.append((6.0, 7.5))
                    else:
                        intensity_pool.append((5.5, 6.5))
                
                random.shuffle(intensity_pool)
                
                for t, (i_min, i_max) in zip(all_steps, intensity_pool):
                     self.seismic_schedule[t] = {
                        "type": "shock",
                        "intensity_min": i_min,
                        "intensity_max": i_max,
                        "num_centers": random.randint(2, 4) if i_max > 6.0 else random.randint(1, 3)
                    }
            
            self.step_count = 0
            
            # [Fix] Physics Matrix Init (Batch)
            self.edge_time_matrix = torch.zeros((batch_size, self.num_nodes, self.num_nodes), device=self.device)
            self.edge_energy_matrix = torch.zeros((batch_size, self.num_nodes, self.num_nodes), device=self.device)
            self.edge_cost_matrix = torch.zeros((batch_size, self.num_nodes, self.num_nodes), device=self.device)
            
            # Logic: If sync_problem, generate 1 scenario and replicate.
            phys_batch = 1 if sync_problem else batch_size

            # Trigger First Shock (t=0) via Schedule
            if self.enable_disaster and 0 in self.seismic_schedule:
                params = self.seismic_schedule[0]
                pga = self._generate_ground_motion(phys_batch, 
                                                   intensity_min=params['intensity_min'], 
                                                   intensity_max=params['intensity_max'],
                                                   num_centers=params['num_centers'])
                self.pga = pga
                                                   
                # Apply Damage (Binary State)
                zero_states = torch.zeros((phys_batch, self.num_physical_edges), dtype=torch.long, device=self.device)
                self.damage_states = self._apply_fragility(zero_states, pga)
                
                # [Cascade]
                self.damage_states = self._apply_cascading_damage(self.damage_states)
            else:
                self.pga = None
                self.damage_states = torch.zeros((phys_batch, self.num_physical_edges), dtype=torch.long, device=self.device)
            
            # Sync Replication
            if sync_problem:
                if self.pga is not None:
                    self.pga = self.pga.repeat(batch_size, 1)
                self.damage_states = self.damage_states.repeat(batch_size, 1)
                if hasattr(self, 'last_center_nodes') and self.last_center_nodes is not None:
                    if self.last_center_nodes.size(0) == 1 and batch_size > 1:
                        self.last_center_nodes = self.last_center_nodes.repeat(batch_size, 1)
            
            # Update Edge Attributes based on States
            # self._update_edge_attributes(batch_size, self.damage_states)
            # MOVED TO AFTER pyg_data CREATION
            


            # 4. Start/Target & Curriculum Learning (POMO Optimized)
            num_problems = 1 if sync_problem else batch_size
            
            start_nodes = torch.randint(0, self.num_nodes, (num_problems,), device=self.device)
            target_nodes = torch.randint(0, self.num_nodes, (num_problems,), device=self.device)
            
            # [Curriculum]
            min_dist_limit = self.max_dist * 0.8 * self.curriculum_ratio
            max_dist_limit = min_dist_limit + (self.max_dist * 0.2)
            
            # Re-sample if out of range (Vectorized rejection sampling)
            for _ in range(10):
                dists = self.apsp_matrix[start_nodes, target_nodes]
                is_invalid = (dists < min_dist_limit) | (dists > max_dist_limit) | (start_nodes == target_nodes)
                
                if not is_invalid.any(): break
                
                count = is_invalid.sum()
                start_nodes[is_invalid] = torch.randint(0, self.num_nodes, (count,), device=self.device)
                target_nodes[is_invalid] = torch.randint(0, self.num_nodes, (count,), device=self.device)

            # If Sync, Replicate
            if sync_problem:
                start_nodes = start_nodes.repeat(batch_size)
                target_nodes = target_nodes.repeat(batch_size)
            
            self.current_node = start_nodes
            self.target_node = target_nodes
            
            # SOC 초기화 (배터리 Full 상태로 시작)
            self.initial_battery = torch.full((batch_size,), MAX_BATTERY_J, device=self.device)
            self.current_soc = torch.ones((batch_size, 1), device=self.device)

            # [New] Dynamic Feature Init
            # 1. Visit Count (Batch, NumNodes)
            self.visit_count = torch.zeros((batch_size, self.num_nodes), device=self.device)
            self.visit_count.scatter_(1, self.current_node.unsqueeze(1), 1.0) # Start node visited once check
            
            # 2. Network Distance from Target (Batch, NumNodes)
            # Extract row from APSP matrix for each target
            # self.apsp_matrix: (N, N)
            # target_nodes: (Batch)
            # We want (Batch, N) where [b, :] = apsp[:, target[b]]
            # Since APSP is symmetric (undirected), apsp[target] is fine.
            self.network_dist_to_target = self.apsp_matrix[target_nodes] # [Batch, N]
            
            # Normalize Network Distance (0~1) for stability
            # Avoid div by zero
            max_d = self.max_dist if self.max_dist > 0 else 1.0
            norm_network_dist = self.network_dist_to_target / max_d
            
            # 5. Batch Data Construction
            x_base = self.base_x.repeat(batch_size, 1) # [Batch*N, 2(Pos)]
            is_robot = torch.zeros((batch_size, self.num_nodes), device=self.device)
            is_target = torch.zeros((batch_size, self.num_nodes), device=self.device)
            is_robot.scatter_(1, self.current_node.unsqueeze(1), 1.0)
            is_target.scatter_(1, self.target_node.unsqueeze(1), 1.0)
            
            # [New Features]
            # Visit Count (Normalized roughly? just raw count is fine for RNN, or log)
            # Let's use raw count for now, maybe clip at 5?
            # Network Dist (Normalized)
            
            # [New] Direction Features
            # Vector from each node to target_node
            # target_pos: [Batch, 2]
            target_pos = self.pos_tensor[target_nodes]
            # self.pos_tensor: [N, 2]
            # We need [Batch, N, 2] - [1, N, 2]? No.
            # We need vector for each batch item.
            # target_pos_expanded: [Batch, 1, 2]
            target_pos_expanded = target_pos.unsqueeze(1)
            # node_pos_expanded: [1, N, 2]
            node_pos_expanded = self.pos_tensor.unsqueeze(0)
            
            # Direction: [Batch, N, 2]
            direction = target_pos_expanded - node_pos_expanded
            norm = direction.norm(dim=2, keepdim=True).clamp(min=1e-8)
            direction_norm = direction / norm
            
            # Flatten features
            f_robot = is_robot.view(-1, 1)
            f_target = is_target.view(-1, 1)
            f_visit = self.visit_count.view(-1, 1)
            f_dist = norm_network_dist.view(-1, 1)
            f_dir = direction_norm.view(-1, 2)
            
            f_final_target = torch.ones((batch_size * self.num_nodes, 1), device=self.device)

            # Concat: [x, y, is_cur, is_tgt, visit, dist, dir_x, dir_y, is_final_target_phase] -> 9 Channels
            x_feat = torch.cat([x_base, f_robot, f_target, f_visit, f_dist, f_dir, f_final_target], dim=1)
            
            offsets = (torch.arange(batch_size, device=self.device) * self.num_nodes).view(batch_size, 1, 1)
            batch_edge_index = (self.base_edge_index.unsqueeze(0) + offsets).view(2, -1)
            
            edge_attr = self.base_edge_attr.repeat(batch_size, 1).clone()
            
            # [Fix] Update Dynamic Features (Crucial for Agent Learning)
            # Col 0: Length (Static)
            # Col 1: Damage (Dynamic)
            # damage_val is not defined here. It should be derived from self.damage_states
            # This part of the snippet is incomplete and potentially incorrect.
            # The _update_edge_attributes function is responsible for setting edge_attr.
            # I will remove the manual edge_attr updates here as _update_edge_attributes handles it.
            # The original code had:
            # self._update_edge_attributes(batch_size, self.damage_states)
            # This function should populate pyg_data.edge_attr correctly.
            # The snippet provided for `reset` was missing the `epicenters` definition for `else` block.
            # I've added random epicenters and magnitudes for the training mode.
            
            # The snippet also had `damage_val`, `time_h`, `energy_j`, `is_complete`, `costs`
            # which are calculated inside `_update_edge_attributes`.
            # So, the manual update of `edge_attr` here is redundant and incorrect.
            # I will rely on `_update_edge_attributes` to correctly set `self.pyg_data.edge_attr`.
            
            batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)
            
            self.pyg_data = Batch(x=x_feat, edge_index=batch_edge_index, edge_attr=edge_attr, batch=batch_vec)
            self.pyg_data.num_graphs = batch_size
            
            # [Fix] Update Dynamic Attributes NOW that pyg_data exists
            self._update_edge_attributes(batch_size, self.damage_states)

        # --- Common Init ---
        self.visited = torch.zeros((self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)
        self.visited.scatter_(1, self.current_node.unsqueeze(1), True)
        self.history = [self.current_node]
        
        mask = self._get_mask(self.current_node)
        return self.pyg_data, mask

    def update_target_features(self, new_targets, is_final_target_phase=None):
        """
        Update Target-dependent features in pyg_data.x
        Called by Trainer when subgoal changes.
        Features to update:
        - Col 3: is_target (Reset old, Set new)
        - Col 5: net_dist (APSP distance to new target)
        - Col 6,7: direction (Vector to new target)
        - Col 8: is_final_target_phase (Broadcast per graph)
        """
        self.target_node = new_targets # Update internal state
        batch_indices = torch.arange(self.batch_size, device=self.device)
        # [Fix] 매니저가 EOS 토큰(N) 또는 범위 외 인덱스를 생성할 수 있으므로 안전하게 클램핑
        new_targets = new_targets.clamp(0, self.num_nodes - 1)
        
        # 1. Update is_target
        self.pyg_data.x[:, 3] = 0.0
        flat_tgt_idx = batch_indices * self.num_nodes + new_targets
        self.pyg_data.x[flat_tgt_idx, 3] = 1.0
        
        # 2. Update net_dist
        # self.apsp_matrix: [N, N]
        # We need [Batch, N] -> apsp[new_targets, :] due to symmetry
        # Flatten to [Batch*N, 1]
        new_dists = self.apsp_matrix[new_targets] # [Batch, N]
        norm_dists = (new_dists / self.max_dist).view(-1, 1)
        self.pyg_data.x[:, 5] = norm_dists.squeeze(1)
        
        # 3. Update Direction
        # target_pos: [Batch, 2]
        target_pos = self.pos_tensor[new_targets]
        target_pos_expanded = target_pos.unsqueeze(1) # [Batch, 1, 2]
        node_pos_expanded = self.pos_tensor.unsqueeze(0) # [1, N, 2]
        
        direction = target_pos_expanded - node_pos_expanded # [Batch, N, 2]
        norm = direction.norm(dim=2, keepdim=True).clamp(min=1e-8)
        direction_norm = (direction / norm).view(-1, 2)
        
        self.pyg_data.x[:, 6:8] = direction_norm

        if is_final_target_phase is None:
            is_final_target_phase = torch.ones(self.batch_size, device=self.device)
        final_target_feat = is_final_target_phase.float().unsqueeze(1).expand(-1, self.num_nodes).reshape(-1, 1)
        self.pyg_data.x[:, 8:9] = final_target_feat

    def step(self, next_node_idx):
        # 1. Update Energy (SOC)
        curr = self.current_node
        next_n = next_node_idx
        
        # Extract energy cost for this step
        # indices: [Batch]
        batch_indices = torch.arange(self.batch_size, device=self.device)
        energy_step = self.edge_energy_matrix[batch_indices, curr, next_n] # [Batch]
        
        # Energy can be 0 if self-loop (wait)
        
        # Update SOC (Ratio)
        # SOC = (Initial_J - Consumed_J) / Max_J
        # But we track current ratio. 
        # Easier: Track Cumulative Consumption? 
        # Let's track consumed energy in step?
        # Actually simplest is: self.current_soc -= (energy_step / MAX_BATTERY_J)
        
        # [Simplified] Removed Energy Calculation
        # MAX_BATTERY_J = 10.0 * 3.6e6
        # soc_drop = energy_step / MAX_BATTERY_J
        # self.current_soc = self.current_soc - soc_drop.unsqueeze(1)
        # self.current_soc = torch.clamp(self.current_soc, 0.0, 1.0)
        
        # [Aftershock Trigger]
        # Probability: 1% per step
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.step_count += 1
        
        # [Seismic Schedule Logic]

        # [Seismic Schedule Logic]
        # Check if current time step matches any scheduled event
        # Assume synchronous steps for batch (all batch items have same step_count)
        current_t = self.step_count
        
        if self.enable_disaster and current_t in self.seismic_schedule and current_t > 0:
            params = self.seismic_schedule[current_t]
            
            # [Dynamic Aftershock Logic]
            # 1. Select New Epicenters based on Previous Centers (Spatial Shift)
            # self.last_center_nodes: [Batch, NumCenters_Prev]
            
            # Pick one center from Previous Shock
            # Note: last_center_nodes updated in _generate_ground_motion
            
            num_prev_centers = self.last_center_nodes.size(1)
            rand_indices = torch.randint(0, num_prev_centers, (self.batch_size, 1), device=self.device)
            base_centers = torch.gather(self.last_center_nodes, 1, rand_indices).squeeze(1) # [Batch]
            
            # Shift to Neighbor
            new_centers_list = []
            for b in range(self.batch_size):
                u = base_centers[b].item()
                neighbors = list(self.map_core.graph.neighbors(self.idx_to_node[u]))
                if neighbors:
                    v_id = random.choice(neighbors)
                    v_idx = self.node_mapping[v_id]
                    new_centers_list.append(v_idx)
                else:
                    new_centers_list.append(u)
            
            new_centers_tensor = torch.tensor(new_centers_list, device=self.device).unsqueeze(1) # [Batch, 1]
            
            # 2. Call Physics Engine with Scheduled Params
            # center_nodes override num_centers if provided
            as_pga = self._generate_ground_motion(self.batch_size, 
                                                  intensity_min=params['intensity_min'], 
                                                  intensity_max=params['intensity_max'], 
                                                  center_nodes=new_centers_tensor)
            
            # Apply Fragility
            calc_states = self._apply_fragility(self.damage_states, as_pga)
            self.damage_states = torch.maximum(self.damage_states, calc_states)
            
            # [Cascade]
            self.damage_states = self._apply_cascading_damage(self.damage_states)
            
            # Update Edge Attributes
            self._update_edge_attributes(self.batch_size, self.damage_states)
             
        # Keep SOC at 1.0 just in case model uses it
        pass
        
        # 2. Move & Visit
        self.current_node = next_node_idx
        # Use out-of-place scatter to avoid RL in-place error
        self.visited = self.visited.scatter(1, next_node_idx.unsqueeze(1), True)
        
        # [New] Update Visit Count
        # Add +1 to visit count at current node
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.visit_count[batch_indices, next_node_idx] += 1.0
        
        # [Sync PyG Data]
        # Update pyg_data.x to reflect new state
        # x structure: [x, y, is_curr(2), is_tgt(3), visit(4), dist(5), dir_x(6), dir_y(7), is_final_target_phase(8)]
        
        # 1. Update is_current (Reset old, Set new)
        # We need to zero out previous current nodes? No, we can just rebuild 
        # or scatter 0 to prev and 1 to curr.
        # But we don't track prev_node tensors easily here without history.
        # Actually simplest is: Zero out col 2, then scatter 1.
        
        self.pyg_data.x[:, 2] = 0.0 # Clear current
        # Flattened index for scatter: batch_idx * num_nodes + node_idx
        flat_curr_idx = batch_indices * self.num_nodes + self.current_node
        self.pyg_data.x[flat_curr_idx, 2] = 1.0
        
        # 2. Update visit_count (Col 4)
        # We can just update the specific values
        self.pyg_data.x[:, 4] = self.visit_count.view(-1)
        
        if not hasattr(self, 'history'): self.history = []
        self.history.append(next_node_idx)
        
        return self._get_mask(self.current_node)
        
        return self._get_mask(self.current_node)

    def get_mask(self):
        return self._get_mask(self.current_node)

    def _get_mask(self, current_node):
        connectivity = self.adj_matrix[current_node] 
        valid_mask = connectivity & (~self.visited)
        has_valid_moves = valid_mask.any(dim=1, keepdim=True)
        final_mask = torch.where(has_valid_moves, valid_mask, connectivity)
        
        # [New] Battery Constraints
        # However, Masking based on battery is complex because we need to know if edge is traversable with remaining battery.
        # For RL, we usually let it fail and punish.
        return final_mask

    def get_reward(self, path_list, actions_mask=None):
        batch_size, seq_len = path_list.size()
        device = self.device
        
        # 1. Reconstruct Path
        curr_nodes = path_list 
        prev_nodes = torch.cat([path_list[:, 0:1], path_list[:, :-1]], dim=1)
        
        # 2. Basic Masks
        is_target = (curr_nodes == self.target_node.unsqueeze(1))
        has_arrived_mask = torch.cummax(is_target, dim=1)[0]
        
        # Active: Not yet arrived
        active_mask = (~has_arrived_mask)
        # Note: We don't have 'depletion' anymore.
        
        step_rewards = torch.zeros(batch_size, seq_len, device=device)
        
        # 3. Step Penalties (Time/Distance Cost)
        # Goal: Find Shortest Path -> Constant penalty per step
        # Value: -0.05 (tuned relative to +10 arrival)
        step_rewards[active_mask] -= 0.05
        
        # 4. Edge Specific Penalties
        # [User Request] Remove 'Closed' penalty (Assume all passable)
        # Use penalty matrix (pre-calculated in reset)
        # flat_p_cost = self.edge_cost_matrix.view(-1)[flat_indices]
        # penalty_costs = flat_p_cost.view(batch_size, seq_len)
        # step_rewards[active_mask] -= penalty_costs[active_mask]
        
        # Wait Penalty (Stationary) - Keep this to prevent camping
        is_wait = (prev_nodes == curr_nodes)
        step_rewards[active_mask & is_wait] -= self.reward_config['wait_penalty']

        # 5. Arrival Bonus
        # Only for the FIRST arrival step
        has_arrived_prev = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=device), has_arrived_mask[:, :-1]], dim=1)
        first_arrival = is_target & (~has_arrived_prev)
        
        step_rewards[first_arrival] += self.reward_config['arrival_bonus_base']
        
        # 6. Distance Shaping (Potential Field)
        # Reward = Gamma * Phi(next) - Phi(curr)
        # Phi = -Distance_to_Target * Scale
        # Reduces sparsity.
        
        curr_net_dist = self.apsp_matrix[curr_nodes, self.target_node.unsqueeze(1)] # [Batch, Seq]
        prev_net_dist = torch.cat([curr_net_dist[:, 0:1] + 1.0, curr_net_dist[:, :-1]], dim=1) # +1.0 for first step momentum
        
        # Closer = Positive Reward
        # Scale: 0.1 per km
        dist_improvement = (prev_net_dist - curr_net_dist) * 0.1
        step_rewards[active_mask] += dist_improvement[active_mask]
        
        # 7. Loop Penalty (Progressive)
        # [Vectorized Optimization]
        # Instead of looping t=0..Seq (O(Seq) kernels), use broadcasting.
        # matches[b, t1, t2] = (node[b,t1] == node[b,t2])
        # We want count for t1, summing over t2 <= t1.
        
        # 1. Expand dims: [Batch, Seq, 1] == [Batch, 1, Seq] -> [Batch, Seq, Seq]
        c_exp = curr_nodes.unsqueeze(2)
        h_exp = curr_nodes.unsqueeze(1)
        pairwise_matches = (c_exp == h_exp).float()
        
        # 2. Mask future (Lower Triangular)
        # We only care about t2 <= t1
        mask_future = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0) # [1, Seq, Seq]
        
        # 3. Sum matches
        valid_matches = pairwise_matches * mask_future
        cumulative_counts = valid_matches.sum(dim=2) # [Batch, Seq]
        
        # 4. Penalty formula: Base * (Count - 1)
        revisit_count = (cumulative_counts - 1).float()
        revisit_count = torch.clamp(revisit_count, min=0.0)
        
        step_rewards[active_mask] -= self.reward_config['loop_penalty_base'] * revisit_count[active_mask]
        
        # 8. Timeout Penalty (Last step if not arrived)
        ever_arrived = has_arrived_mask[:, -1]
        is_timeout = (~ever_arrived)
        step_rewards[is_timeout, -1] -= self.reward_config['fail_penalty']
        
        if actions_mask is not None:
             step_rewards = step_rewards * actions_mask

        return step_rewards.sum(dim=1)

    # --- Physics Engine Implementation (Vectorized) ---
    def _init_physics_engine(self):
        """
        [Physics] Pre-compute Spatial Correlation Matrix & Cholesky Factor
        """
        # print("🌋 Initializing Physics Engine (Spatial Correlation)...")
        
        # 1. Extract Edge Centers
        edge_centers = []
        edge_indices_list = [] # [New] Store (u, v) indices for self.edge_index
        
        # Pre-filter edges to match self.num_physical_edges derived from graph
        # Note: self.map_core.graph is undirected.
        for u, v in self.map_core.graph.edges():
            u_pos = np.array(self.map_core.pos[u])
            v_pos = np.array(self.map_core.pos[v])
            center = (u_pos + v_pos) / 2.0
            edge_centers.append(center)
            
            # [New] Capture Node Indices (u_idx, v_idx)
            u_idx = self.node_mapping[u]
            v_idx = self.node_mapping[v]
            edge_indices_list.append([u_idx, v_idx])
            
        edge_centers = torch.tensor(np.array(edge_centers), dtype=torch.float32, device=self.device)
        self.num_physical_edges = len(edge_centers)

        # [New] Create self.edge_index (2, NumEdges)
        # Used in _generate_ground_motion for APSP lookups
        self.edge_index = torch.tensor(edge_indices_list, dtype=torch.long, device=self.device).t().contiguous()

        
        # 2. Compute Distance Matrix (Edge vs Edge)
        diff = edge_centers.unsqueeze(1) - edge_centers.unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1)) # [NumEdges, NumEdges]
        
        # [Fix] Cache Static Edge Attributes (Capacity, Speed, IsHighway) for Fragility Logic
        # These are used in _apply_fragility before pyg_data is created in reset()
        capacities = []
        speeds = []
        is_highways = [] # [New]
        for u, v, d in self.map_core.graph.edges(data=True):
            capacities.append(d.get('capacity', 0.0))
            speeds.append(d.get('speed', 0.0))
            is_highways.append(d.get('is_highway', 0.0))
            
        self.edge_capacities = torch.tensor(capacities, dtype=torch.float, device=self.device)
        self.edge_speeds = torch.tensor(speeds, dtype=torch.float, device=self.device)
        self.edge_is_highways = torch.tensor(is_highways, dtype=torch.float, device=self.device)
        
        # 3. Correlation Matrix (Exponential Model)
        correlation_range = 2.0 
        correlation_matrix = torch.exp(-dist_matrix / correlation_range)
        
        # 4. Cholesky Decomposition
        epsilon = 1e-6
        eye = torch.eye(self.num_physical_edges, device=self.device)
        
        try:
            self.L_matrix = torch.linalg.cholesky(correlation_matrix + epsilon * eye)
        except RuntimeError:
            print("⚠️ Cholesky failed. Using Eigendecomposition fallback.")
            e, v = torch.linalg.eigh(correlation_matrix)
            e = torch.clamp(e, min=0.0)
            self.L_matrix = v @ torch.diag(torch.sqrt(e))
            
        if self.verbose: print(f"✅ Physics Engine Ready. {self.num_physical_edges} edges correlated.")
        
        # [New] 액상화 취약도 (Liquefaction Susceptibility) 초기화
        # User Request: 애너하임 (Basin) 특성 상 Moderate (3.0)으로 고정.
        # 랜덤성을 제거하여 학습 인과관계 명확화 (PGA/Magnitude -> Damage)
        self.susceptibility = torch.full((self.num_physical_edges,), 3.0, device=self.device)
        # self.susceptibility = torch.clamp(self.susceptibility, 0.0, 5.0) # Not needed for constant
        
        # print("💧 Soil Liquefaction Susceptibility Map Initialized.")
        
        # [Cascade] Init Underpass Logic
        self._init_cascading_pairs()

    def _init_cascading_pairs(self):
        """
        [Physics] 고가도로(Highway) 붕괴 시 하부 도로(Normal) 연쇄 파괴를 위한 교차 쌍 계산
        O(N_highway * N_normal) Complexity. Run once at init.
        """
        import numpy as np
        
        # 1. Edge Coordinates
        G = self.map_core.graph
        pos = self.map_core.pos
        edge_list = list(G.edges())
        
        highway_indices = []
        normal_indices = []
        
        # Pre-classify edges
        # edge_is_highways is tensor on device. Need cpu numpy.
        is_h = self.edge_is_highways.cpu().numpy()
        
        for i, (u, v) in enumerate(edge_list):
            if is_h[i] == 1.0:
                highway_indices.append(i)
            else:
                normal_indices.append(i)
                
        # 2. Geometric Intersection Check
        # Segment A-B (Highway) vs C-D (Normal)
        pairs = []
        
        def ccw(A, B, C):
            val = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            if abs(val) < 1e-9: return 0 # Parallel/Collinear
            return 1 if val > 0 else -1
            
        def on_segment(P, A, B):
            # Check if P lies on segment AB
            return min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and \
                   min(A[1], B[1]) <= P[1] <= max(A[1], B[1])
                   
        def intersect(i_h, i_n):
            # Coordinates
            uh, vh = edge_list[i_h]
            un, vn = edge_list[i_n]
            
            p1, p2 = np.array(pos[uh]), np.array(pos[vh])
            p3, p4 = np.array(pos[un]), np.array(pos[vn])
            
            # Bounding Box Check (Optimization)
            if max(p1[0], p2[0]) < min(p3[0], p4[0]) or \
               min(p1[0], p2[0]) > max(p3[0], p4[0]) or \
               max(p1[1], p2[1]) < min(p3[1], p4[1]) or \
               min(p1[1], p2[1]) > max(p3[1], p4[1]):
                return False
                
            # CCW Check
            d1 = ccw(p3, p4, p1)
            d2 = ccw(p3, p4, p2)
            d3 = ccw(p1, p2, p3)
            d4 = ccw(p1, p2, p4)
            
            # Standard Intersection
            if ((d1 * d2 < 0) and (d3 * d4 < 0)):
                return True
                
            # Special Case: Collinear/Touching?
            # User wants "Overlapping or Under". 
            # If they share a node, they strictly don't "cross" in a simple segment check usually,
            # unless we count endpoints.
            # If they share a node, does collapse propagate?
            # Case: Highway leads to Ramp. Highway collapses -> Ramp blocked? Maybe.
            # Let's be aggressive: If they touch or cross, we count it.
            # But simple intersection handles crossing.
            return False
            
        # Brute Force Check (Num Edges ~900 -> 450x450 ~ 200k ops, fast enough)
        # However, limit check to non-sharing nodes? 
        # If they share nodes, they are connected components. Connectivity handles routing.
        # We specifically want "Underpass" (Geometrically overlapping but not topologically connected or distinct edges).
        
        for h_idx in highway_indices:
            for n_idx in normal_indices:
                # Check Intersection
                if intersect(h_idx, n_idx):
                    pairs.append([h_idx, n_idx])
                    
        self.underpass_pairs = torch.tensor(pairs, dtype=torch.long, device=self.device)
        # print(f"🌉 Cascading Failure Init: Found {len(pairs)} Highway-Normal intersections.")

    def _apply_cascading_damage(self, current_states):
        """
        [Physics] Apply Cascading Failure
        If Highway (Bridge) is Damaged (1), Underpass (Road) -> Damaged (1) with High Prob.
        [User Request] Sync damage or one level lower (but binary state -> same level).
        """
        if not hasattr(self, 'underpass_pairs') or len(self.underpass_pairs) == 0:
            return current_states
            
        new_states = current_states.clone()
        
        # Extract Pairs
        highway_indices = self.underpass_pairs[:, 0] # [NumPairs]
        normal_indices = self.underpass_pairs[:, 1]  # [NumPairs]
        
        # For each batch
        for b in range(current_states.shape[0]):
            h_s = new_states[b, highway_indices] # [NumPairs]
            
            # If Highway is Damaged (1)
            damaged_mask = (h_s == 1)
            
            if damaged_mask.any():
                target_normals = normal_indices[damaged_mask]
                
                # Probabilistic Sync (0.8)
                # 80% chance to also damage the underpass
                probs = torch.rand(target_normals.shape, device=self.device)
                sync_mask = (probs < 0.8)
                
                final_targets = target_normals[sync_mask]
                
                if len(final_targets) > 0:
                    new_states[b, final_targets] = torch.maximum(new_states[b, final_targets], torch.tensor(1, device=self.device))
                
        return new_states

    def _generate_ground_motion(self, batch_size, intensity_min=None, intensity_max=None, center_nodes=None, num_centers=None):
        """
        [New] Stochastic Network-Based Intensity Model
        Generates Intensity Score based on Graph Distance (not Euclidean).
        
        Args:
            batch_size (int): Batch size
            intensity_min (float, optional): Min intensity score. Defaults to 7.0 (Main Shock).
            intensity_max (float, optional): Max intensity score. Defaults to 8.5 (Main Shock).
            center_nodes (torch.Tensor, optional): Pre-defined epicenter nodes [Batch, NumCenters].
            num_centers (int, optional): Number of epicenters to pick if center_nodes is None. 
                                         If None, randomly picks 2~4.
        """
        # Defaults
        if intensity_min is None: intensity_min = 7.0
        if intensity_max is None: intensity_max = 8.5
            
        # 1. Pick Epicenter Nodes
        num_nodes = self.num_nodes
        
        if center_nodes is None:
            # Determine Number of Centers (Random 2~4 if not specified)
            if num_centers is None:
                # Random integer between 2 and 4 (inclusive)
                n_centers = random.randint(2, 4)
            else:
                n_centers = num_centers
                
            # Randomly select epicenter nodes for each batch
            # [Batch, NumEpicenters]
            epoch_centers = torch.randint(0, num_nodes, (batch_size, n_centers), device=self.device)
            # Save for Aftershock shift logic
            self.last_center_nodes = epoch_centers
        else:
            epoch_centers = center_nodes
            n_centers = center_nodes.size(1)

        # 2. Get Network Distance for all edges
        # We need distance from Edge(u, v) to Epicenter(e).
        # Distance = min(dist(u, e), dist(v, e)) + edge_weight/2 ? 
        # Approx: min(dist(u, e)) for all u in edge.
        
        # self.apsp_matrix is [NumNodes, NumNodes]
        # We want [Batch, NumEdges]
        # Extract edge endpoints
        # edge_index: [2, NumEdges]
        u_nodes = self.edge_index[0] # [NumEdges]
        v_nodes = self.edge_index[1] # [NumEdges]
        
        # We need dists from all u_nodes to the chosen epicenter_nodes
        # This is expensive to do naive gather if O(N^2).
        # But APSP is already computed.
        # We want dists: [Batch, NumEdges] = min_k( APSP[u, epicenters[b, k]] )
        
        dists_list = []
        for b in range(batch_size):
            sources = epoch_centers[b] # [NumEpicenters]
            # Get rows for sources from APSP: [NumEpicenters, NumNodes]
            
            source_dists = self.apsp_matrix[sources, :] # [k, N]
            
            # Map to edges
            d_u = source_dists[:, u_nodes] # [k, E]
            d_v = source_dists[:, v_nodes] # [k, E]
            
            # Edge dist = Average of endpoint dists (or min)
            d_edge = (d_u + d_v) * 0.5
            
            # Min over all epicenters
            min_dist, _ = torch.min(d_edge, dim=0) # [E]
            dists_list.append(min_dist)
            
        network_dists = torch.stack(dists_list, dim=0) # [Batch, E]
        
        # Convert Unit? APSP is likely in meters or weights.
        network_dists_km = network_dists / 1000.0
        
        # 3. Calculate Score (Stochastic Intensity)
        # Base Intensity: Uniform Distribution
        base_intensity = torch.empty(batch_size, 1, device=self.device).uniform_(intensity_min, intensity_max)
        
        # Decay: Network distance decay
        # decay = 2.0 * log(dist + 1)
        decay_factor = 2.5 * torch.log10(network_dists_km + 1.0)
        
        # Calculate base score before adding noise
        base_score = base_intensity - decay_factor
        
        # Noise: Spatial Correlation + Strong Randomness
        z = torch.randn(batch_size, self.num_physical_edges, device=self.device)
        spatial_epsilon = torch.matmul(z, self.L_matrix.t())
        spatial_epsilon = torch.clamp(spatial_epsilon, min=-2.0, max=2.0) # Stronger noise
        
        # Final Score
        pga = base_score + (0.5 * spatial_epsilon)
        pga = torch.clamp(pga, min=0.0) # Score cannot be negative
        
        # Store for Visualization Access
        self.last_score = pga.detach()
        
        return pga

    def _apply_fragility(self, current_states, pga):
        """
        [Physics] Structural Damage Logic
        - Calculates 'Structural Score' (Effective Intensity) based on Durability.
        - Updates self.last_score for Visualization consistency.
        - Determines Damage based on HAZUS 'Extensive' threshold (4.0).
        """
        batch_size = pga.shape[0]
        capacities = self.edge_capacities.unsqueeze(0).expand(batch_size, -1)
        
        # 1. Structural Resistance (Bonuses)
        # - Bridge: Vulnerable (Resistance 0.0) -> Score = Ground Motion
        # - Road: Strong (Resistance 2.0) -> Score = Ground Motion - 2.0
        #   (e.g., Ground 6.0 -> Score 4.0 (Extensive) -> Damaged)
        
        resistance = torch.zeros_like(pga)
        mask_bridge = (capacities > 8000)
        mask_road = (~mask_bridge)
        
        resistance[mask_bridge] = 0.0
        resistance[mask_road] = 2.0
        
        # 2. Effective Structural Score
        # Matches the 'HAZUS Scale' logged/visualized
        structural_score = pga - resistance
        structural_score = torch.clamp(structural_score, min=0.0)
        
        # [CRITICAL] Update Visualization Source
        self.last_score = structural_score.detach()
        
        # 3. Damage Determination
        # Criterion: HAZUS 'Extensive' (Score > 4.0) = Damaged
        base_threshold = 4.0
        
        # 3a. Actual Damage (With Noise)
        noise = torch.randn_like(structural_score) * 0.4
        # Logic: Score > Threshold + Noise  <=>  Score - Noise > Threshold
        is_damaged = (structural_score > (base_threshold + noise))
        
        new_states = current_states.clone()
        new_states[is_damaged] = 1
        new_states = torch.maximum(current_states, new_states)
        
        # 3b. Theoretical Damage (No Noise)
        # Strictly Score > 4.0
        if hasattr(self, 'damage_states_theoretical'):
            is_damaged_theo = (structural_score > base_threshold)
            
            current_theo = self.damage_states_theoretical
            new_theo = current_theo.clone()
            new_theo[is_damaged_theo] = 1
            self.damage_states_theoretical = torch.maximum(current_theo, new_theo)
        
        return new_states

        # ---------------------------------------------------
        # Case B: Surface Roads (Probabilistic - Durable)
        # ---------------------------------------------------
        return new_states

    def _update_edge_attributes(self, batch_size, states):
        """
        [Stochastic] Map Discrete States to physical edge attributes.
        Binary State:
        - 0: Normal -> Speed 1.0, Cost 0.0
        - 1: Damaged -> Speed 0.0 (Blocked), Cost 1.0 (High Penalty)
        """
        states_expanded = states.repeat_interleave(2, dim=1)
        states_flat = states_expanded.view(-1)
        
        # Speed Factors
        speed_fac = torch.ones_like(states_flat, dtype=torch.float)
        speed_fac[states_flat == 0] = 1.0
        speed_fac[states_flat == 1] = 0.0 # Blocked
        
        # Costs
        costs = torch.zeros_like(states_flat, dtype=torch.float)
        costs[states_flat == 0] = 0.0
        costs[states_flat == 1] = 1.0 # High Penalty
        
        lengths = self.pyg_data.edge_attr[:, 0]
        base_speed = 40.0
        
        real_speed = torch.clamp(base_speed * speed_fac, min=1e-6)
        time_h = lengths / real_speed
        time_h[speed_fac <= 1e-6] = 100.0 # Infinity
        
        self.pyg_data.edge_attr[:, 2] = time_h
        
        energy_j = lengths * (50.0 * 1.0) * 3600.0
        energy_j[speed_fac <= 1e-6] = 100.0
        self.pyg_data.edge_attr[:, 3] = energy_j
        
        # Feature 4: Blocked? (Binary)
        self.pyg_data.edge_attr[:, 4] = (states_flat == 1).float()
        
        self.pyg_data.edge_attr[:, 6] = costs
        
        num_pyg_edges = self.base_edge_index.size(1)
        
        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1).expand(-1, num_pyg_edges).flatten()
        src_nodes = self.base_edge_index[0].unsqueeze(0).expand(batch_size, -1).flatten()
        dst_nodes = self.base_edge_index[1].unsqueeze(0).expand(batch_size, -1).flatten()
        
        flat_indices = batch_indices * (self.num_nodes ** 2) + src_nodes * self.num_nodes + dst_nodes
        
        self.edge_time_matrix.view(-1).scatter_(0, flat_indices, time_h)
        self.edge_energy_matrix.view(-1).scatter_(0, flat_indices, energy_j)
        self.edge_cost_matrix.view(-1).scatter_(0, flat_indices, costs)

    def get_final_rewards(self):
        if not hasattr(self, 'history'):
            return torch.zeros(self.batch_size, device=self.device)
        path = torch.stack(self.history, dim=1)
        return self.get_reward(path)
