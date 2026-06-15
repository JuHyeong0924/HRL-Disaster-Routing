"""
HRL Closed-Loop 환경: Manager-Worker 상호작용 래퍼 (Zone 기반)

Manager가 인접한 Zone 1개를 선택 → Worker가 해당 Zone을 향해 이동 → 도착/타임아웃 시 Manager 재호출
이 과정을 반복하여 최종 목적지에 도달하는 Closed-loop Re-planning 환경.
"""
import random
import json
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.envs.disaster_map import DisasterMap


class ManagerEnv:
    def __init__(
        self,
        node_file: str,
        net_file: str,
        worker: torch.nn.Module,
        zone_json: str = 'data/grid_Anaheim_node_to_zone.json',
        zone_graph_json: str = 'data/grid_Anaheim_zone_graph.json',
        c_max: int = 20,
        max_manager_turns: int = 20,
        goal_bonus: float = 10.0,
        step_penalty_scale: float = 0.1,
        device: str = 'cpu',
    ) -> None:
        self.dm = DisasterMap(node_file, net_file)
        self.G = self.dm.graph
        self.nodes = sorted(list(self.G.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        self.device = torch.device(device)

        # Worker (동결)
        self.worker = worker
        self.worker.eval()
        for p in self.worker.parameters():
            p.requires_grad_(False)

        # 환경 파라미터
        self.c_max = c_max
        self.goal_bonus = goal_bonus
        self.step_penalty_scale = step_penalty_scale

        # Zone 데이터 로드
        with open(zone_json, 'r') as f:
            n2z = json.load(f)
            
        with open(zone_graph_json, 'r') as f:
            zg_data = json.load(f)
            
        self.k_zones = zg_data['k']
        self.num_zones = self.k_zones
        self.zone_adj = {int(k): v for k, v in zg_data['zone_adjacency'].items()}
        
        # 맵 크기(Zone 개수)에 비례하여 최대 턴 수 동적 할당 (최소 20턴)
        self.max_manager_turns = max(20, self.k_zones // 2)
        
        # 노드 -> Zone 매핑 텐서
        self._node_zone_tensor = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        for i in range(self.num_nodes):
            node_id = self.idx_to_node[i]
            z = n2z.get(str(node_id), n2z.get(node_id, 0))
            self._node_zone_tensor[i] = z

        # Zone별 재난 강도(Disaster Intensity) 초기화 (추후 재난 정보 연동)
        self.zone_disaster_intensity = torch.zeros(self.k_zones, dtype=torch.float32, device=self.device)
        self.max_zone_disaster = 1.0 # 0 division 방지

        # Zone Centroids (for Direction Feature)
        self.zone_centroids = torch.zeros((self.k_zones, 2), dtype=torch.float32, device=self.device)
        for z in range(self.k_zones):
            z_mask = (self._node_zone_tensor == z)
            if z_mask.any():
                z_nodes = z_mask.nonzero(as_tuple=True)[0]
                x_sum, y_sum = 0.0, 0.0
                for idx in z_nodes:
                    node_id = self.idx_to_node[idx.item()]
                    x, y = self.dm.pos[node_id]
                    x_sum += x
                    y_sum += y
                self.zone_centroids[z, 0] = x_sum / len(z_nodes)
                self.zone_centroids[z, 1] = y_sum / len(z_nodes)

        # Zone Graph 생성 및 APSP (Manager State 용)
        self.ZG = nx.Graph()
        for i in range(self.k_zones):
            self.ZG.add_node(i)
            
        zone_edge_list = []
        for u, neighbors in self.zone_adj.items():
            for v in neighbors:
                # 중심점 간 유클리디안 거리를 엣지 가중치로 부여
                p1 = self.zone_centroids[u]
                p2 = self.zone_centroids[v]
                dist = float(torch.norm(p1 - p2))
                self.ZG.add_edge(u, v, weight=dist)
                zone_edge_list.append((u, v))
                
        self.zone_edge_index = torch.tensor(zone_edge_list, dtype=torch.long).t().to(self.device) if zone_edge_list else torch.zeros(2, 0, dtype=torch.long, device=self.device)
                
        z_apsp = dict(nx.all_pairs_dijkstra_path_length(self.ZG, weight='weight'))
        self.zone_dist_matrix = np.full((self.k_zones, self.k_zones), np.inf)
        for u, lengths in z_apsp.items():
            for v, length in lengths.items():
                self.zone_dist_matrix[u, v] = length
        self.max_zone_dist = float(self.zone_dist_matrix[self.zone_dist_matrix < np.inf].max())

        # 노드 레벨 APSP 물리적 거리 행렬 (PBRS 보상 용)
        import os
        import hashlib
        cache_key = hashlib.md5(f"{node_file}_{net_file}_{self.num_nodes}".encode()).hexdigest()[:8]
        cache_path = f"data/dist_matrix_{cache_key}.npy"

        if os.path.exists(cache_path):
            self.dist_matrix = np.load(cache_path)
        else:
            apsp = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
            self.dist_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)
            for u, lengths in apsp.items():
                u_idx = self.node_to_idx[u]
                for v, length in lengths.items():
                    v_idx = self.node_to_idx[v]
                    self.dist_matrix[u_idx, v_idx] = length
            np.save(cache_path, self.dist_matrix)

        self.max_dist = float(self.dist_matrix[self.dist_matrix < np.inf].max())

        # g(n) 계산을 위한 [num_nodes, k_zones] 캐시 테이블 사전 연산
        self.node_to_zone_dist_matrix = torch.zeros((self.num_nodes, self.k_zones), dtype=torch.float32, device=self.device)
        dist_tensor = torch.tensor(self.dist_matrix, dtype=torch.float32, device=self.device)
        for z in range(self.k_zones):
            z_mask = (self._node_zone_tensor == z)
            if z_mask.any():
                self.node_to_zone_dist_matrix[:, z] = dist_tensor[:, z_mask].min(dim=1).values
            else:
                self.node_to_zone_dist_matrix[:, z] = 50000.0

        # 엣지 인덱스 (정적, Worker용)
        edge_list = []
        for u, v in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            edge_list.append((ui, vi))
            edge_list.append((vi, ui))
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        self.edge_attr = torch.zeros((self.edge_index.size(1), 1), dtype=torch.float32).to(self.device)

        # 인접 리스트 (Worker 이동용)
        self._adj_list: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for u, v in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            self._adj_list[ui].append(vi)
            self._adj_list[vi].append(ui)

        # 에피소드 상태
        self.current_idx: int = 0
        self.goal_idx: int = 0
        self.manager_turns: int = 0
        self.total_worker_steps: int = 0
        self.done: bool = False
        self.visited_zones = set()

    def reset(self) -> Tuple[int, int]:
        while True:
            s = random.choice(self.nodes)
            t = random.choice(self.nodes)
            if s != t:
                s_idx = self.node_to_idx[s]
                t_idx = self.node_to_idx[t]
                if self.dist_matrix[s_idx, t_idx] < np.inf:
                    break

        self.current_idx = s_idx
        self.goal_idx = t_idx
        self.manager_turns = 0
        self.total_worker_steps = 0
        self.done = False
        self.visited_zones = set([int(self._node_zone_tensor[s_idx].item())])
        # Worker 노드 방문 이력 (use_is_visited 지원)
        self.worker_visited_nodes = torch.zeros(self.num_nodes, device=self.device)
        self.worker_visited_nodes[s_idx] = 1.0
        return self.current_idx, self.goal_idx

    def get_manager_state(self) -> torch.Tensor:
        """Manager 입력 State 생성 (Zone-level: [K, 7])."""
        x = torch.zeros(self.k_zones, 7, device=self.device)
        
        curr_z = int(self._node_zone_tensor[self.current_idx].item())
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())
        
        # 방향 벡터 (나침반) 계산
        curr_centroid = self.zone_centroids[curr_z]
        goal_centroid = self.zone_centroids[goal_z]
        goal_vec = goal_centroid - curr_centroid
        goal_norm = torch.norm(goal_vec) + 1e-8
        
        # 채널 0: is_curr_zone
        x[curr_z, 0] = 1.0
        # 채널 1: is_tgt_zone
        x[goal_z, 1] = 1.0
        # 채널 2: is_visited_zone
        for z in self.visited_zones:
            x[z, 2] = 1.0
        # 채널 3: zone_dist (목적지 구역에서 해당 구역까지의 최소 물리적 거리, h(n))
        node_dists_to_goal = torch.tensor(self.dist_matrix[:, self.goal_idx], dtype=torch.float32, device=self.device)
        zone_min_dists = torch.zeros(self.k_zones, device=self.device)
        for z in range(self.k_zones):
            z_mask = (self._node_zone_tensor == z)
            if z_mask.any():
                zone_min_dists[z] = node_dists_to_goal[z_mask].min()
            else:
                zone_min_dists[z] = 50000.0
        
        # 정규화 (max_dist 기준)
        x[:, 3] = torch.clamp(zone_min_dists, max=100000.0) / max(self.max_dist, 1.0)
        
        # 채널 4: distance from current node (현재 노드에서 해당 구역까지의 최소 물리적 거리, g(n))
        x[:, 4] = torch.clamp(self.node_to_zone_dist_matrix[self.current_idx, :], max=100000.0) / max(self.max_dist, 1.0)
        
        # 채널 5: zone disaster intensity (해당 구역의 재난 피해도)
        x[:, 5] = self.zone_disaster_intensity / max(self.max_zone_disaster, 1.0)
        
        # 채널 6: 방향 코사인 유사도 (목적지 방향과의 일치도)
        for z in range(self.k_zones):
            if z == curr_z:
                x[z, 6] = 0.0
            else:
                cand_vec = self.zone_centroids[z] - curr_centroid
                cand_norm = torch.norm(cand_vec) + 1e-8
                cos_sim = torch.dot(goal_vec, cand_vec) / (goal_norm * cand_norm)
                x[z, 6] = cos_sim

        return x

    def get_candidate_mask(self) -> torch.Tensor:
        """현재 Zone과 인접한 Zone만 허용."""
        mask = torch.zeros(self.k_zones, device=self.device)
        curr_z = int(self._node_zone_tensor[self.current_idx].item())
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())
        
        for neighbor_z in self.zone_adj.get(curr_z, []):
            mask[neighbor_z] = 1.0
            
        # [수정] 제자리 걸음(Self-loop Cowardice) 방지: 현재 존을 다시 선택하는 것은 논리적으로 무의미하며 무한 루프 유발
        if curr_z != goal_z:
            mask[curr_z] = 0.0
            
        if curr_z == goal_z:
            mask[curr_z] = 1.0
            
        if mask.sum() == 0:
            mask[curr_z] = 1.0
            
        return mask

    def _get_worker_state(self, subgoal_zone_idx: int) -> torch.Tensor:
        """Worker State [N, D]: is_curr, is_tgt, zone_info, dist, [is_visited]."""
        # Worker의 use_is_visited에 따라 차원 결정
        state_dim = 5 if getattr(self.worker, 'use_is_visited', False) else 4
        x = torch.zeros(self.num_nodes, state_dim, device=self.device)

        x[self.current_idx, 0] = 1.0  # is_curr
        x[self.goal_idx, 1] = 1.0     # is_tgt
        
        # zone_info (채널 2): -1(금지구역), 0(현재구역), 1(목표구역)
        curr_z = int(self._node_zone_tensor[self.current_idx].item())
        x[:, 2] = -1.0
        x[self._node_zone_tensor == curr_z, 2] = 0.0
        x[self._node_zone_tensor == subgoal_zone_idx, 2] = 1.0

        # 채널 3: dist (Worker 학습 시의 use_relative_hop 설정과 완벽하게 동기화)
        dists = torch.from_numpy(self.dist_matrix[:, self.goal_idx].copy()).float().to(self.device)
        if getattr(self.worker, 'use_relative_hop', False):
            curr_dist = float(self.dist_matrix[self.current_idx, self.goal_idx])
            rel_dists = (curr_dist - dists)
            scale = max(self.max_dist * 0.1, 1.0)
            rel_dists = torch.clamp(rel_dists, -scale, scale) / scale
            x[:, 3] = rel_dists
        else:
            abs_dists = torch.clamp(dists, max=100000.0) / max(self.max_dist, 1.0)
            x[:, 3] = abs_dists

        # 채널 4: is_visited (Worker가 방문한 노드 이력)
        if state_dim == 5:
            x[:, 4] = self.worker_visited_nodes

        return x

    def _get_worker_action_mask(self, subgoal_zone_idx: int = -1) -> torch.Tensor:
        """Worker 행동 마스크: Soft Masking (모든 이웃 노드 허용)."""
        mask = torch.zeros(self.num_nodes, device=self.device)
        for neighbor_idx in self._adj_list[self.current_idx]:
            mask[neighbor_idx] = 1.0
        if mask.sum() == 0:
            mask[self.current_idx] = 1.0
        return mask

    @torch.no_grad()
    def execute_worker(self, subgoal_zone_idx: int) -> Tuple[int, float, bool]:
        """Worker를 실행하여 서브골 존까지 이동. 결정적(argmax) 이동."""
        time_taken = 0.0
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())
        
        for _ in range(self.c_max):
            curr_z = int(self._node_zone_tensor[self.current_idx].item())
            
            # 도착 확인 (Worker가 목표 노드 도달 시 즉시 성공 반환)
            if self.current_idx == self.goal_idx:
                return self.current_idx, time_taken, True

            w_state = self._get_worker_state(subgoal_zone_idx)
            w_mask = self._get_worker_action_mask()
            
            # Worker forward: batch=None으로 단일 그래프 추론
            logits, _, _ = self.worker(
                w_state, self.edge_index, batch=None,
                neighbors_mask=w_mask,
                edge_attr=self.edge_attr,
            )

            # 🚨 확률적 이동(sample) 삭제
            # dist = torch.distributions.Categorical(logits)
            # action = dist.sample().item()
            
            # ✅ 확정적 이동(argmax)으로 직진성 보장
            action = logits.argmax().item()
            
            u_node = self.idx_to_node[self.current_idx]
            v_node = self.idx_to_node[action]
            edge_weight = self.G[u_node][v_node].get('weight', 1.0)
            time_taken += edge_weight
            
            self.current_idx = action
            self.total_worker_steps += 1
            
            new_z = int(self._node_zone_tensor[self.current_idx].item())
            self.visited_zones.add(new_z)
            self.worker_visited_nodes[self.current_idx] = 1.0

            if self.current_idx == self.goal_idx:
                return self.current_idx, time_taken, True
                
            # 워커가 새로운 구역(Zone) 경계선을 넘을 경우 즉시 이동 중단 (매니저 재개입 강제)
            if new_z != curr_z:
                break

        return self.current_idx, time_taken, (self.current_idx == self.goal_idx)

    def step(self, subgoal_zone_idx: int) -> Tuple[float, bool, Dict]:
        if self.done:
            return 0.0, True, {'reason': 'already_done'}

        curr_z = int(self._node_zone_tensor[self.current_idx].item())
        
        # [수정] 재방문(핑퐁) 시 강제 종료/마스킹은 하지 않지만, 신경망이 "재방문은 나쁘다"는 것을 
        # 학습할 수 있도록 명시적인 페널티(음수 보상)를 부여해야 함. (PBRS만으로는 가치함수 오차로 인해 학습 실패)
        revisit_penalty = 0.0
        if subgoal_zone_idx in self.visited_zones and subgoal_zone_idx != curr_z:
            revisit_penalty = -5.0

        self.manager_turns += 1
        start_idx = self.current_idx
        start_z = int(self._node_zone_tensor[start_idx].item())
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())

        # PBRS: zone-level 거리 기반 포텐셜
        # [수정] 무한 핑퐁 루프의 수학적 원인(Negative Potential Exploit) 완벽 차단!
        # Potential Phi(s)가 음수일 경우 gamma를 곱해 차이를 구하면 사이클에서 양의 보상이 나오는 버그가 발생합니다.
        max_dist_pbrs = max(self.max_zone_dist, 50000.0)
        phi_before = max_dist_pbrs - float(self.zone_dist_matrix[start_z, goal_z])

        end_idx, time_taken, reached_goal = self.execute_worker(subgoal_zone_idx)
        
        end_z = int(self._node_zone_tensor[end_idx].item())
        reached_subgoal = (end_z == subgoal_zone_idx)

        # Base step cost + Manager Turn Penalty + Revisit Penalty
        manager_turn_penalty = -0.5
        
        # [수정] 워커의 이동(steps)이 아닌 소요 물리 시간(time_taken) 기반 페널티 적용
        worker_step_penalty = -0.1 * time_taken
        
        step_cost = manager_turn_penalty + revisit_penalty + worker_step_penalty
            
        reward = step_cost
        pbrs = 0.0
        
        # PBRS (Potential Based Reward Shaping) - Node-level (Stochastic Trauma 방지 체제 하에서 수학적 정합성)
        node_dists = self.dist_matrix[:, self.goal_idx]
        start_z_mask = (self._node_zone_tensor == start_z).cpu().numpy()
        end_z_mask = (self._node_zone_tensor == end_z).cpu().numpy()
        
        dist_before = node_dists[start_z_mask].min() if start_z_mask.any() else 50000.0
        dist_after = node_dists[end_z_mask].min() if end_z_mask.any() else 50000.0
        
        # [추가] 무한대(inf)가 -inf로 폭발하는 것을 원천 차단
        dist_before = min(float(dist_before), 50000.0)
        dist_after = min(float(dist_after), 50000.0)
        
        # 양수 포텐셜 유지 (Negative Potential Exploit 차단)
        max_dist_node = max(self.max_dist, 50000.0)
        phi_before = max_dist_node - dist_before
        phi_after = max_dist_node - dist_after
        
        # 거리 스케일에 맞춰 PBRS 스케일링 필요
        scale_factor = 0.5 / max(self.max_dist * 0.1, 1.0)
        pbrs = (0.99 * phi_after - phi_before) * scale_factor
        
        if reached_subgoal:
            reward += pbrs
        else:
            reward += pbrs - 15.0

        info = {
            'start_idx': start_idx,
            'subgoal_zone': subgoal_zone_idx,
            'end_idx': end_idx,
            'time_taken': time_taken,
            'manager_turns': self.manager_turns,
            'total_worker_steps': self.total_worker_steps,
            'reached_subgoal': reached_subgoal,
            'pbrs': pbrs,
        }

        if reached_goal:
            reward += 50.0  # Ultimate Node Bonus
            self.done = True
            info['reason'] = 'success'
        elif end_z == goal_z:
            reward += 20.0  # Manager Zone Bonus
            self.done = True
            info['reason'] = 'success'
        elif self.manager_turns >= self.max_manager_turns:
            self.done = True
            info['reason'] = 'max_turns'
        else:
            info['reason'] = 'continue'

        return reward, self.done, info
