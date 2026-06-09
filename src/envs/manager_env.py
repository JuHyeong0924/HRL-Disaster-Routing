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

        # Zone Graph 생성 및 APSP (Manager State 용)
        self.ZG = nx.Graph()
        for i in range(self.k_zones):
            self.ZG.add_node(i)
            
        zone_edge_list = []
        for u, neighbors in self.zone_adj.items():
            for v in neighbors:
                self.ZG.add_edge(u, v)
                zone_edge_list.append((u, v))
                
        self.zone_edge_index = torch.tensor(zone_edge_list, dtype=torch.long).t().to(self.device) if zone_edge_list else torch.zeros(2, 0, dtype=torch.long, device=self.device)
                
        z_apsp = dict(nx.all_pairs_shortest_path_length(self.ZG))
        self.zone_hop_matrix = np.full((self.k_zones, self.k_zones), np.inf)
        for u, lengths in z_apsp.items():
            for v, length in lengths.items():
                self.zone_hop_matrix[u, v] = length
        self.max_zone_hop = float(self.zone_hop_matrix[self.zone_hop_matrix < np.inf].max())

        # 노드 레벨 APSP 홉 행렬 (PBRS 보상 용)
        import os
        import hashlib
        cache_key = hashlib.md5(f"{node_file}_{net_file}_{self.num_nodes}".encode()).hexdigest()[:8]
        cache_path = f"data/hop_matrix_{cache_key}.npy"

        if os.path.exists(cache_path):
            self.hop_matrix = np.load(cache_path)
        else:
            apsp = dict(nx.all_pairs_shortest_path_length(self.G))
            self.hop_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)
            for u, lengths in apsp.items():
                u_idx = self.node_to_idx[u]
                for v, length in lengths.items():
                    v_idx = self.node_to_idx[v]
                    self.hop_matrix[u_idx, v_idx] = length
            np.save(cache_path, self.hop_matrix)

        self.max_hop = float(self.hop_matrix[self.hop_matrix < np.inf].max())

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
                if self.hop_matrix[s_idx, t_idx] < np.inf:
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
        """Manager 입력 State 생성 (Zone-level: [K, 4])."""
        x = torch.zeros(self.k_zones, 4, device=self.device)
        
        curr_z = int(self._node_zone_tensor[self.current_idx].item())
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())
        
        # 채널 0: is_curr_zone
        x[curr_z, 0] = 1.0
        # 채널 1: is_tgt_zone
        x[goal_z, 1] = 1.0
        # 채널 2: is_visited_zone
        for z in self.visited_zones:
            x[z, 2] = 1.0
        # 채널 3: zone_hop_dist to goal_zone
        hops = torch.from_numpy(self.zone_hop_matrix[:, goal_z].copy()).float().to(self.device)
        hops = torch.clamp(hops, max=50.0) / max(self.max_zone_hop, 1.0)
        x[:, 3] = hops

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
        """Worker State [N, D]: is_curr, is_tgt, is_next_zone, relative_hop, [is_visited]."""
        # Worker의 use_is_visited에 따라 차원 결정
        state_dim = 5 if getattr(self.worker, 'use_is_visited', False) else 4
        x = torch.zeros(self.num_nodes, state_dim, device=self.device)

        x[self.current_idx, 0] = 1.0  # is_curr
        x[self.goal_idx, 1] = 1.0     # is_tgt
        mask_tz = (self._node_zone_tensor == subgoal_zone_idx)
        x[mask_tz, 2] = 1.0           # is_next_zone

        # 채널 3: Relative hop — 현재 위치 기준 상대적 거리 이점
        hops = torch.from_numpy(self.hop_matrix[:, self.goal_idx].copy()).float().to(self.device)
        curr_dist = float(self.hop_matrix[self.current_idx, self.goal_idx])
        rel_hops = (curr_dist - hops)
        rel_hops = torch.clamp(rel_hops, -5.0, 5.0) / 5.0
        x[:, 3] = rel_hops

        # 채널 4: is_visited (Worker가 방문한 노드 이력)
        if state_dim == 5:
            x[:, 4] = self.worker_visited_nodes

        return x

    def _get_worker_action_mask(self, subgoal_zone_idx: int = -1) -> torch.Tensor:
        """Worker 행동 마스크: 현재 노드의 인접 노드만 허용."""
        mask = torch.zeros(self.num_nodes, device=self.device)
        for neighbor_idx in self._adj_list[self.current_idx]:
            mask[neighbor_idx] = 1.0
        if mask.sum() == 0:
            mask[self.current_idx] = 1.0
        return mask

    @torch.no_grad()
    def execute_worker(self, subgoal_zone_idx: int) -> Tuple[int, int, bool]:
        """Worker를 실행하여 서브골 존까지 이동. 결정적(argmax) 이동."""
        steps_taken = 0
        goal_z = int(self._node_zone_tensor[self.goal_idx].item())
        
        for _ in range(self.c_max):
            curr_z = int(self._node_zone_tensor[self.current_idx].item())
            
            # 도착 확인 (Worker가 목표 노드 도달 시 즉시 성공 반환)
            if self.current_idx == self.goal_idx:
                return self.current_idx, steps_taken, True
                
            # 서브골 존에 도달했는지 확인 (단, 최종 목표 존인 경우는 목표 노드까지 가야 하므로 멈추지 않음)
            if curr_z == subgoal_zone_idx and subgoal_zone_idx != goal_z:
                break

            w_state = self._get_worker_state(subgoal_zone_idx)
            w_mask = self._get_worker_action_mask()
            
            # Worker forward: batch=None으로 단일 그래프 추론
            logits, _, _ = self.worker(
                w_state, self.edge_index, batch=None,
                neighbors_mask=w_mask,
                edge_attr=self.edge_attr if self.worker.use_edge_attr else None,
            )

            # 확률적 이동 (sample) — argmax는 동일 상태에서 무한 루프 유발
            dist = torch.distributions.Categorical(logits)
            action = dist.sample().item()
            self.current_idx = action
            steps_taken += 1
            self.total_worker_steps += 1
            
            new_z = int(self._node_zone_tensor[self.current_idx].item())
            self.visited_zones.add(new_z)
            self.worker_visited_nodes[self.current_idx] = 1.0

            if self.current_idx == self.goal_idx:
                return self.current_idx, steps_taken, True

        return self.current_idx, steps_taken, (self.current_idx == self.goal_idx)

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
        max_dist = max(self.max_zone_hop, 50.0)
        phi_before = max_dist - float(self.zone_hop_matrix[start_z, goal_z])

        end_idx, steps_taken, reached_goal = self.execute_worker(subgoal_zone_idx)
        
        end_z = int(self._node_zone_tensor[end_idx].item())
        reached_subgoal = (end_z == subgoal_zone_idx)

        # Base step cost + Manager Turn Penalty + Revisit Penalty
        manager_turn_penalty = -0.5  # 턴 패널티 정상화
        
        # [수정] Stochastic Trauma 방지: 매니저 보상의 완전한 계층적 분리(Decoupling)
        # 훈련 중 워커의 확률적 탐색으로 인해 발생하는 스텝(steps_taken)을 매니저에게 전가하면, 
        # 매니저는 "최단 경로"가 아니라 "워커가 덜 헤매는 쉬운 경로"로 우회하거나 핑퐁을 칩니다.
        # 이를 방지하기 위해 매니저는 오직 '턴 페널티'와 '재방문 페널티'만 받도록 계층간 보상을 분리합니다.
        step_cost = manager_turn_penalty + revisit_penalty
            
        reward = step_cost
        pbrs = 0.0
        
        if reached_subgoal:
            # 양수로 정의된 수학적 Potential을 기반으로 완벽한 PBRS 계산
            phi_after = max_dist - float(self.zone_hop_matrix[subgoal_zone_idx, goal_z])
            gamma = 0.99
            pbrs = (gamma * phi_after - phi_before) * 0.5
            reward += pbrs
        else:
            phi_after = max_dist - float(self.zone_hop_matrix[end_z, goal_z])
            gamma = 0.99
            pbrs = (gamma * phi_after - phi_before) * 0.5
            reward += pbrs - 1.0

        info = {
            'start_idx': start_idx,
            'subgoal_zone': subgoal_zone_idx,
            'end_idx': end_idx,
            'steps_taken': steps_taken,
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
