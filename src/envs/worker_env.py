import os
import json
import networkx as nx
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional
from scipy.sparse.csgraph import shortest_path

from src.envs.disaster_map import DisasterMap


class WorkerEnv:
    """
    Phase 1: 재난이 없는 상태에서의 HRL 길찾기 검증용 환경
    - Manager: Zone Graph에서 A* 알고리즘으로 최단 Zone 시퀀스 일괄 생성 (Dummy)
    - Worker: Action Masking + Sliding Window 방식으로 다음 Zone 타겟만 제공받음
    - POMO 배치 병렬 처리 지원: reset(batch_size=N)으로 N개 에피소드 동시 진행
    """
    def __init__(self, 
                 node_file: str,
                 net_file: str,
                 zone_json: str = 'data/grid_Anaheim_node_to_zone.json',
                 zone_graph_json: str = 'data/grid_Anaheim_zone_graph.json',
                 c_max: int = 20,
                 subgoal_mode: str = 'zone', # 'zone' or 'node'
                 masking_mode: str = 'hard', # 'hard', 'hard_full_seq', 'soft_curr_next', 'soft_flex'
                 oob_penalty: float = -1.0,
                 disaster_prob: float = 0.0,
                 dynamic_disaster: bool = False,
                 device: str = 'cpu'):
        # 1. 원본 맵 로드
        self.dm = DisasterMap(node_file, net_file)
        self.G = self.dm.graph
        self.nodes = sorted(list(self.G.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        
        
        # Dijkstra 거리 행렬 (APSP) — 캐시 파일 우선 로드
        import hashlib
        cache_key = hashlib.md5(f"{node_file}_{net_file}_{self.num_nodes}".encode()).hexdigest()[:8]
        cache_path = f"data/dist_matrix_{cache_key}.npy"
        
        if os.path.exists(cache_path):
            self.dist_matrix = np.load(cache_path)
        else:
            self.apsp = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
            self.dist_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)
            for u, lengths in self.apsp.items():
                u_idx = self.node_to_idx[u]
                for v, length in lengths.items():
                    v_idx = self.node_to_idx[v]
                    self.dist_matrix[u_idx, v_idx] = length
            np.save(cache_path, self.dist_matrix)
                
        valid_dists = self.dist_matrix[self.dist_matrix < np.inf]
        self.max_dist = float(np.max(valid_dists)) if len(valid_dists) > 0 else 25.0
        # log1p 변환 후 최대값 (dijkstra 정규화용)
        self.max_log_dist = float(np.log1p(self.max_dist))
        
                
        # 2. Zone 데이터 로드
        with open(zone_json, 'r') as f:
            self.n2z = {int(k): int(v) for k, v in json.load(f).items()}
            
        with open(zone_graph_json, 'r') as f:
            self.zone_graph_data = json.load(f)
            
        self.k = self.zone_graph_data['k']
        
        # 역방향 매핑 (Zone -> Nodes)
        self.z2n = {z: [] for z in range(self.k)}
        for n, z in self.n2z.items():
            self.z2n[z].append(n)
            
        # 노드별 Zone 인덱스 텐서 (정적, GPU 전송용)
        self._node_zone_tensor = torch.tensor(
            [self.n2z[self.idx_to_node[i]] for i in range(self.num_nodes)],
            dtype=torch.long,
            device=device
        )
        
        # Zone Centroids 계산 (물리적 거리 가중치용)
        self.zone_centroids = torch.zeros((self.k, 2), dtype=torch.float32)
        for z in range(self.k):
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

        # Manager를 위한 Zone Graph (NetworkX 객체)
        self.ZG = nx.Graph()
        for z in range(self.k):
            self.ZG.add_node(z)
        for z_str, neighbors in self.zone_graph_data['zone_adjacency'].items():
            z = int(z_str)
            for neighbor in neighbors:
                w = 1.0  # uniform
                self.ZG.add_edge(z, neighbor, weight=w)
        
        # 인접 리스트 사전 계산 (idx 기반)
        self._adj_list = [[] for _ in range(self.num_nodes)]
        for u, v in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            self._adj_list[ui].append(vi)
            self._adj_list[vi].append(ui)
            
        # Dense Adjacency Matrix Tensor 생성 (Vectorized Masking용)
        self._adj_matrix_tensor = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool, device=device)
        for u in range(self.num_nodes):
            for v in self._adj_list[u]:
                self._adj_matrix_tensor[u, v] = True
                
        # Zone Adjacency Matrix Tensor 생성 (Manager Masking용)
        self._zone_adj_matrix_tensor = torch.zeros((self.k, self.k), dtype=torch.bool, device=device)
        for u in self.ZG.nodes():
            self._zone_adj_matrix_tensor[u, u] = True # 자기 자신도 허용
            for v in self.ZG.neighbors(u):
                self._zone_adj_matrix_tensor[u, v] = True
                
        # 보상 설정
        self.GOAL_REWARD = 50.0
        self.INVALID_PENALTY = -10.0
        self.MAX_STEPS = 200
        self.zone_progress_reward = False  # [P0] Ablation 제어 플래그
        
        # [v3 Ablation] 마스킹 모드 및 PBRS 제어
        # hard: 기존 방식 (현재/다음 Zone만, 위반 시 종료)
        # hard_full_seq: Zone 시퀀스 전체 허용, 위반 시 종료
        # soft_curr_next: 물리적 제약 없음, 현재/다음 Zone 이탈 시 OOB 페널티
        # soft_flex: 물리적 제약 없음, 전체 시퀀스 이탈 시 OOB 페널티
        self.masking_mode = masking_mode
        self.oob_penalty = oob_penalty

        self.device = torch.device(device)
        self.subgoal_mode = subgoal_mode  # 'zone' or 'node'
        
        self.disaster_prob = disaster_prob # 기본값 (재난 없음)
        self.dynamic_disaster = dynamic_disaster # 동적 재난 모드
        
        # 배치 상태 관리 (reset에서 초기화)
        self.batch_size = 1
        self.curr_nodes = None      # [B] 현재 노드 인덱스
        self.target_nodes = None    # [B] 목적지 노드 인덱스
        self.zone_sequences = None  # List[List[int]], 길이 B
        self.zone_seq_idxs = None   # [B] 현재 zone sequence 인덱스
        self.node_sequences = None
        self.node_seq_idxs = None   # [B] 현재 node sequence 인덱스
        self.steps_count = None     # [B] 스텝 카운터
        self.dones = None           # [B] 종료 플래그
        self.visited_nodes = None   # [B, N] 방문 이력
        
    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """배치 에피소드 초기화 및 Manager Zone 시퀀스 생성.
        
        Returns:
            state: [B, N, 4] 텐서 (is_curr, is_tgt, zone_info, dist)
        """
        self.batch_size = batch_size
        self.curr_nodes = torch.zeros(batch_size, dtype=torch.long)
        self.target_nodes = torch.zeros(batch_size, dtype=torch.long)
        self.zone_sequences = []
        self.node_sequences = []
        self.subgoal_nodes = torch.zeros(batch_size, dtype=torch.long)
        self.zone_seq_idxs = torch.zeros(batch_size, dtype=torch.long)
        self.node_seq_idxs = torch.zeros(batch_size, dtype=torch.long)
        self.steps_count = torch.zeros(batch_size, dtype=torch.long)
        self.dones = torch.zeros(batch_size, dtype=torch.bool)
        self.total_dist = torch.zeros(batch_size, dtype=torch.float)  # 물리적 이동 거리 누적
        
        # [Phase 1 Stage 2] 정적 재난 (에피소드 시작 시)
        if self.disaster_prob > 0 and not self.dynamic_disaster:
            self.dm.apply_disaster_damage(damage_prob=self.disaster_prob)
            self._update_zone_graph_weights()
            self._update_dist_matrix()
        elif self.dynamic_disaster:
            # 동적 모드일 경우 시작 시엔 재난 없이 시작
            self.dm.apply_disaster_damage(damage_prob=0.0)
            self._update_zone_graph_weights()
            self._update_dist_matrix()
            
        for b in range(batch_size):
            # 무작위 시종착점 선택 (서로 다른 Zone에 속하도록)
            while True:
                s = random.choice(self.nodes)
                t = random.choice(self.nodes)
                if s != t and self.n2z[s] != self.n2z[t]:
                    break
            
            self.curr_nodes[b] = self.node_to_idx[s]
            self.target_nodes[b] = self.node_to_idx[t]
            
            # A* 기반 Zone 시퀀스 생성 (Zone 모드용)
            sz = self.n2z[s]
            tz = self.n2z[t]
            try:
                zseq = nx.astar_path(self.ZG, sz, tz, weight='weight')
            except nx.NetworkXNoPath:
                zseq = [sz, tz]
            self.zone_sequences.append(zseq)
            
            # Shortest Path 기반 Node 시퀀스 생성 (Node 모드용)
            try:
                nseq = nx.shortest_path(self.G, s, t)
            except nx.NetworkXNoPath:
                nseq = [s, t]
            self.node_sequences.append(nseq)
            
            # Node 모드 초기 서브골 설정 (다음 노드, 단 1칸 이동 시 자기 자신이 되지 않도록)
            nxt_idx = 1 if len(nseq) > 1 else 0
            # 5-hop 룩어헤드로 좀 더 먼 서브골을 줄 수도 있으나 기본은 바로 다음 노드(또는 몇 칸 앞)로 설정 가능.
            # 훈련 난이도를 낮추기 위해 3칸 앞을 서브골로 줘보자.
            sub_idx = min(len(nseq) - 1, 3)
            self.subgoal_nodes[b] = self.node_to_idx[nseq[sub_idx]]
            
        self.visited_nodes = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.float32, device=self.device)
        for b in range(self.batch_size):
            self.visited_nodes[b, int(self.curr_nodes[b].item())] = 1.0
            
        return self._get_state_batch()
        
    def _update_zone_graph_weights(self):
        """재난 발생으로 인한 Edge damage를 기반으로 Zone Graph의 weight 갱신"""
        for u_z, v_z in self.ZG.edges():
            cross_damages = []
            for node_u in self.z2n[u_z]:
                for node_v in self.z2n[v_z]:
                    if self.G.has_edge(node_u, node_v):
                        cross_damages.append(self.G[node_u][node_v].get('damage', 0.0))
            avg_damage = sum(cross_damages) / max(len(cross_damages), 1)
            
            # 원래 weight에 비례하여 증가 (baseline: uniform weight 1.0)
            base_weight = 1.0
                
            self.ZG[u_z][v_z]['weight'] = base_weight * (1 + avg_damage * 10)
            
    def _update_dist_matrix(self):
        """Scipy를 사용하여 O(V^3) 플로이드 워셜을 밀리초 단위로 초고속 수행하여 dist_matrix 실시간 최신화"""
        adj = nx.to_scipy_sparse_array(self.G, weight='weight')
        self.dist_matrix = shortest_path(csgraph=adj, directed=False)
        # 주의: 신경망 정규화 스케일 안정을 위해 self.max_dist는 초기 깨끗한 맵 기준값을 유지합니다.
            
    def apply_dynamic_disaster(self):
        """[Phase 1 Stage 3] 에피소드 진행 중 동적 재난 발생 시뮬레이션"""
        self.dm.apply_disaster_damage(damage_prob=self.disaster_prob)
        self._update_zone_graph_weights()
        self._update_dist_matrix()
        
        # 기존 경로를 무효화하고 진행중인(완료되지 않은) 에피소드의 Zone Sequence 재계산
        for b in range(self.batch_size):
            if not self.dones[b]:
                curr_node_idx = int(self.curr_nodes[b].item())
                target_node_idx = int(self.target_nodes[b].item())
                sz = self.n2z[self.idx_to_node[curr_node_idx]]
                tz = self.n2z[self.idx_to_node[target_node_idx]]
                
                try:
                    zseq = nx.astar_path(self.ZG, sz, tz, weight='weight')
                except nx.NetworkXNoPath:
                    zseq = [sz, tz]
                    
                self.zone_sequences[b] = zseq
                self.zone_seq_idxs[b] = 0
                
    def apply_aftershock(self):
        """[Phase 2 HRL] 에피소드 진행 중 동적 재난(여진) 발생 시뮬레이션 (Manager 경로 덮어쓰기 방지)"""
        self.dm.apply_disaster_damage(damage_prob=self.disaster_prob)
        self._update_zone_graph_weights()
        self._update_dist_matrix()
        # 주의: HRL 구조에서는 zone_sequences를 재계산하지 않음 (Manager의 z_act를 유지하기 위함)
                
    def _get_current_and_next_zone_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치별 현재 Zone과 다음 Zone 반환. [B] 텐서 2개."""
        curr_z = torch.zeros(self.batch_size, dtype=torch.long)
        next_z = torch.zeros(self.batch_size, dtype=torch.long)
        for b in range(self.batch_size):
            idx = int(self.zone_seq_idxs[b].item())
            seq = self.zone_sequences[b]
            curr_z[b] = seq[idx]
            if idx + 1 < len(seq):
                next_z[b] = seq[idx + 1]
            else:
                next_z[b] = seq[idx]
        return curr_z, next_z
    
    def _get_state_batch(self) -> torch.Tensor:
        """배치 Worker 입력 상태 구성 [B, N, D] (Fully Vectorized)."""
        B = self.batch_size
        N = self.num_nodes
        state_dim = 5
        state = torch.zeros(B, N, state_dim, device=self.device)
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        curr_z = curr_z.to(self.device)
        next_z = next_z.to(self.device)
        nz_tensor = self._node_zone_tensor.to(self.device)
        
        batch_idx = torch.arange(B, device=self.device)
        curr_nodes = self.curr_nodes.to(self.device)
        target_nodes = self.target_nodes.to(self.device)
        
        # is_curr, is_tgt 채널 일괄 업데이트
        state[batch_idx, curr_nodes, 0] = 1.0
        state[batch_idx, target_nodes, 1] = 1.0
        
        if self.subgoal_mode == 'zone':
            # binary: 0(기타), 1(다음 Zone에 속한 노드)
            state[:, :, 2] = (nz_tensor.unsqueeze(0) == next_z.unsqueeze(1)).float()
        elif self.subgoal_mode == 'node':
            state[:, :, 2] = -1.0
            subgoal_nodes = self.subgoal_nodes.to(self.device)
            state[batch_idx, subgoal_nodes, 2] = 1.0
            
        # dist (Ch.3): Dijkstra (Normalized) 일괄 연산
        dist_tensor = torch.from_numpy(self.dist_matrix).float().to(self.device)
        
        # raw: 타겟 노드에서 모든 노드까지의 거리 [B, N]
        raw = torch.log1p(dist_tensor[:, target_nodes].T) 
        
        # curr_val: 현재 노드에서 타겟 노드까지의 거리 [B, 1]
        curr_val = torch.log1p(dist_tensor[curr_nodes, target_nodes]).unsqueeze(1)
        scale = 3.0
        rel = (curr_val - raw)
        rel = torch.clamp(rel, -scale, scale) / scale
        state[:, :, 3] = rel
                
        # is_visited 채널
        state[:, :, 4] = self.visited_nodes
            
        return state
    
    def get_action_mask_batch(self) -> torch.Tensor:
        """배치별 Action Masking [B, N] (Fully Vectorized).
        
        masking_mode에 따라 허용 범위가 달라짐:
        - hard: {현재 Zone, 다음 Zone} 이웃만 허용
        - hard_full_seq: {Zone Sequence 전체} 이웃만 허용
        - soft_curr_next / soft_flex: 모든 인접 노드 허용 (물리적 제약 없음)
        """
        B = self.batch_size
        N = self.num_nodes
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        curr_z = curr_z.to(self.device)
        next_z = next_z.to(self.device)
        nz_tensor = self._node_zone_tensor.to(self.device)
        
        curr_nodes = self.curr_nodes.to(self.device)
        
        # Dense 인접 행렬을 통해 모든 에이전트의 인접 노드 일괄 마스킹
        mask = self._adj_matrix_tensor[curr_nodes].clone().float() # [B, N]
        
        # 하드 마스킹 제약 추가
        if self.masking_mode == 'hard':
            allowed = (nz_tensor.unsqueeze(0) == curr_z.unsqueeze(1)) | (nz_tensor.unsqueeze(0) == next_z.unsqueeze(1))
            mask = mask * allowed.float()
        elif self.masking_mode == 'hard_full_seq':
            # 시퀀스 길이가 달라서 배칭이 까다로우므로 순차 처리
            for b in range(B):
                if not self.dones[b]:
                    allowed_zones = set(self.zone_sequences[b])
                    for n in range(N):
                        if int(nz_tensor[n].item()) not in allowed_zones:
                            mask[b, n] = 0.0
                            
        # 종료된 에이전트 마스킹 해제 및 stagnation 방지 (갈 곳이 없으면 제자리 허용)
        dones_gpu = self.dones.to(self.device)
        mask[dones_gpu] = 0.0
        
        # 갈 곳이 없는 경우(Stagnation) 현재 노드라도 1.0으로 만들어줌
        mask_sums = mask.sum(dim=1)
        stagnation_batches = (mask_sums == 0) & (~dones_gpu)
        if stagnation_batches.any():
            mask[stagnation_batches, curr_nodes[stagnation_batches]] = 1.0
            
        return mask
    
    def step_batch(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """배치 스텝 실행.
        
        masking_mode에 따라 Zone 위반 처리가 달라짐:
        - hard / hard_full_seq: 위반 시 즉시 종료 + INVALID_PENALTY
        - soft_curr_next / soft_flex: 위반 시 OOB_PENALTY만 부과, 계속 진행
        
        use_pbrs=True 시 hop_dist 차이 기반 PBRS 보상 추가.
        """
        B = self.batch_size
        rewards = torch.zeros(B)
        infos = [{} for _ in range(B)]
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor
        
        # [PBRS] 이동 전 거리 기록
        prev_dists = torch.zeros(B)
        for b in range(B):
            if not self.dones[b]:
                ci = int(self.curr_nodes[b].item())
                if self.subgoal_mode == 'node':
                    ti = int(self.subgoal_nodes[b].item())
                else:
                    ti = int(self.target_nodes[b].item())
                prev_dists[b] = float(np.log1p(self.dist_matrix[ci, ti]))
        
        for b in range(B):
            if self.dones[b]:
                continue
                
            self.steps_count[b] += 1
            action_idx = int(actions[b].item())
            curr_idx = int(self.curr_nodes[b].item())
            action_zone = int(nz_tensor[action_idx].item())
            
            # 물리적 인접성 검사 (모든 모드 공통)
            if action_idx not in self._adj_list[curr_idx]:
                # 물리적으로 연결되지 않은 노드 선택 → 무조건 에피소드 종료
                rewards[b] = self.INVALID_PENALTY
                self.dones[b] = True
                infos[b] = {'reason': 'invalid', 'path_len': int(self.steps_count[b].item())}
                continue
            
            # 제자리 선택 → stagnation 종료 (모든 모드 공통)
            if action_idx == curr_idx:
                rewards[b] = self.INVALID_PENALTY
                self.dones[b] = True
                infos[b] = {'reason': 'stagnation', 'path_len': int(self.steps_count[b].item())}
                continue
            
            # 재방문 패널티 추가 (무한루프 방지)
            if self.visited_nodes[b, action_idx] == 1.0:
                rewards[b] -= 5.0
            
            # Zone 위반 여부 판정 (masking_mode별 분기)
            is_oob = False
            if self.masking_mode == 'hard':
                allowed = {int(curr_z[b].item()), int(next_z[b].item())}
                if action_zone not in allowed:
                    rewards[b] = self.INVALID_PENALTY
                    self.dones[b] = True
                    infos[b] = {'reason': 'invalid', 'path_len': int(self.steps_count[b].item())}
                    continue
            elif self.masking_mode == 'hard_full_seq':
                allowed = set(self.zone_sequences[b])
                if action_zone not in allowed:
                    rewards[b] = self.INVALID_PENALTY
                    self.dones[b] = True
                    infos[b] = {'reason': 'invalid', 'path_len': int(self.steps_count[b].item())}
                    continue
            elif self.masking_mode == 'soft_curr_next':
                # Soft: 종료하지 않고 OOB 페널티만 부과
                allowed = {int(curr_z[b].item()), int(next_z[b].item())}
                if action_zone not in allowed:
                    is_oob = True
            elif self.masking_mode == 'soft_flex':
                # Soft: Zone Sequence 전체 기준으로 OOB 판정
                allowed = set(self.zone_sequences[b])
                if action_zone not in allowed:
                    is_oob = True
            
            # Update current nodes
            self.curr_nodes[b] = action_idx
            # 물리적 이동 거리 누적
            u_node = self.idx_to_node[curr_idx]
            v_node = self.idx_to_node[action_idx]
            edge_weight = self.G[u_node][v_node].get('weight', 1.0)
            self.total_dist[b] += edge_weight
            if not self.dones[b]:
                self.visited_nodes[b, action_idx] = 1.0
            
            # 1) Transition Info (다음 Zone 진입 시 - Zone 모드이거나 soft_curr_next 마스킹을 위해 항상 추적)
            if action_zone == int(next_z[b].item()) and int(self.zone_seq_idxs[b].item()) + 1 < len(self.zone_sequences[b]):
                self.zone_seq_idxs[b] += 1
                # [P0] Zone 전환 중간 보상 (Zone 모드에서만)
                if self.subgoal_mode == 'zone' and self.zone_progress_reward:
                    progress = float(self.zone_seq_idxs[b].item()) / len(self.zone_sequences[b])
                    rewards[b] += 5.0 * progress
                    
            if self.subgoal_mode == 'node':
                if action_idx == int(self.subgoal_nodes[b].item()):
                    idx = int(self.node_seq_idxs[b].item())
                    if idx + 3 < len(self.node_sequences[b]):
                        self.node_seq_idxs[b] += 3
                        sub_idx = min(len(self.node_sequences[b]) - 1, idx + 6)
                        self.subgoal_nodes[b] = self.node_to_idx[self.node_sequences[b][sub_idx]]
                        if self.zone_progress_reward:
                            rewards[b] += 2.0
                
            # 목적지 도착 검사
            time_penalty = -0.1 * edge_weight
            
            if action_idx == int(self.target_nodes[b].item()):
                rewards[b] = self.GOAL_REWARD
                self.dones[b] = True
                infos[b] = {'reason': 'success', 'path_len': int(self.steps_count[b].item()), 'total_dist': float(self.total_dist[b].item())}
            elif int(self.steps_count[b].item()) >= self.MAX_STEPS:
                rewards[b] += time_penalty
                self.dones[b] = True
                infos[b] = {'reason': 'max_steps', 'path_len': int(self.steps_count[b].item()), 'total_dist': float(self.total_dist[b].item())}
            else:
                rewards[b] += time_penalty
                # [v3] OOB 페널티 추가 (soft 모드에서 Zone 이탈 시)
                if is_oob:
                    rewards[b] += self.oob_penalty
        
        # [v3 PBRS] 이동 후 dist 차이 기반 Dense Reward 추가
        if prev_dists is not None:
            for b in range(B):
                # 이미 종료(success/invalid/stagnation)된 에피소드는 PBRS 적용 안 함
                if infos[b].get('reason') in ('success', 'invalid', 'stagnation'):
                    continue
                ci = int(self.curr_nodes[b].item())
                if self.subgoal_mode == 'node':
                    ti = int(self.subgoal_nodes[b].item())
                else:
                    ti = int(self.target_nodes[b].item())
                new_dist = float(np.log1p(self.dist_matrix[ci, ti]))
                # PBRS: Φ(s) = -dist → 가까워지면 양수, 멀어지면 음수
                scale_factor = 0.5  # log 스케일에서는 hop과 유사한 범위
                pbrs = (prev_dists[b].item() - new_dist) * scale_factor
                rewards[b] += pbrs
                
        return self._get_state_batch(), rewards, self.dones.clone(), infos
