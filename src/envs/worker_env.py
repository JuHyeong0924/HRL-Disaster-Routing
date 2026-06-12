import os
import json
import networkx as nx
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional

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
                 use_pbrs: bool = True,
                 use_relative_hop: bool = False,
                 use_is_visited: bool = False,
                 baseline: bool = False,
                 device: str = 'cpu'):
        self.baseline = baseline
        # 1. 원본 맵 로드
        self.dm = DisasterMap(node_file, net_file)
        self.G = self.dm.graph
        self.nodes = sorted(list(self.G.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        
        # APSP (가장 짧은 홉 거리 계산) — 캐시 파일 우선 로드
        import hashlib
        # 맵 파일 기반 캐시 키 생성 (노드/엣지 변경 시 자동 무효화)
        cache_key = hashlib.md5(f"{node_file}_{net_file}_{self.num_nodes}".encode()).hexdigest()[:8]
        cache_path = f"data/hop_matrix_{cache_key}.npy"
        
        if os.path.exists(cache_path):
            # 캐시 파일에서 즉시 로드 (수 ms)
            self.hop_matrix = np.load(cache_path)
        else:
            # 최초 1회: BFS로 계산 후 캐시 저장
            self.apsp = dict(nx.all_pairs_shortest_path_length(self.G))
            self.hop_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)
            for u, lengths in self.apsp.items():
                u_idx = self.node_to_idx[u]
                for v, length in lengths.items():
                    v_idx = self.node_to_idx[v]
                    self.hop_matrix[u_idx, v_idx] = length
            np.save(cache_path, self.hop_matrix)
                
        # 최대 홉 거리(Graph Diameter) 계산 및 저장 (정규화에 사용)
        valid_hops = self.hop_matrix[self.hop_matrix < np.inf]
        self.max_hop = float(np.max(valid_hops)) if len(valid_hops) > 0 else 25.0
                
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
            
        # Manager를 위한 Zone Graph (NetworkX 객체)
        self.ZG = nx.Graph()
        for z in range(self.k):
            self.ZG.add_node(z)
        for z_str, neighbors in self.zone_graph_data['zone_adjacency'].items():
            z = int(z_str)
            for neighbor in neighbors:
                self.ZG.add_edge(z, neighbor, weight=1.0)
        
        # 노드별 Zone 인덱스 텐서 (정적, GPU 전송용)
        self._node_zone_tensor = torch.tensor(
            [self.n2z[self.idx_to_node[i]] for i in range(self.num_nodes)],
            dtype=torch.long,
        )
        
        # 인접 리스트 사전 계산 (idx 기반)
        self._adj_list = [[] for _ in range(self.num_nodes)]
        for u, v in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            self._adj_list[ui].append(vi)
            self._adj_list[vi].append(ui)
                
        # 보상 설정
        self.GOAL_REWARD = 50.0
        self.STEP_PENALTY = -0.1
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
        self.use_pbrs = use_pbrs
        self.use_relative_hop = use_relative_hop
        self.use_is_visited = use_is_visited

        self.device = torch.device(device)
        self.subgoal_mode = subgoal_mode  # 'zone' or 'node'
        
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
            state: [B, N, 4] 텐서 (is_curr, is_tgt, is_next_zone, hop_dist)
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
        """배치 Worker 입력 상태 구성 [B, N, D]."""
        B = self.batch_size
        N = self.num_nodes
        state_dim = 5 if self.use_is_visited else 4
        state = torch.zeros(B, N, state_dim)
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor  # [N]
        
        for b in range(B):
            # is_curr
            state[b, self.curr_nodes[b], 0] = 1.0
            # is_tgt
            state[b, self.target_nodes[b], 1] = 1.0
            
            if not self.baseline:
                if self.subgoal_mode == 'zone':
                    # is_next_zone: 해당 배치의 다음 목표 Zone에 속한 노드들
                    state[b, :, 2] = (nz_tensor == next_z[b]).float()
                elif self.subgoal_mode == 'node':
                    # is_next_target: 서브골 Node 1개만 활성화
                    state[b, int(self.subgoal_nodes[b].item()), 2] = 1.0
                
            # hop_dist
            tgt_idx = int(self.target_nodes[b].item())
            hops = torch.from_numpy(self.hop_matrix[:, tgt_idx].copy()).float()
            
            if self.use_relative_hop:
                curr_idx = int(self.curr_nodes[b].item())
                curr_dist = float(self.hop_matrix[curr_idx, tgt_idx])
                # 상대적 홉 기울기: (현재 위치에서의 거리 - 임의 노드에서의 거리)
                # 정규화: 범위 클리핑 (-5 to 5) 후 / 5.0
                rel_hops = (curr_dist - hops)
                rel_hops = torch.clamp(rel_hops, -5.0, 5.0) / 5.0
                state[b, :, 3] = rel_hops
            else:
                hops = torch.clamp(hops, max=100.0) / max(self.max_hop, 1.0)
                state[b, :, 3] = hops
                
            # is_visited 채널 (use_is_visited 활성화 시)
            if self.use_is_visited:
                state[b, :, 4] = self.visited_nodes[b]

            
        return state
    
    def get_action_mask_batch(self) -> torch.Tensor:
        """배치별 Action Masking [B, N].
        
        masking_mode에 따라 허용 범위가 달라짐:
        - hard: {현재 Zone, 다음 Zone} 이웃만 허용
        - hard_full_seq: {Zone Sequence 전체} 이웃만 허용
        - soft_curr_next / soft_flex: 모든 인접 노드 허용 (물리적 제약 없음)
        """
        B = self.batch_size
        N = self.num_nodes
        mask = torch.zeros(B, N)
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor
        
        for b in range(B):
            if self.dones[b]:
                mask[b, int(self.curr_nodes[b].item())] = 1.0
                continue
            
            curr_idx = int(self.curr_nodes[b].item())
            
            if self.masking_mode in ('soft_curr_next', 'soft_flex'):
                # Soft 모드: 물리적 인접 노드 전부 허용 (페널티로 유도)
                for neighbor_idx in self._adj_list[curr_idx]:
                    mask[b, neighbor_idx] = 1.0
            elif self.masking_mode == 'hard_full_seq':
                # Hard Full Seq: Zone Sequence 전체에 속한 이웃만 허용
                allowed = set(self.zone_sequences[b])
                for neighbor_idx in self._adj_list[curr_idx]:
                    if int(nz_tensor[neighbor_idx].item()) in allowed:
                        mask[b, neighbor_idx] = 1.0
            else:
                # Hard (기본): 현재/다음 Zone 이웃만 허용
                allowed = {int(curr_z[b].item()), int(next_z[b].item())}
                for neighbor_idx in self._adj_list[curr_idx]:
                    if int(nz_tensor[neighbor_idx].item()) in allowed:
                        mask[b, neighbor_idx] = 1.0
            
            # 갈 곳이 없으면 제자리 허용 (Stagnation 방지)
            if mask[b].sum() == 0:
                mask[b, curr_idx] = 1.0
                
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
        
        # [PBRS] 이동 전 hop_dist 기록 (use_pbrs=True 시에만 사용)
        prev_hop_dists = None
        if self.use_pbrs:
            prev_hop_dists = torch.zeros(B)
            for b in range(B):
                if not self.dones[b]:
                    ci = int(self.curr_nodes[b].item())
                    if self.subgoal_mode == 'node':
                        ti = int(self.subgoal_nodes[b].item())
                    else:
                        ti = int(self.target_nodes[b].item())
                    prev_hop_dists[b] = float(self.hop_matrix[ci, ti])
        
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
            if action_idx == int(self.target_nodes[b].item()):
                rewards[b] = self.GOAL_REWARD
                self.dones[b] = True
                infos[b] = {'reason': 'success', 'path_len': int(self.steps_count[b].item())}
            elif int(self.steps_count[b].item()) >= self.MAX_STEPS:
                rewards[b] += self.STEP_PENALTY
                self.dones[b] = True
                infos[b] = {'reason': 'max_steps', 'path_len': int(self.steps_count[b].item())}
            else:
                rewards[b] += self.STEP_PENALTY
                # [v3] OOB 페널티 추가 (soft 모드에서 Zone 이탈 시)
                if is_oob:
                    rewards[b] += self.oob_penalty
        
        # [v3 PBRS] 이동 후 hop_dist 차이 기반 Dense Reward 추가
        if self.use_pbrs and prev_hop_dists is not None:
            for b in range(B):
                # 이미 종료(success/invalid/stagnation)된 에피소드는 PBRS 적용 안 함
                if infos[b].get('reason') in ('success', 'invalid', 'stagnation'):
                    continue
                ci = int(self.curr_nodes[b].item())
                if self.subgoal_mode == 'node':
                    ti = int(self.subgoal_nodes[b].item())
                else:
                    ti = int(self.target_nodes[b].item())
                new_hop = float(self.hop_matrix[ci, ti])
                # PBRS: Φ(s) = -hop_dist → 가까워지면 양수, 멀어지면 음수
                pbrs = (prev_hop_dists[b].item() - new_hop) * 0.5  # 스케일 계수 0.5
                rewards[b] += pbrs
                
        return self._get_state_batch(), rewards, self.dones.clone(), infos
