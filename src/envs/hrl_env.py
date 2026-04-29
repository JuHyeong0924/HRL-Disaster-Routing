import os
import json
import networkx as nx
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional

from src.envs.disaster_map import DisasterMap


class HRLZoneEnv:
    """
    Phase 1: 재난이 없는 상태에서의 HRL 길찾기 검증용 환경
    - Manager: Zone Graph에서 A* 알고리즘으로 최단 Zone 시퀀스 일괄 생성 (Dummy)
    - Worker: Action Masking + Sliding Window 방식으로 다음 Zone 타겟만 제공받음
    - POMO 배치 병렬 처리 지원: reset(batch_size=N)으로 N개 에피소드 동시 진행
    """
    def __init__(self, node_file: str, net_file: str, 
                 zone_json: str = 'data/node_to_zone_k30.json', 
                 zone_graph_json: str = 'data/zone_graph_k30.json'):
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
        
        # 배치 상태 관리 (reset에서 초기화)
        self.batch_size = 1
        self.curr_nodes = None      # [B] 현재 노드 인덱스
        self.target_nodes = None    # [B] 목적지 노드 인덱스
        self.zone_sequences = None  # List[List[int]], 길이 B
        self.seq_idxs = None        # [B] 현재 zone sequence 인덱스
        self.steps_count = None     # [B] 스텝 카운터
        self.dones = None           # [B] 종료 플래그
        
    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """배치 에피소드 초기화 및 Manager Zone 시퀀스 생성.
        
        Returns:
            state: [B, N, 4] 텐서 (is_curr, is_tgt, is_next_zone, hop_dist)
        """
        self.batch_size = batch_size
        self.curr_nodes = torch.zeros(batch_size, dtype=torch.long)
        self.target_nodes = torch.zeros(batch_size, dtype=torch.long)
        self.zone_sequences = []
        self.seq_idxs = torch.zeros(batch_size, dtype=torch.long)
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
            
            # A* 기반 Zone 시퀀스 생성
            sz = self.n2z[s]
            tz = self.n2z[t]
            try:
                zseq = nx.astar_path(self.ZG, sz, tz, weight='weight')
            except nx.NetworkXNoPath:
                zseq = [sz, tz]
            self.zone_sequences.append(zseq)
            
        return self._get_state_batch()
    
    def _get_current_and_next_zone_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치별 현재 Zone과 다음 Zone 반환. [B] 텐서 2개."""
        curr_z = torch.zeros(self.batch_size, dtype=torch.long)
        next_z = torch.zeros(self.batch_size, dtype=torch.long)
        for b in range(self.batch_size):
            idx = int(self.seq_idxs[b].item())
            seq = self.zone_sequences[b]
            curr_z[b] = seq[idx]
            if idx + 1 < len(seq):
                next_z[b] = seq[idx + 1]
            else:
                next_z[b] = seq[idx]
        return curr_z, next_z
    
    def _get_state_batch(self) -> torch.Tensor:
        """배치 Worker 입력 상태 구성 [B, N, 4]."""
        B = self.batch_size
        N = self.num_nodes
        state = torch.zeros(B, N, 4)
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor  # [N]
        
        for b in range(B):
            # is_curr
            state[b, self.curr_nodes[b], 0] = 1.0
            # is_tgt
            state[b, self.target_nodes[b], 1] = 1.0
            # is_next_zone: 해당 배치의 다음 목표 Zone에 속한 노드들
            state[b, :, 2] = (nz_tensor == next_z[b]).float()
            # hop_dist (정규화: 25.0 대신 맵의 실제 max_hop 사용)
            tgt_idx = int(self.target_nodes[b].item())
            hops = torch.from_numpy(self.hop_matrix[:, tgt_idx].copy()).float()
            hops = torch.clamp(hops, max=100.0) / max(self.max_hop, 1.0)
            state[b, :, 3] = hops
            
        return state
    
    def get_action_mask_batch(self) -> torch.Tensor:
        """배치별 Action Masking [B, N]."""
        B = self.batch_size
        N = self.num_nodes
        mask = torch.zeros(B, N)
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor
        
        for b in range(B):
            if self.dones[b]:
                # 이미 끝난 에피소드는 아무 노드나 허용 (dummy)
                mask[b, int(self.curr_nodes[b].item())] = 1.0
                continue
                
            allowed = {int(curr_z[b].item()), int(next_z[b].item())}
            curr_idx = int(self.curr_nodes[b].item())
            
            for neighbor_idx in self._adj_list[curr_idx]:
                if int(nz_tensor[neighbor_idx].item()) in allowed:
                    mask[b, neighbor_idx] = 1.0
                    
            # 갈 곳이 없으면 제자리 허용 (Stagnation 방지)
            if mask[b].sum() == 0:
                mask[b, curr_idx] = 1.0
                
        return mask
    
    def step_batch(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """배치 스텝 실행.
        
        Args:
            actions: [B] 노드 인덱스 텐서
        Returns:
            state: [B, N, 4]
            rewards: [B]
            dones: [B]
            infos: List[dict] 길이 B
        """
        B = self.batch_size
        rewards = torch.zeros(B)
        infos = [{} for _ in range(B)]
        
        curr_z, next_z = self._get_current_and_next_zone_batch()
        nz_tensor = self._node_zone_tensor
        
        for b in range(B):
            if self.dones[b]:
                continue
                
            self.steps_count[b] += 1
            action_idx = int(actions[b].item())
            action_node = self.idx_to_node[action_idx]
            curr_idx = int(self.curr_nodes[b].item())
            curr_node = self.idx_to_node[curr_idx]
            
            action_zone = int(nz_tensor[action_idx].item())
            allowed = {int(curr_z[b].item()), int(next_z[b].item())}
            
            # Invalid action 체크
            if action_zone not in allowed or action_idx not in self._adj_list[curr_idx]:
                if action_idx == curr_idx:
                    rewards[b] = self.INVALID_PENALTY
                    self.dones[b] = True
                    infos[b] = {'reason': 'stagnation', 'path_len': int(self.steps_count[b].item())}
                else:
                    rewards[b] = self.INVALID_PENALTY
                    self.dones[b] = True
                    infos[b] = {'reason': 'invalid', 'path_len': int(self.steps_count[b].item())}
                continue
            
            # 이동
            self.curr_nodes[b] = action_idx
            
            # Sliding Window 업데이트
            if action_zone == int(next_z[b].item()) and int(self.seq_idxs[b].item()) + 1 < len(self.zone_sequences[b]):
                self.seq_idxs[b] += 1
                # [P0] Zone 전환 중간 보상: 진행률에 비례하여 Dense Signal 제공
                if self.zone_progress_reward:
                    progress = float(self.seq_idxs[b].item()) / len(self.zone_sequences[b])
                    rewards[b] += 5.0 * progress
                
            # 목적지 도착 검사
            if action_idx == int(self.target_nodes[b].item()):
                rewards[b] = self.GOAL_REWARD
                self.dones[b] = True
                infos[b] = {'reason': 'success', 'path_len': int(self.steps_count[b].item())}
            elif int(self.steps_count[b].item()) >= self.MAX_STEPS:
                rewards[b] = self.STEP_PENALTY
                self.dones[b] = True
                infos[b] = {'reason': 'max_steps', 'path_len': int(self.steps_count[b].item())}
            else:
                rewards[b] = self.STEP_PENALTY
                
        return self._get_state_batch(), rewards, self.dones.clone(), infos
