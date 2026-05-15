"""
HRL Closed-Loop 환경: Manager-Worker 상호작용 래퍼

Manager가 서브골 1개를 선택 → Worker가 서브골을 향해 이동 → 도착/타임아웃 시 Manager 재호출
이 과정을 반복하여 최종 목적지에 도달하는 Closed-loop Re-planning 환경.

핵심 설계:
- Manager는 매 턴마다 K-hop 반경 내에서 서브골 1개를 선택
- Worker는 서브골 방향으로 c_max 스텝 내에서 이동
- PBRS 보상: Φ(end_node) - Φ(start_node) - step_penalty × 소요_스텝
- 에피소드 종료: 최종 목적지 도달 OR max_manager_turns 초과
"""
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.envs.disaster_map import DisasterMap


class HRLClosedLoopEnv:
    """Manager-Worker Closed-Loop 상호작용 환경.

    Manager는 K-hop 반경 내에서 서브골 1개를 선택하고,
    Worker가 서브골을 향해 이동한 후 PBRS 보상을 받는다.

    Args:
        node_file: 노드 파일 경로
        net_file: 네트워크 파일 경로
        worker: 학습된 Worker 모델 (동결 상태)
        k_hop: Manager 서브골 선택 반경
        c_max: Worker 최대 허용 스텝 (타임아웃)
        max_manager_turns: 에피소드당 최대 Manager 호출 수
        goal_bonus: 최종 목적지 도착 보상
        step_penalty_scale: 스텝당 패널티 계수
        device: 연산 디바이스
    """

    def __init__(
        self,
        node_file: str,
        net_file: str,
        worker: torch.nn.Module,
        k_hop: int = 5,
        c_max: int = 8,
        max_manager_turns: int = 50,
        goal_bonus: float = 10.0,
        step_penalty_scale: float = 0.1,
        device: str = 'cpu',
    ) -> None:
        # 맵 로드
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
        self.k_hop = k_hop
        self.c_max = c_max
        self.max_manager_turns = max_manager_turns
        self.goal_bonus = goal_bonus
        self.step_penalty_scale = step_penalty_scale

        # APSP 홉 행렬 (캐싱)
        import os
        import hashlib
        cache_key = hashlib.md5(
            f"{node_file}_{net_file}_{self.num_nodes}".encode()
        ).hexdigest()[:8]
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
        edge_attr_list = []
        for u, v, data in self.G.edges(data=True):
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            cap = data.get('capacity', 0.0)
            length = data.get('length', 0.0)
            speed = data.get('speed', 0.0)
            feat = [length, cap, speed]
            edge_list.append((ui, vi))
            edge_attr_list.append(feat)
            edge_list.append((vi, ui))
            edge_attr_list.append(feat)

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        self.edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(self.device)

        # Edge attr 정규화
        if self.edge_attr.size(0) > 0:
            feat_min = self.edge_attr.min(dim=0, keepdim=True)[0]
            feat_max = self.edge_attr.max(dim=0, keepdim=True)[0]
            scale = (feat_max - feat_min).clamp(min=1e-8)
            self.edge_attr = (self.edge_attr - feat_min) / scale

        # 인접 리스트 (Worker 이동용)
        self._adj_list: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for u, v in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            self._adj_list[ui].append(vi)
            self._adj_list[vi].append(ui)

        # Degree 텐서 (Manager State용, 정규화)
        deg_dict = dict(self.G.degree())
        max_deg = max(deg_dict.values()) if deg_dict else 1.0
        self.degree_tensor = torch.zeros(self.num_nodes, dtype=torch.float)
        for i in range(self.num_nodes):
            node_id = self.idx_to_node[i]
            self.degree_tensor[i] = float(deg_dict.get(node_id, 0)) / max(max_deg, 1.0)

        # 에피소드 상태
        self.current_idx: int = 0
        self.goal_idx: int = 0
        self.manager_turns: int = 0
        self.total_worker_steps: int = 0
        self.done: bool = False

    def reset(self) -> Tuple[int, int]:
        """에피소드 초기화. 랜덤 출발지-목적지 쌍 생성.

        Returns:
            (current_idx, goal_idx) 튜플
        """
        while True:
            s = random.choice(self.nodes)
            t = random.choice(self.nodes)
            if s != t:
                s_idx = self.node_to_idx[s]
                t_idx = self.node_to_idx[t]
                # 도달 가능한 쌍인지 확인
                if self.hop_matrix[s_idx, t_idx] < np.inf:
                    break

        self.current_idx = s_idx
        self.goal_idx = t_idx
        self.manager_turns = 0
        self.total_worker_steps = 0
        self.done = False
        return self.current_idx, self.goal_idx

    def get_manager_state(self) -> torch.Tensor:
        """Manager 입력 State 생성 (S7: is_curr, is_tgt, hop_dist, degree).

        Returns:
            x: [N, 4] 노드 피처 텐서
        """
        N = self.num_nodes
        x = torch.zeros(N, 4, device=self.device)

        # 채널 0: is_curr (현재 위치)
        x[self.current_idx, 0] = 1.0
        # 채널 1: is_tgt (최종 목적지)
        x[self.goal_idx, 1] = 1.0
        # 채널 2: hop_dist (목적지까지 정규화 홉 거리)
        hops = torch.from_numpy(
            self.hop_matrix[:, self.goal_idx].copy()
        ).float().to(self.device)
        hops = torch.clamp(hops, max=100.0) / max(self.max_hop, 1.0)
        x[:, 2] = hops
        # 채널 3: degree (정규화된 노드 차수)
        x[:, 3] = self.degree_tensor.to(self.device)

        return x

    def get_candidate_mask(self) -> torch.Tensor:
        """K-hop 반경 내 후보 서브골 마스크 생성.

        현재 위치에서 정확히 1~K hop 거리에 있는 노드들만 후보로 선택.
        (0-hop = 현재 위치 자신은 제외)

        Returns:
            mask: [N] 바이너리 마스크 (1=후보, 0=비후보)
        """
        mask = torch.zeros(self.num_nodes, device=self.device)
        hops = self.hop_matrix[self.current_idx, :]
        for i in range(self.num_nodes):
            if 1 <= hops[i] <= self.k_hop:
                mask[i] = 1.0

        # 최종 목적지가 K-hop 이내이면 직접 선택 가능
        if hops[self.goal_idx] <= self.k_hop:
            mask[self.goal_idx] = 1.0

        return mask

    def _get_worker_state(self, subgoal_idx: int) -> torch.Tensor:
        """Worker 입력 State 생성 (is_curr, is_tgt, is_subgoal, hop_dist).

        Args:
            subgoal_idx: Manager가 지정한 서브골 노드 인덱스

        Returns:
            x: [N, 4] 노드 피처 텐서
        """
        N = self.num_nodes
        x = torch.zeros(N, 4, device=self.device)

        x[self.current_idx, 0] = 1.0  # is_curr
        x[self.goal_idx, 1] = 1.0     # is_tgt
        x[subgoal_idx, 2] = 1.0       # is_subgoal (서브골 노드 1개)

        # hop_dist: 서브골까지의 거리 (Worker는 서브골을 향해 이동)
        hops = torch.from_numpy(
            self.hop_matrix[:, subgoal_idx].copy()
        ).float().to(self.device)
        hops = torch.clamp(hops, max=100.0) / max(self.max_hop, 1.0)
        x[:, 3] = hops

        return x

    def _get_worker_action_mask(self) -> torch.Tensor:
        """Worker의 Action Mask (물리적 인접 노드만 허용).

        Returns:
            mask: [N] 바이너리 마스크
        """
        mask = torch.zeros(self.num_nodes, device=self.device)
        for neighbor_idx in self._adj_list[self.current_idx]:
            mask[neighbor_idx] = 1.0

        # 갈 곳이 없으면 제자리 허용 (방어)
        if mask.sum() == 0:
            mask[self.current_idx] = 1.0

        return mask

    @torch.no_grad()
    def execute_worker(self, subgoal_idx: int) -> Tuple[int, int, bool]:
        """Worker가 서브골을 향해 이동. 최대 c_max 스텝.

        Args:
            subgoal_idx: Manager가 지정한 서브골 노드 인덱스

        Returns:
            end_idx: Worker가 최종적으로 멈춘 노드 인덱스
            steps_taken: 소요된 스텝 수
            reached_goal: 최종 목적지 도달 여부
        """
        steps_taken = 0

        for _ in range(self.c_max):
            # 서브골 도착 체크
            if self.current_idx == subgoal_idx:
                break

            # 최종 목적지 도착 체크
            if self.current_idx == self.goal_idx:
                return self.current_idx, steps_taken, True

            # Worker State 및 Action Mask 생성
            w_state = self._get_worker_state(subgoal_idx)
            w_mask = self._get_worker_action_mask()

            # Worker Forward
            probs, _, _ = self.worker(
                w_state, self.edge_index, batch=None,
                neighbors_mask=w_mask,
                edge_attr=self.edge_attr if self.worker.use_edge_attr else None,
            )

            # 행동 선택 (Greedy — Worker는 이미 학습 완료)
            action = probs.argmax().item()

            # 이동
            self.current_idx = action
            steps_taken += 1
            self.total_worker_steps += 1

            # 최종 목적지 도착 체크
            if self.current_idx == self.goal_idx:
                return self.current_idx, steps_taken, True

        return self.current_idx, steps_taken, (self.current_idx == self.goal_idx)

    def step(self, subgoal_idx: int) -> Tuple[float, bool, Dict]:
        """Manager 1턴 실행: 서브골 지정 → Worker 이동 → PBRS 보상 산출.

        Args:
            subgoal_idx: Manager가 선택한 서브골 노드 인덱스

        Returns:
            reward: PBRS 보상
            done: 에피소드 종료 여부
            info: 디버그 정보 딕셔너리
        """
        if self.done:
            return 0.0, True, {'reason': 'already_done'}

        self.manager_turns += 1
        start_idx = self.current_idx

        # PBRS 포텐셜: Φ(s) = -hop_dist(s, goal)
        phi_before = -float(self.hop_matrix[start_idx, self.goal_idx])

        # Worker 실행
        end_idx, steps_taken, reached_goal = self.execute_worker(subgoal_idx)

        phi_after = -float(self.hop_matrix[end_idx, self.goal_idx])

        # 보상 계산
        pbrs = phi_after - phi_before  # 목적지에 가까워지면 양수
        step_cost = -self.step_penalty_scale * steps_taken
        reward = pbrs + step_cost

        info = {
            'start_idx': start_idx,
            'subgoal_idx': subgoal_idx,
            'end_idx': end_idx,
            'steps_taken': steps_taken,
            'manager_turns': self.manager_turns,
            'total_worker_steps': self.total_worker_steps,
            'phi_before': phi_before,
            'phi_after': phi_after,
            'pbrs': pbrs,
        }

        # 종료 조건 체크
        if reached_goal:
            reward += self.goal_bonus
            self.done = True
            info['reason'] = 'success'
        elif self.manager_turns >= self.max_manager_turns:
            self.done = True
            info['reason'] = 'max_turns'
        else:
            info['reason'] = 'continue'

        return reward, self.done, info
