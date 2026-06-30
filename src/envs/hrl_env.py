import torch
import numpy as np
import networkx as nx
import random
from collections import deque

class HRLEnv:
    """
    HRL Manager 환경.
    Frozen Worker를 감싸고 있으며, Manager의 (Target, Zone) 액션을 받아
    Worker가 특정 이벤트를 발생시킬 때까지 내부적으로 스텝을 진행합니다.
    """
    def __init__(self, worker, worker_env):
        self.worker = worker
        self.worker.eval()
        if hasattr(self.worker, 'parameters'):
            for p in self.worker.parameters():
                p.requires_grad_(False)
            
        self.env = worker_env
        self.max_time = 200
        self.max_manager_turns = 50
        
        # ManagerTrainer 호환성을 위한 zone_dist_matrix 사전 계산
        self._recompute_zone_dist_matrix()
        
        # Graph caching for fast rebuilding in _build_graph_data
        self.el = [(self.env.node_to_idx[u], self.env.node_to_idx[v]) for u, v in self.env.G.edges()]
        self.bidir = self.el + [(v, u) for u, v in self.el]
        self.cached_edge_index = torch.tensor(self.bidir, dtype=torch.long).t().to(self.env.device)

        # _build_graph_data용 bidir 인덱스 텐서 캐싱 (런타임 NetworkX dict 순회 제거)
        self._bidir_src = torch.tensor([p[0] for p in self.bidir], dtype=torch.long, device=self.env.device)
        self._bidir_dst = torch.tensor([p[1] for p in self.bidir], dtype=torch.long, device=self.env.device)

        # zone_edge_index 캐싱 (Zone Graph 토폴로지는 불변)
        el_z = list(self.env.ZG.edges())
        bidir_z = el_z + [(v, u) for u, v in el_z]
        self._cached_zone_edge_index = torch.tensor(bidir_z, dtype=torch.long, device=self.env.device).t()
        
        # 상태 변수
        self.batch_size = 1
        self.num_targets = 10
        self.targets = None         # [B, N] target node indices
        self.target_zones = None    # [B, N]
        self.deadlines = None       # [B, N]
        self.target_rescued = None  # [B, N] boolean mask
        
        self.current_time = None    # [B]
        self.manager_turns = None   # [B]
        self.worker_steps = None    # [B]
        self.num_rescued = None     # [B]
        
        self.curr_target_idx = None # [B] (0 ~ N-1)
        self.curr_zone_action = None # [B] (0 ~ K-1)
        
        self.dones = None           # [B]
    
    def _recompute_zone_dist_matrix(self):
        """Zone Graph의 현재 weight 기반으로 zone_dist_matrix 재계산.
        
        호출 시점: __init__, reset() 내 재난 적용 후, aftershock 후.
        Manager의 Zone Score Network에 정확한 거리를 제공하기 위함.
        """
        z_apsp = dict(nx.all_pairs_dijkstra_path_length(self.env.ZG, weight='weight'))
        z_mat = np.full((self.env.k, self.env.k), np.inf)
        for u, lengths in z_apsp.items():
            for v, length in lengths.items():
                z_mat[u, v] = length
        self.zone_dist_matrix = torch.tensor(z_mat, dtype=torch.float32, device=self.env.device)
        
    def reset(self, batch_size=1, num_targets=10):
        """새로운 시나리오(타겟 목록) 생성"""
        self.batch_size = batch_size
        self.num_targets = num_targets
        
        # 동적 한계치 설정 (타겟 개수에 비례)
        self.max_manager_turns = num_targets * 20
        self.max_time = num_targets * 80
        
        # [NEW] 에피소드 시작 시 초기 재난 부여 (Phase 1)
        if getattr(self.env, 'disaster_prob', 0.0) > 0:
            self.env.dm.apply_disaster_damage(damage_prob=0.0)  # [BUG FIX] Reset previous cumulative damages
            self.env.dm.apply_disaster_damage(damage_prob=self.env.disaster_prob)
            self.env.sync_tensors_from_graph()        # ← 텐서 먼저 동기화 (필수)
            self.env._update_zone_graph_weights()
            self.env._update_dist_matrix()
            self._recompute_zone_dist_matrix()        # ← [BUG FIX] 재난 후 Zone 거리 갱신
        
        # 1. 출발지 무작위 설정 (WorkerEnv의 reset 이용하지 않고 수동 설정)
        self.env.batch_size = batch_size
        self.env.curr_nodes = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.env.target_nodes = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.env.visited_nodes = torch.zeros(batch_size, self.env.num_nodes, dtype=torch.float32, device=self.env.device)
        self.env.total_dist = torch.zeros(batch_size, dtype=torch.float)
        self.env.zone_sequences = [[] for _ in range(batch_size)]
        self.env.zone_seq_idxs = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.env.dones = torch.zeros(batch_size, dtype=torch.bool, device=self.env.device)
        self.env.steps_count = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        
        self.targets = torch.zeros(batch_size, num_targets, dtype=torch.long, device=self.env.device)
        self.target_zones = torch.zeros(batch_size, num_targets, dtype=torch.long, device=self.env.device)
        self.deadlines = torch.zeros(batch_size, num_targets, dtype=torch.float, device=self.env.device)
        
        self.current_time = torch.zeros(batch_size, dtype=torch.float, device=self.env.device)
        self.worker_steps = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)

        self.dones = torch.zeros(batch_size, dtype=torch.bool, device=self.env.device)

        for b in range(batch_size):
            # 연결된 노드가 충분히 많은 출발지 선택
            attempts = 0
            is_isolated = False
            while True:
                s = random.choice(self.env.nodes)
                s_idx = self.env.node_to_idx[s]
                reachable = [n for n in self.env.nodes if self.env.dist_matrix[s_idx, self.env.node_to_idx[n]] < float('inf')]
                if len(reachable) > num_targets:
                    break
                attempts += 1
                if attempts > 100:
                    print(f"[WARNING] Batch {b} is isolated! Ending episode immediately. len(reachable)={len(reachable)} <= {num_targets}")
                    is_isolated = True
                    reachable = self.env.nodes  # 구색만 맞추기 위해 전체 노드 할당
                    break
            
            self.env.curr_nodes[b] = s_idx
            self.env.visited_nodes[b, int(self.env.curr_nodes[b])] = 1.0
            if is_isolated:
                self.dones[b] = True
            
            # 타겟 N개 생성 (도달 가능한 노드 중에서만)
            selected = set([s])
            for i in range(num_targets):
                attempts = 0
                while True:
                    t = random.choice(reachable)
                    if t not in selected:
                        selected.add(t)
                        self.targets[b, i] = self.env.node_to_idx[t]
                        self.target_zones[b, i] = self.env.n2z[t]
                        
                        # Deadline (휴리스틱: 홉 수 기반 여유 시간)
                        s_idx = self.env.curr_nodes[b].item()
                        t_idx = self.targets[b, i].item()
                        
                        try:
                            weight_dist = nx.shortest_path_length(self.env.G, source=s, target=t, weight='weight')
                        except nx.NetworkXNoPath:
                            weight_dist = 50.0
                            
                        # 가중치 거리 기준 최소 거리 + 랜덤 여유 (탐색을 위해 여유 시간 대폭 증가)
                        self.deadlines[b, i] = int(min(weight_dist * 2.0 + random.uniform(50, 100), self.max_time))
                        break
                    
                    attempts += 1
                    if attempts > 100:
                        # 긴급 조치: 그냥 아무 노드나 하나 추가
                        remaining = list(set(self.env.nodes) - selected)
                        t = random.choice(remaining) if remaining else random.choice(self.env.nodes)
                        selected.add(t)
                        self.targets[b, i] = self.env.node_to_idx[t]
                        self.target_zones[b, i] = self.env.n2z[t]
                        self.deadlines[b, i] = self.max_time * random.uniform(0.5, 1.0)
                        break

        self.target_rescued = torch.zeros(batch_size, num_targets, dtype=torch.bool, device=self.env.device)
        self.target_failed = torch.zeros(batch_size, num_targets, dtype=torch.bool, device=self.env.device)
        self.current_time = torch.zeros(batch_size, dtype=torch.float, device=self.env.device)
        self.manager_turns = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.worker_steps = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.num_rescued = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        
        self.curr_target_idx = torch.full((batch_size,), -1, dtype=torch.long, device=self.env.device)
        self.curr_zone_action = torch.full((batch_size,), -1, dtype=torch.long, device=self.env.device)
        
        # [Phase 1B] 시간 기반 Continuous Aftershock 스케줄링
        # 여진은 Manager/Worker 턴과 무관하게 current_time 축에서 독립적으로 발생
        # 근거: Omori's Law — 여진은 시간 축에서 독립 발생 (물리적 타당성)
        self.global_step = 0
        if getattr(self.env, 'dynamic_disaster', False):
            num_aftershocks = random.randint(15, 25)
            total_time = self.max_time
            interval = total_time / (num_aftershocks + 1)
            self.aftershock_times = sorted([
                interval * (i + 1) + random.uniform(-interval * 0.3, interval * 0.3)
                for i in range(num_aftershocks)
            ])
        else:
            self.aftershock_times = []
        self.aftershock_cursor = 0  # 다음 발생할 여진의 인덱스
        self.ugv_destroys = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
            
        # 내부 상태 저장 (POMO에서 같은 시나리오 반복을 위해)
        self._save_scenario()
        return self._get_scenario_dict()
        
    def _save_scenario(self):
        self._saved = {
            'start_nodes': self.env.curr_nodes.clone(),
            'targets': self.targets.clone(),
            'target_zones': self.target_zones.clone(),
            'deadlines': self.deadlines.clone()
        }
        
    def restore(self):
        """저장된 시나리오로 되돌리기 (POMO K-rollout 용)"""
        b = self.batch_size
        self.env.batch_size = b
        self.env.curr_nodes = self._saved['start_nodes'].clone()
        self.env.target_nodes = torch.zeros(b, dtype=torch.long, device=self.env.device)
        self.env.visited_nodes = torch.zeros(b, self.env.num_nodes, dtype=torch.float32, device=self.env.device)
        self.env.dones = torch.zeros(b, dtype=torch.bool, device=self.env.device)
        self.env.steps_count = torch.zeros(b, dtype=torch.long, device=self.env.device)
        for i in range(b):
            self.env.visited_nodes[i, int(self.env.curr_nodes[i])] = 1.0
        self.env.total_dist = torch.zeros(b, dtype=torch.float)
        
        self.target_rescued = torch.zeros(b, self.num_targets, dtype=torch.bool, device=self.env.device)
        self.target_failed = torch.zeros(b, self.num_targets, dtype=torch.bool, device=self.env.device)
        self.current_time = torch.zeros(b, dtype=torch.float, device=self.env.device)
        self.manager_turns = torch.zeros(b, dtype=torch.long, device=self.env.device)
        self.worker_steps = torch.zeros(b, dtype=torch.long, device=self.env.device)
        self.num_rescued = torch.zeros(b, dtype=torch.long, device=self.env.device)
        
        self.curr_target_idx = torch.full((b,), -1, dtype=torch.long, device=self.env.device)
        self.curr_zone_action = torch.full((b,), -1, dtype=torch.long, device=self.env.device)
        self.dones = torch.zeros(b, dtype=torch.bool, device=self.env.device)
        
    def _get_scenario_dict(self):
        return {
            'zone_features': self.get_zone_features(),
            'zone_edge_index': self.get_zone_edge_index(),
            'target_zones': self.target_zones.clone(),
            'deadlines': self.deadlines.clone()
        }
        
    def get_zone_features(self):
        """Zone 레벨 features [B, K_zones, 7]"""
        B = self.batch_size
        K_zones = self.env.k
        feats = torch.zeros(B, K_zones, 7, device=self.env.device)
        
        # ===== Ch.3 disaster_intensity 벡터화 (배치 공유, 1회 계산) =====
        adj_float = self.env._adj_matrix_tensor.float()                          # [N, N]
        node_adj_damage_sum = (self.env._damage_matrix * adj_float).sum(dim=1)   # [N]
        node_adj_count = adj_float.sum(dim=1)                                     # [N]
        zone_one_hot = torch.nn.functional.one_hot(
            self.env._node_zone_tensor, num_classes=K_zones
        ).float()                                                                 # [N, K]
        zone_damage_sum = zone_one_hot.T @ node_adj_damage_sum                    # [K]
        zone_adj_count_sum = zone_one_hot.T @ node_adj_count                      # [K]
        zone_avg_damage = zone_damage_sum / zone_adj_count_sum.clamp(min=1)       # [K]
        
        # [NEW] Ch.6 node_damage 기반 disaster_intensity 벡터화 (Max로 변경)
        zone_max_node_dmg = torch.zeros(K_zones, device=self.env.device)
        zone_max_node_dmg.scatter_reduce_(0, self.env._node_zone_tensor.long(), self.env._node_damage_tensor, reduce='amax', include_self=False)
        zone_avg_node_dmg = zone_max_node_dmg   # 변수명은 하위 호환성을 위해 유지하되 Max 값이 들어감
        
        # ===== [NEW] 100% Pure PyTorch Tensor Vectorization =====
        # 1. Ch.0: is_current
        curr_zones = self.env._node_zone_tensor[self.env.curr_nodes]  # [B]
        feats[:, :, 0].scatter_(1, curr_zones.unsqueeze(1), 1.0)
        
        # 2. Ch.1: has_target
        ch1 = torch.zeros(B, K_zones, device=self.env.device)
        ch1.scatter_add_(1, self.target_zones, (~self.target_rescued).float())
        feats[:, :, 1] = (ch1 > 0).float()
        
        # 3. Ch.2: is_visited
        ch2 = self.env.visited_nodes @ zone_one_hot  # [B, N] @ [N, K] -> [B, K]
        feats[:, :, 2] = (ch2 > 0).float()
        
        # 4. Ch.3 & Ch.6: Broadcast Damage metrics
        feats[:, :, 3] = zone_avg_damage.unsqueeze(0).expand(B, K_zones)
        feats[:, :, 6] = zone_avg_node_dmg.unsqueeze(0).expand(B, K_zones)
        
        # 5. Ch.4: dist_from_curr
        all_centroids = self.env.zone_centroids.to(self.env.device)  # [K, 2]
        curr_centroids = all_centroids[curr_zones]  # [B, 2]
        diff = curr_centroids.unsqueeze(1) - all_centroids.unsqueeze(0)  # [B, K, 2]
        ch4 = torch.norm(diff, dim=-1)  # [B, K]
        feats[:, :, 4] = ch4 / max(self.env.max_dist, 1.0)
                
        return feats
        
    def get_zone_edge_index(self):
        return self._cached_zone_edge_index
        
    def get_target_features(self):
        """
        [B, N, 6] — 데드라인 인식 강화 Target Features (완전 벡터화)
        0: deadline (정규화)
        1: time_remaining (정규화)
        2: dist_from_curr (정규화)
        3: urgency_ratio (dist/time_remaining, 높을수록 긴급, clamp [0,5])
        4: feasibility (데드라인 내 도달 가능 여부, 0 or 1)
        5: normalized_slack (여유 시간, clamp [-1,1])
        """
        B, N = self.batch_size, self.num_targets
        feats = torch.zeros(B, N, 6, device=self.env.device)
        
        c_idxs = self.env.curr_nodes.long()  # [B]
        t_idxs = self.targets.long()          # [B, N]
        
        # GPU Tensor fancy indexing으로 거리 일괄 조회 [B, N]
        dists = self.env._dist_matrix_tensor[
            c_idxs.unsqueeze(1).expand_as(t_idxs), t_idxs
        ]
        dists = torch.where(torch.isinf(dists),
                            torch.full_like(dists, self.env.max_dist), dists)
        
        d = self.deadlines                              # [B, N]
        rem = d - self.current_time.unsqueeze(1)        # [B, N]
        
        feats[:, :, 0] = d / self.max_time
        feats[:, :, 1] = rem / self.max_time
        feats[:, :, 2] = dists / max(self.env.max_dist, 1.0)
        
        rem_safe = rem.clamp(min=1e-6)
        feats[:, :, 3] = torch.where(rem > 0,
                                      (dists / rem_safe).clamp(max=5.0),
                                      torch.full_like(dists, 5.0))
        feats[:, :, 4] = ((rem > 0) & (dists < rem)).float()
        feats[:, :, 5] = ((rem - dists) / self.max_time).clamp(-1.0, 1.0)
        
        return feats
        
    def get_target_mask(self):
        """[B, N] 유효한 타겟=1, 구조완료/시간초과=0 (벡터화)"""
        B, N = self.batch_size, self.num_targets
        mask = torch.ones(B, N, dtype=torch.long, device=self.env.device)
        mask[self.target_rescued] = 0
        expired = self.current_time.unsqueeze(1) > self.deadlines  # [B, N]
        mask[expired] = 0
        return mask
        
    def get_zone_adj_mask(self):
        """[B, K] 현재 Zone과 인접한 Zone = 1 (벡터화)"""
        c_zones = self.env._node_zone_tensor[self.env.curr_nodes.long()]  # [B]
        return self.env._zone_adj_matrix_tensor[c_zones].long()           # [B, K]
        
    def step_manager(self, target_actions, zone_actions, visualizer=None, save_dir=None, frame_idx_ref=None):
        """
        target_actions: [B] target index (0 ~ N-1)
        zone_actions: [B] zone index (0 ~ K-1)
        
        선택된 Zone으로 Worker를 진행시킵니다.
        Manager가 zone을 지정하면, Worker는 해당 zone에 도착하거나
        타겟을 구출하거나 Trap될 때까지 진행합니다.
        """
        B = self.batch_size
        self.curr_target_idx = target_actions
        self.curr_zone_action = zone_actions
        
        self.global_step += 1
        
        # [Phase 1B] 시간 기반 여진은 Worker 루프 내부에서 체크 (아래 참조)
        # 기존 턴 기반 글로벌 여진은 제거됨
                
        # Worker에게 부여할 목표 세팅
        self.env.zone_sequences = []
        self.env.zone_seq_idxs = torch.zeros(B, dtype=torch.long)
        
        for b in range(B):
            if not self.dones[b]:
                # Worker 내부 상태 초기화
                self.env.dones[b] = False
                self.env.steps_count[b] = 0
                
                # 매 턴마다 방문 이력 리셋 (단, 현재 위치는 방문 처리)
                self.env.visited_nodes[b].zero_()
                self.env.visited_nodes[b, int(self.env.curr_nodes[b].item())] = 1.0
                
                self.manager_turns[b] += 1
                t_idx = self.curr_target_idx[b].item()
                self.env.target_nodes[b] = self.targets[b, t_idx]
                
            # Worker에게 다음 가야할 zone sequence를 설정
            c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
            c_zone = self.env.n2z[c_node]
            z_act = self.curr_zone_action[b].item()
            self.env.zone_sequences.append([c_zone, z_act])
            
        # Worker Execution Loop
        events = ['none'] * B
        worker_done = self.dones.clone()
        need_rebuild = True  # 첫 반복에서 반드시 구축
        
        # 이전 Zone 추적 (aftershock 트리거용)
        prev_zones = {}
        for b in range(B):
            c = self.env.idx_to_node[int(self.env.curr_nodes[b])]
            prev_zones[b] = self.env.n2z[c]
        
        for step in range(50):  # max inner steps
            if worker_done.all():
                break
            
            # 그래프 데이터 (재)구축
            if need_rebuild:
                edge_index, edge_attr = self._build_graph_data()
                need_rebuild = False
                
            active = [b for b in range(B) if not worker_done[b]]
            A = len(active)
            N_nodes = self.env.num_nodes
            
            st = self.env._get_state_batch()
            action_masks = self.env.get_action_mask_batch()
            xs = torch.stack([st[b].to(self.env.device) for b in active])
            ms = torch.stack([action_masks[b].to(self.env.device) for b in active])
            
            xf = xs.view(-1, xs.shape[-1])
            mf = ms.view(-1)
            ai = torch.arange(A, device=self.env.device).repeat_interleave(N_nodes)
            aei = torch.cat([edge_index + i*N_nodes for i in range(A)], dim=1)
            eaf = edge_attr.repeat(A, 1) if edge_attr is not None else None
            
            if hasattr(self.worker, 'is_heuristic') and self.worker.is_heuristic:
                aa = self.worker.get_actions(active)
            else:
                with torch.no_grad():
                    probs, _, _ = self.worker(xf, aei, edge_attr=eaf, batch=ai, neighbors_mask=mf)
                    
                acts = [probs[i*N_nodes:(i+1)*N_nodes].argmax().item() for i in range(A)]
                
                aa = []
                p = 0
                for b in range(B):
                    if not worker_done[b]:
                        aa.append(acts[p])
                        p += 1
                    else:
                        aa.append(0)
                    
            prev_nodes = self.env.curr_nodes.clone()
            _, _, step_dones, infos = self.env.step_batch(torch.tensor(aa))
            
            for b in active:
                prev_idx = int(prev_nodes[b])
                new_idx = int(self.env.curr_nodes[b])
                if prev_idx == new_idx:
                    edge_w = 1.0
                else:
                    edge_w = self.env._weight_matrix[prev_idx, new_idx].item()
                    if edge_w == 0.0:  # 간선이 없는 경우 (adj=False이면 weight=0)
                        edge_w = 1.0
                    
                self.worker_steps[b] += 1
                self.current_time[b] += edge_w
                
                # [Phase 1C] UGV 파괴 판정 — HAZUS Complete (Closed) 간선 통과 시
                # UGV는 특수 구조 장비로 강행 돌파 가능하나, 30% 확률로 파괴
                if prev_idx != new_idx and self.env._adj_matrix_tensor[prev_idx, new_idx]:
                    status = self.env._status_matrix[prev_idx, new_idx].item()
                    # status: 0=Normal, 1=Caution, 2=Danger, 3=Closed
                    if status == 3 and random.random() < 0.5:
                        events[b] = 'agent_destroyed'
                        self.dones[b] = True
                        worker_done[b] = True
                        self.ugv_destroys[b] += 1
                    elif status == 2 and random.random() < 0.2:
                        # Extensive 등급: 20% 확률로 즉사 (Trap -> Destroy로 강화)
                        events[b] = 'agent_destroyed'
                        self.dones[b] = True
                        worker_done[b] = True
                        self.ugv_destroys[b] += 1
            
            # [BUG FIX] 시간 기반 여진 체크 — per-batch 루프 밖에서 max(current_time) 기준으로 통합
            # 여진은 물리적 글로벌 이벤트이므로, 가장 빠른 배치가 해당 시각을 넘기면 모든 배치에 적용
            if getattr(self.env, 'dynamic_disaster', False) and active:
                max_time = max(self.current_time[b].item() for b in active)
                while (self.aftershock_cursor < len(self.aftershock_times) and
                       max_time >= self.aftershock_times[self.aftershock_cursor]):
                    # 미세 여진 적용 (damage_prob=0.05: 5% 간선 영향)
                    affected_nodes = self.env.dm.apply_disaster_damage(damage_prob=0.05)
                    self.env.sync_tensors_from_graph()
                    self.env._update_zone_graph_weights()
                    self.env._update_dist_matrix()
                    self._recompute_zone_dist_matrix()  # ← [BUG FIX] 여진 후 Zone 거리 갱신
                    need_rebuild = True  # Worker edge_attr 갱신
                    
                    # [NEW] Dynamic Time-Window (동적 데드라인 축소)
                    if affected_nodes:
                        affected_zones = {self.env.n2z[n] for n in affected_nodes}
                        # [PERF] GPU-CPU Sync Lock 방지: target_zones 일괄 다운로드
                        tz_cpu = self.target_zones.cpu().numpy()
                        for b in active:
                            for i in range(self.num_targets):
                                if not self.target_rescued[b, i] and tz_cpu[b, i] in affected_zones:
                                    reduction = self.deadlines[b, i] * random.uniform(0.2, 0.4)
                                    self.deadlines[b, i] = max(self.current_time[b].item() + 10.0, self.deadlines[b, i] - reduction)
                                    
                        # [NEW] Aftershock Strike (변화량 기반 직격타 파괴 및 타겟 압사)
                        for b in active:
                            # 1. Agent KIA (UGV 파괴: 순간 충격량 >= 0.3)
                            c_node = int(self.env.curr_nodes[b].item())
                            if c_node in affected_nodes and affected_nodes[c_node] >= 0.3:
                                events[b] = 'agent_destroyed'
                                self.dones[b] = True
                                worker_done[b] = True
                                self.ugv_destroys[b] += 1
                                
                            # 2. Target KIA (타겟 건물의 붕괴로 인한 구조 실패 처리: 순간 충격량 >= 0.5)
                            for i in range(self.num_targets):
                                if not self.target_rescued[b, i] and not self.target_failed[b, i]:
                                    t_node = int(self.targets[b, i].item())
                                    if t_node in affected_nodes and affected_nodes[t_node] >= 0.5:
                                        self.target_failed[b, i] = True
                    
                    self.aftershock_cursor += 1
                
            if visualizer is not None and save_dir is not None and frame_idx_ref is not None:
                # Batch 0에 대해서만 시각화
                if 0 in active:
                    b_prev_n = self.env.idx_to_node[int(prev_nodes[0])]
                    b_new_n = self.env.idx_to_node[int(self.env.curr_nodes[0])]
                    
                    if not hasattr(self, 'worker_path_log'):
                        self.worker_path_log = []
                    self.worker_path_log.append(b_new_n)
                    
                    visualizer.plot_state(
                        self, 
                        step_idx=frame_idx_ref[0], 
                        save_dir=save_dir, 
                        trajectory=self.worker_path_log,
                        mission_zone=self.curr_zone_action[0].item(),
                        worker_edge=(b_prev_n, b_new_n) if b_prev_n != b_new_n else None,
                        global_time=self.current_time[0].item()
                    )
                    frame_idx_ref[0] += 1

            for b in active:
                # Check events
                c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
                c_zone = self.env.n2z[c_node]
                target_node = self.env.idx_to_node[int(self.env.target_nodes[b])]
                target_idx_in_list = self.curr_target_idx[b].item()
                
                # 1. 타겟 도착 체크
                if c_node == target_node:
                    events[b] = 'target_rescued'
                    self.target_rescued[b, target_idx_in_list] = True
                    self.num_rescued[b] += 1
                    worker_done[b] = True
                    continue
                
                # 2. Zone 전환 체크
                # Trap 판정은 이제 Phase 1C에서 통합 처리됨 (위 Worker step 직후)
                if c_zone != prev_zones[b]:
                    # Zone 도착/이탈 판정
                    if c_zone == self.curr_zone_action[b].item():
                        events[b] = 'zone_arrived'
                    else:
                        events[b] = 'zone_escaped'
                    worker_done[b] = True
                    prev_zones[b] = c_zone
                elif step_dones[b].item():
                    # OOB, Stagnation, Destroyed 등
                    reason = infos[b].get('reason', '')
                    if reason in ['destroyed', 'stagnation']:
                        events[b] = 'agent_destroyed'
                        self.dones[b] = True
                    else:
                        events[b] = 'zone_escaped'
                    worker_done[b] = True
                    
        # Update dones
        target_mask = self.get_target_mask()
        for b in range(B):
            for i in range(self.num_targets):
                if not self.target_rescued[b, i] and self.current_time[b] > self.deadlines[b, i]:
                    self.target_failed[b, i] = True
                    
            if self.target_rescued[b].sum() == self.num_targets:
                self.dones[b] = True
            if self.current_time[b] >= self.max_time:
                self.dones[b] = True
            if self.manager_turns[b] >= self.max_manager_turns:
                self.dones[b] = True
            # 데드라인 만료 등으로 선택할 타겟이 전혀 없을 경우 조기 종료 처리
            if target_mask[b].sum() == 0:
                self.dones[b] = True
                
        return events, self.dones.clone()
    
    def _build_graph_data(self):
        """현재 그래프 상태 기반 edge_index, edge_attr 구축 [length, damage] (텐서 인덱싱)."""
        length_vals = self.env._length_matrix[self._bidir_src, self._bidir_dst]  # [E_bidir]
        damage_vals = self.env._damage_matrix[self._bidir_src, self._bidir_dst]  # [E_bidir]
        edge_attr = torch.stack([length_vals, damage_vals], dim=1)                # [E_bidir, 2]
        
        # Per-channel min-max normalization
        mn = edge_attr.min(0, keepdim=True)[0]
        mx = edge_attr.max(0, keepdim=True)[0]
        edge_attr = (edge_attr - mn) / (mx - mn).clamp(min=1e-8)
        
        return self.cached_edge_index, edge_attr
