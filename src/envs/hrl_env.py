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
        for p in self.worker.parameters():
            p.requires_grad_(False)
            
        self.env = worker_env
        self.max_time = 200
        self.max_manager_turns = 50
        
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
        
    def reset(self, batch_size=1, num_targets=10):
        """새로운 시나리오(타겟 목록) 생성"""
        self.batch_size = batch_size
        self.num_targets = num_targets
        
        # 동적 한계치 설정 (타겟 개수에 비례)
        self.max_manager_turns = num_targets * 20
        self.max_time = num_targets * 80
        
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

        for b in range(batch_size):
            s = random.choice(self.env.nodes)
            self.env.curr_nodes[b] = self.env.node_to_idx[s]
            self.env.visited_nodes[b, int(self.env.curr_nodes[b])] = 1.0
            
            # 타겟 N개 생성
            selected = set([s])
            for i in range(num_targets):
                while True:
                    t = random.choice(self.env.nodes)
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
                            
                        # 가중치 거리 기준 최소 거리 + 랜덤 여유
                        self.deadlines[b, i] = int(min(weight_dist * 1.5 + random.uniform(20, 50), self.max_time))
                        break

        self.target_rescued = torch.zeros(batch_size, num_targets, dtype=torch.bool, device=self.env.device)
        self.target_failed = torch.zeros(batch_size, num_targets, dtype=torch.bool, device=self.env.device)
        self.current_time = torch.zeros(batch_size, dtype=torch.float, device=self.env.device)
        self.manager_turns = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.worker_steps = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        self.num_rescued = torch.zeros(batch_size, dtype=torch.long, device=self.env.device)
        
        self.curr_target_idx = torch.full((batch_size,), -1, dtype=torch.long, device=self.env.device)
        self.curr_zone_action = torch.full((batch_size,), -1, dtype=torch.long, device=self.env.device)
        
        self.dones = torch.zeros(batch_size, dtype=torch.bool, device=self.env.device)
        
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
        """
        [B, K, 6] 텐서
        0: is_curr_zone
        1: has_target
        2: is_visited
        3: disaster_intensity (현재는 0 고정, 나중에 동적 재난 시 연동)
        4: dist_from_curr
        5: cos_sim
        """
        B = self.batch_size
        K_zones = self.env.k
        feats = torch.zeros(B, K_zones, 6, device=self.env.device)
        
        for b in range(B):
            c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
            c_zone = self.env.n2z[c_node]
            
            # is_curr_zone
            feats[b, c_zone, 0] = 1.0
            
            # has_target
            for i in range(self.num_targets):
                if not self.target_rescued[b, i]:
                    tz = self.target_zones[b, i].item()
                    feats[b, tz, 1] = 1.0
                    
            # is_visited (Worker 방문 내역을 Zone 단위로 변환)
            v_nodes = self.env.visited_nodes[b].nonzero(as_tuple=True)[0]
            for vn in v_nodes:
                vz = self.env.n2z[self.env.idx_to_node[int(vn)]]
                feats[b, vz, 2] = 1.0
                
            # dist_from_curr (centroid 거리 기반)
            p1 = self.env.zone_centroids[c_zone]
            for z in range(K_zones):
                p2 = self.env.zone_centroids[z]
                dist = float(torch.norm(p1 - p2))
                feats[b, z, 4] = dist / max(self.env.max_dist, 1.0)
                
            # cos_sim 생략 (필요 시 복구)
            
        return feats
        
    def get_zone_edge_index(self):
        # [2, E]
        el = list(self.env.ZG.edges())
        bidir = el + [(v, u) for u, v in el]
        return torch.tensor(bidir, dtype=torch.long, device=self.env.device).t()
        
    def get_target_features(self):
        """
        [B, N, 4]
        0: deadline (정규화)
        1: time_remaining
        2: dist_from_curr
        3: rescue_priority
        """
        B, N = self.batch_size, self.num_targets
        feats = torch.zeros(B, N, 4, device=self.env.device)
        for b in range(B):
            c_idx = int(self.env.curr_nodes[b])
            for i in range(N):
                t_idx = int(self.targets[b, i])
                d = self.deadlines[b, i]
                rem = d - self.current_time[b]
                dist = self.env.dist_matrix[c_idx, t_idx]
                
                feats[b, i, 0] = d / self.max_time
                feats[b, i, 1] = rem / self.max_time
                feats[b, i, 2] = dist / max(self.env.max_dist, 1.0)
                feats[b, i, 3] = 1.0  # priority default
        return feats
        
    def get_target_mask(self):
        """[B, N] 유효한 타겟=1, 구조완료/시간초과=0"""
        B, N = self.batch_size, self.num_targets
        mask = torch.ones(B, N, dtype=torch.long, device=self.env.device)
        for b in range(B):
            for i in range(N):
                if self.target_rescued[b, i]:
                    mask[b, i] = 0
                if self.current_time[b] > self.deadlines[b, i]:
                    mask[b, i] = 0
        return mask
        
    def get_zone_adj_mask(self):
        """[B, K] 현재 Zone과 인접한 Zone(혹은 자기 자신) = 1"""
        B, K = self.batch_size, self.env.k
        mask = torch.zeros(B, K, dtype=torch.long, device=self.env.device)
        for b in range(B):
            c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
            c_zone = self.env.n2z[c_node]
            mask[b, c_zone] = 1
            for nbr in self.env.ZG.neighbors(c_zone):
                mask[b, nbr] = 1
        return mask
        
    def step_manager(self, target_actions, zone_actions):
        """
        target_actions: [B] target index (0 ~ N-1)
        zone_actions: [B] zone index (0 ~ K-1)
        
        선택된 Zone으로 Worker를 진행시킵니다.
        """
        B = self.batch_size
        self.curr_target_idx = target_actions
        self.curr_zone_action = zone_actions
        
        # Worker에게 부여할 목표 세팅
        self.env.zone_sequences = []
        self.env.zone_seq_idxs = torch.zeros(B, dtype=torch.long)
        
        for b in range(B):
            if not self.dones[b]:
                self.manager_turns[b] += 1
                t_idx = self.curr_target_idx[b].item()
                self.env.target_nodes[b] = self.targets[b, t_idx]
                
            # Worker에게 다음 가야할 zone sequence를 설정 (done이어도 shape 맞추기 위해 추가)
            c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
            c_zone = self.env.n2z[c_node]
            z_act = self.curr_zone_action[b].item()
            self.env.zone_sequences.append([c_zone, z_act])
            
        # Worker Execution Loop
        events = ['none'] * B
        
        # [Phase 2 Stage 3] 여진(Aftershock) 동적 발생
        import random
        if self.env.dynamic_disaster and random.random() < 0.05: # 매 턴마다 5% 확률로 여진 발생
            self.env.apply_aftershock()
            
        worker_done = self.dones.clone()
        
        # Graph data for worker
        el = [(self.env.node_to_idx[u], self.env.node_to_idx[v]) for u, v in self.env.G.edges()]
        bidir = el + [(v, u) for u, v in el]
        edge_index = torch.tensor(bidir, dtype=torch.long).t().to(self.env.device)
        N_nodes = self.env.num_nodes
        
        ea = []
        for ui, vi in bidir:
            u, v = self.env.idx_to_node[ui], self.env.idx_to_node[vi]
            d = self.env.dm.graph[u][v]
            ea.append([d.get('length', 0.0)])
        edge_attr = torch.tensor(ea, dtype=torch.float).to(self.env.device)
        mn = edge_attr.min(0,keepdim=True)[0]; mx = edge_attr.max(0,keepdim=True)[0]
        edge_attr = (edge_attr - mn) / (mx - mn).clamp(min=1e-8)

        # Execute Worker
        for _ in range(50): # max inner steps
            if worker_done.all():
                break
                
            active = [b for b in range(B) if not worker_done[b]]
            A = len(active)
            
            st = self.env._get_state_batch()
            xs = torch.stack([st[b].to(self.env.device) for b in active])
            ms = torch.stack([self.env.get_action_mask_batch()[b].to(self.env.device) for b in active])
            
            xf = xs.view(-1, xs.shape[-1])
            mf = ms.view(-1)
            ai = torch.arange(A, device=self.env.device).repeat_interleave(N_nodes)
            aei = torch.cat([edge_index + i*N_nodes for i in range(A)], dim=1)
            eaf = edge_attr.repeat(A, 1) if edge_attr is not None else None
            
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
                prev_n = self.env.idx_to_node[int(prev_nodes[b])]
                new_n = self.env.idx_to_node[int(self.env.curr_nodes[b])]
                if prev_n == new_n:
                    edge_w = 1.0 # Self-loop fallback
                else:
                    edge_w = self.env.G[prev_n][new_n].get('weight', 1.0)
                    
                self.worker_steps[b] += 1
                self.current_time[b] += edge_w
                # Check events
                c_node = self.env.idx_to_node[int(self.env.curr_nodes[b])]
                c_zone = self.env.n2z[c_node]
                target_node = self.env.idx_to_node[int(self.env.target_nodes[b])]
                target_idx_in_list = self.curr_target_idx[b].item()
                
                if c_node == target_node:
                    events[b] = 'target_rescued'
                    self.target_rescued[b, target_idx_in_list] = True
                    self.num_rescued[b] += 1
                    worker_done[b] = True
                elif c_zone != self.env.zone_sequences[b][0]:
                    if c_zone == self.curr_zone_action[b].item():
                        events[b] = 'zone_arrived'
                    else:
                        events[b] = 'zone_escaped'
                    worker_done[b] = True
                elif step_dones[b].item():
                    # OOB
                    events[b] = 'zone_escaped'
                    worker_done[b] = True
                    
        # Update dones
        target_mask = self.get_target_mask()
        for b in range(B):
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
