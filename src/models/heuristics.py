import torch
import random
import networkx as nx
import numpy as np

class Dijkstra_Worker:
    """휴리스틱 워커 (다익스트라 기반 최단 경로 이동)"""
    def __init__(self, worker_env):
        self.env = worker_env
        self.is_heuristic = True

    def get_actions(self, active):
        """HRLEnv.step_manager에서 넘겨준 active batch index 리스트에 대해 전체 batch size 크기의 action list 반환"""
        B = self.env.batch_size
        actions = [0] * B
        for b in active:
            c_idx = int(self.env.curr_nodes[b].item())
            c_node = self.env.idx_to_node[c_idx]
            
            # 목적지 결정 (subgoal_mode에 따라)
            if self.env.subgoal_mode == 'node':
                t_idx = int(self.env.subgoal_nodes[b].item())
            else:
                t_idx = int(self.env.target_nodes[b].item())
                
            t_node = self.env.idx_to_node[t_idx]
            
            # 현재 노드가 목적지면 제자리 반환
            if c_node == t_node:
                actions[b] = c_idx
                continue
                
            # 다익스트라 경로 탐색
            try:
                path = nx.shortest_path(self.env.G, source=c_node, target=t_node, weight='weight')
                if len(path) > 1:
                    next_node = path[1]
                    actions[b] = self.env.node_to_idx[next_node]
                else:
                    actions[b] = c_idx
            except nx.NetworkXNoPath:
                # 고립되었을 경우 (Stagnation 유발을 위해 제자리 반환)
                actions[b] = c_idx
                
        return actions
    
    def eval(self):
        pass


class ALNS_Manager:
    """ALNS 기반 매니저 (간단한 VRP/OP 용 파괴-복구 연산자 적용)"""
    def __init__(self, hrl_env, iterations=50):
        self.env = hrl_env
        self.iterations = iterations
        self.current_plan = {} # batch_idx -> target_sequence

    def get_action(self):
        """현재 상태를 보고 최적의 target과 zone을 반환"""
        B = self.env.batch_size
        t_acts = torch.zeros(B, dtype=torch.long, device=self.env.env.device)
        z_acts = torch.zeros(B, dtype=torch.long, device=self.env.env.device)
        
        for b in range(B):
            c_idx = int(self.env.env.curr_nodes[b].item())
            c_node = self.env.env.idx_to_node[c_idx]
            c_zone = self.env.env.n2z[c_node]
            
            # 남은 유효 타겟 도출
            valid_targets = []
            for i in range(self.env.num_targets):
                if not self.env.target_rescued[b, i] and not getattr(self.env, 'target_failed', torch.zeros_like(self.env.target_rescued))[b, i]:
                    valid_targets.append(i)
                    
            if not valid_targets:
                t_acts[b] = 0
                z_acts[b] = c_zone
                continue
                
            # ALNS Solve
            seq = self._solve_alns(b, c_node, valid_targets)
            if not seq:
                t_acts[b] = valid_targets[0]
                t_idx = int(self.env.targets[b, valid_targets[0]].item())
                t_node = self.env.env.idx_to_node[t_idx]
                z_acts[b] = self._get_next_zone(c_node, t_node)
            else:
                best_t = seq[0]
                t_acts[b] = best_t
                t_idx = int(self.env.targets[b, best_t].item())
                t_node = self.env.env.idx_to_node[t_idx]
                z_acts[b] = self._get_next_zone(c_node, t_node)
                
        return t_acts, z_acts

    def _solve_alns(self, b, start_node, valid_targets):
        if len(valid_targets) <= 1:
            return valid_targets
            
        # 1. 초기해: Nearest Neighbor
        curr = start_node
        unvisited = set(valid_targets)
        curr_time = self.env.current_time[b].item()
        solution = []
        
        while unvisited:
            best_t = None
            best_dist = float('inf')
            for t in unvisited:
                t_idx = int(self.env.targets[b, t].item())
                t_node = self.env.env.idx_to_node[t_idx]
                try:
                    d = nx.shortest_path_length(self.env.env.G, curr, t_node, weight='weight')
                except nx.NetworkXNoPath:
                    d = float('inf')
                if d < best_dist:
                    best_dist = d
                    best_t = t
            if best_t is not None and best_dist != float('inf'):
                # 데드라인 체크
                if curr_time + best_dist <= self.env.deadlines[b, best_t].item():
                    solution.append(best_t)
                    curr_time += best_dist
                    curr = self.env.env.idx_to_node[int(self.env.targets[b, best_t].item())]
                unvisited.remove(best_t)
            else:
                break
                
        best_solution = solution[:]
        # 목적함수: 구조 수 최우선 (가중치 10000), 이동 거리(시간) 최소화 (Tie-breaker)
        best_score = len(best_solution) * 10000 - curr_time
        
        # 2. ALNS Iterations
        for _ in range(self.iterations):
            # Destroy: 무작위 하나 제거
            if not solution:
                break
            temp_sol = solution[:]
            removed = random.choice(temp_sol)
            temp_sol.remove(removed)
            
            # Repair: 남은 타겟(valid_targets) 중 가장 짧은 거리로 삽입 가능한 놈 하나 추가
            unv = set(valid_targets) - set(temp_sol)
            if unv:
                t = random.choice(list(unv))
                temp_sol.append(t)
                
            # Evaluate temp_sol
            curr_t = self.env.current_time[b].item()
            c_node = start_node
            valid = True
            rescued = 0
            for t in temp_sol:
                t_node = self.env.env.idx_to_node[int(self.env.targets[b, t].item())]
                try:
                    d = nx.shortest_path_length(self.env.env.G, c_node, t_node, weight='weight')
                    if curr_t + d <= self.env.deadlines[b, t].item():
                        curr_t += d
                        c_node = t_node
                        rescued += 1
                    else:
                        valid = False
                        break
                except nx.NetworkXNoPath:
                    valid = False
                    break
            
            temp_score = rescued * 10000 - curr_t
            if valid and temp_score >= best_score:
                solution = temp_sol[:]
                best_score = temp_score
                best_solution = solution[:]
                
        return best_solution if best_solution else valid_targets
        
    def _get_next_zone(self, start_node, target_node):
        start_z = self.env.env.n2z[start_node]
        target_z = self.env.env.n2z[target_node]
        if start_z == target_z:
            return start_z
            
        try:
            z_path = nx.shortest_path(self.env.env.ZG, start_z, target_z, weight='weight')
            if len(z_path) > 1:
                return z_path[1]
            return start_z
        except nx.NetworkXNoPath:
            return start_z


class GA_Manager:
    """GA 기반 매니저"""
    def __init__(self, hrl_env, pop_size=20, generations=20):
        self.env = hrl_env
        self.pop_size = pop_size
        self.generations = generations

    def get_action(self):
        B = self.env.batch_size
        t_acts = torch.zeros(B, dtype=torch.long, device=self.env.env.device)
        z_acts = torch.zeros(B, dtype=torch.long, device=self.env.env.device)
        
        for b in range(B):
            c_idx = int(self.env.env.curr_nodes[b].item())
            c_node = self.env.env.idx_to_node[c_idx]
            c_zone = self.env.env.n2z[c_node]
            
            valid_targets = []
            for i in range(self.env.num_targets):
                if not self.env.target_rescued[b, i] and not getattr(self.env, 'target_failed', torch.zeros_like(self.env.target_rescued))[b, i]:
                    valid_targets.append(i)
                    
            if not valid_targets:
                t_acts[b] = 0
                z_acts[b] = c_zone
                continue
                
            seq = self._solve_ga(b, c_node, valid_targets)
            best_t = seq[0] if seq else valid_targets[0]
            
            t_acts[b] = best_t
            t_idx = int(self.env.targets[b, best_t].item())
            t_node = self.env.env.idx_to_node[t_idx]
            
            start_z = c_zone
            target_z = self.env.env.n2z[t_node]
            if start_z == target_z:
                z_acts[b] = start_z
            else:
                try:
                    z_path = nx.shortest_path(self.env.env.ZG, start_z, target_z, weight='weight')
                    z_acts[b] = z_path[1] if len(z_path) > 1 else start_z
                except nx.NetworkXNoPath:
                    z_acts[b] = start_z
                    
        return t_acts, z_acts

    def _solve_ga(self, b, start_node, valid_targets):
        if len(valid_targets) <= 1:
            return valid_targets
            
        population = [random.sample(valid_targets, len(valid_targets)) for _ in range(self.pop_size)]
        best_chromosome = None
        best_fitness = -float('inf')
        
        for _ in range(self.generations):
            scored_pop = []
            for chrom in population:
                curr_t = self.env.current_time[b].item()
                c_node = start_node
                rescued = 0
                for t in chrom:
                    t_node = self.env.env.idx_to_node[int(self.env.targets[b, t].item())]
                    try:
                        d = nx.shortest_path_length(self.env.env.G, c_node, t_node, weight='weight')
                        if curr_t + d <= self.env.deadlines[b, t].item():
                            curr_t += d
                            c_node = t_node
                            rescued += 1
                        else:
                            break # Time window exceed
                    except nx.NetworkXNoPath:
                        break
                score = rescued * 10000 - curr_t
                scored_pop.append((score, chrom))
                if score > best_fitness:
                    best_fitness = score
                    best_chromosome = chrom[:]
                    
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            survivors = [x[1] for x in scored_pop[:self.pop_size//2]]
            
            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                cut = len(p1) // 2
                child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
                if random.random() < 0.1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                next_gen.append(child)
                
            population = next_gen
            
        return best_chromosome if best_chromosome else valid_targets
