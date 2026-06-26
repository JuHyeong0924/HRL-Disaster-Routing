"""
Phase 2: Manager HRL 트레이너 (정통 PPO 도입 버전)

- POMO의 K-복제 롤아웃을 제거하고 단일 궤적 기반 PPO로 전환.
- 타겟(Target)과 구역(Zone) 선택에 대한 Actor-Critic PPO 업데이트 수행.
- GAE(Generalized Advantage Estimation)를 활용하여 안정적인 학습 도모.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any
from torch.distributions import Categorical

class ManagerTrainer:
    def __init__(self, manager, hrl_env, K=1, lr=1e-4, max_grad_norm=1.0, config=None):
        self.manager = manager
        self.env = hrl_env
        self.K = K # K=1 (PPO rollout)
        self.device = next(self.manager.parameters()).device
        
        # PPO Hyperparameters
        if config is None:
            config = type('Config', (), {})()
            
        self.lr = getattr(config, 'lr', lr)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.clip_ratio = getattr(config, 'clip_ratio', 0.2)
        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.01)
        self.vf_coeff = getattr(config, 'vf_coeff', 0.5)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 4)
        self.mini_batch_size = getattr(config, 'mini_batch_size', 64)
        self.max_grad_norm = getattr(config, 'max_grad_norm', max_grad_norm)
        
        self.optimizer = optim.Adam(self.manager.parameters(), lr=self.lr)

    def _compute_gae(self, rewards: List[float], values: List[float]) -> torch.Tensor:
        advantages = []
        gae = 0.0
        next_value = 0.0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = v
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _run_batch_episodes(self, batch_size=32, num_targets=10):
        self.manager.eval()
        self.env.reset(batch_size=batch_size, num_targets=num_targets)
        B = batch_size
        
        # 에피소드 궤적
        ep_zone_features = [[] for _ in range(B)]
        ep_target_features = [[] for _ in range(B)]
        ep_target_mask = [[] for _ in range(B)]
        ep_target_zones = [[] for _ in range(B)]
        ep_zone_adj_mask = [[] for _ in range(B)]
        
        # [Phase 2F] h_last 제거 → num_feasible, avg_urgency로 교체
        ep_num_feasible = [[] for _ in range(B)]
        ep_avg_urgency = [[] for _ in range(B)]
        ep_elapsed = [[] for _ in range(B)]
        ep_rescued = [[] for _ in range(B)]
        
        ep_t_acts = [[] for _ in range(B)]
        ep_z_acts = [[] for _ in range(B)]
        
        ep_rewards = [[] for _ in range(B)]
        ep_values = [[] for _ in range(B)]
        ep_log_probs = [[] for _ in range(B)]
        
        done_flags = [False] * B
        
        # 초기 상태
        prev_start_zones = {b: -1 for b in range(B)}
        prev_z_acts = {b: -1 for b in range(B)}
        prev_target_failed = {b: self.env.target_failed[b].clone() for b in range(B)}
        
        # Manager Turns Execution Loop
        while not all(done_flags):
            active = [b for b in range(B) if not done_flags[b]]
            A = len(active)
            if A == 0: break
            
            # 피처 추출
            zone_features = self.env.get_zone_features()[active] # [A, K_zones, 6]
            zone_edge_index = self.env.get_zone_edge_index() # [2, E]
            target_features = self.env.get_target_features()[active] # [A, N, 6]
            target_mask = self.env.get_target_mask()[active] # [A, N]
            target_zones = self.env.target_zones[active] # [A, N]
            zone_adj_mask = self.env.get_zone_adj_mask()[active] # [A, K_zones]
            zone_dist_matrix = self.env.zone_dist_matrix.unsqueeze(0).expand(A, -1, -1) # [A, K, K]
            
            elapsed = self.env.current_time[active].unsqueeze(1) / self.env.max_time
            rescued = self.env.num_rescued[active].unsqueeze(1).float() / num_targets
            
            # [Phase 2F] num_feasible, avg_urgency 계산
            num_feasible = target_mask.sum(dim=-1, keepdim=True).float() / num_targets  # [A, 1]
            urgency_ch = target_features[:, :, 3]  # [A, N] urgency_ratio
            mask_float = target_mask.float()  # [A, N]
            avg_urgency = (urgency_ch * mask_float).sum(dim=-1, keepdim=True) / mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # [A, 1]
            
            # 버퍼에 저장 (CPU)
            for i, b in enumerate(active):
                ep_zone_features[b].append(zone_features[i].clone().cpu())
                ep_target_features[b].append(target_features[i].clone().cpu())
                ep_target_mask[b].append(target_mask[i].clone().cpu())
                ep_target_zones[b].append(target_zones[i].clone().cpu())
                ep_zone_adj_mask[b].append(zone_adj_mask[i].clone().cpu())
                ep_num_feasible[b].append(num_feasible[i].clone().cpu())
                ep_avg_urgency[b].append(avg_urgency[i].clone().cpu())
                ep_elapsed[b].append(elapsed[i].clone().cpu())
                ep_rescued[b].append(rescued[i].clone().cpu())
                
            # ── 모델 추론 (No Grad) ──
            K_zones = self.env.env.k
            flat_zf = zone_features.view(A * K_zones, 6)
            ai = torch.arange(A, device=self.device).repeat_interleave(K_zones)
            aei = torch.cat([zone_edge_index + i*K_zones for i in range(A)], dim=1)
            
            zone_emb = self.manager.encode_zones(flat_zf, aei, batch=ai) # [A * K_zones, 128]
            
            t_ai = torch.arange(A, device=self.device).repeat_interleave(num_targets)
            act_offsets = torch.arange(A, device=self.device).repeat_interleave(num_targets) * K_zones
            flat_tz_idx = act_offsets + target_zones.view(-1)
            
            t_emb = self.manager.get_target_embeddings(zone_emb, target_features.view(-1, 6), flat_tz_idx)
            
            # [Phase 2F] context generator: h_last 제거
            query = self.manager.generate_context(elapsed, rescued, num_feasible, avg_urgency)
            
            t_logits, t_inv, t_emb_dense = self.manager.get_target_logits(query, t_emb, target_mask.view(-1), t_ai)
            t_dist = Categorical(logits=t_logits)
            t_act = t_dist.sample()
            t_log_prob = t_dist.log_prob(t_act)
            
            z_ai = torch.arange(A, device=self.device).repeat_interleave(K_zones)
            selected_t_emb = t_emb_dense[torch.arange(A), t_act]
            
            selected_tz = target_zones[torch.arange(A), t_act]
            z_logits, z_inv = self.manager.get_zone_logits(
                query, selected_t_emb, zone_emb, zone_adj_mask.view(-1), z_ai, 
                selected_tz, zone_dist_matrix
            )
            z_dist = Categorical(logits=z_logits)
            z_act = z_dist.sample()
            z_log_prob = z_dist.log_prob(z_act)
            
            # Value
            value = self.manager.get_value(query)
            
            full_log_prob = t_log_prob + z_log_prob
            
            for i, b in enumerate(active):
                ep_t_acts[b].append(t_act[i].item())
                ep_z_acts[b].append(z_act[i].item())
                ep_log_probs[b].append(full_log_prob[i].item())
                ep_values[b].append(value[i].item())
            
            # 환경 진행 전의 리워드 스냅샷
            prev_manager_turns = self.env.manager_turns.clone()
            prev_worker_steps = self.env.worker_steps.clone()
            prev_rescued = self.env.num_rescued.clone()
            prev_elapsed_time = self.env.current_time.clone()
            
            # 액션 적용 전: 이번 턴의 출발 구역 기록 (재방문 판별용) 및 PBRS 거리 스냅샷
            curr_start_zones = {}
            prev_dist_to_target = {}
            for i, b in enumerate(active):
                c_node_idx = int(self.env.env.curr_nodes[b].item())
                c_node = self.env.env.idx_to_node[c_node_idx]
                curr_start_zones[b] = self.env.env.n2z[c_node]
                
                # PBRS용: 현재 위치에서 타겟까지의 초기 다익스트라 거리
                t_idx = t_act[i].item()
                target_node_idx = int(self.env.targets[b, t_idx].item())
                d_prev = self.env.env.dist_matrix[c_node_idx, target_node_idx]
                if d_prev == float('inf') or np.isinf(d_prev):
                    d_prev = self.env.env.max_dist
                prev_dist_to_target[b] = d_prev
            
            # 액션 적용 (Active 배치에 대해서만 추출하여 패스)
            act_t_act = torch.zeros(B, dtype=torch.long, device=self.device)
            act_z_act = torch.zeros(B, dtype=torch.long, device=self.device)
            for i, b in enumerate(active):
                act_t_act[b] = t_act[i]
                act_z_act[b] = z_act[i]
                
            events, new_dones = self.env.step_manager(act_t_act, act_z_act)
            
            # [Phase 2C] 보상 재설계 — HAZUS 환경에 맞는 reward shaping
            for i, b in enumerate(active):
                # 1. 구출 보상 증폭: 10.0 → 20.0
                reward_rescued = (self.env.num_rescued[b].item() - prev_rescued[b].item()) * 20.0
                reward_turns = (self.env.manager_turns[b].item() - prev_manager_turns[b].item()) * -0.5
                
                # 워커 이동 횟수(Step) 대신 물리적 소요 시간(Time) 기준 페널티
                elapsed_time = self.env.current_time[b].item() - prev_elapsed_time[b].item()
                reward_time = elapsed_time * -0.1
                
                # PBRS: 계수 축소 5.0 → 2.0 (과도한 밀집 보상이 스파스 보상을 압도하는 문제 해결)
                c_node_idx_after = int(self.env.env.curr_nodes[b].item())
                t_idx = t_act[i].item()
                target_node_idx = int(self.env.targets[b, t_idx].item())
                d_curr = self.env.env.dist_matrix[c_node_idx_after, target_node_idx]
                if d_curr == float('inf') or np.isinf(d_curr):
                    d_curr = self.env.env.max_dist
                curr_dist_to_target = d_curr
                reward_pbrs = (np.log1p(prev_dist_to_target[b]) - np.log1p(curr_dist_to_target)) * 2.0
                
                turn_reward = reward_rescued + reward_turns + reward_time + reward_pbrs
                
                # [Phase 2C] 데드라인 만료 페널티 (-5.0 per newly expired target)
                for ti in range(num_targets):
                    if (not self.env.target_rescued[b, ti] and
                        not prev_target_failed[b][ti] and
                        self.env.target_failed[b, ti]):
                        turn_reward -= 5.0
                
                # [Phase 2C] 여유 시간 보너스 (데드라인 내 충분한 여유로 구출 시)
                if reward_rescued > 0:
                    t_idx_for_slack = t_act[i].item()
                    slack = self.env.deadlines[b, t_idx_for_slack].item() - self.env.current_time[b].item()
                    if slack > 0:
                        turn_reward += min(slack / self.env.max_time * 5.0, 3.0)
                
                # 명령 불복종(제자리 이탈) 페널티
                if events[b] == 'zone_escaped':
                    turn_reward -= 2.0
                    
                # 재방문(Tabu) 페널티
                if z_act[i].item() == prev_start_zones[b]:
                    turn_reward -= 1.0
                    
                # 정체 감점(Stagnation Penalty)
                if z_act[i].item() == prev_z_acts[b] and reward_rescued <= 0:
                    turn_reward -= 2.0
                    
                prev_z_acts[b] = z_act[i].item()
                prev_start_zones[b] = curr_start_zones[b]
                prev_target_failed[b] = self.env.target_failed[b].clone()
                
                ep_rewards[b].append(turn_reward)
                done_flags[b] = new_dones[b].item()

        # GAE 계산 및 버퍼 취합
        buffer = {
            'zone_features': [], 'target_features': [], 'target_mask': [], 'target_zones': [],
            'zone_adj_mask': [], 'num_feasible': [], 'avg_urgency': [], 'elapsed': [], 'rescued': [],
            't_acts': [], 'z_acts': [], 'old_log_probs': [], 'returns': [], 'advantages': []
        }
        
        batch_mean_reward = 0
        batch_mean_rescued = 0
        batch_mean_turns = 0
        batch_mean_steps = 0
        
        for b in range(B):
            r_list = ep_rewards[b]
            v_list = ep_values[b]
            adv_tensor = self._compute_gae(r_list, v_list)
            v_tensor = torch.tensor(v_list, dtype=torch.float32, device=self.device)
            ret_tensor = adv_tensor + v_tensor
            
            buffer['zone_features'].extend(ep_zone_features[b])
            buffer['target_features'].extend(ep_target_features[b])
            buffer['target_mask'].extend(ep_target_mask[b])
            buffer['target_zones'].extend(ep_target_zones[b])
            buffer['zone_adj_mask'].extend(ep_zone_adj_mask[b])
            buffer['num_feasible'].extend(ep_num_feasible[b])
            buffer['avg_urgency'].extend(ep_avg_urgency[b])
            buffer['elapsed'].extend(ep_elapsed[b])
            buffer['rescued'].extend(ep_rescued[b])
            buffer['t_acts'].extend(ep_t_acts[b])
            buffer['z_acts'].extend(ep_z_acts[b])
            buffer['old_log_probs'].extend(ep_log_probs[b])
            buffer['returns'].extend(ret_tensor.cpu().numpy())
            buffer['advantages'].extend(adv_tensor.cpu().numpy())
            
            batch_mean_reward += sum(r_list)
            batch_mean_rescued += self.env.num_rescued[b].item()
            batch_mean_turns += self.env.manager_turns[b].item()
            batch_mean_steps += self.env.worker_steps[b].item()
            
        # 텐서 스택
        for k in buffer.keys():
            if k in ['t_acts', 'z_acts']:
                buffer[k] = torch.tensor(buffer[k], dtype=torch.long)
            elif k in ['old_log_probs', 'returns', 'advantages']:
                buffer[k] = torch.tensor(buffer[k], dtype=torch.float32)
            else:
                buffer[k] = torch.stack(buffer[k])
                
        # Advantage 정규화
        adv = buffer['advantages']
        if adv.numel() > 1:
            buffer['advantages'] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        else:
            buffer['advantages'] = adv - adv.mean()
            
        stats = {
            'mean_reward': batch_mean_reward / B,
            'mean_rescued': batch_mean_rescued / B,
            'mean_manager_turns': batch_mean_turns / B,
            'mean_worker_steps': batch_mean_steps / B
        }
        return buffer, stats

    def train_step(self, batch_size=32, num_targets=10, current_ent_coef=0.01):
        """PPO Update"""
        buffer, stats = self._run_batch_episodes(batch_size, num_targets)
        
        self.manager.train()
        T_total = buffer['t_acts'].size(0)
        indices = np.arange(T_total)
        
        num_targets = buffer['target_features'].size(1)
        K_zones = self.env.env.k
        zone_edge_index = self.env.get_zone_edge_index()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T_total, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                # 미니배치 데이터 로드 (GPU)
                mb_zf = buffer['zone_features'][mb_idx].to(self.device)
                mb_tf = buffer['target_features'][mb_idx].to(self.device)
                mb_tm = buffer['target_mask'][mb_idx].to(self.device)
                mb_tz = buffer['target_zones'][mb_idx].to(self.device)
                mb_zam = buffer['zone_adj_mask'][mb_idx].to(self.device)
                mb_num_feasible = buffer['num_feasible'][mb_idx].to(self.device)
                mb_avg_urgency = buffer['avg_urgency'][mb_idx].to(self.device)
                mb_elapsed = buffer['elapsed'][mb_idx].to(self.device)
                mb_rescued = buffer['rescued'][mb_idx].to(self.device)
                
                mb_t_acts = buffer['t_acts'][mb_idx].to(self.device)
                mb_z_acts = buffer['z_acts'][mb_idx].to(self.device)
                mb_old_log_probs = buffer['old_log_probs'][mb_idx].to(self.device)
                mb_returns = buffer['returns'][mb_idx].to(self.device)
                mb_advantages = buffer['advantages'][mb_idx].to(self.device)
                
                A = mb_zf.size(0)
                
                # GNN 포워드 (미분 활성화)
                flat_zf = mb_zf.view(A * K_zones, 6)
                ai = torch.arange(A, device=self.device).repeat_interleave(K_zones)
                aei = torch.cat([zone_edge_index + i*K_zones for i in range(A)], dim=1)
                
                zone_emb = self.manager.encode_zones(flat_zf, aei, batch=ai)
                
                t_ai = torch.arange(A, device=self.device).repeat_interleave(num_targets)
                act_offsets = torch.arange(A, device=self.device).repeat_interleave(num_targets) * K_zones
                flat_tz_idx = act_offsets + mb_tz.view(-1)
                
                t_emb = self.manager.get_target_embeddings(zone_emb, mb_tf.view(-1, 6), flat_tz_idx)
                
                query = self.manager.generate_context(mb_elapsed, mb_rescued, mb_num_feasible, mb_avg_urgency)
                
                t_logits, t_inv, t_emb_dense = self.manager.get_target_logits(query, t_emb, mb_tm.view(-1), t_ai)
                t_dist = Categorical(logits=t_logits)
                new_t_log_prob = t_dist.log_prob(mb_t_acts)
                
                z_ai = torch.arange(A, device=self.device).repeat_interleave(K_zones)
                selected_t_emb = t_emb_dense[torch.arange(A), mb_t_acts]
                
                mb_selected_tz = mb_tz[torch.arange(A), mb_t_acts]
                mb_zone_dist_matrix = self.env.zone_dist_matrix.unsqueeze(0).expand(A, -1, -1).to(self.device)
                z_logits, z_inv = self.manager.get_zone_logits(
                    query, selected_t_emb, zone_emb, mb_zam.view(-1), z_ai,
                    mb_selected_tz, mb_zone_dist_matrix
                )
                z_dist = Categorical(logits=z_logits)
                new_z_log_prob = z_dist.log_prob(mb_z_acts)
                
                value = self.manager.get_value(query)
                
                new_log_probs = new_t_log_prob + new_z_log_prob
                entropy = (t_dist.entropy() + z_dist.entropy()).mean()
                
                # PPO Objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(value, mb_returns)
                
                loss = policy_loss + self.vf_coeff * value_loss - current_ent_coef * entropy
                
                if torch.isnan(loss):
                    continue
                    
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1
                
        stats['loss'] = total_policy_loss / max(num_updates, 1) + self.vf_coeff * total_value_loss / max(num_updates, 1)
        stats['policy_loss'] = total_policy_loss / max(num_updates, 1)
        stats['value_loss'] = total_value_loss / max(num_updates, 1)
        stats['entropy'] = total_entropy_loss / max(num_updates, 1)
        
        return stats
