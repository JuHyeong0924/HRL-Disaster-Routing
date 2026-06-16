"""
Phase 1: Worker 사전 학습 트레이너 (정통 PPO 도입 버전)

- Rollout 과정에서 torch.no_grad()를 사용하여 초고속 데이터 수집.
- 수집된 궤적(State, Action, Reward, Value 등)을 미니배치로 분할.
- Epoch 반복을 통한 PPO 학습 효율 극대화.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from src.envs.worker_env import WorkerEnv
from src.models.worker import Worker


class HRLWorkerTrainer:
    """Worker 사전 학습(PPO) 트레이너"""

    def __init__(self, env: WorkerEnv, logger: Any, worker_model: Worker, config: Any) -> None:
        self.env = env
        self.logger = logger
        self.worker = worker_model
        
        self.device = next(self.worker.parameters()).device
        
        # PPO 하이퍼파라미터
        self.lr = getattr(config, 'lr', 3e-4)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.clip_ratio = getattr(config, 'clip_ratio', 0.2)
        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.01)
        self.vf_coeff = getattr(config, 'vf_coeff', 0.5)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 4)
        # 미니배치 크기: GNN forward OOM 방지 핵심
        self.mini_batch_size = getattr(config, 'mini_batch_size', 128)
        self.batch_size = getattr(config, 'batch_size', 24)
        
        self.optimizer = optim.Adam(self.worker.parameters(), lr=self.lr)
        
        self.save_dir = getattr(config, 'save_dir', 'logs/rl_worker_stage')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 그래프 데이터 (동적 재구축 필요 시 _build_graph_data 호출)
        self.edge_index, self.edge_attr = self._build_graph_data()
    
    def _build_graph_data(self):
        """현재 그래프 상태를 반영한 edge_index, edge_attr 구축 (damage 채널 포함)."""
        el = [(self.env.node_to_idx[u], self.env.node_to_idx[v])
              for u, v in self.env.G.edges()]
        bidir = el + [(v, u) for u, v in el]
        edge_index = torch.tensor(bidir, dtype=torch.long).t().to(self.device)
        
        ea = []
        for ui, vi in bidir:
            u, v = self.env.idx_to_node[ui], self.env.idx_to_node[vi]
            d = self.env.G[u][v]
            ea.append([d.get('length', 0.0), d.get('damage', 0.0)])
        edge_attr = torch.tensor(ea, dtype=torch.float32).to(self.device)
        
        # Per-channel min-max normalization
        mn = edge_attr.min(0, keepdim=True)[0]
        mx = edge_attr.max(0, keepdim=True)[0]
        edge_attr = (edge_attr - mn) / (mx - mn).clamp(min=1e-8)
        
        return edge_index, edge_attr
        
    def _compute_gae(self, rewards: List[float], values: List[float]) -> torch.Tensor:
        """GAE (Generalized Advantage Estimation) 계산"""
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
    def _run_batch_episodes(self, batch_size: int) -> tuple:
        """초고속 Rollout (No Grad) — 계산 그래프 없이 데이터만 수집."""
        state = self.env.reset(batch_size=batch_size)  # [B, N, 4]
        B = batch_size
        N = state.shape[1]

        # 에피소드별 궤적 저장소
        ep_states = [[] for _ in range(B)]
        ep_actions = [[] for _ in range(B)]
        ep_rewards = [[] for _ in range(B)]
        ep_values = [[] for _ in range(B)]
        ep_log_probs = [[] for _ in range(B)]
        ep_masks = [[] for _ in range(B)]
        
        done_flags = [False] * B
        final_infos = [{} for _ in range(B)]

        while not all(done_flags):
            active = [b for b in range(B) if not done_flags[b]]
            A = len(active)

            active_states = torch.stack([state[b].to(self.device) for b in active])
            active_masks = torch.stack([self.env.get_action_mask_batch()[b].to(self.device) for b in active])

            x_flat = active_states.view(-1, active_states.shape[-1])
            mask_flat = active_masks.view(-1)
            
            ai = torch.arange(A, device=self.device).repeat_interleave(N)
            aei = torch.cat([self.edge_index + i * N for i in range(A)], dim=1)
            ae_attr = self.edge_attr.repeat(A, 1)

            # 모델 추론 (No Grad — 계산 그래프 없음)
            probs_all, values_all, _ = self.worker(x_flat, aei, batch=ai, neighbors_mask=mask_flat, edge_attr=ae_attr)
            
            probs_b = probs_all.view(A, N)
            values_b = values_all.view(A)
            
            actions_cpu = []
            
            for i, b in enumerate(active):
                prob = probs_b[i]
                v = values_b[i].item()
                
                dist = torch.distributions.Categorical(prob)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                
                act_item = action.item()
                actions_cpu.append(act_item)
                
                # CPU 텐서로 저장 (GPU 메모리 절약)
                ep_states[b].append(state[b].clone().cpu())
                ep_actions[b].append(act_item)
                ep_values[b].append(v)
                ep_log_probs[b].append(log_prob)
                ep_masks[b].append(self.env.get_action_mask_batch()[b].clone().cpu())
            
            all_actions = []
            ai_ptr = 0
            for b in range(B):
                if not done_flags[b]:
                    all_actions.append(actions_cpu[ai_ptr])
                    ai_ptr += 1
                else:
                    all_actions.append(0)
            
            # 환경 스텝
            next_state, rewards, dones, infos = self.env.step_batch(torch.tensor(all_actions))
            
            for i, b in enumerate(active):
                ep_rewards[b].append(rewards[b].item())
                if dones[b].item():
                    done_flags[b] = True
                    final_infos[b] = infos[b]
                else:
                    state[b] = next_state[b]

        # GAE 계산 및 버퍼 취합
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_masks = []
        
        batch_results = []
        
        for b in range(B):
            r_list = ep_rewards[b]
            v_list = ep_values[b]
            adv_tensor = self._compute_gae(r_list, v_list)
            v_tensor = torch.tensor(v_list, dtype=torch.float32, device=self.device)
            ret_tensor = adv_tensor + v_tensor
            
            batch_results.append({
                'reward': sum(r_list),
                'success': 1.0 if final_infos[b].get('reason') == 'success' else 0.0,
                'path_len': final_infos[b].get('path_len', len(r_list)),
            })
            
            all_states.extend(ep_states[b])
            all_actions.extend(ep_actions[b])
            all_old_log_probs.extend(ep_log_probs[b])
            all_returns.extend(ret_tensor.cpu().numpy())
            all_advantages.extend(adv_tensor.cpu().numpy())
            all_masks.extend(ep_masks[b])
            
        buffer = {}
        buffer['states'] = torch.stack(all_states)
        buffer['actions'] = torch.tensor(all_actions, dtype=torch.long)
        buffer['old_log_probs'] = torch.tensor(all_old_log_probs, dtype=torch.float32)
        buffer['returns'] = torch.tensor(all_returns, dtype=torch.float32)
        buffer['advantages'] = torch.tensor(all_advantages, dtype=torch.float32)
        buffer['masks'] = torch.stack(all_masks)
        
        # Advantage 정규화
        adv = buffer['advantages']
        if adv.numel() > 1:
            buffer['advantages'] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        else:
            buffer['advantages'] = adv - adv.mean()
        
        return buffer, batch_results

    def _update_ppo(self, buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """수집된 궤적(Buffer)을 미니배치 단위 PPO로 업데이트."""
        states = buffer['states'].to(self.device)
        actions = buffer['actions'].to(self.device)
        old_log_probs = buffer['old_log_probs'].to(self.device)
        returns = buffer['returns'].to(self.device)
        advantages = buffer['advantages'].to(self.device)
        masks = buffer['masks'].to(self.device)
        
        T_total = states.size(0)
        N = states.size(1)
        indices = np.arange(T_total)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T_total, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_masks = masks[mb_idx]
                
                A = mb_states.size(0)
                
                x_flat = mb_states.view(-1, mb_states.shape[-1])
                mask_flat = mb_masks.view(-1)
                
                ai = torch.arange(A, device=self.device).repeat_interleave(N)
                aei = torch.cat([self.edge_index + i * N for i in range(A)], dim=1)
                ae_attr = self.edge_attr.repeat(A, 1)
                
                # GNN forward (미분 활성화)
                probs_all, values_all, logits_all = self.worker(x_flat, aei, batch=ai, neighbors_mask=mask_flat, edge_attr=ae_attr)
                
                probs_b = probs_all.view(A, N)
                values_b = values_all.view(A)
                logits_b = logits_all.view(A, N)
                
                dist = torch.distributions.Categorical(logits=logits_b)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # PPO Clipped Objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(values_b, mb_returns)
                
                loss = policy_loss + self.vf_coeff * value_loss - self.entropy_coeff * entropy
                
                if torch.isnan(loss):
                    print("⚠️ Loss is NaN! Skipping update.")
                    continue
                    
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1
                
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy_loss / max(num_updates, 1),
        }

    def train(self, episodes: int) -> None:
        """메인 학습 루프: Rollout(No Grad) → PPO Update(Mini-batch)."""
        from collections import deque
        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)
        
        best_reward = -float('inf')
        B = self.batch_size
        
        print(f"📋 HRL Phase 1: Worker PPO 학습 (batch={B}, mini_batch={self.mini_batch_size}, ppo_epochs={self.ppo_epochs})")
        
        ep_count = 0
        with tqdm(total=episodes, desc="Phase 1 Worker", ncols=140, unit="ep") as pbar:
            while ep_count < episodes:
                
                # ---------------------------------------------------------
                # HRL 단계별 커리큘럼 전환 로직 (25% : 25% : 50% 비율)
                # ---------------------------------------------------------
                if ep_count <= int(episodes * 0.25):
                    self.env.disaster_prob = 0.0
                    self.env.dynamic_disaster = False
                    phase_str = 'P1:Normal'
                elif ep_count <= int(episodes * 0.50):
                    self.env.disaster_prob = 0.2
                    self.env.dynamic_disaster = False
                    phase_str = 'P2:Static'
                else:
                    self.env.disaster_prob = 0.2
                    self.env.dynamic_disaster = True
                    phase_str = 'P3:Dynamic'

                # 이번 배치에서 시뮬레이션할 에피소드 수 (마지막 배치를 위해)
                current_batch_size = min(B, episodes - ep_count)
                if current_batch_size <= 0:
                    break
                    
                buffer, batch_results = self._run_batch_episodes(current_batch_size)
                
                for r in batch_results:
                    recent_rewards.append(r['reward'])
                    recent_success.append(r['success'])
                    
                loss_info = self._update_ppo(buffer)
                
                avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
                avg_success = np.mean(recent_success) if len(recent_success) > 0 else 0.0
                
                # Update progress bar
                pbar.set_postfix({
                    'Phase': phase_str,
                    'SR': f"{avg_success*100:.1f}%",
                    'Rw': f"{avg_reward:.1f}",
                    'P-Loss': f"{loss_info['policy_loss']:.3f}",
                    'V-Loss': f"{loss_info['value_loss']:.3f}"
                })
                pbar.update(current_batch_size)
                ep_count += current_batch_size
                
                # Best 모델 저장
                if avg_reward > best_reward and len(recent_success) >= 50:
                    best_reward = avg_reward
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(self.worker.state_dict(), os.path.join(self.save_dir, 'best.pt'))

        # Final 저장
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.worker.state_dict(), os.path.join(self.save_dir, 'last.pt'))
        print(f'✅ HRL Worker Phase 1 PPO 학습 완료! Best Reward: {best_reward:.1f}')
