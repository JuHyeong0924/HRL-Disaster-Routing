"""
Manager v2 PPO Trainer: 비자기회귀 Manager를 PPO + GAE로 학습.

핵심 설계:
- SL(A* 모방) 완전 제거, 순수 RL만으로 학습
- Rollout Buffer로 에피소드 데이터 수집
- GAE로 토큰별(턴별) Advantage 계산
- PPO Clipped Objective로 안정적 정책 업데이트
- HRLClosedLoopEnv와 연동: Manager → Worker → PBRS → 반복
"""
import os
import json
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[Dict] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.truncations: List[bool] = []
        self.next_values: List[float] = []

        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def store(
        self,
        state_info: Dict,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        truncated: bool = False,
        next_value: float = 0.0,
    ) -> None:
        self.states.append(state_info)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.truncations.append(truncated)
        self.next_values.append(next_value)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                gae = 0.0
                next_val = self.next_values[t] if self.truncations[t] else 0.0
            else:
                next_val = self.values[t+1]

            delta = self.rewards[t] + gamma * next_val - self.values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32)

        if len(advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
            # 🚨 [삭제] 아래 self.returns 정규화 코드는 무조건 삭제하세요!
            # self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.truncations.clear()
        self.next_values.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)


class ManagerTrainer:
    def __init__(self, env, manager: nn.Module, config) -> None:
        self.env = env
        self.manager = manager
        self.config = config
        self.device = next(manager.parameters()).device

        self.lr = getattr(config, 'lr', 3e-4)
        self.optimizer = optim.Adam(manager.parameters(), lr=self.lr)

        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.clip_range = getattr(config, 'clip_range', 0.2)
        self.n_epochs = getattr(config, 'n_epochs', 4)
        self.value_coeff = getattr(config, 'value_coeff', 0.5)
        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.0)
        
        # [수정] 사용자님의 제안대로 SAC 스타일 자동 튜닝을 복구하되,
        # 이산 행동 공간에 맞게 target_entropy를 양수로 설정. (마스킹된 유효 액션 3~5개 기준 적절한 양수인 0.5로 설정)
        self.target_entropy = 0.5
        init_alpha = getattr(config, 'entropy_coeff', 0.02)
        safe_init_alpha = max(init_alpha, 1e-8)
        self.log_alpha = torch.tensor(np.log(safe_init_alpha), dtype=torch.float32,
                                       device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.batch_size = getattr(config, 'batch_size', 16)

        self.n_rollout_episodes = getattr(config, 'num_pomo', 16)
        self.save_dir = getattr(config, 'save_dir', 'logs/rl_manager_stage')
        os.makedirs(self.save_dir, exist_ok=True)

        self.buffer = RolloutBuffer()

    def collect_rollouts(self, pbar=None) -> Dict:
        """에피소드 n_rollout_episodes개를 수집하여 버퍼에 저장."""
        self.manager.eval()
        self.buffer.clear()

        episode_rewards = []
        episode_successes = []
        episode_turns = []
        episode_worker_steps = []
        episode_worker_successes = []

        for _ in range(self.n_rollout_episodes):
            current_idx, goal_idx = self.env.reset()
            ep_reward = 0.0
            worker_successes = 0
            manager_steps = 0
            
            while not self.env.done:
                x = self.env.get_manager_state()
                candidate_mask = self.env.get_candidate_mask()

                # 유효 후보가 없으면 강제 종료
                if candidate_mask.sum() == 0:
                    self.buffer.store(
                        state_info={'x': x.cpu(), 'curr_z': 0, 'goal_z': 0, 'mask': candidate_mask.cpu()},
                        action=0, reward=-1.0, value=0.0, log_prob=0.0, done=True
                    )
                    self.env.done = True
                    break

                curr_z = int(self.env._node_zone_tensor[self.env.current_idx].item())
                goal_z = int(self.env._node_zone_tensor[self.env.goal_idx].item())
                
                # Manager 정책으로 순수하게 액션 선택 (Heuristic 없음)
                action, log_prob, value, entropy = self.manager.select_action(
                    x, self.env.zone_edge_index, curr_z,
                    goal_z, candidate_mask,
                )

                reward, done, info = self.env.step(action)
                ep_reward += reward
                manager_steps += 1
                if info.get('reached_subgoal', False):
                    worker_successes += 1

                truncated = False
                next_val = 0.0
                if done and info.get('reason') == 'max_turns':
                    truncated = True
                    # Truncation 시 next value bootstrap
                    nx_x = self.env.get_manager_state()
                    nx_mask = self.env.get_candidate_mask()
                    nx_curr_z = int(self.env._node_zone_tensor[self.env.current_idx].item())
                    nx_goal_z = int(self.env._node_zone_tensor[self.env.goal_idx].item())
                    with torch.no_grad():
                        _, n_val, _ = self.manager.forward(
                            nx_x, self.env.zone_edge_index, 
                            torch.tensor(nx_curr_z, device=self.device), 
                            torch.tensor(nx_goal_z, device=self.device), 
                            nx_mask
                        )
                        next_val = n_val.item()

                self.buffer.store(
                    state_info={'x': x.cpu(), 'curr_z': curr_z, 'goal_z': goal_z, 'mask': candidate_mask.cpu()},
                    action=action,
                    reward=reward,
                    value=value.item() if isinstance(value, torch.Tensor) else value,
                    log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                    done=done,
                    truncated=truncated,
                    next_value=next_val
                )

            episode_rewards.append(ep_reward)
            episode_successes.append(1.0 if info.get('reason') == 'success' else 0.0)
            episode_turns.append(self.env.manager_turns)
            episode_worker_steps.append(self.env.total_worker_steps)
            episode_worker_successes.append(worker_successes / max(1, manager_steps))

            if pbar is not None:
                pbar.update(1)

        self.buffer.compute_gae(gamma=self.gamma, lam=self.gae_lambda)

        return {
            'mean_reward': np.mean(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'worker_success_rate': np.mean(episode_worker_successes),
            'mean_turns': np.mean(episode_turns),
            'mean_worker_steps': np.mean(episode_worker_steps),
        }

    def update(self) -> Dict:
        """PPO Clipped Objective로 Manager 정책 업데이트."""
        self.manager.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # PyG DataLoader용 데이터셋 구성
        dataset = []
        for i in range(len(self.buffer)):
            state = self.buffer.states[i]
            data = Data(x=state['x'], edge_index=self.env.zone_edge_index.cpu())
            data.curr_z = torch.tensor(state['curr_z'], dtype=torch.long)
            data.goal_z = torch.tensor(state['goal_z'], dtype=torch.long)
            data.mask = state['mask']
            data.action = torch.tensor(self.buffer.actions[i], dtype=torch.long)
            data.old_log_prob = torch.tensor(self.buffer.log_probs[i], dtype=torch.float32)
            data.advantage = self.buffer.advantages[i].clone().detach()
            data.target_return = self.buffer.returns[i].clone().detach()
            dataset.append(data)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for batch_data in loader:
                batch_data = batch_data.to(self.device)

                if (batch_data.mask == 0).all():
                    continue

                probs, value, logits = self.manager.forward(
                    batch_data.x, 
                    batch_data.edge_index, 
                    batch_data.curr_z, 
                    batch_data.goal_z, 
                    batch_data.mask,
                    batch_data.batch
                )

                # logits로 Categorical 분포 생성
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(batch_data.action)
                entropy = dist.entropy().mean()

                # PPO Clipped Surrogate Objective
                ratio = torch.exp(new_log_prob - batch_data.old_log_prob)
                surr1 = ratio * batch_data.advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_data.advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss (MSE)
                critic_loss = nn.functional.mse_loss(value.squeeze(-1) if value.dim() > 1 else value, batch_data.target_return)

                # 자동 alpha 적용
                # alpha = self.log_alpha.exp().detach()
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 🚨 [삭제] 아래 4줄의 alpha_loss 및 optimizer 로직을 반드시 전체 주석 처리하세요!
                # alpha_loss = self.log_alpha * (entropy.detach() - self.target_entropy)
                # self.alpha_optimizer.zero_grad()
                # alpha_loss.backward()
                # self.alpha_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        if n_updates == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}

        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def train(self, total_episodes: int) -> None:
        best_sr = -1.0
        t_start = time.time()

        pbar = tqdm(total=total_episodes, desc="Mgr-PPO", smoothing=0.1)

        episodes_done = 0
        epoch = 0
        while episodes_done < total_episodes:
            stats = self.collect_rollouts(pbar=pbar)
            update_info = self.update()

            t_elapsed = time.time() - t_start
            alpha_val = self.log_alpha.exp().item()
            
            pbar.set_postfix({
                'Rwd': f"{stats['mean_reward']:.2f}",
                'MgrS': f"{stats['success_rate']*100:.1f}%",
                'WkrS': f"{stats['worker_success_rate']*100:.1f}%",
                'Trn': f"{stats['mean_turns']:.1f}",
                'WStp': f"{stats['mean_worker_steps']:.1f}",
                'Loss': f"{update_info['actor_loss']:.2f}/{update_info['critic_loss']:.2f}",
                'Ent': f"{update_info['entropy']:.2f}",
                'α': f"{alpha_val:.4f}"
            })

            episodes_done += self.n_rollout_episodes
            epoch += 1

            sr = stats['success_rate'] * 100
            if sr > best_sr:
                best_sr = sr
                torch.save(self.manager.state_dict(), os.path.join(self.save_dir, 'best.pt'))

        pbar.close()
        torch.save(self.manager.state_dict(), os.path.join(self.save_dir, 'final.pt'))
