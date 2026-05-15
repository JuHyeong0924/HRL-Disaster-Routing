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
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class RolloutBuffer:
    """PPO 학습용 경험 버퍼.

    에피소드 실행 중 수집된 (state_info, action, reward, value, log_prob, done)를
    저장하고, GAE Advantage를 계산한 뒤 미니배치로 제공한다.
    """

    def __init__(self) -> None:
        self.states: List[Dict] = []       # Manager 상태 정보 (x, current_idx, goal_idx, mask)
        self.actions: List[int] = []        # 선택된 서브골 인덱스
        self.rewards: List[float] = []
        self.values: List[float] = []       # Critic V(s) 추정값
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # GAE 계산 결과
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
    ) -> None:
        """한 스텝의 경험을 저장."""
        self.states.append(state_info)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """GAE(λ) Advantage 및 Returns 계산.

        Args:
            gamma: 할인율
            lam: GAE λ (편향-분산 트레이드오프)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = 0.0  # 에피소드 종료 후 가치 = 0

        # 역순 순회 (마지막 스텝부터)
        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                gae = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            next_value = self.values[t]

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32)

        # Advantage 정규화 (학습 안정성)
        if len(advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def clear(self) -> None:
        """버퍼 초기화."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)


class ManagerPPOTrainer:
    """PPO + GAE 기반 비자기회귀 Manager 학습기.

    Args:
        env: HRLClosedLoopEnv 인스턴스
        manager: ReactiveManager 모델
        config: 학습 설정 (argparse Namespace)
    """

    def __init__(self, env, manager: nn.Module, config) -> None:
        self.env = env
        self.manager = manager
        self.config = config
        self.device = next(manager.parameters()).device

        # 옵티마이저
        self.lr = getattr(config, 'lr', 3e-4)
        self.optimizer = optim.Adam(manager.parameters(), lr=self.lr)

        # PPO 하이퍼파라미터 (초기값, 추후 Ablation 튜닝 예정)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.clip_range = getattr(config, 'clip_range', 0.2)
        self.n_epochs = getattr(config, 'n_epochs', 4)
        self.value_coeff = getattr(config, 'value_coeff', 0.5)
        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.01)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)

        # 학습 설정
        self.n_rollout_episodes = getattr(config, 'num_pomo', 16)

        # 저장 경로
        self.save_dir = getattr(config, 'save_dir', 'logs/rl_manager_v2')
        os.makedirs(self.save_dir, exist_ok=True)

        # Rollout Buffer
        self.buffer = RolloutBuffer()

    def collect_rollouts(self) -> Dict:
        """N개 에피소드를 실행하여 Rollout Buffer에 경험 저장.

        Returns:
            rollout_stats: 에피소드 통계 (성공률, 평균 보상, 평균 턴 수 등)
        """
        self.manager.eval()
        self.buffer.clear()

        episode_rewards = []
        episode_successes = []
        episode_turns = []
        episode_worker_steps = []

        for _ in range(self.n_rollout_episodes):
            # 에피소드 초기화
            current_idx, goal_idx = self.env.reset()
            ep_reward = 0.0

            while not self.env.done:
                # Manager State 생성
                x = self.env.get_manager_state()
                candidate_mask = self.env.get_candidate_mask()

                # 유효한 후보가 없으면 에피소드 강제 종료
                if candidate_mask.sum() == 0:
                    self.buffer.store(
                        state_info={
                            'x': x.cpu(), 'current_idx': self.env.current_idx,
                            'goal_idx': self.env.goal_idx, 'mask': candidate_mask.cpu(),
                        },
                        action=self.env.current_idx, reward=-1.0,
                        value=0.0, log_prob=0.0, done=True,
                    )
                    self.env.done = True
                    break

                # Manager 행동 선택
                action, log_prob, value, entropy = self.manager.select_action(
                    x, self.env.edge_index, self.env.current_idx,
                    self.env.goal_idx, candidate_mask,
                )

                # 환경 스텝 (Worker 실행 포함)
                reward, done, info = self.env.step(action)
                ep_reward += reward

                # 버퍼에 경험 저장
                self.buffer.store(
                    state_info={
                        'x': x.cpu(),
                        'current_idx': info['start_idx'],
                        'goal_idx': self.env.goal_idx,
                        'mask': candidate_mask.cpu(),
                    },
                    action=action,
                    reward=reward,
                    value=value.item() if isinstance(value, torch.Tensor) else value,
                    log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                    done=done,
                )

            # 에피소드 통계
            episode_rewards.append(ep_reward)
            episode_successes.append(1.0 if info.get('reason') == 'success' else 0.0)
            episode_turns.append(self.env.manager_turns)
            episode_worker_steps.append(self.env.total_worker_steps)

        # GAE Advantage 계산
        self.buffer.compute_gae(gamma=self.gamma, lam=self.gae_lambda)

        return {
            'mean_reward': np.mean(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'mean_turns': np.mean(episode_turns),
            'mean_worker_steps': np.mean(episode_worker_steps),
        }

    def update(self) -> Dict:
        """PPO Clipped Objective로 정책 업데이트.

        Returns:
            update_stats: 손실 통계 (actor_loss, critic_loss, entropy 등)
        """
        self.manager.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # n_epochs 반복 (동일 Rollout 재사용)
        for epoch in range(self.n_epochs):
            # 전체 버퍼를 순회 (미니배치 분할 없이 — 데이터가 크지 않으므로)
            for i in range(len(self.buffer)):
                state_info = self.buffer.states[i]
                old_action = self.buffer.actions[i]
                old_log_prob = self.buffer.log_probs[i]
                advantage = self.buffer.advantages[i].to(self.device)
                target_return = self.buffer.returns[i].to(self.device)

                # 현재 정책으로 재평가
                x = state_info['x'].to(self.device)
                current_idx = state_info['current_idx']
                goal_idx = state_info['goal_idx']
                mask = state_info['mask'].to(self.device)

                probs, value, logits = self.manager(
                    x, self.env.edge_index, current_idx, goal_idx, mask,
                )

                # 유효하지 않은 상태 건너뛰기
                if (mask == 0).all():
                    continue

                dist = torch.distributions.Categorical(probs)
                action_tensor = torch.tensor(old_action, device=self.device)
                new_log_prob = dist.log_prob(action_tensor)
                entropy = dist.entropy()

                # PPO Clipped Loss
                old_lp = torch.tensor(old_log_prob, device=self.device)
                ratio = torch.exp(new_log_prob - old_lp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantage
                actor_loss = -torch.min(surr1, surr2)

                # Critic Loss
                critic_loss = nn.functional.mse_loss(value.squeeze(), target_return)

                # 총 손실
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def train(self, episodes: int) -> None:
        """메인 학습 루프.

        Args:
            episodes: 총 에피소드 수 (iterations = episodes / n_rollout_episodes)
        """
        total_iterations = max(1, episodes // self.n_rollout_episodes)

        # 런타임 설정 저장
        runtime_config = {
            'stage': 'manager_v2 (PPO + PBRS)',
            'lr': self.lr,
            'episodes': episodes,
            'n_rollout_episodes': self.n_rollout_episodes,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'n_epochs': self.n_epochs,
            'k_hop': self.env.k_hop,
            'c_max': self.env.c_max,
            'max_manager_turns': self.env.max_manager_turns,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.save_dir, 'runtime_config.json'), 'w') as f:
            json.dump(runtime_config, f, indent=2)

        # 학습 로그
        recent_rewards = deque(maxlen=50)
        recent_success = deque(maxlen=50)
        best_success_rate = 0.0
        history = {'rewards': [], 'success_rates': [], 'turns': [], 'losses': []}

        pbar = tqdm(range(1, total_iterations + 1), desc="MgrV2-PPO", ncols=120)

        for iteration in pbar:
            # 1. Rollout 수집
            rollout_stats = self.collect_rollouts()

            # 2. PPO 업데이트
            update_stats = self.update()

            # 3. 로깅
            recent_rewards.append(rollout_stats['mean_reward'])
            recent_success.append(rollout_stats['success_rate'])

            ema_reward = np.mean(recent_rewards)
            ema_success = np.mean(recent_success) * 100

            history['rewards'].append(rollout_stats['mean_reward'])
            history['success_rates'].append(rollout_stats['success_rate'])
            history['turns'].append(rollout_stats['mean_turns'])
            history['losses'].append(update_stats['actor_loss'])

            pbar.set_postfix({
                'SR': f'{ema_success:5.1f}%',
                'Rw': f'{ema_reward:6.2f}',
                'Turns': f'{rollout_stats["mean_turns"]:.1f}',
                'ALoss': f'{update_stats["actor_loss"]:.3f}',
                'Ent': f'{update_stats["entropy"]:.3f}',
            })

            # 20 iteration마다 상세 로그
            if iteration % 20 == 0:
                ep_count = iteration * self.n_rollout_episodes
                pbar.write(
                    f'[Iter {iteration:4d}/{total_iterations} | Ep {ep_count}] '
                    f'SR={ema_success:5.1f}% | Rw={ema_reward:6.2f} | '
                    f'Turns={rollout_stats["mean_turns"]:.1f} | '
                    f'WkrSteps={rollout_stats["mean_worker_steps"]:.1f} | '
                    f'ALoss={update_stats["actor_loss"]:.4f} | '
                    f'CLoss={update_stats["critic_loss"]:.4f}'
                )

            # Best 모델 저장 (최소 5 iteration 이후)
            if ema_success > best_success_rate and len(recent_success) >= 5:
                best_success_rate = ema_success
                self._save_checkpoint('best.pt', iteration, ema_success)

        # 최종 모델 저장
        self._save_checkpoint('final.pt', total_iterations, ema_success)

        # 학습 곡선 저장
        self._save_learning_curve(history)

        pbar.write(
            f'✅ Manager v2 (PPO) 학습 완료! '
            f'Best Success Rate: {best_success_rate:.1f}%'
        )

    def _save_checkpoint(self, filename: str, iteration: int, metric: float) -> None:
        """모델 체크포인트 저장."""
        payload = {
            'iteration': iteration,
            'stage': 'manager_v2_ppo',
            'manager_state': self.manager.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metric': metric,
            'metric_name': 'success_rate_ema',
        }
        torch.save(payload, os.path.join(self.save_dir, filename))

    def _save_learning_curve(self, history: Dict) -> None:
        """학습 곡선 그래프 저장."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Manager v2 (PPO + PBRS) Learning Curve', fontsize=14)

            axes[0, 0].plot(history['rewards'], alpha=0.7)
            axes[0, 0].set_title('Mean Reward per Iteration')
            axes[0, 0].set_xlabel('Iteration')

            axes[0, 1].plot(
                [s * 100 for s in history['success_rates']], alpha=0.7, color='green'
            )
            axes[0, 1].set_title('Success Rate (%)')
            axes[0, 1].set_xlabel('Iteration')

            axes[1, 0].plot(history['turns'], alpha=0.7, color='orange')
            axes[1, 0].set_title('Mean Manager Turns')
            axes[1, 0].set_xlabel('Iteration')

            axes[1, 1].plot(history['losses'], alpha=0.7, color='red')
            axes[1, 1].set_title('Actor Loss')
            axes[1, 1].set_xlabel('Iteration')

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, 'learning_curve.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f'📈 Learning curve saved to {save_path}')
        except Exception as e:
            print(f'⚠️ Learning curve 저장 실패: {e}')
