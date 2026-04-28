"""
HRL Phase 1: Zone-Guided Worker 학습 Trainer

Gradient Accumulation 방식으로 N개 에피소드의 gradient를 누적하여
대형 배치와 동일한 학습 효과를 달성. GATv2의 VRAM 제약을 회피하면서도
분산 감소(Variance Reduction) 효과를 확보함.
"""
import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm


class HRLWorkerTrainer:
    """HRL Phase 1: Gradient Accumulation 기반 Worker 학습."""

    def __init__(self, env, manager, worker, config):
        self.env = env
        self.manager = manager
        self.worker = worker
        self.config = config
        self.device = next(worker.parameters()).device
        
        # Gradient Accumulation 배치 크기 (POMO 대체)
        self.accum_batch = getattr(config, 'num_pomo', 16)

        # 정적 edge_index (단일 그래프, 양방향)
        edge_list = []
        for u, v in env.G.edges():
            edge_list.append((env.node_to_idx[u], env.node_to_idx[v]))
        edge_list_bidir = edge_list + [(v, u) for u, v in edge_list]
        self.edge_index = torch.tensor(
            edge_list_bidir, dtype=torch.long
        ).t().to(self.device)

        # 옵티마이저
        self.lr = getattr(config, 'lr', 3e-4)
        self.optimizer = optim.Adam(worker.parameters(), lr=self.lr)

        # 하이퍼파라미터
        self.gamma = 0.99
        self.max_grad_norm = 0.5

        # 저장 경로
        self.save_dir = getattr(config, 'save_dir', 'logs/rl_worker_stage')
        os.makedirs(self.save_dir, exist_ok=True)

        # Manager 동결
        if manager is not None:
            manager.eval()
            for p in manager.parameters():
                p.requires_grad_(False)

        # Worker 학습 활성화
        worker.train()
        for p in worker.parameters():
            p.requires_grad_(True)

    def _run_single_episode(self) -> dict:
        """단일 에피소드 실행 후 loss를 반환 (backward 수행하지 않음)."""
        state = self.env.reset(batch_size=1)  # [1, N, 4]
        state_np = state[0].numpy()  # [N, 4]
        
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state_np).to(self.device)
            mask_tensor = self.env.get_action_mask_batch()[0].to(self.device)  # [N]

            probs, value, _ = self.worker(
                state_tensor, self.edge_index,
                batch=None, neighbors_mask=mask_tensor,
            )

            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward_t, done_t, infos = self.env.step_batch(
                torch.tensor([action.item()])
            )

            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward_t[0].item())
            done = done_t[0].item()
            state_np = next_state[0].numpy()

        info = infos[0]

        # Returns 계산
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Loss 계산 (backward는 호출하지 않음)
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        for lp, val, ret in zip(log_probs, values, returns_t):
            advantage = ret.item() - val.item()
            policy_loss = policy_loss + (-lp * advantage)
            value_loss = value_loss + nn.functional.mse_loss(val, ret.detach())

        ep_loss = policy_loss + value_loss
        ep_reward = sum(rewards)
        is_success = 1.0 if info.get('reason') == 'success' else 0.0
        path_len = info.get('path_len', len(rewards))

        return {
            'loss': ep_loss,
            'reward': ep_reward,
            'success': is_success,
            'path_len': path_len,
        }

    def train(self, episodes: int) -> None:
        """메인 학습 루프 (Gradient Accumulation)."""
        from collections import deque

        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        best_success_rate = 0.0
        K = self.accum_batch  # Gradient Accumulation 크기

        # 런타임 설정 저장
        runtime_config = {
            'stage': 'worker (HRL Phase 1)',
            'lr': self.lr,
            'episodes': episodes,
            'accum_batch': K,
            'gamma': self.gamma,
            'save_dir': self.save_dir,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.save_dir, 'runtime_config.json'), 'w') as f:
            json.dump(runtime_config, f, indent=2)

        pbar = tqdm(range(1, episodes + 1), desc="HRL Worker", ncols=120)
        self.optimizer.zero_grad()
        accum_count = 0

        for ep in pbar:
            result = self._run_single_episode()

            # Gradient 누적 (K개의 에피소드마다 업데이트)
            scaled_loss = result['loss'] / K
            scaled_loss.backward()
            accum_count += 1

            # 로깅
            recent_rewards.append(result['reward'])
            recent_success.append(result['success'])
            recent_lengths.append(result['path_len'])

            # K개 누적 완료 시 업데이트
            if accum_count >= K:
                nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                accum_count = 0

            # 진행률 표시
            ema_succ = np.mean(recent_success) * 100
            pbar.set_postfix({
                'EMA': f'{ema_succ:5.1f}%',
                'Rw': f'{np.mean(recent_rewards):6.1f}',
                'Len': f'{np.mean(recent_lengths):5.1f}',
            })

            if ep % 200 == 0:
                pbar.write(
                    f'[EP {ep:5d}] EMA={ema_succ:5.1f}% | '
                    f'Rw={np.mean(recent_rewards):6.1f} | '
                    f'Len={np.mean(recent_lengths):5.1f}'
                )

            # Best 모델 저장
            if ema_succ > best_success_rate and len(recent_success) >= 50:
                best_success_rate = ema_succ
                self._save_checkpoint('best.pt', ep, ema_succ)

        # 잔여 gradient flush
        if accum_count > 0:
            nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        self._save_checkpoint('final.pt', episodes, ema_succ)
        pbar.write(f'✅ HRL Worker Phase 1 학습 완료! Best EMA Success: {best_success_rate:.1f}%')

    def _save_checkpoint(self, filename: str, ep: int, metric: float) -> None:
        payload = {
            'epoch': ep,
            'stage': 'worker_hrl_phase1',
            'worker_state': self.worker.state_dict(),
            'metric': metric,
            'metric_name': 'success_rate_ema',
        }
        torch.save(payload, os.path.join(self.save_dir, filename))
