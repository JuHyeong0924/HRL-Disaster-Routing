"""
HRL Phase 1: Zone-Guided Worker 학습 Trainer

Gradient Accumulation 방식으로 N개 에피소드의 gradient를 누적하여
대형 배치와 동일한 학습 효과를 달성. GATv2의 VRAM 제약을 회피하면서도
분산 감소(Variance Reduction) 효과를 확보함.

Ablation 플래그:
- use_gae: GAE(λ) 적용 여부 (기본 False → Monte Carlo)
- entropy_coeff: Entropy Bonus 계수 (기본 0.0 → 없음)
- use_cosine_lr: Cosine LR Scheduler 사용 여부 (기본 False)
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
    """HRL Phase 1: Gradient Accumulation 기반 Worker 학습.
    
    Ablation 실험을 위해 GAE, Entropy, LR Scheduler를 플래그로 제어.
    """

    def __init__(self, env, manager, worker, config):
        self.env = env
        self.manager = manager
        self.worker = worker
        self.config = config
        self.device = next(worker.parameters()).device
        
        # Gradient Accumulation 배치 크기
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
        
        # [P1] GAE 및 Entropy Ablation 플래그
        self.use_gae = getattr(config, 'use_gae', False)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.0)
        
        # [P2] Cosine LR Scheduler
        self.use_cosine_lr = getattr(config, 'use_cosine_lr', False)
        self._episodes_total = getattr(config, 'episodes', 5000)
        if self.use_cosine_lr:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self._episodes_total, eta_min=1e-5
            )
        else:
            self.scheduler = None

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

    def _compute_gae(self, rewards: list, values: list) -> torch.Tensor:
        """GAE(λ) 계산. 분산-편향 트레이드오프 제어."""
        advantages = []
        gae = 0.0
        next_value = 0.0  # 에피소드 종료이므로 terminal value = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            v_scalar = v.item()
            delta = r + self.gamma * next_value - v_scalar
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = v_scalar
            
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def _run_batch_episodes(self, batch_size: int) -> list:
        """B개 에피소드를 환경에서 동시에(병렬) 실행하고 각각의 결과를 반환.

        HRLZoneEnv의 reset(batch_size=B) + step_batch(actions[B])를 활용하여
        B개의 경로 탐색을 하나의 반복문으로 동시에 처리함.
        POMO(Policy Optimization with Multiple Optima) 병렬화와 동일한 효과.
        """
        state = self.env.reset(batch_size=batch_size)  # [B, N, 4]
        B = batch_size
        N = state.shape[1]

        # 배치별 궤적 저장소
        traj_log_probs  = [[] for _ in range(B)]  # 배치 b: 로그 확률 리스트
        traj_values     = [[] for _ in range(B)]  # 배치 b: 가치 추정 리스트
        traj_rewards    = [[] for _ in range(B)]  # 배치 b: 보상 리스트
        traj_entropies  = [[] for _ in range(B)]  # 배치 b: 엔트로피 리스트
        done_flags      = [False] * B             # 종료 여부

        # 환경 공유 edge_index (동일 그래프, batch 인덱스로 구분)
        # batch 텐서: [N*B] — 각 노드가 어느 그래프(에피소드)에 속하는지
        batch_idx = torch.arange(B, device=self.device).repeat_interleave(N)

        # 배치 edge_index: 각 에피소드의 엣지 인덱스를 i*N만큼 offset
        batch_ei_list = []
        for b in range(B):
            batch_ei_list.append(self.edge_index + b * N)
        batch_ei = torch.cat(batch_ei_list, dim=1)  # [2, E*B]

        # 종료 시점의 info를 배치별로 보존 (step_batch가 매번 infos를 새로 생성하므로)
        final_infos = [{} for _ in range(B)]

        while not all(done_flags):
            # 현재 활성(미종료) 에피소드 인덱스
            active = [b for b in range(B) if not done_flags[b]]

            # 활성 에피소드만 묶어서 GNN 한 번에 처리
            active_states = torch.stack(
                [state[b].to(self.device) for b in active]
            )  # [|active|, N, 4]
            active_masks = torch.stack(
                [self.env.get_action_mask_batch()[b].to(self.device) for b in active]
            )  # [|active|, N]

            x_flat = active_states.view(-1, active_states.shape[-1])   # [|A|*N, 4]
            mask_flat = active_masks.view(-1)                           # [|A|*N]
            A = len(active)
            ai = torch.arange(A, device=self.device).repeat_interleave(N)
            aei = torch.cat([self.edge_index + i * N for i in range(A)], dim=1)

            probs_all, values_all, _ = self.worker(
                x_flat, aei, batch=ai, neighbors_mask=mask_flat,
            )  # probs_all: [|A|*N], values_all: [|A|, 1]

            # 각 활성 에피소드별 행동 샘플링
            actions = []
            for i, b in enumerate(active):
                node_probs = probs_all[i * N: (i + 1) * N]  # [N]
                node_val   = values_all[i].squeeze()         # scalar
                dist = Categorical(node_probs)
                action = dist.sample()
                lp = dist.log_prob(action)
                ent = dist.entropy()

                traj_log_probs[b].append(lp)
                traj_values[b].append(node_val)
                traj_entropies[b].append(ent)
                actions.append(action.item())

            # 종료된 에피소드는 dummy 0 행동 전달 (환경 내부에서 무시됨)
            all_actions = []
            ai_ptr = 0
            for b in range(B):
                if not done_flags[b]:
                    all_actions.append(actions[ai_ptr])
                    ai_ptr += 1
                else:
                    all_actions.append(0)  # dummy

            next_state, reward_t, done_t, infos = self.env.step_batch(
                torch.tensor(all_actions)
            )

            for b in range(B):
                if not done_flags[b]:
                    traj_rewards[b].append(reward_t[b].item())
                    if done_t[b].item():
                        done_flags[b] = True
                        final_infos[b] = infos[b]  # 종료 시점의 info 보존

            state = next_state

        # 배치별 loss 계산
        results = []
        for b in range(B):
            rewards_b  = traj_rewards[b]
            values_b   = traj_values[b]
            log_probs_b = traj_log_probs[b]
            entropies_b = traj_entropies[b]

            if self.use_gae:
                advantages = self._compute_gae(rewards_b, values_b)
                values_t = torch.stack(values_b).detach()
                returns_t = advantages + values_t
            else:
                returns_list, R = [], 0.0
                for r in reversed(rewards_b):
                    R = r + self.gamma * R
                    returns_list.insert(0, R)
                returns_t = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
                advantages = returns_t - torch.stack(values_b).detach()

            policy_loss = torch.tensor(0.0, device=self.device)
            value_loss  = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            for lp, val, ret, adv, ent in zip(log_probs_b, values_b, returns_t, advantages, entropies_b):
                policy_loss  = policy_loss  + (-lp * adv.detach())
                value_loss   = value_loss   + nn.functional.mse_loss(val, ret.detach())
                entropy_loss = entropy_loss + ent

            ep_loss = policy_loss + value_loss - self.entropy_coeff * entropy_loss
            info_b  = final_infos[b]  # 종료 시점에 보존한 info 사용

            results.append({
                'loss':     ep_loss,
                'reward':   sum(rewards_b),
                'success':  1.0 if info_b.get('reason') == 'success' else 0.0,
                'path_len': info_b.get('path_len', len(rewards_b)),
            })

        return results

    def train(self, episodes: int) -> None:
        """메인 학습 루프 (POMO 배치 병렬 실행).

        매 스텝마다 num_pomo(K)개의 에피소드를 동시에 실행하고
        K개의 gradient를 평균 합산하여 1회 업데이트.
        총 업데이트 횟수 = episodes // K 회.
        """
        from collections import deque

        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        best_success_rate = 0.0
        K = self.accum_batch  # POMO 배치 크기 (동시 에피소드 수)

        # 런타임 설정 저장
        ablation_flags = {
            'zone_progress_reward': getattr(self.env, 'zone_progress_reward', False),
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda if self.use_gae else None,
            'entropy_coeff': self.entropy_coeff,
            'use_cosine_lr': self.use_cosine_lr,
            'pomo_batch_size': K,
        }
        runtime_config = {
            'stage': 'worker (HRL Phase 1)',
            'lr': self.lr,
            'episodes': episodes,
            'pomo_batch': K,
            'gamma': self.gamma,
            'ablation': ablation_flags,
            'save_dir': self.save_dir,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.save_dir, 'runtime_config.json'), 'w') as f:
            json.dump(runtime_config, f, indent=2)

        # 총 업데이트 스텝 수 (K개 에피소드 단위)
        total_steps = max(1, episodes // K)
        ep_counter = 0  # 완료된 에피소드 수 추적
        ema_succ = 0.0

        pbar = tqdm(range(1, total_steps + 1), desc="Phase 1", ncols=120)

        for step in pbar:
            self.optimizer.zero_grad()

            # K개 에피소드 동시 병렬 실행
            batch_results = self._run_batch_episodes(batch_size=K)

            # 배치 내 모든 에피소드의 loss를 평균 합산하여 역전파
            total_loss = torch.stack([r['loss'] for r in batch_results]).mean()
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # [P2] LR Scheduler Step (매 업데이트마다)
            if self.scheduler is not None:
                self.scheduler.step()

            # 로깅 (배치 내 모든 에피소드 결과 반영)
            for r in batch_results:
                recent_rewards.append(r['reward'])
                recent_success.append(r['success'])
                recent_lengths.append(r['path_len'])

            ep_counter += K
            ema_succ = np.mean(recent_success) * 100
            pbar.set_postfix({
                'Step': f'{step}/{total_steps}',
                'SR': f'{ema_succ:5.1f}%',
                'Rw': f'{np.mean(recent_rewards):6.1f}',
                'Len': f'{np.mean(recent_lengths):5.1f}',
            })

            # 100 스텝마다 로그 출력
            if step % 100 == 0:
                pbar.write(
                    f'[Step {step:4d}/{total_steps}] SR={ema_succ:5.1f}% | '
                    f'Rw={np.mean(recent_rewards):6.1f} | '
                    f'Len={np.mean(recent_lengths):5.1f}'
                )

            # Best 모델 저장
            if ema_succ > best_success_rate and len(recent_success) >= 50:
                best_success_rate = ema_succ
                self._save_checkpoint('best.pt', ep_counter, ema_succ)

        self._save_checkpoint('final.pt', ep_counter, ema_succ)
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
