import csv
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler  # AMP: VRAM 절감용 혼합 정밀도
from tqdm import tqdm

class DOMOTrainer: 
    """
    Disaster Optimization with Multiple Optima (DOMO) - Vectorized Implementation
    Refactored 2026-02-05: Fully Vectorized Execution for Parallel Trajectories
    v2 2026-02-11: 6가지 핵심 버그 수정 (Gradient, Advantage, Loss Scale, Curriculum 등)
    """
    def __init__(self, env, manager, worker, config):
        self.env = env
        self.manager = manager
        self.worker = worker
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.save_dir = getattr(config, 'save_dir', "logs/rl_finetune")
        # [Diagnostic] 디버그 모드 플래그
        self.debug_mode = getattr(config, 'debug', False)
        self.debug_interval = int(getattr(config, 'debug_interval', 200))
        self.stagnation_patience = int(getattr(config, 'stagnation_patience', 24))
        self._collect_debug_this_episode = False
        self._debug_window = []
        self._debug_csv_fields = None
        self._debug_log_path = None
        self._debug_csv_path = None
        self._debug_jsonl_path = None
        self.reward_cfg = {
            'POTENTIAL_SCALE': 6.5,
            'GAMMA_PBRS': 0.99,
            'SUBGOAL_BASE': 0.5,
            'SUBGOAL_SCALE': 1.5,
            'OPTIMALITY_BONUS': 0.5,
            'GOAL_REWARD': 40.0,
            'EFFICIENCY_MAX': 8.0,
            'MILESTONE_25': 0.75,
            'MILESTONE_50': 1.5,
            'MILESTONE_75': 3.0,
            'FAIL_PENALTY': -20.0,
            'LOOP_PENALTY_SCALE': 0.1,
            'EXPLORATION_BONUS': 0.0,
            'BASE_STEP_PENALTY': -0.02,
            'TIME_PRESSURE_SCALE': 2.0,
            'PLAN_DENSITY_TARGET': 0.20,
            'PLAN_DENSITY_WEIGHT': 7.0,
            'PLAN_CORRIDOR_TARGET': 0.6,
            'PLAN_CORRIDOR_WEIGHT': 1.5,
            'TARGET_SEGMENT_HOPS': 4.5,
            'PLAN_COUNT_BAND': 1,
            'PLAN_COUNT_UNDER_PENALTY': 1.0,
            'PLAN_COUNT_OVER_PENALTY': 0.2,
            'ANCHOR_HOP_PENALTY': 0.15,
            'ANCHOR_NEAR_BONUS': 0.50,
            'FIRST_ANCHOR_PENALTY': 0.30,
            'SPACING_PENALTY_SCALE': 0.80,
            'MONOTONIC_PENALTY_SCALE': 0.50,
            'EMPTY_PLAN_PENALTY': -2.0,
            'PLAN_ADJUST_MIN': -6.0,
            'PLAN_ADJUST_MAX': 4.0,
            'CHECKPOINT_MAX_REWARD': 4.0,
            'VISIT_LOGIT_PENALTY': 0.35,
            'RECENT_NODE_PENALTY': 1.0,
            'RECENT_WINDOW': 4,
            'STAGNATION_PATIENCE': self.stagnation_patience,
        }
        self.critic_cfg = {
            'return_momentum': 0.99,
            'return_eps': 1e-6,
            'huber_beta': 1.0,
        }
        self.mgr_max_grad_norm = float(getattr(config, 'mgr_max_grad_norm', 5.0))
        self.wkr_max_grad_norm = float(getattr(config, 'wkr_max_grad_norm', 5.0))
        self.mgr_aux_start = float(getattr(config, 'mgr_aux_start', 0.20))
        self.mgr_aux_end = float(getattr(config, 'mgr_aux_end', 0.15))
        self.wkr_aux_start = float(getattr(config, 'wkr_aux_start', 0.20))
        self.wkr_aux_end = float(getattr(config, 'wkr_aux_end', 0.05))
        self.mgr_lr_scale = float(getattr(config, 'mgr_lr_scale', 0.7))
        self.mgr_eta_min_scale = float(getattr(config, 'mgr_eta_min_scale', 0.1))
        self.stage = str(getattr(config, 'stage', 'joint'))
        self.ret_ema_mean = None
        self.ret_ema_std = None
        self._handoff_target_warned = False
        # [Refactor: Task 4] Goal에서 멀어지는 행동 페널티 (Post-Handoff 시 3.0으로 강화)
        self.goal_regression_penalty_large = float(getattr(config, "goal_regression_penalty_large", 0.35))
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # AMP: Mixed Precision 설정 (VRAM 30~40% 절감)
        self.use_amp = torch.cuda.is_available()
        self.grad_scaler = GradScaler('cuda', enabled=self.use_amp)

        self.manager.to(self.device).train()
        self.worker.to(self.device).train()
        
        # [Fix: Adaptive Fine-tuning] Worker 초기엔 완전히 동결하여 Manager 폼 교정에 집중
        for p in self.worker.parameters():
            p.requires_grad_(False)
        
        # Manager에만 초기부터 별도 LR 적용
        self.mgr_opt = optim.Adam(self.manager.parameters(), lr=config.lr * self.mgr_lr_scale)
        self.wkr_opt = None  # 추후 특정 진척도에서 언프리즈 후 할당
        
        self.num_pomo = config.num_pomo # Batch Size for Parallel Simulation
        
        # LR Schedulers will be initialized in train() with total episodes
        self.mgr_scheduler = None
        self.wkr_scheduler = None
        
        # [Fix: Adaptive DAgger]
        self.manager_entropy_ema = 0.0
        self.manager_clip_hit_ema = 0.0
        self.dagger_cooldown = 0

    def _should_collect_debug(self, ep, episodes):
        if not self.debug_mode:
            return False
        if ep == 0 or ep == episodes - 1:
            return True
        return (ep % self.debug_interval) == 0
        
    def train(self, episodes):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.debug_mode:
            self._init_debug_outputs()
        
        # [Fix 2026-03-15] 단조 감소 스케줄러 (WarmRestart는 주기적 LR 스파이크로 지식 파괴)
        self.mgr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.mgr_opt, T_max=episodes, eta_min=self.config.lr * self.mgr_eta_min_scale
        )
        self.wkr_scheduler = None  # 언프리즈 시점에 설정
        
        avg_reward = 0
        wkr_unfreeze_progress = 0.50  # 전체 에피소드의 50% 시점에서 동결 해제
        wkr_unfreeze_ep = int(episodes * wkr_unfreeze_progress)
        pbar = tqdm(range(episodes), desc="RL", ncols=100)
        
        # 학습 곡선 기록용
        rl_history = {
            'rewards': [], 'losses': [], 'path_lengths': [],
            'success_rates': [],  # 에피소드별 성공률 (이동 평균)
        }
        success_ema = 0  # Exponential Moving Average
        
        for ep in pbar:
            # [Fix: Micro Fine-Tuning] 진척도 50% 시점에 Worker 잠금 해제
            if ep == wkr_unfreeze_ep:
                # [Fix 2026-04-20] OOM 방지: Worker Unfreeze 전 메모리 확보
                # Why: Worker의 LSTM/Scorer/Critic 역전파에 필요한 activations이
                #      매 스텝마다 누적되어 기존 48 POMO로는 24GB GPU 메모리를 초과함.
                self._save_unified_checkpoint(
                    "pre_unfreeze.pt", ep,
                    metric=success_ema, metric_name="success_ema",
                )
                torch.cuda.empty_cache()
                # [Fix 2026-04-20] POMO 16으로 축소: 32에서도 OOM 발생
                # Why: Worker LSTM (최대 170스텝) activations + Manager Aux CE가
                #      32 POMO에서도 24GB를 초과함. 16이면 충분한 여유 확보.
                prev_pomo = self.num_pomo
                self.num_pomo = 24
                pbar.write(
                    f"[Ep {ep}] Worker Unfreeze: num_pomo {prev_pomo} → {self.num_pomo}, "
                    f"CUDA cache cleared, checkpoint saved."
                )
                for p in self.worker.parameters():
                    p.requires_grad_(True)
                wkr_lr_ft = 1e-5  # 마이크로 파인튜닝 학습률
                self.wkr_opt = optim.Adam(self.worker.parameters(), lr=wkr_lr_ft)
                self.wkr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.wkr_opt, T_max=max(episodes - ep, 1), eta_min=wkr_lr_ft * 0.1
                )
                
            self._collect_debug_this_episode = self._should_collect_debug(ep, episodes)
            self._last_debug_tensors = None

            # [Fix 5] Curriculum Learning 적용
            curriculum_ratio = min(1.0, ep / (episodes * 0.8))
            self.env.set_curriculum_ratio(curriculum_ratio)
            
            # 1. Reset Env (Vectorized Batch Reset)
            self.env.reset(batch_size=self.num_pomo, sync_problem=True)
            
            s0 = self.env.current_node[0].item()
            g0 = self.env.target_node[0].item()
            
            # 2. Manager Plan Generation (Vectorized)
            x_mgr_in = self.env.pyg_data.x[:, :4] 
            edge_index = self.env.pyg_data.edge_index 
            batch_vec = self.env.pyg_data.batch
            
            temperature = max(0.5, 1.5 - curriculum_ratio)
            # edge_attr 슬라이싱: Env 9D → 모델 5D [length, damage, is_closed, is_danger, speed]
            ea = self.env.pyg_data.edge_attr[:, 0:1]  # Phase 1: length만 사용
            # AMP: Manager generate는 autocast로 감싸 메모리 절감
            with autocast('cuda', enabled=self.use_amp):
                sequences, _ = self.manager.generate(
                    x_mgr_in,
                    edge_index,
                    batch_vec,
                    max_len=50,  # [성능 고도화] 대형 맵에서도 끝까지 서브골 생성 가능하도록 확장 (20→50)
                    temperature=temperature,
                    apsp_matrix=self.env.hop_matrix,
                    node_positions=self.env.pos_tensor,
                    edge_attr=ea,
                )
            
            # 3. Vectorized Execution
            x_pos = self.env.pyg_data.x[:, :2].clone()
            plan_diag = self._compute_plan_reward_adjustment(s0, g0, sequences)
            
            # [Fix 1] torch.no_grad() 제거 - Worker gradient 완전 복원
            log_probs_sum, rewards, norm_returns, path_lengths, val_loss_sum, entropy_sum, worker_aux_ce_loss = self.execute_batch_plan(
                s0,
                g0,
                sequences,
                x_pos,
                plan_adjustment=plan_diag['plan_adjustment'],
                plan_diag=plan_diag,
                detach_spatial=True,
                collect_worker_aux=True,
            )
            
            # [Fix 2026-03] Advantage 계산 (Critic Value Base)
            # POMO Baseline(rewards.mean())은 배치 크기가 4로 줄어들어 분산 노이즈가 너무 심함.
            # 따라서 Worker Critic이 예측한 초기 Step Value를 빼주는 정석 Actor-Critic 사용
            traj_values = getattr(self, "_last_initial_values", None)
            if traj_values is not None:
                baseline = traj_values.detach()
            else:
                baseline = norm_returns.mean() # Fallback
            advantages = norm_returns - baseline
            
            # [Fix 2026-03-15] Advantage 정규화 (분산 제어)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Manager Loss
            max_len = sequences.size(1)
            target_seq = sequences.to(self.device)
            
            # [Inductive] Fetch Target Embeddings for Teacher Forcing
            # sequences contains indices [0~N-1] and EOS [N] and PAD/Dummy
            # We need to gather from node_enc(x_mgr_in) and eos_emb
            
            # AMP: Manager loss 계산을 autocast로 감싸 VRAM 절감
            with autocast('cuda', enabled=self.use_amp):
                # 1. Base Node Embeddings (GATv2Conv)
                node_emb_all = self.manager.topology_enc(x_mgr_in, edge_index, edge_attr=ea) # [B*N, H]
                from torch_geometric.utils import to_dense_batch
                node_emb_dense, mask_dense = to_dense_batch(node_emb_all, batch_vec) # [B, N, H]
                
                # 2. Append EOS Embedding
                B, N, H = node_emb_dense.shape
                eos_node_emb = self.manager.eos_token_emb.expand(B, 1, H)
                full_ref_embs = torch.cat([node_emb_dense, eos_node_emb], dim=1) # [B, N+1, H]
                
                # 3. Gather teacher-forcing embeddings for sampled rollout sequence
                target_seq_emb = self._gather_manager_teacher_embeddings(full_ref_embs, target_seq, N)
                
                # Forward
                mgr_logits = self.manager(x_mgr_in, edge_index, batch_vec, target_seq_emb, edge_attr=ea)
                mgr_logits = mgr_logits[:, :-1, :].contiguous()
                # [Fix] Logit clamping: NLL 폭발 방지 (logit이 ±20 이상이면 CE가 수십~수백으로 폭주)
                mgr_logits = mgr_logits.clamp(-20.0, 20.0)
            
            # AMP 경고: CE Loss는 float32에서 수행 (수치 안정성)
            mgr_logits_f32 = mgr_logits.float()
            # [Fix 3] Log-Prob 정규화
            # vocab_size is dynamic (N+1)
            vocab_size = mgr_logits_f32.size(-1) 
            
            mgr_nll = F.cross_entropy(
                mgr_logits_f32.view(-1, vocab_size), 
                target_seq.view(-1), 
                reduction='none', 
                ignore_index=self.manager.PAD_TOKEN
            )
            mgr_nll = mgr_nll.view(self.num_pomo, max_len)
            
            valid_mask = (target_seq != self.manager.PAD_TOKEN).float()
            valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
            mgr_log_probs = -(mgr_nll * valid_mask).sum(dim=1) / valid_counts

            reference_anchor_nodes = plan_diag['reference_anchor_nodes'].to(self.device)
            ref_token_count = int(plan_diag['reference_token_count'])
            aux_seq_len = max(max_len, ref_token_count + 1)
            aux_target_seq = torch.full(
                (self.num_pomo, aux_seq_len),
                self.manager.PAD_TOKEN,
                dtype=torch.long,
                device=self.device,
            )
            if ref_token_count > 0:
                aux_target_seq[:, :ref_token_count] = reference_anchor_nodes.unsqueeze(0).expand(self.num_pomo, -1)
            if ref_token_count < aux_seq_len:
                aux_target_seq[:, ref_token_count] = N

            # AMP: Aux forward도 autocast (Pointer Network NaN은 manager.py 내부에서 FP32 격리로 해결)
            with autocast('cuda', enabled=self.use_amp):
                aux_target_seq_emb = self._gather_manager_teacher_embeddings(full_ref_embs, aux_target_seq, N)
                aux_logits = self.manager(x_mgr_in, edge_index, batch_vec, aux_target_seq_emb, edge_attr=ea)
                aux_logits = aux_logits[:, :-1, :].contiguous()
                aux_logits = aux_logits.clamp(-20.0, 20.0)  # [Fix] Aux CE도 logit clamping

            # Aux CE도 float32에서 수행 (수치 안정성)
            aux_logits_f32 = aux_logits.float()
            aux_vocab_size = aux_logits_f32.size(-1)
            aux_nll = F.cross_entropy(
                aux_logits_f32.view(-1, aux_vocab_size),
                aux_target_seq.view(-1),
                reduction='none',
                ignore_index=self.manager.PAD_TOKEN,
            )
            aux_nll = aux_nll.view(self.num_pomo, aux_seq_len)
            aux_valid_mask = (aux_target_seq != self.manager.PAD_TOKEN).float()
            aux_valid_counts = aux_valid_mask.sum(dim=1).clamp(min=1.0)
            mgr_aux_ce_loss = ((aux_nll * aux_valid_mask).sum(dim=1) / aux_valid_counts).mean()
            
            # [Fix: Adaptive DAgger] EMA 기반으로 망각 징조가 보이면 강제로 과외 스케줄 당김
            if self.dagger_cooldown == 0 and (self.manager_clip_hit_ema > 0.8 or self.manager_entropy_ema > 2.5) and ep > 100:
                self.dagger_cooldown = max(10, int(episodes * 0.05))  # 전체 에피소드의 5% 쿨다운
                
            base_mgr_aux = self._manager_aux_weight(ep, episodes)
            if self.dagger_cooldown > 0:
                cooldown_ratio = self.dagger_cooldown / max(1.0, float(int(episodes * 0.05)))
                mgr_aux_weight = (0.90 * cooldown_ratio) + (base_mgr_aux * (1.0 - cooldown_ratio))
                self.dagger_cooldown -= 1
            else:
                mgr_aux_weight = base_mgr_aux
                
            wkr_aux_weight = self._worker_aux_weight(ep, episodes)
            
            # Worker trajectory는 길수록 log-prob가 커지므로 sqrt(step)로만 완만하게 정규화
            wkr_log_probs = log_probs_sum / path_lengths.float().clamp(min=1.0).sqrt()
            
            # Combined Policy Loss
            total_log_probs = mgr_log_probs + wkr_log_probs
            policy_loss = -(advantages.detach() * total_log_probs).mean()
            
            # Critic Loss
            critic_loss = 0.5 * val_loss_sum.mean()
            
            mgr_probs = F.softmax(mgr_logits_f32, dim=-1)
            mgr_entropy = -(mgr_probs * torch.log(mgr_probs + 1e-8)).sum(dim=-1)
            entropy_bonus = -0.03 * (mgr_entropy * valid_mask).mean()
            
            # [Fix 2026-03] Worker Entropy Bonus 추가
            # execute_batch_plan에서 반환받는 entropy_sum 활용
            entropy_bonus = entropy_bonus - 0.01 * (entropy_sum.mean() / path_lengths.float().clamp(min=1.0).mean())
            
            # [성능 고도화] Entropy Decay: 커리큘럼 80% 이후 엔트로피 보너스를 0으로 점진 감소
            # Why: 학습 후반부에 탐색(Exploration)을 줄이고 최적 경로 집중(Exploitation)으로 전환
            if curriculum_ratio >= 0.8:
                entropy_decay = max(0.0, 1.0 - (curriculum_ratio - 0.8) / 0.2)
            else:
                entropy_decay = 1.0
            entropy_bonus = entropy_bonus * entropy_decay
            
            loss = (
                policy_loss
                + critic_loss
                + entropy_bonus
                + mgr_aux_weight * mgr_aux_ce_loss
                + wkr_aux_weight * worker_aux_ce_loss
            )
            
            self.mgr_opt.zero_grad()
            if self.wkr_opt is not None:
                self.wkr_opt.zero_grad()
                
            if loss.requires_grad:
                # AMP: GradScaler로 loss를 스케일링하여 backward 수행
                self.grad_scaler.scale(loss).backward()
                
                # 각 optimizer에 실제 gradient가 존재하는지 확인
                # (loss 연산 그래프에 참여하지 않은 optimizer는 inf 체크가 등록되지 않아 step() 시 AssertionError 발생)
                mgr_has_grad = any(p.grad is not None for p in self.manager.parameters())
                wkr_has_grad = getattr(self, 'wkr_opt', None) is not None and any(p.grad is not None for p in self.worker.parameters())
                
                # AMP: unscale 후 gradient clipping (스케일링된 상태에서 clip하면 안 됨)
                if mgr_has_grad:
                    self.grad_scaler.unscale_(self.mgr_opt)
                if wkr_has_grad:
                    self.grad_scaler.unscale_(self.wkr_opt)
                
                # Manager/Worker는 gradient scale이 달라 별도 threshold를 사용한다.
                mgr_preclip_norm = float(
                    nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=self.mgr_max_grad_norm)
                ) if mgr_has_grad else 0.0
                wkr_preclip_norm = float(
                    nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=self.wkr_max_grad_norm)
                ) if wkr_has_grad else 0.0
                
                # AMP: scaler가 optimizer.step() 대행 (inf/nan 감지 시 스킵)
                if mgr_has_grad:
                    self.grad_scaler.step(self.mgr_opt)
                if wkr_has_grad:
                    self.grad_scaler.step(self.wkr_opt)
                self.grad_scaler.update()
            else:
                mgr_preclip_norm = 0.0
                wkr_preclip_norm = 0.0
            
            # Step LR Schedulers
            self.mgr_scheduler.step()
            if self.wkr_scheduler is not None:
                self.wkr_scheduler.step()
            
            # [Update DAgger EMA]
            is_clip_hit = 1.0 if mgr_preclip_norm >= self.mgr_max_grad_norm else 0.0
            self.manager_clip_hit_ema = self.manager_clip_hit_ema * 0.90 + is_clip_hit * 0.10
            current_entropy = 0.0
            if valid_mask.sum().item() > 0:
                current_entropy = float((mgr_entropy * valid_mask).sum().item() / valid_mask.sum().item())
            self.manager_entropy_ema = self.manager_entropy_ema * 0.90 + current_entropy * 0.10
            
            avg_reward = (avg_reward * 0.95) + (rewards.mean().item() * 0.05)

            # 성공률 EMA (tqdm 표시를 위해 postfix 직전에 계산)
            is_success = (self.env.current_node == g0)
            ep_success = is_success.float().mean().item()
            success_ema = success_ema * 0.95 + ep_success * 0.05
            rl_history['success_rates'].append(success_ema)

            pbar.set_postfix({
                'EMA': f"{success_ema*100:.1f}%",
                'Succ': f"{ep_success*100:.1f}%",
                'Loss': f"{loss.item():.2f}",
                'Rw': f"{rewards.mean().item():.1f}",
            })

            # 학습 곡선 기록
            rl_history['rewards'].append(rewards.mean().item())
            rl_history['losses'].append(loss.item())
            rl_history['path_lengths'].append(path_lengths.float().mean().item())

            if self._collect_debug_this_episode and self._last_debug_tensors is not None:
                diag, sample_payload = self._build_debug_episode(
                    ep=ep,
                    sequences=sequences,
                    rewards=rewards.detach(),
                    checkpoint_reward=plan_diag['plan_adjustment'].detach(),
                    plan_penalty=plan_diag['plan_penalty'].detach(),
                    far_plan_penalty=plan_diag['far_plan_penalty'].detach(),
                    plan_len_ref=plan_diag['plan_len_ref'].detach(),
                    plan_len_min=plan_diag['plan_len_min'].detach(),
                    plan_len_max=plan_diag['plan_len_max'].detach(),
                    anchor_hop_error_mean=plan_diag['anchor_hop_error_mean'].detach(),
                    anchor_near_ratio=plan_diag['anchor_near_ratio'].detach(),
                    checkpoint_quality=plan_diag['checkpoint_quality'].detach(),
                    normalized_returns=norm_returns.detach(),
                    success_ema=success_ema,
                    advantages=advantages.detach(),
                    mgr_nll=mgr_nll.detach(),
                    mgr_entropy=mgr_entropy.detach(),
                    policy_loss=policy_loss.detach(),
                    critic_loss=critic_loss.detach(),
                    entropy_bonus=entropy_bonus.detach(),
                    mgr_aux_ce_loss=mgr_aux_ce_loss.detach(),
                    mgr_aux_weight=mgr_aux_weight,
                    worker_aux_ce_loss=worker_aux_ce_loss.detach(),
                    worker_aux_weight=wkr_aux_weight,
                    spacing_error_mean=plan_diag['spacing_error_mean'].detach(),
                    monotonic_violation_rate=plan_diag['monotonic_violation_rate'].detach(),
                    segment_budget_error_mean=plan_diag['segment_budget_error_mean'].detach(),
                    first_segment_budget_err=plan_diag['first_segment_budget_err'].detach(),
                    first_segment_overshoot=plan_diag['first_segment_overshoot'].detach(),
                    frontloaded_overshoot_rate=plan_diag['frontloaded_overshoot_rate'].detach(),
                    total_loss=loss.detach(),
                    mgr_preclip_norm=mgr_preclip_norm,
                    wkr_preclip_norm=wkr_preclip_norm,
                    mgr_lr=self.mgr_scheduler.get_last_lr()[0],
                    wkr_lr=self.wkr_scheduler.get_last_lr()[0] if self.wkr_scheduler is not None else 0.0,
                )
                self._last_diag = diag
                self._record_debug_episode(diag, sample_payload)
            
            # VRAM 정리: 100 에피소드마다 캐시 해제 (매 에피소드는 오버헤드만 증가)
            if torch.cuda.is_available() and ep % 100 == 0:
                torch.cuda.empty_cache()
            
            if ep % self.debug_interval == 0:
                wkr_lr_val = self.wkr_scheduler.get_last_lr()[0] if self.wkr_scheduler is not None else 0.0
                # 터미널에는 핵심 1줄만 출력
                pbar.write(f"[Ep {ep}] EMA={success_ema*100:.1f}% | Loss={loss.item():.2f} | MgrLR={self.mgr_scheduler.get_last_lr()[0]:.1e} | WkrLR={wkr_lr_val:.1e}")
                
                # [Diagnostic] 디버그 모드: 상세 로그는 파일에만 기록, 터미널 출력 생략
                if self.debug_mode and self._debug_window:
                    self._emit_debug_window(
                        ep=ep,
                        success_ema=success_ema,
                        current_loss=loss.item(),
                        pbar=pbar,
                    )
                
                self.save_models(ep)

            if ep == 0 or success_ema >= getattr(self, '_best_success_ema', float('-inf')):
                self._best_success_ema = success_ema
                self._save_unified_checkpoint(
                    "best.pt",
                    ep,
                    metric=success_ema,
                    metric_name="success_ema",
                )

        if self.debug_mode and self._debug_window and rl_history['losses']:
            self._emit_debug_window(
                ep=episodes - 1,
                success_ema=success_ema,
                current_loss=rl_history['losses'][-1],
                pbar=pbar,
            )
        
        # === RL 학습 곡선 그래프 생성 ===
        self._plot_rl_curves(rl_history)
        if rl_history['losses']:
            self._save_unified_checkpoint(
                "final.pt",
                episodes - 1,
                metric=success_ema,
                metric_name="success_ema",
            )

    def _init_debug_outputs(self):
        self._debug_window = []
        self._debug_csv_fields = None
        self._debug_log_path = os.path.join(self.save_dir, "rl_debug_log.txt")
        self._debug_csv_path = os.path.join(self.save_dir, "debug_metrics.csv")
        self._debug_jsonl_path = os.path.join(self.save_dir, "debug_episode_sample.jsonl")

        with open(self._debug_log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== RL Debug Log (LR={self.config.lr}, POMO={self.num_pomo}) ===\n\n")
        with open(self._debug_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write("")
        with open(self._debug_jsonl_path, 'w', encoding='utf-8') as f:
            f.write("")
        runtime_payload = {
            'stage': self.stage,
            'save_dir': self.save_dir,
            'run_id': self._run_id,
            'lr': float(getattr(self.config, 'lr', 0.0)),
            'episodes': int(getattr(self.config, 'episodes', 0)),
            'num_pomo': int(getattr(self.config, 'num_pomo', 0)),
            'debug': bool(self.debug_mode),
            'run_type': str(getattr(self.config, 'run_type', 'train')),
            'parent_checkpoints': list(getattr(self.config, 'parent_checkpoints', [])),
            'reward_cfg': self.reward_cfg,
            'critic_cfg': self.critic_cfg,
            'mgr_max_grad_norm': float(self.mgr_max_grad_norm),
            'wkr_max_grad_norm': float(self.wkr_max_grad_norm),
            'mgr_aux_start': float(self.mgr_aux_start),
            'mgr_aux_end': float(self.mgr_aux_end),
            'wkr_aux_start': float(self.wkr_aux_start),
            'wkr_aux_end': float(self.wkr_aux_end),
            'mgr_lr_scale': float(self.mgr_lr_scale),
            'mgr_eta_min_scale': float(self.mgr_eta_min_scale),
        }
        with open(os.path.join(self.save_dir, "runtime_config.json"), 'w', encoding='utf-8') as f:
            json.dump(runtime_payload, f, ensure_ascii=True, indent=2)
        manifest_payload = {
            'run_id': self._run_id,
            'stage': self.stage,
            'save_dir': self.save_dir,
            'started_at': datetime.now().isoformat(timespec='seconds'),
            'episodes': int(getattr(self.config, 'episodes', 0)),
            'run_type': str(getattr(self.config, 'run_type', 'train')),
            'parent_checkpoints': list(getattr(self.config, 'parent_checkpoints', [])),
        }
        with open(os.path.join(self.save_dir, "run_manifest.json"), 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _compute_grad_norm(parameters):
        total_sq = 0.0
        for p in parameters:
            if p.grad is not None:
                total_sq += p.grad.data.norm(2).item() ** 2
        return total_sq ** 0.5

    @staticmethod
    def _safe_div(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    @staticmethod
    def _safe_tensor_mean_item(value):
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0.0
            return float(value.mean().item())
        return float(value)

    def _build_revisit_penalty(self, candidate_nodes, visit_counts, recent_nodes):
        penalty = visit_counts.float() * self.reward_cfg['VISIT_LOGIT_PENALTY']
        if recent_nodes is None or recent_nodes.numel() == 0:
            return penalty

        recent_hits = (candidate_nodes.unsqueeze(-1) == recent_nodes.unsqueeze(1)).any(dim=-1)
        penalty = penalty + recent_hits.float() * self.reward_cfg['RECENT_NODE_PENALTY']
        return penalty

    def _record_debug_episode(self, diag, sample_payload):
        self._debug_window.append({
            'diag': diag,
            'sample': sample_payload,
        })

    def _normalize_returns(self, returns):
        detached = returns.detach()
        batch_mean = float(detached.mean().item())
        batch_std = float(detached.std(unbiased=False).item())
        batch_std = max(batch_std, 1.0)
        momentum = self.critic_cfg['return_momentum']

        if self.ret_ema_mean is None:
            self.ret_ema_mean = batch_mean
            self.ret_ema_std = batch_std
        else:
            self.ret_ema_mean = momentum * self.ret_ema_mean + (1.0 - momentum) * batch_mean
            self.ret_ema_std = momentum * self.ret_ema_std + (1.0 - momentum) * batch_std
            self.ret_ema_std = max(self.ret_ema_std, 1.0)

        eps = self.critic_cfg['return_eps']
        normalized = (returns - self.ret_ema_mean) / (self.ret_ema_std + eps)
        return normalized, self.ret_ema_mean, self.ret_ema_std

    def _manager_aux_weight(self, ep, episodes):
        if episodes <= 1:
            return self.mgr_aux_end
        ratio = float(ep) / float(max(episodes - 1, 1))
        return self.mgr_aux_start + (self.mgr_aux_end - self.mgr_aux_start) * ratio

    def _worker_aux_weight(self, ep, episodes):
        if episodes <= 1:
            return self.wkr_aux_end
        ratio = float(ep) / float(max(episodes - 1, 1))
        return self.wkr_aux_start + (self.wkr_aux_end - self.wkr_aux_start) * ratio

    def _build_hold_then_cosine_scheduler(self, optimizer, episodes, hold_ratio=0.0, min_factor=0.1):
        hold_ratio = float(max(0.0, min(1.0, hold_ratio)))
        min_factor = float(max(0.0, min(1.0, min_factor)))
        hold_steps = int(episodes * hold_ratio)

        def lr_lambda(step_idx):
            if episodes <= 1:
                return min_factor
            if step_idx < hold_steps:
                return 1.0
            cosine_steps = max(episodes - hold_steps - 1, 1)
            progress = min(max(step_idx - hold_steps, 0), cosine_steps) / cosine_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _save_unified_checkpoint(self, filename, ep, metric=None, metric_name=None, extra_payload=None):
        os.makedirs(self.save_dir, exist_ok=True)
        payload = {
            'epoch': int(ep),
            'stage': self.stage,
            'manager_state': self.manager.state_dict(),
            'worker_state': self.worker.state_dict(),
        }
        if metric is not None:
            payload['metric'] = float(metric)
        if metric_name is not None:
            payload['metric_name'] = str(metric_name)
        if extra_payload:
            payload.update(extra_payload)
        torch.save(payload, os.path.join(self.save_dir, filename))

    def _build_reference_metadata(self, start_idx, goal_idx):
        shortest_hops = max(float(self.env.hop_matrix[start_idx, goal_idx].item()), 1.0)
        target_segment_hops = max(float(self.reward_cfg['TARGET_SEGMENT_HOPS']), 1.0)
        plan_count_band = float(self.reward_cfg['PLAN_COUNT_BAND'])
        plan_len_ref_value = max(1.0, float(np.ceil(shortest_hops / target_segment_hops)))
        plan_len_min_value = 1.0 if shortest_hops <= 4.0 else plan_len_ref_value
        plan_len_max_value = plan_len_ref_value + plan_count_band

        optimal_path = self.env.reconstruct_hop_shortest_path_indices(int(start_idx), int(goal_idx))
        if not optimal_path:
            optimal_path = [int(start_idx), int(goal_idx)] if int(start_idx) != int(goal_idx) else [int(start_idx)]

        path_hops = max(len(optimal_path) - 1, 1)
        max_path_index = max(len(optimal_path) - 1, 0)
        if max_path_index <= 0:
            min_anchor_pos = 0
            max_anchor_pos = 0
        else:
            min_anchor_pos = 1
            max_anchor_pos = max(1, max_path_index - 1)

        ref_token_count = max(1, int(plan_len_ref_value))
        reference_anchor_positions = []
        for step_idx in range(1, ref_token_count + 1):
            pos = int(round(step_idx * path_hops / (ref_token_count + 1)))
            pos = max(min_anchor_pos, min(max_anchor_pos, pos))
            if reference_anchor_positions and pos < reference_anchor_positions[-1]:
                pos = reference_anchor_positions[-1]
            reference_anchor_positions.append(pos)

        reference_anchor_nodes = [optimal_path[pos] for pos in reference_anchor_positions]
        reference_progress_targets = [
            float(step_idx) / float(ref_token_count + 1)
            for step_idx in range(1, ref_token_count + 1)
        ]
        return {
            'shortest_hops': shortest_hops,
            'plan_len_ref_value': plan_len_ref_value,
            'plan_len_min_value': plan_len_min_value,
            'plan_len_max_value': plan_len_max_value,
            'optimal_path': optimal_path,
            'reference_anchor_positions': reference_anchor_positions,
            'reference_anchor_nodes': reference_anchor_nodes,
            'reference_progress_targets': reference_progress_targets,
            'reference_token_count': ref_token_count,
        }

    def _build_reference_sequence(self, start_idx, goal_idx, batch_size, max_len=50):
        ref_meta = self._build_reference_metadata(start_idx, goal_idx)
        reference_anchor_nodes = ref_meta['reference_anchor_nodes']
        ref_token_count = int(ref_meta['reference_token_count'])
        seq = torch.full(
            (batch_size, max_len),
            self.manager.PAD_TOKEN,
            dtype=torch.long,
            device=self.device,
        )
        if ref_token_count > 0:
            ref_nodes_tensor = torch.tensor(
                reference_anchor_nodes[:max_len],
                dtype=torch.long,
                device=self.device,
            )
            seq[:, :ref_nodes_tensor.numel()] = ref_nodes_tensor.unsqueeze(0).expand(batch_size, -1)
        return seq, ref_meta

    def _gather_manager_teacher_embeddings(self, full_ref_embs, target_seq, num_nodes):
        _, _, hidden_dim = full_ref_embs.shape
        valid_indices_mask = (target_seq < (num_nodes + 1)) & (target_seq >= 0)
        safe_target_seq = target_seq.clone()
        safe_target_seq[~valid_indices_mask] = 0
        target_seq_emb = torch.gather(
            full_ref_embs,
            1,
            safe_target_seq.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
        padding_mask = (~valid_indices_mask).unsqueeze(-1).expand(-1, -1, hidden_dim)
        return target_seq_emb.masked_fill(padding_mask, 0.0)

    def _compute_plan_reward_adjustment(self, start_idx, goal_idx, sequences):
        valid_mask = (sequences >= 0) & (sequences < self.env.num_nodes)
        safe_plan = sequences.clamp(min=0, max=max(self.env.num_nodes - 1, 0))
        plan_lengths = valid_mask.sum(dim=1).float()

        ref_meta = self._build_reference_metadata(start_idx, goal_idx)
        shortest_hops = ref_meta['shortest_hops']
        plan_density = plan_lengths / shortest_hops
        plan_len_ref_value = ref_meta['plan_len_ref_value']
        plan_len_min_value = ref_meta['plan_len_min_value']
        plan_len_max_value = ref_meta['plan_len_max_value']
        plan_len_ref = torch.full_like(plan_lengths, plan_len_ref_value)
        plan_len_min = torch.full_like(plan_lengths, plan_len_min_value)
        plan_len_max = torch.full_like(plan_lengths, plan_len_max_value)

        start_nodes = torch.full_like(safe_plan, int(start_idx))
        goal_nodes = torch.full_like(safe_plan, int(goal_idx))
        start_to_subgoal_hops = self.env.hop_matrix[start_nodes, safe_plan]
        subgoal_to_goal_hops = self.env.hop_matrix[safe_plan, goal_nodes]
        corridor_ok = valid_mask & (
            (start_to_subgoal_hops + subgoal_to_goal_hops) <= (shortest_hops + 2.0)
        )
        corridor_ratio = corridor_ok.float().sum(dim=1) / plan_lengths.clamp(min=1.0)

        corridor_deficit = torch.clamp(
            self.reward_cfg['PLAN_CORRIDOR_TARGET'] - corridor_ratio,
            min=0.0,
        )
        empty_plan_penalty = torch.where(
            plan_lengths == 0,
            torch.full_like(plan_lengths, self.reward_cfg['EMPTY_PLAN_PENALTY']),
            torch.zeros_like(plan_lengths),
        )
        under_plan = torch.clamp(plan_len_min - plan_lengths, min=0.0)
        over_plan = torch.clamp(plan_lengths - plan_len_max, min=0.0)
        under_plan_penalty = self.reward_cfg['PLAN_COUNT_UNDER_PENALTY'] * under_plan
        over_plan_penalty = self.reward_cfg['PLAN_COUNT_OVER_PENALTY'] * over_plan
        far_plan_penalty = under_plan_penalty

        reference_anchor_nodes = ref_meta['reference_anchor_nodes']
        reference_progress_targets = ref_meta['reference_progress_targets']
        ref_token_count = ref_meta['reference_token_count']
        reference_anchor_nodes_tensor = torch.tensor(
            reference_anchor_nodes,
            dtype=safe_plan.dtype,
            device=safe_plan.device,
        )
        reference_progress_targets_tensor = torch.tensor(
            reference_progress_targets,
            dtype=torch.float32,
            device=safe_plan.device,
        )

        anchor_targets = torch.full_like(safe_plan, -1)
        progress_targets = torch.zeros_like(plan_lengths.unsqueeze(1).expand_as(safe_plan).float())
        anchor_compare_counts = torch.zeros_like(plan_lengths)
        first_anchor_errors = torch.zeros_like(plan_lengths)
        for row_idx in range(safe_plan.size(0)):
            plan_len = int(plan_lengths[row_idx].item())
            if plan_len <= 0:
                continue
            compare_len = min(plan_len, ref_token_count)
            anchor_compare_counts[row_idx] = float(compare_len)
            if compare_len <= 0:
                continue
            anchor_nodes = torch.tensor(
                reference_anchor_nodes[:compare_len],
                dtype=safe_plan.dtype,
                device=safe_plan.device,
            )
            anchor_targets[row_idx, :compare_len] = anchor_nodes
            progress_targets[row_idx, :compare_len] = reference_progress_targets_tensor[:compare_len]
            if ref_token_count >= 2:
                first_anchor_errors[row_idx] = float(
                    self.env.hop_matrix[safe_plan[row_idx, 0], int(reference_anchor_nodes[0])].item()
                )

        anchor_valid_mask = valid_mask & (anchor_targets >= 0)
        safe_anchor_targets = anchor_targets.clamp(min=0, max=max(self.env.num_nodes - 1, 0))
        anchor_hop_errors = self.env.hop_matrix[safe_plan, safe_anchor_targets].float()
        anchor_hop_errors = torch.where(
            anchor_valid_mask,
            anchor_hop_errors,
            torch.zeros_like(anchor_hop_errors),
        )
        anchor_hop_error_mean = (
            torch.clamp(anchor_hop_errors, max=4.0).sum(dim=1)
            / anchor_compare_counts.clamp(min=1.0)
        )
        anchor_near_ratio = (
            ((anchor_hop_errors <= 1.0) & anchor_valid_mask).float().sum(dim=1)
            / anchor_compare_counts.clamp(min=1.0)
        )
        anchor_reward = (
            -self.reward_cfg['ANCHOR_HOP_PENALTY'] * anchor_hop_error_mean
            + self.reward_cfg['ANCHOR_NEAR_BONUS'] * anchor_near_ratio
        )
        remaining_goal_hops = subgoal_to_goal_hops.float()
        sampled_progress = torch.clamp(
            1.0 - (remaining_goal_hops / max(shortest_hops, 1.0)),
            min=0.0,
            max=1.0,
        )
        spacing_errors = torch.where(
            anchor_valid_mask,
            (sampled_progress - progress_targets).abs(),
            torch.zeros_like(sampled_progress),
        )
        spacing_error_mean = spacing_errors.sum(dim=1) / anchor_compare_counts.clamp(min=1.0)

        prev_remaining_goal_hops = torch.full_like(remaining_goal_hops, shortest_hops)
        if remaining_goal_hops.size(1) > 1:
            prev_remaining_goal_hops[:, 1:] = remaining_goal_hops[:, :-1]
        monotonic_violation = torch.where(
            anchor_valid_mask,
            torch.clamp(remaining_goal_hops - prev_remaining_goal_hops + 0.5, min=0.0),
            torch.zeros_like(remaining_goal_hops),
        )
        monotonic_violation_mean = monotonic_violation.sum(dim=1) / anchor_compare_counts.clamp(min=1.0)
        monotonic_violation_rate = (
            ((monotonic_violation > 1e-6) & anchor_valid_mask).float().sum(dim=1)
            / anchor_compare_counts.clamp(min=1.0)
        )
        first_anchor_penalty = torch.where(
            (plan_lengths > 0) & (plan_len_ref >= 2.0),
            -self.reward_cfg['FIRST_ANCHOR_PENALTY'] * torch.clamp(first_anchor_errors, max=4.0),
            torch.zeros_like(plan_lengths),
        )
        has_plan = plan_lengths > 0
        first_subgoal_hops = torch.where(
            has_plan,
            self.env.hop_matrix[start_nodes[:, 0], safe_plan[:, 0]].float(),
            torch.zeros_like(plan_lengths),
        )
        prev_plan_nodes = torch.full_like(safe_plan, int(start_idx))
        if safe_plan.size(1) > 1:
            prev_plan_nodes[:, 1:] = safe_plan[:, :-1]
        actual_segment_hops = self.env.hop_matrix[prev_plan_nodes, safe_plan].float()
        actual_segment_hops = torch.where(
            valid_mask,
            actual_segment_hops,
            torch.zeros_like(actual_segment_hops),
        )
        remaining_hops_before_rank = self.env.hop_matrix[prev_plan_nodes, goal_nodes].float()
        rank_positions = torch.arange(safe_plan.size(1), device=safe_plan.device, dtype=torch.float32).unsqueeze(0)
        remaining_slots = torch.clamp(plan_len_ref.unsqueeze(1) - rank_positions, min=1.0)
        expected_segment_hops = torch.where(
            valid_mask,
            remaining_hops_before_rank / remaining_slots,
            torch.zeros_like(actual_segment_hops),
        )
        segment_budget_error = torch.where(
            valid_mask,
            (actual_segment_hops - expected_segment_hops).abs(),
            torch.zeros_like(actual_segment_hops),
        )
        segment_overshoot = torch.where(
            valid_mask,
            torch.clamp(actual_segment_hops - expected_segment_hops - 1.0, min=0.0),
            torch.zeros_like(actual_segment_hops),
        )
        segment_budget_error_mean = (
            segment_budget_error.sum(dim=1) / plan_lengths.clamp(min=1.0)
        )
        first_segment_budget_err = torch.where(
            has_plan,
            segment_budget_error[:, 0],
            torch.zeros_like(plan_lengths),
        )
        first_segment_overshoot = torch.where(
            has_plan,
            segment_overshoot[:, 0],
            torch.zeros_like(plan_lengths),
        )
        early_weight = torch.ones_like(segment_overshoot)
        if segment_overshoot.size(1) > 0:
            early_weight[:, 0] = 1.5
        if segment_overshoot.size(1) > 1:
            early_weight[:, 1] = 1.25
        frontloaded_overshoot_rate = (
            ((segment_overshoot * early_weight) > 1e-6).float()
            * valid_mask.float()
        ).sum(dim=1) / plan_lengths.clamp(min=1.0)
        # [성능 고도화] 밀도(Density) 초과 페널티: 서브골이 너무 촘촘하면 강한 벌점 부여
        # Why: Density가 목표(0.20)보다 높으면(촘촘하면) PLR이 비효율적으로 높아짐.
        #      이 페널티가 없으면 매니저가 1~2칸마다 서브골을 남발하여 경로 비용이 3~4x로 폭등.
        density_excess = torch.clamp(
            plan_density - self.reward_cfg['PLAN_DENSITY_TARGET'],
            min=0.0,
        )
        density_penalty = self.reward_cfg['PLAN_DENSITY_WEIGHT'] * density_excess
        
        plan_penalty = (
            - self.reward_cfg['PLAN_CORRIDOR_WEIGHT'] * corridor_deficit
            - far_plan_penalty
            - over_plan_penalty
            - self.reward_cfg['SPACING_PENALTY_SCALE'] * spacing_error_mean
            - self.reward_cfg['MONOTONIC_PENALTY_SCALE'] * monotonic_violation_mean
            - density_penalty  # [성능 고도화] 밀도 초과 시 페널티 실제 적용
            + anchor_reward
            + first_anchor_penalty
            + empty_plan_penalty
        )

        checkpoint_quality = self._compute_checkpoint_quality(
            start_idx,
            goal_idx,
            sequences,
            max_reward=self.reward_cfg['CHECKPOINT_MAX_REWARD'],
        )
        plan_adjustment = torch.clamp(
            checkpoint_quality + plan_penalty,
            min=self.reward_cfg['PLAN_ADJUST_MIN'],
            max=self.reward_cfg['PLAN_ADJUST_MAX'],
        )

        return {
            'valid_mask': valid_mask,
            'corridor_ok': corridor_ok,
            'corridor_ratio': corridor_ratio,
            'plan_lengths': plan_lengths,
            'plan_density': plan_density,
            'density_excess': torch.clamp(
                plan_density - self.reward_cfg['PLAN_DENSITY_TARGET'],
                min=0.0,
            ),
            'corridor_deficit': corridor_deficit,
            'plan_len_ref': plan_len_ref,
            'plan_len_min': plan_len_min,
            'plan_len_max': plan_len_max,
            'under_plan_penalty': under_plan_penalty,
            'over_plan_penalty': over_plan_penalty,
            'far_plan_penalty': far_plan_penalty,
            'reference_anchor_nodes': reference_anchor_nodes_tensor,
            'reference_token_count': ref_token_count,
            'anchor_targets': anchor_targets,
            'anchor_compare_counts': anchor_compare_counts,
            'anchor_hop_error_mean': anchor_hop_error_mean,
            'anchor_near_ratio': anchor_near_ratio,
            'anchor_reward': anchor_reward,
            'spacing_error_mean': spacing_error_mean,
            'monotonic_violation_mean': monotonic_violation_mean,
            'monotonic_violation_rate': monotonic_violation_rate,
            'first_anchor_errors': first_anchor_errors,
            'first_subgoal_hops': first_subgoal_hops,
            'actual_segment_hops': actual_segment_hops,
            'expected_segment_hops': expected_segment_hops,
            'segment_budget_error': segment_budget_error,
            'segment_budget_error_mean': segment_budget_error_mean,
            'first_segment_budget_err': first_segment_budget_err,
            'first_segment_overshoot': first_segment_overshoot,
            'frontloaded_overshoot_rate': frontloaded_overshoot_rate,
            'first_anchor_penalty': first_anchor_penalty,
            'plan_penalty': plan_penalty,
            'checkpoint_quality': checkpoint_quality,
            'plan_adjustment': plan_adjustment,
            'shortest_hops': shortest_hops,
        }

    def _build_debug_episode(
        self,
        ep,
        sequences,
        rewards,
        checkpoint_reward,
        plan_penalty,
        far_plan_penalty,
        plan_len_ref,
        plan_len_min,
        plan_len_max,
        anchor_hop_error_mean,
        anchor_near_ratio,
        checkpoint_quality,
        normalized_returns,
        success_ema,
        advantages,
        mgr_nll,
        mgr_entropy,
        policy_loss,
        critic_loss,
        entropy_bonus,
        mgr_aux_ce_loss,
        mgr_aux_weight,
        worker_aux_ce_loss,
        worker_aux_weight,
        spacing_error_mean,
        monotonic_violation_rate,
        segment_budget_error_mean,
        first_segment_budget_err,
        first_segment_overshoot,
        frontloaded_overshoot_rate,
        total_loss,
        mgr_preclip_norm,
        wkr_preclip_norm,
        mgr_lr,
        wkr_lr,
    ):
        dbg = self._last_debug_tensors
        batch_size = sequences.size(0)
        device = sequences.device

        valid_plan_mask = (sequences >= 0) & (sequences < self.env.num_nodes)
        safe_plan = sequences.clamp(min=0, max=max(self.env.num_nodes - 1, 0))
        plan_lengths = valid_plan_mask.sum(dim=1).float()
        eos_seen = (sequences == self.env.num_nodes).any(dim=1).float()

        unique_counts = []
        for seq in sequences:
            valid_nodes = seq[(seq >= 0) & (seq < self.env.num_nodes)]
            unique_counts.append(
                float(torch.unique(valid_nodes).numel()) if valid_nodes.numel() > 0 else 0.0
            )
        unique_counts = torch.tensor(unique_counts, dtype=torch.float32, device=device)
        unique_ratio = unique_counts / plan_lengths.clamp(min=1.0)

        goal_reward_value = max(float(dbg['goal_reward_value']), 1e-6)
        shortest_hops = max(float(self.env.hop_matrix[int(dbg['start_idx']), int(dbg['goal_idx'])].item()), 1.0)
        expert_density_low = 1.0 / 11.0
        expert_density_high = 1.0 / 5.0

        plan_density = plan_lengths / shortest_hops
        expert_density_gap = torch.where(
            plan_density < expert_density_low,
            expert_density_low - plan_density,
            torch.where(
                plan_density > expert_density_high,
                plan_density - expert_density_high,
                torch.zeros_like(plan_density),
            ),
        )

        start_nodes = torch.full_like(safe_plan, int(dbg['start_idx']))
        goal_nodes = torch.full_like(safe_plan, int(dbg['goal_idx']))
        start_to_subgoal_hops = self.env.hop_matrix[start_nodes, safe_plan]
        subgoal_to_goal_hops = self.env.hop_matrix[safe_plan, goal_nodes]
        corridor_ok = valid_plan_mask & (
            (start_to_subgoal_hops + subgoal_to_goal_hops) <= (shortest_hops + 2.0)
        )
        corridor_ratio = corridor_ok.float().sum(dim=1) / plan_lengths.clamp(min=1.0)
        prev_nodes = torch.full_like(safe_plan, int(dbg['start_idx']))
        if safe_plan.size(1) > 1:
            prev_nodes[:, 1:] = safe_plan[:, :-1]
        current_goal_hops_by_rank = self.env.hop_matrix[prev_nodes, goal_nodes].float()
        selected_goal_hops_by_rank = subgoal_to_goal_hops.float()
        manager_nonprogress_mask = valid_plan_mask & (
            selected_goal_hops_by_rank >= (current_goal_hops_by_rank - 1e-6)
        )
        manager_detour_hops = torch.clamp(
            (start_to_subgoal_hops + subgoal_to_goal_hops).float() - shortest_hops,
            min=0.0,
        )
        manager_step_total = int(valid_plan_mask.sum().item())

        last_plan_rank = (plan_lengths.long() - 1).clamp(min=0)
        row_idx = torch.arange(batch_size, device=device)
        last_plan_nodes = torch.where(
            plan_lengths > 0,
            safe_plan[row_idx, last_plan_rank],
            torch.full((batch_size,), int(dbg['start_idx']), dtype=safe_plan.dtype, device=device),
        )
        remaining_goal_hops_at_eos = self.env.hop_matrix[last_plan_nodes, int(dbg['goal_idx'])].float()
        eos_near_goal_mask = eos_seen.bool() & (remaining_goal_hops_at_eos <= 2.0)
        eos_count = int(eos_seen.sum().item())

        has_plan = plan_lengths > 0
        first_nodes = safe_plan[:, 0]
        first_subgoal_hops = torch.where(
            has_plan,
            self.env.hop_matrix[int(dbg['start_idx']), first_nodes],
            torch.zeros_like(plan_lengths),
        ).float()

        pair_valid = valid_plan_mask[:, 1:] & valid_plan_mask[:, :-1]
        pair_hops = self.env.hop_matrix[safe_plan[:, :-1], safe_plan[:, 1:]].float()
        pair_counts = pair_valid.sum(dim=1).float()
        mean_inter_subgoal_hops = (
            (pair_hops * pair_valid.float()).sum(dim=1) / pair_counts.clamp(min=1.0)
        )
        mean_inter_subgoal_hops = torch.where(
            pair_counts > 0,
            mean_inter_subgoal_hops,
            torch.zeros_like(mean_inter_subgoal_hops),
        )
        max_inter_subgoal_hops = pair_hops.masked_fill(~pair_valid, 0.0).max(dim=1).values
        max_inter_subgoal_hops = torch.where(
            pair_counts > 0,
            max_inter_subgoal_hops,
            torch.zeros_like(max_inter_subgoal_hops),
        )

        hit_mask = dbg['subgoal_hit_mask'].to(device)
        reached_subgoals = (hit_mask & valid_plan_mask).sum(dim=1).float()
        generated_subgoals = plan_lengths
        plan_utilization = reached_subgoals / generated_subgoals.clamp(min=1.0)
        reached_last_subgoal_mask = dbg['reached_last_subgoal_mask'].to(device).bool()
        goal_after_last_subgoal_mask = dbg['goal_after_last_subgoal_mask'].to(device).bool()
        soft_arrival_mask = dbg['soft_arrival_mask'].to(device).bool()
        skip_mask = dbg['skip_mask'].to(device).bool()
        skip_due_to_passed_mask = dbg['skip_due_to_passed_mask'].to(device).bool()
        skip_due_to_goal_progress_mask = dbg['skip_due_to_goal_progress_mask'].to(device).bool()
        skip_due_to_unreachable_mask = dbg['skip_due_to_unreachable_mask'].to(device).bool()
        switch_reason_exact_mask = dbg['switch_reason_exact_mask'].to(device).bool()
        switch_reason_soft_mask = dbg['switch_reason_soft_mask'].to(device).bool()
        post_last_sg_success_mask = dbg['post_last_sg_success_mask'].to(device).bool()
        post_last_sg_stagnation_mask = dbg['post_last_sg_stagnation_mask'].to(device).bool()
        goal_dist_after_last_subgoal = dbg['goal_dist_after_last_subgoal'].to(device)
        steps_after_last_subgoal = dbg['steps_after_last_subgoal'].to(device)
        ptr_dwell_mean = dbg['ptr_dwell_mean'].to(device)
        remaining_goal_hops_at_handoff = dbg['remaining_goal_hops_at_handoff'].to(device)
        goal_hops_1step_after_handoff = dbg['goal_hops_1step_after_handoff'].to(device)

        success_mask = dbg['success_mask'].to(device).bool()
        fail_mask = ~success_mask
        total_rewards_with_ckpt = rewards.to(device)
        norm_returns = normalized_returns.to(device)

        fail_shaping = dbg['r1_pbrs'] + dbg['r2_subgoal'] + dbg['r5_milestone'] + dbg['r6_explore']
        success_rewards = total_rewards_with_ckpt[success_mask]
        valid_goal_share_mask = success_rewards > 1e-6
        success_goal_share = dbg['r3_goal'][success_mask][valid_goal_share_mask] / success_rewards[valid_goal_share_mask]

        goal_progress_per_100_steps = dbg['progress_ratio'] * 100.0 / dbg['step_counts'].clamp(min=1.0)
        subgoal_progress_per_100_steps = reached_subgoals * 100.0 / dbg['step_counts'].clamp(min=1.0)

        min_hop_to_active = dbg['min_hop_to_active_subgoal'].clone()
        min_hop_to_active = torch.where(
            torch.isfinite(min_hop_to_active),
            min_hop_to_active,
            torch.zeros_like(min_hop_to_active),
        )
        max_subgoal_progress = dbg['max_subgoal_progress']

        mgr_postclip_norm = self._compute_grad_norm(self.manager.parameters())
        wkr_postclip_norm = self._compute_grad_norm(self.worker.parameters())
        manager_clip_hit = float(mgr_preclip_norm > self.mgr_max_grad_norm)
        worker_clip_hit = float(wkr_preclip_norm > self.wkr_max_grad_norm)

        diag = {
            'ep': int(ep),
            'sample_count': int(batch_size),
            'mgr_lr': float(mgr_lr),
            'wkr_lr': float(wkr_lr),
            'loss': float(total_loss.item()),
            'success_ema': float(success_ema),
            'r1_pbrs': float(dbg['r1_pbrs'].mean().item()),
            'r2_subgoal': float(dbg['r2_subgoal'].mean().item()),
            'r3_goal': float(dbg['r3_goal'].mean().item()),
            'r4_efficiency': float(dbg['r4_efficiency'].mean().item()),
            'r5_milestone': float(dbg['r5_milestone'].mean().item()),
            'r6_explore': float(dbg['r6_explore'].mean().item()),
            'r7_plan_penalty': float(plan_penalty.mean().item()),
            'p1_time': float(dbg['p1_time'].mean().item()),
            'p2_loop': float(dbg['p2_loop'].mean().item()),
            'p3_fail': float(dbg['p3_fail'].mean().item()),
            'total_reward': float(dbg['base_reward'].mean().item()),
            'reward_std': float(dbg['base_reward'].std(unbiased=False).item()),
            'reward_min': float(dbg['base_reward'].min().item()),
            'reward_max': float(dbg['base_reward'].max().item()),
            'ckpt_reward_mean': float(checkpoint_reward.mean().item()),
            'ckpt_reward_std': float(checkpoint_reward.std(unbiased=False).item()),
            'ckpt_quality_mean': float(checkpoint_quality.mean().item()),
            'reward_mean_with_ckpt': float(total_rewards_with_ckpt.mean().item()),
            'reward_std_with_ckpt': float(total_rewards_with_ckpt.std(unbiased=False).item()),
            'reward_min_with_ckpt': float(total_rewards_with_ckpt.min().item()),
            'reward_max_with_ckpt': float(total_rewards_with_ckpt.max().item()),
            'goal_hit_rate': float(dbg['r3_goal'].mean().item() / goal_reward_value),
            'success_count': int(success_mask.sum().item()),
            'fail_count': int(fail_mask.sum().item()),
            'success_rate': float(success_mask.float().mean().item()),
            'goal_hit_count': int(success_mask.sum().item()),
            'loop_fail_count': int(dbg['loop_fail_mask'].sum().item()),
            'loop_fail_rate': float(dbg['loop_fail_mask'].float().mean().item()),
            'stagnation_fail_count': int(dbg['stagnation_fail_mask'].sum().item()),
            'stagnation_fail_rate': float(dbg['stagnation_fail_mask'].float().mean().item()),
            'success_total_reward_sum': float(total_rewards_with_ckpt[success_mask].sum().item()),
            'fail_total_reward_sum': float(total_rewards_with_ckpt[fail_mask].sum().item()),
            'success_norm_return_sum': float(norm_returns[success_mask].sum().item()),
            'fail_norm_return_sum': float(norm_returns[fail_mask].sum().item()),
            'success_ckpt_reward_sum': float(checkpoint_reward[success_mask].sum().item()),
            'fail_ckpt_reward_sum': float(checkpoint_reward[fail_mask].sum().item()),
            'fail_shaping_sum': float(fail_shaping[fail_mask].sum().item()),
            'goal_share_sum': float(success_goal_share.sum().item()),
            'goal_share_count': int(success_goal_share.numel()),
            'final_goal_dist_mean': float(dbg['final_goal_dist'].mean().item()),
            'final_goal_dist_min': float(dbg['final_goal_dist'].min().item()),
            'final_goal_dist_max': float(dbg['final_goal_dist'].max().item()),
            'progress_mean': float(dbg['progress_ratio'].mean().item()),
            'progress_min': float(dbg['progress_ratio'].min().item()),
            'progress_max': float(dbg['progress_ratio'].max().item()),
            'plan_len_mean': float(plan_lengths.mean().item()),
            'plan_len_min': float(plan_lengths.min().item()),
            'plan_len_max': float(plan_lengths.max().item()),
            'plan_len_ref_mean': float(plan_len_ref.mean().item()),
            'plan_len_ref_min': float(plan_len_ref.min().item()),
            'plan_len_ref_max': float(plan_len_ref.max().item()),
            'plan_len_target_sum': float(plan_len_ref.sum().item()),
            'plan_len_min_target_sum': float(plan_len_min.sum().item()),
            'plan_len_max_target_sum': float(plan_len_max.sum().item()),
            'plan_terminated_rate': float(eos_seen.mean().item()),
            'plan_empty_rate': float((plan_lengths == 0).float().mean().item()),
            'plan_unique_ratio': float(unique_ratio.mean().item()),
            'generated_subgoal_count': float(generated_subgoals.sum().item()),
            'reached_subgoal_count': float(reached_subgoals.sum().item()),
            'far_plan_count': int((plan_lengths < plan_len_min).sum().item()),
            'far_plan_rate': float((plan_lengths < plan_len_min).float().mean().item()),
            'plan_under_count': int((plan_lengths < plan_len_min).sum().item()),
            'plan_over_count': int((plan_lengths > plan_len_max).sum().item()),
            'plan_under_rate': float((plan_lengths < plan_len_min).float().mean().item()),
            'plan_over_rate': float((plan_lengths > plan_len_max).float().mean().item()),
            'plan_utilization': self._safe_div(reached_subgoals.sum().item(), generated_subgoals.sum().item()),
            'plan_density_mean': float(plan_density.mean().item()),
            'plan_density_min': float(plan_density.min().item()),
            'plan_density_max': float(plan_density.max().item()),
            'plan_density_sum': float(plan_density.sum().item()),
            'expert_density_low': expert_density_low,
            'expert_density_high': expert_density_high,
            'expert_density_gap_mean': float(expert_density_gap.mean().item()),
            'expert_density_gap_sum': float(expert_density_gap.sum().item()),
            'anchor_hop_err_mean': float(anchor_hop_error_mean.mean().item()),
            'anchor_hop_err_sum': float(anchor_hop_error_mean.sum().item()),
            'anchor_near_rate': float(anchor_near_ratio.mean().item()),
            'anchor_near_sum': float(anchor_near_ratio.sum().item()),
            'spacing_error_mean': float(spacing_error_mean.mean().item()),
            'spacing_error_sum': float(spacing_error_mean.sum().item()),
            'monotonic_violation_rate': float(monotonic_violation_rate.mean().item()),
            'monotonic_violation_sum': float(monotonic_violation_rate.sum().item()),
            'manager_step_total': manager_step_total,
            'manager_corridor_step_count': int(corridor_ok.sum().item()),
            'manager_nonprogress_step_count': int(manager_nonprogress_mask.sum().item()),
            'manager_detour_hops_sum': float((manager_detour_hops * valid_plan_mask.float()).sum().item()),
            'manager_eos_count': eos_count,
            'manager_eos_near_goal_count': int(eos_near_goal_mask.sum().item()),
            'remaining_hops_when_eos_sum': float((remaining_goal_hops_at_eos * eos_seen).sum().item()),
            'corridor_ratio_mean': float(corridor_ratio.mean().item()),
            'corridor_ratio_sum': float(corridor_ratio.sum().item()),
            'corridor_ratio_success_sum': float(corridor_ratio[success_mask].sum().item()),
            'corridor_ratio_fail_sum': float(corridor_ratio[fail_mask].sum().item()),
            'first_subgoal_hops_mean': float(first_subgoal_hops[has_plan].mean().item()) if has_plan.any() else 0.0,
            'first_subgoal_hops_sum': float(first_subgoal_hops[has_plan].sum().item()),
            'first_subgoal_hops_count': int(has_plan.sum().item()),
            'segment_budget_error_mean': float(segment_budget_error_mean.mean().item()),
            'segment_budget_error_sum': float(segment_budget_error_mean.sum().item()),
            'first_segment_budget_err_mean': float(first_segment_budget_err.mean().item()),
            'first_segment_budget_err_sum': float(first_segment_budget_err.sum().item()),
            'first_segment_overshoot_mean': float(first_segment_overshoot.mean().item()),
            'first_segment_overshoot_sum': float(first_segment_overshoot.sum().item()),
            'frontloaded_overshoot_rate': float(frontloaded_overshoot_rate.mean().item()),
            'frontloaded_overshoot_sum': float(frontloaded_overshoot_rate.sum().item()),
            'manager_corridor_step_rate': self._safe_div(corridor_ok.sum().item(), manager_step_total),
            'manager_nonprogress_step_rate': self._safe_div(manager_nonprogress_mask.sum().item(), manager_step_total),
            'manager_avg_detour_hops': self._safe_div(
                (manager_detour_hops * valid_plan_mask.float()).sum().item(),
                manager_step_total,
            ),
            'manager_eos_near_goal_rate': self._safe_div(eos_near_goal_mask.sum().item(), eos_count),
            'mean_inter_subgoal_hops_mean': float(mean_inter_subgoal_hops.mean().item()),
            'mean_inter_subgoal_hops_sum': float(mean_inter_subgoal_hops.sum().item()),
            'max_inter_subgoal_hops_mean': float(max_inter_subgoal_hops.mean().item()),
            'max_inter_subgoal_hops_max': float(max_inter_subgoal_hops.max().item()),
            'subgoal_reached': float(reached_subgoals.sum().item()),
            'subgoal_total': float(generated_subgoals.sum().item()),
            'subgoal_rate': self._safe_div(reached_subgoals.sum().item(), generated_subgoals.sum().item()),
            'reached_last_subgoal_count': int(reached_last_subgoal_mask.sum().item()),
            'goal_after_last_subgoal_count': int(goal_after_last_subgoal_mask.sum().item()),
            'soft_arrival_count': int(soft_arrival_mask.sum().item()),
            'skip_count': int(skip_mask.sum().item()),
            'switch_reason_exact_count': int(switch_reason_exact_mask.sum().item()),
            'switch_reason_soft_count': int(switch_reason_soft_mask.sum().item()),
            'skip_due_to_passed_count': int(skip_due_to_passed_mask.sum().item()),
            'skip_due_to_goal_progress_count': int(skip_due_to_goal_progress_mask.sum().item()),
            'skip_due_to_unreachable_count': int(skip_due_to_unreachable_mask.sum().item()),
            'post_last_sg_success_count': int(post_last_sg_success_mask.sum().item()),
            'post_last_sg_stagnation_count': int(post_last_sg_stagnation_mask.sum().item()),
            'goal_dist_after_last_sg_sum': float(goal_dist_after_last_subgoal[goal_after_last_subgoal_mask].sum().item()),
            'steps_after_last_sg_sum': float(steps_after_last_subgoal[goal_after_last_subgoal_mask].sum().item()),
            'remaining_goal_hops_at_handoff_sum': float(
                remaining_goal_hops_at_handoff[goal_after_last_subgoal_mask].sum().item()
            ),
            'goal_hops_1step_after_handoff_sum': float(
                goal_hops_1step_after_handoff[goal_after_last_subgoal_mask].sum().item()
            ),
            'ptr_dwell_sum': float(ptr_dwell_mean.sum().item()),
            'wkr_entropy_mean': float(dbg['wkr_entropy_mean'].mean().item()),
            'avg_steps': float(dbg['step_counts'].mean().item()),
            'steps_min': float(dbg['step_counts'].min().item()),
            'steps_max': float(dbg['step_counts'].max().item()),
            'min_hop_to_active_subgoal_mean': float(min_hop_to_active.mean().item()),
            'min_hop_to_active_subgoal_min': float(min_hop_to_active.min().item()),
            'min_hop_to_active_subgoal_max': float(min_hop_to_active.max().item()),
            'min_hop_to_active_subgoal_sum': float(min_hop_to_active.sum().item()),
            'max_subgoal_progress_mean': float(max_subgoal_progress.mean().item()),
            'max_subgoal_progress_min': float(max_subgoal_progress.min().item()),
            'max_subgoal_progress_max': float(max_subgoal_progress.max().item()),
            'max_subgoal_progress_sum': float(max_subgoal_progress.sum().item()),
            'goal_progress_per_100_steps_mean': float(goal_progress_per_100_steps.mean().item()),
            'goal_progress_per_100_steps_sum': float(goal_progress_per_100_steps.sum().item()),
            'subgoal_progress_per_100_steps_mean': float(subgoal_progress_per_100_steps.mean().item()),
            'subgoal_progress_per_100_steps_sum': float(subgoal_progress_per_100_steps.sum().item()),
            'goal_progress_per100': float(goal_progress_per_100_steps.mean().item()),
            'subgoal_progress_per100': float(subgoal_progress_per_100_steps.mean().item()),
            'critic_v0': float(dbg['critic_v0']),
            'critic_mse': float(dbg['critic_mse']),
            'return_mean': float(dbg['return_mean']),
            'return_std': float(dbg['return_std']),
            'norm_return_mean': float(dbg['norm_return_mean']),
            'norm_return_std': float(dbg['norm_return_std']),
            'value_mean': float(dbg['value_mean']),
            'value_std': float(dbg['value_std']),
            'mean_abs_td_error': float(dbg['mean_abs_td_error']),
            'explained_variance': float(dbg['explained_variance']),
            'adv_mean': float(advantages.mean().item()),
            'adv_std': float(advantages.std(unbiased=False).item()),
            'mgr_grad_norm_preclip': float(mgr_preclip_norm),
            'mgr_grad_norm_postclip': float(mgr_postclip_norm),
            'wkr_grad_norm_preclip': float(wkr_preclip_norm),
            'wkr_grad_norm_postclip': float(wkr_postclip_norm),
            'manager_clip_hit': manager_clip_hit,
            'worker_clip_hit': worker_clip_hit,
            'mgr_entropy': float(mgr_entropy.mean().item()),
            'mgr_nll': float(mgr_nll.mean().item()),
            'policy_loss': float(policy_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'entropy_bonus': float(entropy_bonus.item()),
            'mgr_aux_ce_loss': float(mgr_aux_ce_loss.item()),
            'mgr_aux_weight': float(mgr_aux_weight),
            'worker_aux_ce_loss': float(worker_aux_ce_loss.item()),
            'worker_aux_weight': float(worker_aux_weight),
        }

        reward_components = {
            'r1_pbrs': dbg['r1_pbrs'],
            'r2_subgoal': dbg['r2_subgoal'],
            'r3_goal': dbg['r3_goal'],
            'r4_efficiency': dbg['r4_efficiency'],
            'r5_milestone': dbg['r5_milestone'],
            'r6_explore': dbg['r6_explore'],
            'r7_plan_penalty': plan_penalty,
            'p1_time': dbg['p1_time'],
            'p2_loop': dbg['p2_loop'],
            'p3_fail': dbg['p3_fail'],
        }
        for name, values in reward_components.items():
            diag[f'success_{name}_sum'] = float(values[success_mask].sum().item())
            diag[f'fail_{name}_sum'] = float(values[fail_mask].sum().item())

        num_rank_cols = hit_mask.size(1)
        for rank in range(4):
            opp_mask = plan_lengths >= float(rank + 1)
            if rank < num_rank_cols:
                diag[f'subgoal_rank{rank + 1}_hit_count'] = int(hit_mask[:, rank][opp_mask].sum().item())
                diag[f'subgoal_rank{rank + 1}_opp_count'] = int(opp_mask.sum().item())
            else:
                diag[f'subgoal_rank{rank + 1}_hit_count'] = 0
                diag[f'subgoal_rank{rank + 1}_opp_count'] = 0
        rank5plus_valid = valid_plan_mask[:, 4:]
        diag['subgoal_rank5plus_hit_count'] = int((hit_mask[:, 4:] & rank5plus_valid).sum().item())
        diag['subgoal_rank5plus_opp_count'] = int(rank5plus_valid.sum().item())

        sample_score = (
            success_mask.float() * 1_000_000.0
            + dbg['progress_ratio'] * 1_000.0
            + plan_utilization * 100.0
            - dbg['final_goal_dist'] * 1e-3
        )
        best_idx = int(sample_score.argmax().item())
        sample_plan = sequences[best_idx][valid_plan_mask[best_idx]].detach().cpu().tolist()
        sample_hit_mask = hit_mask[best_idx][: int(plan_lengths[best_idx].item())].detach().cpu().tolist()
        sample_payload = {
            'ep': int(ep),
            'start_idx': int(dbg['start_idx']),
            'goal_idx': int(dbg['goal_idx']),
            'batch_index': best_idx,
            'success': bool(success_mask[best_idx].item()),
            'final_reward': float(total_rewards_with_ckpt[best_idx].item()),
            'normalized_return': float(norm_returns[best_idx].item()),
            'checkpoint_reward': float(checkpoint_reward[best_idx].item()),
            'plan': [int(x) for x in sample_plan],
            'plan_length': int(plan_lengths[best_idx].item()),
            'hit_ranks': [idx + 1 for idx, hit in enumerate(sample_hit_mask) if hit],
            'generated_subgoals': int(plan_lengths[best_idx].item()),
            'reached_subgoals': int(reached_subgoals[best_idx].item()),
            'plan_utilization': float(plan_utilization[best_idx].item()),
            'plan_density': float(plan_density[best_idx].item()),
            'plan_len_ref': float(plan_len_ref[best_idx].item()),
            'plan_len_min': float(plan_len_min[best_idx].item()),
            'plan_len_max': float(plan_len_max[best_idx].item()),
            'corridor_ratio': float(corridor_ratio[best_idx].item()),
            'anchor_hop_err': float(anchor_hop_error_mean[best_idx].item()),
            'anchor_near': float(anchor_near_ratio[best_idx].item()),
            'spacing_error': float(spacing_error_mean[best_idx].item()),
            'monotonic_violation_rate': float(monotonic_violation_rate[best_idx].item()),
            'manager_corridor_step_rate': float(
                corridor_ok[best_idx].float().sum().item() / max(plan_lengths[best_idx].item(), 1.0)
            ),
            'manager_nonprogress_step_rate': float(
                manager_nonprogress_mask[best_idx].float().sum().item() / max(plan_lengths[best_idx].item(), 1.0)
            ),
            'manager_avg_detour_hops': float(
                (manager_detour_hops[best_idx] * valid_plan_mask[best_idx].float()).sum().item()
                / max(plan_lengths[best_idx].item(), 1.0)
            ),
            'manager_eos_near_goal': bool(eos_near_goal_mask[best_idx].item()),
            'remaining_hops_when_eos': float(remaining_goal_hops_at_eos[best_idx].item()),
            'first_subgoal_hops': float(first_subgoal_hops[best_idx].item()),
            'segment_budget_error': float(segment_budget_error_mean[best_idx].item()),
            'first_segment_budget_err': float(first_segment_budget_err[best_idx].item()),
            'first_segment_overshoot': float(first_segment_overshoot[best_idx].item()),
            'frontloaded_overshoot_rate': float(frontloaded_overshoot_rate[best_idx].item()),
            'mean_inter_subgoal_hops': float(mean_inter_subgoal_hops[best_idx].item()),
            'max_inter_subgoal_hops': float(max_inter_subgoal_hops[best_idx].item()),
            'final_goal_distance': float(dbg['final_goal_dist'][best_idx].item()),
            'progress': float(dbg['progress_ratio'][best_idx].item()),
            'goal_progress_per_100_steps': float(goal_progress_per_100_steps[best_idx].item()),
            'subgoal_progress_per_100_steps': float(subgoal_progress_per_100_steps[best_idx].item()),
            'switch_reason_exact': bool(switch_reason_exact_mask[best_idx].item()),
            'switch_reason_soft': bool(switch_reason_soft_mask[best_idx].item()),
            'soft_arrival': bool(soft_arrival_mask[best_idx].item()),
            'skip': bool(skip_mask[best_idx].item()),
            'skip_due_to_passed': bool(skip_due_to_passed_mask[best_idx].item()),
            'skip_due_to_goal_progress': bool(skip_due_to_goal_progress_mask[best_idx].item()),
            'skip_due_to_unreachable': bool(skip_due_to_unreachable_mask[best_idx].item()),
            'reached_last_subgoal': bool(reached_last_subgoal_mask[best_idx].item()),
            'goal_after_last_subgoal': bool(goal_after_last_subgoal_mask[best_idx].item()),
            'remaining_goal_hops_at_handoff': float(remaining_goal_hops_at_handoff[best_idx].item()),
            'goal_hops_1step_after_handoff': float(goal_hops_1step_after_handoff[best_idx].item()),
            'goal_dist_after_last_subgoal': float(goal_dist_after_last_subgoal[best_idx].item()),
            'steps_after_last_subgoal': float(steps_after_last_subgoal[best_idx].item()),
            'ptr_dwell_mean': float(ptr_dwell_mean[best_idx].item()),
            'post_last_sg_success': bool(post_last_sg_success_mask[best_idx].item()),
            'post_last_sg_stagnation': bool(post_last_sg_stagnation_mask[best_idx].item()),
            'steps': int(dbg['step_counts'][best_idx].item()),
            'min_hop_to_active_subgoal': float(min_hop_to_active[best_idx].item()),
            'max_subgoal_progress': float(max_subgoal_progress[best_idx].item()),
            'selection_score': float(sample_score[best_idx].item()),
        }
        return diag, sample_payload

    def _aggregate_debug_window(self):
        if not self._debug_window:
            return None, None

        diags = [entry['diag'] for entry in self._debug_window]
        samples = [entry['sample'] for entry in self._debug_window]
        num_eps = len(diags)

        def avg(key):
            return sum(d.get(key, 0.0) for d in diags) / max(num_eps, 1)

        def total(key):
            return sum(d.get(key, 0.0) for d in diags)

        def minv(key):
            vals = [d.get(key, 0.0) for d in diags]
            return min(vals) if vals else 0.0

        def maxv(key):
            vals = [d.get(key, 0.0) for d in diags]
            return max(vals) if vals else 0.0

        total_samples = total('sample_count')
        total_success = total('success_count')
        total_fail = total('fail_count')
        total_generated = total('generated_subgoal_count')
        total_reached = total('reached_subgoal_count')

        summary = {
            'interval_episodes': num_eps,
            'ep': int(diags[-1]['ep']),
            'mgr_lr': diags[-1]['mgr_lr'],
            'wkr_lr': diags[-1]['wkr_lr'],
            'loss_mean': avg('loss'),
            'success_ema': diags[-1]['success_ema'],
            'sample_count': int(total_samples),
            'success_count': int(total_success),
            'fail_count': int(total_fail),
            'success_rate': self._safe_div(total_success, total_samples),
            'goal_hit_rate': self._safe_div(total('goal_hit_count'), total_samples),
            'loop_fail_rate': self._safe_div(total('loop_fail_count'), total_samples),
            'stagnation_fail_rate': self._safe_div(total('stagnation_fail_count'), total_samples),
            'far_plan_rate': self._safe_div(total('far_plan_count'), total_samples),
            'plan_under_rate': self._safe_div(total('plan_under_count'), total_samples),
            'plan_over_rate': self._safe_div(total('plan_over_count'), total_samples),
            'reached_last_subgoal_rate': self._safe_div(total('reached_last_subgoal_count'), total_samples),
            'goal_after_last_subgoal_rate': self._safe_div(total('goal_after_last_subgoal_count'), total_samples),
            'soft_arrival_rate': self._safe_div(total('soft_arrival_count'), total_samples),
            'skip_rate': self._safe_div(total('skip_count'), total_samples),
            'switch_reason_exact_rate': self._safe_div(total('switch_reason_exact_count'), total_samples),
            'switch_reason_soft_rate': self._safe_div(total('switch_reason_soft_count'), total_samples),
            'skip_due_to_passed_rate': self._safe_div(total('skip_due_to_passed_count'), total_samples),
            'skip_due_to_goal_progress_rate': self._safe_div(total('skip_due_to_goal_progress_count'), total_samples),
            'skip_due_to_unreachable_rate': self._safe_div(total('skip_due_to_unreachable_count'), total_samples),
            'post_last_sg_success_rate': self._safe_div(total('post_last_sg_success_count'), total('goal_after_last_subgoal_count')),
            'post_last_sg_stagnation_rate': self._safe_div(total('post_last_sg_stagnation_count'), total('goal_after_last_subgoal_count')),
        }

        mean_keys = [
            'r1_pbrs', 'r2_subgoal', 'r3_goal', 'r4_efficiency', 'r5_milestone', 'r6_explore', 'r7_plan_penalty',
            'p1_time', 'p2_loop', 'p3_fail', 'total_reward', 'reward_std', 'ckpt_reward_mean',
            'ckpt_reward_std', 'ckpt_quality_mean', 'reward_mean_with_ckpt', 'reward_std_with_ckpt', 'final_goal_dist_mean',
            'progress_mean', 'plan_len_mean', 'plan_len_ref_mean', 'anchor_hop_err_mean', 'anchor_near_rate',
            'spacing_error_mean', 'monotonic_violation_rate', 'plan_terminated_rate', 'plan_empty_rate',
            'plan_unique_ratio', 'plan_density_mean', 'expert_density_gap_mean',
            'corridor_ratio_mean', 'mean_inter_subgoal_hops_mean', 'max_inter_subgoal_hops_mean',
            'wkr_entropy_mean', 'avg_steps', 'critic_v0', 'critic_mse', 'return_mean',
            'return_std', 'norm_return_mean', 'norm_return_std', 'value_mean', 'value_std', 'mean_abs_td_error', 'explained_variance',
            'adv_mean', 'adv_std', 'mgr_grad_norm_preclip', 'mgr_grad_norm_postclip',
            'wkr_grad_norm_preclip', 'wkr_grad_norm_postclip', 'mgr_entropy', 'mgr_nll',
            'policy_loss', 'critic_loss', 'entropy_bonus', 'mgr_aux_ce_loss', 'mgr_aux_weight',
            'worker_aux_ce_loss', 'worker_aux_weight'
        ]
        for key in mean_keys:
            summary[key] = avg(key)

        minmax_pairs = [
            ('reward_min', minv), ('reward_max', maxv),
            ('reward_min_with_ckpt', minv), ('reward_max_with_ckpt', maxv),
            ('final_goal_dist_min', minv), ('final_goal_dist_max', maxv),
            ('progress_min', minv), ('progress_max', maxv),
            ('plan_len_min', minv), ('plan_len_max', maxv),
            ('plan_len_ref_min', minv), ('plan_len_ref_max', maxv),
            ('plan_density_min', minv), ('plan_density_max', maxv),
            ('max_inter_subgoal_hops_max', maxv),
            ('steps_min', minv), ('steps_max', maxv),
            ('min_hop_to_active_subgoal_min', minv), ('min_hop_to_active_subgoal_max', maxv),
            ('max_subgoal_progress_min', minv), ('max_subgoal_progress_max', maxv),
        ]
        for key, reducer in minmax_pairs:
            summary[key] = reducer(key)

        summary['subgoal_reached'] = total_reached
        summary['subgoal_total'] = total_generated
        summary['subgoal_rate'] = self._safe_div(total_reached, total_generated)
        summary['plan_utilization'] = self._safe_div(total_reached, total_generated)
        summary['expert_density_low'] = diags[-1]['expert_density_low']
        summary['expert_density_high'] = diags[-1]['expert_density_high']
        summary['plan_len_target_mean'] = self._safe_div(total('plan_len_target_sum'), total_samples)
        summary['plan_len_min_target_mean'] = self._safe_div(total('plan_len_min_target_sum'), total_samples)
        summary['plan_len_max_target_mean'] = self._safe_div(total('plan_len_max_target_sum'), total_samples)
        summary['anchor_hop_err_mean'] = self._safe_div(total('anchor_hop_err_sum'), total_samples)
        summary['anchor_near_rate'] = self._safe_div(total('anchor_near_sum'), total_samples)
        summary['spacing_error_mean'] = self._safe_div(total('spacing_error_sum'), total_samples)
        summary['monotonic_violation_rate'] = self._safe_div(total('monotonic_violation_sum'), total_samples)

        summary['success_total_reward_mean'] = self._safe_div(total('success_total_reward_sum'), total_success)
        summary['fail_total_reward_mean'] = self._safe_div(total('fail_total_reward_sum'), total_fail)
        summary['success_norm_return_mean'] = self._safe_div(total('success_norm_return_sum'), total_success)
        summary['fail_norm_return_mean'] = self._safe_div(total('fail_norm_return_sum'), total_fail)
        summary['fail_shaping_mean'] = self._safe_div(total('fail_shaping_sum'), total_fail)
        summary['goal_share_mean'] = self._safe_div(total('goal_share_sum'), total('goal_share_count'))
        summary['corridor_ratio_success_mean'] = self._safe_div(total('corridor_ratio_success_sum'), total_success)
        summary['corridor_ratio_fail_mean'] = self._safe_div(total('corridor_ratio_fail_sum'), total_fail)
        summary['manager_corridor_step_rate'] = self._safe_div(
            total('manager_corridor_step_count'),
            total('manager_step_total'),
        )
        summary['manager_nonprogress_step_rate'] = self._safe_div(
            total('manager_nonprogress_step_count'),
            total('manager_step_total'),
        )
        summary['manager_avg_detour_hops'] = self._safe_div(
            total('manager_detour_hops_sum'),
            total('manager_step_total'),
        )
        summary['manager_eos_near_goal_rate'] = self._safe_div(
            total('manager_eos_near_goal_count'),
            total('manager_eos_count'),
        )
        summary['remaining_hops_when_eos_mean'] = self._safe_div(
            total('remaining_hops_when_eos_sum'),
            total('manager_eos_count'),
        )
        summary['first_subgoal_hops_mean'] = self._safe_div(total('first_subgoal_hops_sum'), total('first_subgoal_hops_count'))
        summary['segment_budget_error_mean'] = self._safe_div(total('segment_budget_error_sum'), total_samples)
        summary['first_segment_budget_err_mean'] = self._safe_div(total('first_segment_budget_err_sum'), total_samples)
        summary['first_segment_overshoot_mean'] = self._safe_div(total('first_segment_overshoot_sum'), total_samples)
        summary['frontloaded_overshoot_rate'] = self._safe_div(total('frontloaded_overshoot_sum'), total_samples)
        summary['min_hop_to_active_subgoal_mean'] = self._safe_div(total('min_hop_to_active_subgoal_sum'), total_samples)
        summary['max_subgoal_progress_mean'] = self._safe_div(total('max_subgoal_progress_sum'), total_samples)
        summary['goal_progress_per_100_steps_mean'] = self._safe_div(total('goal_progress_per_100_steps_sum'), total_samples)
        summary['subgoal_progress_per_100_steps_mean'] = self._safe_div(total('subgoal_progress_per_100_steps_sum'), total_samples)
        summary['goal_progress_per100'] = summary['goal_progress_per_100_steps_mean']
        summary['subgoal_progress_per100'] = summary['subgoal_progress_per_100_steps_mean']
        summary['goal_dist_after_last_sg_mean'] = self._safe_div(
            total('goal_dist_after_last_sg_sum'),
            total('goal_after_last_subgoal_count'),
        )
        summary['steps_after_last_sg_mean'] = self._safe_div(
            total('steps_after_last_sg_sum'),
            total('goal_after_last_subgoal_count'),
        )
        summary['remaining_goal_hops_at_handoff_mean'] = self._safe_div(
            total('remaining_goal_hops_at_handoff_sum'),
            total('goal_after_last_subgoal_count'),
        )
        summary['goal_hops_1step_after_handoff_mean'] = self._safe_div(
            total('goal_hops_1step_after_handoff_sum'),
            total('goal_after_last_subgoal_count'),
        )
        summary['ptr_dwell_steps_mean'] = self._safe_div(total('ptr_dwell_sum'), total_samples)

        split_components = ['r1_pbrs', 'r2_subgoal', 'r3_goal', 'r4_efficiency', 'r5_milestone', 'r6_explore', 'r7_plan_penalty', 'p1_time', 'p2_loop', 'p3_fail']
        for name in split_components:
            summary[f'success_{name}_mean'] = self._safe_div(total(f'success_{name}_sum'), total_success)
            summary[f'fail_{name}_mean'] = self._safe_div(total(f'fail_{name}_sum'), total_fail)

        for rank in range(4):
            hits = total(f'subgoal_rank{rank + 1}_hit_count')
            opps = total(f'subgoal_rank{rank + 1}_opp_count')
            summary[f'subgoal_rank{rank + 1}_rate'] = self._safe_div(hits, opps)
        summary['subgoal_rank5plus_rate'] = self._safe_div(
            total('subgoal_rank5plus_hit_count'),
            total('subgoal_rank5plus_opp_count'),
        )

        summary['manager_clip_hit_rate'] = self._safe_div(total('manager_clip_hit'), num_eps)
        summary['worker_clip_hit_rate'] = self._safe_div(total('worker_clip_hit'), num_eps)

        representative = max(samples, key=lambda item: item.get('selection_score', float('-inf')))
        return summary, representative

    def _append_debug_csv(self, row):
        if self._debug_csv_fields is None:
            self._debug_csv_fields = list(row.keys())
            with open(self._debug_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._debug_csv_fields)
                writer.writeheader()
                writer.writerow(row)
            return

        with open(self._debug_csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._debug_csv_fields)
            writer.writerow(row)

    def _append_debug_sample(self, ep, summary, sample):
        payload = {
            'interval_end_ep': int(ep),
            'interval_episodes': int(summary['interval_episodes']),
            'summary': {
                'success_rate': float(summary['success_rate']),
                'goal_hit_rate': float(summary['goal_hit_rate']),
                'plan_utilization': float(summary['plan_utilization']),
                'stagnation_fail_rate': float(summary['stagnation_fail_rate']),
                'corridor_ratio_mean': float(summary['corridor_ratio_mean']),
                'corridor_ratio_success_mean': float(summary['corridor_ratio_success_mean']),
                'corridor_ratio_fail_mean': float(summary['corridor_ratio_fail_mean']),
                'manager_corridor_step_rate': float(summary['manager_corridor_step_rate']),
                'manager_nonprogress_step_rate': float(summary['manager_nonprogress_step_rate']),
                'manager_avg_detour_hops': float(summary['manager_avg_detour_hops']),
                'manager_eos_near_goal_rate': float(summary['manager_eos_near_goal_rate']),
                'plan_len_ref_mean': float(summary['plan_len_ref_mean']),
                'plan_under_rate': float(summary['plan_under_rate']),
                'plan_over_rate': float(summary['plan_over_rate']),
                'anchor_hop_err_mean': float(summary['anchor_hop_err_mean']),
                'anchor_near_rate': float(summary['anchor_near_rate']),
                'spacing_error_mean': float(summary['spacing_error_mean']),
                'monotonic_violation_rate': float(summary['monotonic_violation_rate']),
                'segment_budget_error_mean': float(summary['segment_budget_error_mean']),
                'first_segment_budget_err_mean': float(summary['first_segment_budget_err_mean']),
                'first_segment_overshoot_mean': float(summary['first_segment_overshoot_mean']),
                'frontloaded_overshoot_rate': float(summary['frontloaded_overshoot_rate']),
                'remaining_hops_when_eos_mean': float(summary['remaining_hops_when_eos_mean']),
                'reached_last_subgoal_rate': float(summary['reached_last_subgoal_rate']),
                'goal_after_last_subgoal_rate': float(summary['goal_after_last_subgoal_rate']),
                'goal_dist_after_last_sg_mean': float(summary['goal_dist_after_last_sg_mean']),
                'steps_after_last_sg_mean': float(summary['steps_after_last_sg_mean']),
                'ptr_dwell_steps_mean': float(summary['ptr_dwell_steps_mean']),
                'switch_reason_exact_rate': float(summary['switch_reason_exact_rate']),
                'switch_reason_soft_rate': float(summary['switch_reason_soft_rate']),
                'soft_arrival_rate': float(summary['soft_arrival_rate']),
                'skip_rate': float(summary['skip_rate']),
                'skip_due_to_passed_rate': float(summary['skip_due_to_passed_rate']),
                'skip_due_to_goal_progress_rate': float(summary['skip_due_to_goal_progress_rate']),
                'skip_due_to_unreachable_rate': float(summary['skip_due_to_unreachable_rate']),
                'remaining_goal_hops_at_handoff_mean': float(summary['remaining_goal_hops_at_handoff_mean']),
                'goal_hops_1step_after_handoff_mean': float(summary['goal_hops_1step_after_handoff_mean']),
                'post_last_sg_success_rate': float(summary['post_last_sg_success_rate']),
                'post_last_sg_stagnation_rate': float(summary['post_last_sg_stagnation_rate']),
                'plan_density_mean': float(summary['plan_density_mean']),
                'expert_density_gap_mean': float(summary['expert_density_gap_mean']),
                'fail_shaping_mean': float(summary['fail_shaping_mean']),
                'goal_share_mean': float(summary['goal_share_mean']),
                'explained_variance': float(summary['explained_variance']),
            },
            'sample': sample,
        }
        with open(self._debug_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _emit_debug_window(self, ep, success_ema, current_loss, pbar):
        summary, representative = self._aggregate_debug_window()
        if summary is None:
            return

        lines = [
            f"  ┌─── [DEBUG Ep {ep} | last {summary['interval_episodes']} eps] Reward Breakdown (window mean) ───",
            f"  │ R1 PBRS:      {summary['r1_pbrs']:.3f}",
            f"  │ R2 Subgoal:   {summary['r2_subgoal']:.3f}",
            f"  │ R3 Goal:      {summary['r3_goal']:.3f}",
            f"  │ R4 Efficiency:{summary['r4_efficiency']:.3f}",
            f"  │ R5 Milestone: {summary['r5_milestone']:.3f}",
            f"  │ R6 Explore:   {summary['r6_explore']:.3f}",
            f"  │ R7 PlanPen.:  {summary['r7_plan_penalty']:.3f}",
            f"  │ P1 Time:      {summary['p1_time']:.3f}",
            f"  │ P2 Loop:      {summary['p2_loop']:.3f}",
            f"  │ P3 Fail:      {summary['p3_fail']:.3f}",
            f"  │ Total(Base):  {summary['total_reward']:.3f}",
            f"  │ Ckpt Reward:  {summary['ckpt_reward_mean']:.3f} (quality {summary['ckpt_quality_mean']:.3f}, std {summary['ckpt_reward_std']:.3f})",
            f"  │ Total(Final): {summary['reward_mean_with_ckpt']:.3f}",
            f"  │ Reward Stats: std={summary['reward_std_with_ckpt']:.3f}, min={summary['reward_min_with_ckpt']:.3f}, max={summary['reward_max_with_ckpt']:.3f}",
            f"  ├─── Reward Alignment ───",
            f"  │ Goal Hit Rate: {summary['goal_hit_rate']*100:.1f}%",
            f"  │ Success/Fail:  {summary['success_total_reward_mean']:.3f} / {summary['fail_total_reward_mean']:.3f}",
            f"  │ Norm Ret S/F: {summary['success_norm_return_mean']:.3f} / {summary['fail_norm_return_mean']:.3f}",
            f"  │ Fail Shaping:  {summary['fail_shaping_mean']:.3f}",
            f"  │ Goal Share:    {summary['goal_share_mean']*100:.1f}%",
            f"  ├─── Batch Outcome ───",
            f"  │ Success Rate: {summary['success_rate']*100:.1f}% ({summary['success_count']:.0f}/{summary['sample_count']:.0f})",
            f"  │ Success EMA:  {success_ema*100:.1f}%",
            f"  │ Loop Fail:    {summary['loop_fail_rate']*100:.1f}%",
            f"  │ Stagnation Fail: {summary['stagnation_fail_rate']*100:.1f}%",
            f"  │ Goal Dist:    mean={summary['final_goal_dist_mean']:.2f}, min={summary['final_goal_dist_min']:.2f}, max={summary['final_goal_dist_max']:.2f}",
            f"  │ Progress:     mean={summary['progress_mean']*100:.1f}%, min={summary['progress_min']*100:.1f}%, max={summary['progress_max']*100:.1f}%",
            f"  ├─── Plan Diagnostics ───",
            f"  │ Plan Length:  mean={summary['plan_len_mean']:.1f}, min={summary['plan_len_min']:.0f}, max={summary['plan_len_max']:.0f}",
            f"  │ Target Len:   ref={summary['plan_len_ref_mean']:.1f}, band={summary['plan_len_min_target_mean']:.1f}~{summary['plan_len_max_target_mean']:.1f}",
            f"  │ EOS Rate:     {summary['plan_terminated_rate']*100:.1f}%",
            f"  │ Empty Plans:  {summary['plan_empty_rate']*100:.1f}%",
            f"  │ Unique Ratio: {summary['plan_unique_ratio']*100:.1f}%",
            f"  │ Far ShortPlan:{summary['far_plan_rate']*100:.1f}%",
            f"  │ Plan Under/Over: {summary['plan_under_rate']*100:.1f}% / {summary['plan_over_rate']*100:.1f}%",
            f"  │ Anchor Err/Near: {summary['anchor_hop_err_mean']:.2f} / {summary['anchor_near_rate']*100:.1f}%",
            f"  │ Spacing/Mono: {summary['spacing_error_mean']:.3f} / {summary['monotonic_violation_rate']*100:.1f}%",
            f"  │ SegBudget/FstErr: {summary['segment_budget_error_mean']:.2f} / {summary['first_segment_budget_err_mean']:.2f}",
            f"  │ FirstOvershoot/Frontload: {summary['first_segment_overshoot_mean']:.2f} / {summary['frontloaded_overshoot_rate']*100:.1f}%",
            f"  │ Density:      mean={summary['plan_density_mean']:.3f}, band={summary['expert_density_low']:.3f}~{summary['expert_density_high']:.3f}, gap={summary['expert_density_gap_mean']:.3f}",
            f"  │ Corridor:     mean={summary['corridor_ratio_mean']*100:.1f}%, succ={summary['corridor_ratio_success_mean']*100:.1f}%, fail={summary['corridor_ratio_fail_mean']*100:.1f}%",
            f"  │ SG Hops:      first={summary['first_subgoal_hops_mean']:.2f}, inter={summary['mean_inter_subgoal_hops_mean']:.2f}, max={summary['max_inter_subgoal_hops_max']:.2f}",
            f"  ├─── Worker Diagnostics ───",
            f"  │ Subgoal Reach: {summary['subgoal_reached']:.0f}/{summary['subgoal_total']:.0f} ({summary['subgoal_rate']*100:.1f}%)",
            f"  │ Plan Utiliz.: {summary['plan_utilization']*100:.1f}%",
            f"  │ Hit@1/2/3/4/5+: {summary['subgoal_rank1_rate']*100:.1f}/{summary['subgoal_rank2_rate']*100:.1f}/{summary['subgoal_rank3_rate']*100:.1f}/{summary['subgoal_rank4_rate']*100:.1f}/{summary['subgoal_rank5plus_rate']*100:.1f}",
            f"  │ Active SG Hop: mean={summary['min_hop_to_active_subgoal_mean']:.2f}, min={summary['min_hop_to_active_subgoal_min']:.2f}, max={summary['min_hop_to_active_subgoal_max']:.2f}",
            f"  │ Max SG Prog.: mean={summary['max_subgoal_progress_mean']*100:.1f}%, min={summary['max_subgoal_progress_min']*100:.1f}%, max={summary['max_subgoal_progress_max']*100:.1f}%",
            f"  │ LastSG/Handoff: {summary['reached_last_subgoal_rate']*100:.1f}% / {summary['goal_after_last_subgoal_rate']*100:.1f}%",
            f"  │ PostLastSG: succ={summary['post_last_sg_success_rate']*100:.1f}%, stag={summary['post_last_sg_stagnation_rate']*100:.1f}%",
            f"  │ Switch Exact/Soft: {summary['switch_reason_exact_rate']*100:.1f}% / {summary['switch_reason_soft_rate']*100:.1f}%",
            f"  │ GoalDist/Steps after LastSG: {summary['goal_dist_after_last_sg_mean']:.2f} / {summary['steps_after_last_sg_mean']:.2f}",
            f"  │ Handoff GoalHop now/next: {summary['remaining_goal_hops_at_handoff_mean']:.2f} / {summary['goal_hops_1step_after_handoff_mean']:.2f}",
            f"  │ Soft/Skip(T/P/G/U): {summary['soft_arrival_rate']*100:.1f}% / {summary['skip_rate']*100:.1f}% / {summary['skip_due_to_passed_rate']*100:.1f}% / {summary['skip_due_to_goal_progress_rate']*100:.1f}% / {summary['skip_due_to_unreachable_rate']*100:.1f}%",
            f"  │ Ptr Dwell:    mean={summary['ptr_dwell_steps_mean']:.2f}",
            f"  │ Goal/SG per100: {summary['goal_progress_per_100_steps_mean']:.2f} / {summary['subgoal_progress_per_100_steps_mean']:.2f}",
            f"  │ Wkr Entropy:  {summary['wkr_entropy_mean']:.3f}",
            f"  │ Wkr Steps:    mean={summary['avg_steps']:.1f}, min={summary['steps_min']:.0f}, max={summary['steps_max']:.0f}",
            f"  ├─── Critic & Advantage ───",
            f"  │ Critic V0:    {summary['critic_v0']:.3f}",
            f"  │ Critic MSE:   {summary['critic_mse']:.3f}",
            f"  │ Return Mean:  {summary['return_mean']:.3f} (std {summary['return_std']:.3f})",
            f"  │ Norm Return:  {summary['norm_return_mean']:.3f} (std {summary['norm_return_std']:.3f})",
            f"  │ Value Mean:   {summary['value_mean']:.3f} (std {summary['value_std']:.3f})",
            f"  │ TD |err|:     {summary['mean_abs_td_error']:.3f}",
            f"  │ Expl. Var:    {summary['explained_variance']:.3f}",
            f"  │ Adv Mean:     {summary['adv_mean']:.3f}",
            f"  │ Adv Std:      {summary['adv_std']:.3f}",
            f"  ├─── Gradient Norms ───",
            f"  │ Manager:      pre={summary['mgr_grad_norm_preclip']:.4f}, post={summary['mgr_grad_norm_postclip']:.4f}, clip-hit={summary['manager_clip_hit_rate']*100:.1f}%",
            f"  │ Worker:       pre={summary['wkr_grad_norm_preclip']:.4f}, post={summary['wkr_grad_norm_postclip']:.4f}, clip-hit={summary['worker_clip_hit_rate']*100:.1f}%",
            f"  ├─── Manager Diagnostics ───",
            f"  │ Mgr Entropy:  {summary['mgr_entropy']:.3f}",
            f"  │ Mgr NLL:      {summary['mgr_nll']:.3f}",
            f"  │ CorridorStep: {summary['manager_corridor_step_rate']*100:.1f}%",
            f"  │ NonProgress:  {summary['manager_nonprogress_step_rate']*100:.1f}%",
            f"  │ Avg Detour:   {summary['manager_avg_detour_hops']:.2f}",
            f"  │ EOS<=2 Rate:  {summary['manager_eos_near_goal_rate']*100:.1f}%",
            f"  │ RemHops@EOS:  {summary['remaining_hops_when_eos_mean']:.2f}",
            f"  │ Policy Loss:  {summary['policy_loss']:.3f}",
            f"  │ Critic Loss:  {summary['critic_loss']:.3f}",
            f"  │ Ent Bonus:    {summary['entropy_bonus']:.3f}",
            f"  │ Aux CE:       {summary['mgr_aux_ce_loss']:.3f} @ w={summary['mgr_aux_weight']:.3f}",
            f"  │ Wkr Aux CE:   {summary['worker_aux_ce_loss']:.3f} @ w={summary['worker_aux_weight']:.3f}",
            f"  └────────────────────────",
        ]

        # [터미널 출력 정리] 상세 로그는 파일에만 기록, 터미널에는 핵심 지표 1줄만 표시
        compact_line = (
            f"  📊 SR={summary['success_rate']*100:.1f}% | "
            f"Plan={summary['plan_density_mean']:.3f} | "
            f"Corr={summary['corridor_mean']*100:.0f}% | "
            f"Under/Over={summary['plan_under_rate']*100:.0f}/{summary['plan_over_rate']*100:.0f}% | "
            f"Hit@1={summary['subgoal_rank1_rate']*100:.0f}%"
        )
        pbar.write(compact_line)

        # 상세 로그는 파일에만 기록
        with open(self._debug_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[Ep {ep}] SuccessEMA: {success_ema*100:.1f}%, Loss: {current_loss:.2f}\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

        csv_row = dict(summary)
        csv_row['success_ema'] = float(success_ema)
        csv_row['loss_last'] = float(current_loss)
        self._append_debug_csv(csv_row)
        self._append_debug_sample(ep, summary, representative)
        self._debug_window = []

    def execute_batch_plan(
        self,
        start_idx,
        goal_idx,
        sequences,
        x_pos,
        plan_adjustment=None,
        plan_diag=None,
        detach_spatial=True,
        collect_worker_aux=False,
        handoff_aux_boost=2.0,
        enable_skip=True,
    ):
        """
        Vectorized Worker Execution
        [Fix 1] torch.no_grad() 제거 → Worker gradient 완전 복원
        VRAM 관리를 위해 batch_size를 줄이는 방식으로 대응
        """
        batch_size = sequences.size(0)
        device = self.device
        row_index = torch.arange(batch_size, device=device)
        goal_idx_tensor = torch.full((batch_size,), int(goal_idx), dtype=torch.long, device=device)

        # Subgoal pointer 초기화
        subgoal_ptrs = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # PAD/EOS/Invalid 토큰을 Goal Index로 대체하여 안전한 시퀀스 생성
        valid_mask = (sequences < self.env.num_nodes) & (sequences >= 0)
        safe_sequences = torch.where(valid_mask, sequences, goal_idx_tensor.unsqueeze(1))
        safe_sequences = torch.cat([safe_sequences, goal_idx_tensor.unsqueeze(1)], dim=1)
        generated_plan_lengths = valid_mask.sum(dim=1)
        subgoal_hit_mask = torch.zeros_like(valid_mask, dtype=torch.bool, device=device)
        corridor_by_rank = plan_diag['corridor_ok'].to(device) if plan_diag is not None else torch.zeros_like(valid_mask, dtype=torch.bool, device=device)
        
        # LSTM 상태 초기화
        hid_dim = self.worker.lstm.hidden_size
        h = torch.zeros(batch_size, hid_dim, device=device)
        c = torch.zeros(batch_size, hid_dim, device=device)
        
        # 추적 변수 초기화
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        log_probs_sum = torch.zeros(batch_size, device=device)
        entropy_sum = torch.zeros(batch_size, device=device) # Worker Entropy
        step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        worker_aux_ce_sum = torch.tensor(0.0, device=device)
        worker_aux_ce_count = torch.tensor(0.0, device=device)
        
        # Value Loss 누적기
        val_loss_sum = torch.zeros(batch_size, device=device)
        
        # Subgoal 보상 누적기
        subgoal_rewards = torch.zeros(batch_size, device=device)
        
        # Loop 감지
        visit_counts = torch.zeros(batch_size, self.env.num_nodes, device=device)
        path_history = [self.env.current_node.clone()]
        failed_early_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stagnated_early_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stagnation_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        reached_last_subgoal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        goal_after_last_subgoal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        soft_arrival_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        skip_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        skip_due_to_passed_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        skip_due_to_goal_progress_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        skip_due_to_unreachable_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        post_last_sg_stagnation_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        goal_dist_after_last_subgoal = torch.zeros(batch_size, dtype=torch.float, device=device)
        steps_after_last_subgoal = torch.zeros(batch_size, dtype=torch.long, device=device)
        ptr_dwell_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        ptr_dwell_total = torch.zeros(batch_size, dtype=torch.float, device=device)
        ptr_dwell_count = torch.zeros(batch_size, dtype=torch.float, device=device)
        in_post_last_subgoal_phase = torch.zeros(batch_size, dtype=torch.bool, device=device)
        switch_reason_exact_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        switch_reason_soft_mask_any = torch.zeros(batch_size, dtype=torch.bool, device=device)
        remaining_goal_hops_at_handoff = torch.zeros(batch_size, dtype=torch.float, device=device)
        goal_hops_1step_after_handoff = torch.zeros(batch_size, dtype=torch.float, device=device)
        pending_goal_hops_after_handoff = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # APSP 기반 거리 추적 초기화
        optimal_dist = self.env.apsp_matrix[start_idx, goal_idx].item()
        optimal_dist = max(optimal_dist, 1.0)  # 0 방지
        
        # Value 예측 저장 (MC Target으로 Critic Loss 계산용)
        traj_values = []
        traj_value_masks = []
        self._last_initial_values = torch.zeros(self.num_pomo, device=device)
        
        # [최적화] 동적 MAX_TOTAL_STEPS 설정 및 한도 축소
        # A* 최단 스텝 수의 대략적인 추정치 기반으로 Max Step 제한 (최소 150, 최대 400)
        optimal_steps_est = optimal_dist / (self.env.max_dist / self.env.num_nodes)
        MAX_TOTAL_STEPS = int(min(400, max(150, optimal_steps_est * 5)))
        
        # [최적화] Loop 감지 기준점 완화 (3 -> 6)
        # 탐험(Exploration)을 보장하기 위해 한도를 너무 타이트하게 잡지 않음
        LOOP_LIMIT = 6
        
        reward_cfg = self.reward_cfg
        POTENTIAL_SCALE = reward_cfg['POTENTIAL_SCALE']
        GAMMA_PBRS = reward_cfg['GAMMA_PBRS']
        SUBGOAL_BASE = reward_cfg['SUBGOAL_BASE']
        SUBGOAL_SCALE = reward_cfg['SUBGOAL_SCALE']
        OPTIMALITY_BONUS = reward_cfg['OPTIMALITY_BONUS']
        GOAL_REWARD = reward_cfg['GOAL_REWARD']
        EFFICIENCY_MAX = reward_cfg['EFFICIENCY_MAX']
        MILESTONE_25 = reward_cfg['MILESTONE_25']
        MILESTONE_50 = reward_cfg['MILESTONE_50']
        MILESTONE_75 = reward_cfg['MILESTONE_75']
        FAIL_PENALTY = reward_cfg['FAIL_PENALTY']
        LOOP_PENALTY_SCALE = reward_cfg['LOOP_PENALTY_SCALE']
        EXPLORATION_BONUS = reward_cfg['EXPLORATION_BONUS']
        BASE_STEP_PENALTY = reward_cfg['BASE_STEP_PENALTY']
        TIME_PRESSURE_SCALE = reward_cfg['TIME_PRESSURE_SCALE']
        POST_HANDOFF_WINDOW = 10
        GOAL_FINISH_PBRS_MULT = 2.0
        GOAL_FINISH_PENALTY_SCALE = 0.5
        
        # 보상 누적기 초기화
        pbrs_sum = torch.zeros(batch_size, device=device)
        subgoal_rewards = torch.zeros(batch_size, device=device)
        step_penalty_sum = torch.zeros(batch_size, device=device)
        loop_penalty_sum = torch.zeros(batch_size, device=device)
        exploration_sum = torch.zeros(batch_size, device=device)
        milestone_sum = torch.zeros(batch_size, device=device)
        
        # PBRS 초기 포텐셜
        max_dist = max(self.env.max_dist, 1.0)
        
        # 마일스톤 플래그
        milestone_25_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        milestone_50_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        milestone_75_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 서브골 최적성 추적
        steps_since_last_subgoal = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_subgoal_node = torch.full((batch_size,), start_idx, dtype=torch.long, device=device)
        
        # 초기 타겟 설정
        target_ptrs = torch.minimum(subgoal_ptrs, generated_plan_lengths)
        targets = torch.gather(safe_sequences, 1, target_ptrs.unsqueeze(1)).squeeze(1)
        final_target_phase = (~(subgoal_ptrs < generated_plan_lengths)).float()
        self.env.update_target_features(targets, final_target_phase)
        current_subgoal_start_hops = self.env.hop_matrix[self.env.current_node, targets].float().clamp(min=1.0)
        best_goal_hops = self.env.hop_matrix[self.env.current_node, goal_idx].float()
        best_subgoal_hops = current_subgoal_start_hops.clone()
        segment_start_goal_dist = self.env.apsp_matrix[self.env.current_node, goal_idx].float()
        min_hop_to_active_subgoal = torch.full((batch_size,), float('inf'), device=device)
        max_subgoal_progress = torch.zeros(batch_size, device=device)
        phi_target_old = -self.env.apsp_matrix[self.env.current_node, targets] / max_dist
        phi_goal_old = -self.env.apsp_matrix[self.env.current_node, goal_idx] / max_dist
        
        # Critic BPTT 관리: 중간 끊김 없이 MAX_TOTAL_STEPS까지 유지하여 Bias 편향 제거
        CRITIC_WINDOW = MAX_TOTAL_STEPS
        
        # [Fix 1] torch.no_grad() 완전 제거
        # Worker의 policy gradient가 학습에 반영되도록 gradient 그래프 유지
        for t in range(MAX_TOTAL_STEPS):
            if not active_mask.any():
                break
            
            # Loop 감지 & Early Exit
            curr_nodes = self.env.current_node.unsqueeze(1)
            visit_counts.scatter_add_(1, curr_nodes, active_mask.float().unsqueeze(1))
            
            curr_visits = torch.gather(visit_counts, 1, curr_nodes).squeeze(1)
            is_looping = (curr_visits > LOOP_LIMIT)
            
            new_failures = is_looping & active_mask
            failed_early_mask = failed_early_mask | new_failures
            active_mask = active_mask & (~is_looping)
            
            if not active_mask.any():
                break
            
            # Subgoal / Goal handoff 관리
            valid_active_subgoal = subgoal_ptrs < generated_plan_lengths
            target_ptrs = torch.minimum(subgoal_ptrs, generated_plan_lengths)
            current_subgoals = torch.gather(safe_sequences, 1, target_ptrs.unsqueeze(1)).squeeze(1)
            current_subgoals = torch.where(valid_active_subgoal, current_subgoals, goal_idx_tensor)
            next_subgoal_ptrs_raw = torch.clamp(subgoal_ptrs + 1, max=safe_sequences.size(1) - 1)
            next_target_ptrs = torch.minimum(next_subgoal_ptrs_raw, generated_plan_lengths)
            next_targets = torch.gather(safe_sequences, 1, next_target_ptrs.unsqueeze(1)).squeeze(1)
            next_exists = (subgoal_ptrs + 1) < generated_plan_lengths

            arrived_subgoal = (self.env.current_node == current_subgoals)
            active_subgoal_hops = self.env.hop_matrix[self.env.current_node, current_subgoals].float()
            next_target_hops = self.env.hop_matrix[self.env.current_node, next_targets].float()
            current_to_next_hops = self.env.hop_matrix[current_subgoals, next_targets].float()
            near_subgoal = valid_active_subgoal & (active_subgoal_hops <= 1.0)
            passed_subgoal = valid_active_subgoal & next_exists & (
                next_target_hops + 1.0 <= current_to_next_hops
            )
            last_target = valid_active_subgoal & (~next_exists)
            # [Refactor: Task 3] Soft-Arrival 조건 완화 - 다음 목표 접근 시 도착 인정
            # Why: 정확히 노드를 밟으려다 Worker가 갇히는 문제 해결
            approaching_next = next_exists & (next_target_hops < current_to_next_hops)
            soft_arrived_subgoal = (~arrived_subgoal) & near_subgoal & (
                passed_subgoal | last_target | approaching_next
            )
            valid_arrived_subgoal = valid_active_subgoal & (arrived_subgoal | soft_arrived_subgoal)
            exact_arrived_subgoal = valid_active_subgoal & arrived_subgoal
            switch_reason_exact_mask_any = switch_reason_exact_mask_any | exact_arrived_subgoal
            switch_reason_soft_mask_any = switch_reason_soft_mask_any | soft_arrived_subgoal
            soft_arrival_mask_any = soft_arrival_mask_any | soft_arrived_subgoal

            adaptive_patience = torch.clamp(
                10 + 2 * current_subgoal_start_hops.long(),
                min=12,
                max=40,
            )
            post_handoff_patience = torch.clamp(
                8 + 2 * self.env.hop_matrix[self.env.current_node, goal_idx].long(),
                min=12,
                max=24,
            )
            effective_patience = torch.where(
                in_post_last_subgoal_phase,
                post_handoff_patience,
                adaptive_patience,
            )
            goal_dist_now = self.env.apsp_matrix[self.env.current_node, goal_idx].float()
            improved_goal = goal_dist_now < (segment_start_goal_dist - 1e-6)
            raw_skip_mask = (
                valid_active_subgoal
                & next_exists
                & (~valid_arrived_subgoal)
                & (stagnation_steps >= effective_patience)
                & (passed_subgoal | improved_goal)
            )
            skip_mask = raw_skip_mask if enable_skip else torch.zeros_like(raw_skip_mask)
            skip_due_to_passed_mask_any = skip_due_to_passed_mask_any | (skip_mask & passed_subgoal)
            skip_due_to_goal_progress_mask_any = skip_due_to_goal_progress_mask_any | (skip_mask & (~passed_subgoal) & improved_goal)
            skip_mask_any = skip_mask_any | skip_mask
            advance_mask = valid_arrived_subgoal | skip_mask

            masked_active_subgoal_hops = torch.where(
                valid_active_subgoal,
                active_subgoal_hops,
                torch.full_like(active_subgoal_hops, float('inf')),
            )
            min_hop_to_active_subgoal = torch.minimum(min_hop_to_active_subgoal, masked_active_subgoal_hops)
            subgoal_progress_now = 1.0 - (
                active_subgoal_hops / current_subgoal_start_hops.clamp(min=1.0)
            )
            subgoal_progress_now = torch.clamp(subgoal_progress_now, min=0.0, max=1.0)
            subgoal_progress_now = torch.where(
                valid_active_subgoal,
                subgoal_progress_now,
                torch.zeros_like(subgoal_progress_now),
            )
            max_subgoal_progress = torch.maximum(max_subgoal_progress, subgoal_progress_now)
            if valid_arrived_subgoal.any():
                row_idx = row_index[valid_arrived_subgoal]
                col_idx = subgoal_ptrs[valid_arrived_subgoal]
                subgoal_hit_mask[row_idx, col_idx] = True
            
            # B: 서브골 도달 보상 (진행률 비례 + 최적성 보너스)
            total_subgoals_count = generated_plan_lengths.float().clamp(min=1.0)
            progress_ratio = torch.clamp(subgoal_ptrs.float() / total_subgoals_count, min=0.0, max=1.0)
            subgoal_bonus = 0.5 * (SUBGOAL_BASE + SUBGOAL_SCALE * progress_ratio)
            corridor_ok_now = torch.gather(corridor_by_rank, 1, subgoal_ptrs.unsqueeze(1)).squeeze(1)
            eligible_subgoal = valid_arrived_subgoal & corridor_ok_now & improved_goal
            subgoal_rewards += eligible_subgoal.float() * subgoal_bonus
            
            # C: 서브골 최적성 보너스 (A* 최단경로 대비 실제 스텝)
            optimal_subgoal_dist = self.env.apsp_matrix[prev_subgoal_node, current_subgoals]
            optimal_subgoal_steps = torch.clamp(optimal_subgoal_dist / max(self.env.max_dist / self.env.num_nodes, 1.0), min=1.0)
            actual_steps = steps_since_last_subgoal.float().clamp(min=1.0)
            opt_ratio = optimal_subgoal_steps / actual_steps
            opt_bonus = OPTIMALITY_BONUS * torch.clamp(opt_ratio, 0.0, 1.0)
            subgoal_rewards += eligible_subgoal.float() * opt_bonus
            
            last_subgoal_advance = advance_mask & (subgoal_ptrs == (generated_plan_lengths - 1))
            reached_last_subgoal_mask = reached_last_subgoal_mask | (valid_arrived_subgoal & last_subgoal_advance)

            # 서브골 도달/스킵 시 추적 변수 리셋
            steps_since_last_subgoal = torch.where(advance_mask, torch.zeros_like(steps_since_last_subgoal), steps_since_last_subgoal)
            prev_subgoal_node = torch.where(advance_mask, current_subgoals, prev_subgoal_node)

            ptr_dwell_total = torch.where(
                advance_mask,
                ptr_dwell_total + ptr_dwell_steps.float(),
                ptr_dwell_total,
            )
            ptr_dwell_count = torch.where(
                advance_mask,
                ptr_dwell_count + 1.0,
                ptr_dwell_count,
            )
            ptr_dwell_steps = torch.where(
                advance_mask,
                torch.zeros_like(ptr_dwell_steps),
                ptr_dwell_steps,
            )

            next_subgoal_ptrs = torch.where(advance_mask, subgoal_ptrs + 1, subgoal_ptrs)
            next_subgoal_ptrs = torch.clamp(next_subgoal_ptrs, max=safe_sequences.size(1) - 1)
            ptr_changed = (next_subgoal_ptrs != subgoal_ptrs)
            subgoal_ptrs = next_subgoal_ptrs

            target_ptrs = torch.minimum(subgoal_ptrs, generated_plan_lengths)
            targets = torch.gather(safe_sequences, 1, target_ptrs.unsqueeze(1)).squeeze(1)
            valid_active_subgoal = subgoal_ptrs < generated_plan_lengths
            final_target_phase = (~valid_active_subgoal).float()
            goal_handoff = ptr_changed & (~valid_active_subgoal)
            goal_after_last_subgoal_mask = goal_after_last_subgoal_mask | goal_handoff
            goal_dist_after_last_subgoal = torch.where(
                goal_handoff & (~in_post_last_subgoal_phase),
                self.env.apsp_matrix[self.env.current_node, goal_idx].float(),
                goal_dist_after_last_subgoal,
            )
            remaining_goal_hops_at_handoff = torch.where(
                goal_handoff,
                self.env.hop_matrix[self.env.current_node, goal_idx].float(),
                remaining_goal_hops_at_handoff,
            )
            pending_goal_hops_after_handoff = pending_goal_hops_after_handoff | goal_handoff
            in_post_last_subgoal_phase = in_post_last_subgoal_phase | goal_handoff
            self.env.update_target_features(targets, final_target_phase)
            if ptr_changed.any():
                next_start_hops = self.env.hop_matrix[self.env.current_node, targets].float().clamp(min=1.0)
                current_subgoal_start_hops = torch.where(
                    ptr_changed,
                    next_start_hops,
                    current_subgoal_start_hops,
                )
                best_subgoal_hops = torch.where(
                    ptr_changed,
                    next_start_hops,
                    best_subgoal_hops,
                )
                next_goal_dist = self.env.apsp_matrix[self.env.current_node, goal_idx].float()
                segment_start_goal_dist = torch.where(
                    ptr_changed,
                    next_goal_dist,
                    segment_start_goal_dist,
                )
                phi_target_rebase = -self.env.apsp_matrix[self.env.current_node, targets] / max_dist
                phi_target_old = torch.where(
                    ptr_changed,
                    phi_target_rebase,
                    phi_target_old,
                )
                stagnation_steps = torch.where(
                    ptr_changed,
                    torch.zeros_like(stagnation_steps),
                    stagnation_steps,
                )
                steps_since_last_subgoal = torch.where(
                    ptr_changed,
                    torch.zeros_like(steps_since_last_subgoal),
                    steps_since_last_subgoal,
                )
                best_goal_hops = torch.where(
                    goal_handoff,
                    self.env.hop_matrix[self.env.current_node, goal_idx].float(),
                    best_goal_hops,
                )
                steps_after_last_subgoal = torch.where(
                    goal_handoff,
                    torch.zeros_like(steps_after_last_subgoal),
                    steps_after_last_subgoal,
                )
                ptr_dwell_steps = torch.where(
                    goal_handoff,
                    torch.zeros_like(ptr_dwell_steps),
                    ptr_dwell_steps,
                )

            if goal_handoff.any():
                obs_target_idx = self.env.target_node.clone()
                pbrs_target_idx = targets.clone()
                aux_target_idx = targets.clone()
                debug_target_idx = targets.clone()
                mismatch = (
                    (~torch.eq(obs_target_idx, targets))
                    | (~torch.eq(pbrs_target_idx, targets))
                    | (~torch.eq(aux_target_idx, targets))
                    | (~torch.eq(debug_target_idx, targets))
                ) & goal_handoff
                if mismatch.any():
                    msg = (
                        "Handoff target mismatch detected: "
                        f"obs={obs_target_idx[mismatch].tolist()} active={targets[mismatch].tolist()}"
                    )
                    if self.debug_mode:
                        raise AssertionError(msg)
                    if not self._handoff_target_warned:
                        print(f"⚠️ {msg}")
                        self._handoff_target_warned = True
                if self.debug_mode:
                    handoff_goal_hops_now = self.env.hop_matrix[self.env.current_node, goal_idx].float()
                    assert torch.equal(self.env.target_node[goal_handoff], targets[goal_handoff]), "handoff env target mismatch"
                    assert torch.all(stagnation_steps[goal_handoff] == 0), "handoff stagnation_steps not reset"
                    assert torch.all(steps_after_last_subgoal[goal_handoff] == 0), "handoff steps_after_last_subgoal not reset"
                    assert torch.allclose(
                        best_goal_hops[goal_handoff],
                        handoff_goal_hops_now[goal_handoff],
                    ), "handoff best_goal_hops not rebased"
                    assert torch.allclose(
                        best_subgoal_hops[ptr_changed],
                        current_subgoal_start_hops[ptr_changed],
                    ), "handoff best_subgoal_hops not rebased"

            # Goal 도달 체크
            arrived_goal = (self.env.current_node == goal_idx)
            active_mask = active_mask & (~arrived_goal)
            
            if not active_mask.any():
                break

            active_before_step = active_mask.clone()
            goal_finish_window = in_post_last_subgoal_phase & (steps_after_last_subgoal < POST_HANDOFF_WINDOW)

            # Worker Action (Gradient 유지)
            # Env X: [x, y, is_curr(2), is_tgt(3), visit(4), dist(5), dir_x(6), dir_y(7), is_final_target_phase(8), hop_dist(9)]
            # Worker: [x, y, is_curr, is_tgt, dist, dir_x, dir_y, is_final_target_phase, hop_dist] (visit 제외 = 9채널)
            env_x = self.env.pyg_data.x
            wkr_in = torch.cat([env_x[:, :4], env_x[:, 5:]], dim=1) # [B*N, 9]

            # [Fix] 동적으로 edge_attr 5D 슬라이싱 (환경이 변할 수 있으므로 매 스텝 계산)
            ea = self.env.pyg_data.edge_attr[:, 0:1]  # Phase 1: length만 사용
            
            # AMP: Worker forward pass를 autocast로 감싸 VRAM 절감 (LSTM BPTT 메모리 핵심)
            # RL에서는 spatial encoder를 고정해 매-step full-graph backward 비용을 줄인다.
            with autocast('cuda', enabled=self.use_amp):
                scores, h_n, c_n, value_pred = self.worker.predict_next_hop(
                    wkr_in, self.env.pyg_data.edge_index, h, c, self.env.pyg_data.batch,
                    detach_spatial=detach_spatial, edge_attr=ea
                )
            
            # Value 예측 저장 (Critic 학습 및 초기 예측 Advantage용)
            v_sq = value_pred.squeeze(-1)
            if t == 0:
                self._last_initial_values = v_sq # Save for Actor-Critic Advantage Baseline
            
            h, c = h_n, c_n
            # [Fix] Truncated BPTT: 50스텝마다 hidden state detach → VRAM 일정 유지
            if t > 0 and t % 50 == 0:
                h = h.detach()
                c = c.detach()
            
            # Masking & Sampling
            mask = self.env.get_mask() # [Batch, Num_Nodes]
            scores = scores.view(self.num_pomo, -1)
            candidate_nodes = torch.arange(scores.size(1), device=self.device).unsqueeze(0).expand(batch_size, -1)
            recent_window = self.reward_cfg['RECENT_WINDOW']
            recent_nodes = None
            if path_history:
                recent_nodes = torch.stack(path_history[-recent_window:], dim=1)
            revisit_penalty_logits = self._build_revisit_penalty(candidate_nodes, visit_counts, recent_nodes)
            if goal_finish_window.any():
                revisit_penalty_logits = torch.where(
                    goal_finish_window.unsqueeze(1),
                    revisit_penalty_logits * GOAL_FINISH_PENALTY_SCALE,
                    revisit_penalty_logits,
                )
            scores = scores - revisit_penalty_logits
            scores = scores.masked_fill(~mask.bool(), -float('inf'))

            if collect_worker_aux:
                expert_next = self.env.weighted_next_hop_matrix[self.env.current_node, targets]
                expert_is_valid = (
                    active_mask
                    & (expert_next >= 0)
                    & torch.gather(mask, 1, expert_next.unsqueeze(1)).squeeze(1).bool()
                )
                if expert_is_valid.any():
                    aux_target_idx = targets.clone()
                    if goal_handoff.any():
                        aux_mismatch = (~torch.eq(aux_target_idx, targets)) & goal_handoff
                        if aux_mismatch.any():
                            msg = (
                                "Handoff aux target mismatch detected: "
                                f"aux={aux_target_idx[aux_mismatch].tolist()} active={targets[aux_mismatch].tolist()}"
                            )
                            if self.debug_mode:
                                raise AssertionError(msg)
                            if not self._handoff_target_warned:
                                print(f"⚠️ {msg}")
                                self._handoff_target_warned = True
                    handoff_aux_mask = in_post_last_subgoal_phase & (steps_after_last_subgoal < POST_HANDOFF_WINDOW)
                    aux_weights = torch.where(
                        handoff_aux_mask,
                        torch.full((batch_size,), float(handoff_aux_boost), device=device),
                        torch.ones(batch_size, device=device),
                    )
                    aux_loss_vec = F.cross_entropy(
                        scores[expert_is_valid],
                        expert_next[expert_is_valid],
                        reduction='none',
                    )
                    worker_aux_ce_sum = worker_aux_ce_sum + (aux_loss_vec * aux_weights[expert_is_valid]).sum()
                    worker_aux_ce_count = worker_aux_ce_count + aux_weights[expert_is_valid].sum()
            
            probs = F.softmax(scores, dim=1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            # Log-Prob & Entropy 저장 (gradient 유지 - Policy Gradient용)
            log_p = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # [Fix] active_before_step 사용: 이 스텝에서 실제 행동한 Worker만 log_prob/entropy 누적
            log_probs_sum = torch.where(active_before_step, log_probs_sum + log_p, log_probs_sum)
            entropy_sum = torch.where(active_before_step, entropy_sum + entropy, entropy_sum)
            if active_before_step.any():
                traj_values.append(v_sq)
                traj_value_masks.append(active_before_step.float())
            
            # Env Step (비활성 Worker는 현재 위치에 고정)
            actions = torch.where(active_before_step, actions, self.env.current_node)
            self.env.step(actions)
            path_history.append(self.env.current_node.clone())

            arrived_goal_after = self.env.current_node == goal_idx
            goal_hops_after = self.env.hop_matrix[self.env.current_node, goal_idx].float()
            if pending_goal_hops_after_handoff.any():
                goal_hops_1step_after_handoff = torch.where(
                    pending_goal_hops_after_handoff,
                    goal_hops_after,
                    goal_hops_1step_after_handoff,
                )
                pending_goal_hops_after_handoff = torch.zeros_like(pending_goal_hops_after_handoff)
            active_subgoal_hops_after = self.env.hop_matrix[self.env.current_node, targets].float()
            # [Refactor: Task 4] Goal Regression 감지 (best_goal_hops 업데이트 전 계산 필수)
            goal_regressed = active_before_step & (goal_hops_after > best_goal_hops + 0.5)
            goal_progress_mask = goal_hops_after < (best_goal_hops - 1e-6)
            subgoal_progress_mask = valid_active_subgoal & (active_subgoal_hops_after < (best_subgoal_hops - 1e-6))
            progress_mask = active_before_step & (
                goal_progress_mask
                | subgoal_progress_mask
                | valid_arrived_subgoal
                | soft_arrived_subgoal
                | skip_mask
                | ptr_changed
                | arrived_goal_after
            )
            best_goal_hops = torch.where(
                active_before_step,
                torch.minimum(best_goal_hops, goal_hops_after),
                best_goal_hops,
            )
            best_subgoal_hops = torch.where(
                valid_active_subgoal & active_before_step,
                torch.minimum(best_subgoal_hops, active_subgoal_hops_after),
                best_subgoal_hops,
            )
            ptr_dwell_steps = ptr_dwell_steps + active_before_step.long()
            steps_after_last_subgoal = steps_after_last_subgoal + (active_before_step & in_post_last_subgoal_phase).long()
            stagnation_steps = torch.where(
                progress_mask,
                torch.zeros_like(stagnation_steps),
                stagnation_steps + active_before_step.long(),
            )
            post_handoff_patience_after = torch.clamp(
                8 + 2 * goal_hops_after.long(),
                min=12,
                max=24,
            )
            effective_patience_after = torch.where(
                in_post_last_subgoal_phase,
                post_handoff_patience_after,
                effective_patience,
            )
            new_stagnations = active_before_step & (~progress_mask) & (stagnation_steps >= effective_patience_after)
            stagnated_early_mask = stagnated_early_mask | new_stagnations
            post_last_sg_stagnation_mask = post_last_sg_stagnation_mask | (new_stagnations & in_post_last_subgoal_phase)
            
            # Step 카운트 증가
            step_counts = step_counts + active_before_step.long()
            steps_since_last_subgoal = steps_since_last_subgoal + active_before_step.long()
            
            # F: 시간 압박 페널티 (스텝이 길어질수록 점점 강해짐)
            time_ratio = t / max(MAX_TOTAL_STEPS, 1)
            step_p = BASE_STEP_PENALTY * (1.0 + TIME_PRESSURE_SCALE * time_ratio)
            step_penalty_sum = torch.where(
                active_before_step, step_penalty_sum + step_p, step_penalty_sum
            )
            
            # P2: 루프 페널티 (방문횟수에 비례하여 점진 증가)
            curr_nodes_flat = self.env.current_node
            curr_visit_count = torch.gather(visit_counts, 1, curr_nodes_flat.unsqueeze(1)).squeeze(1)
            revisit_penalty = LOOP_PENALTY_SCALE * torch.clamp(curr_visit_count - 1.0, min=0.0)
            revisit_penalty = torch.where(
                goal_finish_window,
                revisit_penalty * GOAL_FINISH_PENALTY_SCALE,
                revisit_penalty,
            )
            loop_penalty_sum = torch.where(
                active_before_step, loop_penalty_sum - revisit_penalty, loop_penalty_sum
            )
            
            # [Refactor: Task 4] Goal Regression 페널티 (Handoff 후 3.0x 스케일)
            # Why: Post-Handoff 구간에서 코앞의 Goal을 놓치면 치명적 → 강력한 페널티
            regression_scale = torch.where(
                in_post_last_subgoal_phase,
                torch.full_like(goal_hops_after, 3.0),
                torch.full_like(goal_hops_after, self.goal_regression_penalty_large),
            )
            goal_regression_penalty = goal_regressed.float() * regression_scale
            loop_penalty_sum = torch.where(
                active_before_step, loop_penalty_sum - goal_regression_penalty, loop_penalty_sum
            )
            
            # E: 탐색 보너스 (처음 방문하는 노드에 소액 보상)
            is_first_visit = (curr_visit_count <= 1.0) & active_before_step
            exploration_sum = torch.where(
                is_first_visit, exploration_sum + EXPLORATION_BONUS, exploration_sum
            )
            
            # A: Stage-aware PBRS (active target 중심 + optional goal tether)
            phi_target_new = -self.env.apsp_matrix[curr_nodes_flat, targets] / max_dist
            phi_goal_new = -self.env.apsp_matrix[curr_nodes_flat, goal_idx] / max_dist
            active_target_weight = torch.where(
                valid_active_subgoal,
                torch.ones_like(phi_target_new),
                torch.zeros_like(phi_target_new),
            )
            if self.stage == "worker":
                lambda_sg = active_target_weight
                lambda_goal = 1.0 - active_target_weight
            else:
                lambda_sg = active_target_weight
                lambda_goal = torch.where(
                    valid_active_subgoal,
                    torch.full_like(phi_goal_new, 0.2),
                    torch.ones_like(phi_goal_new),
                )
            goal_finish_multiplier = torch.where(
                goal_finish_window,
                torch.full_like(phi_goal_new, GOAL_FINISH_PBRS_MULT),
                torch.ones_like(phi_goal_new),
            )
            pbrs_step = (
                lambda_sg * (GAMMA_PBRS * phi_target_new - phi_target_old)
                + (lambda_goal * goal_finish_multiplier) * (GAMMA_PBRS * phi_goal_new - phi_goal_old)
            )
            pbrs_sum = torch.where(active_before_step, pbrs_sum + pbrs_step, pbrs_sum)
            phi_target_old = phi_target_new
            phi_goal_old = phi_goal_new
            
            # D: 마일스톤 보상 (25/50/75% 지점 통과 시 일회성 보너스)
            progress = 1.0 - self.env.apsp_matrix[curr_nodes_flat, goal_idx] / max(optimal_dist, 1.0)
            
            hit_25 = (progress >= 0.25) & (~milestone_25_reached) & active_before_step
            milestone_sum = torch.where(hit_25, milestone_sum + MILESTONE_25, milestone_sum)
            milestone_25_reached = milestone_25_reached | hit_25
            
            hit_50 = (progress >= 0.50) & (~milestone_50_reached) & active_before_step
            milestone_sum = torch.where(hit_50, milestone_sum + MILESTONE_50, milestone_sum)
            milestone_50_reached = milestone_50_reached | hit_50
            
            hit_75 = (progress >= 0.75) & (~milestone_75_reached) & active_before_step
            milestone_sum = torch.where(hit_75, milestone_sum + MILESTONE_75, milestone_sum)
            milestone_75_reached = milestone_75_reached | hit_75

            active_mask = active_mask & (~new_stagnations)
            
            # Critic value trajectory는 최종 Monte Carlo loss까지 gradient를 유지한다.
        
        trailing_dwell_mask = ptr_dwell_steps > 0
        ptr_dwell_total = torch.where(
            trailing_dwell_mask,
            ptr_dwell_total + ptr_dwell_steps.float(),
            ptr_dwell_total,
        )
        ptr_dwell_count = torch.where(
            trailing_dwell_mask,
            ptr_dwell_count + 1.0,
            ptr_dwell_count,
        )
        post_last_sg_success_mask = goal_after_last_subgoal_mask & (self.env.current_node == goal_idx)

        # === [v6] 최종 보상 계산 (PBRS + 7요소) ===
        is_success = (self.env.current_node == goal_idx)
        
        # A: PBRS (포텐셜 기반 거리 가이드)
        final_rewards = pbrs_sum * POTENTIAL_SCALE
        
        # B: 서브골 보상 (진행률 + 최적성 보너스 포함)
        final_rewards += subgoal_rewards
        
        # C: 목표 도달 보상
        final_rewards += is_success.float() * GOAL_REWARD
        
        # D: 마일스톤 보상
        final_rewards += milestone_sum
        
        # E: 탐색 보너스
        final_rewards += exploration_sum
        
        # R4: 효율성 보너스 (성공 시에만)
        optimal_steps = max(optimal_dist / (self.env.max_dist / self.env.num_nodes), 1.0)
        ratio = step_counts.float() / optimal_steps
        efficiency_bonus = torch.clamp(EFFICIENCY_MAX * (2.0 - ratio), min=0.0, max=EFFICIENCY_MAX)
        final_rewards += torch.where(is_success, efficiency_bonus, torch.zeros_like(efficiency_bonus))
        
        # F: 시간 압박 페널티 (누적)
        final_rewards += step_penalty_sum
        
        # P2: 루프 페널티 (누적)
        final_rewards += loop_penalty_sum
        
        # P3: 실패 페널티
        fail_penalty = (~is_success).float() * FAIL_PENALTY
        final_rewards += fail_penalty
        base_rewards = final_rewards.clone()
        if plan_adjustment is not None:
            final_rewards = final_rewards + plan_adjustment.to(self.device)

        # --- Critic Value Loss (Monte Carlo Targets) ---
        target = final_rewards.detach()
        norm_target, ret_mean, ret_std = self._normalize_returns(target)
        
        valid_value_weight = torch.zeros(batch_size, device=device)
        for t, val_pred in enumerate(traj_values):
            step_mask = traj_value_masks[t] if t < len(traj_value_masks) else torch.ones(batch_size, device=device)
            loss_t = F.smooth_l1_loss(
                val_pred,
                norm_target,
                reduction='none',
                beta=self.critic_cfg['huber_beta'],
            )
            val_loss_sum = val_loss_sum + loss_t * step_mask
            valid_value_weight = valid_value_weight + step_mask
        
        if traj_values:
            val_loss_sum = val_loss_sum / valid_value_weight.clamp(min=1.0)
        
        # [Diagnostic] 디버그 모드에서 보상 분해 및 진단 정보 저장
        if self._collect_debug_this_episode:
            final_goal_dist = self.env.apsp_matrix[self.env.current_node, goal_idx]
            progress_ratio = 1.0 - (final_goal_dist / max(optimal_dist, 1.0))
            traj_value_tensor = None
            if traj_values:
                traj_value_tensor = torch.stack([v.detach() for v in traj_values], dim=1)
                repeated_target = norm_target.unsqueeze(1).expand_as(traj_value_tensor)
                abs_td_error = (repeated_target - traj_value_tensor).abs()
                target_var = repeated_target.var(unbiased=False)
                if target_var.item() > 1e-8:
                    explained_variance = 1.0 - ((repeated_target - traj_value_tensor).var(unbiased=False) / target_var)
                else:
                    explained_variance = torch.tensor(0.0, device=self.device)
                value_mean = traj_value_tensor.mean().item()
                value_std = traj_value_tensor.std(unbiased=False).item()
                mean_abs_td_error = abs_td_error.mean().item()
            else:
                explained_variance = torch.tensor(0.0, device=self.device)
                value_mean = 0.0
                value_std = 0.0
                mean_abs_td_error = 0.0

            self._last_debug_tensors = {
                'start_idx': int(start_idx),
                'goal_idx': int(goal_idx),
                'goal_reward_value': float(GOAL_REWARD),
                'r1_pbrs': (pbrs_sum * POTENTIAL_SCALE).detach(),
                'r2_subgoal': subgoal_rewards.detach(),
                'r3_goal': (is_success.float() * GOAL_REWARD).detach(),
                'r4_efficiency': torch.where(is_success, efficiency_bonus, torch.zeros_like(efficiency_bonus)).detach(),
                'r5_milestone': milestone_sum.detach(),
                'r6_explore': exploration_sum.detach(),
                'p1_time': step_penalty_sum.detach(),
                'p2_loop': loop_penalty_sum.detach(),
                'p3_fail': fail_penalty.detach(),
                'r7_plan_penalty': plan_diag['plan_penalty'].detach() if plan_diag is not None else torch.zeros_like(final_rewards),
                'base_reward': base_rewards.detach(),
                'final_reward': final_rewards.detach(),
                'success_mask': is_success.detach(),
                'loop_fail_mask': failed_early_mask.detach(),
                'stagnation_fail_mask': stagnated_early_mask.detach(),
                'final_goal_dist': final_goal_dist.detach(),
                'progress_ratio': progress_ratio.detach(),
                'step_counts': step_counts.float().detach(),
                'subgoal_hit_mask': subgoal_hit_mask.detach(),
                'reached_last_subgoal_mask': reached_last_subgoal_mask.detach(),
                'goal_after_last_subgoal_mask': goal_after_last_subgoal_mask.detach(),
                'goal_dist_after_last_subgoal': goal_dist_after_last_subgoal.detach(),
                'steps_after_last_subgoal': steps_after_last_subgoal.float().detach(),
                'ptr_dwell_mean': (ptr_dwell_total / ptr_dwell_count.clamp(min=1.0)).detach(),
                'soft_arrival_mask': soft_arrival_mask_any.detach(),
                'skip_mask': skip_mask_any.detach(),
                'switch_reason_exact_mask': switch_reason_exact_mask_any.detach(),
                'switch_reason_soft_mask': switch_reason_soft_mask_any.detach(),
                'skip_due_to_passed_mask': skip_due_to_passed_mask_any.detach(),
                'skip_due_to_goal_progress_mask': skip_due_to_goal_progress_mask_any.detach(),
                'skip_due_to_unreachable_mask': skip_due_to_unreachable_mask_any.detach(),
                'blocked_target_skip_mask': skip_due_to_unreachable_mask_any.detach(),
                'post_last_sg_success_mask': post_last_sg_success_mask.detach(),
                'post_last_sg_stagnation_mask': post_last_sg_stagnation_mask.detach(),
                'remaining_goal_hops_at_handoff': remaining_goal_hops_at_handoff.detach(),
                'goal_hops_1step_after_handoff': goal_hops_1step_after_handoff.detach(),
                'min_hop_to_active_subgoal': min_hop_to_active_subgoal.detach(),
                'max_subgoal_progress': max_subgoal_progress.detach(),
                'wkr_entropy_mean': (entropy_sum / step_counts.float().clamp(min=1.0)).detach(),
                'critic_v0': self._safe_tensor_mean_item(getattr(self, '_last_initial_values', None)),
                'critic_mse': val_loss_sum.mean().item(),
                'return_mean': target.mean().item(),
                'return_std': target.std(unbiased=False).item(),
                'norm_return_mean': norm_target.mean().item(),
                'norm_return_std': norm_target.std(unbiased=False).item(),
                'ret_ema_mean': ret_mean,
                'ret_ema_std': ret_std,
                'value_mean': value_mean,
                'value_std': value_std,
                'mean_abs_td_error': mean_abs_td_error,
                'explained_variance': float(explained_variance.item()),
            }
        
        worker_aux_ce_loss = worker_aux_ce_sum / worker_aux_ce_count.clamp(min=1.0)
        return (
            log_probs_sum,
            final_rewards,
            norm_target.detach(),
            step_counts,
            val_loss_sum,
            entropy_sum,
            worker_aux_ce_loss,
        )

    def _compute_checkpoint_quality(self, start_idx, goal_idx, sequences, max_reward=4.0):
        """
        방안 C: 체크포인트 품질 보상 (효율성 + 균등분할 결합).
        
        효율성: APSP(S,C1) + APSP(C1,C2) + ... + APSP(Cn,G) ≈ APSP(S,G)이면 최대 보상
        균등분할: 세그먼트 길이의 분산이 작을수록 높은 보상
        
        Returns: [Batch] 크기의 checkpoint quality reward 텐서
        """
        batch_size = sequences.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        optimal_dist = self.env.apsp_matrix[start_idx, goal_idx].item()
        if optimal_dist < 1.0:
            return rewards  # S와 G가 같거나 매우 가까운 경우
        
        alpha = 0.6   # 효율성 가중치
        beta = 0.4    # 균등분할 가중치
        max_reward = float(max_reward)
        
        for b in range(batch_size):
            # 유효 체크포인트 추출 (PAD/EOS 제외)
            seq = sequences[b]
            valid = (seq < self.env.num_nodes) & (seq >= 0)
            checkpoints = seq[valid].tolist()
            
            if len(checkpoints) == 0:
                # 체크포인트 없으면 보상 없음 (감점도 없음)
                continue
            
            # 전체 경유지 시퀀스: [Start, C1, C2, ..., Goal]
            waypoints = [start_idx] + checkpoints + [goal_idx]
            
            # === 효율성 (α) ===
            # 체크포인트를 경유한 총 거리 vs 직행 거리
            total_segment_dist = 0.0
            segment_lengths = []
            for i in range(len(waypoints) - 1):
                u, v = waypoints[i], waypoints[i+1]
                seg_dist = self.env.apsp_matrix[u, v].item()
                total_segment_dist += seg_dist
                segment_lengths.append(seg_dist)
            
            # efficiency: 1.0이면 완벽 (우회 없음), 0에 가까우면 큰 우회
            efficiency = optimal_dist / max(total_segment_dist, 1.0)
            efficiency = min(efficiency, 1.0)  # 1.0 초과 방지
            
            # === 균등분할 (β) ===
            # 세그먼트 길이의 변동계수(CV) 기반: CV가 작을수록 균등
            if len(segment_lengths) > 1:
                mean_len = sum(segment_lengths) / len(segment_lengths)
                variance = sum((s - mean_len)**2 for s in segment_lengths) / len(segment_lengths)
                std_dev = variance ** 0.5
                cv = std_dev / max(mean_len, 1.0)  # 변동계수 (0이면 완벽 균등)
                balance = 1.0 / (1.0 + cv)  # 0~1 범위
            else:
                balance = 0.5  # 세그먼트 1개면 중립 점수
            
            # === 결합 ===
            quality = alpha * efficiency + beta * balance
            rewards[b] = quality * max_reward

        return rewards

    def save_models(self, ep):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.manager.state_dict(), os.path.join(self.save_dir, f"manager_{ep}.pt"))
        torch.save(self.worker.state_dict(), os.path.join(self.save_dir, f"worker_{ep}.pt"))
    
    def _plot_rl_curves(
        self,
        history,
        title='RL Fine-tuning Learning Curves',
        reward_title='Reward per Episode',
        reward_label='Reward (MA50)',
        loss_title='Policy Loss per Episode',
        loss_label='Loss (MA50)',
        path_title='Average Path Length',
        path_label='Path Length (MA50)',
        path_ylabel='Steps',
        success_title='Success Rate (EMA)',
        success_label='Success Rate (EMA)',
        success_target=0.7,
        success_target_label='Target (70%)',
    ):
        """RL 학습 곡선 그래프 생성 및 저장"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        episodes = range(1, len(history['rewards']) + 1)
        
        # 이동 평균 함수
        def smooth(data, window=50):
            if len(data) < window:
                return data
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window)
                smoothed.append(sum(data[start:i+1]) / (i - start + 1))
            return smoothed
        
        # 1. Reward 곡선
        axes[0][0].plot(episodes, history['rewards'], alpha=0.3, color='blue', linewidth=0.5)
        axes[0][0].plot(episodes, smooth(history['rewards']), 'b-', linewidth=2, label=reward_label)
        axes[0][0].set_title(reward_title, fontsize=13, fontweight='bold')
        axes[0][0].set_xlabel('Episode')
        axes[0][0].set_ylabel('Reward')
        axes[0][0].legend()
        axes[0][0].grid(True, alpha=0.3)
        
        # 2. Loss 곡선
        axes[0][1].plot(episodes, history['losses'], alpha=0.3, color='red', linewidth=0.5)
        axes[0][1].plot(episodes, smooth(history['losses']), 'r-', linewidth=2, label=loss_label)
        axes[0][1].set_title(loss_title, fontsize=13, fontweight='bold')
        axes[0][1].set_xlabel('Episode')
        axes[0][1].set_ylabel('Loss')
        axes[0][1].legend()
        axes[0][1].grid(True, alpha=0.3)
        
        # 3. Path Length 곡선
        axes[1][0].plot(episodes, history['path_lengths'], alpha=0.3, color='green', linewidth=0.5)
        axes[1][0].plot(episodes, smooth(history['path_lengths']), 'g-', linewidth=2, label=path_label)
        axes[1][0].set_title(path_title, fontsize=13, fontweight='bold')
        axes[1][0].set_xlabel('Episode')
        axes[1][0].set_ylabel(path_ylabel)
        axes[1][0].legend()
        axes[1][0].grid(True, alpha=0.3)
        
        # 4. Success Rate EMA 곡선 (퍼센트 단위로 표시, Y축 자동 스케일)
        success_pct = [v * 100.0 for v in history['success_rates']]
        axes[1][1].plot(episodes, success_pct, 'purple', linewidth=2, label=success_label)
        if success_target is not None:
            axes[1][1].axhline(y=success_target * 100, color='orange', linestyle='--', alpha=0.7, label=success_target_label)
        axes[1][1].set_title(success_title, fontsize=13, fontweight='bold')
        axes[1][1].set_xlabel('Episode')
        axes[1][1].set_ylabel('Rate (%)')
        axes[1][1].set_ylim(0, 100)
        axes[1][1].legend()
        axes[1][1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        curve_path = os.path.join(self.save_dir, 'rl_learning_curve.png')
        curve_run_path = os.path.join(self.save_dir, f'rl_learning_curve_{self._run_id}.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.savefig(curve_run_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 RL learning curve saved to {curve_path}")
