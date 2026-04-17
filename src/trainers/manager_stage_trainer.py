import json
import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.trainers.pomo_trainer import DOMOTrainer


class ManagerStageTrainer(DOMOTrainer):
    def __init__(self, env, manager, worker, config):
        super().__init__(env, manager, worker, config)
        self.stage = "manager"
        self.reward_cfg.update(
            {
                'PLAN_CORRIDOR_WEIGHT': 0.5,
                'PLAN_COUNT_UNDER_PENALTY': 1.5,
                'PLAN_COUNT_OVER_PENALTY': 0.15,
                'ANCHOR_HOP_PENALTY': 0.40,
                'ANCHOR_NEAR_BONUS': 0.95,
                'FIRST_ANCHOR_PENALTY': 0.80,
                'SPACING_PENALTY_SCALE': 1.20,
                'MONOTONIC_PENALTY_SCALE': 0.80,
                'PLAN_ADJUST_MIN': -8.0,
                'PLAN_ADJUST_MAX': 4.0,
            }
        )
        self.mgr_max_grad_norm = 20.0
        self.manager.to(self.device).train()
        for p in self.manager.parameters():
            p.requires_grad_(True)  # [Fix] Worker Stage에서 Freeze된 상태가 넘어올 수 있으므로 명시적 해제
            
        self.worker.to(self.device).eval()
        for p in self.worker.parameters():
            p.requires_grad_(False)
        self.mgr_opt = optim.Adam(self.manager.parameters(), lr=self.config.lr)
        self.wkr_opt = None
        self.sparse_warmstart_ratio = 0.20  # [Refactor: Task 1] 초반 20%% A* Anchor CE 웜스타트

    def _compute_manager_plan_score(self, plan_diag):
        plan_lengths = plan_diag['plan_lengths']
        under_count = torch.clamp(plan_diag['plan_len_min'] - plan_lengths, min=0.0)
        over_count = torch.clamp(plan_lengths - plan_diag['plan_len_max'], min=0.0)
        r_count = -1.20 * under_count - 0.15 * over_count

        r_anchor = -0.40 * plan_diag['anchor_hop_error_mean'] + 0.95 * plan_diag['anchor_near_ratio']
        r_spacing = -1.20 * plan_diag['spacing_error_mean']
        r_mono = -0.80 * plan_diag['monotonic_violation_mean']
        r_budget = -1.10 * plan_diag['segment_budget_error_mean']
        r_front = -1.50 * plan_diag['first_segment_overshoot']

        expected_first_hops = plan_diag['shortest_hops'] / (plan_diag['plan_len_ref'] + 1.0)
        first_hop_excess = torch.clamp(
            plan_diag['first_subgoal_hops'] - expected_first_hops - 1.0,
            min=0.0,
            max=4.0,
        )
        r_first = -0.80 * torch.clamp(plan_diag['first_segment_budget_err'] - 1.0, min=0.0) - 0.40 * first_hop_excess
        corridor_deficit = torch.clamp(0.80 - plan_diag['corridor_ratio'], min=0.0)
        r_corr = -0.20 * corridor_deficit
        r_empty = torch.where(
            plan_lengths == 0,
            torch.full_like(plan_lengths, -2.0),
            torch.zeros_like(plan_lengths),
        )
        plan_score = r_count + r_anchor + r_spacing + r_mono + r_budget + r_front + r_first + r_corr + r_empty
        return {
            'plan_score': plan_score,
            'r_count': r_count,
            'r_anchor': r_anchor,
            'r_spacing': r_spacing,
            'r_mono': r_mono,
            'r_budget': r_budget,
            'r_front': r_front,
            'r_first': r_first,
            'r_corr': r_corr,
            'r_empty': r_empty,
        }

    def _append_manager_sample(self, ep, sequences, plan_diag, score_bundle, summary):
        valid_mask = (sequences >= 0) & (sequences < self.env.num_nodes)
        best_idx = int(score_bundle['plan_score'].argmax().item())
        payload = {
            'ep': int(ep),
            'stage': self.stage,
            'start_idx': int(self.env.current_node[0].item()),
            'goal_idx': int(self.env.target_node[0].item()),
            'plan': sequences[best_idx][valid_mask[best_idx]].detach().cpu().tolist(),
            'reference_anchors': plan_diag['reference_anchor_nodes'].detach().cpu().tolist(),
            'plan_length': int(plan_diag['plan_lengths'][best_idx].item()),
            'plan_len_ref': float(plan_diag['plan_len_ref'][best_idx].item()),
            'plan_under': bool(plan_diag['plan_lengths'][best_idx].item() < plan_diag['plan_len_min'][best_idx].item()),
            'anchor_hop_err': float(plan_diag['anchor_hop_error_mean'][best_idx].item()),
            'anchor_near': float(plan_diag['anchor_near_ratio'][best_idx].item()),
            'spacing_error': float(plan_diag['spacing_error_mean'][best_idx].item()),
            'monotonic_violation_rate': float(plan_diag['monotonic_violation_rate'][best_idx].item()),
            'first_subgoal_hops': float(plan_diag['first_subgoal_hops'][best_idx].item()),
            'segment_budget_error': float(plan_diag['segment_budget_error_mean'][best_idx].item()),
            'first_segment_budget_err': float(plan_diag['first_segment_budget_err'][best_idx].item()),
            'first_segment_overshoot': float(plan_diag['first_segment_overshoot'][best_idx].item()),
            'frontloaded_overshoot_rate': float(plan_diag['frontloaded_overshoot_rate'][best_idx].item()),
            'plan_score': float(score_bundle['plan_score'][best_idx].item()),
            'summary': summary,
        }
        with open(self._debug_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def train(self, episodes):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.debug_mode:
            self._init_debug_outputs()
            with open(self._debug_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Manager Stage Debug Log (LR={self.config.lr}, POMO={self.num_pomo}) ===\n\n")

        self.mgr_scheduler = self._build_hold_then_cosine_scheduler(
            self.mgr_opt,
            episodes,
            hold_ratio=0.7,
            min_factor=0.3,
        )

        history = {'rewards': [], 'losses': [], 'path_lengths': [], 'success_rates': []}
        quality_ema = 0.0
        best_metric = float('-inf')
        pbar = tqdm(range(episodes), desc="MgrStage", ncols=100)
        warmstart_episodes = max(1, int(round(episodes * self.sparse_warmstart_ratio)))

        for ep in pbar:
            self._collect_debug_this_episode = self._should_collect_debug(ep, episodes)
            curriculum_ratio = min(1.0, ep / max(episodes * 0.8, 1.0))
            self.env.set_curriculum_ratio(curriculum_ratio)
            self.env.reset(batch_size=self.num_pomo, sync_problem=True)

            s0 = self.env.current_node[0].item()
            g0 = self.env.target_node[0].item()
            x_mgr_in = self.env.pyg_data.x[:, :4]
            edge_index = self.env.pyg_data.edge_index
            batch_vec = self.env.pyg_data.batch
            ea = self.env.pyg_data.edge_attr[:, 0:1]  # Phase 1: length만 사용

            temperature = max(0.5, 1.5 - curriculum_ratio)
            sequences, _ = self.manager.generate(
                x_mgr_in,
                edge_index,
                batch_vec,
                max_len=20,
                temperature=temperature,
                apsp_matrix=self.env.hop_matrix,
                node_positions=self.env.pos_tensor,
                edge_attr=ea,
            )
            plan_diag = self._compute_plan_reward_adjustment(s0, g0, sequences)
            score_bundle = self._compute_manager_plan_score(plan_diag)
            plan_score = score_bundle['plan_score']
            normalized_plan_score = (plan_score - plan_score.mean()) / (plan_score.std(unbiased=False) + 1e-8)

            max_len = sequences.size(1)
            target_seq = sequences.to(self.device)
            node_emb_all = self.manager.topology_enc(x_mgr_in, edge_index, edge_attr=ea)
            from torch_geometric.utils import to_dense_batch
            node_emb_dense, _ = to_dense_batch(node_emb_all, batch_vec)
            _, num_nodes, hidden_dim = node_emb_dense.shape
            eos_node_emb = self.manager.eos_token_emb.expand(self.num_pomo, 1, hidden_dim)
            full_ref_embs = torch.cat([node_emb_dense, eos_node_emb], dim=1)

            target_seq_emb = self._gather_manager_teacher_embeddings(full_ref_embs, target_seq, num_nodes)
            mgr_logits = self.manager(x_mgr_in, edge_index, batch_vec, target_seq_emb, edge_attr=ea)
            mgr_logits = mgr_logits[:, :-1, :].contiguous()
            vocab_size = mgr_logits.size(-1)
            mgr_nll = F.cross_entropy(
                mgr_logits.view(-1, vocab_size),
                target_seq.view(-1),
                reduction='none',
                ignore_index=self.manager.PAD_TOKEN,
            ).view(self.num_pomo, max_len)
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
                aux_target_seq[:, ref_token_count] = num_nodes

            aux_target_seq_emb = self._gather_manager_teacher_embeddings(full_ref_embs, aux_target_seq, num_nodes)
            aux_logits = self.manager(x_mgr_in, edge_index, batch_vec, aux_target_seq_emb, edge_attr=ea)
            aux_logits = aux_logits[:, :-1, :].contiguous()
            aux_vocab_size = aux_logits.size(-1)
            aux_nll = F.cross_entropy(
                aux_logits.view(-1, aux_vocab_size),
                aux_target_seq.view(-1),
                reduction='none',
                ignore_index=self.manager.PAD_TOKEN,
            ).view(self.num_pomo, aux_seq_len)
            aux_valid_mask = (aux_target_seq != self.manager.PAD_TOKEN).float()
            aux_valid_counts = aux_valid_mask.sum(dim=1).clamp(min=1.0)
            aux_ce_loss = ((aux_nll * aux_valid_mask).sum(dim=1) / aux_valid_counts).mean()

            mgr_probs = F.softmax(mgr_logits, dim=-1)
            mgr_entropy = -(mgr_probs * torch.log(mgr_probs + 1e-8)).sum(dim=-1)
            progress = float(ep) / float(max(episodes - 1, 1))
            warmstart_active = ep < warmstart_episodes
            if progress < self.sparse_warmstart_ratio:
                rl_weight = 0.0
                aux_weight = 1.0
            elif progress < 0.55:
                rl_weight = 0.30
                aux_weight = 0.70
            elif progress < 0.80:
                rl_weight = 0.50
                aux_weight = 0.50
            else:
                rl_weight = 0.70
                aux_weight = 0.30
            policy_loss = -(normalized_plan_score.detach() * mgr_log_probs).mean()
            entropy_bonus = -0.02 * (mgr_entropy * valid_mask).mean()
            loss = rl_weight * policy_loss + aux_weight * aux_ce_loss + entropy_bonus

            self.mgr_opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
                # [Fix] 확실한 그라디언트 추적을 위해 명시적으로 norm 계산
                mgr_preclip_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=self.mgr_max_grad_norm)
                )
                self.mgr_opt.step()
            else:
                mgr_preclip_norm = -1.0 # 0.0과 구별하기 위해 -1로 기록 (requires_grad=False 감지용)
            self.mgr_scheduler.step()

            quality_rate = (
                (plan_diag['plan_lengths'] >= plan_diag['plan_len_min'])
                & (plan_diag['anchor_near_ratio'] >= 0.5)
            ).float().mean().item()
            quality_ema = quality_ema * 0.95 + quality_rate * 0.05

            plan_score_mean = plan_score.mean().item()
            plan_len_mean = plan_diag['plan_lengths'].mean().item()
            pbar.set_postfix({'Score': f"{plan_score_mean:.2f}", 'Plan': f"{plan_len_mean:.1f}"})

            history['rewards'].append(plan_score_mean)
            history['losses'].append(loss.item())
            history['path_lengths'].append(plan_len_mean)
            history['success_rates'].append(float(plan_diag['anchor_near_ratio'].mean().item()))

            hard_mask = plan_diag['plan_len_ref'] >= 4.0
            if hard_mask.any():
                hard_anchor_near = float(plan_diag['anchor_near_ratio'][hard_mask].mean().item())
                hard_under_rate = float(
                    (plan_diag['plan_lengths'][hard_mask] < plan_diag['plan_len_min'][hard_mask]).float().mean().item()
                )
                hard_first_segment_overshoot = float(plan_diag['first_segment_overshoot'][hard_mask].mean().item())
                hard_segment_budget_error = float(plan_diag['segment_budget_error_mean'][hard_mask].mean().item())
                metric = (
                    hard_anchor_near
                    - hard_under_rate
                    - hard_first_segment_overshoot
                    - 0.5 * hard_segment_budget_error
                )
            else:
                hard_anchor_near = 0.0
                hard_under_rate = 0.0
                hard_first_segment_overshoot = 0.0
                hard_segment_budget_error = 0.0
                metric = plan_score_mean
            if metric >= best_metric:
                best_metric = metric
                self._save_unified_checkpoint(
                    "best.pt",
                    ep,
                    metric=metric,
                    metric_name="hard_bin_plan_score",
                    extra_payload={
                        'hard_anchor_near': hard_anchor_near,
                        'hard_under_rate': hard_under_rate,
                        'hard_first_segment_overshoot': hard_first_segment_overshoot,
                        'hard_segment_budget_error': hard_segment_budget_error,
                    },
                )

            if self._collect_debug_this_episode:
                mgr_postclip_norm = self._compute_grad_norm(self.manager.parameters())
                row = {
                    'ep': int(ep),
                    'mgr_lr': float(self.mgr_scheduler.get_last_lr()[0]),
                    'loss': float(loss.item()),
                    'plan_score_mean': float(plan_score_mean),
                    'quality_ema': float(quality_ema),
                    'plan_len_mean': float(plan_len_mean),
                    'plan_len_ref_mean': float(plan_diag['plan_len_ref'].mean().item()),
                    'plan_under_rate': float((plan_diag['plan_lengths'] < plan_diag['plan_len_min']).float().mean().item()),
                    'plan_over_rate': float((plan_diag['plan_lengths'] > plan_diag['plan_len_max']).float().mean().item()),
                    'anchor_hop_err_mean': float(plan_diag['anchor_hop_error_mean'].mean().item()),
                    'anchor_near_rate': float(plan_diag['anchor_near_ratio'].mean().item()),
                    'spacing_error_mean': float(plan_diag['spacing_error_mean'].mean().item()),
                    'monotonic_violation_rate': float(plan_diag['monotonic_violation_rate'].mean().item()),
                    'first_subgoal_hops_mean': float(plan_diag['first_subgoal_hops'].mean().item()),
                    'segment_budget_error_mean': float(plan_diag['segment_budget_error_mean'].mean().item()),
                    'first_segment_budget_err_mean': float(plan_diag['first_segment_budget_err'].mean().item()),
                    'first_segment_overshoot_mean': float(plan_diag['first_segment_overshoot'].mean().item()),
                    'frontloaded_overshoot_rate': float(plan_diag['frontloaded_overshoot_rate'].mean().item()),
                    'hard_anchor_near_rate': float(hard_anchor_near),
                    'hard_under_rate': float(hard_under_rate),
                    'hard_first_segment_overshoot': float(hard_first_segment_overshoot),
                    'hard_segment_budget_error': float(hard_segment_budget_error),
                    'corridor_ratio_mean': float(plan_diag['corridor_ratio'].mean().item()),
                    'mgr_grad_norm_preclip': float(mgr_preclip_norm),
                    'mgr_grad_norm_postclip': float(mgr_postclip_norm),
                    'manager_clip_hit_rate': float(mgr_preclip_norm > self.mgr_max_grad_norm),
                    'mgr_entropy': float(mgr_entropy.mean().item()),
                    'mgr_nll': float(mgr_nll.mean().item()),
                    'aux_ce_loss': float(aux_ce_loss.item()),
                    'warmstart_active': float(warmstart_active),
                    'policy_weight': float(rl_weight),
                    'aux_weight': float(aux_weight),
                }
                self._append_debug_csv(row)
                summary = {
                    'plan_score_mean': row['plan_score_mean'],
                    'quality_ema': row['quality_ema'],
                    'plan_len_mean': row['plan_len_mean'],
                    'plan_len_ref_mean': row['plan_len_ref_mean'],
                    'plan_under_rate': row['plan_under_rate'],
                    'anchor_hop_err_mean': row['anchor_hop_err_mean'],
                    'anchor_near_rate': row['anchor_near_rate'],
                    'spacing_error_mean': row['spacing_error_mean'],
                    'monotonic_violation_rate': row['monotonic_violation_rate'],
                    'first_subgoal_hops_mean': row['first_subgoal_hops_mean'],
                    'segment_budget_error_mean': row['segment_budget_error_mean'],
                    'first_segment_budget_err_mean': row['first_segment_budget_err_mean'],
                    'first_segment_overshoot_mean': row['first_segment_overshoot_mean'],
                    'frontloaded_overshoot_rate': row['frontloaded_overshoot_rate'],
                    'hard_anchor_near_rate': row['hard_anchor_near_rate'],
                    'hard_under_rate': row['hard_under_rate'],
                    'hard_first_segment_overshoot': row['hard_first_segment_overshoot'],
                    'hard_segment_budget_error': row['hard_segment_budget_error'],
                }
                self._append_manager_sample(ep, sequences, plan_diag, score_bundle, summary)
                lines = [
                    f"[Ep {ep}] Mgr LR: {row['mgr_lr']:.6f}, QualEMA: {quality_ema*100:.1f}%, Score: {plan_score_mean:.2f}",
                    f"  Plan mean/ref: {row['plan_len_mean']:.2f} / {row['plan_len_ref_mean']:.2f}",
                    f"  Under/Over: {row['plan_under_rate']*100:.1f}% / {row['plan_over_rate']*100:.1f}%",
                    f"  Anchor Err/Near: {row['anchor_hop_err_mean']:.2f} / {row['anchor_near_rate']*100:.1f}%",
                    f"  Spacing/Mono: {row['spacing_error_mean']:.3f} / {row['monotonic_violation_rate']*100:.1f}%",
                    f"  SegBudget/FstErr/Ov: {row['segment_budget_error_mean']:.2f} / {row['first_segment_budget_err_mean']:.2f} / {row['first_segment_overshoot_mean']:.2f}",
                    f"  Frontload Overshoot: {row['frontloaded_overshoot_rate']*100:.1f}%",
                    f"  HardBin Near/Under/Fst/Budget: {row['hard_anchor_near_rate']*100:.1f}% / {row['hard_under_rate']*100:.1f}% / {row['hard_first_segment_overshoot']:.2f} / {row['hard_segment_budget_error']:.2f}",
                    f"  First SG Hops: {row['first_subgoal_hops_mean']:.2f}",
                    f"  Corridor: {row['corridor_ratio_mean']*100:.1f}%",
                    f"  Grad pre/post: {row['mgr_grad_norm_preclip']:.4f} / {row['mgr_grad_norm_postclip']:.4f} (clip {row['manager_clip_hit_rate']*100:.1f}%)",
                    f"  NLL/Entropy/Aux: {row['mgr_nll']:.3f} / {row['mgr_entropy']:.3f} / {row['aux_ce_loss']:.3f}",
                    f"  Warmstart/Weights: {bool(warmstart_active)} / RL {row['policy_weight']:.2f} / Aux {row['aux_weight']:.2f}",
                ]
                for line in lines:
                    pbar.write(line)
                with open(self._debug_log_path, 'a', encoding='utf-8') as f:
                    for line in lines:
                        f.write(line + "\n")
                    f.write("\n")

        self._plot_rl_curves(
            history,
            title='Manager Stage Learning Curves',
            reward_title='Plan Score per Episode',
            reward_label='Plan Score (MA50)',
            path_title='Average Plan Length',
            path_label='Plan Length (MA50)',
            path_ylabel='Tokens',
            success_title='Anchor Near Rate',
            success_label='Anchor Near Rate',
            success_target=None,
        )
        if history['losses']:
            self._save_unified_checkpoint("final.pt", episodes - 1, metric=best_metric, metric_name="hard_bin_plan_score")
