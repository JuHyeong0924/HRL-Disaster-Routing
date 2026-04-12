import json
import json
import os
import random

import torch
import torch.optim as optim
from tqdm import tqdm

from src.trainers.pomo_trainer import DOMOTrainer


class WorkerStageTrainer(DOMOTrainer):
    def __init__(self, env, manager, worker, config):
        super().__init__(env, manager, worker, config)
        self.stage = "worker"
        self.worker.to(self.device).train()
        self.manager.to(self.device).eval()
        for p in self.manager.parameters():
            p.requires_grad_(False)
        self.wkr_opt = optim.Adam(self.worker.parameters(), lr=self.config.lr)
        self.mgr_opt = None
        self.wkr_aux_start = 0.50
        self.wkr_aux_end = 0.10

    def _append_worker_sample(self, ep, start_idx, goal_idx, sequences, row, summary):
        valid_mask = (sequences >= 0) & (sequences < self.env.num_nodes)
        payload = {
            'ep': int(ep),
            'stage': self.stage,
            'start_idx': int(start_idx),
            'goal_idx': int(goal_idx),
            'reference_plan': sequences[0][valid_mask[0]].detach().cpu().tolist(),
            'summary': summary,
            'step_metrics': row,
        }
        with open(self._debug_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _build_sequence_from_nodes(self, nodes, batch_size, max_len=20):
        seq = torch.full(
            (batch_size, max_len),
            self.manager.PAD_TOKEN,
            dtype=torch.long,
            device=self.device,
        )
        if nodes:
            node_tensor = torch.tensor(nodes[:max_len], dtype=torch.long, device=self.device)
            seq[:, :node_tensor.numel()] = node_tensor.unsqueeze(0).expand(batch_size, -1)
        return seq

    def _perturb_reference_sequence(self, ref_meta, batch_size, max_len=20):
        optimal_path = list(ref_meta['optimal_path'])
        anchor_positions = list(ref_meta['reference_anchor_positions'])
        if not anchor_positions:
            return self._build_reference_sequence(
                optimal_path[0],
                optimal_path[-1],
                batch_size,
                max_len=max_len,
            )[0]

        max_path_index = max(len(optimal_path) - 1, 0)
        min_anchor_pos = 0 if max_path_index <= 0 else 1
        max_anchor_pos = 0 if max_path_index <= 0 else max(1, max_path_index - 1)

        mode = random.choice(["drop", "shift", "overshoot"])
        new_positions = anchor_positions[:]
        if mode == "drop" and len(new_positions) > 1:
            drop_idx = random.randrange(len(new_positions))
            del new_positions[drop_idx]
        elif mode == "shift" and new_positions:
            shift_idx = random.randrange(len(new_positions))
            shift_delta = random.choice([-1, 1])
            new_positions[shift_idx] = max(min_anchor_pos, min(max_anchor_pos, new_positions[shift_idx] + shift_delta))
        elif mode == "overshoot" and new_positions:
            overshoot_delta = 1 if max_path_index <= 3 else random.choice([1, 2])
            new_positions[0] = max(min_anchor_pos, min(max_anchor_pos, new_positions[0] + overshoot_delta))

        for idx in range(1, len(new_positions)):
            if new_positions[idx] < new_positions[idx - 1]:
                new_positions[idx] = new_positions[idx - 1]

        new_nodes = [optimal_path[pos] for pos in new_positions]
        return self._build_sequence_from_nodes(new_nodes, batch_size, max_len=max_len)

    def _sample_manager_sequences(self, x_mgr_in, edge_index, batch_vec, edge_attr, max_len=20):
        with torch.no_grad():
            sequences, _ = self.manager.generate(
                x_mgr_in,
                edge_index,
                batch_vec,
                max_len=max_len,
                temperature=1.0,
                apsp_matrix=self.env.hop_matrix,
                node_positions=self.env.pos_tensor,
                edge_attr=edge_attr,
            )
        return sequences

    def _build_worker_training_plan(self, ep, episodes, start_idx, goal_idx, x_mgr_in, edge_index, batch_vec, edge_attr, max_len=20):
        ref_sequences, ref_meta = self._build_reference_sequence(start_idx, goal_idx, self.num_pomo, max_len=max_len)
        progress = ep / max(episodes - 1, 1)
        draw = random.random()

        if progress < 0.70:
            if draw < 0.80:
                return ref_sequences, "clean"
            return self._perturb_reference_sequence(ref_meta, self.num_pomo, max_len=max_len), "perturbed"

        if draw < 0.50:
            return ref_sequences, "clean"
        if draw < 0.80:
            return self._perturb_reference_sequence(ref_meta, self.num_pomo, max_len=max_len), "perturbed"
        return self._sample_manager_sequences(x_mgr_in, edge_index, batch_vec, edge_attr, max_len=max_len), "manager"

    def train(self, episodes):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.debug_mode:
            self._init_debug_outputs()
            with open(self._debug_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Worker Stage Debug Log (LR={self.config.lr}, POMO={self.num_pomo}) ===\n\n")

        self.wkr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.wkr_opt,
            T_max=episodes,
            eta_min=max(self.config.lr * 0.1, 1e-5),  # [Refactor: Task 5] 최소 학습률 1e-5 보장
        )

        history = {'rewards': [], 'losses': [], 'path_lengths': [], 'success_rates': []}
        success_ema = 0.0
        best_metric = float('-inf')
        pbar = tqdm(range(episodes), desc="WkrStage", ncols=100)

        for ep in pbar:
            self._collect_debug_this_episode = self._should_collect_debug(ep, episodes)
            self._last_debug_tensors = None
            curriculum_ratio = min(1.0, ep / max(episodes * 0.8, 1.0))
            self.env.set_curriculum_ratio(curriculum_ratio)
            self.env.reset(batch_size=self.num_pomo, sync_problem=True)

            s0 = self.env.current_node[0].item()
            g0 = self.env.target_node[0].item()
            x_mgr_in = self.env.pyg_data.x[:, :4]
            edge_index = self.env.pyg_data.edge_index
            batch_vec = self.env.pyg_data.batch
            edge_attr = self.env.pyg_data.edge_attr[:, [0, 1, 4, 6, 8]]
            train_sequences, plan_source = self._build_worker_training_plan(
                ep,
                episodes,
                s0,
                g0,
                x_mgr_in,
                edge_index,
                batch_vec,
                edge_attr,
                max_len=20,
            )
            x_pos = self.env.pyg_data.x[:, :2].clone()
            plan_diag = self._compute_plan_reward_adjustment(s0, g0, train_sequences)

            log_probs_sum, rewards, norm_returns, path_lengths, val_loss_sum, entropy_sum, worker_aux_ce_loss = self.execute_batch_plan(
                s0,
                g0,
                train_sequences,
                x_pos,
                plan_adjustment=None,
                plan_diag=plan_diag,
                detach_spatial=False,
                collect_worker_aux=True,
                handoff_aux_boost=5.0,  # [Refactor: Task 4] Handoff 직후 A* 모방 가중치 5배 증폭
                enable_skip=False,
            )

            traj_values = getattr(self, "_last_initial_values", None)
            if traj_values is not None:
                baseline = traj_values.detach()
            else:
                baseline = norm_returns.mean()
            advantages = norm_returns - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            wkr_log_probs = log_probs_sum / path_lengths.float().clamp(min=1.0).sqrt()
            policy_loss = -(advantages.detach() * wkr_log_probs).mean()
            critic_loss = 0.5 * val_loss_sum.mean()
            entropy_bonus = -0.01 * (entropy_sum.mean() / path_lengths.float().clamp(min=1.0).mean())
            worker_aux_weight = self._worker_aux_weight(ep, episodes)
            loss = policy_loss + 0.5 * critic_loss + worker_aux_weight * worker_aux_ce_loss + entropy_bonus

            self.wkr_opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
                wkr_preclip_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=self.wkr_max_grad_norm)
                )
                self.wkr_opt.step()
            else:
                wkr_preclip_norm = 0.0
            self.wkr_scheduler.step()

            is_success = (self.env.current_node == g0)
            ep_success = is_success.float().mean().item()
            success_ema = success_ema * 0.95 + ep_success * 0.05
            reward_mean = rewards.mean().item()
            avg_len = path_lengths.float().mean().item()
            pbar.set_postfix({'Rw': f"{reward_mean:.1f}", 'Len': f"{avg_len:.1f}"})

            history['rewards'].append(reward_mean)
            history['losses'].append(loss.item())
            history['path_lengths'].append(avg_len)

            subgoal_rate = 0.0
            plan_utilization = 0.0
            stagnation_fail_rate = 0.0
            if self._last_debug_tensors is not None:
                dbg = self._last_debug_tensors
                hit_mask = dbg['subgoal_hit_mask']
                valid_plan_mask = (train_sequences >= 0) & (train_sequences < self.env.num_nodes)
                reached = (hit_mask & valid_plan_mask).sum().item()
                generated = valid_plan_mask.sum().item()
                subgoal_rate = self._safe_div(reached, generated)
                plan_utilization = subgoal_rate
                stagnation_fail_rate = float(dbg['stagnation_fail_mask'].float().mean().item())
            history['success_rates'].append(subgoal_rate)

            metric = subgoal_rate + plan_utilization - stagnation_fail_rate + success_ema
            if metric >= best_metric:
                best_metric = metric
                self._save_unified_checkpoint("best.pt", ep, metric=metric, metric_name="worker_stage_score")

            if self._collect_debug_this_episode and self._last_debug_tensors is not None:
                dbg = self._last_debug_tensors
                valid_plan_mask = (train_sequences >= 0) & (train_sequences < self.env.num_nodes)
                reached = (dbg['subgoal_hit_mask'] & valid_plan_mask).sum().item()
                generated = valid_plan_mask.sum().item()
                subgoal_rate = self._safe_div(reached, generated)
                wkr_postclip_norm = self._compute_grad_norm(self.worker.parameters())
                row = {
                    'ep': int(ep),
                    'wkr_lr': float(self.wkr_scheduler.get_last_lr()[0]),
                    'loss': float(loss.item()),
                    'reward_mean': float(reward_mean),
                    'success_ema': float(success_ema),
                    'plan_source': plan_source,
                    'plan_len_ref_mean': float(plan_diag['plan_len_ref'].mean().item()),
                    'subgoal_rate': float(subgoal_rate),
                    'plan_utilization': float(subgoal_rate),
                    'stagnation_fail_rate': float(dbg['stagnation_fail_mask'].float().mean().item()),
                    'loop_fail_rate': float(dbg['loop_fail_mask'].float().mean().item()),
                    'active_subgoal_hops_mean': float(
                        torch.where(
                            torch.isfinite(dbg['min_hop_to_active_subgoal']),
                            dbg['min_hop_to_active_subgoal'],
                            torch.zeros_like(dbg['min_hop_to_active_subgoal']),
                        ).mean().item()
                    ),
                    'max_subgoal_progress_mean': float(dbg['max_subgoal_progress'].mean().item()),
                    'reached_last_subgoal_rate': float(dbg['reached_last_subgoal_mask'].float().mean().item()),
                    'goal_after_last_subgoal_rate': float(dbg['goal_after_last_subgoal_mask'].float().mean().item()),
                    'goal_dist_after_last_sg_mean': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['goal_dist_after_last_subgoal'],
                            torch.zeros_like(dbg['goal_dist_after_last_subgoal']),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'steps_after_last_sg_mean': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['steps_after_last_subgoal'],
                            torch.zeros_like(dbg['steps_after_last_subgoal']),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'ptr_dwell_steps_mean': float(dbg['ptr_dwell_mean'].mean().item()),
                    'switch_reason_exact_rate': float(dbg['switch_reason_exact_mask'].float().mean().item()),
                    'switch_reason_soft_rate': float(dbg['switch_reason_soft_mask'].float().mean().item()),
                    'soft_arrival_rate': float(dbg['soft_arrival_mask'].float().mean().item()),
                    'skip_rate': float(dbg['skip_mask'].float().mean().item()),
                    'skip_due_to_passed_rate': float(dbg['skip_due_to_passed_mask'].float().mean().item()),
                    'skip_due_to_goal_progress_rate': float(dbg['skip_due_to_goal_progress_mask'].float().mean().item()),
                    'skip_due_to_unreachable_rate': float(dbg['skip_due_to_unreachable_mask'].float().mean().item()),
                    'remaining_goal_hops_at_handoff_mean': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['remaining_goal_hops_at_handoff'],
                            torch.zeros_like(dbg['remaining_goal_hops_at_handoff']),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'goal_hops_1step_after_handoff_mean': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['goal_hops_1step_after_handoff'],
                            torch.zeros_like(dbg['goal_hops_1step_after_handoff']),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'post_last_sg_success_rate': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['post_last_sg_success_mask'].float(),
                            torch.zeros_like(dbg['post_last_sg_success_mask'].float()),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'post_last_sg_stagnation_rate': float(
                        torch.where(
                            dbg['goal_after_last_subgoal_mask'],
                            dbg['post_last_sg_stagnation_mask'].float(),
                            torch.zeros_like(dbg['post_last_sg_stagnation_mask'].float()),
                        ).sum().item()
                        / max(float(dbg['goal_after_last_subgoal_mask'].float().sum().item()), 1.0)
                    ),
                    'goal_progress_per100': float(
                        (dbg['progress_ratio'] * 100.0 / dbg['step_counts'].clamp(min=1.0)).mean().item()
                    ),
                    'subgoal_progress_per100': float(
                        (((dbg['subgoal_hit_mask'] & valid_plan_mask).sum(dim=1).float() * 100.0)
                         / dbg['step_counts'].clamp(min=1.0)).mean().item()
                    ),
                    'worker_entropy': float((dbg['wkr_entropy_mean']).mean().item()),
                    'worker_aux_ce_loss': float(worker_aux_ce_loss.item()),
                    'worker_aux_weight': float(worker_aux_weight),
                    'wkr_grad_norm_preclip': float(wkr_preclip_norm),
                    'wkr_grad_norm_postclip': float(wkr_postclip_norm),
                    'worker_clip_hit_rate': float(wkr_preclip_norm > self.wkr_max_grad_norm),
                    'avg_steps': float(dbg['step_counts'].float().mean().item()),
                }
                self._append_debug_csv(row)
                summary = {
                    'plan_source': row['plan_source'],
                    'subgoal_rate': row['subgoal_rate'],
                    'plan_utilization': row['plan_utilization'],
                    'stagnation_fail_rate': row['stagnation_fail_rate'],
                    'active_subgoal_hops_mean': row['active_subgoal_hops_mean'],
                    'max_subgoal_progress_mean': row['max_subgoal_progress_mean'],
                    'reached_last_subgoal_rate': row['reached_last_subgoal_rate'],
                    'goal_after_last_subgoal_rate': row['goal_after_last_subgoal_rate'],
                    'switch_reason_exact_rate': row['switch_reason_exact_rate'],
                    'switch_reason_soft_rate': row['switch_reason_soft_rate'],
                    'skip_due_to_passed_rate': row['skip_due_to_passed_rate'],
                    'skip_due_to_goal_progress_rate': row['skip_due_to_goal_progress_rate'],
                    'skip_due_to_unreachable_rate': row['skip_due_to_unreachable_rate'],
                    'remaining_goal_hops_at_handoff_mean': row['remaining_goal_hops_at_handoff_mean'],
                    'goal_hops_1step_after_handoff_mean': row['goal_hops_1step_after_handoff_mean'],
                    'post_last_sg_success_rate': row['post_last_sg_success_rate'],
                    'post_last_sg_stagnation_rate': row['post_last_sg_stagnation_rate'],
                    'goal_progress_per100': row['goal_progress_per100'],
                    'subgoal_progress_per100': row['subgoal_progress_per100'],
                    'worker_aux_ce_loss': row['worker_aux_ce_loss'],
                    'success_ema': row['success_ema'],
                }
                self._append_worker_sample(ep, s0, g0, train_sequences, row, summary)
                lines = [
                    f"[Ep {ep}] Wkr LR: {row['wkr_lr']:.6f}, SuccessEMA: {success_ema*100:.1f}%, Loss: {loss.item():.2f} ({plan_source})",
                    f"  Subgoal/Util: {row['subgoal_rate']*100:.1f}% / {row['plan_utilization']*100:.1f}%",
                    f"  Stag/Loop: {row['stagnation_fail_rate']*100:.1f}% / {row['loop_fail_rate']*100:.1f}%",
                    f"  Active SG Hop: {row['active_subgoal_hops_mean']:.2f}",
                    f"  Max SG Prog: {row['max_subgoal_progress_mean']*100:.1f}%",
                    f"  LastSG/Handoff: {row['reached_last_subgoal_rate']*100:.1f}% / {row['goal_after_last_subgoal_rate']*100:.1f}%",
                    f"  PostLastSG: succ={row['post_last_sg_success_rate']*100:.1f}%, stag={row['post_last_sg_stagnation_rate']*100:.1f}%",
                    f"  Switch Exact/Soft: {row['switch_reason_exact_rate']*100:.1f}% / {row['switch_reason_soft_rate']*100:.1f}%",
                    f"  Skip(T/P/G/U): {row['skip_rate']*100:.1f}% / {row['skip_due_to_passed_rate']*100:.1f}% / {row['skip_due_to_goal_progress_rate']*100:.1f}% / {row['skip_due_to_unreachable_rate']*100:.1f}%",
                    f"  Handoff GoalHop now/next: {row['remaining_goal_hops_at_handoff_mean']:.2f} / {row['goal_hops_1step_after_handoff_mean']:.2f}",
                    f"  GoalDist/Steps after LastSG: {row['goal_dist_after_last_sg_mean']:.2f} / {row['steps_after_last_sg_mean']:.2f}",
                    f"  Goal/SG per100: {row['goal_progress_per100']:.2f} / {row['subgoal_progress_per100']:.2f}",
                    f"  Aux CE: {row['worker_aux_ce_loss']:.3f} @ w={row['worker_aux_weight']:.3f}",
                    f"  Grad pre/post: {row['wkr_grad_norm_preclip']:.4f} / {row['wkr_grad_norm_postclip']:.4f} (clip {row['worker_clip_hit_rate']*100:.1f}%)",
                ]
                for line in lines:
                    pbar.write(line)
                with open(self._debug_log_path, 'a', encoding='utf-8') as f:
                    for line in lines:
                        f.write(line + "\n")
                    f.write("\n")

        self._plot_rl_curves(
            history,
            title='Worker Stage Learning Curves',
            reward_title='Worker Reward per Episode',
            reward_label='Reward (MA50)',
            path_title='Average Rollout Length',
            path_label='Rollout Length (MA50)',
            path_ylabel='Steps',
            success_title='Subgoal Reach Rate',
            success_label='Subgoal Reach Rate',
            success_target=None,
        )
        if history['losses']:
            self._save_unified_checkpoint("final.pt", episodes - 1, metric=best_metric, metric_name="worker_stage_score")
