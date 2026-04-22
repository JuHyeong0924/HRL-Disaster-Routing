import heapq
import json
import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.trainers.pomo_trainer import DOMOTrainer


class WorkerNavTrainer(DOMOTrainer):
    """Phase 1: 순수 네비게이션 학습. 교사 경로(A*) 기반 hidden checkpoint으로 Worker를 안내."""

    def __init__(self, env, manager, worker, config):
        super().__init__(env, manager, worker, config)
        self.stage = "phase1"

        # Manager is legacy scaffolding only in this APTE branch.
        self.manager.eval()
        for param in self.manager.parameters():
            param.requires_grad_(False)
        self.mgr_opt = None
        self.mgr_scheduler = None

        # [Critical Fix] DOMOTrainer.__init__에서 Joint Stage용으로 Worker를 동결(requires_grad=False)함.
        # Worker Stage에서는 Worker가 학습 대상이므로 반드시 다시 활성화해야 함.
        for param in self.worker.parameters():
            param.requires_grad_(True)
        self.wkr_opt = optim.Adam(self.worker.parameters(), lr=self.config.lr)
        self.wkr_scheduler = None

        self.max_steps = int(getattr(config, "max_steps", 400))
        self.worker_temperature = float(getattr(config, "worker_temperature", 1.0))
        self.target_segment_hops = float(getattr(config, "target_segment_hops", 6.0))
        self.min_hops_for_hidden_checkpoint = int(
            getattr(config, "min_hops_for_hidden_checkpoint", 4)
        )
        self.two_hidden_checkpoint_min_hops = int(
            getattr(config, "two_hidden_checkpoint_min_hops", 10)
        )
        self.loop_limit = int(getattr(config, "loop_limit", 6))
        self.checkpoint_hit_radius = int(getattr(config, "checkpoint_hit_radius", 1))
        self.hidden_bonus_start = float(getattr(config, "hidden_bonus_start", 2.0))
        self.hidden_bonus_mid = float(getattr(config, "hidden_bonus_mid", 1.5))
        self.hidden_bonus_end = float(getattr(config, "hidden_bonus_end", 0.75))
        self.guidance_schedule_ep_1 = int(getattr(config, "guidance_schedule_ep_1", 400))
        self.guidance_schedule_ep_2 = int(getattr(config, "guidance_schedule_ep_2", 800))
        self.wkr_lr_floor = float(getattr(config, "wkr_lr_floor", 1e-6))
        self.total_hidden_bonus_cap = float(
            getattr(
                config,
                "total_hidden_bonus_cap",
                0.25 * float(self.reward_cfg["GOAL_REWARD"]),
            )
        )
        goal_reward = float(self.reward_cfg["GOAL_REWARD"])
        assert (
            self.total_hidden_bonus_cap <= 0.25 * goal_reward + 1e-8
        ), "total_hidden_bonus_cap must stay <= 0.25 * GOAL_REWARD"

        self.wkr_aux_start = float(getattr(config, "wkr_aux_start", 0.20))
        self.wkr_aux_mid = float(getattr(config, "wkr_aux_mid", 0.17))
        self.wkr_aux_end = float(getattr(config, "wkr_aux_end", 0.12))
        self.goal_hop_bonus_8 = float(getattr(config, "goal_hop_bonus_8", 0.75))
        self.goal_hop_bonus_4 = float(getattr(config, "goal_hop_bonus_4", 1.0))
        self.goal_hop_bonus_2 = float(getattr(config, "goal_hop_bonus_2", 1.25))
        self.goal_neighbor_action_bonus = float(
            getattr(config, "goal_neighbor_action_bonus", 1.0)
        )
        self.goal_neighbor_miss_penalty = float(
            getattr(config, "goal_neighbor_miss_penalty", 0.35)
        )
        self.near_goal_ce_mult = float(getattr(config, "near_goal_ce_mult", 1.75))
        self.terminal_entropy_mult = float(
            getattr(config, "terminal_entropy_mult", 0.5)
        )
        self.goal_regression_penalty_small = float(
            getattr(config, "goal_regression_penalty_small", 0.15)
        )
        self.goal_regression_penalty_large = float(
            getattr(config, "goal_regression_penalty_large", 0.35)
        )
        self.near_goal_patience_bonus = int(getattr(config, "near_goal_patience_bonus", 8))

        self.reward_cfg.update(
            {
                "PHASE1_MODE": "apte_guided_worker",
                "TARGET_SEGMENT_HOPS": self.target_segment_hops,
                "MIN_HOPS_FOR_HIDDEN_CHECKPOINT": self.min_hops_for_hidden_checkpoint,
                "TWO_HIDDEN_CHECKPOINT_MIN_HOPS": self.two_hidden_checkpoint_min_hops,
                "HIDDEN_CHECKPOINT_BONUS_START": self.hidden_bonus_start,
                "HIDDEN_CHECKPOINT_BONUS_MID": self.hidden_bonus_mid,
                "HIDDEN_CHECKPOINT_BONUS_END": self.hidden_bonus_end,
                "GUIDANCE_SCHEDULE_EP_1": self.guidance_schedule_ep_1,
                "GUIDANCE_SCHEDULE_EP_2": self.guidance_schedule_ep_2,
                "CHECKPOINT_HIT_RADIUS": self.checkpoint_hit_radius,
                "TOTAL_HIDDEN_BONUS_CAP": self.total_hidden_bonus_cap,
                "WORKER_AUX_START": self.wkr_aux_start,
                "WORKER_AUX_MID": self.wkr_aux_mid,
                "WORKER_AUX_END": self.wkr_aux_end,
                "WORKER_LR_FLOOR": self.wkr_lr_floor,
                "GOAL_HOP_BONUS_8": self.goal_hop_bonus_8,
                "GOAL_HOP_BONUS_4": self.goal_hop_bonus_4,
                "GOAL_HOP_BONUS_2": self.goal_hop_bonus_2,
                "GOAL_NEIGHBOR_ACTION_BONUS": self.goal_neighbor_action_bonus,
                "GOAL_NEIGHBOR_MISS_PENALTY": self.goal_neighbor_miss_penalty,
                "NEAR_GOAL_CE_MULT": self.near_goal_ce_mult,
                "TERMINAL_ENTROPY_MULT": self.terminal_entropy_mult,
                "GOAL_REGRESSION_PENALTY_SMALL": self.goal_regression_penalty_small,
                "GOAL_REGRESSION_PENALTY_LARGE": self.goal_regression_penalty_large,
                "NEAR_GOAL_PATIENCE_BONUS": self.near_goal_patience_bonus,
                "LOOP_LIMIT": self.loop_limit,
                "MAX_ROLLOUT_STEPS": self.max_steps,
                "TEACHER_PATH_POLICY": "reset_only",
                "CHECKPOINT_FORMAT": "worker_centric",
            }
        )

        self._neighbor_lists = [
            torch.where(self.env.adj_matrix[node_idx])[0].tolist()
            for node_idx in range(self.env.num_nodes)
        ]

    def _init_debug_outputs(self):
        super()._init_debug_outputs()
        runtime_path = os.path.join(self.save_dir, "runtime_config.json")
        with open(runtime_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.update(
            {
                "phase1_semantics": {
                    "training": "teacher_guided_hidden_checkpoints",
                    "inference": "worker_only_goal_conditioned",
                    "teacher_replan": "reset_only",
                    "checkpoint_format": "worker_centric",
                }
            }
        )
        with open(runtime_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _select_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
        """엣지 피처 선택 + Min-Max 정규화.
        
        [v3] 정규화 추가: length/capacity/speed의 스케일 불일치로 인한
        GATv2 attention 편향을 방지. 각 피처를 [0, 1] 범위로 정규화.
        """
        if edge_attr is None:
            return None
        if edge_attr.size(1) == 0:
            return None
        # edge_dim=3: [length, capacity, speed] (인덱스 0, 7, 8)
        selected = edge_attr[:, [0, 7, 8]]
        # Min-Max 정규화: 각 피처를 [0, 1] 범위로 스케일링
        feat_min = selected.min(dim=0, keepdim=True)[0]
        feat_max = selected.max(dim=0, keepdim=True)[0]
        scale = (feat_max - feat_min).clamp(min=1e-8)  # 0-division 방지
        return (selected - feat_min) / scale

    def _build_worker_input(self, time_pct: float = 0.0) -> torch.Tensor:
        """환경 x(10채널)에서 Worker 입력(8채널)으로 변환.
        
        [v3 변경]
        - x,y 절대좌표 제거: cross-map 일반화를 위해 맵 종속 피처 배제
        - time_pct 추가: 남은 스텝 비율로 MDP 완전 관측성 보장
        
        Env X: [x(0), y(1), is_curr(2), is_tgt(3), visit(4), dist(5),
                dir_x(6), dir_y(7), is_final(8), hop_dist(9)]
        
        Worker 입력: [is_curr, is_tgt, net_dist, dir_x, dir_y, is_final, hop_dist, time_pct]
                      (8 channels)
        """
        x = self.env.pyg_data.x
        if x.size(1) < 10:
            pad = torch.zeros(
                x.size(0),
                10 - x.size(1),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
            self.env.pyg_data.x = x
        # x,y(0,1) 제거, visit(4) 제거 → [is_curr(2), is_tgt(3), dist(5), dir(6,7), final(8), hop(9)]
        spatial = torch.cat([x[:, 2:4], x[:, 5:]], dim=1)  # [N, 7]
        # time_pct 브로드캐스트: 모든 노드에 동일한 시간 진행률 부여
        t_feat = torch.full(
            (spatial.size(0), 1), time_pct,
            device=x.device, dtype=x.dtype,
        )
        return torch.cat([spatial, t_feat], dim=1)  # [N, 8]

    def _hidden_bonus_weight(self, episode_idx):
        if int(episode_idx) < self.guidance_schedule_ep_1:
            return self.hidden_bonus_start
        if int(episode_idx) < self.guidance_schedule_ep_2:
            return self.hidden_bonus_mid
        return self.hidden_bonus_end

    def _worker_aux_weight(self, episode_idx):
        if int(episode_idx) < self.guidance_schedule_ep_1:
            return self.wkr_aux_start
        if int(episode_idx) < self.guidance_schedule_ep_2:
            return self.wkr_aux_mid
        return self.wkr_aux_end

    @staticmethod
    def _explained_variance(targets, preds):
        if targets.numel() == 0 or preds.numel() == 0:
            return 0.0
        var_y = torch.var(targets, unbiased=False)
        if float(var_y.item()) <= 1e-8:
            return 0.0
        residual = targets - preds
        return float((1.0 - torch.var(residual, unbiased=False) / var_y).item())

    def _compute_dynamic_shortest_path(self, batch_idx, start_idx, goal_idx):
        start = int(start_idx)
        goal = int(goal_idx)
        if start == goal:
            return [start]

        weights = self.env.edge_time_matrix[int(batch_idx)]
        dist = [math.inf] * self.env.num_nodes
        prev = [-1] * self.env.num_nodes
        dist[start] = 0.0
        heap = [(0.0, start)]

        while heap:
            cur_dist, node = heapq.heappop(heap)
            if cur_dist > dist[node]:
                continue
            if node == goal:
                break
            for neighbor in self._neighbor_lists[node]:
                edge_cost = float(weights[node, neighbor].item())
                if not math.isfinite(edge_cost) or edge_cost >= 99.0:
                    continue
                cand = cur_dist + edge_cost
                if cand < dist[neighbor]:
                    dist[neighbor] = cand
                    prev[neighbor] = node
                    heapq.heappush(heap, (cand, neighbor))

        if not math.isfinite(dist[goal]):
            return []

        path = [goal]
        cursor = goal
        while cursor != start:
            cursor = prev[cursor]
            if cursor < 0:
                return []
            path.append(cursor)
        path.reverse()
        return path

    def _compress_hidden_checkpoints(self, path_nodes):
        hop_len = max(len(path_nodes) - 1, 0)
        if hop_len <= 3:
            return []

        if hop_len < self.two_hidden_checkpoint_min_hops:
            target_positions = [0.5]
        else:
            target_positions = [1.0 / 3.0, 2.0 / 3.0]
        checkpoints = []
        seen = set()

        for ratio in target_positions:
            path_pos = int(round(ratio * hop_len))
            path_pos = max(1, min(path_pos, len(path_nodes) - 2))
            node = int(path_nodes[path_pos])
            if node in seen:
                continue
            seen.add(node)
            checkpoints.append(node)

        return checkpoints

    def _build_hidden_checkpoint_batch(self):
        start_nodes = self.env.current_node.detach().cpu().tolist()
        goal_nodes = self.env.target_node.detach().cpu().tolist()

        shared_problem = (
            len(set(start_nodes)) == 1
            and len(set(goal_nodes)) == 1
            and (
                self.env.edge_time_matrix.size(0) == 1
                or torch.allclose(
                    self.env.edge_time_matrix,
                    self.env.edge_time_matrix[0:1].expand_as(self.env.edge_time_matrix),
                )
            )
        )

        hidden_lists = []
        teacher_paths = []
        optimal_hops = []

        if shared_problem:
            path = self._compute_dynamic_shortest_path(0, start_nodes[0], goal_nodes[0])
            if not path:
                path = self.env.reconstruct_hop_shortest_path_indices(int(start_nodes[0]), int(goal_nodes[0]))
            hidden = self._compress_hidden_checkpoints(path)
            hop_val = max(len(path) - 1, 1) if path else max(
                1,
                int(self.env.hop_matrix[int(start_nodes[0]), int(goal_nodes[0])].item()),
            )
            for _ in range(self.num_pomo):
                hidden_lists.append(list(hidden))
                teacher_paths.append(list(path))
                optimal_hops.append(float(hop_val))
            return hidden_lists, teacher_paths, torch.tensor(optimal_hops, device=self.device, dtype=torch.float32)

        for batch_idx, (start_idx, goal_idx) in enumerate(zip(start_nodes, goal_nodes)):
            path = self._compute_dynamic_shortest_path(batch_idx, start_idx, goal_idx)
            if not path:
                path = self.env.reconstruct_hop_shortest_path_indices(int(start_idx), int(goal_idx))
            hidden = self._compress_hidden_checkpoints(path)
            hop_val = max(len(path) - 1, 1) if path else max(
                1,
                int(self.env.hop_matrix[int(start_idx), int(goal_idx)].item()),
            )
            hidden_lists.append(hidden)
            teacher_paths.append(path)
            optimal_hops.append(float(hop_val))

        return hidden_lists, teacher_paths, torch.tensor(optimal_hops, device=self.device, dtype=torch.float32)

    def _save_worker_checkpoint(self, filename, ep, metric=None, metric_name=None, extra_payload=None):
        os.makedirs(self.save_dir, exist_ok=True)
        payload = {
            "epoch": int(ep),
            "stage": self.stage,
            "checkpoint_format": "worker_centric",
            "worker_state": self.worker.state_dict(),
            "manager_state": self.manager.state_dict(),  # Manager도 함께 저장
            "worker_optimizer_state": self.wkr_opt.state_dict() if self.wkr_opt is not None else None,
        }
        if self.wkr_scheduler is not None:
            payload["scheduler_state"] = self.wkr_scheduler.state_dict()
        if metric is not None:
            payload["metric"] = float(metric)
        if metric_name is not None:
            payload["metric_name"] = str(metric_name)
        if extra_payload:
            payload.update(extra_payload)
        torch.save(payload, os.path.join(self.save_dir, filename))

    def _execute_goal_conditioned_rollout(self, ep, episodes):
        batch_size = self.num_pomo
        batch_indices = torch.arange(batch_size, device=self.device)
        goal_targets = self.env.target_node.clone()
        final_phase = torch.ones(batch_size, device=self.device)
        self.env.update_target_features(goal_targets, final_phase)

        hidden_lists, teacher_paths, optimal_hops = self._build_hidden_checkpoint_batch()
        checkpoint_counts = torch.tensor(
            [len(items) for items in hidden_lists],
            device=self.device,
            dtype=torch.float32,
        )
        checkpoint_ptrs = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        checkpoint_hits = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        checkpoint_has_any_hit = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        teacher_hidden_checkpoint_count_actual = checkpoint_counts.clone()
        goal_neighbor_ever = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        goal_neighbor_success_steps = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        goal_neighbor_opportunity_steps = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        # [Perf] checkpoint 노드를 2D 패딩 텐서로 사전 변환 (for-loop 벡터화용)
        _max_ckpt = max((len(h) for h in hidden_lists), default=1)
        _max_ckpt = max(_max_ckpt, 1)  # 최소 1로 보장
        ckpt_tensor = torch.full(
            (batch_size, _max_ckpt), -1, device=self.device, dtype=torch.long
        )
        for _b, _h in enumerate(hidden_lists):
            if _h:
                ckpt_tensor[_b, : len(_h)] = torch.tensor(_h, device=self.device)
        checkpoint_counts_long = checkpoint_counts.long()

        per_hit_bonus = self._hidden_bonus_weight(ep)
        aux_weight = self._worker_aux_weight(ep)
        cap_tensor = torch.full((batch_size,), self.total_hidden_bonus_cap, device=self.device)
        hit_bonus_tensor = torch.where(
            checkpoint_counts > 0,
            torch.minimum(
                torch.full_like(checkpoint_counts, per_hit_bonus),
                cap_tensor / checkpoint_counts.clamp(min=1.0),
            ),
            torch.zeros_like(checkpoint_counts),
        )

        h = torch.zeros(batch_size, self.worker.lstm.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.worker.lstm.hidden_size, device=self.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        loop_fail = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        stagnation_fail = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        timeout_fail = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        step_counts = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        best_goal_dist = self.env.apsp_matrix[self.env.current_node, goal_targets].clone()
        current_goal_hops = self.env.hop_matrix[self.env.current_node, goal_targets].float()
        best_goal_hops = current_goal_hops.clone()
        stagnation_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        teacher_hops_int = optimal_hops.long()
        stagnation_patience_budget = (
            int(self.stagnation_patience)
            + torch.minimum(
                torch.full_like(teacher_hops_int, 8),
                torch.div(teacher_hops_int, 3, rounding_mode="floor"),
            )
        )
        goal_threshold_hit_8 = current_goal_hops <= 8.0
        goal_threshold_hit_4 = current_goal_hops <= 4.0
        goal_threshold_hit_2 = current_goal_hops <= 2.0
        goal_hop_1_hit = current_goal_hops <= 1.0
        goal4_patience_granted = current_goal_hops <= 4.0
        goal2_patience_granted = current_goal_hops <= 2.0
        stagnation_reset_on_best_goal_improve = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        stagnation_reset_goal4 = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        stagnation_reset_goal2 = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        goal_regression_after_best4_hit = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        near_goal_ce_applied_steps = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        ce_eligible_steps = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        start_goal_dist = best_goal_dist.clone()

        log_probs = []
        values = []
        entropies = []
        entropy_weights = []
        rewards = []
        valid_masks = []
        aux_losses = []

        # [Perf] path_traces는 디버그 에피소드에서만 수집 (매 스텝 GPU→CPU 동기화 제거)
        _collect_path = self._collect_debug_this_episode
        if _collect_path:
            path_traces = [[int(node)] for node in self.env.current_node.detach().cpu().tolist()]
        else:
            path_traces = [[] for _ in range(batch_size)]

        gamma = float(self.reward_cfg["GAMMA_PBRS"])
        pbrs_scale = float(self.reward_cfg["POTENTIAL_SCALE"])
        step_penalty = float(self.reward_cfg["BASE_STEP_PENALTY"])
        revisit_penalty_scale = float(self.reward_cfg["LOOP_PENALTY_SCALE"])
        fail_penalty = float(self.reward_cfg["FAIL_PENALTY"])
        goal_reward = float(self.reward_cfg["GOAL_REWARD"])

        # [v3] GATv2 encoder detach 여부: config 기반 (기본: True)
        # Why: GATv2 3레이어 × 400스텝 BPTT는 ~100GB VRAM 필요 → 24GB 4090에서 불가능
        detach_spatial = bool(getattr(self.config, "detach_spatial", True))
        
        # [Perf] edge_attr는 스텝마다 되풀이되지 않으므로 루프 밖에서 1회만 계산
        _cached_edge_attr = self._select_edge_attr(self.env.pyg_data.edge_attr)
        
        # [VRAM Fix] Truncated BPTT: N스텝마다 LSTM 상태를 detach하여 연산 그래프 길이를 제한
        # 400스텝 전체 BPTT는 ~100GB VRAM 필요 → tbptt_len=20이면 ~6GB로 축소
        _tbptt_len = int(getattr(self.config, "tbptt_len", 20))
        _step_counter = 0
        
        for _ in range(self.max_steps):
            active = ~done
            if not bool(active.any()):
                break

            # [VRAM Fix] Truncated BPTT: N스텝마다 h,c를 detach하여 이전 그래프 해제
            _step_counter += 1
            if _step_counter % _tbptt_len == 0:
                h = h.detach()
                c = c.detach()

            current_nodes = self.env.current_node.clone()
            # [v3] time_pct: 현재 스텝 / 최대 스텝 → 에이전트가 마감 시간 인지
            time_pct = float(_step_counter) / float(self.max_steps)
            worker_input = self._build_worker_input(time_pct=time_pct)

            scores, h_next, c_next, value_pred = self.worker.predict_next_hop(
                worker_input,
                self.env.pyg_data.edge_index,
                h,
                c,
                self.env.pyg_data.batch,
                detach_spatial=detach_spatial, 
                edge_attr=_cached_edge_attr,
            )
            # inactive 에이전트의 LSTM 상태는 detach하여 불필요한 연산 그래프 VRAM 누적 방지
            h = torch.where(active.unsqueeze(1), h_next, h.detach())
            c = torch.where(active.unsqueeze(1), c_next, c.detach())

            row_scores = scores.view(batch_size, -1)
            mask = self.env.get_mask().bool()
            masked_scores = row_scores.masked_fill(~mask, -1e9)
            probs = F.softmax(masked_scores, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=1)

            sample_scores = masked_scores
            if self.worker_temperature > 1e-5 and abs(self.worker_temperature - 1.0) > 1e-8:
                sample_scores = masked_scores / self.worker_temperature
                probs = F.softmax(sample_scores, dim=-1)
                entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=1)

            sampled = torch.multinomial(probs, 1).squeeze(1)
            sampled = torch.where(active, sampled, self.env.current_node)
            selected_prob = torch.gather(probs, 1, sampled.unsqueeze(1)).squeeze(1).clamp(min=1e-12)
            log_prob = selected_prob.log()
            log_prob = torch.where(active, log_prob, torch.zeros_like(log_prob))
            entropy = torch.where(active, entropy, torch.zeros_like(entropy))
            goal_neighbor_now = active & mask[batch_indices, goal_targets]
            goal_neighbor_ever |= goal_neighbor_now
            goal_neighbor_opportunity_steps += goal_neighbor_now.float()
            goal_neighbor_chosen = goal_neighbor_now & (sampled == goal_targets)
            goal_neighbor_success_steps += goal_neighbor_chosen.float()

            goal_dist_before = self.env.apsp_matrix[self.env.current_node, goal_targets].float()
            goal_hops_before = self.env.hop_matrix[self.env.current_node, goal_targets].float()
            had_best4_before = best_goal_hops <= 4.0
            self.env.step(sampled)
            self.env.update_target_features(goal_targets, final_phase)
            next_nodes = self.env.current_node.clone()
            goal_dist_after = self.env.apsp_matrix[next_nodes, goal_targets].float()
            goal_hops_after = self.env.hop_matrix[next_nodes, goal_targets].float()

            phi_prev = -goal_dist_before / max(float(self.env.max_dist), 1.0)
            phi_next = -goal_dist_after / max(float(self.env.max_dist), 1.0)
            reward = torch.full((batch_size,), step_penalty, device=self.device)
            reward = reward + pbrs_scale * (gamma * phi_next - phi_prev)

            revisit_count = self.env.visit_count[batch_indices, next_nodes].float() - 1.0
            reward = reward - revisit_penalty_scale * revisit_count.clamp(min=0.0)
            reward = reward + goal_neighbor_chosen.float() * self.goal_neighbor_action_bonus
            reward = reward - (
                (goal_neighbor_now & (~goal_neighbor_chosen)).float()
                * self.goal_neighbor_miss_penalty
            )

            # [Perf] checkpoint hit 판정: bool() GPU 동기화 없이 무조건 실행 (empty mask는 비용 없음)
            if self.checkpoint_hit_radius >= 0:
                ptr_valid = active & (checkpoint_ptrs < checkpoint_counts_long)
                safe_ptrs = checkpoint_ptrs.clamp(max=_max_ckpt - 1)
                current_ckpt = ckpt_tensor[batch_indices, safe_ptrs]
                exact_hit = ptr_valid & (next_nodes == current_ckpt)
                if self.checkpoint_hit_radius > 0:
                    hop_dist = self.env.hop_matrix[next_nodes, current_ckpt]
                    radius_hit = ptr_valid & (~exact_hit) & (
                        hop_dist <= float(self.checkpoint_hit_radius)
                    )
                    hit_mask = exact_hit | radius_hit
                else:
                    hit_mask = exact_hit
                reward = reward + hit_mask.float() * hit_bonus_tensor
                checkpoint_ptrs = checkpoint_ptrs + hit_mask.long()
                checkpoint_hits = checkpoint_hits + hit_mask.float()
                checkpoint_has_any_hit = checkpoint_has_any_hit | hit_mask

            reached_goal = active & (next_nodes == goal_targets)
            reward = reward + reached_goal.float() * goal_reward

            threshold_8_now = active & (goal_hops_after <= 8.0) & (~goal_threshold_hit_8)
            threshold_4_now = active & (goal_hops_after <= 4.0) & (~goal_threshold_hit_4)
            threshold_2_now = active & (goal_hops_after <= 2.0) & (~goal_threshold_hit_2)
            reward = reward + threshold_8_now.float() * self.goal_hop_bonus_8
            reward = reward + threshold_4_now.float() * self.goal_hop_bonus_4
            reward = reward + threshold_2_now.float() * self.goal_hop_bonus_2
            goal_threshold_hit_8 |= threshold_8_now
            goal_threshold_hit_4 |= threshold_4_now
            goal_threshold_hit_2 |= threshold_2_now
            goal_hop_1_hit |= active & (goal_hops_after <= 1.0)

            regression_now = had_best4_before & active & (goal_hops_after > goal_hops_before)
            regression_large = regression_now & (goal_hops_after >= (best_goal_hops + 2.0))
            regression_small = regression_now & (~regression_large)
            reward = reward - regression_small.float() * self.goal_regression_penalty_small
            reward = reward - regression_large.float() * self.goal_regression_penalty_large
            goal_regression_after_best4_hit |= regression_now

            improved = goal_dist_after < (best_goal_dist - 1e-6)
            best_goal_dist = torch.where(improved, goal_dist_after, best_goal_dist)
            best_goal_hops = torch.minimum(best_goal_hops, goal_hops_after)
            stagnation_steps = torch.where(
                improved | reached_goal,
                torch.zeros_like(stagnation_steps),
                stagnation_steps + active.long(),
            )
            stagnation_reset_on_best_goal_improve += improved.float()

            # [Perf] bool(any()) GPU 동기화 제거: torch.where는 빈 마스크에도 비용이 거의 없음
            goal4_reset_now = (
                active
                & (best_goal_hops <= 4.0)
                & (~goal4_patience_granted)
            )
            stagnation_steps = torch.where(
                goal4_reset_now,
                torch.zeros_like(stagnation_steps),
                stagnation_steps,
            )
            stagnation_patience_budget = torch.where(
                goal4_reset_now,
                stagnation_patience_budget + self.near_goal_patience_bonus,
                stagnation_patience_budget,
            )
            goal4_patience_granted |= goal4_reset_now
            stagnation_reset_goal4 += goal4_reset_now.float()

            goal2_reset_now = (
                active
                & (best_goal_hops <= 2.0)
                & (~goal2_patience_granted)
            )
            stagnation_steps = torch.where(
                goal2_reset_now,
                torch.zeros_like(stagnation_steps),
                stagnation_steps,
            )
            goal2_patience_granted |= goal2_reset_now
            stagnation_reset_goal2 += goal2_reset_now.float()

            loop_fail_now = active & (self.env.visit_count[batch_indices, next_nodes] > self.loop_limit)
            stagnation_now = active & (stagnation_steps >= stagnation_patience_budget)
            fail_now = (loop_fail_now | stagnation_now) & (~reached_goal)
            reward = reward + fail_now.float() * fail_penalty

            success |= reached_goal
            loop_fail |= loop_fail_now
            stagnation_fail |= stagnation_now
            done |= reached_goal | loop_fail_now | stagnation_now
            step_counts = step_counts + active.float()

            expert_next = self.env.weighted_next_hop_matrix[current_nodes, goal_targets]
            # [Perf] 디버그 에피소드에서만 path_traces 수집 (GPU→CPU 동기화 제거)
            if _collect_path:
                _next_cpu = next_nodes.detach().cpu().tolist()
                for batch_idx in range(batch_size):
                    path_traces[batch_idx].append(_next_cpu[batch_idx])

            ce_mask = active & (expert_next >= 0)
            if bool(ce_mask.any()):
                ce_loss = F.cross_entropy(
                    row_scores[ce_mask], expert_next[ce_mask], reduction="none"
                )
                near_goal_ce_mask = goal_hops_after[ce_mask] <= 2.0
                ce_weights = torch.ones_like(ce_loss)
                ce_weights = torch.where(
                    near_goal_ce_mask,
                    torch.full_like(ce_weights, self.near_goal_ce_mult),
                    ce_weights,
                )
                aux_losses.append((ce_loss * ce_weights).mean())
                near_goal_ce_applied_steps += (
                    active & (goal_hops_after <= 2.0) & (expert_next >= 0)
                ).float()
                ce_eligible_steps += ce_mask.float()

            entropy_weight = torch.where(
                active & (goal_hops_after <= 2.0),
                torch.full_like(entropy, self.terminal_entropy_mult),
                torch.ones_like(entropy),
            )

            log_probs.append(log_prob)
            values.append(value_pred.view(-1))
            entropies.append(entropy)
            entropy_weights.append(entropy_weight)
            rewards.append(reward)
            valid_masks.append(active)

        if bool((~done).any()):
            remaining = ~done
            rewards[-1] = rewards[-1] + remaining.float() * fail_penalty
            timeout_fail |= remaining
            done |= remaining

        rewards_t = torch.stack(rewards, dim=0)
        values_t = torch.stack(values, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)
        entropies_t = torch.stack(entropies, dim=0)
        entropy_weights_t = torch.stack(entropy_weights, dim=0)
        valid_t = torch.stack(valid_masks, dim=0).bool()

        returns_t = torch.zeros_like(rewards_t)
        running = torch.zeros(batch_size, device=self.device)
        for step_idx in range(rewards_t.size(0) - 1, -1, -1):
            running = rewards_t[step_idx] + gamma * running
            returns_t[step_idx] = running
            running = torch.where(valid_t[step_idx], running, torch.zeros_like(running))

        valid_returns = returns_t[valid_t]
        norm_returns = torch.zeros_like(returns_t)
        if valid_returns.numel() > 0:
            norm_valid_returns, _, _ = self._normalize_returns(valid_returns)
            norm_returns[valid_t] = norm_valid_returns
        else:
            norm_valid_returns = valid_returns

        flat_log_probs = log_probs_t[valid_t]
        flat_values = values_t[valid_t]
        flat_entropies = entropies_t[valid_t]
        flat_entropy_weights = entropy_weights_t[valid_t]
        flat_norm_returns = norm_returns[valid_t]

        advantages = flat_norm_returns - flat_values.detach()

        # Advantage 정규화: 학습 안정화 및 Critic 설명력(ExplVar) 복구
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(advantages * flat_log_probs).mean() if flat_log_probs.numel() > 0 else torch.tensor(0.0, device=self.device)
        critic_loss = F.smooth_l1_loss(flat_values, flat_norm_returns) if flat_values.numel() > 0 else torch.tensor(0.0, device=self.device)
        entropy_bonus = (
            -0.01 * (flat_entropies * flat_entropy_weights).mean()
            if flat_entropies.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )
        worker_aux_ce = (
            torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=self.device)
        )
        total_loss = policy_loss + 0.5 * critic_loss + entropy_bonus + aux_weight * worker_aux_ce

        final_goal_dist = self.env.apsp_matrix[self.env.current_node, goal_targets].float()
        final_goal_hops = self.env.hop_matrix[self.env.current_node, goal_targets].float()
        progress = 1.0 - final_goal_dist / start_goal_dist.clamp(min=1.0)
        goal_progress_per100 = (progress * 100.0 / step_counts.clamp(min=1.0)).mean()
        path_length_ratio = step_counts / optimal_hops.clamp(min=1.0)
        goal_neighbor_rate = goal_neighbor_ever.float().mean() * 100.0
        goal_neighbor_success_rate = (
            goal_neighbor_success_steps.sum()
            / goal_neighbor_opportunity_steps.sum().clamp(min=1.0)
        ) * 100.0
        goal_hop_1_hit_rate = goal_hop_1_hit.float().mean() * 100.0
        if bool(goal_hop_1_hit.any()):
            goal_hop_1_to_success_rate = (
                success[goal_hop_1_hit].float().mean() * 100.0
            )
        else:
            goal_hop_1_to_success_rate = torch.tensor(0.0, device=self.device)
        goal_regression_after_best4_rate = (
            goal_regression_after_best4_hit.float().mean() * 100.0
        )
        near_goal_ce_mult_applied_rate = (
            near_goal_ce_applied_steps.sum()
            / ce_eligible_steps.sum().clamp(min=1.0)
        ) * 100.0

        any_ckpt_mask = checkpoint_counts > 0
        if bool(any_ckpt_mask.any()):
            checkpoint_hit_rate = checkpoint_has_any_hit[any_ckpt_mask].float().mean() * 100.0
            checkpoint_completion = (
                checkpoint_hits[any_ckpt_mask].sum() / checkpoint_counts[any_ckpt_mask].sum().clamp(min=1.0)
            ) * 100.0
        else:
            checkpoint_hit_rate = torch.tensor(0.0, device=self.device)
            checkpoint_completion = torch.tensor(0.0, device=self.device)

        checkpoint_ptrs_f = checkpoint_ptrs.float()
        has_rank_1 = checkpoint_counts >= 1.0
        has_rank_2 = checkpoint_counts >= 2.0
        has_rank_3plus = checkpoint_counts > 2.0
        if bool(has_rank_1.any()):
            hit_at_rank_1 = (checkpoint_ptrs_f[has_rank_1] >= 1.0).float().mean() * 100.0
        else:
            hit_at_rank_1 = torch.tensor(0.0, device=self.device)
        if bool(has_rank_2.any()):
            hit_at_rank_2 = (checkpoint_ptrs_f[has_rank_2] >= 2.0).float().mean() * 100.0
        else:
            hit_at_rank_2 = torch.tensor(0.0, device=self.device)
        if bool(has_rank_3plus.any()):
            rank3plus_hits = (checkpoint_ptrs_f[has_rank_3plus] - 2.0).clamp(min=0.0).sum()
            rank3plus_avail = (checkpoint_counts[has_rank_3plus] - 2.0).clamp(min=0.0).sum()
            hit_at_rank_3plus = (rank3plus_hits / rank3plus_avail.clamp(min=1.0)) * 100.0
        else:
            hit_at_rank_3plus = torch.tensor(0.0, device=self.device)

        sample_payload = {
            "episode": int(ep),
            "start_node": int(path_traces[0][0]) if _collect_path and path_traces[0] else -1,
            "goal_node": int(goal_targets[0].item()),
            "teacher_path": teacher_paths[0],
            "hidden_checkpoints": hidden_lists[0],
            "checkpoint_hits": int(checkpoint_hits[0].item()),
            "path_trace": path_traces[0] if _collect_path else [],
            "success": bool(success[0].item()),
            "final_goal_distance": float(final_goal_dist[0].item()),
        }
        if self._collect_debug_this_episode:
            with open(self._debug_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample_payload, ensure_ascii=True) + "\n")

        diag = {
            "reward_mean": float(rewards_t.sum(dim=0).mean().item()),
            "success_rate": float(success.float().mean().item() * 100.0),
            "goal_dist_mean": float(final_goal_dist.mean().item()),
            "goal_hop_mean": float(final_goal_hops.mean().item()),
            "best_goal_dist_mean": float(best_goal_dist.mean().item()),
            "best_goal_hops_mean": float(best_goal_hops.mean().item()),
            "progress_mean": float(progress.mean().item() * 100.0),
            "goal_progress_per100": float(goal_progress_per100.item()),
            "stagnation_fail_rate": float(stagnation_fail.float().mean().item() * 100.0),
            "loop_fail_rate": float(loop_fail.float().mean().item() * 100.0),
            "timeout_fail_rate": float(timeout_fail.float().mean().item() * 100.0),
            "path_length_mean": float(step_counts.mean().item()),
            "path_length_ratio": float(path_length_ratio.mean().item()),
            "teacher_path_hops_mean": float(optimal_hops.mean().item()),
            "teacher_hidden_checkpoint_count_actual": float(teacher_hidden_checkpoint_count_actual.mean().item()),
            "worker_aux_ce_loss": float(worker_aux_ce.item()),
            "worker_critic_loss": float(critic_loss.item()) if 'critic_loss' in locals() else 0.0,
            "worker_entropy": float(flat_entropies.mean().item()) if flat_entropies.numel() > 0 else 0.0,
            "critic_explained_variance": self._explained_variance(flat_norm_returns, flat_values.detach()),
            "hidden_checkpoint_count_mean": float(checkpoint_counts.mean().item()),
            "hidden_checkpoint_hit_rate": float(checkpoint_hit_rate.item()),
            "ordered_checkpoint_completion_ratio": float(checkpoint_completion.item()),
            "hidden_checkpoint_hit_at_rank_1": float(hit_at_rank_1.item()),
            "hidden_checkpoint_hit_at_rank_2": float(hit_at_rank_2.item()),
            "hidden_checkpoint_hit_at_rank_3plus": float(hit_at_rank_3plus.item()),
            "goal_neighbor_rate": float(goal_neighbor_rate.item()),
            "goal_neighbor_success_rate": float(goal_neighbor_success_rate.item()),
            "goal_hop_1_hit_rate": float(goal_hop_1_hit_rate.item()),
            "goal_hop_1_to_success_rate": float(goal_hop_1_to_success_rate.item()),
            "goal_regression_after_best4_rate": float(goal_regression_after_best4_rate.item()),
            "near_goal_ce_mult_applied_rate": float(near_goal_ce_mult_applied_rate.item()),
            "goal_threshold_hit_8_rate": float(goal_threshold_hit_8.float().mean().item() * 100.0),
            "goal_threshold_hit_4_rate": float(goal_threshold_hit_4.float().mean().item() * 100.0),
            "goal_threshold_hit_2_rate": float(goal_threshold_hit_2.float().mean().item() * 100.0),
            "stagnation_reset_on_best_goal_improve_count": float(
                stagnation_reset_on_best_goal_improve.mean().item()
            ),
            "stagnation_reset_near_goal_count": float(
                (stagnation_reset_goal4 + stagnation_reset_goal2).mean().item()
            ),
            "stagnation_reset_goal4_count": float(
                stagnation_reset_goal4.mean().item()
            ),
            "stagnation_reset_goal2_count": float(
                stagnation_reset_goal2.mean().item()
            ),
            "policy_loss": float(policy_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy_bonus": float(entropy_bonus.item()),
            "aux_weight": float(aux_weight),
            "hidden_bonus_weight": float(per_hit_bonus),
            "worker_grad_pre": 0.0,
            "worker_grad_post": 0.0,
            "worker_grad_clip_hit": 0.0,
        }
        return total_loss, diag

    def train(self, episodes):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.debug_mode:
            self._init_debug_outputs()
            with open(self._debug_log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"=== Worker Stage Debug Log (LR={self.config.lr}, POMO={self.num_pomo}) ===\n\n"
                )

        # min → max 변경: LR이 wkr_lr_floor 이하로 떨어지지 않도록 하한선 보장
        self.wkr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.wkr_opt,
            T_max=max(episodes, 1),
            eta_min=max(float(self.config.lr) * 0.1, self.wkr_lr_floor),
        )

        rl_history = {
            "rewards": [],
            "losses": [],
            "path_lengths": [],
            "success_rates": [],
        }
        success_ema = 0.0

        # [Gradient Accumulation] REINFORCE 고분산 억제를 위한 가상 배치 확대
        ACCUM_STEPS = 4
        self.wkr_opt.zero_grad(set_to_none=True)

        tqdm_disabled = getattr(self.config, 'disable_tqdm', False)
        pbar = tqdm(range(episodes), desc="Worker", dynamic_ncols=True, disable=tqdm_disabled)
        for ep in pbar:
            self._collect_debug_this_episode = self._should_collect_debug(ep, episodes)

            self.env.set_curriculum_ratio(min(1.0, ep / max(int(episodes * 0.8), 1)))
            self.env.reset(batch_size=self.num_pomo, sync_problem=True)

            total_loss, diag = self._execute_goal_conditioned_rollout(ep, episodes)

            # [Gradient Accumulation] Loss를 ACCUM_STEPS로 나눠서 누적
            scaled_loss = total_loss / ACCUM_STEPS

            if getattr(scaled_loss, 'requires_grad', False):
                scaled_loss.backward()
            
            grad_pre = 0.0
            grad_post = 0.0
            clip_hit = 0.0
            
            # ACCUM_STEPS 마다 가중치 업데이트
            if (ep + 1) % ACCUM_STEPS == 0:
                grad_pre = self._compute_grad_norm(self.worker.parameters())
                grad_post = grad_pre
                if self.wkr_max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.worker.parameters(), self.wkr_max_grad_norm)
                    grad_post = self._compute_grad_norm(self.worker.parameters())
                    clip_hit = 100.0 if grad_pre > self.wkr_max_grad_norm + 1e-8 else 0.0
                self.wkr_opt.step()
                self.wkr_opt.zero_grad(set_to_none=True)  # 업데이트 후 gradient 초기화
            self.wkr_scheduler.step()
            diag["worker_grad_pre"] = float(grad_pre)
            diag["worker_grad_post"] = float(grad_post)
            diag["worker_grad_clip_hit"] = float(clip_hit)

            success_ema = 0.95 * success_ema + 0.05 * diag["success_rate"]
            reward_mean = diag["reward_mean"]
            rl_history["rewards"].append(reward_mean)
            rl_history["losses"].append(float(total_loss.item()))
            rl_history["path_lengths"].append(diag["path_length_mean"])

            if getattr(self.config, 'disable_tqdm', False):
                import sys
                info_str = f"Loss={total_loss.item():.2f}, Succ={diag['success_rate']:.1f}%, EMA={success_ema:.1f}%, Rw={reward_mean:.2f}, Len={diag['path_length_mean']:.1f}"
                sys.stderr.write(f"PROGRESS_UPDATE|{ep + 1}|{info_str}\n")
                
                # [주기적 디버그 로그 설정] 200 에피소드마다 자세한 지표 IPC 전송
                if (ep + 1) % 200 == 0:
                    ent = diag.get("worker_entropy", 0.0)
                    v_loss = diag.get("worker_critic_loss", 0.0)
                    expl_var = diag.get("critic_explained_variance", 0.0)
                    hop_dist = diag.get("teacher_path_hops_mean", 0.0)  # 시작점 기준 목표까지의 평균 홉
                    prog = diag.get("progress_mean", 0.0)
                    grad = diag.get("worker_grad_pre", 0.0)
                    
                    debug_str = (
                        f"📊 [EP {ep+1:5d}] W_Ent: {ent:.3f} | V_Loss: {v_loss:.3f} | Grad: {grad:.4f} | ExplVar: {expl_var:.2f} | "
                        f"Hop_Dist: {hop_dist:.1f} | Prog: {prog:.1f}% | EMA: {success_ema:.1f}%"
                    )
                    sys.stderr.write(f"DEBUG_UPDATE|{debug_str}\n")
                
                sys.stderr.flush()
            else:
                pbar.set_postfix_str(f"Loss={total_loss.item():.2f}, Succ={diag['success_rate']:.1f}%, EMA={success_ema:.1f}%, Rw={reward_mean:.2f}, Len={diag['path_length_mean']:.1f}")

            # _plot_rl_curves는 0~1 범위를 가정하므로 % → 비율로 변환 저장
            rl_history["success_rates"].append(success_ema / 100.0)

            if self._collect_debug_this_episode:
                row = {
                    "episode": int(ep),
                    "success_rate": diag["success_rate"],
                    "success_ema": float(success_ema),
                    "goal_dist_mean": diag["goal_dist_mean"],
                    "goal_hop_mean": diag["goal_hop_mean"],
                    "best_goal_dist_mean": diag["best_goal_dist_mean"],
                    "best_goal_hops_mean": diag["best_goal_hops_mean"],
                    "progress_mean": diag["progress_mean"],
                    "goal_progress_per100": diag["goal_progress_per100"],
                    "stagnation_fail_rate": diag["stagnation_fail_rate"],
                    "loop_fail_rate": diag["loop_fail_rate"],
                    "timeout_fail_rate": diag["timeout_fail_rate"],
                    "path_length_ratio": diag["path_length_ratio"],
                    "teacher_path_hops_mean": diag["teacher_path_hops_mean"],
                    "teacher_hidden_checkpoint_count_actual": diag["teacher_hidden_checkpoint_count_actual"],
                    "worker_aux_ce_loss": diag["worker_aux_ce_loss"],
                    "worker_entropy": diag["worker_entropy"],
                    "critic_explained_variance": diag["critic_explained_variance"],
                    "hidden_checkpoint_count_mean": diag["hidden_checkpoint_count_mean"],
                    "hidden_checkpoint_hit_rate": diag["hidden_checkpoint_hit_rate"],
                    "ordered_checkpoint_completion_ratio": diag["ordered_checkpoint_completion_ratio"],
                    "hidden_checkpoint_hit_at_rank_1": diag["hidden_checkpoint_hit_at_rank_1"],
                    "hidden_checkpoint_hit_at_rank_2": diag["hidden_checkpoint_hit_at_rank_2"],
                    "hidden_checkpoint_hit_at_rank_3plus": diag["hidden_checkpoint_hit_at_rank_3plus"],
                    "goal_neighbor_rate": diag["goal_neighbor_rate"],
                    "goal_neighbor_success_rate": diag["goal_neighbor_success_rate"],
                    "goal_hop_1_hit_rate": diag["goal_hop_1_hit_rate"],
                    "goal_hop_1_to_success_rate": diag["goal_hop_1_to_success_rate"],
                    "goal_regression_after_best4_rate": diag["goal_regression_after_best4_rate"],
                    "near_goal_ce_mult_applied_rate": diag["near_goal_ce_mult_applied_rate"],
                    "goal_threshold_hit_8_rate": diag["goal_threshold_hit_8_rate"],
                    "goal_threshold_hit_4_rate": diag["goal_threshold_hit_4_rate"],
                    "goal_threshold_hit_2_rate": diag["goal_threshold_hit_2_rate"],
                    "stagnation_reset_on_best_goal_improve_count": diag["stagnation_reset_on_best_goal_improve_count"],
                    "stagnation_reset_near_goal_count": diag["stagnation_reset_near_goal_count"],
                    "stagnation_reset_goal4_count": diag["stagnation_reset_goal4_count"],
                    "stagnation_reset_goal2_count": diag["stagnation_reset_goal2_count"],
                    "reward_mean": reward_mean,
                    "loss": float(total_loss.item()),
                    "wkr_lr": float(self.wkr_scheduler.get_last_lr()[0]),
                    "worker_grad_pre": diag["worker_grad_pre"],
                    "worker_grad_post": diag["worker_grad_post"],
                    "worker_grad_clip_hit": diag["worker_grad_clip_hit"],
                    "hidden_bonus_weight": diag["hidden_bonus_weight"],
                    "aux_weight": diag["aux_weight"],
                }
                self._append_debug_csv(row)
                lines = [
                    f"[Ep {ep}] Wkr LR: {row['wkr_lr']:.6f}, SuccessEMA: {success_ema:.1f}%, Loss: {row['loss']:.2f}",
                    f"  Success / GoalDist / Progress: {row['success_rate']:.1f}% / "
                    f"{row['goal_dist_mean']:.2f} / {row['progress_mean']:.1f}%",
                    f"  TeacherHop / HiddenCount / GoalHop / BestHop: {row['teacher_path_hops_mean']:.2f} / "
                    f"{row['teacher_hidden_checkpoint_count_actual']:.2f} / {row['goal_hop_mean']:.2f} / {row['best_goal_hops_mean']:.2f}",
                    f"  BestGoalDist / Goal per100: {row['best_goal_dist_mean']:.2f} / {row['goal_progress_per100']:.2f}",
                    f"  Stag / Loop / Timeout: {row['stagnation_fail_rate']:.1f}% / "
                    f"{row['loop_fail_rate']:.1f}% / {row['timeout_fail_rate']:.1f}%",
                    f"  PathRatio / HiddenCount / HiddenHit / Completion: "
                    f"{row['path_length_ratio']:.3f} / {row['hidden_checkpoint_count_mean']:.2f} / "
                    f"{row['hidden_checkpoint_hit_rate']:.1f}% / {row['ordered_checkpoint_completion_ratio']:.1f}%",
                    f"  Hit@1 / Hit@2 / Hit@3+: {row['hidden_checkpoint_hit_at_rank_1']:.1f}% / "
                    f"{row['hidden_checkpoint_hit_at_rank_2']:.1f}% / {row['hidden_checkpoint_hit_at_rank_3plus']:.1f}%",
                    f"  GoalNbr / GoalNbrSucc / Hop1 / Hop1->Succ: {row['goal_neighbor_rate']:.1f}% / "
                    f"{row['goal_neighbor_success_rate']:.1f}% / {row['goal_hop_1_hit_rate']:.1f}% / {row['goal_hop_1_to_success_rate']:.1f}%",
                    f"  GoalRegress / NearGoalCE: {row['goal_regression_after_best4_rate']:.1f}% / "
                    f"{row['near_goal_ce_mult_applied_rate']:.1f}%",
                    f"  Goal<=8/4/2: {row['goal_threshold_hit_8_rate']:.1f}% / "
                    f"{row['goal_threshold_hit_4_rate']:.1f}% / {row['goal_threshold_hit_2_rate']:.1f}%",
                    f"  StagReset best/near/4/2: {row['stagnation_reset_on_best_goal_improve_count']:.2f} / "
                    f"{row['stagnation_reset_near_goal_count']:.2f} / {row['stagnation_reset_goal4_count']:.2f} / {row['stagnation_reset_goal2_count']:.2f}",
                    f"  AuxCE / Entropy / ExplVar: {row['worker_aux_ce_loss']:.3f} / "
                    f"{row['worker_entropy']:.3f} / {row['critic_explained_variance']:.3f}",
                    f"  Grad pre/post: {row['worker_grad_pre']:.4f} / {row['worker_grad_post']:.4f} "
                    f"(clip {row['worker_grad_clip_hit']:.1f}%)",
                    f"  HiddenBonus/AuxWeight: {row['hidden_bonus_weight']:.3f} / {row['aux_weight']:.3f}",
                ]
                # pbar.write("\n".join(lines)) # Disabled to prevent console scrolling in parallel mode
                with open(self._debug_log_path, "a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")
                    f.write("\n")

            pbar.set_postfix(
                Succ=f"{diag['success_rate']:.1f}%",
                EMA=f"{success_ema:.1f}%",
                Loss=f"{float(total_loss.item()):.2f}",
            )

        self._save_worker_checkpoint(
            "final.pt",
            episodes - 1,
            metric=success_ema,
            metric_name="success_ema",
            extra_payload={"phase1_mode": "apte_guided_worker"},
        )
        best_path = os.path.join(self.save_dir, "best.pt")
        if os.path.exists(best_path):
            os.remove(best_path)
        self._plot_rl_curves(rl_history, title="Worker Stage Learning Curves")
