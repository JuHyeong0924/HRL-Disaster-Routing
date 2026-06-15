import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ManagerPOMOTrainer:
    def __init__(self, manager, hrl_env, K=10, lr=1e-4, max_grad_norm=1.0):
        self.manager = manager
        self.env = hrl_env
        self.K = K
        self.optimizer = optim.Adam(manager.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        
    def train_step(self, batch_size=32, num_targets=10):
        self.manager.train()
        B_true = batch_size
        K = self.K
        B_env = B_true * K
        
        # 1. 시나리오 생성
        self.env.reset(batch_size=B_true, num_targets=num_targets)
        
        # 2. POMO를 위한 K개 복제
        # (B_true) -> (B_true * K)
        saved = self.env._saved
        self.env.batch_size = B_env
        self.env.targets = saved['targets'].repeat_interleave(K, dim=0)
        self.env.target_zones = saved['target_zones'].repeat_interleave(K, dim=0)
        self.env.deadlines = saved['deadlines'].repeat_interleave(K, dim=0)
        saved['start_nodes'] = saved['start_nodes'].repeat_interleave(K, dim=0)
        
        # Restore will initialize the B_env size buffers
        self.env.restore()
        
        device = self.env.env.device
        all_log_probs = []
        
        # ── Zone Encoder (1회) ──
        zone_features = self.env.get_zone_features() # [B_env, K_zones, 6]
        zone_edge_index = self.env.get_zone_edge_index() # [2, E]
        
        # 배치 병합을 위한 PyG 형태 변환
        K_zones = self.env.env.k
        flat_zf = zone_features.view(B_env * K_zones, 6)
        
        ai = torch.arange(B_env, device=device).repeat_interleave(K_zones)
        aei = torch.cat([zone_edge_index + i*K_zones for i in range(B_env)], dim=1)
        
        zone_emb = self.manager.encode_zones(flat_zf, aei, batch=ai) # [B_env * K_zones, 128]
        
        # ── 첫 번째 Target 강제 선택 (POMO K rollout) ──
        # B_true 마다 K개의 복제본이 있으므로, 각 복제본은 첫 번째로 갈 타겟을 (k % num_targets) 로 강제 선택
        k_indices = torch.arange(B_env, device=device) % num_targets
        curr_target = k_indices # [B_env]
        
        # 첫 번째 타겟에 대한 예측은 강제이므로 로그 확률 0, 하지만 Zone은 Manager가 직접 예측해야 함!
        target_features = self.env.get_target_features() # [B_env, N, 4]
        tz_idx = self.env.target_zones.view(-1)
        offsets = torch.arange(B_env, device=device).repeat_interleave(num_targets) * K_zones
        flat_tz_idx = offsets + tz_idx
        t_emb = self.manager.get_target_embeddings(zone_emb, target_features.view(-1, 4), flat_tz_idx) # [B_env*N, 128]
        
        t_emb_dense = t_emb.view(B_env, num_targets, 128)
        selected_t_emb = t_emb_dense[torch.arange(B_env), curr_target] # [B_env, 128]
        
        # 초기 Context 생성
        h_last = torch.zeros(B_env, 128, device=device)
        elapsed = self.env.current_time.unsqueeze(1) / self.env.max_time
        rescued = self.env.num_rescued.unsqueeze(1).float() / num_targets
        query = self.manager.generate_context(h_last, elapsed, rescued) # [B_env, 128]
        
        # 첫 번째 Zone Action 네트워크 예측
        z_ai = torch.arange(B_env, device=device).repeat_interleave(K_zones)
        zone_adj_mask = self.env.get_zone_adj_mask()
        z_logits, z_inv = self.manager.get_zone_logits(query, selected_t_emb, zone_emb, zone_adj_mask.view(-1), z_ai)
        z_dist = Categorical(logits=z_logits)
        curr_zone_action = z_dist.sample() # [B_env]
        z_log_prob = z_dist.log_prob(curr_zone_action) # [B_env]
        
        # POMO 강제 선택은 타겟뿐이므로 타겟의 log_prob는 0으로 간주하고, Zone의 확률만 초기 기록
        full_log_prob = z_log_prob
        all_log_probs.append(full_log_prob)
        
        manager_turns = torch.zeros(B_env, dtype=torch.long, device=device)
        total_worker_steps = torch.zeros(B_env, dtype=torch.long, device=device)
        num_rescued = torch.zeros(B_env, dtype=torch.long, device=device)
        dones = torch.zeros(B_env, dtype=torch.bool, device=device)
        
        # Target Fusion (한번 계산해둠, Target 피처는 시간에 따라 바뀌지만 일단 고정 피처로 근사)
        # 시간 변경 등을 실시간 반영하려면 step마다 호출 필요
        
        while not dones.all():
            # ── Worker 실행 (Frozen) ──
            events, new_dones = self.env.step_manager(curr_target, curr_zone_action)
            dones = new_dones
            
            # 이벤트 판정 및 보상 집계
            for b in range(B_env):
                if events[b] == 'target_rescued':
                    if self.env.current_time[b] <= self.env.deadlines[b, curr_target[b]]:
                        num_rescued[b] += 1
            
            if dones.all():
                break
                
            active = [b for b in range(B_env) if not dones[b]]
            A = len(active)
            if A == 0: break
            
            # ── Manager 재호출 ──
            # 활성 배치에 대해서만 Feature 추출
            target_features = self.env.get_target_features() # [B_env, N, 4]
            target_mask = self.env.get_target_mask() # [B_env, N]
            zone_adj_mask = self.env.get_zone_adj_mask() # [B_env, K_zones]
            
            # active 배치만 슬라이싱
            act_target_features = target_features[active] # [A, N, 4]
            act_target_mask = target_mask[active] # [A, N]
            act_zone_adj_mask = zone_adj_mask[active] # [A, K_zones]
            
            # Target Embeddings
            t_ai = torch.arange(A, device=device).repeat_interleave(num_targets)
            tz_idx = self.env.target_zones[active].view(-1) # [A*N]
            # tz_idx는 각 batch instance 내의 offset이 아님.
            # zone_emb가 [B_env * K_zones, 128] 이므로 올바른 offset 계산 필요
            act_offsets = torch.tensor(active, device=device).repeat_interleave(num_targets) * K_zones
            flat_target_zone_idx = act_offsets + tz_idx
            
            t_emb = self.manager.get_target_embeddings(
                zone_emb, act_target_features.view(-1, 4), flat_target_zone_idx
            ) # [A*N, 128]
            
            # Context
            h_last = torch.zeros(A, 128, device=device) # 임시로 0 (추후 이전 타겟 임베딩 재활용)
            elapsed = self.env.current_time[active].unsqueeze(1) / self.env.max_time
            rescued = self.env.num_rescued[active].unsqueeze(1).float() / num_targets
            query = self.manager.generate_context(h_last, elapsed, rescued) # [A, 128]
            
            # Target Action
            t_logits, t_inv, t_emb_dense = self.manager.get_target_logits(query, t_emb, act_target_mask.view(-1), t_ai)
            t_dist = Categorical(logits=t_logits)
            t_act = t_dist.sample() # [A]
            t_log_prob = t_dist.log_prob(t_act) # [A]
            
            # Zone Action
            z_ai = torch.arange(A, device=device).repeat_interleave(K_zones)
            # zone_emb 중 active한 것만 추출
            act_zone_idx = (torch.tensor(active, device=device).unsqueeze(1) * K_zones + torch.arange(K_zones, device=device).unsqueeze(0)).view(-1)
            act_zone_emb = zone_emb[act_zone_idx] # [A*K_zones, 128]
            
            selected_t_emb = t_emb_dense[torch.arange(A), t_act] # [A, 128]
            z_logits, z_inv = self.manager.get_zone_logits(query, selected_t_emb, act_zone_emb, act_zone_adj_mask.view(-1), z_ai)
            z_dist = Categorical(logits=z_logits)
            z_act = z_dist.sample() # [A]
            z_log_prob = z_dist.log_prob(z_act) # [A]
            
            # 확률 업데이트
            full_log_prob = torch.zeros(B_env, device=device)
            full_log_prob[active] = t_log_prob + z_log_prob
            all_log_probs.append(full_log_prob)
            
            # Action 반영
            new_curr_target = curr_target.clone()
            new_curr_zone_action = curr_zone_action.clone()
            for idx, b in enumerate(active):
                new_curr_target[b] = t_act[idx]
                new_curr_zone_action[b] = z_act[idx]
            curr_target = new_curr_target
            curr_zone_action = new_curr_zone_action

        # ── 에피소드 보상 계산 ──
        # [B_true * K]
        rewards = (num_rescued.float() * 10.0 
                   - 0.5 * self.env.manager_turns.float() 
                   - 0.1 * self.env.worker_steps.float())
                   
        rewards = rewards.view(B_true, K)
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = (rewards - baseline).view(-1) # [B_true * K]
        
        # Loss 계산
        loss = 0.0
        if len(all_log_probs) > 0:
            sum_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) # [B_env]
            loss = -(advantages * sum_log_probs).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        return {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else 0.0,
            'mean_reward': rewards.mean().item(),
            'mean_rescued': num_rescued.float().mean().item(),
            'mean_manager_turns': self.env.manager_turns.float().mean().item(),
            'mean_worker_steps': self.env.worker_steps.float().mean().item()
        }
