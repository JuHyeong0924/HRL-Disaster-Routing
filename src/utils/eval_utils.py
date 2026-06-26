import os
import torch
import networkx as nx
import numpy as np
import time

from src.envs.worker_env import WorkerEnv
from src.models.worker import Worker
from src.models.manager import Manager
from src.envs.hrl_env import HRLEnv

def load_eval_env(map_name: str, device: torch.device, disaster_prob: float = 0.05, dynamic_disaster: bool = True) -> tuple[WorkerEnv, HRLEnv]:
    worker_env = WorkerEnv(
        f'data/{map_name}_node.tntp', f'data/{map_name}_net.tntp',
        zone_json=f'data/grid_{map_name}_node_to_zone.json',
        zone_graph_json=f'data/grid_{map_name}_zone_graph.json',
        masking_mode='soft_curr_next',
        disaster_prob=disaster_prob,
        dynamic_disaster=dynamic_disaster,
        device=str(device)
    )
    
    # Dummy worker for env creation, it will be loaded later
    worker = Worker(node_dim=6, hidden_dim=256, num_layers=2, dropout=0.0).to(device)
    hrl_env = HRLEnv(worker, worker_env)
    return worker_env, hrl_env

def load_neural_models(device: torch.device, worker_ckpt: str, manager_ckpt: str) -> tuple[Worker, Manager]:
    worker = Worker(node_dim=6, hidden_dim=256, num_layers=2, dropout=0.0).to(device)
    if os.path.exists(worker_ckpt):
        ckpt = torch.load(worker_ckpt, map_location=device, weights_only=False)
        state = ckpt.get('worker_state', ckpt.get('state_dict', ckpt))
        cur = worker.state_dict()
        compat = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
        worker.load_state_dict(compat, strict=False)
    worker.eval()
    
    manager = Manager(
        zone_dim=6, target_dim=6, hidden_dim=256,  # [Phase 2B] target_dim 4→6
        num_gat_layers=3, gat_heads=4, num_transformer_layers=3, transformer_heads=4
    ).to(device)
    if os.path.exists(manager_ckpt):
        m_ckpt = torch.load(manager_ckpt, map_location=device, weights_only=False)
        m_state = m_ckpt.get('manager_state', m_ckpt.get('state_dict', m_ckpt.get('model_state_dict', m_ckpt)))
        manager.load_state_dict(m_state, strict=False)
    manager.eval()
    
    return worker, manager

def get_manager_action(manager: Manager, hrl_env: HRLEnv, device: torch.device, forced_t_act: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scene = hrl_env._get_scenario_dict()
        zf = scene['zone_features']
        zei = scene['zone_edge_index']
        tz = scene['target_zones'].view(-1)
        
        tf = hrl_env.get_target_features().view(-1, 6)  # [Phase 2A] 4→6
        tm = hrl_env.get_target_mask().view(-1)
        zam = hrl_env.get_zone_adj_mask().view(-1)
        
        c_nodes = hrl_env.env.curr_nodes
        elapsed = hrl_env.current_time.unsqueeze(-1) / max(hrl_env.max_time, 1.0)
        rescued = hrl_env.num_rescued.float().unsqueeze(-1) / float(hrl_env.num_targets)
        
        B = hrl_env.batch_size
        num_targets = hrl_env.num_targets
        K_zones = hrl_env.env.k
        
        flat_zf = zf.view(K_zones, 6)
        ai = torch.arange(B, device=device).repeat_interleave(K_zones)
        zam = hrl_env.get_zone_adj_mask()[0]
        
        # [Phase 2F] num_feasible, avg_urgency 계산
        target_mask_full = hrl_env.get_target_mask()  # [1, N]
        target_features_full = hrl_env.get_target_features()  # [1, N, 6]
        num_feasible = target_mask_full.sum(dim=-1, keepdim=True).float() / num_targets  # [1, 1]
        urgency_ch = target_features_full[:, :, 3]  # [1, N]
        mask_float = target_mask_full.float()
        avg_urgency = (urgency_ch * mask_float).sum(dim=-1, keepdim=True) / mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # [1, 1]
        
        zone_emb = manager.encode_zones(flat_zf, zei, batch=ai)
        
        t_ai = torch.zeros(num_targets, dtype=torch.long, device=device)
        t_emb = manager.get_target_embeddings(zone_emb, tf, tz)
        
        # [Phase 2F] context generator: h_last 제거
        query = manager.generate_context(elapsed, rescued, num_feasible, avg_urgency)
        
        t_logits, _, t_emb_dense = manager.get_target_logits(query, t_emb, tm, t_ai)
        t_act = t_logits.argmax(dim=-1)
        
        if forced_t_act is not None:
            t_act[0] = forced_t_act
            
        z_ai = torch.zeros(K_zones, dtype=torch.long, device=device)
        selected_t_emb = t_emb_dense[0, t_act[0]]
        selected_tz = tz[t_act[0]].unsqueeze(0)
        
        # ⚡ Prevent Infinite Loop: Mask out current zone if target is in a different zone
        c_node = hrl_env.env.idx_to_node[int(hrl_env.env.curr_nodes[0])]
        c_zone = hrl_env.env.n2z[c_node]
        if selected_tz[0].item() != c_zone:
            zam[c_zone] = 0
            
        z_dist_mat = getattr(hrl_env, 'zone_dist_matrix', None)
        if z_dist_mat is None:
            z_apsp = dict(nx.all_pairs_dijkstra_path_length(hrl_env.env.ZG, weight='weight'))
            zone_dist_matrix = np.full((hrl_env.env.k, hrl_env.env.k), np.inf)
            for u, lengths in z_apsp.items():
                for v, length in lengths.items():
                    zone_dist_matrix[u, v] = length
            z_dist_mat = torch.tensor(zone_dist_matrix, dtype=torch.float32, device=device).unsqueeze(0)
            hrl_env.zone_dist_matrix = z_dist_mat
        elif len(z_dist_mat.shape) == 2:
            z_dist_mat = z_dist_mat.unsqueeze(0)
            
        z_logits, _ = manager.get_zone_logits(
            query, selected_t_emb.unsqueeze(0), zone_emb, zam, z_ai, selected_tz, z_dist_mat
        )
        z_act = z_logits.argmax(dim=-1)

        
        return torch.tensor([t_act[0].item()], device=device), torch.tensor([z_act.item()], device=device)

def reset_graph(worker_env: WorkerEnv):
    """[Fairness Fix] 그래프를 완전히 초기 상태로 복원."""
    worker_env.dm.apply_disaster_damage(damage_prob=0.0)
    worker_env._update_zone_graph_weights()
    worker_env._update_dist_matrix()

def run_evaluation_episode(manager, worker, hrl_env, num_targets, device, mode="benchmark", visualizer=None, frame_dir=None, frame_idx_ref=None, episode_idx=0):
    from src.models.heuristics import GA_Manager, ALNS_Manager
    
    reset_graph(hrl_env.env)
    hrl_env.reset(batch_size=1, num_targets=num_targets)
    
    if mode in ['analyze', 'visualize']:
        print(f"\n==================================================")
        print(f"🚀 Episode {episode_idx+1} Start")
        print(f"==================================================\n")
        
    done = False
    start_time = time.time()
    recompute_count = 0
    rescued = 0
    failed = False
    
    is_neural_manager = not isinstance(manager, (GA_Manager, ALNS_Manager))
    step_idx = 0
    
    if frame_idx_ref is None:
        frame_idx_ref = [0]
        
    locked_target_idx = -1
    
    while not done:
        c_node = hrl_env.env.idx_to_node[int(hrl_env.env.curr_nodes[0])]
        c_zone = hrl_env.env.n2z[c_node]
        
        valid_mask = hrl_env.get_target_mask()[0]
        if locked_target_idx != -1 and valid_mask[locked_target_idx] == 0:
            locked_target_idx = -1
            
        if mode == 'analyze':
            print(f"\n--- Manager Turn {step_idx} ---")
            
        if is_neural_manager:
            t_act_tensor, z_act_tensor = get_manager_action(manager, hrl_env, device, forced_t_act=locked_target_idx if locked_target_idx != -1 else None)
            locked_target_idx = t_act_tensor[0].item()
            recompute_count += 1
        else:
            manager.env = hrl_env
            t_act, z_act = manager.get_action()
            t_act_tensor = t_act.clone().detach().to(device)
            z_act_tensor = z_act.clone().detach().to(device)
            recompute_count += 1
            
        t_idx = t_act_tensor[0].item()
        t_node = hrl_env.env.idx_to_node[int(hrl_env.targets[0, t_idx])]
        t_zone = hrl_env.env.n2z[t_node]
        
        if mode == 'analyze':
            print(f"📍 Current: Node {c_node} (Zone {c_zone})  🎯 Target: Node {t_node} (Zone {t_zone})")
            
        subgoal_zone = z_act_tensor.item()
        
        if mode == 'analyze':
            adj_mask = hrl_env.env._zone_adj_matrix_tensor[c_zone]
            adj_zones = torch.nonzero(adj_mask).squeeze(-1).tolist()
            if subgoal_zone == c_zone:
                val_str = "🟡 SAME ZONE"
            elif subgoal_zone in adj_zones:
                val_str = "🟢 ADJACENT ZONE"
            else:
                val_str = "🔴 INVALID/FARAWAY ZONE"
            print(f"🤖 Manager Action: Go to Zone {subgoal_zone} ({val_str})")
            
        if mode == 'visualize' and visualizer and frame_dir and frame_idx_ref is not None:
            if frame_idx_ref[0] == 0:
                if not hasattr(hrl_env, 'worker_path_log'):
                    hrl_env.worker_path_log = [c_node]
                visualizer.plot_state(
                    hrl_env, step_idx=frame_idx_ref[0], save_dir=frame_dir, 
                    trajectory=hrl_env.worker_path_log, mission_zone=subgoal_zone, 
                    global_time=hrl_env.current_time[0].item()
                )
                frame_idx_ref[0] += 1
                
            events, dones = hrl_env.step_manager(
                t_act_tensor, z_act_tensor, 
                visualizer=visualizer, save_dir=frame_dir, frame_idx_ref=frame_idx_ref
            )
        else:
            events, dones = hrl_env.step_manager(t_act_tensor, z_act_tensor)
        
        if mode == 'analyze':
            worker_steps = hrl_env.worker_steps[0].item()
            event = events[0]
            print(f"🏃 Worker Result: Event={event}, Steps taken={worker_steps}")
            print(f"   Worker Path: {hrl_env.worker_path_log if hasattr(hrl_env, 'worker_path_log') else 'Not tracked'}")
            
            hrl_env.worker_steps[0] = 0
            if hasattr(hrl_env, 'worker_path_log'):
                hrl_env.worker_path_log = []
                
        if 'agent_destroyed' in events:
            failed = True
            
        step_idx += 1
        if dones[0]:
            rescued = hrl_env.num_rescued[0].item()
            done = True
            if mode == 'analyze':
                print("\n🏁 Episode Finished!")
                print(f"Total Rescued: {rescued}")
            elif mode == 'visualize':
                print(f"🏁 Episode Finished! Total Rescued: {rescued}/{num_targets}")
            
    latency = time.time() - start_time
    total_dist = hrl_env.env.total_dist[0].item()
    return rescued, latency, recompute_count, failed, total_dist
