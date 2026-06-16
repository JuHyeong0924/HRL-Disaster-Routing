import os
import sys
import torch

sys.path.insert(0, '.')
from src.envs.worker_env import WorkerEnv
from src.models.worker import Worker
from src.models.manager_unified import ManagerUnified
from src.envs.hrl_env import HRLEnv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    map_name = 'Anaheim'
    worker_ckpt_path = 'logs/rl_worker_stage/2026-06-16_142234_worker/best.pt'
    manager_ckpt_path = 'logs/rl_manager_stage/2026-06-16_020215_manager_ppo/best_manager.pt'
    
    print(f"Loading WorkerEnv...")
    worker_env = WorkerEnv(
        f'data/{map_name}_node.tntp', f'data/{map_name}_net.tntp',
        zone_json=f'data/grid_{map_name}_node_to_zone.json',
        zone_graph_json=f'data/grid_{map_name}_zone_graph.json',
        masking_mode='soft_curr_next',
        disaster_prob=0.0,
        dynamic_disaster=False,
        device=device
    )
    
    print(f"Loading Worker...")
    worker = Worker(node_dim=5, hidden_dim=256, num_layers=2, dropout=0.0).to(device)
    ckpt = torch.load(worker_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('worker_state', ckpt.get('state_dict', ckpt))
    cur = worker.state_dict()
    compat = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
    worker.load_state_dict(compat, strict=False)
    worker.eval()
    
    print(f"Loading HRLEnv...")
    hrl_env = HRLEnv(worker, worker_env)
    
    print(f"Loading Manager...")
    manager = ManagerUnified(
        zone_dim=6, target_dim=4, hidden_dim=128,
        num_gat_layers=2, gat_heads=4, num_transformer_layers=1, transformer_heads=4
    ).to(device)
    manager_ckpt = torch.load(manager_ckpt_path, map_location=device, weights_only=False)
    m_state = manager_ckpt.get('manager_state', manager_ckpt.get('state_dict', manager_ckpt.get('model_state_dict', manager_ckpt)))
    manager.load_state_dict(m_state, strict=False)
    manager.eval()
    
    # Compute zone_dist_matrix
    import networkx as nx
    import numpy as np
    z_apsp = dict(nx.all_pairs_dijkstra_path_length(hrl_env.env.ZG, weight='weight'))
    zone_dist_matrix = np.full((hrl_env.env.k, hrl_env.env.k), np.inf)
    for u, lengths in z_apsp.items():
        for v, length in lengths.items():
            zone_dist_matrix[u, v] = length
    z_dist_mat = torch.tensor(zone_dist_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    
    episodes = 5
    for ep in range(episodes):
        print(f"\n{'='*50}")
        print(f"🚀 Episode {ep+1} Start")
        print(f"{'='*50}")
        
        state = hrl_env.reset(batch_size=1)
        done = False
        step_idx = 0
        
        while not done:
            print(f"\n--- Manager Turn {step_idx} ---")
            
            # Print current state
            c_node = hrl_env.env.idx_to_node[int(hrl_env.env.curr_nodes[0])]
            c_zone = hrl_env.env.n2z[c_node]
            t_idx = hrl_env.curr_target_idx[0].item()
            t_node = hrl_env.env.idx_to_node[int(hrl_env.targets[0, t_idx])]
            t_zone = hrl_env.env.n2z[t_node]
            
            print(f"📍 Current: Node {c_node} (Zone {c_zone})  🎯 Target: Node {t_node} (Zone {t_zone})")
            
            # Manager Action
            with torch.no_grad():
                scene = hrl_env._get_scenario_dict()
                zf = scene['zone_features']
                zei = scene['zone_edge_index']
                tz = scene['target_zones']
                
                tf = hrl_env.get_target_features()
                tm = hrl_env.get_target_mask()
                zam = hrl_env.get_zone_adj_mask()
                
                h_last = torch.zeros(1, 128, device=device)
                elapsed = hrl_env.current_time.unsqueeze(1)
                rescued = hrl_env.num_rescued.unsqueeze(1).float()
                
                K_zones = hrl_env.env.k
                num_targets = hrl_env.num_targets
                
                flat_zf = zf.view(K_zones, 6)
                ai = torch.zeros(K_zones, dtype=torch.long, device=device)
                
                zone_emb = manager.encode_zones(flat_zf, zei, batch=ai)
                
                t_ai = torch.zeros(num_targets, dtype=torch.long, device=device)
                t_emb = manager.get_target_embeddings(zone_emb, tf.view(-1, 4), tz.view(-1))
                
                query = manager.generate_context(h_last, elapsed, rescued)
                
                t_logits, _, t_emb_dense = manager.get_target_logits(query, t_emb, tm.view(-1), t_ai)
                t_act = t_logits.argmax(dim=-1)
                
                z_ai = torch.zeros(K_zones, dtype=torch.long, device=device)
                selected_t_emb = t_emb_dense[0, t_act[0]]
                
                selected_tz = tz.view(-1)[t_act[0]].unsqueeze(0)
                
                z_logits, _ = manager.get_zone_logits(query, selected_t_emb.unsqueeze(0), zone_emb, zam.view(-1), z_ai, selected_tz, z_dist_mat)
                z_act = z_logits.argmax(dim=-1)
            
            subgoal_zone = z_act.item()
            
            # Validation
            adj_mask = hrl_env.env._zone_adj_matrix_tensor[c_zone]
            adj_zones = torch.nonzero(adj_mask).squeeze(-1).tolist()
            if subgoal_zone == c_zone:
                val_str = "🟡 SAME ZONE"
            elif subgoal_zone in adj_zones:
                val_str = "🟢 ADJACENT ZONE"
            else:
                val_str = "🔴 INVALID/FARAWAY ZONE"
                
            print(f"🤖 Manager Action: Go to Zone {subgoal_zone} ({val_str})")
            
            # Execute
            t_act_tensor = torch.tensor([t_act[0].item()], device=device)
            z_act_tensor = torch.tensor([z_act.item()], device=device)
            events, dones = hrl_env.step_manager(t_act_tensor, z_act_tensor)
            
            # Inspect worker execution result
            worker_steps = hrl_env.worker_steps[0].item()
            event = events[0]
            
            print(f"🏃 Worker Result: Event={event}, Steps taken={worker_steps}")
            print(f"   Worker Path: {hrl_env.worker_path_log if hasattr(hrl_env, 'worker_path_log') else 'Not tracked'}")
            
            # Reset worker step tracker for print clarity next turn
            hrl_env.worker_steps[0] = 0
            if hasattr(hrl_env, 'worker_path_log'):
                hrl_env.worker_path_log = []
            
            step_idx += 1
            if dones[0]:
                print("\n🏁 Episode Finished!")
                print(f"Total Rescued: {hrl_env.num_rescued[0].item()}")
                break

if __name__ == '__main__':
    main()
