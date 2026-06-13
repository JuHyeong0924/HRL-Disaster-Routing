import os
import sys
import json
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from collections import defaultdict
import argparse
import glob
import math

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트 허용
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.manager_env import ManagerEnv
from src.envs.worker_env import WorkerEnv
from src.models.manager import Manager
from src.models.worker import Worker
from scripts.train_rl import _load_worker_checkpoint

MAP_CONFIGS = {
    'anaheim': {
        'node_file': 'data/Anaheim_node.tntp',
        'net_file': 'data/Anaheim_net.tntp',
        'zone_json': 'data/grid_Anaheim_node_to_zone.json',
        'zone_graph_json': 'data/grid_Anaheim_zone_graph.json',
        'name': 'Anaheim'
    },
    'chicago': {
        'node_file': 'data/ChicagoSketch_node.tntp',
        'net_file': 'data/ChicagoSketch_net.tntp',
        'zone_json': 'data/grid_ChicagoSketch_node_to_zone.json',
        'zone_graph_json': 'data/grid_ChicagoSketch_zone_graph.json',
        'name': 'ChicagoSketch'
    },
    'berlin-mitte': {
        'node_file': 'data/Berlin-Mitte_node.tntp',
        'net_file': 'data/Berlin-Mitte_net.tntp',
        'zone_json': 'data/grid_Berlin-Mitte_node_to_zone.json',
        'zone_graph_json': 'data/grid_Berlin-Mitte_zone_graph.json',
        'name': 'Berlin-Mitte'
    },
    'berlin-friedrichshain': {
        'node_file': 'data/Berlin-Friedrichshain_node.tntp',
        'net_file': 'data/Berlin-Friedrichshain_net.tntp',
        'zone_json': 'data/grid_Berlin-Friedrichshain_node_to_zone.json',
        'zone_graph_json': 'data/grid_Berlin-Friedrichshain_zone_graph.json',
        'name': 'Berlin-Friedrichshain'
    },
    'goldcoast': {
        'node_file': 'data/Goldcoast_node.tntp',
        'net_file': 'data/Goldcoast_net.tntp',
        'zone_json': 'data/grid_Goldcoast_node_to_zone.json',
        'zone_graph_json': 'data/grid_Goldcoast_zone_graph.json',
        'name': 'Goldcoast'
    }
}

def load_coordinates(node_file: str):
    coords = {}
    with open(node_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('~') or line.startswith('Node') or line.startswith('<'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node_id] = (x, y)
                except ValueError:
                    pass
    return coords


def evaluate_crossmap(env: WorkerEnv, worker: Worker, device: torch.device, num_episodes: int = 200, batch_size: int = 16) -> dict:
    """Worker 모델의 Zero-shot 성능 평가 (Phase 1만 평가)"""
    worker.eval()
    
    edge_list = []
    for u, v in env.G.edges():
        edge_list.append((env.node_to_idx[u], env.node_to_idx[v]))
    edge_list_bidir = edge_list + [(v, u) for u, v in edge_list]
    edge_index = torch.tensor(edge_list_bidir, dtype=torch.long).t().to(device)
    
    static_edge_attr = None
    if getattr(worker, 'use_edge_attr', False) or getattr(worker, 'config', {}).get('use_edge_attr', False) or True:
        curr_edge_attr = []
        for u_idx, v_idx in edge_list_bidir:
            u = env.idx_to_node[u_idx]
            v = env.idx_to_node[v_idx]
            data = env.dm.graph[u][v]
            length = data.get('length', 0.0)
            capacity = data.get('capacity', 0.0)
            speed = data.get('speed', 0.0)
            curr_edge_attr.append([length, capacity, speed])
        static_edge_attr = torch.tensor(curr_edge_attr, dtype=torch.float).to(device)
        
        if static_edge_attr.size(0) > 0:
            feat_min = static_edge_attr.min(dim=0, keepdim=True)[0]
            feat_max = static_edge_attr.max(dim=0, keepdim=True)[0]
            scale = (feat_max - feat_min).clamp(min=1e-8)
            static_edge_attr = (static_edge_attr - feat_min) / scale
    
    N = env.num_nodes
    all_results = []
    num_batches = (num_episodes + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            actual_batch = min(batch_size, num_episodes - batch_idx * batch_size)
            state = env.reset(batch_size=actual_batch)
            B = actual_batch
            
            batch_rewards = [0.0] * B
            done_flags = [False] * B
            final_infos = [{} for _ in range(B)]
            
            while not all(done_flags):
                active = [b for b in range(B) if not done_flags[b]]
                active_states = torch.stack([state[b].to(device) for b in active])
                active_masks = torch.stack([env.get_action_mask_batch()[b].to(device) for b in active])
                
                A = len(active)
                x_flat = active_states.view(-1, active_states.shape[-1])
                mask_flat = active_masks.view(-1)
                ai = torch.arange(A, device=device).repeat_interleave(N)
                aei = torch.cat([edge_index + i * N for i in range(A)], dim=1)
                
                edge_attr_flat = None
                if static_edge_attr is not None:
                    edge_attr_flat = static_edge_attr.repeat(A, 1)
                
                probs_all, _, _ = worker(x_flat, aei, edge_attr=edge_attr_flat, batch=ai, neighbors_mask=mask_flat)
                
                actions = []
                for i, b in enumerate(active):
                    node_probs = probs_all[i * N: (i + 1) * N]
                    actions.append(node_probs.argmax().item())
                
                all_actions = []
                ai_ptr = 0
                for b in range(B):
                    if not done_flags[b]:
                        all_actions.append(actions[ai_ptr])
                        ai_ptr += 1
                    else:
                        all_actions.append(0)
                
                next_state, reward_t, done_t, infos = env.step_batch(torch.tensor(all_actions))
                
                for b in range(B):
                    if not done_flags[b]:
                        batch_rewards[b] += reward_t[b].item()
                        if done_t[b].item():
                            done_flags[b] = True
                            final_infos[b] = infos[b]
                state = next_state
            
            for b in range(B):
                all_results.append({
                    'success': 1.0 if final_infos[b].get('reason') == 'success' else 0.0,
                    'reward': batch_rewards[b],
                    'path_len': final_infos[b].get('path_len', 200),
                    'reason': final_infos[b].get('reason', 'unknown'),
                })
    
    success_rate = np.mean([r['success'] for r in all_results])
    avg_reward = np.mean([r['reward'] for r in all_results])
    avg_path_len = np.mean([r['path_len'] for r in all_results])
    success_lens = [r['path_len'] for r in all_results if r['success'] > 0]
    avg_success_path = np.mean(success_lens) if success_lens else 0.0
    
    reason_counts = defaultdict(int)
    for r in all_results:
        reason_counts[r['reason']] += 1
    
    metrics = {
        'success_rate': float(success_rate),
        'avg_reward': float(avg_reward),
        'avg_path_len': float(avg_path_len),
        'avg_success_path': float(avg_success_path),
        'num_episodes': len(all_results),
        'reason_distribution': dict(reason_counts),
    }
    return metrics


def visualize_map(map_name, worker, manager, device, save_dir=None, save_prefix="", num_episodes=100, manager_max_turns=50, worker_c_max=20):
    """지정된 맵에서 모델 성능을 평가하고 첫 에피소드의 HRL 경로를 시각화 (Closed-Loop Manager)"""
    config = MAP_CONFIGS[map_name]
    env = ManagerEnv(
        node_file=config['node_file'],
        net_file=config['net_file'],
        worker=worker,
        c_max=100,
        zone_json=config['zone_json'],
        zone_graph_json=config['zone_graph_json'],
        device=device
    )

    avg_nodes_per_zone = env.num_nodes / env.k_zones
    worker_c_max = max(50, int(avg_nodes_per_zone * 4))
    manager_max_turns = max(50, env.k_zones * 2)
    env.c_max = worker_c_max

    pos = load_coordinates(config['node_file'])
    if not pos:
        pos = nx.spring_layout(env.G)

    success_count = 0
    expansion_sum = 0.0
    total_episodes = num_episodes
    
    first_success_path = None
    first_success_zones = None
    first_start_node = None
    first_goal_node = None
    
    for ep in range(total_episodes):
        while True:
            curr_idx, goal_idx = env.reset()
            if env.hop_matrix[curr_idx, goal_idx] > 10:
                break
        
        start_node = env.nodes[curr_idx]
        goal_node = env.nodes[goal_idx]
        
        path_nodes = [start_node]
        manager_zones = []
        visited_zones = set()
        max_manager_turns = 100 

        done = False
        success = False
        while not done:
            if len(manager_zones) > max_manager_turns:
                print(f"Failed: Max manager turns ({max_manager_turns}) exceeded.", flush=True)
                break
                
            x = env.get_manager_state()
            mask = env.get_candidate_mask()
            curr_z = int(env._node_zone_tensor[env.current_idx].item())
            goal_z = int(env._node_zone_tensor[env.goal_idx].item())
            
            if mask[goal_z] == 1.0:
                action = goal_z
                current_worker_c_max = 500
            else:
                action, _, _, _ = manager.select_action(x, env.zone_edge_index, curr_z, goal_z, mask, deterministic=True)
                current_worker_c_max = worker_c_max
                
            manager_zones.append(action)
            visited_zones.add(action)
            env.visited_zones.add(action)
            
            subgoal_z = action
            zone_nodes = (env._node_zone_tensor == subgoal_z).nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            if len(zone_nodes) == 0:
                print(f"Warning: Manager selected empty zone {subgoal_z}")
                break
                
            w_done = False
            turns = 0
            
            while not w_done and turns < current_worker_c_max:
                wx = env._get_worker_state(subgoal_z)
                w_mask = env._get_worker_action_mask(subgoal_z)
                w_batch = torch.zeros(wx.size(0), dtype=torch.long, device=device)
                
                with torch.no_grad():
                    probs, value, _ = worker(wx, env.edge_index, w_batch, neighbors_mask=w_mask, edge_attr=env.edge_attr)
                
                for past_node in path_nodes[-5:]:
                    if past_node in env.node_to_idx:
                        probs[env.node_to_idx[past_node]] *= 0.001
                        
                next_idx = torch.argmax(probs).item()
                
                curr_z_eval = int(env._node_zone_tensor[env.current_idx].item())
                next_z_eval = int(env._node_zone_tensor[next_idx].item())
                
                env.current_idx = next_idx
                path_nodes.append(env.nodes[next_idx])
                turns += 1
                
                if env.current_idx == env.goal_idx:
                    w_done = True
                elif next_z_eval != curr_z_eval:
                    w_done = True
            
            if env.current_idx == env.goal_idx:
                done = True
                success = True
                break
                
            env.manager_turns += 1
            if env.manager_turns >= manager_max_turns:
                done = True

        if success:
            success_count += 1
            try:
                opt_path = nx.shortest_path(env.G, start_node, goal_node)
                opt_len = len(opt_path) - 1
                agent_len = len(path_nodes) - 1
                if opt_len > 0:
                    expansion_sum += (agent_len / opt_len)
                elif agent_len == 0:
                    expansion_sum += 1.0
                print(f"[{config['name']}] Ep {ep+1}/{total_episodes}: Success (Opt: {opt_len}, Agent: {agent_len})", flush=True)
            except nx.NetworkXNoPath:
                pass 
                
            if first_success_path is None:
                first_success_path = path_nodes
                first_success_zones = manager_zones
                first_start_node = start_node
                first_goal_node = goal_node
        
        if not success:
            print(f"[{config['name']}] Ep {ep+1}/{total_episodes}: Failed (Manager Zones: {manager_zones})", flush=True)

    success_rate = (success_count / total_episodes) * 100
    avg_expansion = (expansion_sum / success_count) if success_count > 0 else 0.0
    print(f"[{config['name']}] Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%) | Avg Path Expansion: {avg_expansion:.2f}배")

    if first_success_path is None:
        print("Failed to find any successful path for visualization.")
        return

    path_nodes = first_success_path
    manager_zones = first_success_zones
    start_node = first_start_node
    goal_node = first_goal_node

    try:
        opt_path = nx.shortest_path(env.G, start_node, goal_node)
        opt_edges = [(opt_path[i], opt_path[i+1]) for i in range(len(opt_path)-1)]
    except nx.NetworkXNoPath:
        opt_edges = []

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    nx.draw_networkx_edges(env.G, pos, ax=ax, edge_color='lightgrey', alpha=0.5, width=1.0)
    nx.draw_networkx_nodes(env.G, pos, ax=ax, node_color='lightgrey', node_size=10, alpha=0.5)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    N_grid = math.ceil(math.sqrt(len(pos) / 16.0))
    N_grid = max(2, N_grid)
    eps_x = (max_x - min_x) * 1e-6
    eps_y = (max_y - min_y) * 1e-6
    max_x += eps_x
    max_y += eps_y
    dx = (max_x - min_x) / N_grid
    dy = (max_y - min_y) / N_grid

    zone_to_cells = defaultdict(list)
    for n in env.nodes:
        x, y = pos[n]
        gx = int((x - min_x) / dx)
        gy = int((y - min_y) / dy)
        gx = min(max(gx, 0), N_grid-1)
        gy = min(max(gy, 0), N_grid-1)
        z = env._node_zone_tensor[env.node_to_idx[n]].item()
        zone_to_cells[z].append((gx, gy))

    def draw_grid_zone(z_idx, color, label=None, draw_text=False, step_idx=None):
        if z_idx not in zone_to_cells: return None
        cells = set(zone_to_cells[z_idx])
        sum_cx, sum_cy = 0, 0
        for gx, gy in cells:
            cx, cy = min_x + gx * dx, min_y + gy * dy
            rect = patches.Rectangle((cx, cy), dx, dy, linewidth=0, facecolor=color, alpha=0.5, zorder=0)
            ax.add_patch(rect)
            sum_cx += cx + dx/2
            sum_cy += cy + dy/2
            
        center_pt = (sum_cx / len(cells), sum_cy / len(cells))
        if draw_text and step_idx is not None:
            ax.text(center_pt[0], center_pt[1], str(step_idx), color='black', fontsize=14, weight='bold', 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        return center_pt

    start_z = int(env._node_zone_tensor[env.node_to_idx[start_node]].item())
    goal_z = int(env._node_zone_tensor[env.node_to_idx[goal_node]].item())
    draw_grid_zone(start_z, 'lightgreen', label='Start Zone')
    draw_grid_zone(goal_z, 'lightcoral', label='Goal Zone')

    drawn_zones = {start_z: True, goal_z: True}
    draw_counter = 1
    for z in manager_zones:
        if z not in drawn_zones:
            draw_grid_zone(z, 'lightblue', draw_text=True, step_idx=draw_counter)
            drawn_zones[z] = True
            draw_counter += 1

    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    nx.draw_networkx_edges(env.G, pos, edgelist=path_edges, ax=ax, edge_color='black', style='solid', width=2.5, arrows=True, arrowsize=15, node_size=0)
    nx.draw_networkx_edges(env.G, pos, edgelist=opt_edges, ax=ax, edge_color='red', style='dashed', width=2.0, alpha=0.7)

    if start_node in pos and goal_node in pos:
        nx.draw_networkx_nodes(env.G, pos, nodelist=[start_node], ax=ax, node_color='green', node_size=150)
        nx.draw_networkx_nodes(env.G, pos, nodelist=[goal_node], ax=ax, node_color='red', node_size=150)

    plt.tight_layout()
    if save_dir is None:
        save_dir = 'tests'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{save_prefix}hrl_path_{map_name}.png')
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, choices=['anaheim', 'chicago', 'berlin-mitte', 'berlin-friedrichshain', 'goldcoast', 'all'], default='all')
    parser.add_argument('--cross_map', action='store_true', help="Cross-Map Zero-shot (Worker-only) 평가 모드")
    
    parser.add_argument('--worker_ckpt', type=str, default=None, help="Path to worker checkpoint")
    parser.add_argument('--manager_ckpt', type=str, default=None, help="Path to manager checkpoint (Closed-loop)")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the visualization")
    parser.add_argument('--save_prefix', type=str, default="", help="Prefix for the saved file name")
    
    parser.add_argument('--num_episodes', type=int, default=100, help='평가 에피소드 수 (Cross-Map은 기본 200 등)')
    parser.add_argument('--batch_size', type=int, default=16, help='Cross-Map 평가용 배치 크기')
    parser.add_argument("--use_is_visited", action="store_true", help="Worker 방문 이력 노드 상태 채널")
    parser.add_argument("--use_global_pool", action="store_true")
    
    # Cross-Map 용 HRL 환경 설정
    parser.add_argument('--masking_mode', type=str, default='soft_curr_next')
    parser.add_argument('--use_pbrs', action='store_true')
    parser.add_argument('--subgoal_mode', type=str, default='zone')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--use_jk_net', action='store_true')
    parser.add_argument('--use_edge_attr', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_latest_ckpt(dir_path):
        dirs = sorted(glob.glob(f"{dir_path}/*"))
        if dirs:
            return f"{dirs[-1]}/best.pt"
        return None
        
    w_ckpt = args.worker_ckpt if args.worker_ckpt else get_latest_ckpt('logs/rl_worker_stage')
    
    use_is_visited = getattr(args, 'use_is_visited', False)
    use_global_pool = getattr(args, 'use_global_pool', False)
    node_dim = 5 if use_is_visited else 4
    worker = Worker(node_dim=node_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, use_is_visited=use_is_visited, use_global_pool=use_global_pool, use_jk_net=args.use_jk_net, use_edge_attr=args.use_edge_attr, use_checkpoint=False).to(device)
    
    payload = torch.load(w_ckpt, map_location=device, weights_only=False)
    if 'worker_state_dict' in payload:
        worker.load_state_dict(payload['worker_state_dict'], strict=False)
    elif 'worker_state' in payload:
        worker.load_state_dict(payload['worker_state'], strict=False)
    else:
        worker.load_state_dict(payload, strict=False)
    worker.eval()
    print(f"📦 Loaded worker checkpoint from {w_ckpt}")

    if not args.cross_map:
        manager = Manager(node_dim=7, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        m_ckpt = args.manager_ckpt if args.manager_ckpt else get_latest_ckpt('logs/rl_manager_stage')
        if m_ckpt and os.path.exists(m_ckpt):
            manager.load_state_dict(torch.load(m_ckpt, map_location=device, weights_only=True), strict=False)
            print(f"📦 Loaded manager checkpoint from {m_ckpt}")
        manager.eval()
        
        maps_to_run = list(MAP_CONFIGS.keys()) if args.map == 'all' else [args.map]
        for m in maps_to_run:
            visualize_map(m, worker, manager, device, args.save_dir, args.save_prefix, num_episodes=args.num_episodes)
    else:
        # Cross-Map Evaluation (Zero-shot Worker)
        maps_to_run = list(MAP_CONFIGS.keys()) if args.map == 'all' else [args.map]
        for m in maps_to_run:
            print(f"🗺️ Cross-Map 평가 진행 맵: {m}")
            # Zone 파일 자동 탐색
            zone_json = MAP_CONFIGS[m]['zone_json']
            zone_graph_json = MAP_CONFIGS[m]['zone_graph_json']
            
            env = WorkerEnv(
                f"data/{MAP_CONFIGS[m]['name']}_node.tntp",
                f"data/{MAP_CONFIGS[m]['name']}_net.tntp",
                zone_json=zone_json,
                zone_graph_json=zone_graph_json,
                masking_mode=args.masking_mode,
                use_pbrs=args.use_pbrs,
                subgoal_mode=args.subgoal_mode,
            )
            
            metrics = evaluate_crossmap(env, worker, device, num_episodes=args.num_episodes, batch_size=args.batch_size)
            print(f"\n📊 {m} Zero-shot 평가 결과")
            print(f"  Success Rate:    {metrics['success_rate']*100:.1f}%")
            print(f"  Avg Reward:      {metrics['avg_reward']:.1f}")
            print(f"  Avg Path Len:    {metrics['avg_path_len']:.1f}")
            
            result_path = f"logs/eval_zeroshot_{m}_{args.subgoal_mode}_results.json"
            os.makedirs('logs', exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump({'map': m, **metrics}, f, indent=2)
            print(f"  ✅ 결과 저장: {result_path}")

if __name__ == '__main__':
    main()
