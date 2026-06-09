import os
import json
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from collections import defaultdict
import argparse

import math
from src.envs.manager_env import ManagerEnv
from src.models.manager import Manager
from src.models.worker import Worker
from train_rl import _load_worker_checkpoint

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

def visualize_map(map_name, worker, manager, device, save_dir=None, save_prefix="", num_episodes=100, manager_max_turns=50, worker_c_max=20):
    """
    지정된 맵에서 모델 성능을 평가하고 첫 에피소드의 HRL 경로를 시각화.
    """
    config = MAP_CONFIGS[map_name]
    node_file = config['node_file']
    net_file = config['net_file']
    zone_json = config['zone_json']
    zone_graph_json = config['zone_graph_json']
    

    env = ManagerEnv(
        node_file=node_file,
        net_file=net_file,
        worker=worker,
        c_max=100, # Initial placeholder, will update dynamically below
        goal_bonus=10.0,
        step_penalty_scale=0.1,


        zone_json=zone_json,
        zone_graph_json=zone_graph_json,
        device=device
    )

    # Calculate dynamic turn limits based on map size
    avg_nodes_per_zone = env.num_nodes / env.k_zones
    worker_c_max = max(50, int(avg_nodes_per_zone * 4)) # Allow plenty of steps inside a zone
    manager_max_turns = max(50, env.k_zones * 2)
    env.c_max = worker_c_max # Update env limit

    # Load Coordinates
    pos = load_coordinates(node_file)
    if not pos:
        # Fallback to spring layout if coordinates missing
        pos = nx.spring_layout(env.G)

    # 3. Run evaluation for N episodes
    success_count = 0
    efficiency_sum = 0.0
    total_episodes = 100 # Run 100 episodes sequentially
    
    first_success_path = None
    first_success_zones = None
    first_start_node = None
    first_goal_node = None
    
    for ep in range(total_episodes):
        while True:
            curr_idx, goal_idx = env.reset()
            # Ensure the start and goal are far enough to require multiple zones
            if env.hop_matrix[curr_idx, goal_idx] > 10:
                break
        
        start_node = env.nodes[curr_idx]
        goal_node = env.nodes[goal_idx]
        
        path_nodes = [start_node]
        manager_zones = []
        visited_zones = set()
        manager_turns = 0

        done = False
        success = False
        while not done:
            manager_turns += 1
            if manager_turns > env.max_manager_turns:
                print(f"Failed: Max manager turns ({env.max_manager_turns}) exceeded.")
                break
                
            x = env.get_manager_state()
            mask = env.get_candidate_mask()
            curr_z = int(env._node_zone_tensor[env.current_idx].item())
            goal_z = int(env._node_zone_tensor[env.goal_idx].item())
            
            if mask[goal_z] == 1.0:
                action = goal_z
                # 목적지 구역에 진입했으므로 워커에게 충분한 스텝을 주어 노드를 끈질기게 찾게 함 (매니저 턴 소모 중지 효과)
                current_worker_c_max = 500
            else:
                action, _, _, _ = manager.select_action(x, env.zone_edge_index, curr_z, goal_z, mask, deterministic=True)
                current_worker_c_max = worker_c_max
                
            manager_zones.append(action)
            
            if action in visited_zones and action != curr_z:
                # Cyclic path 로직 제거 (허용)
                pass
            visited_zones.add(action)
            env.visited_zones.add(action)  # 핵심 버그 수정: env 내부 상태 업데이트 누락 수정
            
            subgoal_z = action
            zone_nodes = (env._node_zone_tensor == subgoal_z).nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            if len(zone_nodes) == 0:
                print(f"Warning: Manager selected empty zone {subgoal_z}")
                break
                
            w_done = False
            turns = 0
            start_z_for_worker = int(env._node_zone_tensor[env.current_idx].item())
            
            while not w_done and turns < current_worker_c_max:
                wx = env._get_worker_state(subgoal_z)
                w_mask = env._get_worker_action_mask(subgoal_z)
                w_batch = torch.zeros(wx.size(0), dtype=torch.long, device=device)
                
                with torch.no_grad():
                    probs, value, _ = worker(wx, env.edge_index, w_batch, neighbors_mask=w_mask, edge_attr=env.edge_attr)
                dist = torch.distributions.Categorical(probs)
                next_idx = dist.sample().item()
                
                path_nodes.append(env.nodes[next_idx])
                env.current_idx = next_idx
                turns += 1
                
                next_z = int(env._node_zone_tensor[next_idx].item())
                
                if env.current_idx == env.goal_idx:
                    w_done = True
                elif next_z == subgoal_z and subgoal_z != goal_z:
                    w_done = True
                elif next_z != start_z_for_worker and next_z != subgoal_z and next_z != goal_z:
                    w_done = True
            
            if env.current_idx == env.goal_idx:
                done = True
                success = True
                break
                
            # 목적지 Zone에 도착했더라도 최종 목적지 Node(env.goal_idx)에 도달할 때까지
            # 매니저에게 계속 턴을 주어 워커가 노드를 찾도록 기회를 줌.

            env.manager_turns += 1
            if env.manager_turns >= manager_max_turns:
                done = True

        if success:
            success_count += 1
            
            # Calculate path efficiency
            try:
                opt_path = nx.shortest_path(env.G, start_node, goal_node)
                opt_len = len(opt_path) - 1
                agent_len = len(path_nodes) - 1
                if agent_len > 0:
                    efficiency_sum += (opt_len / agent_len)
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
    avg_efficiency = (efficiency_sum / success_count * 100) if success_count > 0 else 0.0
    print(f"[{config['name']}] Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%) | Avg Path Efficiency: {avg_efficiency:.1f}%")

    if first_success_path is None:
        print("Failed to find any successful path for visualization.")
        return

    path_nodes = first_success_path
    manager_zones = first_success_zones
    start_node = first_start_node
    goal_node = first_goal_node

    # Calculate Optimal Path (A* / shortest path)
    try:
        opt_path = nx.shortest_path(env.G, start_node, goal_node)
        opt_edges = [(opt_path[i], opt_path[i+1]) for i in range(len(opt_path)-1)]
    except nx.NetworkXNoPath:
        opt_edges = []

    # 4. Plotting
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # 4.1 Base Graph (Grey edges)
    nx.draw_networkx_edges(env.G, pos, ax=ax, edge_color='lightgrey', alpha=0.5, width=1.0)
    nx.draw_networkx_nodes(env.G, pos, ax=ax, node_color='lightgrey', node_size=10, alpha=0.5)

    # 4.2 Manager Zones (Light Blue Grid Areas)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin_x = (max(all_x) - min(all_x)) * 0.02
    margin_y = (max(all_y) - min(all_y)) * 0.02

    # Reconstruct Grid Geometry
    N = math.ceil(math.sqrt(len(pos) / 16.0))
    N = max(2, N)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    eps_x = (max_x - min_x) * 1e-6
    eps_y = (max_y - min_y) * 1e-6
    max_x += eps_x
    max_y += eps_y
    
    dx = (max_x - min_x) / N
    dy = (max_y - min_y) / N

    # Map each zone to a list of (gx, gy) grid cells
    zone_to_cells = defaultdict(list)
    for n in env.nodes:
        x, y = pos[n]
        gx = int((x - min_x) / dx)
        gy = int((y - min_y) / dy)
        gx = min(max(gx, 0), N-1)
        gy = min(max(gy, 0), N-1)
        z = env._node_zone_tensor[env.node_to_idx[n]].item()
        zone_to_cells[z].append((gx, gy))

    # Helper function to draw a grid zone
    def draw_grid_zone(z_idx, color, label=None, draw_text=False, step_idx=None):
        if z_idx not in zone_to_cells: return None
        cells = set(zone_to_cells[z_idx])
        
        # Calculate center for text
        sum_cx, sum_cy = 0, 0
        for gx, gy in cells:
            cx, cy = min_x + gx * dx, min_y + gy * dy
            rect = patches.Rectangle((cx, cy), dx, dy, linewidth=0, facecolor=color, alpha=0.5, zorder=0, label=label if label and (gx, gy) == list(cells)[0] else None)
            ax.add_patch(rect)
            sum_cx += cx + dx/2
            sum_cy += cy + dy/2
            
        center_pt = (sum_cx / len(cells), sum_cy / len(cells))
        if draw_text and step_idx is not None:
            ax.text(center_pt[0], center_pt[1], str(step_idx), color='black', fontsize=14, weight='bold', 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        return center_pt

    # Draw Start and Goal Zones
    start_z = int(env._node_zone_tensor[env.node_to_idx[start_node]].item())
    goal_z = int(env._node_zone_tensor[env.node_to_idx[goal_node]].item())
    draw_grid_zone(start_z, 'lightgreen', label='Start Zone')
    draw_grid_zone(goal_z, 'lightcoral', label='Goal Zone')

    drawn_zones = {start_z: True, goal_z: True}
    zone_centers = []
    
    # Manager Zones Sequence
    for step_idx, z in enumerate(manager_zones):
        if z not in drawn_zones:
            drawn_zones[z] = True
            c_pt = draw_grid_zone(z, 'lightblue', draw_text=True, step_idx=step_idx+1)
            if c_pt: zone_centers.append(c_pt)
        else:
            if z in zone_to_cells:
                cells = set(zone_to_cells[z])
                sum_cx = sum([min_x + gx * dx + dx/2 for gx, gy in cells])
                sum_cy = sum([min_y + gy * dy + dy/2 for gx, gy in cells])
                c_pt = (sum_cx / len(cells), sum_cy / len(cells))
                zone_centers.append(c_pt)
                ax.text(c_pt[0], c_pt[1], str(step_idx + 1), color='black', fontsize=14, weight='bold', 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Draw arrows between zone centers to show manager sequence
    # Removed as per user request

    # 4.3 Path (Dark Orange Line for HRL, Dashed Blue for Optimal)
    # (Optimal path will be drawn after Worker path to be on top)
    
    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    # Restrict to bounding box of path to make it visible
    path_coords = np.array([pos[n] for n in path_nodes if n in pos])
    if len(path_coords) > 0:
        min_x, max_x = path_coords[:,0].min(), path_coords[:,0].max()
        min_y, max_y = path_coords[:,1].min(), path_coords[:,1].max()
        margin_x = (max_x - min_x) * 0.5 + 0.01
        margin_y = (max_y - min_y) * 0.5 + 0.01
        
        # Only draw nodes within this extended bounding box
        visible_nodes = [n for n in env.G.nodes() if n in pos and 
                        min_x - margin_x <= pos[n][0] <= max_x + margin_x and 
                        min_y - margin_y <= pos[n][1] <= max_y + margin_y]
    else:
        visible_nodes = list(env.G.nodes())

    # Draw visible edges
    visible_edges = [(u, v) for u, v in env.G.edges() if u in visible_nodes and v in visible_nodes]
    nx.draw_networkx_edges(env.G, pos, edgelist=visible_edges, ax=ax, edge_color='gray', alpha=0.3)
    
    # Draw visible nodes
    nx.draw_networkx_nodes(env.G, pos, nodelist=visible_nodes, ax=ax, node_color='lightgray', node_size=10, alpha=0.5)

    # Draw the path
    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    # Set node_size=0 so there are no gaps for arrow heads!
    nx.draw_networkx_edges(env.G, pos, edgelist=path_edges, ax=ax, edge_color='black', style='solid', width=2.5, arrows=True, arrowsize=15, node_size=0)
    
    # Draw A* Optimal Path as a dashed red line on top
    nx.draw_networkx_edges(env.G, pos, edgelist=opt_edges, ax=ax, edge_color='red', style='dashed', width=2.0, alpha=0.7)

    if start_node in pos and goal_node in pos:
        # Draw start and goal nodes
        nx.draw_networkx_nodes(env.G, pos, nodelist=[start_node], ax=ax, node_color='green', node_size=150)
        nx.draw_networkx_nodes(env.G, pos, nodelist=[goal_node], ax=ax, node_color='red', node_size=150)

    plt.tight_layout()
    # 시각화 저장
    if save_dir is None:
        save_dir = 'tests'
    os.makedirs(save_dir, exist_ok=True)
    out_filename = f'{save_prefix}hrl_path_{map_name}.png'
    out_path = os.path.join(save_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to {out_path}")
    plt.close()

    # Save Legend separately if not already saved
    legend_path = os.path.join(save_dir, "legend.png")
    if not os.path.exists(legend_path):
        fig_leg = plt.figure(figsize=(10, 1.5))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis('off')
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
            Line2D([0], [0], marker='o', color='w', label='Goal Node', markerfacecolor='red', markersize=12),
            Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Worker Path'),
            Line2D([0], [0], color='red', lw=2.0, linestyle='--', label='A* Optimal Path'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', alpha=0.5, markersize=10, label='Manager Subgoal Zones')
        ]
        ax_leg.legend(handles=custom_lines, loc='center', ncol=5, frameon=False, fontsize=12)
        fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')
        plt.close(fig_leg)
        print(f"Legend saved to {legend_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, choices=['anaheim', 'chicago', 'berlin-mitte', 'berlin-friedrichshain', 'goldcoast', 'all'], default='all')

    parser.add_argument('--worker_ckpt', type=str, default=None, help="Path to worker checkpoint")
    parser.add_argument('--manager_ckpt', type=str, default=None, help="Path to manager checkpoint")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the visualization")
    parser.add_argument('--save_prefix', type=str, default="", help="Prefix for the saved file name")
    parser.add_argument("--use_is_visited", action="store_true", help="Worker에 방문 노드 이력 상태 채널 추가")
    parser.add_argument("--use_global_pool", action="store_true", help="Worker Critic에 Global Mean Pooling 추가")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_latest_ckpt(dir_path):
        import glob
        dirs = sorted(glob.glob(f"{dir_path}/*"))
        if dirs:
            return f"{dirs[-1]}/best.pt"
        return None
        
    w_ckpt = args.worker_ckpt if args.worker_ckpt else get_latest_ckpt('logs/rl_worker_stage')
    
    use_is_visited = getattr(args, 'use_is_visited', False)
    use_global_pool = getattr(args, 'use_global_pool', False)
    node_dim = 5 if use_is_visited else 4
    worker = Worker(node_dim=node_dim, hidden_dim=256, num_layers=2, use_is_visited=use_is_visited, use_global_pool=use_global_pool).to(device)
    worker.load_state_dict(torch.load(w_ckpt, map_location=device, weights_only=True))
    worker.eval()
    print(f"📦 Loaded worker checkpoint from {w_ckpt}")
    
    manager = Manager(node_dim=4, hidden_dim=256, num_layers=2).to(device)
    m_ckpt = args.manager_ckpt if args.manager_ckpt else get_latest_ckpt('logs/rl_manager_stage')
    manager.load_state_dict(torch.load(m_ckpt, map_location=device, weights_only=True))
    manager.eval()
    print(f"📦 Loaded manager checkpoint from {m_ckpt}")
    
    maps_to_run = list(MAP_CONFIGS.keys()) if args.map == 'all' else [args.map]
    for m in maps_to_run:
        visualize_map(m, worker, manager, device, args.save_dir, args.save_prefix)

if __name__ == '__main__':
    main()
