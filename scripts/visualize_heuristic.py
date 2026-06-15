import os
import sys
import torch
import networkx as nx

sys.path.insert(0, '.')
from src.envs.worker_env import WorkerEnv
from src.envs.hrl_env import HRLEnv
from src.utils.visualizer import DisasterVisualizer

def run_heuristic_visualization(map_name):
    print(f"🚀 Starting Heuristic Visualization Test (Map: {map_name})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 환경 초기화
    try:
        worker_env = WorkerEnv(
            f'data/{map_name}_node.tntp', f'data/{map_name}_net.tntp',
            zone_json=f'data/grid_{map_name}_node_to_zone.json',
            zone_graph_json=f'data/grid_{map_name}_zone_graph.json',
            masking_mode='soft_curr_next',
            device=device
        )
    except FileNotFoundError:
        print(f"⚠️ Data for {map_name} not found. Skipping.")
        return
    
    # Dummy worker model since we are not using RL
    class DummyWorker(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return None, None, None
            
    worker = DummyWorker().to(device)
    hrl_env = HRLEnv(worker, worker_env, max_time=200, max_manager_turns=50)
    
    # 2. Visualizer 및 디렉토리 구조 초기화
    visualizer = DisasterVisualizer(worker_env.dm, worker_env.n2z)
    
    ep_name = 'ep_0'
    base_dir = os.path.join('figs', map_name, ep_name)
    png_dir = os.path.join(base_dir, 'png')
    gif_dir = os.path.join(base_dir, 'gif')
    
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    
    # 기존 스냅샷 지우기
    for f in os.listdir(png_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(png_dir, f))
            
    # 3. 단일 에피소드 시뮬레이션 시작
    batch_size = 1
    num_targets = 10
    
    hrl_env.reset(batch_size=batch_size, num_targets=num_targets)
    
    # 초기에 재난 0
    worker_env.disaster_prob = 0.0
    worker_env.apply_dynamic_disaster()
    
    b = 0
    done = False
    global_time = 0.0
    frame_idx = 0
    worker_busy_time = 0.0
    total_edge_time = 1.0
    path = []
    path_idx = 0
    c_node = hrl_env.env.idx_to_node[int(hrl_env.env.curr_nodes[b].item())]
    next_node = c_node
    target_node = -1
    best_t_idx = -1
    mission_zone = worker_env.n2z.get(c_node)
    trajectory = [c_node]
    zone_sequence = []
    last_disaster_time = 0.0
    snapshot_paths = []
    
    # 첫 스냅샷 (Step 0)
    p = visualizer.plot_state(hrl_env, frame_idx, png_dir, trajectory=trajectory, global_time=global_time)
    snapshot_paths.append(p)
    frame_idx += 1
    
    while not done and global_time < 200.0:
        hrl_env.current_time[b] = global_time
        
        # 1. 재난 발생 체크
        if global_time >= 30.0 and last_disaster_time < 30.0:
            print(f"💥 [Time {global_time:.1f}] Main Disaster Triggered!")
            worker_env.disaster_prob = 0.3
            worker_env.apply_dynamic_disaster()
            last_disaster_time = 30.0
            path = []
            c_node = next_node
            worker_busy_time = 0.0
            
            p = visualizer.plot_state(hrl_env, frame_idx, png_dir, filename_prefix="disaster", trajectory=trajectory, mission_zone=mission_zone, zone_sequence=zone_sequence, global_time=global_time)
            snapshot_paths.append(p)
            frame_idx += 1
            
        elif global_time >= 60.0 and last_disaster_time < 60.0:
            print(f"💥 [Time {global_time:.1f}] Aftershock Triggered!")
            worker_env.disaster_prob = 0.1
            worker_env.apply_dynamic_disaster()
            last_disaster_time = 60.0
            path = []
            c_node = next_node
            worker_busy_time = 0.0
            
            p = visualizer.plot_state(hrl_env, frame_idx, png_dir, filename_prefix="aftershock", trajectory=trajectory, mission_zone=mission_zone, zone_sequence=zone_sequence, global_time=global_time)
            snapshot_paths.append(p)
            frame_idx += 1
            
        # 1. Continuous Time Window Check
        active_targets = 0
        for i in range(num_targets):
            if not hrl_env.target_rescued[b, i] and not hrl_env.target_failed[b, i]:
                # If target deadline expired during the tick
                if global_time > hrl_env.deadlines[b, i].item():
                    hrl_env.target_failed[b, i] = True
                    print(f"[{global_time:.1f}] Target {int(hrl_env.targets[b, i])} failed due to TW timeout.")
                    # Abort current path if it was our target
                    if i == best_t_idx:
                        path = []
                        best_t_idx = -1
                        target_node = -1
                        worker_busy_time = 0.0
                else:
                    active_targets += 1
                    
        if active_targets == 0 and best_t_idx == -1:
            if not done:
                print(f"[{global_time:.1f}] 모든 타겟을 완료하거나 구할 수 없습니다.")
                done = True
                
        # 2. Worker 의사결정 (Free 상태일 때만)
        if worker_busy_time <= 0 and not done:
            if c_node != next_node:
                c_node = next_node
                
            if c_node == target_node and best_t_idx != -1:
                if not hrl_env.target_failed[b, best_t_idx]:
                    hrl_env.target_rescued[b, best_t_idx] = True
                path = []
                best_t_idx = -1
                target_node = -1
                
            if path_idx >= len(path) - 1 or len(path) == 0:
                best_dist = float('inf')
                
                # 타겟 탐색
                for i in range(num_targets):
                    if not hrl_env.target_rescued[b, i] and not hrl_env.target_failed[b, i]:
                        t_node = int(hrl_env.targets[b, i].item())
                        try:
                            hop_len = nx.shortest_path_length(worker_env.G, c_node, t_node, weight=None)
                            path_weight = nx.shortest_path_length(worker_env.G, c_node, t_node, weight='weight')
                        except nx.NetworkXNoPath:
                            continue
                            
                        # EDF + Shortest Path Heuristic
                        urgency = max(0.0, hrl_env.deadlines[b, i].item() - global_time)
                        score = path_weight + urgency * 0.5
                        
                        if score < best_dist:
                            best_dist = score
                            best_t_idx = i
                            target_node = t_node
                                
                if best_t_idx == -1:
                    pass # Already handled by continuous check above
                else:
                    mission_zone = worker_env.n2z.get(target_node)
                    print(f"[{global_time:.1f}] Manager chose target {target_node} in zone {mission_zone} (TW: {hrl_env.deadlines[b, best_t_idx].item()})")
                    hrl_env.curr_target_idx[b] = best_t_idx
                    
                    try:
                        path = nx.shortest_path(worker_env.G, c_node, target_node, weight='weight')
                        path_idx = 0
                    except nx.NetworkXNoPath:
                        print(f"No path found to target {target_node}. Stopping.")
                        done = True
                        
                if not done and len(path) > 1:
                    c_node = path[path_idx]
                    next_node = path[path_idx + 1]
                    
                    if c_node == next_node:
                        edge_weight = 0.5
                    else:
                        edge_weight = worker_env.G[c_node][next_node].get('weight', 1.0)
                        
                    worker_busy_time = edge_weight
                    total_edge_time = edge_weight
                    
                    hrl_env.env.curr_nodes[b] = worker_env.node_to_idx[next_node]
                    hrl_env.env.visited_nodes[b, worker_env.node_to_idx[next_node]] = 1.0
                    trajectory.append(next_node)
                    path_idx += 1
                    
                    remaining_path = path[path_idx:]
                    zone_sequence = []
                    for n in remaining_path:
                        z = worker_env.n2z.get(n)
                        if z is not None:
                            if not zone_sequence or zone_sequence[-1] != z:
                                zone_sequence.append(z)
                        
        # 3. Time Tick and Rendering
        if worker_busy_time > 0 or done:
            progress = max(0.0, 1.0 - (worker_busy_time / total_edge_time)) if total_edge_time > 0 else 1.0
            
            p = visualizer.plot_state(hrl_env, frame_idx, png_dir, 
                                      trajectory=trajectory, 
                                      mission_zone=mission_zone, 
                                      zone_sequence=zone_sequence,
                                      worker_edge=(c_node, next_node),
                                      worker_progress=progress,
                                      global_time=global_time)
            snapshot_paths.append(p)
            
            if not done:
                worker_busy_time -= 0.5
                global_time += 0.5
            frame_idx += 1
            
    print(f"✅ Simulation ended at Time {global_time:.1f} for {map_name}. Rendering final pause frames...")
    
    # 4. Final Pause Rendering (10 frames = 2 seconds at 5 FPS)
    for _ in range(10):
        p = visualizer.plot_state(hrl_env, frame_idx, png_dir, 
                                  trajectory=trajectory, 
                                  mission_zone=mission_zone, 
                                  zone_sequence=zone_sequence,
                                  worker_edge=(c_node, next_node),
                                  worker_progress=1.0,
                                  global_time=global_time)
        snapshot_paths.append(p)
        frame_idx += 1
    
    # GIF 생성
    gif_path = os.path.join(gif_dir, "simulation_result.gif")
    DisasterVisualizer.create_gif(snapshot_paths, gif_path, fps=5)

if __name__ == '__main__':
    maps = ['Anaheim', 'Berlin-Friedrichshain']
    for m in maps:
        run_heuristic_visualization(m)
