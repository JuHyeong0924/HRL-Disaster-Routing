
import os
import sys
import pickle
import math
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse
from collections import deque


def compute_distance_to_path(G, path_nodes, node_to_idx, num_nodes):
    """
    그래프 BFS를 사용하여 각 노드에서 A* 경로까지의 최소 거리(홉 수)를 계산.
    Returns: torch.Tensor [num_nodes] - 정수 거리. 경로 위 노드 = 0, 1-hop 이웃 = 1, ...
    Soft Label 생성에 사용: 경로 근처 노드에 높은 확률 부여.
    """
    distance_map = torch.full((num_nodes,), fill_value=float('inf'), dtype=torch.float32)
    
    # 경로 위 노드 거리 = 0
    queue = deque()
    visited = set()
    
    for node in path_nodes:
        idx = node_to_idx[node]
        distance_map[idx] = 0.0
        queue.append((node, 0))
        visited.add(node)
    
    # BFS 탐색
    while queue:
        current_node, dist = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                n_idx = node_to_idx[neighbor]
                distance_map[n_idx] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return distance_map


# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.graph_loader import GraphLoader

def generate_expert_data(map_name='Anaheim', count=50000, manager_ratio=1, save_dir='data'):
    """
    A* 최적 경로 기반 Expert Demonstration 데이터 생성.
    
    [v2 2026-02-11] Manager 데이터 개선:
    - 기존: 전체 A* 경로를 시퀀스로 사용 (30~50 노드) → RL max_len=10과 불일치
    - 개선: Sparse Waypoints (매 step_size번째 + 방향 변화점) → 5~15개 핵심 웨이포인트
    
    이 변경으로 SL에서 배운 "간결한 계획"이 RL의 생성 방식과 일치하게 됨.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    node_file = os.path.join(base_dir, f"{map_name}_node.tntp")
    net_file = os.path.join(base_dir, f"{map_name}_net.tntp")
    
    loader = GraphLoader(node_file, net_file)
    G = loader.get_nx_graph()
    nodes = list(G.nodes())
    
    # 노드 좌표 사전 (방향 변화 감지용)
    node_positions = {}
    pyg_data = loader.get_pyg_data()
    for i, node in enumerate(nodes):
        node_positions[node] = (pyg_data.x[i, 0].item(), pyg_data.x[i, 1].item())
    
    print(f"Using Full Node Set as Vocabulary ({len(nodes)} nodes).")
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'checkpoints.pkl'), 'wb') as f:
        pickle.dump(nodes, f)
    print(f"Saved Metadata to {os.path.join(save_dir, 'checkpoints.pkl')}")

    print(f"Generating {count} Expert Paths (Manager Ratio: {manager_ratio}x)...")
    
    # Manager Containers
    mgr_start = []
    mgr_goal = []
    mgr_checkpoints = []     # List of Tensors (Sparse Waypoints)
    mgr_distance_maps = []   # 거리 맵 (Soft Label용)
    
    # Worker Containers
    wkr_curr = []
    wkr_target = []
    wkr_next = []
    
    mgr_count = 0
    wkr_count_paths = 0 # Worker 데이터를 생성한 경로 수
    
    # Target Counts
    target_mgr_count = count * manager_ratio
    target_wkr_count = count
    
    pbar = tqdm(total=target_mgr_count, desc="Generating Paths")
    
    # 노드 인덱스 매핑
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    while mgr_count < target_mgr_count:
        u, v = np.random.choice(nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, source=u, target=v, weight='weight')
        except nx.NetworkXNoPath:
            continue
            
        if len(path) < 5: 
            continue
            
        mgr_count += 1
        pbar.update(1)
        
        # === Manager Data: Sparse Waypoints (Always Collect) ===
        # [v6] 경로 위의 구조적 포인트만 남기는 sparse subgoal planner
        # - 짧은 문제(shortest_hops <= 4)는 goal-only 계획
        # - 그 외는 A* 경로 위에서 교차로/turn point를 우선 유지
        # - 3~6 hop 간격을 유지해 지나치게 촘촘한 manager label을 방지
        sparse_waypoints = _extract_sparse_waypoints(
            G,
            path,
            node_positions,
            min_gap=3,
            max_gap=6,
            angle_threshold=25.0,
        )
        
        # 시작 노드 제거 (Manager는 Start→첫 웨이포인트를 예측)
        label_seq = sparse_waypoints[1:] if sparse_waypoints[0] == u else sparse_waypoints
        
        # 목표 노드가 마지막에 포함되도록 보장
        if label_seq[-1] != v:
            label_seq.append(v)
        
        mgr_start.append(node_to_idx[u])
        mgr_goal.append(node_to_idx[v])
        
        seq_indices = [node_to_idx[n] for n in label_seq]
        mgr_checkpoints.append(torch.tensor(seq_indices, dtype=torch.long))
        
        # 거리 맵 생성: A* 경로까지의 BFS 거리
        dist_map = compute_distance_to_path(G, path, node_to_idx, len(nodes))
        mgr_distance_maps.append(dist_map)
        
        # === Worker Data: Collect only up to 'count' paths ===
        if wkr_count_paths < target_wkr_count:
            wkr_count_paths += 1
            
            # Worker는 sparse_waypoints 사이의 구간을 네비게이션
            worker_targets = sparse_waypoints[1:]
            if not worker_targets:
                worker_targets = [path[-1]]
            
            target_ptr = 0
            current_target_node = worker_targets[0]
            
            # 한 경로에 대한 시퀀스 저장용 리스트
            path_curr = []
            path_target = []
            path_next = []
            
            for i, curr_node in enumerate(path[:-1]):
                # 현재 타겟에 도달하면 다음 타겟으로 전환
                if curr_node == current_target_node:
                    target_ptr += 1
                    if target_ptr < len(worker_targets):
                        current_target_node = worker_targets[target_ptr]
                    else:
                        current_target_node = worker_targets[-1] 
                
                path_curr.append(node_to_idx[curr_node])
                path_target.append(node_to_idx[current_target_node])
                path_next.append(node_to_idx[path[i+1]])
                
            wkr_curr.append(torch.tensor(path_curr, dtype=torch.long))
            wkr_target.append(torch.tensor(path_target, dtype=torch.long))
            wkr_next.append(torch.tensor(path_next, dtype=torch.long))
            
    pbar.close()
    
    # 시퀀스 길이 통계 출력
    seq_lengths = [len(seq) for seq in mgr_checkpoints]
    print(f"\n📊 Manager Sequence Length Stats ({len(mgr_start)} samples):")
    print(f"   Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, "
          f"Mean: {np.mean(seq_lengths):.1f}, Median: {np.median(seq_lengths):.1f}")
    
    # 3. Convert and Save
    print("Converting to Tensors...")
    
    manager_data = {
        'start_nodes': torch.tensor(mgr_start, dtype=torch.long),
        'goal_nodes': torch.tensor(mgr_goal, dtype=torch.long),
        'checkpoint_seqs': mgr_checkpoints,
        'distance_maps': mgr_distance_maps
    }
    torch.save(manager_data, os.path.join(save_dir, 'manager_data.pt'))
    print(f"Saved Manager Data ({len(mgr_start)} samples)")
    
    worker_data = {
        'curr_nodes': wkr_curr,
        'target_nodes': wkr_target,
        'next_hops': wkr_next
    }
    torch.save(worker_data, os.path.join(save_dir, 'worker_data.pt'))
    # Flatten된 총 스텝 수 계산
    total_wkr_steps = sum(len(seq) for seq in wkr_curr)
    print(f"Saved Worker Data ({wkr_count_paths} paths, {total_wkr_steps} steps in sequences)")


def _compute_turn_angle(path, node_positions, idx):
    if idx <= 0 or idx >= len(path) - 1:
        return 0.0

    a = node_positions.get(path[idx - 1])
    b = node_positions.get(path[idx])
    c = node_positions.get(path[idx + 1])
    if a is None or b is None or c is None:
        return 0.0

    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    len_ab = math.hypot(ab[0], ab[1])
    len_bc = math.hypot(bc[0], bc[1])
    if len_ab <= 1e-8 or len_bc <= 1e-8:
        return 0.0

    cos_angle = (ab[0] * bc[0] + ab[1] * bc[1]) / (len_ab * len_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _is_structural_candidate(G, path, node_positions, idx, angle_threshold):
    if idx <= 0 or idx >= len(path) - 1:
        return False
    node = path[idx]
    if G.degree(node) >= 3:
        return True
    return _compute_turn_angle(path, node_positions, idx) >= angle_threshold


def _extract_sparse_waypoints(G, path, node_positions, min_gap=3, max_gap=6, angle_threshold=25.0):
    """
    A* 경로 위에서만 sparse waypoint를 선택하는 manager expert planner.

    규칙:
    1. shortest_hops <= 4 이면 goal-only plan.
    2. 후보는 A* 경로 위의 structural node(교차로 or 큰 turn)만 사용.
    3. 마지막으로 고른 지점에서 3~6 hop 구간을 sweep하면서 첫 structural candidate를 채택.
    4. 구조적 후보가 없으면 max_gap 위치의 anchor를 강제로 채택.
    5. 마지막 waypoint는 항상 goal.
    """
    if len(path) <= 2:
        return list(path)

    shortest_hops = len(path) - 1
    if shortest_hops <= 4:
        return [path[0], path[-1]]

    min_gap = max(int(min_gap), 1)
    max_gap = max(int(max_gap), min_gap)
    structural_indices = {
        idx
        for idx in range(1, len(path) - 1)
        if _is_structural_candidate(G, path, node_positions, idx, angle_threshold)
    }

    selected_indices = [0]
    last_kept = 0
    goal_idx = len(path) - 1

    while last_kept < goal_idx:
        remaining = goal_idx - last_kept
        if remaining <= max_gap:
            selected_indices.append(goal_idx)
            break

        scan_start = min(last_kept + min_gap, goal_idx)
        scan_end = min(last_kept + max_gap, goal_idx)
        chosen_idx = next(
            (idx for idx in range(scan_start, scan_end + 1) if idx in structural_indices),
            None,
        )
        if chosen_idx is None:
            chosen_idx = scan_end

        if chosen_idx <= last_kept:
            chosen_idx = min(last_kept + max_gap, goal_idx)

        selected_indices.append(chosen_idx)
        last_kept = chosen_idx

    if selected_indices[-1] != goal_idx:
        selected_indices.append(goal_idx)

    dedup_indices = []
    for idx in selected_indices:
        if not dedup_indices or dedup_indices[-1] != idx:
            dedup_indices.append(idx)

    return [path[idx] for idx in dedup_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50000, help='Base number of paths (for Worker)')
    parser.add_argument('--manager_ratio', type=int, default=10, help='Ratio of Manager data relative to count (Default: 10x)')
    parser.add_argument('--map', type=str, default='Anaheim')
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()
    
    generate_expert_data(map_name=args.map, count=args.count, manager_ratio=args.manager_ratio, save_dir=args.save_dir)
