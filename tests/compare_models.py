import os
import sys
import argparse
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing
import json

# Add Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.envs.disaster_env import DisasterEnv
from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM

# Global variables for multiprocessing workers
g_env = None
g_manager_sl = None
g_worker_sl = None
g_manager_rl = None
g_worker_rl = None
g_args = None

def worker_init(map_name, hidden_dim, device_str, manager_sl_sd, worker_sl_sd, manager_rl_sd, worker_rl_sd, max_steps):
    """Initializer for worker processes to load env and models once."""
    global g_env, g_manager_sl, g_worker_sl, g_manager_rl, g_worker_rl, g_args
    
    # 1. Load Env
    g_env = DisasterEnv(f'data/{map_name}_node.tntp', f'data/{map_name}_net.tntp', device=device_str, verbose=False, enable_disaster=False)
    
    # 2. Load Models
    device = torch.device(device_str)
    
    # SL Models
    g_manager_sl = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim).to(device)
    if manager_sl_sd: g_manager_sl.load_state_dict(manager_sl_sd)
    g_manager_sl.eval()
    
    g_worker_sl = WorkerLSTM(node_dim=7, hidden_dim=hidden_dim).to(device)
    if worker_sl_sd: g_worker_sl.load_state_dict(worker_sl_sd)
    g_worker_sl.eval()
    
    # RL Models
    g_manager_rl = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim).to(device)
    if manager_rl_sd: g_manager_rl.load_state_dict(manager_rl_sd)
    g_manager_rl.eval()
    
    g_worker_rl = WorkerLSTM(node_dim=7, hidden_dim=hidden_dim).to(device)
    if worker_rl_sd: g_worker_rl.load_state_dict(worker_rl_sd)
    g_worker_rl.eval()
    
    g_args = {'max_steps': max_steps}

@torch.no_grad()
def rollout(env, manager, worker, start_node, goal_node, max_steps=400):
    """Simplified Rollout Function without Visualization Overhead"""
    # 1. Reset Env
    env.reset(batch_size=1)
    env.current_node = torch.tensor([env.node_mapping[start_node]]).to(env.device)
    env.target_node = torch.tensor([env.node_mapping[goal_node]]).to(env.device)
    
    # Update Features & Mask (Initial)
    # env.get_mask() expects visited to be tracked. Reset handles this mostly, but we need to init batch 1 manually for safe rollout
    # Easier to just bypass pyG batch and do exact steps if needed, or use the env.
    
    # Actually, to be safe and use exact RL env logic, we must use `train_rl` style loop or the visualization rollout.
    # Let's import the rollout function from visualize_result to maintain 100% consistency.
    from tests.visualize_result import hierarchical_rollout
    
    try:
        steps_data, final_path, _ = hierarchical_rollout(
            env, manager, worker, None, start_node, goal_node, 
            max_steps=max_steps, worker_only=False, verbose=False
        )
        
        # Calculate A* Optimal Length
        optimal_path = nx.shortest_path(env.map_core.graph, source=start_node, target=goal_node, weight='length')
        opt_len = sum(env.map_core.graph[u][v].get('length', 1.0) for u, v in zip(optimal_path[:-1], optimal_path[1:]))
        
        agent_len = sum(env.map_core.graph[u][v].get('length', 1.0) for u, v in zip(final_path[:-1], final_path[1:]))
        
        is_success = (final_path[-1] == goal_node)
        ratio = agent_len / opt_len if opt_len > 0 and is_success else 0.0
        
        return {
            'success': is_success,
            'steps': len(final_path) - 1,
            'opt_steps': len(optimal_path) - 1,
            'agent_len': agent_len,
            'opt_len': opt_len,
            'ratio': ratio
        }
    except Exception as e:
        return {'success': False, 'steps': max_steps, 'opt_steps': 1, 'agent_len': 0, 'opt_len': 1, 'ratio': 0}

@torch.no_grad()
def evaluate_single_episode_mp(ep_idx):
    """Function executed by each worker process."""
    torch.cuda.empty_cache()
    global g_env, g_manager_sl, g_worker_sl, g_manager_rl, g_worker_rl, g_args
    
    np.random.seed(ep_idx)
    nodes = list(g_env.map_core.graph.nodes())
    start, goal = np.random.choice(nodes, 2, replace=False)
    
    res_sl = rollout(g_env, g_manager_sl, g_worker_sl, start, goal, g_args['max_steps'])
    res_rl = rollout(g_env, g_manager_rl, g_worker_rl, start, goal, g_args['max_steps'])
    
    return {'ep': ep_idx, 'sl': res_sl, 'rl': res_rl}

def load_latest_model(dir_path, mgr_prefix, wkr_prefix):
    if not os.path.exists(dir_path): return None, None
    files = os.listdir(dir_path)
    mgr_files = [f for f in files if f.startswith(mgr_prefix) and f.endswith(".pt") or f.endswith(".pth")]
    if not mgr_files: return None, None
    
    def extract_ep(fname):
        try: return int(fname.split('_')[1].split('.')[0])
        except: return -1
        
    latest_mgr = max(mgr_files, key=extract_ep)
    ep_num = extract_ep(latest_mgr)
    
    if ep_num >= 0: latest_wkr = f"{wkr_prefix}{ep_num}.pt"
    else: latest_wkr = f"{wkr_prefix}.pth"
        
    manager_path = os.path.join(dir_path, latest_mgr)
    worker_path = os.path.join(dir_path, latest_wkr)
    
    print(f"Loaded: {manager_path}, {worker_path}")
    return manager_path, worker_path

def main(args):
    print("="*60)
    print(f"🚀 SL vs RL Phase 1 (Static) Model Comparison on {args.map}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load SL Models
    mgr_sl_path, wkr_sl_path = load_latest_model("logs/sl_pretrain", "manager_sl", "worker_sl")
    mgr_sl_sd = torch.load(mgr_sl_path, map_location='cpu') if mgr_sl_path else None
    wkr_sl_sd = torch.load(wkr_sl_path, map_location='cpu') if wkr_sl_path else None
    
    # 2. Load RL Phase 1 Models
    mgr_rl_path, wkr_rl_path = load_latest_model("logs/rl_finetune_phase1", "manager_", "worker_")
    mgr_rl_sd = torch.load(mgr_rl_path, map_location='cpu') if mgr_rl_path else None
    wkr_rl_sd = torch.load(wkr_rl_path, map_location='cpu') if wkr_rl_path else None
    
    if not mgr_sl_sd or not mgr_rl_sd:
        print("❌ Error: Both SL and RL Phase 1 models are required for comparison.")
        return
        
    # 3. Parallel Evaluation
    print(f"\nEvaluating {args.episodes} random routes in parallel...")
    num_cores = 8 if args.cpu else max(1, torch.cuda.device_count() * 2) # Assume 2 workers per GPU
    worker_device = 'cpu' if args.cpu else str(device)
    
    ctx = multiprocessing.get_context('spawn')
    all_metrics = []
    
    with ctx.Pool(processes=num_cores, initializer=worker_init, initargs=(args.map, 256, worker_device, mgr_sl_sd, wkr_sl_sd, mgr_rl_sd, wkr_rl_sd, args.max_steps)) as pool:
        ep_range = range(args.episodes)
        results = list(tqdm(pool.imap(evaluate_single_episode_mp, ep_range), total=len(ep_range), desc="Comparing Models"))
        for res in results:
            if res: all_metrics.append(res)
            
    # 4. Compile Statistics
    sl_success = sum(1 for m in all_metrics if m['sl']['success'])
    rl_success = sum(1 for m in all_metrics if m['rl']['success'])
    
    sl_ratios = [m['sl']['ratio'] for m in all_metrics if m['sl']['success']]
    rl_ratios = [m['rl']['ratio'] for m in all_metrics if m['rl']['success']]
    
    sl_avg_ratio = np.mean(sl_ratios) if sl_ratios else 0.0
    rl_avg_ratio = np.mean(rl_ratios) if rl_ratios else 0.0
    
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON RESULTS OVER {args.episodes} EPISODES ({args.map})")
    print(f"{'='*60}")
    print(f"                      | SL Pre-trained | RL Phase 1 (Static) |")
    print(f"----------------------|----------------|---------------------|")
    print(f" 🎯 Success Rate      | {sl_success/args.episodes:>13.1%} | {rl_success/args.episodes:>18.1%} |")
    print(f" 📏 Path Length Ratio | {sl_avg_ratio:>13.3f} | {rl_avg_ratio:>18.3f} |")
    print(f"{'='*60}")
    print("* Path Length Ratio: 낮을수록 좋음 (1.0 = A* 최적경로)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Anaheim')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--cpu', action='store_true', help='Force CPU multiprocessing')
    
    args = parser.parse_args()
    main(args)
