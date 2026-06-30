import os
import sys
import torch
import random
import numpy as np
import glob
import argparse

sys.path.insert(0, '.')
from src.utils.eval_utils import load_eval_env, load_neural_models, run_evaluation_episode
from src.models.heuristics import GA_Manager, ALNS_Manager, Dijkstra_Worker
from src.utils.visualizer import DisasterVisualizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['benchmark', 'analyze', 'visualize'],
                        help="평가 모드 선택 (benchmark: 성능표, analyze: 스텝로그, visualize: GIF생성)")
    parser.add_argument('--episodes', type=int, default=None, help="실행할 에피소드 수 (미지정시 mode별 기본값)")
    parser.add_argument('--map', type=str, default='Anaheim')
    parser.add_argument('--num_targets', type=int, default=15)
    parser.add_argument('--worker_ckpt', type=str, default='logs/rl_worker_stage/2026-06-26_151453_worker/best.pt')
    parser.add_argument('--manager_ckpt', type=str, default='logs/rl_manager_stage/2026-06-26_190100_manager/best_manager.pt')
    parser.add_argument('--num_layers', type=int, default=4, help="Worker GNN 레이어 수")
    parser.add_argument('--disaster_prob', type=float, default=0.2, help="재난 발생 확률")
    parser.add_argument('--dynamic_disaster', action='store_true', default=True, help="동적 여진 활성화")
    parser.add_argument('--no_dynamic_disaster', action='store_true', help="동적 여진 비활성화")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default episodes by mode
    if args.episodes is None:
        args.episodes = {'benchmark': 10, 'analyze': 5, 'visualize': 1}[args.mode]
        
    if args.no_dynamic_disaster:
        args.dynamic_disaster = False

    print(f"Loading Models on {device}...")
    neural_worker, neural_manager = load_neural_models(device, args.worker_ckpt, args.manager_ckpt, num_layers=args.num_layers)
    
    if args.mode == 'benchmark':
        models_to_test = [
            ('HRL', neural_manager, neural_worker),
            ('HRL-Dijkstra', neural_manager, None),
            ('GA-Neural', GA_Manager(None), neural_worker),
            ('ALNS-Neural', ALNS_Manager(None), neural_worker),
            ('GA-Dijkstra', GA_Manager(None), None),
            ('ALNS-Dijkstra', ALNS_Manager(None), None)
        ]
        
        results = {m[0]: {'rescued': [], 'latency': [], 'recompute': [], 'ugv_destroys': [], 'total_dist': []} for m in models_to_test}
        print(f"\n🚀 Starting Benchmark on {args.map} ({args.episodes} Episodes)")
        print("=" * 80)
        
        envs_cache = {}
        for ep in range(args.episodes):
            seed = 42 + ep
            for name, manager, worker in models_to_test:
                set_seed(seed)
                if name not in envs_cache:
                    worker_env, hrl_env = load_eval_env(
                        args.map, device, 
                        disaster_prob=args.disaster_prob, 
                        dynamic_disaster=args.dynamic_disaster, 
                        num_layers=args.num_layers
                    )
                    envs_cache[name] = (worker_env, hrl_env)
                else:
                    worker_env, hrl_env = envs_cache[name]
                
                actual_worker = worker if worker is not None else Dijkstra_Worker(worker_env)
                hrl_env.worker = actual_worker
                
                rescued, latency, recompute, ugv_destroys, total_dist = run_evaluation_episode(
                    manager, actual_worker, hrl_env, args.num_targets, device, mode='benchmark'
                )
                
                results[name]['rescued'].append(rescued)
                results[name]['latency'].append(latency)
                results[name]['recompute'].append(recompute)
                results[name]['ugv_destroys'].append(ugv_destroys)
                results[name]['total_dist'].append(total_dist)
                    
            print(f"Episode {ep+1}/{args.episodes} completed.")
            
        print("\n" + "=" * 100)
        print(f"{'Model':<15} | {'Rescue Rate (%)':<15} | {'Latency (s)':<15} | {'Total Dist':<12} | {'Recomputes':<12} | {'UGV Destroys':<12}")
        print("-" * 100)
        for name in results.keys():
            resc = np.mean(results[name]['rescued']) / args.num_targets * 100
            lat = np.mean(results[name]['latency'])
            dist = np.mean(results[name]['total_dist'])
            recomp = np.mean(results[name]['recompute'])
            destroys = np.mean(results[name]['ugv_destroys'])
            print(f"{name:<15} | {resc:>13.2f} % | {lat:>13.3f} s | {dist:>10.1f} | {recomp:>10.1f} | {destroys:>10.1f}")
        print("=" * 100)

    elif args.mode == 'analyze':
        for ep in range(args.episodes):
            print(f"\n{'='*50}")
            print(f"🚀 Episode {ep+1} Start")
            print(f"{'='*50}")
            
            set_seed(42 + ep)
            worker_env, hrl_env = load_eval_env(args.map, device, disaster_prob=0.0, dynamic_disaster=False)
            hrl_env.worker = neural_worker
            
            run_evaluation_episode(neural_manager, neural_worker, hrl_env, args.num_targets, device, mode='analyze')

    elif args.mode == 'visualize':
        save_base_dir = f'figs/{args.map}/dynamic_hrl'
        frame_dir = os.path.join(save_base_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        
        for ep in range(args.episodes):
            print(f"\n{'='*40}")
            print(f"Generating GIF for {args.map} (Episode {ep+1})...")
            print(f"{'='*40}")
            
            set_seed(42 + ep)
            worker_env, hrl_env = load_eval_env(args.map, device, disaster_prob=0.05, dynamic_disaster=True)
            hrl_env.worker = neural_worker
            
            visualizer = DisasterVisualizer(dm=hrl_env.env.dm, n2z=hrl_env.env.n2z)
            
            for f in glob.glob(os.path.join(frame_dir, "*.png")):
                os.remove(f)
                
            frame_idx_ref = [0]
            run_evaluation_episode(
                neural_manager, neural_worker, hrl_env, args.num_targets, device, 
                mode='visualize', visualizer=visualizer, frame_dir=frame_dir, frame_idx_ref=frame_idx_ref
            )
            
            print("Generating GIF...")
            png_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
            gif_path = os.path.join(save_base_dir, f"{args.map}_dynamic_routing_ep{ep+1}.gif")
            if png_paths:
                visualizer.create_gif(png_paths, gif_path, fps=10)
                print(f"Saved: {gif_path}")
            else:
                print("No frames were generated!")

if __name__ == '__main__':
    main()
