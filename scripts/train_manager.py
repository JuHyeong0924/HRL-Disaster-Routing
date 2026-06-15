import os
import sys
import argparse
import warnings
import torch
from datetime import datetime

# PyTorch Nested Tensor 경고 무시
warnings.filterwarnings("ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*")

sys.path.insert(0, '.')
from src.envs.worker_env import WorkerEnv
from src.models.worker import Worker
from src.models.manager_unified import ManagerUnified
from src.envs.hrl_env import HRLEnv
from src.trainers.manager_ppo_trainer import ManagerPPOTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Anaheim')
    parser.add_argument('--worker_ckpt', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mini_batch_size', type=int, default=2048)
    parser.add_argument('--num_targets', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disaster_prob', type=float, default=0.0)
    parser.add_argument('--dynamic_disaster', action='store_true')
    parser.add_argument('--exp_name', type=str, default='manager_ppo')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Worker
    worker_env = WorkerEnv(
        f'data/{args.map}_node.tntp', f'data/{args.map}_net.tntp',
        zone_json=f'data/grid_{args.map}_node_to_zone.json',
        zone_graph_json=f'data/grid_{args.map}_zone_graph.json',
        masking_mode='soft_curr_next',
        disaster_prob=args.disaster_prob,
        dynamic_disaster=args.dynamic_disaster,
        device=device
    )
    
    worker = Worker(node_dim=5, hidden_dim=256, num_layers=2, dropout=0.0).to(device)
    ckpt = torch.load(args.worker_ckpt, map_location=device, weights_only=False)
    state = ckpt.get('worker_state', ckpt.get('state_dict', ckpt))
    cur = worker.state_dict()
    compat = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
    worker.load_state_dict(compat, strict=False)
    
    # 2. Init HRLEnv (동적 한계치 적용되므로 파라미터 제외)
    hrl_env = HRLEnv(worker, worker_env)
    
    # 3. Init Manager
    manager = ManagerUnified(
        zone_dim=6, target_dim=4, hidden_dim=128,
        num_gat_layers=2, gat_heads=4, num_transformer_layers=1, transformer_heads=4
    ).to(device)
    
    # 4. Init Trainer
    trainer = ManagerPPOTrainer(manager, hrl_env, config=args)
    
    # 5. Training Loop
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_dir = f'logs/rl_manager_stage/{timestamp}_{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    best_reward = -float('inf')
    
    from tqdm import tqdm
    print(f"🚀 Manager PPO Training Start (Worker: {args.worker_ckpt})")
    num_steps = (args.episodes + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=args.episodes, desc="Manager PPO", ncols=140, unit="ep") as pbar:
        for step in range(1, num_steps + 1):
            logs = trainer.train_step(batch_size=args.batch_size, num_targets=args.num_targets)
            
            pbar.set_postfix({
                'Loss': f"{logs.get('loss', 0):.4f}",
                'P': f"{logs.get('policy_loss', 0):.4f}",
                'V': f"{logs.get('value_loss', 0):.4f}",
                'Rw': f"{logs['mean_reward']:.2f}",
                'Rsc': f"{logs['mean_rescued']:.1f}/{args.num_targets}",
                'Trn': f"{logs['mean_manager_turns']:.1f}",
                'WStp': f"{logs['mean_worker_steps']:.1f}"
            })
            pbar.update(args.batch_size)
            
            if logs['mean_reward'] > best_reward:
                best_reward = logs['mean_reward']
                torch.save(manager.state_dict(), f'{save_dir}/best_manager.pt')
                tqdm.write(f"  => Best model saved (Reward: {best_reward:.2f})")
                
    print(f"✅ Training Complete. Best Reward: {best_reward:.2f}")
            
if __name__ == '__main__':
    main()
