
import os
import argparse
import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"🚀 {description}...")
    print(f"   Command: {command}")
    print(f"{'='*60}\n")
    
    try:
        # Inherit stdout/stderr to preserve TTY (tqdm support)
        result = subprocess.run(command, shell=True)
        
        if result.returncode != 0:
            print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        
        print(f"\n✅ {description} Completed Successfully.")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by User.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Integrated Training Pipeline for Disaster Context UGV")
    
    # Global Args
    parser.add_argument('--map', default='Anaheim', help='Map Name (Default: Anaheim)')
    parser.add_argument('--data', default='data', help='Data Directory containing .pt files')
    
    # SL Args
    parser.add_argument('--skip_sl', action='store_true', help='Skip Supervised Learning Phase')
    parser.add_argument('--force_sl', action='store_true', help='Force SL Pretraining (Ignore logs)')
    parser.add_argument('--epochs_sl', type=int, default=50, help='Epochs for SL Phase (Default: 50)')
    parser.add_argument('--batch_sl', type=int, default=64, help='Batch size for SL Phase')
    parser.add_argument('--hidden_dim_sl', type=int, default=256, help='Hidden dim for SL Phase (Default: 256)')
    parser.add_argument('--lr_manager_sl', type=float, default=5e-4, help='Manager Learning rate (SL)')
    parser.add_argument('--lr_worker_sl', type=float, default=1e-4, help='Worker Learning rate (SL)')
    
    # RL Args
    parser.add_argument('--skip_rl', action='store_true', help='Skip Reinforcement Learning Phase')
    parser.add_argument('--episodes_rl', type=int, default=5000, help='Episodes for RL Phase (Default: 5000)')
    parser.add_argument('--hidden_dim_rl', type=int, default=256, help='Hidden dim for RL Phase (Default: 256)')
    parser.add_argument('--lr_rl', type=float, default=1e-4, help='Learning rate for RL Phase')
    parser.add_argument('--batch_rl', type=int, default=16, help='Batch size (POMO) for RL Phase')
    
    # Viz Args
    parser.add_argument('--skip_viz', action='store_true', help='Skip Visualization Phase')
    parser.add_argument('--viz_episodes', type=int, default=5, help='Number of Episodes to Visualize')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Total Number of Episodes to Evaluate (for Metrics)')
    
    args = parser.parse_args()
    
    # 1. Supervised Learning (Pre-training)
    if not args.skip_sl:
        cmd_sl = (
            f'"{sys.executable}" train_sl.py '
            f'--map {args.map} '
            f'--data {args.data} '
            f'--epochs {args.epochs_sl} '
            f'--batch_size {args.batch_sl} '
            f'--hidden_dim {args.hidden_dim_sl} '
            f'--lr_manager {args.lr_manager_sl} '
            f'--lr_worker {args.lr_worker_sl}'
        )
        run_command(cmd_sl, "Phase 1: Supervised Learning (Pre-training)")
    
    # 2. Reinforcement Learning (Fine-tuning)
    if not args.skip_rl:
        # Phase 1: Static Environment
        cmd_rl_p1 = (
            f'"{sys.executable}" train_rl.py '
            f'--data {args.data} '
            f'--map {args.map} '
            f'--episodes {args.episodes_rl // 2} '
            f'--hidden_dim {args.hidden_dim_rl} '
            f'--lr {args.lr_rl} '
            f'--batch_size {args.batch_rl}'
        )
        run_command(cmd_rl_p1, "Phase 2-1: Reinforcement Learning (Phase 1: Static Route Optimization)")
        
        # Phase 2: Dynamic Environment
        cmd_rl_p2 = (
            f'"{sys.executable}" train_rl.py '
            f'--data {args.data} '
            f'--map {args.map} '
            f'--episodes {args.episodes_rl // 2} '
            f'--hidden_dim {args.hidden_dim_rl} '
            f'--lr {args.lr_rl} '
            f'--batch_size {args.batch_rl} '
            f'--disaster'
        )
        run_command(cmd_rl_p2, "Phase 2-2: Reinforcement Learning (Phase 2: Dynamic Disaster Adaptation)")
        
    # 3. Visualization
    if not args.skip_viz:
        # Viz Phase 1 Model
        cmd_viz_p1 = f'"{sys.executable}" tests/visualize_result.py --map {args.map} --episodes {args.eval_episodes} --viz_episodes {args.viz_episodes} --mode rl --phase 1 --hidden_dim {args.hidden_dim_rl}'
        run_command(cmd_viz_p1, "Phase 3-1: Visualization & Verification (RL Phase 1 Model)")
        
        # Viz Phase 2 Model
        cmd_viz_p2 = f'"{sys.executable}" tests/visualize_result.py --map {args.map} --episodes {args.eval_episodes} --viz_episodes {args.viz_episodes} --mode rl --phase 2 --disaster --hidden_dim {args.hidden_dim_rl}'
        run_command(cmd_viz_p2, "Phase 3-2: Visualization & Verification (RL Phase 2 Model)")

if __name__ == "__main__":
    main()
