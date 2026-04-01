import os
import argparse
import csv
import json
import os
import torch
from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM
from src.envs.disaster_env import DisasterEnv
from src.trainers.pomo_trainer import DOMOTrainer
from src.trainers.manager_stage_trainer import ManagerStageTrainer
from src.trainers.worker_stage_trainer import WorkerStageTrainer

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _read_last_debug_row(save_dir):
    csv_path = os.path.join(save_dir, "debug_metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1] if rows else {}


def _safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _build_phase1_readiness():
    manager_dir = os.path.join("logs", "rl_phase1_manager")
    worker_dir = os.path.join("logs", "rl_phase1_worker")
    joint_dir = os.path.join("logs", "rl_phase1_joint")
    manager_row = _read_last_debug_row(manager_dir)
    worker_row = _read_last_debug_row(worker_dir)

    payload = {
        "manager": {
            "plan_under_rate": _safe_float(manager_row, "plan_under_rate", 1.0),
            "anchor_near_rate": _safe_float(manager_row, "anchor_near_rate", 0.0),
            "first_subgoal_hops_mean": _safe_float(manager_row, "first_subgoal_hops_mean", 999.0),
        },
        "worker": {
            "stagnation_fail_rate": _safe_float(worker_row, "stagnation_fail_rate", 1.0),
            "goal_after_last_subgoal_rate": _safe_float(worker_row, "goal_after_last_subgoal_rate", 0.0),
            "post_last_sg_success_rate": _safe_float(worker_row, "post_last_sg_success_rate", 0.0),
        },
    }
    smoke_ready = (
        payload["manager"]["plan_under_rate"] < 0.40
        and payload["manager"]["anchor_near_rate"] > 0.10
        and payload["worker"]["stagnation_fail_rate"] < 0.50
        and payload["worker"]["goal_after_last_subgoal_rate"] > 0.50
    )
    launch_ready = (
        payload["manager"]["plan_under_rate"] < 0.25
        and payload["manager"]["anchor_near_rate"] > 0.20
        and payload["manager"]["first_subgoal_hops_mean"] <= 6.0
        and payload["worker"]["stagnation_fail_rate"] < 0.30
        and payload["worker"]["goal_after_last_subgoal_rate"] > 0.70
        and payload["worker"]["post_last_sg_success_rate"] > 0.60
    )
    payload["smoke_ready"] = bool(smoke_ready)
    payload["launch_ready"] = bool(launch_ready)

    os.makedirs(joint_dir, exist_ok=True)
    with open(os.path.join(joint_dir, "stage_readiness.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return payload


def _write_refbin_artifact():
    joint_dir = os.path.join("logs", "rl_phase1_joint")
    manager_row = _read_last_debug_row(os.path.join("logs", "rl_phase1_manager"))
    worker_row = _read_last_debug_row(os.path.join("logs", "rl_phase1_worker"))
    joint_row = _read_last_debug_row(joint_dir)
    os.makedirs(joint_dir, exist_ok=True)
    out_path = os.path.join(joint_dir, "val_by_refbin.csv")
    fieldnames = ["stage", "ref_bin", "metric", "value"]
    rows = []
    manager_ref = int(round(_safe_float(manager_row, "plan_len_ref_mean", 0.0)))
    worker_ref = int(round(_safe_float(worker_row, "plan_len_ref_mean", 0.0)))
    joint_ref = int(round(_safe_float(joint_row, "plan_len_ref_mean", 0.0)))
    rows.extend([
        {"stage": "manager", "ref_bin": str(manager_ref if manager_ref <= 3 else ">=4"), "metric": "plan_under_rate", "value": _safe_float(manager_row, "plan_under_rate", 0.0)},
        {"stage": "manager", "ref_bin": str(manager_ref if manager_ref <= 3 else ">=4"), "metric": "anchor_near_rate", "value": _safe_float(manager_row, "anchor_near_rate", 0.0)},
        {"stage": "manager", "ref_bin": str(manager_ref if manager_ref <= 3 else ">=4"), "metric": "first_subgoal_hops_mean", "value": _safe_float(manager_row, "first_subgoal_hops_mean", 0.0)},
        {"stage": "manager", "ref_bin": str(manager_ref if manager_ref <= 3 else ">=4"), "metric": "segment_budget_error_mean", "value": _safe_float(manager_row, "segment_budget_error_mean", 0.0)},
        {"stage": "worker", "ref_bin": str(worker_ref if worker_ref <= 3 else ">=4"), "metric": "subgoal_rate", "value": _safe_float(worker_row, "subgoal_rate", 0.0)},
        {"stage": "worker", "ref_bin": str(worker_ref if worker_ref <= 3 else ">=4"), "metric": "goal_after_last_subgoal_rate", "value": _safe_float(worker_row, "goal_after_last_subgoal_rate", 0.0)},
        {"stage": "worker", "ref_bin": str(worker_ref if worker_ref <= 3 else ">=4"), "metric": "post_last_sg_success_rate", "value": _safe_float(worker_row, "post_last_sg_success_rate", 0.0)},
        {"stage": "worker", "ref_bin": str(worker_ref if worker_ref <= 3 else ">=4"), "metric": "stagnation_fail_rate", "value": _safe_float(worker_row, "stagnation_fail_rate", 0.0)},
        {"stage": "joint", "ref_bin": str(joint_ref if joint_ref <= 3 else ">=4"), "metric": "success_rate", "value": _safe_float(joint_row, "success_rate", 0.0)},
        {"stage": "joint", "ref_bin": str(joint_ref if joint_ref <= 3 else ">=4"), "metric": "far_short_plan", "value": _safe_float(joint_row, "far_plan_rate", 0.0)},
        {"stage": "joint", "ref_bin": str(joint_ref if joint_ref <= 3 else ">=4"), "metric": "anchor_hop_err_mean", "value": _safe_float(joint_row, "anchor_hop_err_mean", 0.0)},
    ])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _run_stage(args, stage_override=None):
    effective_stage = "joint" if args.disaster else (stage_override or args.stage)
    loaded_checkpoint_paths = []

    print("🚀 Starting Hierarchical POMO RL Training...")
    
    # 1. Load Meta Data
    # Checkpoints no longer needed (Full Path Strategy)
    checkpoints = None
    
    # 2. Setup Env
    print("Initializing Environment...")
    env = DisasterEnv(f'data/{args.map}_node.tntp', f'data/{args.map}_net.tntp', enable_disaster=args.disaster)
    num_nodes = env.map_core.graph.number_of_nodes() # or len(env.node_mapping)
    
    # 3. Load Models
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ GPU NOT DETECTED! Training will be slow on CPU.")
    
    print(f"Active Device: {device}")
    
    # num_checkpoints unused in model, pass num_nodes for clarity
    manager = GraphTransformerManager(
        node_dim=4, 
        hidden_dim=args.hidden_dim, 
        dropout=0.2
    ).to(device)
    
    worker = WorkerLSTM(
        node_dim=8,
        hidden_dim=args.hidden_dim
    ).to(device)

    def _load_state_compat(module, state_dict, module_name):
        current_state = module.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in current_state:
                skipped.append(f"{key}(missing)")
                continue
            if current_state[key].shape != value.shape:
                skipped.append(
                    f"{key}(shape {tuple(value.shape)} -> {tuple(current_state[key].shape)})"
                )
                continue
            compatible[key] = value
        module.load_state_dict(compatible, strict=False)
        if skipped:
            preview = ", ".join(skipped[:4])
            suffix = "..." if len(skipped) > 4 else ""
            print(f"⚠️ Partial {module_name} load: skipped {len(skipped)} keys [{preview}{suffix}]")

    def _load_unified_checkpoint(path, load_manager=True, load_worker=True):
        if not os.path.exists(path):
            return False, False
        checkpoint = torch.load(path, map_location=device)
        mgr_loaded = False
        wkr_loaded = False
        if load_manager and 'manager_state' in checkpoint:
            manager.load_state_dict(checkpoint['manager_state'])
            mgr_loaded = True
        if load_worker and 'worker_state' in checkpoint:
            _load_state_compat(worker, checkpoint['worker_state'], "worker")
            wkr_loaded = True
        print(f"📦 Loaded checkpoint from {path}")
        loaded_checkpoint_paths.append(path)
        return mgr_loaded, wkr_loaded
    
    loaded_mgr = False
    loaded_wkr = False
    
    if args.disaster:
        phase2_ckpt = os.path.join("logs", "rl_phase1_joint", "best.pt")
        loaded_mgr, loaded_wkr = _load_unified_checkpoint(phase2_ckpt)
    elif effective_stage == "manager":
        sl_ckpt = os.path.join("logs", "sl_pretrain", "model_sl_final.pt")
        loaded_mgr, loaded_wkr = _load_unified_checkpoint(sl_ckpt)
    elif effective_stage == "worker":
        mgr_ckpt = os.path.join("logs", "rl_phase1_manager", "best.pt")
        sl_ckpt = os.path.join("logs", "sl_pretrain", "model_sl_final.pt")
        if os.path.exists(mgr_ckpt):
            loaded_mgr, _ = _load_unified_checkpoint(mgr_ckpt, load_manager=True, load_worker=False)
        fallback_mgr, loaded_wkr = _load_unified_checkpoint(sl_ckpt)
        loaded_mgr = loaded_mgr or fallback_mgr
    else:
        mgr_ckpt = os.path.join("logs", "rl_phase1_manager", "best.pt")
        wkr_ckpt = os.path.join("logs", "rl_phase1_worker", "best.pt")
        if os.path.exists(mgr_ckpt):
            loaded_mgr, _ = _load_unified_checkpoint(mgr_ckpt, load_manager=True, load_worker=False)
        if os.path.exists(wkr_ckpt):
            _, loaded_wkr = _load_unified_checkpoint(wkr_ckpt, load_manager=False, load_worker=True)
        if not loaded_mgr or not loaded_wkr:
            sl_ckpt = os.path.join("logs", "sl_pretrain", "model_sl_final.pt")
            fallback_mgr, fallback_wkr = _load_unified_checkpoint(sl_ckpt)
            loaded_mgr = loaded_mgr or fallback_mgr
            loaded_wkr = loaded_wkr or fallback_wkr
    
    if not loaded_mgr:
        print("⚠️ Warning: Pre-trained Manager not found! Starting from scratch.")
    if not loaded_wkr:
        print("⚠️ Warning: Pre-trained Worker not found! Starting from scratch.")
        
    # 4. Config & Trainer
    if args.disaster:
        save_dir = "logs/rl_finetune_phase2"
    else:
        save_dir = {
            "manager": "logs/rl_phase1_manager",
            "worker": "logs/rl_phase1_worker",
            "joint": "logs/rl_phase1_joint",
        }[effective_stage]
    
    config = Config(
        lr=args.lr,
        num_pomo=args.num_pomo,
        episodes=args.episodes,
        save_dir=save_dir,
        stage=effective_stage,
        debug=args.debug,  # 디버그 모드 플래그 전달
        run_type="smoke" if args.episodes <= 5 else "train",
        parent_checkpoints=loaded_checkpoint_paths,
        mgr_lr_scale=0.5,
        mgr_eta_min_scale=0.1,
        mgr_max_grad_norm=10.0,
        wkr_max_grad_norm=5.0,
        mgr_aux_start=0.20,
        mgr_aux_end=0.05,
        wkr_aux_start=0.20,
        wkr_aux_end=0.05,
    )

    if args.disaster:
        trainer = DOMOTrainer(env, manager, worker, config)
    elif effective_stage == "manager":
        trainer = ManagerStageTrainer(env, manager, worker, config)
    elif effective_stage == "worker":
        trainer = WorkerStageTrainer(env, manager, worker, config)
    else:
        trainer = DOMOTrainer(env, manager, worker, config)
    
    # 5. Run
    trainer.train(args.episodes)

def train_rl(args):
    if args.stage == "phase1":
        if args.disaster:
            raise ValueError("--stage phase1 cannot be combined with --disaster.")

        stage_sequence = ["manager", "worker"]
        total_stages = len(stage_sequence)
        for idx, stage_name in enumerate(stage_sequence, start=1):
            print(f"\n{'=' * 72}")
            print(f"🔁 Phase 1 Pipeline [{idx}/{total_stages}]: {stage_name.upper()} stage")
            print(f"{'=' * 72}")
            stage_args = argparse.Namespace(**{**vars(args), "stage": stage_name, "disaster": False})
            _run_stage(stage_args, stage_override=stage_name)

        readiness = _build_phase1_readiness()
        _write_refbin_artifact()
        if not readiness["launch_ready"] and not args.force_joint:
            print("\n⏸️ Joint stage skipped because launch gate is not satisfied.")
            print(f"   smoke_ready={readiness['smoke_ready']} launch_ready={readiness['launch_ready']}")
            return

        print(f"\n{'=' * 72}")
        print("🔁 Phase 1 Pipeline [3/3]: JOINT stage")
        print(f"{'=' * 72}")
        if not readiness["launch_ready"] and args.force_joint:
            print("⚠️ launch_ready=False but continuing because --force_joint was provided.")
        stage_args = argparse.Namespace(**{**vars(args), "stage": "joint", "disaster": False})
        _run_stage(stage_args, stage_override="joint")
        _write_refbin_artifact()
        return

    if args.stage == "joint" and not args.disaster and not args.force_joint:
        readiness = _build_phase1_readiness()
        if not readiness["launch_ready"]:
            raise RuntimeError(
                "Joint stage launch gate is not satisfied. "
                "Use --force_joint to bypass for research smoke runs."
            )

    _run_stage(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='Anaheim')
    parser.add_argument('--data', default='data', help='Data Directory')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--num_pomo', type=int, default=8, help='POMO 병렬 시뮬레이션 수')
    parser.add_argument('--stage', choices=['manager', 'worker', 'joint', 'phase1'], default='joint', help='RL stage or full Phase 1 pipeline')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5) # Lower LR for Fine-tuning to prevent Catastrophic Forgetting
    parser.add_argument('--disaster', action='store_true', help='Enable disaster scenarios')
    parser.add_argument('--debug', action='store_true', help='디버그 모드: 200에피소드마다 보상 분해/Gradient/Entropy 상세 로그 출력')
    parser.add_argument('--force_joint', action='store_true', help='Launch joint stage even if readiness gate is not satisfied')
    
    args = parser.parse_args()
    
    # Fix Worker Init signature in loop above if needed
    # (Confirmed WorkerLSTM takes node_dim, hidden_dim)
    
    train_rl(args)
