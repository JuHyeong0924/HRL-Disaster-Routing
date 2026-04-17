import argparse
import os
from datetime import datetime

import torch

from src.envs.disaster_env import DisasterEnv
from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM
from src.trainers.worker_nav_trainer import WorkerNavTrainer
from src.trainers.manager_stage_trainer import ManagerStageTrainer  # [Refactor: Task 1]
from src.trainers.worker_stage_trainer import WorkerStageTrainer  # [Refactor: Task 1]
from src.trainers.pomo_trainer import DOMOTrainer  # [Refactor: Task 1]


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_state_compat(module, state_dict, module_name):
    current_state = module.state_dict()
    compatible = {}
    skipped = []
    adapted = []
    worker_input_adapt_keys = {
        "convs.0.lin_l.weight",
        "convs.0.lin_r.weight",
        "input_proj.weight",
    }
    for key, value in state_dict.items():
        if key not in current_state:
            skipped.append(f"{key}(missing)")
            continue
        target_value = current_state[key]
        if target_value.shape != value.shape:
            if (
                module_name == "worker"
                and key in worker_input_adapt_keys
                and value.ndim == 2
                and target_value.ndim == 2
                and value.shape[0] == target_value.shape[0]
                and value.shape[1] == 7
                and target_value.shape[1] == 8
            ):
                padded = target_value.clone()
                padded.zero_()
                padded[:, :7] = value.to(device=target_value.device, dtype=target_value.dtype)
                compatible[key] = padded
                adapted.append(key)
                continue
            skipped.append(
                f"{key}(shape {tuple(value.shape)} -> {tuple(target_value.shape)})"
            )
            continue
        compatible[key] = value.to(device=target_value.device, dtype=target_value.dtype)
    module.load_state_dict(compatible, strict=False)
    if adapted:
        print(
            "🔁 Adapted legacy worker checkpoint from 7-dim input to 8-dim input "
            f"for {len(adapted)} input-layer weights."
        )
    # [Refactor: Task 2] Critic 2-Layer MLP 변경 시 키 불일치 명시적 감지
    critic_skipped = [k for k in skipped if "critic" in k.split("(")[0]]
    if critic_skipped:
        print(
            f"🔄 [{module_name}] Critic architecture changed: "
            f"{len(critic_skipped)} old critic keys skipped. "
            "New 2-Layer MLP Critic will be randomly initialized."
        )
    if skipped:
        preview = ", ".join(skipped[:4])
        suffix = "..." if len(skipped) > 4 else ""
        print(
            f"⚠️ Partial {module_name} load: skipped {len(skipped)} keys "
            f"[{preview}{suffix}]"
        )


def _extract_worker_state(payload):
    if not isinstance(payload, dict):
        return payload
    if "worker_state" in payload:
        return payload["worker_state"]
    if "state_dict" in payload:
        return payload["state_dict"]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError("Could not find worker_state in checkpoint payload.")


# [Refactor: Task 1] Manager 체크포인트 로드 유틸리티
def _extract_manager_state(payload):
    if not isinstance(payload, dict):
        return payload
    if "manager_state" in payload:
        return payload["manager_state"]
    if "state_dict" in payload:
        return payload["state_dict"]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError("Could not find manager_state in checkpoint payload.")


def _load_manager_checkpoint(path, manager, device, loaded_paths):
    if not os.path.exists(path):
        return False
    payload = torch.load(path, map_location=device)
    manager_state = _extract_manager_state(payload)
    _load_state_compat(manager, manager_state, "manager")
    print(f"📦 Loaded manager checkpoint from {path}")
    loaded_paths.append(path)
    return True


def _load_worker_checkpoint(path, worker, device, loaded_paths):
    if not os.path.exists(path):
        return False
    payload = torch.load(path, map_location=device)
    worker_state = _extract_worker_state(payload)
    _load_state_compat(worker, worker_state, "worker")
    print(f"📦 Loaded worker checkpoint from {path}")
    loaded_paths.append(path)
    return True


def _build_config(args, loaded_checkpoint_paths, stage_override=None):
    # stage_override: phase1 순차 실행 시 각 단계별 stage 지정용
    effective_stage = stage_override or args.stage
    # 타임스탬프 서브폴더 생성: logs/<stage>/<YYYY-MM-DD_HHMM>_<stage>_pomo<N>/
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_label = f"{timestamp}_{effective_stage}_pomo{args.num_pomo}"
    stage_base = {
        'manager': os.path.join('logs', 'rl_manager_stage'),
        'worker': os.path.join('logs', 'rl_worker_stage'),
        'joint': os.path.join('logs', 'rl_joint_stage'),
    }.get(effective_stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        num_pomo=args.num_pomo,
        episodes=args.episodes,
        save_dir=save_dir,
        stage=effective_stage,
        debug=args.debug,
        run_type="smoke" if args.episodes <= 5 else "train",
        parent_checkpoints=loaded_checkpoint_paths,
        mgr_max_grad_norm=20.0,  # [Fix] Manager Stage(20.0)과 동일하게 맞춤 (기본값 5.0은 100% clip-hit 유발)
        max_steps=400,
        worker_temperature=1.0,
        target_segment_hops=6.0,
        min_hops_for_hidden_checkpoint=4,
        two_hidden_checkpoint_min_hops=10,
        checkpoint_hit_radius=1,
        hidden_bonus_start=2.0,
        hidden_bonus_mid=1.5,
        hidden_bonus_end=0.75,
        guidance_schedule_ep_1=400,
        guidance_schedule_ep_2=800,
        total_hidden_bonus_cap=10.0,
        wkr_aux_start=0.20,
        wkr_aux_mid=0.17,
        wkr_aux_end=0.12,
        wkr_lr_floor=getattr(args, "wkr_lr_floor", 1e-5),  # [Refactor: Task 5] 최소 학습률 상향
        goal_hop_bonus_8=0.75,
        goal_hop_bonus_4=1.0,
        goal_hop_bonus_2=1.25,
        goal_neighbor_action_bonus=1.0,
        goal_neighbor_miss_penalty=0.35,
        near_goal_ce_mult=1.75,
        terminal_entropy_mult=0.5,
        goal_regression_penalty_small=0.15,
        goal_regression_penalty_large=0.35,
        near_goal_patience_bonus=8,
        loop_limit=6,
        stagnation_patience=24,
    )


def _init_env_and_models(args):
    """환경, 모델, 디바이스 초기화 (공용)."""
    print("Initializing Environment...")
    env = DisasterEnv(
        f"data/{args.map}_node.tntp",
        f"data/{args.map}_net.tntp",
        enable_disaster=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NOT DETECTED! Training will be slow on CPU.")
    print(f"Active Device: {device}")

    # Phase 1: edge_dim=1 (length만 사용. A*/APSP 학습 신호와 일치하는 유일한 엣지 피처)
    manager = GraphTransformerManager(node_dim=4, hidden_dim=args.hidden_dim, dropout=0.2, edge_dim=1).to(device)
    worker = WorkerLSTM(node_dim=8, hidden_dim=args.hidden_dim, edge_dim=1).to(device)

    # 배치 크기 고정 적용 (VRAM 스케일러 제거됨)
    args.num_pomo = int(args.num_pomo) if str(args.num_pomo).isdigit() else 32
    print(f"📐 고정된 배치 크기(POMO): {args.num_pomo}")

    return env, manager, worker, device


def _get_latest_ckpt(base_path: str, fallback_name: str) -> str:
    """base_path 내 최신 서브디렉토리의 fallback_name 체크포인트 경로 반환."""
    ckpt = os.path.join(base_path, fallback_name)
    if os.path.exists(base_path):
        subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            target_file = os.path.join(latest_subdir, fallback_name)
            if os.path.exists(target_file):
                ckpt = target_file
            else:
                # [Fix] best.pt가 없으면 final.pt라도 가져오도록 지원
                alt_file = os.path.join(latest_subdir, "final.pt")
                if os.path.exists(alt_file):
                    ckpt = alt_file
    return ckpt


def _run_single_stage(args, env, manager, worker, device, stage: str,
                      episodes: int) -> None:
    """단일 stage 학습 실행."""
    print(f"\n{'='*60}")
    print(f"🚀 Stage [{stage.upper()}] 학습 시작 ({episodes} episodes)")
    print(f"{'='*60}")

    loaded_checkpoint_paths = []
    sl_ckpt = _get_latest_ckpt(os.path.join("logs", "sl_pretrain"), "model_sl_final.pt")
    mgr_stage_ckpt = _get_latest_ckpt(os.path.join("logs", "rl_manager_stage"), "best.pt")
    wkr_stage_ckpt = _get_latest_ckpt(os.path.join("logs", "rl_worker_stage"), "best.pt")

    if stage == "worker":
        # SL pretrained worker/manager 로드
        if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
            print("⚠️ SL worker checkpoint not found. Starting from scratch.")
        if not _load_manager_checkpoint(sl_ckpt, manager, device, loaded_checkpoint_paths):
            print("⚠️ SL manager checkpoint not found.")
    elif stage == "manager":
        # Worker: worker_stage best → fallback SL
        if not _load_worker_checkpoint(wkr_stage_ckpt, worker, device, loaded_checkpoint_paths):
            if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
                print("⚠️ No worker checkpoint for manager stage.")
        # Manager: SL pretrained로 시작
        if not _load_manager_checkpoint(sl_ckpt, manager, device, loaded_checkpoint_paths):
            print("⚠️ SL manager checkpoint not found.")
    elif stage == "joint":
        # Worker: worker_stage best → fallback SL
        if not _load_worker_checkpoint(wkr_stage_ckpt, worker, device, loaded_checkpoint_paths):
            if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
                print("⚠️ No worker checkpoint for joint stage.")
        # Manager: manager_stage best → fallback SL
        if not _load_manager_checkpoint(mgr_stage_ckpt, manager, device, loaded_checkpoint_paths):
            if not _load_manager_checkpoint(sl_ckpt, manager, device, loaded_checkpoint_paths):
                print("⚠️ No manager checkpoint for joint stage.")

    config = _build_config(args, loaded_checkpoint_paths, stage_override=stage)

    # Stage별 Trainer 분기
    if stage == "manager":
        trainer = ManagerStageTrainer(env, manager, worker, config)
    elif stage == "worker":
        # WorkerNavTrainer를 Worker stage의 기본 Trainer로 사용
        trainer = WorkerNavTrainer(env, manager, worker, config)
    elif stage == "joint":
        trainer = DOMOTrainer(env, manager, worker, config)
    else:
        trainer = WorkerNavTrainer(env, manager, worker, config)

    trainer.train(episodes)
    print(f"\n✅ Stage [{stage.upper()}] 학습 완료!")
    print(f"   저장 위치: {config.save_dir}")


def train_rl(args):
    if args.disaster:
        raise ValueError(
            "--disaster is legacy / unsupported. "
            "Use --stage phase1 for worker→manager→joint pipeline."
        )

    env, manager, worker, device = _init_env_and_models(args)

    if args.stage == "phase1":
        # Phase 1: worker → manager → joint 자동 순차 실행
        # 각 단계별로 입력받은 episodes만큼 독립적으로 실행 (총 3 * episodes)
        worker_eps = args.episodes
        manager_eps = args.episodes
        joint_eps = args.episodes

        print(f"\n🎯 Phase 1 자동 순차 실행: 각 Stage별 {args.episodes} episodes (총 {args.episodes * 3} eps)")
        print(f"   1. Worker:  {worker_eps} eps → logs/rl_worker_stage/")
        print(f"   2. Manager: {manager_eps} eps → logs/rl_manager_stage/")
        print(f"   3. Joint:   {joint_eps} eps → logs/rl_joint_stage/")

        _run_single_stage(args, env, manager, worker, device, "worker", worker_eps)
        _run_single_stage(args, env, manager, worker, device, "manager", manager_eps)
        _run_single_stage(args, env, manager, worker, device, "joint", joint_eps)

        print(f"\n{'='*60}")
        print("🎉 Phase 1 전체 학습 완료!")
        print(f"{'='*60}")
    else:
        # 단일 stage 실행
        _run_single_stage(args, env, manager, worker, device, args.stage, args.episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="Anaheim")
    parser.add_argument("--data", default="data", help="Data Directory")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--num_pomo", default="auto",
                        help="POMO 병렬 시뮬레이션 수 (auto: VRAM 한도 내 최대 자동 탐색)")
    parser.add_argument(
        "--stage",
        default="phase1",
        choices=["manager", "worker", "joint", "phase1"],
        help="학습 단계: manager→worker→joint 커리큘럼 또는 phase1(APTE)",
    )
    parser.add_argument("--wkr_lr_floor", type=float, default=1e-5,
                        help="Worker 최소 학습률 (기본 1e-5)")  # [Refactor: Task 5]
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--disaster",
        action="store_true",
        help="Legacy flag. Unsupported in APTE branch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드: 주기적으로 APTE phase1 지표를 상세 로그로 출력",
    )
    parser.add_argument(
        "--force_joint",
        action="store_true",
        help="Legacy flag. Unsupported in APTE branch.",
    )
    train_rl(parser.parse_args())
