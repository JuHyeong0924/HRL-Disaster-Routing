import argparse
import os
import warnings
from datetime import datetime

# lr_scheduler.step() 순서 경고 억제
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

import torch
import torch.backends.cudnn as cudnn

# [Hardcoded CPU Threads] 최적의 컨텍스트 스위칭 효율을 위해 프로세스당 8개 코어 할당
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

# [Speed Optimization] RTX 4090 (Ada) 및 고정된 입력 형태를 위한 하드웨어 극한 속도 튜닝
# 1. cuDNN Benchmark: 첫 스텝 수행 시 최적의 CUDA 커널을 찾아 고정
cudnn.benchmark = True
# 2. TF32 활성화: 행렬 곱셈 속도 최대 3배 폭증 (정밀도 손실 체감 불가)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# [Active Models & Trainers]
from src.models.worker import Worker
from src.trainers.worker_trainer import HRLWorkerTrainer  # [HRL Phase 1]

# [Manager v2] 비자기회귀 + PPO + PBRS Re-planning
from src.models.reactive_manager import ReactiveManager
from src.trainers.manager_ppo_trainer import ManagerPPOTrainer
from src.envs.hrl_closed_loop_env import HRLClosedLoopEnv
from src.envs.hrl_env import HRLZoneEnv


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_state_compat(module, state_dict, module_name):
    """체크포인트 호환성을 고려한 안전한 state_dict 로드.

    Why: 모델 아키텍처가 변경되어도 호환 가능한 키만 안전하게 로드하여
    기존 체크포인트를 최대한 재활용하기 위함.
    """
    current_state = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current_state:
            skipped.append(f"{key}(missing)")
            continue
        target_value = current_state[key]
        if target_value.shape != value.shape:
            skipped.append(
                f"{key}(shape {tuple(value.shape)} -> {tuple(target_value.shape)})"
            )
            continue
        compatible[key] = value.to(device=target_value.device, dtype=target_value.dtype)
    module.load_state_dict(compatible, strict=False)
    # Critic MLP 구조 변경 시 키 불일치 명시적 감지
    critic_skipped = [k for k in skipped if "critic" in k.split("(")[0]]
    if critic_skipped:
        print(
            f"🔄 [{module_name}] Critic architecture changed: "
            f"{len(critic_skipped)} old critic keys skipped. "
            "New Critic will be randomly initialized."
        )
    if skipped:
        preview = ", ".join(skipped[:4])
        suffix = "..." if len(skipped) > 4 else ""
        print(
            f"⚠️ Partial {module_name} load: skipped {len(skipped)} keys "
            f"[{preview}{suffix}]"
        )


def _extract_worker_state(payload):
    """체크포인트 payload에서 Worker state_dict를 추출."""
    if not isinstance(payload, dict):
        return payload
    if "worker_state" in payload:
        return payload["worker_state"]
    if "state_dict" in payload:
        return payload["state_dict"]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError("Could not find worker_state in checkpoint payload.")


def _extract_manager_state(payload):
    """체크포인트 payload에서 Manager state_dict를 추출."""
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
    payload = torch.load(path, map_location=device, weights_only=False)
    manager_state = _extract_manager_state(payload)
    _load_state_compat(manager, manager_state, "manager")
    print(f"📦 Loaded manager checkpoint from {path}")
    loaded_paths.append(path)
    return True


def _load_worker_checkpoint(path, worker, device, loaded_paths):
    if not os.path.exists(path):
        return False
    payload = torch.load(path, map_location=device, weights_only=False)
    worker_state = _extract_worker_state(payload)
    _load_state_compat(worker, worker_state, "worker")
    print(f"📦 Loaded worker checkpoint from {path}")
    loaded_paths.append(path)
    return True


def _build_config(args, loaded_checkpoint_paths):
    """CLI args로부터 Config 객체를 생성.

    Why: Trainer에 전달할 학습 설정을 일관된 형식으로 묶기 위함.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_label = f"{timestamp}_{args.stage}_B{args.batch_size}"
    stage_base = {
        'worker': os.path.join('logs', 'rl_worker_stage'),
        'manager_v2': os.path.join('logs', 'rl_manager_v2'),
    }.get(args.stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        num_pomo=args.batch_size,
        episodes=args.episodes,
        save_dir=save_dir,
        stage=args.stage,
        debug=args.debug,
        disable_tqdm=getattr(args, 'disable_tqdm', False),
        run_type="smoke" if args.episodes <= 5 else "train",
        parent_checkpoints=loaded_checkpoint_paths,
        max_steps=400,
        worker_temperature=1.0,
        wkr_lr_floor=getattr(args, "wkr_lr_floor", 1e-5),
        # [HRL Phase 1] Worker Ablation 플래그
        use_gae=getattr(args, "use_gae", False),
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        entropy_coeff=getattr(args, "entropy_coeff", 0.0),
        use_cosine_lr=getattr(args, "use_cosine_lr", False),
        zone_progress_reward=getattr(args, "zone_progress_reward", False),
    )


def _init_worker_env(args):
    """Phase 1 Worker 학습용 HRLZoneEnv 환경 초기화."""
    import glob
    # 맵별 Zone 파일 자동 탐색: data/node_to_zone_{map}_k*.json 우선, 없으면 기본 k30
    zone_json = 'data/node_to_zone_k30.json'
    zone_graph_json = 'data/zone_graph_k30.json'
    map_zone_files = glob.glob(f"data/node_to_zone_{args.map}_k*.json")
    if map_zone_files:
        zone_json = sorted(map_zone_files)[-1]  # 가장 큰 K 사용
        k_val = zone_json.split('_k')[-1].replace('.json', '')
        zone_graph_json = f"data/zone_graph_{args.map}_k{k_val}.json"
        print(f"   Zone 파일: {zone_json} (맵별 자동 탐색)")
    env = HRLZoneEnv(
        f"data/{args.map}_node.tntp",
        f"data/{args.map}_net.tntp",
        zone_json=zone_json,
        zone_graph_json=zone_graph_json,
        masking_mode=getattr(args, 'masking_mode', 'hard'),
        use_pbrs=getattr(args, 'use_pbrs', False),
        subgoal_mode=getattr(args, 'subgoal_mode', 'default'),
    )
    env.zone_progress_reward = getattr(args, 'zone_progress_reward', False)
    print(f"   Env: masking_mode={env.masking_mode}, use_pbrs={env.use_pbrs}, subgoal_mode={getattr(args, 'subgoal_mode', 'default')}")
    return env


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
                # best.pt가 없으면 final.pt라도 가져오도록 지원
                alt_file = os.path.join(latest_subdir, "final.pt")
                if os.path.exists(alt_file):
                    ckpt = alt_file
    return ckpt


def _run_worker_stage(args) -> None:
    """Phase 1: Worker 단독 학습 (HRLZoneEnv + HRLWorkerTrainer)."""
    print(f"\n{'='*60}")
    print(f"🚀 Stage [WORKER] 학습 시작 ({args.episodes} episodes)")
    print(f"{'='*60}")

    print("Initializing Environment...")
    env = _init_worker_env(args)

    device = _get_device()

    # Worker 생성
    num_layers = getattr(args, 'num_layers', 2)
    use_jk_net = getattr(args, 'use_jk_net', False)
    use_edge_attr = getattr(args, 'use_edge_attr', False)
    worker = Worker(
        node_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        use_checkpoint=True,  # Worker stage에서는 Gradient Checkpointing 활성
        use_jk_net=use_jk_net,
        use_edge_attr=use_edge_attr,
    ).to(device)
    print(f"   Worker: node_dim=4, hidden_dim={args.hidden_dim}, num_layers={num_layers}, use_jk_net={use_jk_net}, use_edge_attr={use_edge_attr}")

    # Worker는 scratch에서 시작
    print("📋 HRL Phase 1: Worker를 scratch에서 학습합니다.")

    loaded_checkpoint_paths = []
    config = _build_config(args, loaded_checkpoint_paths)

    # HRLWorkerTrainer에는 manager 인자가 필요하지만 실제 사용하지 않으므로 None 전달
    trainer = HRLWorkerTrainer(env, None, worker, config)
    trainer.train(args.episodes)

    print(f"\n✅ Stage [WORKER] 학습 완료!")
    print(f"   저장 위치: {config.save_dir}")


def _run_manager_v2_stage(args) -> None:
    """Phase 2: ReactiveManager PPO 학습 (Closed-loop + 동결된 Worker)."""
    print(f"\n{'='*60}")
    print(f"🚀 Stage [MANAGER_V2] 학습 시작 ({args.episodes} episodes)")
    print(f"{'='*60}")

    device = _get_device()
    loaded_checkpoint_paths = []

    # Worker 생성 및 체크포인트 로드
    num_layers = getattr(args, 'num_layers', 2)
    use_jk_net = getattr(args, 'use_jk_net', False)
    use_edge_attr = getattr(args, 'use_edge_attr', False)
    worker = Worker(
        node_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        use_checkpoint=False,  # Manager stage에서 Worker는 동결
        use_jk_net=use_jk_net,
        use_edge_attr=use_edge_attr,
    ).to(device)

    wkr_ckpt = _get_latest_ckpt(os.path.join('logs', 'rl_worker_stage'), 'best.pt')
    if not _load_worker_checkpoint(wkr_ckpt, worker, device, loaded_checkpoint_paths):
        print("⚠️ Worker 체크포인트 없음. Worker는 랜덤 초기 상태로 사용됩니다.")

    # ReactiveManager 생성
    node_dim = 4  # is_curr, is_tgt, hop_dist, degree
    reactive_mgr = ReactiveManager(
        node_dim=node_dim, hidden_dim=args.hidden_dim,
        num_layers=num_layers, dropout=0.2,
    ).to(device)
    print(f"📋 Manager v2: ReactiveManager (node_dim={node_dim}, hidden={args.hidden_dim})")
    print(f"   파라미터 수: {sum(p.numel() for p in reactive_mgr.parameters()):,}")

    # Closed-loop 환경 생성
    node_file = f"data/{args.map}_node.tntp"
    net_file = f"data/{args.map}_net.tntp"
    cl_env = HRLClosedLoopEnv(
        node_file=node_file, net_file=net_file,
        worker=worker, k_hop=5, c_max=8,
        device=str(device),
    )

    # PPO Trainer 생성 및 학습
    config = _build_config(args, loaded_checkpoint_paths)
    os.makedirs(config.save_dir, exist_ok=True)
    ppo_trainer = ManagerPPOTrainer(cl_env, reactive_mgr, config)
    ppo_trainer.train(args.episodes)

    print(f"\n✅ Stage [MANAGER_V2] 학습 완료!")
    print(f"   저장 위치: {config.save_dir}")


def _get_device() -> torch.device:
    """CUDA 디바이스 탐지 및 반환."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NOT DETECTED! Training will be slow on CPU.")
    print(f"Active Device: {device}")
    return device


def train_rl(args):
    """RL 학습 메인 라우터."""
    # --steps → --episodes 변환 (steps 우선)
    if args.steps is not None:
        args.episodes = args.steps * args.batch_size
        print(f"📐 배치={args.batch_size}, 스텝={args.steps} → 총 에피소드={args.episodes}")
    elif args.episodes is None:
        args.episodes = 5000  # 기본값
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")
    else:
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")

    if args.stage == "worker":
        _run_worker_stage(args)
    elif args.stage == "manager_v2":
        _run_manager_v2_stage(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRL-Disaster-Routing RL Training")
    parser.add_argument("--map", default="Anaheim")
    parser.add_argument("--data", default="data", help="Data Directory")
    parser.add_argument("--episodes", type=int, default=None,
                        help="총 에피소드 수 (--steps 미지정 시 사용, 기본 5000)")
    parser.add_argument("--steps", type=int, default=None,
                        help="총 gradient 업데이트 스텝 수 (지정 시 --episodes보다 우선)")
    parser.add_argument("--batch_size", "--num_pomo", type=int, default=16,
                        help="배치 크기: 스텝당 동시 실행 에피소드 수 (기본 16)")
    parser.add_argument(
        "--stage",
        default="worker",
        choices=["worker", "manager_v2"],
        help="학습 단계: worker(Phase 1), manager_v2(Phase 2 비자기회귀+PPO)",
    )
    parser.add_argument("--wkr_lr_floor", type=float, default=1e-5,
                        help="Worker 최소 학습률 (기본 1e-5)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드: 주기적으로 단계별 지표를 상세 로그로 출력",
    )
    parser.add_argument("--disable_tqdm", action="store_true", help="내부 tqdm 비활성화")
    # [HRL Phase 1] Worker 학습 플래그
    parser.add_argument("--zone_progress_reward", action="store_true",
                        help="Zone 전환 시 중간 보상 부여")
    parser.add_argument("--use_gae", action="store_true",
                        help="GAE(λ) Advantage 사용")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE λ 파라미터 (기본 0.95)")
    parser.add_argument("--entropy_coeff", type=float, default=0.0,
                        help="Entropy Bonus 계수 (0이면 비활성)")
    parser.add_argument("--use_cosine_lr", action="store_true",
                        help="Cosine LR Scheduler 사용")
    # [Worker 환경/모델 제어]
    parser.add_argument("--masking_mode", type=str, default="hard",
                        choices=["hard", "hard_full_seq", "soft_curr_next", "soft_flex"],
                        help="Action Masking 모드 (hard/hard_full_seq/soft_curr_next/soft_flex)")
    parser.add_argument("--use_pbrs", action="store_true",
                        help="hop_dist 기반 PBRS Dense Reward 활성")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="GATv2 레이어 수 (1~4)")
    parser.add_argument("--use_jk_net", action="store_true",
                        help="JK-Net (Jumping Knowledge) 활성")
    parser.add_argument("--use_edge_attr", action="store_true",
                        help="Edge-Conditioned MP (Edge Features) 활성")
    parser.add_argument("--subgoal_mode", type=str, default="zone",
                        choices=["zone", "node"],
                        help="Manager Subgoal 모드 (zone / node)")

    args = parser.parse_args()
    train_rl(args)
