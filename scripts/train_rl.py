import argparse
import os
import sys
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.envs.worker_env import WorkerEnv




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
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    exp_suffix = f"_{args.exp_name}" if getattr(args, 'exp_name', None) else ""
    run_label = f"{timestamp}_{args.stage}{exp_suffix}"
    stage_base = {
        'worker': os.path.join('logs', 'rl_worker_stage'),
        'manager': os.path.join('logs', 'rl_manager_stage'),
    }.get(args.stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=getattr(args, "mini_batch_size", 256),
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


def _init_worker_env(args, device='cpu'):
    """Phase 1 Worker 학습용 HRLZoneEnv 환경 초기화."""
    zone_json = f'data/grid_{args.map}_node_to_zone.json'
    zone_graph_json = f'data/grid_{args.map}_zone_graph.json'
    print(f"   Zone 파일: {zone_json} (Grid 분할)")
    env = WorkerEnv(
        f"data/{args.map}_node.tntp",
        f"data/{args.map}_net.tntp",
        zone_json=zone_json,
        zone_graph_json=zone_graph_json,
        masking_mode=getattr(args, 'masking_mode', 'soft_flex'),
        subgoal_mode=getattr(args, 'subgoal_mode', 'zone'),
        oob_penalty=getattr(args, 'oob_penalty', -1.0),
        device=device
    )
    env.zone_progress_reward = getattr(args, 'zone_progress_reward', False)
    env.disaster_prob = getattr(args, 'disaster_prob', 0.0)
    env.dynamic_disaster = getattr(args, 'dynamic_disaster', False)
    print(f"   Env: masking_mode={env.masking_mode}, subgoal_mode={getattr(args, 'subgoal_mode', 'zone')}")
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

    device = _get_device()

    print("Initializing Environment...")
    env = _init_worker_env(args, device=str(device))

    # Worker 생성
    num_layers = getattr(args, 'num_layers', 2)
    use_jk_net = getattr(args, 'use_jk_net', False)
    node_dim = 7
    worker = Worker(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_checkpoint=False,
        use_jk_net=use_jk_net,
    ).to(device)
    print(f"   Worker: node_dim={node_dim}, hidden_dim={args.hidden_dim}, num_layers={num_layers}")

    loaded_checkpoint_paths = []
    
    if args.worker_ckpt and os.path.exists(args.worker_ckpt):
        print(f"📦 HRL Phase 1: 기존 Worker 체크포인트에서 이어서 학습합니다 ({args.worker_ckpt})")
        payload = torch.load(args.worker_ckpt, map_location=device, weights_only=False)
        if "worker_state" in payload:
            worker.load_state_dict(payload["worker_state"])
        else:
            worker.load_state_dict(payload)
        loaded_checkpoint_paths.append(args.worker_ckpt)
    else:
        print("📋 HRL Phase 1: Worker를 scratch에서 학습합니다.")

    config = _build_config(args, loaded_checkpoint_paths)

    # HRLWorkerTrainer에는 manager 인자가 필요하지만 실제 사용하지 않으므로 None 전달
    trainer = HRLWorkerTrainer(env, None, worker, config)
    trainer.train(args.episodes)

    print(f"\n✅ Stage [WORKER] 학습 완료!")
    print(f"   저장 위치: {config.save_dir}")


def _run_manager_stage(args) -> None:
    """Phase 2: Manager PPO 학습 (Closed-loop + 동결된 Worker)."""
    print(f"\n{'='*60}")
    print(f"🚀 Stage [MANAGER] 학습 시작 ({args.episodes} episodes)")
    print(f"{'='*60}")

    from src.models.manager import Manager
    from src.trainers.manager_trainer import ManagerTrainer
    from src.envs.hrl_env import HRLEnv
    import random
    from tqdm import tqdm

    device = _get_device()
    loaded_checkpoint_paths = []

    # Worker 생성 및 체크포인트 로드
    num_layers = getattr(args, 'num_layers', 2)
    use_jk_net = getattr(args, 'use_jk_net', False)
    node_dim = 7
    worker = Worker(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_checkpoint=False,
        use_jk_net=use_jk_net,
    ).to(device)

    wkr_ckpt = args.worker_ckpt if getattr(args, 'worker_ckpt', None) else _get_latest_ckpt(os.path.join('logs', 'rl_worker_stage'), 'best.pt')
    if not _load_worker_checkpoint(wkr_ckpt, worker, device, loaded_checkpoint_paths):
        print("⚠️ Worker 체크포인트 없음. Worker는 랜덤 초기 상태로 사용됩니다.")
    worker.eval()

    # Init WorkerEnv
    worker_env = _init_worker_env(args, device=str(device))

    # Init HRLEnv
    hrl_env = HRLEnv(worker, worker_env)

    # Init Manager
    manager_model = Manager(
        zone_dim=7, target_dim=6, hidden_dim=256,  # [Phase 2B] target_dim 4→6
        num_gat_layers=3, gat_heads=4, num_transformer_layers=3, transformer_heads=4
    ).to(device)
    print(f"📋 Manager: (hidden=256)")
    print(f"   파라미터 수: {sum(p.numel() for p in manager_model.parameters()):,}")

    # Init Trainer
    config = _build_config(args, loaded_checkpoint_paths)
    os.makedirs(config.save_dir, exist_ok=True)
    trainer = ManagerTrainer(manager_model, hrl_env, config=args)

    best_reward = -float('inf')
    num_steps = (args.episodes + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=args.episodes, desc="Manager PPO", ncols=200, unit="ep") as pbar:
        for step in range(1, num_steps + 1):
            current_ep = step * args.batch_size
            
            # [Phase 2E] Curriculum Learning (5 Phases, 비율 기반)
            # 10% : 20% : 20% : 20% : 30% 비율로 전환
            if current_ep <= int(args.episodes * 0.10):
                # Phase 1: 단일 타겟, 무재해
                current_num_targets = 1
                hrl_env.env.disaster_prob = 0.0
                hrl_env.env.dynamic_disaster = False
                phase_str = 'P1:Single'
            elif current_ep <= int(args.episodes * 0.30):
                # Phase 2: 다중 타겟, 무재해
                current_num_targets = random.randint(3, 7)
                hrl_env.env.disaster_prob = 0.0
                hrl_env.env.dynamic_disaster = False
                phase_str = 'P2:Multi'
            elif current_ep <= int(args.episodes * 0.50):
                # Phase 3: 다중 타겟, 정적 재해 (HAZUS 가중치)
                current_num_targets = random.randint(5, 10)
                hrl_env.env.disaster_prob = 0.15
                hrl_env.env.dynamic_disaster = False
                phase_str = 'P3:Static'
            elif current_ep <= int(args.episodes * 0.70):
                # Phase 4: 다수 타겟, 동적 재해 (Continuous Aftershock)
                current_num_targets = random.randint(5, 12)
                hrl_env.env.disaster_prob = 0.15
                hrl_env.env.dynamic_disaster = True
                phase_str = 'P4:Dynamic'
            else:
                # Phase 5: 전체 범위, 강력한 동적 재해
                current_num_targets = random.randint(5, 15)
                hrl_env.env.disaster_prob = 0.2
                hrl_env.env.dynamic_disaster = True
                phase_str = 'P5:Full'
                
            # 동적 엔트로피 감가 적용 (0.05 -> 0.01)
            current_ent_coef = max(0.01, 0.05 - 0.04 * (step / num_steps))
            
            logs = trainer.train_step(
                batch_size=args.batch_size, 
                num_targets=current_num_targets,
                current_ent_coef=current_ent_coef
            )
            
            sr = (logs['mean_rescued'] / current_num_targets) * 100.0 if current_num_targets > 0 else 0.0
            
            pbar.set_postfix({
                'Phase': phase_str,
                'Loss': f"{logs.get('loss', 0):.4f}",
                'P': f"{logs.get('policy_loss', 0):.4f}",
                'V': f"{logs.get('value_loss', 0):.4f}",
                'Rw': f"{logs['mean_reward']:.2f}",
                'Rsc': f"{logs['mean_rescued']:.1f}/{current_num_targets}",
                'SR': f"{sr:.1f}%",
            })
            pbar.update(args.batch_size)
            
            if logs['mean_reward'] > best_reward:
                best_reward = logs['mean_reward']
                torch.save(manager_model.state_dict(), os.path.join(config.save_dir, 'best_manager.pt'))
                tqdm.write(f"  => Best model saved (Reward: {best_reward:.2f})")
                
    print(f"\n✅ Stage [MANAGER] 학습 완료! Best Reward: {best_reward:.2f}")
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
        args.episodes = 20000  # 기본값
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")
    else:
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")

    if args.stage == "worker":
        _run_worker_stage(args)
    elif args.stage == "manager":
        _run_manager_stage(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRL-Disaster-Routing RL Training")
    parser.add_argument("--map", default="Anaheim")
    parser.add_argument("--data", default="data", help="Data Directory")
    parser.add_argument("--episodes", type=int, default=30000,
                        help="총 에피소드 수 (--steps 미지정 시 사용, 기본 30000)")
    parser.add_argument("--steps", type=int, default=None,
                        help="총 gradient 업데이트 스텝 수 (지정 시 --episodes보다 우선)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="배치 크기 (Worker 기본: 32, Manager 기본: 256)")
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="PPO 미니배치 크기 (Worker 기본: 192, Manager 기본: 2048)")
    parser.add_argument(
        "--stage",
        type=str,
        default="worker",
        choices=["worker", "manager", "all"],
        help="학습 단계: worker(Phase 1), manager(Phase 2 비자기회귀+PPO)",
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
    parser.add_argument("--use_gae", action="store_true", default=True,
                        help="GAE(λ) Advantage 사용")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE λ 파라미터 (기본 0.95)")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy Bonus 계수 (0이면 비활성)")
    parser.add_argument("--use_cosine_lr", action="store_true", default=True,
                        help="Cosine LR Scheduler 사용")
    # [Worker 환경/모델 제어]
    parser.add_argument("--masking_mode", type=str, default="soft_curr_next",
                        choices=["hard", "hard_full_seq", "soft_curr_next", "soft_flex"],
                        help="Action Masking 모드 (hard/hard_full_seq/soft_curr_next/soft_flex)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="GATv2 레이어 수 (1~4)")
    parser.add_argument("--use_jk_net", action="store_true",
                        help="JK-Net (Jumping Knowledge) 활성")
    parser.add_argument("--subgoal_mode", type=str, default="zone",
                        choices=["zone", "node"],
                        help="Manager Subgoal 모드 (zone / node)")
    
    parser.add_argument("--oob_penalty", type=float, default=-1.0)

    # [Manager-Specific Settings] 추가
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for logging directory")
    parser.add_argument("--worker_ckpt", type=str, default=None, help="Path to specific worker checkpoint")
    
    # [Phase 1 Stage 2,3 Disaster Settings]
    parser.add_argument("--disaster_prob", type=float, default=0.0, help="에피소드 내 재난 발생 확률 (Stage 2)")
    parser.add_argument("--dynamic_disaster", action="store_true", help="에피소드 진행 중 동적 재난 발생 활성화 (Stage 3)")

    args = parser.parse_args()
    
    if args.batch_size is None:
        args.batch_size = 32 if args.stage == "worker" else 256
    if args.mini_batch_size is None:
        args.mini_batch_size = 192 if args.stage == "worker" else 1024

    train_rl(args)
