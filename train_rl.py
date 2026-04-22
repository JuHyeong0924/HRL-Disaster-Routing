import argparse
import os
import warnings
from datetime import datetime

# lr_scheduler.step() 순서 경고 억제
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

# [Speed Optimization] RTX 4090 (Ada) 및 고정된 입력 형태를 위한 하드웨어 극한 속도 튜닝
# 1. cuDNN Benchmark: 첫 스텝 수행 시 최적의 CUDA 커널을 찾아 고정
cudnn.benchmark = True
# 2. TF32 활성화: 행렬 곱셈 속도 최대 3배 폭증 (정밀도 손실 체감 불가)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.envs.disaster_env import DisasterEnv
from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM
from src.trainers.worker_nav_trainer import WorkerNavTrainer
from src.trainers.manager_stage_trainer import ManagerStageTrainer  # [Refactor: Task 1]
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
                and value.shape[1] < target_value.shape[1]
            ):
                # 레거시 체크포인트 적응 (7dim→8dim→9dim)
                old_dim = value.shape[1]
                new_dim = target_value.shape[1]
                padded = target_value.clone()
                padded.zero_()
                padded[:, :old_dim] = value.to(device=target_value.device, dtype=target_value.dtype)
                compatible[key] = padded
                adapted.append(f"{key}({old_dim}→{new_dim})")
                continue
            skipped.append(
                f"{key}(shape {tuple(value.shape)} -> {tuple(target_value.shape)})"
            )
            continue
        compatible[key] = value.to(device=target_value.device, dtype=target_value.dtype)
    module.load_state_dict(compatible, strict=False)
    if adapted:
        print(
            f"🔁 Adapted legacy worker checkpoint input dims "
            f"for {len(adapted)} weights: {', '.join(adapted[:3])}"
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
        'alignment': os.path.join('logs', 'rl_alignment_stage'),
    }.get(effective_stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        num_pomo=args.num_pomo,
        episodes=args.episodes,
        save_dir=save_dir,
        stage=effective_stage,
        debug=args.debug,
        disable_tqdm=getattr(args, 'disable_tqdm', False),
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

    # edge_dim=3: [length, capacity, speed] → 인덱스 [0, 7, 8]
    manager = GraphTransformerManager(node_dim=4, hidden_dim=args.hidden_dim, dropout=0.2, edge_dim=3).to(device)
    worker = WorkerLSTM(node_dim=9, hidden_dim=args.hidden_dim, edge_dim=3).to(device)

    # 단계별 배치 크기 (개별 학습은 64, Joint는 OOM 방지를 위해 절반으로 고정)
    raw_pomo = str(args.num_pomo)
    base_pomo = int(raw_pomo) if raw_pomo.isdigit() and raw_pomo != "auto" else 64
    
    # [Design 2026-04-21] Joint 단계에서도 Worker 동결으로 POMO 축소 불필요
    args.num_pomo = int(base_pomo)
        
    print(f"📐 고정된 배치 크기(POMO): {args.num_pomo} (Stage: {args.stage})")
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
    elif stage == "alignment":
        # Worker: worker_stage best → fallback SL
        if not _load_worker_checkpoint(wkr_stage_ckpt, worker, device, loaded_checkpoint_paths):
            if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
                print("⚠️ No worker checkpoint for alignment stage.")
        # Manager: manager_stage best → fallback SL
        if not _load_manager_checkpoint(mgr_stage_ckpt, manager, device, loaded_checkpoint_paths):
            if not _load_manager_checkpoint(sl_ckpt, manager, device, loaded_checkpoint_paths):
                print("⚠️ No manager checkpoint for alignment stage.")

    config = _build_config(args, loaded_checkpoint_paths, stage_override=stage)

    # Stage별 Trainer 분기
    if stage == "manager":
        trainer = ManagerStageTrainer(env, manager, worker, config)
    elif stage == "worker":
        # WorkerNavTrainer를 Worker stage의 기본 Trainer로 사용
        trainer = WorkerNavTrainer(env, manager, worker, config)
    elif stage == "alignment":
        trainer = DOMOTrainer(env, manager, worker, config)
    else:
        trainer = WorkerNavTrainer(env, manager, worker, config)

    trainer.train(episodes)
    print(f"\n✅ Stage [{stage.upper()}] 학습 완료!")
    print(f"   저장 위치: {config.save_dir}")


def _run_parallel_phase1(args) -> None:
    """Worker(GPU 0) + Manager(GPU 1) 병렬 → Joint 순차 실행.

    Why: Worker와 Manager는 완전 독립이므로 각 GPU에서 동시 학습 가능.
         Manager는 학습 난이도가 높으므로 에피소드를 2배로 설정.
    """
    import subprocess
    import sys

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        print("⚠️ GPU가 2개 미만입니다. phase1(순차)로 대체합니다.")
        args.stage = "phase1"
        train_rl(args)
        return

    worker_eps = args.episodes
    manager_eps = args.episodes * 4  # Manager는 4배 에피소드 (Worker보다 ~7.6배 빠르므로 GPU 유휴 최소화)
    alignment_eps = args.episodes

    print(f"\n{'='*60}")
    print("🔀 Phase 1 Parallel: Worker(GPU 0) ∥ Manager(GPU 1) → Alignment")
    print(f"{'='*60}")
    print(f"  Worker:     {worker_eps:,} eps on GPU 0")
    print(f"  Manager:    {manager_eps:,} eps on GPU 1")
    print(f"  Alignment:  {alignment_eps:,} eps (완료 후)")
    print()

    # 공통 인자 구성
    base_args = [
        sys.executable, "train_rl.py",
        "--map", args.map,
        "--data", args.data,
        "--hidden_dim", str(args.hidden_dim),
        "--lr", str(args.lr),
    ]
    if args.debug:
        base_args.append("--debug")

    # Worker subprocess (GPU 0)
    worker_cmd = base_args + [
        "--stage", "worker",
        "--episodes", str(worker_eps),
        "--num_pomo", str(args.num_pomo),
        "--disable_tqdm",
    ]
    worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    # Manager subprocess (GPU 1)
    # [Fix] 사용자 요청: Manager의 POMO 크기를 1.5배(예: 48 * 1.5 = 72)로 상향 조정하여 VRAM을 적절히 활용
    if args.num_pomo == "auto":
        manager_pomo = "auto"
    else:
        manager_pomo = int(float(args.num_pomo) * 1.5)
    manager_cmd = base_args + [
        "--stage", "manager",
        "--episodes", str(manager_eps),
        "--num_pomo", str(manager_pomo),
        "--disable_tqdm",
    ]
    manager_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

    print(f"🚀 Worker  시작 (GPU 0, {worker_eps:,} eps)...")
    print(f"🚀 Manager 시작 (GPU 1, {manager_eps:,} eps)...")
    print("\n" * 2) # reserve lines for tqdm

    os.makedirs("logs", exist_ok=True)

    # 두 프로세스 동시 실행. stdout은 터미널/파일 오염을 막기 위해 폐기하고, stderr은 파이프로 연결하여 단위 진행상태 수신
    worker_proc = subprocess.Popen(
        worker_cmd, env=worker_env,
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    manager_proc = subprocess.Popen(
        manager_cmd, env=manager_env,
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    import threading
    from tqdm.auto import tqdm

    def monitor_progress(task_name, total_steps, pos_id, proc):
        # dynamic_ncols=True는 터미널 리사이즈 시 깨짐을 방지하지만, 
        # 다중 스레드에서는 ncols를 고정하는 것이 화면 깨짐 방지에 더 안정적일 수 있습니다.
        pbar = tqdm(total=total_steps, desc=task_name, position=pos_id, leave=True, ncols=130)
        
        for line in iter(proc.stderr.readline, ''):
            if not line:
                break
                
            clean_line = line.strip()
            
            # 2. 약속된 프로토콜만 UI 업데이트에 반영
            if clean_line.startswith("PROGRESS_UPDATE|"):
                try:
                    parts = clean_line.split('|')
                    ep = int(parts[1])
                    postfix_str = parts[2] if len(parts) > 2 else ""
                    
                    # 상태 업데이트 단일화: 직접 할당 후 단 1회 갱신
                    pbar.n = ep
                    pbar.set_postfix_str(postfix_str)
                    pbar.refresh()
                except Exception:
                    pass
            elif clean_line.startswith("DEBUG_UPDATE|"):
                try:
                    # DEBUG_UPDATE|로그내용
                    parts = clean_line.split('|', 1)
                    if len(parts) > 1:
                        # tqdm.write를 사용하면 진행 표시줄 위로 깔끔하게 로그가 출력됩니다.
                        pbar.write(f"[{task_name}] {parts[1]}")
                except Exception:
                    pass
            else:
                # 3. 에러 추적 보호 기법
                # 서브프로세스의 기타 에러/경고(예: OOM Traceback)는 터미널을 깨지 않도록 파일에 기록
                with open(f"logs/{task_name.lower()}_error.log", "a", encoding="utf-8") as f:
                    f.write(line)
                
        pbar.close()

    t1 = threading.Thread(target=monitor_progress, args=("Worker", worker_eps, 0, worker_proc))
    t2 = threading.Thread(target=monitor_progress, args=("Manager", manager_eps, 1, manager_proc))

    t1.start()
    t2.start()

    # 두 프로세스 완료 대기
    worker_rc = worker_proc.wait()
    manager_rc = manager_proc.wait()

    t1.join()
    t2.join()

    print(f"\n✅ Worker  완료 (exit code: {worker_rc})")
    print(f"\n✅ Manager 완료 (exit code: {manager_rc})")

    if worker_rc != 0 or manager_rc != 0:
        print("❌ Worker 또는 Manager 학습이 실패했습니다. Alignment를 건너뜁니다.")
        return

    # Alignment 실행 (GPU 0)
    print(f"\n{'='*60}")
    print(f"🔗 Alignment Stage 시작 ({alignment_eps:,} eps on GPU 0)")
    print(f"{'='*60}\n")

    alignment_pomo = int(float(args.num_pomo) * 1.5)  # Worker 동결 → VRAM 여유
    alignment_cmd = base_args + [
        "--stage", "alignment",
        "--episodes", str(alignment_eps),
        "--num_pomo", str(alignment_pomo),
    ]
    alignment_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    alignment_rc = subprocess.call(
        alignment_cmd, env=alignment_env,
        cwd=os.getcwd(),
    )

    if alignment_rc == 0:
        print(f"\n{'='*60}")
        print("🎉 Phase 1 Parallel 전체 학습 완료!")
        print(f"{'='*60}")
    else:
        print(f"\n❌ Alignment 학습 실패 (exit code: {alignment_rc})")


def train_rl(args):
    if args.disaster:
        raise ValueError(
            "--disaster is legacy / unsupported. "
            "Use --stage phase1 for worker→manager→alignment pipeline."
        )

    if args.stage == "phase1_parallel":
        _run_parallel_phase1(args)
        return

    env, manager, worker, device = _init_env_and_models(args)

    if args.stage == "phase1":
        # Phase 1: worker → manager → alignment 자동 순차 실행
        worker_eps = args.episodes
        manager_eps = args.episodes
        alignment_eps = args.episodes

        print(f"\n🎯 Phase 1 자동 순차 실행: 각 Stage별 {args.episodes} episodes (총 {args.episodes * 3} eps)")
        print(f"   1. Worker:    {worker_eps} eps → logs/rl_worker_stage/")
        print(f"   2. Manager:   {manager_eps} eps → logs/rl_manager_stage/")
        print(f"   3. Alignment: {alignment_eps} eps → logs/rl_alignment_stage/")

        _run_single_stage(args, env, manager, worker, device, "worker", worker_eps)
        _run_single_stage(args, env, manager, worker, device, "manager", manager_eps)
        _run_single_stage(args, env, manager, worker, device, "alignment", alignment_eps)

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
        choices=["manager", "worker", "alignment", "phase1", "phase1_parallel"],
        help="학습 단계: phase1(순차), phase1_parallel(Worker∥Manager→Joint)",
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
        help="디버그 모드: 주기적으로 단계별 지표를 상세 로그로 출력",
    )
    parser.add_argument("--disable_tqdm", action="store_true", help="내부 tqdm 비활성화 및 stdin 보고")
    parser.add_argument(
        "--force_joint",
        action="store_true",
        help="Legacy flag. Unsupported in APTE branch.",
    )
    train_rl(parser.parse_args())
