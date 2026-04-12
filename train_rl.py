import argparse
import os
from datetime import datetime

import torch

from src.envs.disaster_env import DisasterEnv
from src.models.manager import GraphTransformerManager
from src.models.worker import WorkerLSTM
from src.trainers.phase1_guided_worker_trainer import Phase1GuidedWorkerTrainer
from src.trainers.manager_stage_trainer import ManagerStageTrainer  # [Refactor: Task 1]
from src.trainers.worker_stage_trainer import WorkerStageTrainer  # [Refactor: Task 1]
from src.trainers.pomo_trainer import DOMOTrainer  # [Refactor: Task 1]


def _find_max_batch_size(env, worker, device, max_try: int = 64) -> int:
    """GPU VRAM 한도 내 최대 배치 크기를 자동 탐색.

    [8, 12, 16, 24, 32, 48, 64] 순서로 시도하며,
    OOM 발생 직전 크기를 반환합니다.
    """
    # 탐색 후보: 2의 거듭제곱 + 중간값
    candidates = [bs for bs in [8, 12, 16, 24, 32, 48, 64] if bs <= max_try]
    best = candidates[0]

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            env.reset(batch_size=bs, sync_problem=True)
            h = torch.zeros(bs, worker.lstm.hidden_size, device=device)
            c = torch.zeros(bs, worker.lstm.hidden_size, device=device)
            # Worker는 node_dim=8을 기대 → Phase1의 _build_worker_input과 동일하게 구성
            raw_x = env.pyg_data.x
            if raw_x.size(1) < 9:
                pad = torch.zeros(raw_x.size(0), 9 - raw_x.size(1), device=device)
                raw_x = torch.cat([raw_x, pad], dim=1)
            worker_input = torch.cat([raw_x[:, :4], raw_x[:, 5:9]], dim=1)  # [B*N, 8]
            edge_index = env.pyg_data.edge_index
            batch_vec = env.pyg_data.batch
            # edge_attr도 Worker edge_dim=5에 맞게 필터링
            ea = env.pyg_data.edge_attr
            if ea is not None:
                if ea.size(1) >= 9:
                    ea = ea[:, [0, 1, 4, 6, 8]]
                elif ea.size(1) >= 5:
                    ea = ea[:, :5]
            # 5스텝 forward+backward 시뮬레이션으로 VRAM 사용량 테스트
            worker.train()
            opt = torch.optim.SGD(worker.parameters(), lr=1e-5)
            for _ in range(5):
                scores, h_new, c_new, v = worker.predict_next_hop(
                    worker_input, edge_index, h, c, batch_vec,
                    detach_spatial=True, edge_attr=ea,
                )
                loss = scores.sum() + v.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                h = h_new.detach()
                c = c_new.detach()
            del h, c, scores, h_new, c_new, v, loss, opt
            torch.cuda.empty_cache()
            best = bs
            print(f"  ✅ batch_size={bs} OK")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                torch.cuda.empty_cache()
                print(f"  ❌ batch_size={bs} OOM → 이전 크기 {best} 사용")
                break
            raise  # OOM이 아닌 다른 에러는 전파
    return best


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


def _build_config(args, loaded_checkpoint_paths):
    # 타임스탬프 서브폴더 생성: logs/<stage>/<YYYY-MM-DD_HHMM>_<stage>_pomo<N>/
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_label = f"{timestamp}_{args.stage}_pomo{args.num_pomo}"
    stage_base = {
        'manager': os.path.join('logs', 'rl_manager_stage'),
        'worker': os.path.join('logs', 'rl_worker_stage'),
        'joint': os.path.join('logs', 'rl_joint_stage'),
        'phase1': os.path.join('logs', 'rl_phase1_apte'),
    }.get(args.stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        num_pomo=args.num_pomo,
        episodes=args.episodes,
        save_dir=save_dir,  # 타임스탬프 서브폴더에 저장
        stage=args.stage,  # [Refactor: Task 1] 커리큘럼 stage 전달
        debug=args.debug,
        run_type="smoke" if args.episodes <= 5 else "train",
        parent_checkpoints=loaded_checkpoint_paths,
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


def train_rl(args):
    if args.disaster:
        raise ValueError(
            "--disaster is legacy / unsupported in the APTE branch. "
            "Use worker-only APTE phase1."
        )
    # [Refactor: Task 1] 멀티스테이지 지원 (manager/worker/joint/phase1)

    print(f"🚀 Starting RL Training - Stage: {args.stage}")
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

    # APTE phase1 is worker-centric at runtime. The manager module is carried only
    # as an unused compatibility shell so we can reuse the shared trainer helpers.
    manager = GraphTransformerManager(node_dim=4, hidden_dim=args.hidden_dim, dropout=0.2).to(device)
    worker = WorkerLSTM(node_dim=8, hidden_dim=args.hidden_dim).to(device)

    # [Perf] 배치 크기 자동 탐색
    if args.num_pomo == "auto":
        if device.type == "cuda":
            print("🔍 VRAM 한도 내 최대 배치 크기 탐색 중...")
            args.num_pomo = _find_max_batch_size(env, worker, device)
            print(f"📐 자동 탐색 결과: num_pomo = {args.num_pomo}")
        else:
            args.num_pomo = 8
            print("ℹ️ CPU 모드: num_pomo=8 기본값 사용")
    else:
        args.num_pomo = int(args.num_pomo)

    loaded_checkpoint_paths = []

    def _get_latest_ckpt(base_path, fallback_name):
        ckpt = os.path.join(base_path, fallback_name)
        if os.path.exists(base_path):
            subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)
                ckpt = os.path.join(latest_subdir, fallback_name)
        return ckpt

    # [Refactor: Task 1] Stage별 체크포인트 로드 전략
    sl_ckpt = _get_latest_ckpt(os.path.join("logs", "sl_pretrain"), "model_sl_final.pt")
    mgr_stage_ckpt = _get_latest_ckpt(os.path.join("logs", "rl_manager_stage"), "best.pt")
    wkr_stage_ckpt = _get_latest_ckpt(os.path.join("logs", "rl_worker_stage"), "best.pt")

    if args.stage in ("manager", "worker", "phase1"):
        # SL pretrained worker 로드
        if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
            print("⚠️ SL worker checkpoint not found. Starting from scratch.")
    elif args.stage == "joint":
        # Worker stage best → fallback SL
        if not _load_worker_checkpoint(wkr_stage_ckpt, worker, device, loaded_checkpoint_paths):
            if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
                print("⚠️ No worker checkpoint found for joint stage.")

    if args.stage in ("worker", "joint"):
        # Manager stage checkpoint → fallback SL (Manager가 랜덤이면 Plan이 엉망)
        if not _load_manager_checkpoint(mgr_stage_ckpt, manager, device, loaded_checkpoint_paths):
            if not _load_manager_checkpoint(sl_ckpt, manager, device, loaded_checkpoint_paths):
                print("⚠️ Manager checkpoint not found. Manager starts from scratch.")

    config = _build_config(args, loaded_checkpoint_paths)

    # [Refactor: Task 1] Stage별 Trainer 분기
    if args.stage == "manager":
        trainer = ManagerStageTrainer(env, manager, worker, config)
    elif args.stage == "worker":
        trainer = WorkerStageTrainer(env, manager, worker, config)
    elif args.stage == "joint":
        trainer = DOMOTrainer(env, manager, worker, config)
    else:  # phase1
        trainer = Phase1GuidedWorkerTrainer(env, manager, worker, config)

    trainer.train(args.episodes)


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
    parser.add_argument("--lr", type=float, default=1e-5)
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
