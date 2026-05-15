import argparse
import os
import warnings
from datetime import datetime

# lr_scheduler.step() 순서 경고 억제
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

# [Hardcoded CPU Threads] 최적의 컨텍스트 스위칭 효율을 위해 프로세스당 8개 코어 할당
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

# [Speed Optimization] RTX 4090 (Ada) 및 고정된 입력 형태를 위한 하드웨어 극한 속도 튜닝
# 1. cuDNN Benchmark: 첫 스텝 수행 시 최적의 CUDA 커널을 찾아 고정
cudnn.benchmark = True
# 2. TF32 활성화: 행렬 곱셈 속도 최대 3배 폭증 (정밀도 손실 체감 불가)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.envs.disaster_env import DisasterEnv
from src.models.worker import Worker
from src.trainers.worker_trainer import HRLWorkerTrainer  # [HRL Phase 1]

# [Manager v2] 비자기회귀 + PPO + PBRS Re-planning
from src.models.reactive_manager import ReactiveManager
from src.trainers.manager_ppo_trainer import ManagerPPOTrainer
from src.envs.hrl_closed_loop_env import HRLClosedLoopEnv

# [Legacy] 기존 Manager/Trainer — legacy 폴더로 이동됨
from src.models.legacy.node_manager import GraphTransformerManager, DEFAULT_MANAGER_MASK_CFG
from src.trainers.legacy.worker_nav_trainer import WorkerNavTrainer
from src.trainers.legacy.manager_stage_trainer import ManagerStageTrainer
from src.trainers.legacy.pomo_trainer import DOMOTrainer


# [Ablation] Bias/Reward preset → config dict 매핑
def _get_bias_preset_config(preset: str, khop_K: int = 10) -> dict:
    """bias_preset CLI 인자를 mask_cfg dict로 변환."""
    cfg = dict(DEFAULT_MANAGER_MASK_CFG)
    cfg['khop_K'] = khop_K
    
    if preset == 'full':
        pass  # 모든 bias ON (기존 동작)
    elif preset == 'none':
        # visited만 ON, 나머지 전부 OFF
        cfg['enable_khop_mask'] = False
        cfg['enable_radius_mask'] = False
        cfg['enable_directional_bias'] = False
        cfg['enable_corridor_bonus'] = False
        cfg['enable_progress_bonus'] = False
        cfg['enable_detour_penalty'] = False
        cfg['enable_nonprogress_penalty'] = False
        cfg['enable_eos_control'] = False
        cfg['bias_scale'] = 0.0
    elif preset == 'khop_only':
        # visited + K-hop만 ON
        cfg['enable_radius_mask'] = False
        cfg['enable_directional_bias'] = False
        cfg['enable_corridor_bonus'] = False
        cfg['enable_progress_bonus'] = False
        cfg['enable_detour_penalty'] = False
        cfg['enable_nonprogress_penalty'] = False
        cfg['enable_eos_control'] = False
        cfg['bias_scale'] = 0.0
    elif preset == 'soft_only':
        # K-hop OFF, soft bias만 ON
        cfg['enable_khop_mask'] = False
        cfg['enable_radius_mask'] = False  # radius는 hard에 가까우므로 OFF
        cfg['enable_directional_bias'] = False  # directional도 OFF
    return cfg


def _get_reward_preset_config(preset: str) -> dict:
    """reward_preset CLI 인자를 reward ablation config dict로 변환."""
    cfg = {
        'enable_r1_pbrs': True,
        'enable_r2_subgoal': True,
        'enable_r3_goal': True,
        'enable_r4_efficiency': True,
        'enable_r5_milestone': True,
        'enable_r6_exploration': True,
        'enable_r7_plan_penalty': True,
        'enable_p1_time_pressure': True,
        'enable_p2_loop': True,
        'enable_p3_fail': True,
        'subgoal_reward_mode': 'exact',
        'proximity_K': 10,
    }
    
    if preset == 'full':
        pass
    elif preset == 'minimal':
        cfg['enable_r2_subgoal'] = False
        cfg['enable_r4_efficiency'] = False
        cfg['enable_r5_milestone'] = False
        cfg['enable_r6_exploration'] = False
        cfg['enable_r7_plan_penalty'] = False
        cfg['enable_p1_time_pressure'] = False
    elif preset == 'mid':
        cfg['enable_r4_efficiency'] = False
        cfg['enable_r5_milestone'] = False
        cfg['enable_r6_exploration'] = False
        cfg['enable_p1_time_pressure'] = False
    elif preset == 'proximity':
        cfg['enable_r4_efficiency'] = False
        cfg['enable_r5_milestone'] = False
        cfg['enable_r6_exploration'] = False
        cfg['enable_p1_time_pressure'] = False
        cfg['subgoal_reward_mode'] = 'proximity'
    return cfg

def _apply_overrides(cfg: dict, override_str: str) -> None:
    """'key=value,key=value' 형식의 문자열을 파싱하여 config dict에 적용.
    
    Why: preset으로 큰 틀을 잡고, 개별 항목만 세밀하게 조절하는 Ablation용.
    지원 타입: bool(True/False), int, float, str
    """
    if not override_str or not override_str.strip():
        return
    for pair in override_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            print(f"⚠️ Override 무시 (잘못된 형식): {pair}")
            continue
        key, val = pair.split('=', 1)
        key, val = key.strip(), val.strip()
        if key not in cfg:
            print(f"⚠️ Override 무시 (알 수 없는 키): {key}")
            continue
        # 타입 자동 변환
        if val.lower() == 'true':
            cfg[key] = True
        elif val.lower() == 'false':
            cfg[key] = False
        elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
            cfg[key] = float(val) if '.' in val else int(val)
        else:
            cfg[key] = val


# Ablation Study 설정 로드
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
try:
    from ablation_configs import get_ablation_config, get_worker_kwargs
except ImportError:
    # tests/ablation_configs.py가 없으면 기본값 사용
    def get_ablation_config(ablation_id: str) -> dict:
        return {}
    def get_worker_kwargs(ablation_id: str, base_hidden_dim: int = 256) -> dict:
        return {"node_dim": 7, "hidden_dim": base_hidden_dim, "num_layers": 3, "edge_dim": 3}


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


def _build_config(args, loaded_checkpoint_paths, stage_override=None):
    # stage_override: phase1 순차 실행 시 각 단계별 stage 지정용
    effective_stage = stage_override or args.stage
    # 타임스탬프 서브폴더 생성: logs/<stage>/<YYYY-MM-DD_HHMM>_<stage>_pomo<N>[_<ablation>]/
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    # [Ablation] 실험 ID를 폴더명에 포함하여 동시 실행 시 충돌 방지
    ablation_suffix = ""
    if hasattr(args, 'ablation') and args.ablation.upper() != "BASELINE":
        ablation_suffix = f"_{args.ablation.upper()}"
    run_label = f"{timestamp}_{effective_stage}_B{args.batch_size}{ablation_suffix}"
    stage_base = {
        'manager': os.path.join('logs', 'rl_manager_stage'),
        'worker': os.path.join('logs', 'rl_worker_stage'),
        'alignment': os.path.join('logs', 'rl_alignment_stage'),
    }.get(effective_stage, os.path.join('logs', 'rl_finetune'))
    save_dir = os.path.join(stage_base, run_label)
    return Config(
        lr=args.lr,
        num_pomo=args.batch_size,
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
        ablation_config=getattr(args, "_ablation_config", {}),  # [Ablation] 실험 설정 전달
        # [HRL Phase 1] Worker Ablation 플래그 — CLI → Config → HRLWorkerTrainer 전달
        use_gae=getattr(args, "use_gae", False),
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        entropy_coeff=getattr(args, "entropy_coeff", 0.0),
        use_cosine_lr=getattr(args, "use_cosine_lr", False),
        zone_progress_reward=getattr(args, "zone_progress_reward", False),
        mgr_state_preset=getattr(args, 'mgr_state_preset', 'S0'),
        # [Ablation] Manager bias/reward preset config
        bias_mask_cfg=getattr(args, '_bias_mask_cfg', dict(DEFAULT_MANAGER_MASK_CFG)),
        reward_ablation_cfg=getattr(args, '_reward_ablation_cfg', {}),
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


from src.envs.hrl_env import HRLZoneEnv

def _init_env_and_models(args):
    """환경, 모델, 디바이스 초기화 (공용)."""
    print("Initializing Environment...")
    if args.stage == "worker":
        # Phase 1: Worker 검증용 HRLZoneEnv 사용
        # 맵별 Zone 파일 자동 탐색: data/node_to_zone_{map}_k*.json 우선, 없으면 기본 k30
        import glob
        zone_json = 'data/node_to_zone_k30.json'  # 기본값 (Anaheim 호환)
        zone_graph_json = 'data/zone_graph_k30.json'
        map_zone_files = glob.glob(f"data/node_to_zone_{args.map}_k*.json")
        if map_zone_files:
            zone_json = sorted(map_zone_files)[-1]  # 가장 큰 K 사용
            k_val = zone_json.split('_k')[-1].replace('.json', '')
            zone_graph_json = f"data/zone_graph_{args.map}_k{k_val}.json"
            print(f"   Zone 파일: {zone_json} (맵별 자동 탐색)")
        # [v3 Ablation] masking_mode, use_pbrs 환경 변인 전달
        env = HRLZoneEnv(
            f"data/{args.map}_node.tntp",
            f"data/{args.map}_net.tntp",
            zone_json=zone_json,
            zone_graph_json=zone_graph_json,
            masking_mode=getattr(args, 'masking_mode', 'hard'),
            use_pbrs=getattr(args, 'use_pbrs', False),
            subgoal_mode=getattr(args, 'subgoal_mode', 'default'),
        )
        # [Ablation] P0: Zone 전환 중간 보상 플래그 전달
        env.zone_progress_reward = getattr(args, 'zone_progress_reward', False)
        print(f"   Env: masking_mode={env.masking_mode}, use_pbrs={env.use_pbrs}, subgoal_mode={getattr(args, 'subgoal_mode', 'default')}")
    else:
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

    # [Manager State Ablation] Preset별 Node Dimension 매핑
    mgr_state_dims = {
        "S0": 4, "S1": 2, "S2": 3, "S3": 5, "S4": 5, "S5": 5, "S6": 5,
        "S7": 4, "S8": 5, "S9": 5, "S10": 6, "S11": 6, "S12": 7, "S13": 8
    }
    mgr_preset = getattr(args, 'mgr_state_preset', 'S0')
    mgr_node_dim = mgr_state_dims.get(mgr_preset, 4)

    manager = GraphTransformerManager(node_dim=mgr_node_dim, hidden_dim=args.hidden_dim, dropout=0.2, edge_dim=3).to(device)
    print(f"   Manager: preset={mgr_preset}, node_dim={mgr_node_dim}, hidden_dim={args.hidden_dim}")
    
    # [v3 Ablation] Worker 생성: num_layers를 CLI에서 동적 제어
    use_ckpt = args.stage == "worker"
    num_layers = getattr(args, 'num_layers', 2)
    use_jk_net = getattr(args, 'use_jk_net', False)
    use_edge_attr = getattr(args, 'use_edge_attr', False)
    worker = Worker(
        node_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        use_checkpoint=use_ckpt,
        use_jk_net=use_jk_net,
        use_edge_attr=use_edge_attr,
    ).to(device)
    print(f"   Worker: node_dim=4, hidden_dim={args.hidden_dim}, num_layers={num_layers}, use_jk_net={use_jk_net}, use_edge_attr={use_edge_attr}, use_checkpoint={use_ckpt}")

    # --steps → --episodes 변환 (steps 우선)
    if args.steps is not None:
        args.episodes = args.steps * args.batch_size
        print(f"📐 배치={args.batch_size}, 스텝={args.steps} → 총 에피소드={args.episodes}")
    elif args.episodes is None:
        args.episodes = 5000  # 기본값
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")
    else:
        print(f"📐 배치={args.batch_size}, 에피소드={args.episodes} → 스텝={args.episodes // args.batch_size}")
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
        # [HRL Phase 1] Worker는 scratch에서 시작 (4-Dim 구조 변경으로 SL 체크포인트 비호환)
        print("📋 HRL Phase 1: Worker를 scratch에서 학습합니다. (SL 체크포인트 미사용)")
    elif stage == "manager":
        # Worker: worker_stage best → fallback SL
        if not _load_worker_checkpoint(wkr_stage_ckpt, worker, device, loaded_checkpoint_paths):
            if not _load_worker_checkpoint(sl_ckpt, worker, device, loaded_checkpoint_paths):
                print("⚠️ No worker checkpoint for manager stage.")
        # Manager: State Ablation으로 인해 SL pretrained를 무시하고 무조건 Scratch에서 시작
        print("📋 Manager: State Ablation 실험(node_dim 변경)을 위해 SL 체크포인트를 건너뛰고 Scratch에서 학습합니다.")
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
    if stage == "manager_v2":
        # [Manager v2] ReactiveManager + PPO + PBRS Closed-loop
        node_dim = 4  # S7: is_curr, is_tgt, hop_dist, degree
        reactive_mgr = ReactiveManager(
            node_dim=node_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, dropout=0.2,
        ).to(device)
        print(f"📋 Manager v2: ReactiveManager (node_dim={node_dim}, hidden={args.hidden_dim})")
        print(f"   파라미터 수: {sum(p.numel() for p in reactive_mgr.parameters()):,}")

        # Worker 체크포인트 로드
        wkr_ckpt = _get_latest_ckpt(os.path.join('logs', 'rl_worker_stage'), 'best.pt')
        if wkr_ckpt:
            _load_worker_checkpoint(wkr_ckpt, worker, device, loaded_checkpoint_paths)
        else:
            print("⚠️ Worker 체크포인트 없음. Worker는 랜덤 초기 상태로 사용됩니다.")

        # Closed-loop 환경 생성
        node_file = f"data/{args.map}_node.tntp"
        net_file = f"data/{args.map}_net.tntp"
        cl_env = HRLClosedLoopEnv(
            node_file=node_file, net_file=net_file,
            worker=worker, k_hop=5, c_max=8,
            device=str(device),
        )

        # PPO Trainer 생성 및 학습
        config.save_dir = os.path.join('logs', 'rl_manager_v2', config.save_dir.split('/')[-1])
        os.makedirs(config.save_dir, exist_ok=True)
        ppo_trainer = ManagerPPOTrainer(cl_env, reactive_mgr, config)
        ppo_trainer.train(episodes)
    elif stage == "manager":
        trainer = ManagerStageTrainer(env, manager, worker, config)
        trainer.train(episodes)
    elif stage == "worker":
        # [HRL Phase 1] HRLZoneEnv + Worker 전용 Trainer 사용
        trainer = HRLWorkerTrainer(env, manager, worker, config)
        trainer.train(episodes)
    elif stage == "alignment":
        trainer = DOMOTrainer(env, manager, worker, config)
        trainer.train(episodes)
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
        "--batch_size", str(args.batch_size),
        "--disable_tqdm",
    ]
    worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    # Manager subprocess (GPU 1)
    # [Fix] 사용자 요청: Manager의 POMO 크기를 1.5배(예: 48 * 1.5 = 72)로 상향 조정하여 VRAM을 적절히 활용
    if str(args.batch_size) == "auto":
        manager_pomo = "auto"
    else:
        manager_pomo = int(float(args.batch_size) * 1.5)
    manager_cmd = base_args + [
        "--stage", "manager",
        "--episodes", str(manager_eps),
        "--batch_size", str(manager_pomo),
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

    alignment_pomo = int(float(args.batch_size) * 1.5)  # Worker 동결 → VRAM 여유
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
    parser.add_argument("--episodes", type=int, default=None,
                        help="총 에피소드 수 (--steps 미지정 시 사용, 기본 5000)")
    parser.add_argument("--steps", type=int, default=None,
                        help="총 gradient 업데이트 스텝 수 (지정 시 --episodes보다 우선)")
    parser.add_argument("--batch_size", "--num_pomo", type=int, default=16,
                        help="배치 크기: 스텝당 동시 실행 에피소드 수 (기본 16)")
    parser.add_argument(
        "--stage",
        default="phase1",
        choices=["manager", "manager_v2", "worker", "alignment", "phase1", "phase1_parallel"],
        help="학습 단계: manager_v2(비자기회귀+PPO), phase1(순차), phase1_parallel(Worker∥Manager→Joint)",
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
    parser.add_argument("--ablation", type=str, default="BASELINE",
                        help="Ablation 실험 ID (BASELINE, A1~A7, S1~S5, R1~R5)")
    parser.add_argument(
        "--force_joint",
        action="store_true",
        help="Legacy flag. Unsupported in APTE branch.",
    )
    # [HRL Ablation] Phase 1 Worker 실험 플래그
    parser.add_argument("--zone_progress_reward", action="store_true",
                        help="[P0] Zone 전환 시 중간 보상 부여")
    parser.add_argument("--use_gae", action="store_true",
                        help="[P1] GAE(λ) Advantage 사용")
    parser.add_argument("--entropy_coeff", type=float, default=0.0,
                        help="[P1] Entropy Bonus 계수 (0이면 비활성)")
    parser.add_argument("--use_cosine_lr", action="store_true",
                        help="[P2] Cosine LR Scheduler 사용")
    # [v3 Ablation] 환경/모델 변인 제어
    parser.add_argument("--masking_mode", type=str, default="hard",
                        choices=["hard", "hard_full_seq", "soft_curr_next", "soft_flex"],
                        help="Action Masking 모드 (hard/hard_full_seq/soft_curr_next/soft_flex)")
    parser.add_argument("--use_pbrs", action="store_true",
                        help="hop_dist 기반 PBRS Dense Reward 활성")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="GATv2 레이어 수 (1~4)")
    parser.add_argument("--use_jk_net", action="store_true",
                        help="[v4] JK-Net (Jumping Knowledge) 활성")
    parser.add_argument("--use_edge_attr", action="store_true",
                        help="[v4] Edge-Conditioned MP (Edge Features) 활성")
    parser.add_argument("--subgoal_mode", type=str, default="zone",
                        choices=["zone", "node"],
                        help="[Part2] Manager Subgoal 모드 (zone / node)")
    # [Manager State Ablation] preset 설정
    parser.add_argument("--mgr_state_preset", type=str, default="S0",
                        choices=[f"S{i}" for i in range(14)],
                        help="Manager State Ablation 14개 실험 프리셋 (S0~S13)")
    # [Ablation] Manager Decode Bias / Reward preset
    parser.add_argument("--bias_preset", type=str, default="full",
                        choices=["full", "none", "khop_only", "soft_only"],
                        help="Decode bias ablation preset")
    parser.add_argument("--khop_K", type=int, default=10,
                        help="K-hop masking radius (bias_preset=khop_only 시 사용)")
    parser.add_argument("--reward_preset", type=str, default="full",
                        choices=["full", "minimal", "mid", "proximity"],
                        help="Reward ablation preset")
    # [Ablation] 세밀한 Override: preset 위에 개별 항목 덮어쓰기
    parser.add_argument("--bias_override", type=str, default="",
                        help="Bias config override (예: enable_corridor_bonus=True,enable_eos_control=False)")
    parser.add_argument("--reward_override", type=str, default="",
                        help="Reward config override (예: enable_r6_exploration=True,enable_p3_fail=False)")
    
    args = parser.parse_args()
    # [Ablation] preset → config dict 변환
    args._bias_mask_cfg = _get_bias_preset_config(args.bias_preset, args.khop_K)
    args._reward_ablation_cfg = _get_reward_preset_config(args.reward_preset)
    
    # [Ablation] Override 적용: key=value 쌍을 파싱하여 config에 반영
    _apply_overrides(args._bias_mask_cfg, args.bias_override)
    _apply_overrides(args._reward_ablation_cfg, args.reward_override)
    
    # [Ablation] 최종 설정 출력
    if args.bias_preset != 'full' or args.bias_override:
        print(f"🔧 Bias Config: preset={args.bias_preset}, override={args.bias_override or 'none'}")
    if args.reward_preset != 'full' or args.reward_override:
        print(f"🔧 Reward Config: preset={args.reward_preset}, override={args.reward_override or 'none'}")
    
    train_rl(args)
