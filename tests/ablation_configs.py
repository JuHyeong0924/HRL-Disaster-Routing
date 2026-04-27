"""
ablation_configs.py — Worker Ablation Study 실험 정의
각 실험은 Baseline에서 한 가지만 변경하여 해당 요소의 기여도를 정량화한다.
"""

# 실험별 ablation_config 딕셔너리
# worker.py와 worker_nav_trainer.py가 이 값을 읽어 동작을 조건부 변경한다.
ABLATION_REGISTRY: dict = {
    # === Baseline (변경 없음) ===
    "BASELINE": {
        "description": "현재 v4.1 아키텍처 그대로",
    },

    # ==============================
    # 🏗️ Architecture Ablation (A1~A7)
    # ==============================
    "A1": {
        "description": "GATv2 2-Layer (3→2): receptive field 축소",
        "num_layers": 2,
    },
    "A2": {
        "description": "Residual Connection 제거: Over-smoothing 영향 확인",
        "use_residual": False,
    },
    "A3": {
        "description": "GraphNorm 제거: 정규화 기여도 확인",
        "use_graph_norm": False,
    },
    "A4": {
        "description": "Hidden dim 128 (256→128): 모델 경량화 검증",
        "hidden_dim": 128,
    },
    "A5": {
        "description": "LSTM 제거: 시간적 기억(Temporal Memory) 필요성 검증",
        "use_lstm": False,
    },
    "A6": {
        "description": "GATv2 4-Layer (3→4): 더 깊은 GNN 효과 검증",
        "num_layers": 4,
    },
    "A7": {
        "description": "GATv2 1-Layer (3→1): 최소 GNN으로 충분한가?",
        "num_layers": 1,
    },

    # ==============================
    # 📊 State Ablation (S1~S5)
    # ==============================
    "S1": {
        "description": "hop_dist 제거 (6-Dim): 위상학적 거리 없이 유클리드만으로 충분한가?",
        "state_remove_cols": [6],  # hop_dist 열 인덱스
        "node_dim_override": 6,
    },
    "S2": {
        "description": "dir_x, dir_y 제거 (5-Dim): 방향 벡터 없이 hop만으로 충분한가?",
        "state_remove_cols": [3, 4],  # dir_x, dir_y
        "node_dim_override": 5,
    },
    "S3": {
        "description": "time_to_go 제거: 시간 압박 인식 필요성 검증",
        "remove_time_to_go": True,
    },
    "S4": {
        "description": "net_dist 제거 (6-Dim): 네트워크 거리 vs 홉 거리 비교",
        "state_remove_cols": [2],  # net_dist
        "node_dim_override": 6,
    },
    "S5": {
        "description": "최소 State (3-Dim): is_curr + is_tgt + hop_dist만으로 네비게이션 가능한가?",
        "state_keep_cols": [0, 1, 6],  # is_curr, is_tgt, hop_dist
        "node_dim_override": 3,
    },

    # ==============================
    # 💰 Reward Ablation (R1~R5)
    # ==============================
    "R1": {
        "description": "PBRS 제거: Dense reward 없이 Sparse만으로 학습 가능한가?",
        "disable_pbrs": True,
    },
    "R2": {
        "description": "Hop Threshold Bonus 제거: 마일스톤 보상 기여도",
        "disable_hop_bonus": True,
    },
    "R3": {
        "description": "Checkpoint Bonus 제거: 교사 경유점 기여도",
        "disable_checkpoint_bonus": True,
    },
    "R4": {
        "description": "AuxCE Loss 제거: 보조 Cross-entropy 손실 기여도",
        "disable_aux_ce": True,
    },
    "R5": {
        "description": "간소화 보상: goal_reward + step_penalty만 (모든 bonus/penalty 제거)",
        "simplified_reward": True,
    },
}


def get_ablation_config(ablation_id: str) -> dict:
    """실험 ID로 ablation_config 딕셔너리를 반환한다."""
    key = ablation_id.upper()
    if key not in ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown ablation ID: '{ablation_id}'. "
            f"Available: {list(ABLATION_REGISTRY.keys())}"
        )
    return ABLATION_REGISTRY[key].copy()


def get_worker_kwargs(ablation_id: str, base_hidden_dim: int = 256) -> dict:
    """실험 ID에 따른 WorkerLSTM 생성 인자를 반환한다."""
    cfg = get_ablation_config(ablation_id)
    kwargs = {
        "node_dim": cfg.get("node_dim_override", 7),
        "hidden_dim": cfg.get("hidden_dim", base_hidden_dim),
        "num_layers": cfg.get("num_layers", 3),
        "edge_dim": 3,
    }
    return kwargs


# GPU 2개 병렬 실행을 위한 라운드 구성
# 각 라운드에서 2개씩 동시 실행
ABLATION_ROUNDS: list = [
    ["BASELINE", "A1"],
    ["A2", "A3"],
    ["A4", "A5"],
    ["A6", "A7"],
    ["S1", "S2"],
    ["S3", "S4"],
    ["S5", "R1"],
    ["R2", "R3"],
    ["R4", "R5"],
]

ALL_EXPERIMENT_IDS: list = list(ABLATION_REGISTRY.keys())
