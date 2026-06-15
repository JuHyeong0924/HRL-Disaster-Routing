#!/bin/bash
# ============================================================
# Worker Ablation Study V7: dist_mode / zone_info_mode / zone_weight_mode
# ============================================================
# 기존 best 설정 고정: soft_curr_next + use_pbrs + use_is_visited + use_relative_hop
# 변수: dist_mode, zone_info_mode, zone_weight_mode
#
# | ID | dist_mode | zone_info_mode | zone_weight_mode | GPU |
# |----|-----------|----------------|------------------|-----|
# | B  | dijkstra  | binary         | uniform          | 0   |
# | C  | hop       | ternary        | uniform          | 1   |
# | D  | hop       | binary         | euclidean        | 0   |
# | E  | dijkstra  | ternary        | euclidean        | 1   |
#
# A (baseline): hop + binary + uniform → 기존 체크포인트 사용 (학습 불필요)
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

COMMON_ARGS="--stage worker --map Anaheim --episodes 5000 --batch_size 32 \
--masking_mode soft_curr_next --use_pbrs --use_is_visited --use_relative_hop \
--hidden_dim 256 --lr 1e-4"

echo "============================================"
echo "🔬 Worker Ablation Study V7 시작"
echo "   GPU 0: Exp B → Exp D (순차)"
echo "   GPU 1: Exp C → Exp E (순차)"
echo "============================================"

# --- GPU 0: B → D (순차) ---
(
    # Exp B: dijkstra + binary + uniform
    echo "[GPU 0] 🧪 Exp B 시작: dijkstra + binary + uniform"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode dijkstra --zone_info_mode binary --zone_weight_mode uniform \
        --exp_name ablV7_B_dijkstra_binary_uniform

    # Exp D: hop + binary + euclidean
    echo "[GPU 0] 🧪 Exp D 시작: hop + binary + euclidean"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode hop --zone_info_mode binary --zone_weight_mode euclidean \
        --exp_name ablV7_D_hop_binary_euclidean
) &
GPU0_PID=$!

# --- GPU 1: C → E (순차) ---
(
    # Exp C: hop + ternary + uniform
    echo "[GPU 1] 🧪 Exp C 시작: hop + ternary + uniform"
    CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode hop --zone_info_mode ternary --zone_weight_mode uniform \
        --exp_name ablV7_C_hop_ternary_uniform

    # Exp E: dijkstra + ternary + euclidean (Full)
    echo "[GPU 1] 🧪 Exp E 시작: dijkstra + ternary + euclidean (Full)"
    CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode dijkstra --zone_info_mode ternary --zone_weight_mode euclidean \
        --exp_name ablV7_E_dijkstra_ternary_euclidean
) &
GPU1_PID=$!

echo "⏳ GPU 0 PID: $GPU0_PID, GPU 1 PID: $GPU1_PID"
echo "   대기 중..."

# 둘 다 완료 대기
wait $GPU0_PID
echo "✅ GPU 0 완료 (Exp B, D)"
wait $GPU1_PID
echo "✅ GPU 1 완료 (Exp C, E)"

echo ""
echo "============================================"
echo "🎉 전체 Ablation Study V7 완료!"
echo "   결과: logs/rl_worker_stage/ 디렉토리 확인"
echo "============================================"
