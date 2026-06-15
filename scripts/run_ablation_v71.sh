#!/bin/bash
# ============================================================
# V7.1 Re-ablation: Dijkstra log1p 수정 후 재학습
# B_v2: dijkstra(log1p) + binary + uniform
# E_v2: dijkstra(log1p) + ternary + euclidean
# + 추가: F: hop + ternary + euclidean (최종 후보 조합)
# ============================================================
set -e
cd "$(dirname "$0")/.."

COMMON_ARGS="--stage worker --map Anaheim --episodes 5000 --batch_size 32 \
--masking_mode soft_curr_next --use_pbrs --use_is_visited --use_relative_hop \
--hidden_dim 256 --lr 1e-4"

echo "============================================"
echo "🔬 V7.1 Re-ablation (log1p Dijkstra + F)"
echo "   GPU 0: B_v2 → F (순차)"
echo "   GPU 1: E_v2 (순차)"
echo "============================================"

# GPU 0: B_v2 → F
(
    echo "[GPU 0] 🧪 B_v2: dijkstra(log1p) + binary + uniform"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode dijkstra --zone_info_mode binary --zone_weight_mode uniform \
        --exp_name ablV71_Bv2_dijkstra_binary_uniform

    echo "[GPU 0] 🧪 F: hop + ternary + euclidean (최종 후보)"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode hop --zone_info_mode ternary --zone_weight_mode euclidean \
        --exp_name ablV71_F_hop_ternary_euclidean
) &
PID0=$!

# GPU 1: E_v2
(
    echo "[GPU 1] 🧪 E_v2: dijkstra(log1p) + ternary + euclidean"
    CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py $COMMON_ARGS \
        --dist_mode dijkstra --zone_info_mode ternary --zone_weight_mode euclidean \
        --exp_name ablV71_Ev2_dijkstra_ternary_euclidean
) &
PID1=$!

echo "⏳ GPU 0 PID: $PID0, GPU 1 PID: $PID1"
wait $PID0; echo "✅ GPU 0 완료"
wait $PID1; echo "✅ GPU 1 완료"
echo "🎉 V7.1 Re-ablation 완료!"
