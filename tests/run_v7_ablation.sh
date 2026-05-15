#!/bin/bash

# =========================================================================
# [Manager State Ablation v7] 
# Manager의 입력 State(node_dim)를 2차원부터 8차원까지 다양하게 변형하여 제로샷 최적 구조 도출
# 총 14개 실험 (S0 ~ S13)
# =========================================================================

# 공통 하이퍼파라미터
EPOCHS=5000
BATCH=16
COMMON_FLAGS="--stage manager --episodes $EPOCHS --batch_size $BATCH --bias_preset full --reward_preset full"

# 실험 큐
declare -a experiments=(
    # 그룹 1: 좌표 제거 가능성
    "S0"  # Baseline (x, y, is_curr, is_tgt)
    "S1"  # 좌표 제거 (is_curr, is_tgt)
    "S2"  # 좌표 제거 + hop_dist

    # 그룹 2: 피처 1개씩 추가 (좌표 유지)
    "S3"  # +hop_dist
    "S4"  # +net_dist
    "S5"  # +degree
    "S6"  # +betweenness

    # 그룹 3: 좌표 완전 대체 (제로샷 타겟)
    "S7"  # hop_dist + degree
    "S8"  # hop_dist + degree + betweenness
    "S9"  # hop_dist + net_dist + degree
    "S10" # hop_dist + net_dist + degree + betweenness

    # 그룹 4: 하이브리드 최적 조합 (좌표 유지)
    "S11" # x,y + hop_dist + degree
    "S12" # x,y + hop_dist + degree + betweenness
    "S13" # 풀세트
)

# GPU별 병렬 스케줄러 로직
echo "🚀 Starting Manager State Ablation v7"
echo "Total experiments: ${#experiments[@]}"
echo "Using GPUs 0 and 1"

for i in "${!experiments[@]}"; do
    EXP_ID="${experiments[$i]}"
    GPU_ID=$((i % 2))
    
    echo "[$(date +%T)] Starting $EXP_ID on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n rl python train_rl.py $COMMON_FLAGS \
        --mgr_state_preset $EXP_ID \
        --ablation MGR_STATE_$EXP_ID > "logs/v7_ablation_${EXP_ID}.log" 2>&1 &
        
    if [ $GPU_ID -eq 1 ]; then
        echo "⏳ Waiting for GPUs to finish..."
        wait
    fi
done

wait
echo "🎉 All Manager State Ablation v7 experiments completed!"
