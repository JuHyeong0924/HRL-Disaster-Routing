#!/bin/bash

# =========================================================================
# [Manager State Long-term Convergence Ablation] 
# S7 (is_curr, is_tgt, hop_dist, degree)
# S9 (is_curr, is_tgt, hop_dist, net_dist, degree, betweenness)
# Zero-shot 일반화 능력을 극대화하기 위해 좌표를 배제한 모델을 20,000 에피소드까지 극한 학습
# =========================================================================

EPOCHS=20000
BATCH=16
COMMON_FLAGS="--stage manager --episodes $EPOCHS --batch_size $BATCH --bias_preset full --reward_preset full"

# 실험 큐 (Zero-shot 최적 조합 2개)
declare -a experiments=(
    "S7"
    "S9"
)

echo "🚀 Starting Manager Long-term State Ablation (20,000 episodes)"
echo "Total experiments: ${#experiments[@]}"
echo "Using GPUs 0 and 1"

for i in "${!experiments[@]}"; do
    EXP_ID="${experiments[$i]}"
    GPU_ID=$((i % 2))
    
    echo "[$(date +%T)] Starting $EXP_ID on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n rl python train_rl.py $COMMON_FLAGS \
        --mgr_state_preset $EXP_ID \
        --ablation MGR_STATE_LONG_$EXP_ID > "logs/v7_long_ablation_${EXP_ID}.log" 2>&1 &
done

echo "⏳ Waiting for GPUs to finish..."
wait
echo "🎉 All Manager Long-term Ablation experiments completed!"
