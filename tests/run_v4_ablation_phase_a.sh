#!/bin/bash
# =============================================================================
# v4 HRL Worker Ablation Study Runner (Phase A: Architecture & Physics)
# GPU 0/1 병렬 실행, steps=500으로 빠른 스크리닝
# =============================================================================

set -e

# 공통 파라미터 (v3에서 가장 우수했던 soft_curr_next + PBRS 기본 적용)
COMMON="--stage worker --steps 5000 --batch_size 16 --use_gae --entropy_coeff 0.01 --zone_progress_reward --lr 5e-4 --masking_mode soft_curr_next --use_pbrs"

# 실험 실행 함수
run_exp() {
    local GPU=$1
    local ABLATION_ID=$2
    shift 2
    local EXTRA_ARGS="$@"
    
    echo "🚀 [GPU $GPU] 시작: $ABLATION_ID | $EXTRA_ARGS"
    CUDA_VISIBLE_DEVICES=$GPU python -u train_rl.py $COMMON --ablation "$ABLATION_ID" $EXTRA_ARGS \
        > "logs/ablation_v4_${ABLATION_ID}.log" 2>&1
    echo "✅ [GPU $GPU] 완료: $ABLATION_ID"
}

# 로그 디렉토리 생성
mkdir -p logs

echo "============================================================"
echo "🧪 v4 Phase A Ablation Study 시작 (GPU 0/1 병렬)"
echo "   공통: $COMMON"
echo "============================================================"

# --- Round 1: Baseline(L3) vs JK-Net(L3) ---
echo -e "\n--- Round 1: JK-Net 도입 효과 ---"
run_exp 0 BL_L3   --num_layers 3 &
run_exp 1 JK3     --num_layers 3 --use_jk_net &
wait

# --- Round 2: Edge-Conditioned MP 효과 ---
echo -e "\n--- Round 2: Edge-Conditioned MP 도입 효과 ---"
run_exp 0 EC3     --num_layers 3 --use_edge_attr &
run_exp 1 JK3_EC  --num_layers 3 --use_jk_net --use_edge_attr &
wait

# --- Round 3: 모델 경량화 (Layer Reduction) + Edge-Conditioning ---
echo -e "\n--- Round 3: Layer 축소 (다이어트) ---"
run_exp 0 L1_EC   --num_layers 1 --use_edge_attr &
run_exp 1 L2_EC   --num_layers 2 --use_edge_attr &
wait

echo -e "\n============================================================"
echo "🎉 v4 Phase A Ablation Study 완료!"
echo "   결과 로그: logs/ablation_v4_*.log"
echo "============================================================"
