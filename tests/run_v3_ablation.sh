#!/bin/bash
# =============================================================================
# v3 HRL Worker Ablation Study Runner
# GPU 0/1 병렬 실행, steps=500으로 빠른 스크리닝
# =============================================================================

set -e

# 공통 파라미터 (Baseline 확정 설정 기반)
COMMON="--stage worker --steps 500 --batch_size 16 --use_gae --entropy_coeff 0.01 --zone_progress_reward --lr 5e-4"

# conda 환경 활성화 함수
run_exp() {
    local GPU=$1
    local ABLATION_ID=$2
    shift 2
    local EXTRA_ARGS="$@"
    
    echo "🚀 [GPU $GPU] 시작: $ABLATION_ID | $EXTRA_ARGS"
    CUDA_VISIBLE_DEVICES=$GPU python -u train_rl.py $COMMON --ablation "$ABLATION_ID" $EXTRA_ARGS \
        > "logs/ablation_v3_${ABLATION_ID}.log" 2>&1
    echo "✅ [GPU $GPU] 완료: $ABLATION_ID"
}

# 로그 디렉토리 생성
mkdir -p logs

echo "============================================================"
echo "🧪 v3 Ablation Study 시작 (총 11개 실험, GPU 0/1 병렬)"
echo "   공통: $COMMON"
echo "   Baseline(BL): 기존 실행 결과 재사용"
echo "============================================================"

# --- Round 1: Masking 변인 (hard_full_seq vs soft_curr_next) ---
echo -e "\n--- Round 1 ---"
run_exp 0 M_HFS   --masking_mode hard_full_seq &
run_exp 1 M_SCN   --masking_mode soft_curr_next &
wait

# --- Round 2: Masking 변인 (soft_flex) + PBRS on Hard ---
echo -e "\n--- Round 2 ---"
run_exp 0 M_SF    --masking_mode soft_flex &
run_exp 1 P_HARD  --use_pbrs &
wait

# --- Round 3: PBRS + soft_flex + soft_curr_next+PBRS ---
echo -e "\n--- Round 3 ---"
run_exp 0 P_SF    --masking_mode soft_flex --use_pbrs &
run_exp 1 SCN_P   --masking_mode soft_curr_next --use_pbrs &
wait

# --- Round 4: GATv2 레이어 변인 (1 vs 3) ---
echo -e "\n--- Round 4 ---"
run_exp 0 L1      --num_layers 1 &
run_exp 1 L3      --num_layers 3 &
wait

# --- Round 5: GATv2 레이어 변인 (4) + 넓은 시야+자유도 ---
echo -e "\n--- Round 5 ---"
run_exp 0 L4      --num_layers 4 &
run_exp 1 L3_SF   --masking_mode soft_flex --num_layers 3 &
wait

# --- Round 6: 모든 해결책 결합 (BEST) ---
echo -e "\n--- Round 6 ---"
run_exp 0 BEST    --masking_mode soft_flex --use_pbrs --num_layers 3 &
wait

echo -e "\n============================================================"
echo "🎉 v3 Ablation Study 완료!"
echo "   결과 로그: logs/ablation_v3_*.log"
echo "============================================================"
