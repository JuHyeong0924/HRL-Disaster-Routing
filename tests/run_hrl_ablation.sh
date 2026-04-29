#!/bin/bash
# HRL Worker Phase 1: 확장 Ablation Study
# GPU 당 1개씩 (2개 동시) — OOM 방지 (각 ~10GB VRAM)
# 16개 실험 × 5000ep = 8 Wave × ~5분 = 총 ~40분

source /home/sem/miniconda3/bin/activate rl

EPISODES=5000
POMO=16
BASE_LR=3e-4
LOG_DIR="logs/rl_worker_stage"

echo "=============================================="
echo "  HRL Worker Phase 1: Ablation Study"
echo "  GPU 0 + GPU 1 (각 1개씩, 2개 동시)"
echo "  Episodes: $EPISODES | 총 16개 실험"
echo "=============================================="

run_exp() {
    local gpu_id=$1
    local ablation_id=$2
    shift 2
    echo "[START] GPU $gpu_id | $ablation_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python train_rl.py \
        --stage worker --episodes $EPISODES --num_pomo $POMO \
        --ablation "$ablation_id" \
        "$@" \
        > "${LOG_DIR}/ablation_${ablation_id}.log" 2>&1
    echo "[DONE]  GPU $gpu_id | $ablation_id (exit=$?)"
}

mkdir -p "$LOG_DIR"

echo "=== Wave 1/8 ==="
run_exp 0 "BL"           --lr $BASE_LR &
run_exp 1 "P0_ZONE"      --lr $BASE_LR --zone_progress_reward &
wait

echo "=== Wave 2/8 ==="
run_exp 0 "P1_GAE"       --lr $BASE_LR --use_gae --entropy_coeff 0.01 &
run_exp 1 "P2_COSLR"     --lr 5e-4 --use_cosine_lr &
wait

echo "=== Wave 3/8 ==="
run_exp 0 "P0P1"         --lr $BASE_LR --zone_progress_reward --use_gae --entropy_coeff 0.01 &
run_exp 1 "P0P2"         --lr 5e-4 --zone_progress_reward --use_cosine_lr &
wait

echo "=== Wave 4/8 ==="
run_exp 0 "P0P1P2"       --lr 5e-4 --zone_progress_reward --use_gae --entropy_coeff 0.01 --use_cosine_lr &
run_exp 1 "P1P2"         --lr 5e-4 --use_gae --entropy_coeff 0.01 --use_cosine_lr &
wait

echo "=== Wave 5/8 ==="
run_exp 0 "ENT_005"      --lr $BASE_LR --zone_progress_reward --use_gae --entropy_coeff 0.005 &
run_exp 1 "ENT_02"       --lr $BASE_LR --zone_progress_reward --use_gae --entropy_coeff 0.02 &
wait

echo "=== Wave 6/8 ==="
run_exp 0 "ENT_05"       --lr $BASE_LR --zone_progress_reward --use_gae --entropy_coeff 0.05 &
run_exp 1 "LR_1E4"       --lr 1e-4 --zone_progress_reward --use_gae --entropy_coeff 0.01 &
wait

echo "=== Wave 7/8 ==="
run_exp 0 "LR_5E4"       --lr 5e-4 --zone_progress_reward --use_gae --entropy_coeff 0.01 &
run_exp 1 "LR_1E3"       --lr 1e-3 --zone_progress_reward --use_gae --entropy_coeff 0.01 &
wait

echo "=== Wave 8/8 ==="
run_exp 0 "ACCUM_32"     --lr $BASE_LR --num_pomo 32 --zone_progress_reward --use_gae --entropy_coeff 0.01 &
run_exp 1 "ACCUM_64"     --lr $BASE_LR --num_pomo 64 --zone_progress_reward --use_gae --entropy_coeff 0.01 &
wait

echo ""
echo "=============================================="
echo "  ✅ 전체 Ablation 완료! (16개 실험)"
echo "=============================================="
