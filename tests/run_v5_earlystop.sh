#!/bin/bash
# v5 HRL Worker Early Stopping Ablation (1000 Steps)

source /home/sem/miniconda3/etc/profile.d/conda.sh
conda activate rl

TIMESTAMP=$(date +"%Y-%m-%d_%H%M")
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 공통 하이퍼파라미터
COMMON_ARGS="--stage worker --use_jk_net --use_edge_attr --masking_mode soft_curr_next --use_pbrs --batch_size 16 --num_layers 3 --steps 1000"

echo "=========================================================="
echo "🚀 [v5 Early Stop] 병렬 학습 시작 (총 1000 스텝)"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0 python -u train_rl.py $COMMON_ARGS --subgoal_mode zone --ablation SG_ZONE_early > "${LOG_DIR}/ablation_v5_SG_ZONE_early.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -u train_rl.py $COMMON_ARGS --subgoal_mode node --ablation SG_NODE_early > "${LOG_DIR}/ablation_v5_SG_NODE_early.log" 2>&1 &
PID2=$!

wait $PID1
wait $PID2

echo "✅ 모든 학습이 완료되었습니다."

# 체크포인트 경로 자동 탐색
CKPT_ZONE=$(ls -d ${LOG_DIR}/rl_worker_stage/*worker_B16_SG_ZONE_EARLY | sort | tail -n 1)/best.pt
CKPT_NODE=$(ls -d ${LOG_DIR}/rl_worker_stage/*worker_B16_SG_NODE_EARLY | sort | tail -n 1)/best.pt

echo "=========================================================="
echo "🗺️ Cross-map Zero-shot 평가 시작"
echo "=========================================================="

EVAL_ARGS="--num_episodes 200 --use_jk_net --use_edge_attr --masking_mode soft_curr_next --use_pbrs --num_layers 3"

for MAP in Anaheim SiouxFalls ChicagoSketch Goldcoast; do
    echo "--- $MAP ZONE ---"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_crossmap.py --map "$MAP" --checkpoint "$CKPT_ZONE" $EVAL_ARGS --subgoal_mode zone > "${LOG_DIR}/eval_early_${MAP}_ZONE.log" 2>&1 &
    EPID1=$!
    
    echo "--- $MAP NODE ---"
    CUDA_VISIBLE_DEVICES=1 python -u scripts/evaluate_crossmap.py --map "$MAP" --checkpoint "$CKPT_NODE" $EVAL_ARGS --subgoal_mode node > "${LOG_DIR}/eval_early_${MAP}_NODE.log" 2>&1 &
    EPID2=$!
    
    wait $EPID1
    wait $EPID2
done

echo "✅ [v5 Early Stop] 전체 파이프라인 완료!"
