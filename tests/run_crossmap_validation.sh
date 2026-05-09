#!/bin/bash
# =============================================================================
# Worker Cross-Map Zero-shot 검증
# 1. Anaheim에서 최적 구성으로 학습
# 2. 다른 맵에 Zone 분할 생성
# 3. 학습된 모델로 Zero-shot 평가 (재학습 없음)
# =============================================================================

set -e

# 최적 구성 파라미터 (Ablation Study 결과 확정)
OPTIMAL="--masking_mode soft_curr_next --use_pbrs --num_layers 3 \
         --use_gae --entropy_coeff 0.01 --zone_progress_reward --lr 5e-4"

echo "============================================================"
echo "🚀 Step 1: Anaheim 최적 구성 학습 (steps=1000, batch=48)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u train_rl.py \
  --stage worker --steps 1000 --batch_size 48 \
  $OPTIMAL --ablation FINAL \
  2>&1 | tee logs/train_final_anaheim.log

echo ""
echo "============================================================"
echo "🗺️ Step 2: 다른 맵 Zone 분할 생성"
echo "============================================================"

python scripts/generate_zones.py

echo ""
echo "============================================================"
echo "📊 Step 3: Cross-Map Zero-shot 평가"
echo "============================================================"

# 최신 Anaheim 체크포인트 경로 자동 탐색
CKPT=$(find logs/rl_worker_stage -name "best.pt" -newer logs/train_final_anaheim.log 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  CKPT=$(find logs/rl_worker_stage -name "best.pt" | sort -t/ -k4 -r | head -1)
fi
echo "   사용 체크포인트: $CKPT"

for MAP in Anaheim SiouxFalls ChicagoSketch Goldcoast; do
  echo ""
  echo "--- $MAP 평가 ---"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_crossmap.py \
    --map "$MAP" --checkpoint "$CKPT" \
    --num_episodes 200 --num_layers 3 \
    $OPTIMAL \
    2>&1 | tee "logs/eval_zeroshot_${MAP}.log"
done

echo ""
echo "============================================================"
echo "🎉 Cross-Map 검증 완료!"
echo "============================================================"
