#!/bin/bash
set -e

PYTHON="/home/sem/miniconda3/envs/rl/bin/python"
MAP="Anaheim"
WORKER_BATCH=32
MANAGER_BATCH=32
WORKER_EPISODES=5000
MANAGER_EPISODES=5000
OOB_PENALTY=-1.0

run_experiment() {
  local exp_id=$1
  local use_is_visited=$2
  local use_global_pool=$3

  echo "================================================================="
  echo " Running Ablation ${exp_id}"
  echo " use_is_visited: ${use_is_visited}, use_global_pool: ${use_global_pool}"
  echo "================================================================="

  # Build flags
  local eval_flags=""
  if [ "$use_is_visited" = "true" ]; then eval_flags="$eval_flags --use_is_visited"; fi
  if [ "$use_global_pool" = "true" ]; then eval_flags="$eval_flags --use_global_pool"; fi
  local worker_flags="--mini_batch_size 256 $eval_flags"

  # Step 1: Train Worker
  local worker_exp="ModelB_Ablation${exp_id}_Worker"
  local worker_ckpt=$(ls -td logs/rl_worker_stage/*${worker_exp}*/best.pt 2>/dev/null | head -n 1)
  
  if [ -z "$worker_ckpt" ]; then
    echo "[Step 1] Training Worker: ${worker_exp}"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_rl.py \
      --stage worker \
      --map $MAP \
      --episodes $WORKER_EPISODES \
      --batch_size $WORKER_BATCH \
      --use_pbrs \
      --use_relative_hop \
      --oob_penalty $OOB_PENALTY \
      $worker_flags \
      --exp_name $worker_exp

    worker_ckpt=$(ls -td logs/rl_worker_stage/*${worker_exp}*/best.pt 2>/dev/null | head -n 1)
    if [ -z "$worker_ckpt" ]; then echo "❌ Worker checkpoint not found!"; exit 1; fi
  else
    echo "⏩ Worker already trained. Skipping Step 1."
  fi
  echo "✅ Worker Checkpoint: ${worker_ckpt}"

  # Step 2: Train Manager
  local manager_exp="ModelB_Ablation${exp_id}_Manager"
  local manager_ckpt=$(ls -td logs/rl_manager_stage/*${manager_exp}*/best.pt 2>/dev/null | head -n 1)

  if [ -z "$manager_ckpt" ]; then
    echo "[Step 2] Training Manager: ${manager_exp}"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_rl.py \
      --stage manager \
      --map $MAP \
      --episodes $MANAGER_EPISODES \
      --batch_size $MANAGER_BATCH \
      --worker_ckpt $worker_ckpt \
      $worker_flags \
      --exp_name $manager_exp

    manager_ckpt=$(ls -td logs/rl_manager_stage/*${manager_exp}*/best.pt 2>/dev/null | head -n 1)
    if [ -z "$manager_ckpt" ]; then echo "❌ Manager checkpoint not found!"; exit 1; fi
  else
    echo "⏩ Manager already trained. Skipping Step 2."
  fi
  echo "✅ Manager Checkpoint: ${manager_ckpt}"

  # Step 3: Evaluation
  echo "[Step 3] Evaluation"
  for EVAL_MAP in anaheim chicago berlin-mitte berlin-friedrichshain; do
    echo "-----------------------------------"
    echo "Evaluating on: ${EVAL_MAP}"
    CUDA_VISIBLE_DEVICES=0 $PYTHON evaluate.py \
      --map $EVAL_MAP \
      --worker_ckpt $worker_ckpt \
      --manager_ckpt $manager_ckpt \
      $eval_flags \
      --save_prefix "ModelB_Ablation${exp_id}_"
    echo ""
  done
}

# Run experiments
run_experiment "A" "false" "false"
# run_experiment "B" "true" "false"
# run_experiment "C" "false" "true"
# run_experiment "D" "true" "true"

echo "========================================"
echo " All Ablation Experiments Completed!"
echo "========================================"
