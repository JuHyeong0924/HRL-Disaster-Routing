#!/bin/bash
source /home/sem/miniconda3/etc/profile.d/conda.sh
conda activate rl

WORKER="logs/rl_worker_stage/2026-06-17_141421_worker/best.pt"
MANAGER="logs/rl_manager_stage/2026-06-17_144745_manager/best_manager.pt"

python scripts/evaluate.py --mode benchmark --map Anaheim --episodes 20 --worker_ckpt $WORKER --manager_ckpt $MANAGER
