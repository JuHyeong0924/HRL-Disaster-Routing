#!/bin/bash
set -e

echo "======================================="
echo "🚀 1. Cleaning old logs and checkpoints"
echo "======================================="
rm -rf logs/*
echo "Logs directory cleaned."

echo "======================================="
echo "🚀 2. Training Worker (Phase 1)"
echo "======================================="
python scripts/train_rl.py --stage worker

echo "======================================="
echo "🚀 3. Training Manager (Phase 2)"
echo "======================================="
python scripts/train_rl.py --stage manager

echo "======================================="
echo "🚀 4. Evaluating Algorithms (100 Episodes)"
echo "======================================="
python scripts/evaluate_algorithms.py --episodes 100

echo "======================================="
echo "✅ All pipelines finished successfully!"
echo "======================================="
