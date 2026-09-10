#!/usr/bin/env bash
set -euo pipefail

# Generated sweep commands only. Review and run selected lines when ready.

# Training commands
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p4_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p4_ldrc_fixed_seed202_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p55_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p55_ldrc_fixed_seed202_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p7_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/c_safe/configs/c_safe_0p7_ldrc_fixed_seed202_steps50000.yaml

# Evaluation commands
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p4_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p4/seed_101/c_safe_0p4_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p4/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p4_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p4/seed_202/c_safe_0p4_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p4/seed_202/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p55_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p55/seed_101/c_safe_0p55_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p55/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p55_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p55/seed_202/c_safe_0p55_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p55/seed_202/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p7_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p7/seed_101/c_safe_0p7_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p7/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/c_safe/configs/c_safe_0p7_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/c_safe/train/c_safe_0p7/seed_202/c_safe_0p7_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/c_safe/eval/c_safe_0p7/seed_202/eval_metrics.csv
