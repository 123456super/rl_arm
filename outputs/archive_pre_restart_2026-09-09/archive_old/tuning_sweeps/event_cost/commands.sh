#!/usr/bin/env bash
set -euo pipefail

# Generated sweep commands only. Review and run selected lines when ready.

# Training commands
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_2_k_collision_5_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_2_k_collision_5_ldrc_fixed_seed202_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_3_k_collision_8_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_3_k_collision_8_ldrc_fixed_seed202_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_5_k_collision_10_ldrc_fixed_seed101_steps50000.yaml
conda run -n rl python scripts/train.py --config outputs/sweeps/event_cost/configs/k_violation_5_k_collision_10_ldrc_fixed_seed202_steps50000.yaml

# Evaluation commands
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_2_k_collision_5_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_2_k_collision_5/seed_101/k_violation_2_k_collision_5_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_2_k_collision_5/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_2_k_collision_5_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_2_k_collision_5/seed_202/k_violation_2_k_collision_5_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_2_k_collision_5/seed_202/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_3_k_collision_8_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_3_k_collision_8/seed_101/k_violation_3_k_collision_8_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_3_k_collision_8/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_3_k_collision_8_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_3_k_collision_8/seed_202/k_violation_3_k_collision_8_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_3_k_collision_8/seed_202/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_5_k_collision_10_ldrc_fixed_seed101_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_5_k_collision_10/seed_101/k_violation_5_k_collision_10_ldrc_fixed_seed101_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_5_k_collision_10/seed_101/eval_metrics.csv
conda run -n rl python scripts/evaluate.py --config outputs/sweeps/event_cost/configs/k_violation_5_k_collision_10_ldrc_fixed_seed202_steps50000.yaml --method ldrc_fixed --seed 1001 --episodes 50 --checkpoint outputs/sweeps/event_cost/train/k_violation_5_k_collision_10/seed_202/k_violation_5_k_collision_10_ldrc_fixed_seed202_steps50000/actor.pt --output outputs/sweeps/event_cost/eval/k_violation_5_k_collision_10/seed_202/eval_metrics.csv
