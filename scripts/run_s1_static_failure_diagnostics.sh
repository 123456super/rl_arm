#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python}"

ROOT="outputs/reaching_incremental/s1_static_failure_diagnostics"
FINAL_MANIFEST="configs/experiments/reaching_recovery/manifests/v1_final.json"
FEASIBILITY_CONFIG="configs/experiments/reaching_incremental/s1_static_zero_speed_precheck.yaml"
FEASIBILITY_MERGED="${ROOT}/feasibility/static_zero_speed_v1_final.json"

config_for_seed() {
  local seed="$1"
  echo "configs/experiments/reaching_incremental/s1_static_finetune_seed${seed}.yaml"
}

run_dir_for_seed() {
  local seed="$1"
  case "${seed}" in
    4301) echo "outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000" ;;
    4302) echo "outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000" ;;
    4303) echo "outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000" ;;
    *) echo "unknown seed: ${seed}" >&2; exit 2 ;;
  esac
}

selected_checkpoint_for_seed() {
  local seed="$1"
  local run_dir
  run_dir="$(run_dir_for_seed "${seed}")"
  "${PYTHON_CMD[@]}" -c '
import csv
import pathlib
import sys
path = pathlib.Path(sys.argv[1]) / "checkpoint_selection" / "selected_checkpoint.csv"
with path.open(newline="", encoding="utf-8") as file:
    print(next(csv.DictReader(file))["checkpoint"])
' "${run_dir}"
}

eval_trace() {
  local seed="$1"
  mkdir -p "${ROOT}/eval" "${ROOT}/traces/seed_${seed}"
  "${PYTHON_CMD[@]}" scripts/evaluate.py \
    --config "$(config_for_seed "${seed}")" \
    --checkpoint "$(selected_checkpoint_for_seed "${seed}")" \
    --output "${ROOT}/eval/seed_${seed}_final.csv" \
    --trace-output "${ROOT}/traces/seed_${seed}"
}

feasibility_shard() {
  local shard_index="$1"
  local shard_count="$2"
  mkdir -p "${ROOT}/feasibility"
  "${PYTHON_CMD[@]}" scripts/audit_vaps_task_feasibility.py \
    --config "${FEASIBILITY_CONFIG}" \
    --seed-manifest "${FINAL_MANIFEST}" \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}" \
    --output "${ROOT}/feasibility/shard_${shard_index}_of_${shard_count}.json"
}

merge_feasibility() {
  local shard_count="${1:-4}"
  local args=()
  local index
  for ((index = 0; index < shard_count; index += 1)); do
    args+=(--input "${ROOT}/feasibility/shard_${index}_of_${shard_count}.json")
  done
  "${PYTHON_CMD[@]}" scripts/merge_vaps_task_feasibility_audits.py \
    "${args[@]}" \
    --seed-manifest "${FINAL_MANIFEST}" \
    --output "${FEASIBILITY_MERGED}"
}

analyze() {
  "${PYTHON_CMD[@]}" scripts/analyze_static_obstacle_failures.py \
    --feasibility "${FEASIBILITY_MERGED}" \
    --eval "4301=${ROOT}/eval/seed_4301_final.csv" \
    --eval "4302=${ROOT}/eval/seed_4302_final.csv" \
    --eval "4303=${ROOT}/eval/seed_4303_final.csv" \
    --trace-dir "4301=${ROOT}/traces/seed_4301" \
    --trace-dir "4302=${ROOT}/traces/seed_4302" \
    --trace-dir "4303=${ROOT}/traces/seed_4303" \
    --output-json "${ROOT}/summary.json" \
    --output-csv "${ROOT}/actor_episode_diagnostics.csv" \
    --reset-output-csv "${ROOT}/reset_diagnostics.csv"
}

print_summary() {
  "${PYTHON_CMD[@]}" -c '
import json
from pathlib import Path
path = Path("outputs/reaching_incremental/s1_static_failure_diagnostics/summary.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps({
    "pooled_summary": payload["pooled_summary"],
    "by_failure_mode": {
        key: value["episodes"] for key, value in payload["by_failure_mode"].items()
    },
    "reset_consistency_counts": payload["reset_consistency_counts"],
    "reset_consistency_candidate_path_found": payload["reset_consistency_candidate_path_found"],
}, ensure_ascii=False, indent=2))
'
}

case "${STAGE}" in
  eval-trace-4301) eval_trace 4301 ;;
  eval-trace-4302) eval_trace 4302 ;;
  eval-trace-4303) eval_trace 4303 ;;
  feasibility-shard) feasibility_shard "${2:?shard index required}" "${3:?shard count required}" ;;
  merge-feasibility) merge_feasibility "${2:-4}" ;;
  analyze) analyze ;;
  print-summary) print_summary ;;
  *)
    echo "usage: bash scripts/run_s1_static_failure_diagnostics.sh {eval-trace-4301|eval-trace-4302|eval-trace-4303|feasibility-shard IDX COUNT|merge-feasibility [COUNT]|analyze|print-summary}" >&2
    exit 2
    ;;
esac
