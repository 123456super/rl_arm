#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python}"

ROOT="outputs/reaching_incremental/s1_static_terminal_refine_diagnostics"
FINAL_MANIFEST="configs/experiments/reaching_recovery/manifests/v1_final.json"
FEASIBILITY_CONFIG="configs/experiments/reaching_incremental/s1_static_zero_speed_precheck.yaml"
FEASIBILITY_SOURCE="outputs/reaching_incremental/s1_static_failure_diagnostics/feasibility/static_zero_speed_v1_final.json"
FEASIBILITY_MERGED="${ROOT}/feasibility/static_zero_speed_v1_final.json"

config_for_seed() {
  echo "configs/experiments/reaching_incremental/s1_static_candidate_terminal_refine_seed${1}.yaml"
}

run_dir_for_seed() {
  case "$1" in
    4301) echo "outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4301/link_fixed_static_candidate_terminal_refine_seed4301_steps560000" ;;
    4302) echo "outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4302/link_fixed_static_candidate_terminal_refine_seed4302_steps600000" ;;
    4303) echo "outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4303/link_fixed_static_candidate_terminal_refine_seed4303_steps540000" ;;
    *) echo "unknown seed: $1" >&2; exit 2 ;;
  esac
}

selected_checkpoint_for_seed() {
  "${PYTHON_CMD[@]}" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "checkpoint_selection" / "selected_checkpoint.csv"
with path.open(newline="", encoding="utf-8") as file:
    print(next(csv.DictReader(file))["checkpoint"])
' "$(run_dir_for_seed "$1")"
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

copy_feasibility() {
  mkdir -p "${ROOT}/feasibility"
  if [[ -f "${FEASIBILITY_SOURCE}" ]]; then
    cp "${FEASIBILITY_SOURCE}" "${FEASIBILITY_MERGED}"
  else
    "${PYTHON_CMD[@]}" scripts/audit_vaps_task_feasibility.py \
      --config "${FEASIBILITY_CONFIG}" --seed-manifest "${FINAL_MANIFEST}" \
      --output "${ROOT}/feasibility/full.json"
    cp "${ROOT}/feasibility/full.json" "${FEASIBILITY_MERGED}"
  fi
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

build_manifests() {
  "${PYTHON_CMD[@]}" scripts/build_s1_failure_manifests.py \
    --diagnostics "${ROOT}/actor_episode_diagnostics.csv" \
    --output-dir configs/experiments/reaching_incremental/manifests
}

case "${STAGE}" in
  eval-trace-4301) eval_trace 4301 ;;
  eval-trace-4302) eval_trace 4302 ;;
  eval-trace-4303) eval_trace 4303 ;;
  copy-feasibility) copy_feasibility ;;
  analyze) analyze ;;
  build-manifests) build_manifests ;;
  *) echo "usage: $0 {eval-trace-4301|eval-trace-4302|eval-trace-4303|copy-feasibility|analyze|build-manifests}" >&2; exit 2 ;;
esac
