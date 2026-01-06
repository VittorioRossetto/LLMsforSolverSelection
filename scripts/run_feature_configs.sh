#!/usr/bin/env bash
# Run all feature-related benchmark_chat configurations in sequence.
# By default this script runs in dry-run mode (adds `--dry-run` to each call)
# to avoid making real API calls. To perform real runs set `EXEC_REAL=1` in
# the environment when invoking the script.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
SCRIPT_CHAT="$REPO_ROOT/test/benchmark_chat.py"
SCRIPT_PAR="$REPO_ROOT/test/benchmark_parallel.py"
MZN2FEAT_FILE="test/data/mzn2feat_all_features.json"
LOG_DIR="$REPO_ROOT/logs/feature_runs"
SUMMARY_LOG="$LOG_DIR/summary.log"

mkdir -p "$LOG_DIR"

DRY_FLAG="--dry-run"
if [ "${EXEC_REAL:-0}" = "1" ]; then
  DRY_FLAG=""
fi

declare -a CONFIGS=(

  # # All feature configurations
  # "featOnly|--features-only --include-features"
  # "featOnly_Pdesc|--features-only --include-features --include-problem-desc"
  # "featOnly_Sdesc|--features-only --include-features --include-solver-desc"
  # "featOnly_Pdesc_Sdesc|--features-only --include-features --include-problem-desc --include-solver-desc"

  # "modelFeat|--model-and-features --include-features"
  # "modelFeat_Pdesc|--model-and-features --include-features --include-problem-desc"
  # "modelFeat_Sdesc|--model-and-features --include-features --include-solver-desc"
  # "modelFeat_Pdesc_Sdesc|--model-and-features --include-features --include-problem-desc --include-solver-desc"

  # "features|--include-features"
  # "features_Pdesc|--include-features --include-problem-desc"
  # "features_Sdesc|--include-features --include-solver-desc"
  # "features_Pdesc_Sdesc|--include-features --include-problem-desc --include-solver-desc"

  # Best-performing configurations
  "base|"
  "base_Pdesc_Sdesc|--include-problem-desc --include-solver-desc"
  "featOnly_Pdesc|--features-only --include-features --include-problem-desc"
  "featOnly_Sdesc|--features-only --include-features --include-solver-desc"
  "featOnly_Pdesc_Sdesc|--features-only --include-features --include-problem-desc --include-solver-desc"
)

# Temperature sweep for each configuration (passed through to benchmark_chat.py)
declare -a TEMPS=(
  "0.2"
  "0.0"
  "0.8"
  "0.3"
  "0.7"
)

echo "Run started at $(date -u)" | tee -a "$SUMMARY_LOG"
echo "EXEC_REAL=${EXEC_REAL:-0}" | tee -a "$SUMMARY_LOG"

run_one() {
  local name="$1"
  local args="$2"
  local temp="$3"
  local temp_tag
  temp_tag="t${temp}"
  local run_log="$LOG_DIR/${name}.log"

  # include temp in log filename so we don't overwrite
  run_log="$LOG_DIR/${name}_${temp_tag}.log"
  echo "\n--- [$name] [$temp_tag] START $(date -u) ---" | tee -a "$SUMMARY_LOG" "$run_log"

  # parse args safely into an array (avoid word-splitting surprises)
  local -a arg_array
  if [ -n "${args}" ]; then
    read -r -a arg_array <<< "$args"
  else
    arg_array=()
  fi

  local -a cmd
  if [ "$name" = "base" ]; then
    # Base config uses benchmark_parallel.py (no features/mzn2feat flags)
    cmd=("$PY" "$SCRIPT_PAR" "${arg_array[@]}" --temperature "$temp" --best-only)
  else
    # All other configs use benchmark_chat.py
    cmd=("$PY" "$SCRIPT_CHAT" "${arg_array[@]}" --temperature "$temp" --mzn2feat-file "$MZN2FEAT_FILE" --best-only)
  fi
  if [ -n "${DRY_FLAG}" ]; then
    cmd+=("$DRY_FLAG")
  fi
  echo "Running: ${cmd[*]}" | tee -a "$SUMMARY_LOG" "$run_log"

  # run the command, capture stdout+stderr
  if [ "${EXEC_REAL:-0}" = "1" ]; then
    "${cmd[@]}" 2>&1 | tee -a "$run_log"
    rc=${PIPESTATUS[0]:-0}
  else
    # dry-run: keep output but avoid running long tasks
    "${cmd[@]}" 2>&1 | tee -a "$run_log"
    rc=${PIPESTATUS[0]:-0}
  fi

  echo "--- [$name] [$temp_tag] END $(date -u) rc=$rc ---" | tee -a "$SUMMARY_LOG" "$run_log"
  return $rc
}

# iterate and run
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name args <<< "$entry"
  for temp in "${TEMPS[@]}"; do
    run_one "$name" "$args" "$temp"
  done
done

echo "Run finished at $(date -u)" | tee -a "$SUMMARY_LOG"

echo "Logs written to: $LOG_DIR"

exit 0
