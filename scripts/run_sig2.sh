#!/usr/bin/env bash
# Run selected benchmark_chat configurations.
# By default this script runs in dry-run mode (adds `--dry-run` to each call)
# to avoid making real API calls. To perform real runs set `EXEC_REAL=1`.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
SCRIPT_CHAT="$REPO_ROOT/test/benchmark_chat.py"
SCRIPT_PAR="$REPO_ROOT/test/benchmark_parallel.py"
MZN2FEAT_FILE="$REPO_ROOT/test/data/mzn2feat_all_features.json"
LOG_DIR="$REPO_ROOT/logs/feature_runs"
SUMMARY_LOG="$LOG_DIR/summary.log"

mkdir -p "$LOG_DIR"

DRY_FLAG="--dry-run"
if [ "${EXEC_REAL:-0}" = "1" ]; then
  DRY_FLAG=""
fi

# name | args | temperature
# temperature = "default" means do not pass --temperature
declare -a CONFIGS=(
  # "base_Pdesc_Sdesc|--include-problem-desc --include-solver-desc --solver-set significative|default"
  # "featOnly_Sdesc|--features-only --include-features --include-solver-desc --solver-set significative|0.3"
  "featOnly_Sdesc|--features-only --include-features --include-solver-desc --solver-set significative|0.7"
)

echo "Run started at $(date -u)" | tee -a "$SUMMARY_LOG"
echo "EXEC_REAL=${EXEC_REAL:-0}" | tee -a "$SUMMARY_LOG"

run_one() {
  local name="$1"
  local args="$2"
  local temp="$3"

  local temp_tag
  if [ "$temp" = "default" ]; then
    temp_tag="tdefault"
  else
    temp_tag="t${temp}"
  fi

  local run_log="$LOG_DIR/${name}_${temp_tag}.log"
  echo "" | tee -a "$SUMMARY_LOG" "$run_log"
  echo "--- [$name] [$temp_tag] START $(date -u) ---" | tee -a "$SUMMARY_LOG" "$run_log"

  local -a arg_array
  if [ -n "${args}" ]; then
    read -r -a arg_array <<< "$args"
  else
    arg_array=()
  fi

  local -a cmd

  if [[ "$name" == base* ]]; then
    cmd=("$PY" "$SCRIPT_PAR" "${arg_array[@]}" --best-only)
  else
    cmd=("$PY" "$SCRIPT_CHAT" "${arg_array[@]}" --mzn2feat-file "$MZN2FEAT_FILE" --best-only)
  fi

  if [ "$temp" != "default" ]; then
    cmd+=(--temperature "$temp")
  fi

  if [ -n "$DRY_FLAG" ]; then
    cmd+=("$DRY_FLAG")
  fi

  echo "Running: ${cmd[*]}" | tee -a "$SUMMARY_LOG" "$run_log"

  "${cmd[@]}" 2>&1 | tee -a "$run_log"
  rc=${PIPESTATUS[0]:-0}

  echo "--- [$name] [$temp_tag] END $(date -u) rc=$rc ---" | tee -a "$SUMMARY_LOG" "$run_log"
  return $rc
}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name args temp <<< "$entry"
  run_one "$name" "$args" "$temp"
done

echo ""
echo "Run finished at $(date -u)" | tee -a "$SUMMARY_LOG"
echo "Logs written to: $LOG_DIR"

exit 0