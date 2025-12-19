#!/usr/bin/env bash
# Run all feature-related benchmark_chat configurations in sequence.
# By default this script runs in dry-run mode (adds `--dry-run` to each call)
# to avoid making real API calls. To perform real runs set `EXEC_REAL=1` in
# the environment when invoking the script.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
SCRIPT="$REPO_ROOT/test/benchmark_chat.py"
MZN2FEAT_FILE="test/data/mzn2feat_all_features.json"
LOG_DIR="$REPO_ROOT/logs/feature_runs"
SUMMARY_LOG="$LOG_DIR/summary.log"

mkdir -p "$LOG_DIR"

DRY_FLAG="--dry-run"
if [ "${EXEC_REAL:-0}" = "1" ]; then
  DRY_FLAG=""
fi

declare -a CONFIGS=(
  #"featOnly|--features-only --include-features"
  # "featOnly_Pdesc|--features-only --include-features --include-problem-desc"
  # "featOnly_Sdesc|--features-only --include-features --include-solver-desc"
  # "featOnly_Pdesc_Sdesc|--features-only --include-features --include-problem-desc --include-solver-desc"

  # "modelFeat|--model-and-features --include-features"
  # "modelFeat_Pdesc|--model-and-features --include-features --include-problem-desc"
  "modelFeat_Sdesc|--model-and-features --include-features --include-solver-desc"
  "modelFeat_Pdesc_Sdesc|--model-and-features --include-features --include-problem-desc --include-solver-desc"

  "features|--include-features"
  "features_Pdesc|--include-features --include-problem-desc"
  "features_Sdesc|--include-features --include-solver-desc"
  "features_Pdesc_Sdesc|--include-features --include-problem-desc --include-solver-desc"
)

echo "Run started at $(date -u)" | tee -a "$SUMMARY_LOG"
echo "EXEC_REAL=${EXEC_REAL:-0}" | tee -a "$SUMMARY_LOG"

run_one() {
  local name="$1"
  local args="$2"
  local run_log="$LOG_DIR/${name}.log"
  echo "\n--- [$name] START $(date -u) ---" | tee -a "$SUMMARY_LOG" "$run_log"

  cmd=("$PY" "$SCRIPT" $args --mzn2feat-file "$MZN2FEAT_FILE" --best-only $DRY_FLAG)
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

  echo "--- [$name] END $(date -u) rc=$rc ---" | tee -a "$SUMMARY_LOG" "$run_log"
  return $rc
}

# iterate and run
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name args <<< "$entry"
  run_one "$name" "$args"
done

echo "Run finished at $(date -u)" | tee -a "$SUMMARY_LOG"

echo "Logs written to: $LOG_DIR"

exit 0
