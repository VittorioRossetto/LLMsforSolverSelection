#!/bin/bash

# Run benchmark_all_solvers.py with all combinations of solver set and script version

SOLVER_SETS=(minizinc all)
SCRIPT_VERSIONS=(uncommented)

for solver_set in "${SOLVER_SETS[@]}"; do
  for script_version in "${SCRIPT_VERSIONS[@]}"; do
  log_file="benchmark_${solver_set}_${script_version}.log"
  echo -e "\n=== Running: solver_set=$solver_set, script_version=$script_version ===" | tee -a "$log_file"
  python3 benchmark_all_solvers.py --solver-set "$solver_set" --script-version "$script_version" 2>&1 | tee -a "$log_file"
  done
done

echo "\nAll configurations completed."
