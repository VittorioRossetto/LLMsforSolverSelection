#!/usr/bin/env bash
set -euo pipefail

# Compiles MiniZinc models for each mznc2025_probs subfolder into FlatZinc and
# captures `tools/fzn_parser.py` output per instance in a JSON file.
#
# Default command (per instance):
#   /home/vro5/minizinc-custom/bin/minizinc -O3 -c --solver mzn-fzn \
#     -I /home/vro5/Coding/minizincHLstdlib/std <model>.mzn <instance>.dzn/.json
#
# Outputs (default):
#   <root>/fzn_parser_outputs.json

ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBS_DIR_DEFAULT="$ROOT_DEFAULT/mznc2025_probs"

MINIZINC_BIN_DEFAULT="/home/vro5/minizinc-custom/bin/minizinc"
STDLIB_DIR_DEFAULT="/home/vro5/Coding/minizincHLstdlib/std"
PYTHON_DEFAULT="${PYTHON:-python3}"

PROBS_DIR="$PROBS_DIR_DEFAULT"
OUT_JSON="$PROBS_DIR_DEFAULT/fzn_parser_outputs.json"
OUT_JSON_EXPLICIT=0
ONLY_SUBDIR=""
ALLOW_UNBOUNDED_VARS=0
CATEGORIZE_CONSTRAINTS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--probs-dir DIR] [--out JSON] [--only SUBDIR] [--minizinc BIN] [--stdlib DIR] [--python PY] [--allow-unbounded-vars]
                       [--categorize-constraints]

Defaults:
  --probs-dir  $PROBS_DIR_DEFAULT
  --out        $PROBS_DIR_DEFAULT/fzn_parser_outputs.json
  --minizinc   $MINIZINC_BIN_DEFAULT
  --stdlib     $STDLIB_DIR_DEFAULT
  --python     python3 (or PYTHON env var)

Notes:
  - For each subfolder, the script tries to choose a primary .mzn model:
    1) <subfolder>.mzn if it exists
    2) otherwise, the only *.mzn not ending with _commented.mzn
    3) otherwise, the first *.mzn not ending with _commented.mzn
  - Keys in the output JSON are "<subfolder>/<instance_filename>".
  - --allow-unbounded-vars forwards the MiniZinc/Gecode flag of the same name.
  - --categorize-constraints forwards the parser flag of the same name.
  - If --categorize-constraints is set and --out is not provided, the default output
    changes to <probs-dir>/fzn_parser_outputs_categorized.json.
EOF
}

MINIZINC_BIN="$MINIZINC_BIN_DEFAULT"
STDLIB_DIR="$STDLIB_DIR_DEFAULT"
PYTHON_BIN="$PYTHON_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probs-dir)
      PROBS_DIR="$2"; shift 2 ;;
    --out)
      OUT_JSON="$2"; OUT_JSON_EXPLICIT=1; shift 2 ;;
    --only)
      ONLY_SUBDIR="$2"; shift 2 ;;
    --minizinc)
      MINIZINC_BIN="$2"; shift 2 ;;
    --stdlib)
      STDLIB_DIR="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --allow-unbounded-vars)
      ALLOW_UNBOUNDED_VARS=1; shift 1 ;;
    --categorize-constraints)
      CATEGORIZE_CONSTRAINTS=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# If categorized output is requested and the user didn't pick an explicit output path,
# avoid overwriting the standard output JSON.
if [[ "$CATEGORIZE_CONSTRAINTS" -eq 1 && "$OUT_JSON_EXPLICIT" -eq 0 ]]; then
  OUT_JSON="$PROBS_DIR/fzn_parser_outputs_categorized.json"
fi

if [[ ! -d "$PROBS_DIR" ]]; then
  echo "Problems dir not found: $PROBS_DIR" >&2
  exit 1
fi
if [[ ! -x "$MINIZINC_BIN" ]]; then
  echo "MiniZinc binary not executable: $MINIZINC_BIN" >&2
  exit 1
fi
if [[ ! -d "$STDLIB_DIR" ]]; then
  echo "Stdlib dir not found: $STDLIB_DIR" >&2
  exit 1
fi

PARSER="$ROOT_DEFAULT/tools/fzn_parser.py"
if [[ ! -f "$PARSER" ]]; then
  echo "Parser not found: $PARSER" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"
if [[ ! -f "$OUT_JSON" ]]; then
  echo "{}" > "$OUT_JSON"
fi

json_set_from_stdin() {
  local json_path="$1"
  local key="$2"
  "$PYTHON_BIN" -c 'import json,sys
from pathlib import Path
path = Path(sys.argv[1])
key = sys.argv[2]
val = sys.stdin.read()
try:
  data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
except json.JSONDecodeError:
  data = {}
if not isinstance(data, dict):
  data = {}
data[key] = val
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
' "$json_path" "$key"
}

pick_model_file() {
  local subdir="$1"
  local subname
  subname="$(basename "$subdir")"

  if [[ -f "$subdir/$subname.mzn" ]]; then
    echo "$subdir/$subname.mzn"
    return 0
  fi

  # Candidates: any .mzn not ending in _commented.mzn
  mapfile -t cands < <(find "$subdir" -maxdepth 1 -type f -name "*.mzn" ! -name "*_commented.mzn" | sort)
  if [[ ${#cands[@]} -eq 1 ]]; then
    echo "${cands[0]}"
    return 0
  fi
  if [[ ${#cands[@]} -gt 1 ]]; then
    echo "${cands[0]}"
    return 0
  fi

  return 1
}

run_one_subdir() {
  local subdir="$1"
  local subname
  subname="$(basename "$subdir")"

  local model_path
  if ! model_path="$(pick_model_file "$subdir")"; then
    echo "[skip] $subname: no .mzn model found" >&2
    return 0
  fi

  local model_base
  model_base="$(basename "$model_path" .mzn)"

  mapfile -t instances < <(find "$subdir" -maxdepth 1 -type f \( -name "*.dzn" -o -name "*.json" \) | sort)
  if [[ ${#instances[@]} -eq 0 ]]; then
    echo "[skip] $subname: no .dzn/.json instances found" >&2
    return 0
  fi

  echo "[dir] $subname: model $(basename "$model_path"), instances ${#instances[@]}" >&2

  local inst_path inst_file key fzn_path
  fzn_path="$subdir/$model_base.fzn"

  for inst_path in "${instances[@]}"; do
    inst_file="$(basename "$inst_path")"
    key="$subname/$inst_file"

    echo "  [compile] $key" >&2
    (
      cd "$subdir"
      extra_flags=()
      if [[ "$ALLOW_UNBOUNDED_VARS" -eq 1 ]]; then
        extra_flags+=(--allow-unbounded-vars)
      fi
      "$MINIZINC_BIN" -O3 -c --solver mzn-fzn -I "$STDLIB_DIR" "${extra_flags[@]}" "$model_path" "$inst_path" >/dev/null
    )

    if [[ ! -f "$fzn_path" ]]; then
      echo "[warn] missing fzn after compile: $fzn_path" >&2
      printf 'ERROR: expected %s to be created by MiniZinc compile\n' "$fzn_path" | json_set_from_stdin "$OUT_JSON" "$key"
      continue
    fi

    echo "  [parse] $key" >&2
    # Capture all output, including possible errors.
    parser_flags=()
    if [[ "$CATEGORIZE_CONSTRAINTS" -eq 1 ]]; then
      parser_flags+=(--categorize-constraints)
    fi
    parser_out="$($PYTHON_BIN "$PARSER" "$fzn_path" "${parser_flags[@]}" 2>&1 || true)"
    printf "%s" "$parser_out" | json_set_from_stdin "$OUT_JSON" "$key"
  done
}

if [[ -n "$ONLY_SUBDIR" ]]; then
  target="$PROBS_DIR/$ONLY_SUBDIR"
  if [[ ! -d "$target" ]]; then
    echo "Subdir not found: $target" >&2
    exit 1
  fi
  run_one_subdir "$target"
  echo "Done. Wrote $OUT_JSON" >&2
  exit 0
fi

while IFS= read -r -d '' subdir; do
  run_one_subdir "$subdir"
done < <(find "$PROBS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "Done. Wrote $OUT_JSON" >&2
