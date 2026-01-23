#!/usr/bin/env python3
"""Unify FlatZinc/MiniZinc constraint names by identical definition text.

Reads a JSON mapping {constraint_name: definition}. If multiple constraint names
share the exact same definition string, they will be merged into a single entry
with a canonical name.

Canonical naming heuristic (default):
- Split names on '_' and take the longest common prefix of tokens across the
  group. If the prefix is empty, fall back to the shortest key.

Optional heuristic:
- With --drop-leading-types, also drop common leading type tokens like
  'int', 'float', 'bool', 'set', 'array', 'var' before finding a common prefix.

Outputs a new JSON mapping and (optionally) a report JSON with grouping and
renames.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TYPE_TOKENS = {
    "int",
    "float",
    "bool",
    "set",
    "var",
    "opt",
    "array",
}


def _normalize_definition(text: str) -> str:
    # Keep it conservative: only trim surrounding whitespace.
    return text.strip()


def _common_prefix_tokens(token_lists: List[List[str]]) -> List[str]:
    if not token_lists:
        return []

    min_len = min(len(toks) for toks in token_lists)
    prefix: List[str] = []
    for i in range(min_len):
        token = token_lists[0][i]
        if all(toks[i] == token for toks in token_lists[1:]):
            prefix.append(token)
        else:
            break
    return prefix


def _strip_leading_type_tokens(tokens: List[str]) -> List[str]:
    # Remove a run of leading generic/type tokens (e.g., int_lin_eq -> lin_eq).
    i = 0
    while i < len(tokens) and tokens[i] in TYPE_TOKENS:
        i += 1
    return tokens[i:] if i < len(tokens) else tokens


def derive_canonical_name(keys: List[str], drop_leading_types: bool) -> str:
    if len(keys) == 1:
        return keys[0]

    token_lists = [k.split("_") for k in keys]

    if drop_leading_types:
        token_lists = [_strip_leading_type_tokens(toks) for toks in token_lists]

    prefix = _common_prefix_tokens(token_lists)
    if prefix:
        return "_".join(prefix)

    # Fallback: shortest key is often the most general.
    return min(keys, key=len)


def unify_descriptions(
    descriptions: Dict[str, str], *, drop_leading_types: bool
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Return (unified_map, rename_map, groups).

    - unified_map: canonical_name -> definition
    - rename_map: old_name -> canonical_name (identity included)
    - groups: canonical_name -> list of original names merged
    """

    def_to_keys: Dict[str, List[str]] = defaultdict(list)
    normalized: Dict[str, str] = {}

    for k, v in descriptions.items():
        norm = _normalize_definition(v)
        normalized[k] = norm
        def_to_keys[norm].append(k)

    unified: Dict[str, str] = {}
    rename: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    # Stable processing order for reproducibility.
    for definition, keys in sorted(def_to_keys.items(), key=lambda kv: (len(kv[1]), kv[0])):
        keys_sorted = sorted(keys)
        canonical = derive_canonical_name(keys_sorted, drop_leading_types=drop_leading_types)

        # Handle potential collisions between different definition groups.
        if canonical in unified and unified[canonical] != definition:
            # Collision: fall back to a unique canonical name.
            # Prefer the shortest original key not yet used.
            for alt in sorted(keys_sorted, key=len):
                if alt not in unified:
                    canonical = alt
                    break
            else:
                # Last resort: append numeric suffix until unique.
                suffix = 2
                base = canonical
                while f"{base}__{suffix}" in unified:
                    suffix += 1
                canonical = f"{base}__{suffix}"

        unified[canonical] = definition
        groups[canonical] = keys_sorted
        for k in keys_sorted:
            rename[k] = canonical

    return unified, rename, groups


def main() -> int:
    ap = argparse.ArgumentParser(description="Unify constraint descriptions by identical definition text.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("fzn_descriptions.json"),
        help="Input JSON mapping (default: tools/fzn_descriptions.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("fzn_descriptions_unified.json"),
        help="Output JSON mapping (default: tools/fzn_descriptions_unified.json)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report JSON path containing renames and groups.",
    )
    ap.add_argument(
        "--drop-leading-types",
        action="store_true",
        help="Also drop leading type tokens (int/float/bool/set/array/var/opt) before deriving canonical names.",
    )

    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    descriptions = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(descriptions, dict):
        raise TypeError("Input JSON must be an object mapping constraint -> definition")

    unified, rename, groups = unify_descriptions(descriptions, drop_leading_types=args.drop_leading_types)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unified, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.report is not None:
        report_obj = {
            "input": str(args.input),
            "output": str(args.output),
            "drop_leading_types": bool(args.drop_leading_types),
            "counts": {
                "original": len(descriptions),
                "unified": len(unified),
                "merged_groups": sum(1 for ks in groups.values() if len(ks) > 1),
            },
            "rename_map": rename,
            "groups": groups,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Original entries: {len(descriptions)}")
    print(f"Unified entries:  {len(unified)}")
    merged = sum(1 for ks in groups.values() if len(ks) > 1)
    print(f"Merged groups:    {merged}")
    print(f"Wrote: {args.output}")
    if args.report is not None:
        print(f"Report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
