#!/usr/bin/env python3
"""Categorize (unified) FlatZinc/MiniZinc constraint descriptions into spec sections.

This script is meant to work with the outputs of tools/unify_fzn_descriptions.py:
- fzn_descriptions_unified.json (canonical_name -> definition)
- fzn_descriptions_unified_report.json (canonical_name -> list of original names)

We categorize *canonical* constraints by matching ANY original name in the group
against the category lists (as provided by the user, from the MiniZinc docs).

A canonical constraint can belong to multiple categories (e.g., deprecated +
counting); by default we place it into every matching category.

Outputs a JSON file with:
- _meta: counts
- categories: {category_name: {canonical_name: definition}}
- uncategorized: {canonical_name: definition}
- multi_category: {canonical_name: [category_name, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def build_categories() -> Dict[str, Set[str]]:
    # Category names are kept close to the user-provided section headers.
    return {
        "4.2.2.1 All-Different and related constraints": {
            "all_different",
            "all_different_except",
            "all_different_except_0",
            "all_disjoint",
            "all_equal",
            "nvalue",
            "symmetric_all_different",
        },
        "4.2.2.2 Lexicographic constraints": {
            "lex2",
            "lex2_strict",
            "lex_chain",
            "lex_chain_greater",
            "lex_chain_greatereq",
            "lex_chain_greatereq_orbitope",
            "lex_chain_less",
            "lex_chain_lesseq",
            "lex_chain_lesseq_orbitope",
            "lex_greater",
            "lex_greatereq",
            "lex_less",
            "lex_lesseq",
            "seq_precede_chain",
            "strict_lex2",
            "value_precede",
            "value_precede_chain",
            "var_perm_sym",
            "var_sqr_sym",
        },
        "4.2.2.3 Sorting constraints": {
            "arg_sort",
            "decreasing",
            "increasing",
            "sort",
            "strictly_decreasing",
            "strictly_increasing",
        },
        "4.2.2.4 Channeling constraints": {
            "int_set_channel",
            "inverse",
            "inverse_in_range",
            "inverse_set",
            "link_set_to_booleans",
        },
        "4.2.2.5 Counting constraints": {
            "among",
            "at_least",
            "at_most",
            "at_most1",
            "count",
            "count_eq",
            "count_geq",
            "count_gt",
            "count_leq",
            "count_lt",
            "count_neq",
            "distribute",
            "exactly",
            "global_cardinality",
            "global_cardinality_closed",
        },
        "4.2.2.6 Array-related constraints": {
            "element",
            "member",
            "write",
            "writes",
            "writes_seq",
        },
        "4.2.2.7 Set-related constraints": {
            "disjoint",
            "partition_set",
            "roots",
        },
        "4.2.2.8 Mathematical constraints": {
            "arg_max",
            "arg_max_weak",
            "arg_min",
            "arg_min_weak",
            "arg_val",
            "arg_val_weak",
            "maximum",
            "maximum_arg",
            "minimum",
            "minimum_arg",
            "piecewise_linear",
            "range",
            "sliding_sum",
            "sum_pred",
            "sum_set",
        },
        "4.2.2.9 Packing constraints": {
            "bin_packing",
            "bin_packing_capa",
            "bin_packing_load",
            "diffn",
            "diffn_k",
            "diffn_nonstrict",
            "diffn_nonstrict_k",
            "geost",
            "geost_bb",
            "geost_nonoverlap_k",
            "geost_smallest_bb",
            "knapsack",
        },
        "4.2.2.10 Scheduling constraints": {
            "alternative",
            "cumulative",
            "cumulatives",
            "disjunctive",
            "disjunctive_strict",
            "span",
        },
        "4.2.2.11 Graph constraints": {
            "bounded_dpath",
            "bounded_path",
            "circuit",
            "connected",
            "d_weighted_spanning_tree",
            "dag",
            "dconnected",
            "dpath",
            "dreachable",
            "dsteiner",
            "dtree",
            "network_flow",
            "network_flow_cost",
            "path",
            "reachable",
            "steiner",
            "subcircuit",
            "subgraph",
            "tree",
            "weighted_spanning_tree",
        },
        "4.2.2.12 Extensional constraints": {
            "cost_mdd",
            "cost_regular",
            "mdd",
            "mdd_nondet",
            "regular",
            "regular_nfa",
            "table",
        },
        "4.2.2.13 Machine learning constraints": {
            "neural_net",
        },
        "4.2.2.14 Deprecated constraints": {
            "at_least",
            "at_most",
            "exactly",
            "global_cardinality_low_up",
            "global_cardinality_low_up_closed",
        },
        "4.2.3.1 Integer FlatZinc builtins": {
            "array_int_element",
            "array_var_int_element",
            "int_abs",
            "int_div",
            "int_eq",
            "int_eq_reif",
            "int_le",
            "int_le_reif",
            "int_lin_eq",
            "int_lin_eq_reif",
            "int_lin_le",
            "int_lin_le_reif",
            "int_lin_ne",
            "int_lin_ne_reif",
            "int_lt",
            "int_lt_reif",
            "int_max",
            "int_min",
            "int_mod",
            "int_ne",
            "int_ne_reif",
            "int_plus",
            "int_pow",
            "int_times",
            "set_in",
        },
        "4.2.3.2 Bool FlatZinc builtins": {
            "array_bool_and",
            "array_bool_element",
            "array_bool_xor",
            "array_var_bool_element",
            "bool2int",
            "bool_and",
            "bool_clause",
            "bool_eq",
            "bool_eq_reif",
            "bool_le",
            "bool_le_reif",
            "bool_lin_eq",
            "bool_lin_le",
            "bool_lt",
            "bool_lt_reif",
            "bool_not",
            "bool_or",
            "bool_xor",
        },
        "4.2.3.3 Set FlatZinc builtins": {
            "array_set_element",
            "array_var_set_element",
            "set_card",
            "set_diff",
            "set_eq",
            "set_eq_reif",
            "set_in",
            "set_in_reif",
            "set_intersect",
            "set_le",
            "set_le_reif",
            "set_lt",
            "set_lt_reif",
            "set_ne",
            "set_ne_reif",
            "set_subset",
            "set_subset_reif",
            "set_superset",
            "set_superset_reif",
            "set_symdiff",
            "set_union",
        },
        "4.2.3.4 Float FlatZinc builtins": {
            "array_float_element",
            "array_var_float_element",
            "float_abs",
            "float_acos",
            "float_acosh",
            "float_asin",
            "float_asinh",
            "float_atan",
            "float_atanh",
            "float_cos",
            "float_cosh",
            "float_div",
            "float_eq",
            "float_eq_reif",
            "float_exp",
            "float_le",
            "float_le_reif",
            "float_lin_eq",
            "float_lin_eq_reif",
            "float_lin_le",
            "float_lin_le_reif",
            "float_lin_lt",
            "float_lin_lt_reif",
            "float_lin_ne",
            "float_lin_ne_reif",
            "float_ln",
            "float_log10",
            "float_log2",
            "float_lt",
            "float_lt_reif",
            "float_max",
            "float_min",
            "float_ne",
            "float_ne_reif",
            "float_plus",
            "float_pow",
            "float_sin",
            "float_sinh",
            "float_sqrt",
            "float_tan",
            "float_tanh",
            "float_times",
            "int2float",
        },
        "4.2.3.5 FlatZinc builtins added in MiniZinc 2.0.0": {
            "array_float_maximum",
            "array_float_minimum",
            "array_int_maximum",
            "array_int_minimum",
            "bool_clause_reif",
        },
        "4.2.3.6 FlatZinc builtins added in MiniZinc 2.0.2": {
            "array_var_bool_element_nonshifted",
            "array_var_float_element_nonshifted",
            "array_var_int_element_nonshifted",
            "array_var_set_element_nonshifted",
        },
        "4.2.3.7 FlatZinc builtins added in MiniZinc 2.1.0": {
            "float_dom",
            "float_in",
        },
        "4.2.3.8 FlatZinc builtins added in MiniZinc 2.1.1": {
            "max",
            "min",
        },
        "4.2.3.9 FlatZinc builtins added in MiniZinc 2.2.1": {
            "int_pow_fixed",
        },
        "4.2.3.10 FlatZinc builtins added in MiniZinc 2.3.3": {
            "float_set_in",
        },
        "4.2.3.11 FlatZinc builtins added in MiniZinc 2.5.2": {
            "array_var_bool_element2d_nonshifted",
            "array_var_float_element2d_nonshifted",
            "array_var_int_element2d_nonshifted",
            "array_var_set_element2d_nonshifted",
        },
        "4.2.3.12 FlatZinc builtins added in MiniZinc 2.7.1": {
            "float_ceil",
            "float_floor",
            "float_round",
        },
        "4.2.3.13 Deprecated FlatZinc builtins": {
            "array_bool_or",
        },
    }


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return obj


def categorize(
    unified: Dict[str, str],
    groups: Dict[str, List[str]],
    categories: Dict[str, Set[str]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, List[str]]]:
    categorized: Dict[str, Dict[str, str]] = {cat: {} for cat in categories}
    multi: Dict[str, List[str]] = {}

    for canonical, definition in unified.items():
        originals = set(groups.get(canonical, [canonical]))
        matched = [cat for cat, names in categories.items() if originals & names]

        if matched:
            for cat in matched:
                categorized[cat][canonical] = definition
            if len(matched) > 1:
                multi[canonical] = sorted(matched)

    uncategorized = {
        k: v
        for k, v in unified.items()
        if not any(k in categorized[cat] for cat in categories)
    }

    return categorized, uncategorized, multi


def main() -> int:
    ap = argparse.ArgumentParser(description="Categorize unified constraint descriptions into spec categories.")
    ap.add_argument(
        "--unified",
        type=Path,
        default=Path(__file__).with_name("fzn_descriptions_unified.json"),
        help="Unified descriptions JSON (default: tools/fzn_descriptions_unified.json)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).with_name("fzn_descriptions_unified_report.json"),
        help="Unification report JSON (default: tools/fzn_descriptions_unified_report.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("fzn_descriptions_categorized.json"),
        help="Output categorized JSON (default: tools/fzn_descriptions_categorized.json)",
    )

    args = ap.parse_args()

    unified_obj = load_json(args.unified)
    unified: Dict[str, str] = {str(k): str(v) for k, v in unified_obj.items()}

    report_obj = load_json(args.report)
    report_groups = report_obj.get("groups")
    if not isinstance(report_groups, dict):
        raise TypeError("Report JSON must contain a 'groups' object")
    groups: Dict[str, List[str]] = {}
    for canon, originals in report_groups.items():
        if isinstance(originals, list):
            groups[str(canon)] = [str(x) for x in originals]
        else:
            groups[str(canon)] = [str(originals)]

    categories = build_categories()
    categorized, uncategorized, multi = categorize(unified, groups, categories)

    out = {
        "_meta": {
            "unified_entries": len(unified),
            "categorized_entries": sum(len(v) for v in categorized.values()),
            "uncategorized_entries": len(uncategorized),
            "multi_category_entries": len(multi),
            "category_counts": {k: len(v) for k, v in categorized.items()},
        },
        "categories": {
            k: dict(sorted(v.items())) for k, v in categorized.items() if v
        },
        "uncategorized": dict(sorted(uncategorized.items())),
        "multi_category": dict(sorted(multi.items())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Unified entries: {len(unified)}")
    print(f"Uncategorized:  {len(uncategorized)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
