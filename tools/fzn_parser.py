#!/usr/bin/env python3
import sys
import re
from collections import defaultdict
from statistics import mean
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# ============================================================
# Natural language mappings
# ============================================================

SEARCH_VAR_STRATEGY = {
    "input_order": "the input order",
    "first_fail": "a first-fail strategy",
    "anti_first_fail": "an anti first-fail strategy",
    "smallest": "a smallest-domain strategy",
    "largest": "a largest-domain strategy",
}

SEARCH_VALUE_STRATEGY = {
    "indomain_min": "assigning the minimum value",
    "indomain_max": "assigning the maximum value",
    "indomain_split": "splitting the domain",
    "indomain_split_random": "splitting the domain randomly",
}

SEARCH_COMPLETENESS = {
    "complete": "exploring the entire search space",
    "incomplete": "using an incomplete exploration strategy",
}

CONSTRAINT_TEXT = {
    "int_lin_eq":
        "Linear equality constraints enforce that weighted sums of integer variables equal a constant",
    "int_lin_le":
        "Linear inequality constraints restrict weighted sums of integer variables to be less than or equal to a constant",
    "int_lin_ge":
        "Linear inequality constraints restrict weighted sums of integer variables to be greater than or equal to a constant",
    "int_eq":
        "Equality constraints enforce that pairs of integer variables take the same value",
    "int_ne":
        "Disequality constraints enforce that pairs of integer variables take different values",
    "int_le":
        "Ordering constraints enforce that one integer variable is less than or equal to another",
    "bool_clause":
        "Boolean clause constraints represent disjunctions over Boolean literals",
    "bool_clause_reif":
        "Reified Boolean clauses link the satisfaction of a clause to a Boolean variable",
    "bool2int":
        "Boolean-to-integer channeling constraints map Boolean values to integer variables",
    "all_different":
        "All-different constraints enforce that all involved variables take pairwise distinct values",
    "element":
        "Element constraints link a variable to a value selected from an array using an index",
    "int_times":
        "Multiplicative constraints enforce that one integer variable equals the product of two others",
    "int_max":
        "Maximum constraints bind a variable to the maximum value among a set of variables",
}

# ============================================================
# Model container
# ============================================================

class FlatZincModel:
    def __init__(self):
        self.variables = {}
        self.arrays = {}
        self.constraints = []
        self.problem_type = None
        self.objective = None
        self.search = None


@dataclass(frozen=True)
class Domain:
    min_value: int
    max_value: int

    @property
    def mean_value(self) -> float:
        return (self.min_value + self.max_value) / 2


def _parse_domain_spec(domain_spec: str) -> Optional[Domain]:
    domain_spec = domain_spec.strip()

    m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", domain_spec)
    if m:
        lo, hi = map(int, m.groups())
        return Domain(min(lo, hi), max(lo, hi))

    # Set domain: {1,2,3}
    m = re.fullmatch(r"\{\s*(.*?)\s*\}", domain_spec)
    if m:
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if not items:
            return None
        try:
            values = [int(x) for x in items]
        except ValueError:
            return None
        return Domain(min(values), max(values))

    return None


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0

    for ch in s:
        if ch == "(" and depth_brace == 0:
            depth_paren += 1
        elif ch == ")" and depth_brace == 0:
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[" and depth_brace == 0:
            depth_brack += 1
        elif ch == "]" and depth_brace == 0:
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)

        if ch == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_balanced_call(text: str, start_idx: int) -> Optional[str]:
    """Extracts `name(...balanced...)` starting at start_idx. Returns the substring."""
    if start_idx < 0 or start_idx >= len(text):
        return None
    open_idx = text.find("(", start_idx)
    if open_idx == -1:
        return None

    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _parse_vars_expr(expr: str) -> List[str]:
    expr = expr.strip()
    if not expr:
        return []
    if expr.startswith("[") and expr.endswith("]"):
        inner = expr[1:-1].strip()
        if not inner:
            return []
        return [v.strip() for v in _split_top_level_commas(inner) if v.strip()]
    # Single variable/identifier
    return [expr]


def _parse_search_annotation(ann: str):
    """Parse common FlatZinc search annotations.

    Returns either:
      - {kind: 'int_search', vars: [...], var_strategy: str, val_strategy: str, completeness: str}
      - {kind: 'seq_search', phases: [<search dicts>]}
      - None if no recognizable search annotation
    """
    if not ann:
        return None

    # Find an outermost search call we understand.
    idx = ann.find("seq_search(")
    kind = "seq_search"
    if idx == -1:
        idx = ann.find("int_search(")
        kind = "int_search"
    if idx == -1:
        return None

    call = _extract_balanced_call(ann, idx)
    if not call:
        return None

    def _parse_expr(expr: str):
        expr = expr.strip()
        if expr.startswith("int_search("):
            inner = expr[len("int_search("):-1]
            args = _split_top_level_commas(inner)
            if len(args) != 4:
                return None
            vars_expr, var_sel, val_sel, comp = [a.strip() for a in args]
            return {
                "kind": "int_search",
                "vars": _parse_vars_expr(vars_expr),
                "var_strategy": var_sel,
                "val_strategy": val_sel,
                "completeness": comp,
            }
        if expr.startswith("seq_search("):
            inner = expr[len("seq_search("):-1].strip()
            args = _split_top_level_commas(inner)
            if not args:
                return None
            # Usually seq_search([search1, search2, ...])
            first = args[0].strip()
            phases: List[dict] = []
            if first.startswith("[") and first.endswith("]"):
                inner_list = first[1:-1].strip()
                elems = _split_top_level_commas(inner_list) if inner_list else []
                for e in elems:
                    parsed = _parse_expr(e)
                    if parsed:
                        phases.append(parsed)
            else:
                parsed = _parse_expr(first)
                if parsed:
                    phases.append(parsed)
            if not phases:
                return None
            return {"kind": "seq_search", "phases": phases}
        return None

    if kind == "seq_search":
        return _parse_expr(call)
    return _parse_expr(call)

# ============================================================
# Parsing
# ============================================================

def parse_fzn(path):
    model = FlatZincModel()

    with open(path) as f:
        text = f.read()

    # Variables (scalar)
    # FlatZinc uses forms like:
    #   var int: x;
    #   var bool: b;
    #   var 1..52: X_INTRODUCED_1_;
    #   var {1,3,5}: v;
    for m in re.finditer(r"\bvar\s+(?P<spec>int|bool|-?\d+\.\.-?\d+|\{[^}]*\})\s*:\s*(?P<name>\w+)\s*;", text):
        spec = m.group("spec")
        name = m.group("name")

        if spec == "bool":
            model.variables[name] = {"type": "bool", "domain": Domain(0, 1)}
        elif spec == "int":
            model.variables[name] = {"type": "int", "domain": None}
        else:
            model.variables[name] = {"type": "int", "domain": _parse_domain_spec(spec)}

    # Arrays (optional, for reporting convenience)
    # Example:
    #   array [1..52] of var int: y = [...];
    #   array [1..52] of var int: x:: output_array([1..52]) = [...];
    for m in re.finditer(
        r"\barray\s*\[(?P<index>[^\]]+)\]\s*of\s*var\s+(?P<elem>int|bool)\s*:\s*(?P<name>\w+)(?:\s*::[^=;]+)?\s*=\s*\[(?P<body>.*?)\]\s*;",
        text,
        re.DOTALL,
    ):
        index_spec = m.group("index").strip()
        elem_type = m.group("elem")
        name = m.group("name")
        body = m.group("body")

        # Best-effort parse of elements: variables or integer literals.
        raw_items = [x.strip() for x in body.replace("\n", " ").split(",")]
        items = [x for x in raw_items if x]

        length = None
        idx_m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", index_spec)
        if idx_m:
            lo, hi = map(int, idx_m.groups())
            length = abs(hi - lo) + 1

        model.arrays[name] = {
            "type": f"{elem_type}[]",
            "length": length,
            "items": items,
        }

    # Constraints
    for m in re.finditer(r"constraint\s+(\w+)\((.*?)\);", text, re.DOTALL):
        ctype, args = m.groups()
        model.constraints.append(ctype)

    # Solve + search
    solve_match = re.search(
        r"solve\s*(::\s*(.*?))?\s*(satisfy|maximize|minimize)\s*(\w+)?\s*;",
        text,
        re.DOTALL
    )

    if solve_match:
        _, ann, stype, obj = solve_match.groups()
        model.problem_type = stype
        model.objective = obj

        if ann:
            model.search = _parse_search_annotation(ann)

    return model

# ============================================================
# Descriptions
# ============================================================

def describe_problem(model):
    if model.problem_type == "satisfy":
        return "Find any solution satisfying all constraints."
    if model.problem_type == "minimize":
        return f"Find a solution minimizing {model.objective}."
    if model.problem_type == "maximize":
        return f"Find a solution maximizing {model.objective}."
    return "Problem type could not be determined."

def describe_search(search):
    if not search:
        return "No explicit search strategy is specified."

    def _names_preview(names: List[str], limit: int = 6) -> str:
        names = [n for n in names if n]
        if not names:
            return "(none)"
        names_sorted = sorted(names)
        if len(names_sorted) <= limit:
            return ", ".join(names_sorted)
        head = ", ".join(names_sorted[:limit])
        return f"{head}, … (+{len(names_sorted) - limit})"

    def _describe_int_search(s: dict) -> str:
        vars_ = _names_preview(s.get("vars", []))
        var_sel = SEARCH_VAR_STRATEGY.get(s.get("var_strategy"), s.get("var_strategy"))
        val_sel = SEARCH_VALUE_STRATEGY.get(s.get("val_strategy"), s.get("val_strategy"))
        comp = SEARCH_COMPLETENESS.get(s.get("completeness"), s.get("completeness"))
        return f"integer search on variables {vars_}, using {var_sel}, {val_sel}, and {comp}";

    if isinstance(search, dict) and search.get("kind") == "seq_search":
        phases = search.get("phases", [])
        phase_desc = []
        for p in phases:
            if isinstance(p, dict) and p.get("kind") == "int_search":
                phase_desc.append(_describe_int_search(p))
            elif isinstance(p, dict) and p.get("kind") == "seq_search":
                # Nested seq_search: describe recursively in-line.
                phase_desc.append(describe_search(p).rstrip("."))
        if not phase_desc:
            return "No explicit search strategy is specified."
        joined = "; ".join(f"({i+1}) {d}" for i, d in enumerate(phase_desc))
        return f"The solver applies a sequential search strategy with {len(phase_desc)} phases: {joined}."

    if isinstance(search, dict) and search.get("kind") == "int_search":
        return f"The solver applies an {_describe_int_search(search)}."

    # Backwards compatibility if something else assigned a plain dict.
    if isinstance(search, dict) and {"vars", "var_strategy", "val_strategy", "completeness"}.issubset(search.keys()):
        compat = {"kind": "int_search", **search}
        return f"The solver applies an {_describe_int_search(compat)}."

    return "No explicit search strategy is specified."

def describe_variables_detailed(model):
    lines = []
    scalar_domain_sizes = []

    def _fmt_domain(d: Optional[Domain]) -> str:
        if d is None:
            return "domain unknown"
        return f"domain [{d.min_value}, {d.max_value}], mean {d.mean_value:.2f}"

    def _domain_size(d: Optional[Domain]) -> Optional[int]:
        if d is None:
            return None
        return d.max_value - d.min_value + 1

    objective_name = model.objective if isinstance(model.objective, str) else None

    # Keep the objective variable described (not grouped), but do not print its name.
    if objective_name and objective_name in model.variables:
        v = model.variables[objective_name]
        d = v.get("domain")
        size = _domain_size(d)
        if size is not None:
            scalar_domain_sizes.append(size)
        lines.append(f"  Objective variable ({v['type']}): {_fmt_domain(d)}")

    # Group remaining scalar variables by (type, domain).
    groups: Dict[Tuple[str, Optional[Domain]], List[str]] = defaultdict(list)
    for name, v in model.variables.items():
        if objective_name and name == objective_name:
            continue
        d = v.get("domain")
        size = _domain_size(d)
        if size is not None:
            scalar_domain_sizes.append(size)
        groups[(v["type"], d if isinstance(d, Domain) else None)].append(name)

    for (vtype, d), names in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), len(kv[1]))):
        lines.append(f"  {len(names)}× variables ({vtype}): {_fmt_domain(d)}")

    header = [
        f"The model contains "
        f"{sum(1 for v in model.variables.values() if v['type']=='int')} integer variables and "
        f"{sum(1 for v in model.variables.values() if v['type']=='bool')} Boolean variables."
    ]

    if scalar_domain_sizes:
        header.append(
            f"Scalar variable domain sizes range from {min(scalar_domain_sizes)} to {max(scalar_domain_sizes)}, "
            f"with an average size of {mean(scalar_domain_sizes):.2f}."
        )

    return "\n".join(header + [""] + lines)

def describe_constraints(model):
    counts = defaultdict(int)
    for c in model.constraints:
        counts[c] += 1

    paragraphs = []
    total = sum(counts.values())

    for ctype, count in sorted(counts.items()):
        text = CONSTRAINT_TEXT.get(
            ctype,
            f"Constraints of type {ctype} restrict relationships between variables"
        )
        paragraphs.append(f"  {ctype}: {count}")

    organic = []
    for ctype, count in sorted(counts.items()):
        base = CONSTRAINT_TEXT.get(
            ctype,
            f"Constraints of type {ctype} restrict relationships between variables"
        )
        organic.append(f"{count} instances of {base.lower()}")

    return (
        "\n".join(paragraphs)
        + f"\n\nTotal constraints: {total}\n\n"
        + "Qualitatively, the model is composed of "
        + ", ".join(organic)
        + "."
    )

# ============================================================
# Output
# ============================================================

def summarize(model):
    print("=" * 60)
    print("PROBLEM SUMMARY")
    print("=" * 60)

    print("\nProblem:")
    print(" ", describe_problem(model))

    print("\nSearch strategy:")
    print(" ", describe_search(model.search))

    print("\nVariables:")
    print(describe_variables_detailed(model))

    print("\nConstraints:")
    print(describe_constraints(model))

    print("\nDone.")
    print("=" * 60)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fzn_parser.py model.fzn")
        sys.exit(1)

    model = parse_fzn(sys.argv[1])
    summarize(model)
