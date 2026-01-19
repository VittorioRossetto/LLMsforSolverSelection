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
        # Best-effort: map a defined variable name -> the constraint that defines it
        # (based on FlatZinc annotations like :: defines_var(x)).
        self.definitions = {}
        self.problem_type = None
        self.objective = None
        self.search = None


def is_compiler_introduced_var(name: str, ann: Optional[str]) -> bool:
    """Heuristic: classify variables introduced by the FlatZinc compiler.

    FlatZinc typically marks compiler-introduced (defined) variables with
    annotations like `is_defined_var`. Many toolchains also use naming patterns
    such as `X_INTRODUCED_...`.
    """
    if name and re.fullmatch(r"X_INTRODUCED_\d+_", name):
        return True
    if name and "INTRODUCED" in name:
        return True

    ann_text = (ann or "").lower()
    if not ann_text:
        return False

    # Common FlatZinc annotations for compiler-defined variables.
    # Keep this permissive: different backends use slightly different tokens.
    if "is_defined_var" in ann_text:
        return True
    if "var_is_introduced" in ann_text:
        return True
    if "is_introduced" in ann_text:
        return True
    return False


def compute_variable_degrees(model: "FlatZincModel") -> Dict[str, int]:
    """Return a best-effort mapping var_name -> number of constraints mentioning it."""
    degrees: Dict[str, int] = {name: 0 for name in model.variables.keys()}

    # Tokenize arguments and count mentions of known scalar variables.
    token_re = re.compile(r"\b[A-Za-z]\w*\b")
    for c in model.constraints:
        if isinstance(c, dict):
            args = c.get("args", "")
        else:
            # Backwards compatibility (old format stored only the type string).
            args = ""
        if not args:
            continue
        tokens = token_re.findall(args)
        for name in tokens:
            if name in degrees:
                degrees[name] += 1
    return degrees


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
    # NOTE: Anchor scalar variables at start-of-line to avoid accidentally matching
    # the `of var int:` fragment inside array declarations.
    for m in re.finditer(
        r"^\s*var\s+(?P<spec>int|bool|-?\d+\.\.-?\d+|\{[^}]*\})\s*:\s*(?P<name>\w+)(?:\s*::\s*(?P<ann>[^;]*))?\s*;",
        text,
        re.MULTILINE,
    ):
        spec = m.group("spec")
        name = m.group("name")
        ann = m.group("ann")

        origin = "introduced" if is_compiler_introduced_var(name, ann) else "user"

        if spec == "bool":
            model.variables[name] = {
                "type": "bool",
                "domain": Domain(0, 1),
                "origin": origin,
            }
        elif spec == "int":
            model.variables[name] = {"type": "int", "domain": None, "origin": origin}
        else:
            model.variables[name] = {
                "type": "int",
                "domain": _parse_domain_spec(spec),
                "origin": origin,
            }

    # Arrays (optional, for reporting convenience)
    # Examples:
    #   array [1..52] of var int: y = [...];
    #   array [1..52] of var int: x:: output_array([1..52]) = [...];
    #   array [1..4]  of int: X_INTRODUCED_334_ = [1,1,1,-1];
    for m in re.finditer(
        r"\barray\s*\[(?P<index>[^\]]+)\]\s*of\s*(?P<var>var\s+)?(?P<elem>int|bool)\s*:\s*(?P<name>\w+)(?:\s*::\s*(?P<ann>[^=;]+))?(?:\s*=\s*\[(?P<body>.*?)\])?\s*;",
        text,
        re.DOTALL,
    ):
        index_spec = m.group("index").strip()
        elem_type = m.group("elem")
        is_var = bool(m.group("var"))
        name = m.group("name")
        ann = m.group("ann")
        body = m.group("body")

        # Best-effort parse of elements: variables or integer literals.
        items: List[str] = []
        if body is not None:
            raw_items = [x.strip() for x in body.replace("\n", " ").split(",")]
            items = [x for x in raw_items if x]

        length = None
        idx_m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", index_spec)
        if idx_m:
            lo, hi = map(int, idx_m.groups())
            length = abs(hi - lo) + 1

        model.arrays[name] = {
            "type": f"{elem_type}[]",
            "elem_type": elem_type,
            "is_var": is_var,
            "length": length,
            "items": items,
            "origin": "introduced" if is_compiler_introduced_var(name, ann) else "user",
        }

    # Constraints
    # FlatZinc constraints may include trailing annotations like `:: domain` after the closing ')'.
    # Regex-only parsing is brittle (balanced parentheses), so do a small scan using
    # the existing balanced-call extractor.
    constraint_start_re = re.compile(r"\bconstraint\s+(?P<type>\w+)\s*\(")
    defines_var_re = re.compile(r"\bdefines_var\s*\(\s*(?P<name>\w+)\s*\)")
    pos = 0
    while True:
        m = constraint_start_re.search(text, pos)
        if not m:
            break
        ctype = m.group("type")

        call = _extract_balanced_call(text, m.start("type"))
        if not call:
            pos = m.end()
            continue

        call_end = m.start("type") + len(call)
        semi = text.find(";", call_end)
        if semi == -1:
            break

        args = call[call.find("(") + 1 : -1]
        ann = text[call_end:semi].strip()
        defines = [mm.group("name") for mm in defines_var_re.finditer(ann or "")]
        rendered = f"{ctype}({args})" + (f" {ann}" if ann else "")

        model.constraints.append(
            {"type": ctype, "args": args, "ann": ann, "text": rendered, "defines": defines}
        )
        pos = semi + 1

    # Build a best-effort definitions map from :: defines_var(...) annotations.
    for c in model.constraints:
        if not isinstance(c, dict):
            continue
        for vname in c.get("defines", []) or []:
            if vname and vname not in model.definitions:
                model.definitions[vname] = c

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

def describe_objective_function(model: "FlatZincModel", max_depth: int = 3, max_len: int = 800) -> Optional[str]:
    """Best-effort symbolic objective formulation.

    FlatZinc only gives an objective variable plus constraints; there is no
    guaranteed high-level objective expression. Many backends annotate derived
    variables with `:: defines_var(x)`, which lets us reconstruct a formulation
    for the objective variable as an expression tree.
    """
    if model.problem_type not in {"minimize", "maximize"}:
        return None
    objective_name = model.objective if isinstance(model.objective, str) else None
    if not objective_name:
        return None

    expr = _expr_for_name(model, objective_name, depth=max_depth, visited=set())
    expr = re.sub(r"\s+", " ", (expr or "").strip())
    if not expr:
        return None

    abstract_expr, abstract_obj = _abstract_objective_expression(model, objective_name, expr)
    abstract_expr = re.sub(r"\s+", " ", (abstract_expr or "").strip())

    if len(abstract_expr) > max_len:
        abstract_expr = abstract_expr[: max_len - 1] + "…"

    if expr != objective_name and abstract_expr:
        return (
            f"The objective function is in the form: {model.problem_type} {abstract_obj} "
            f"where {abstract_obj} = {abstract_expr}"
        )
    return f"The objective function is in the form: {model.problem_type} {abstract_obj}"


def _abstract_objective_expression(
    model: "FlatZincModel", objective_name: str, expr: str
) -> Tuple[str, str]:
    """Rewrite variable identifiers in an expression into placeholders a,b,c,...

    This is presentation-only: it keeps the objective structure (e.g., max/min/+/etc.)
    but hides FlatZinc-specific variable names like X_INTRODUCED_123_.
    """

    def _placeholder_for_index(i: int) -> str:
        # 0 -> a, 1 -> b, ..., 25 -> z, 26 -> aa, ...
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        out = ""
        n = i
        while True:
            out = alphabet[n % 26] + out
            n = n // 26 - 1
            if n < 0:
                break
        return out

    reserved = {
        # Expression functions we produce
        "max",
        "min",
        "bool2int",
        # If-then-else keywords (we output them in the reconstructed string)
        "if",
        "then",
        "else",
        # Boolean literals sometimes appear
        "true",
        "false",
    }

    objective_name = (objective_name or "").strip()
    expr = (expr or "").strip()
    if not expr:
        return "", "a"

    # Decide what counts as a "replaceable" identifier.
    replaceable: set[str] = set(model.variables.keys()) | set(model.arrays.keys())
    if objective_name:
        replaceable.add(objective_name)

    mapping: Dict[str, str] = {}
    next_idx = 0

    def _ensure(name: str) -> str:
        nonlocal next_idx
        if name in mapping:
            return mapping[name]
        if name == objective_name:
            mapping[name] = "a"
            return "a"
        # First non-objective placeholder should be b.
        if next_idx == 0 and "a" not in mapping.values():
            # Not expected (objective maps to a), but keep consistent.
            next_idx = 1
        if next_idx == 0:
            next_idx = 1
        mapping[name] = _placeholder_for_index(next_idx)
        next_idx += 1
        return mapping[name]

    token_re = re.compile(r"\b[A-Za-z]\w*\b")

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        if tok.lower() in reserved:
            return tok
        if tok in replaceable:
            return _ensure(tok)
        return tok

    abstract_expr = token_re.sub(_sub, expr)
    abstract_obj = "a"
    return abstract_expr, abstract_obj


def _expr_for_name(model: "FlatZincModel", name: str, depth: int, visited: set[str]) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    if re.fullmatch(r"-?\d+", name):
        return name

    v = model.variables.get(name)
    if v:
        d = v.get("domain")
        if isinstance(d, Domain) and d.min_value == d.max_value:
            return str(d.min_value)

    if depth <= 0 or name in visited:
        return name

    defining = model.definitions.get(name)
    if not isinstance(defining, dict):
        return name

    visited.add(name)
    try:
        expr = _expr_from_defining_constraint(model, name, defining, depth=depth, visited=visited)
        return expr or name
    finally:
        visited.remove(name)


def _expr_from_defining_constraint(
    model: "FlatZincModel",
    defined_name: str,
    constraint: dict,
    depth: int,
    visited: set[str],
) -> Optional[str]:
    ctype = (constraint.get("type") or "").strip()
    args = (constraint.get("args") or "").strip()
    parts = _split_top_level_commas(args) if args else []

    def _e(x: str) -> str:
        return _expr_for_name(model, x, depth=depth - 1, visited=visited)

    if ctype == "int_max" and len(parts) == 3:
        a, b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"max({_e(a)}, {_e(b)})"

    if ctype == "int_min" and len(parts) == 3:
        a, b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"min({_e(a)}, {_e(b)})"

    if ctype == "int_plus" and len(parts) == 3:
        a, b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"({_e(a)} + {_e(b)})"

    if ctype == "int_minus" and len(parts) == 3:
        a, b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"({_e(a)} - {_e(b)})"

    if ctype == "int_times" and len(parts) == 3:
        a, b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"({_e(a)} * {_e(b)})"

    if ctype in {"array_int_element", "array_var_int_element"} and len(parts) == 3:
        idx, arr, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"{_e(arr)}[{_e(idx)}]"

    if ctype in {"fzn_if_then_else_var_int", "fzn_if_then_else_var_bool"}:
        if len(parts) == 4:
            cond, then_val, else_val, out = [p.strip() for p in parts]
            if out == defined_name:
                return f"(if {_e(cond)} then {_e(then_val)} else {_e(else_val)})"
        if len(parts) == 3:
            cond, then_val, out = [p.strip() for p in parts]
            if out == defined_name:
                return f"(if {_e(cond)} then {_e(then_val)} else 0)"

    if ctype == "bool2int" and len(parts) == 2:
        b, out = [p.strip() for p in parts]
        if out == defined_name:
            return f"bool2int({_e(b)})"

    # Linear equality can often define a single variable.
    if ctype == "int_lin_eq" and len(parts) == 3:
        coeffs = _resolve_int_array(model, parts[0])
        vars_ = _resolve_id_array(model, parts[1])
        const = _parse_int_literal(parts[2])
        if coeffs is not None and vars_ is not None and const is not None:
            if len(coeffs) == len(vars_) and defined_name in vars_:
                idx = vars_.index(defined_name)
                a_t = coeffs[idx]
                rest_terms = [(a, v) for i, (a, v) in enumerate(zip(coeffs, vars_)) if i != idx]
                rest_expr = _format_linear_sum(model, rest_terms, depth=depth, visited=visited)

                if a_t == -1:
                    if const == 0:
                        return rest_expr
                    return f"({rest_expr} - {const})"
                if a_t == 1:
                    if const == 0:
                        return f"(-({rest_expr}))" if rest_expr != "0" else "0"
                    if rest_expr == "0":
                        return str(const)
                    return f"({const} - ({rest_expr}))"
                # Generic rearrangement: x = (c - rest) / a
                if rest_expr == "0":
                    return f"({const} / {a_t})"
                return f"(({const} - ({rest_expr})) / {a_t})"

    # Fallback: show the defining constraint call.
    return f"{ctype}({args})" if ctype else None


def _parse_int_literal(s: str) -> Optional[int]:
    s = (s or "").strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None


def _parse_int_array(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts = _split_top_level_commas(inner)
    out: List[int] = []
    for p in parts:
        lit = _parse_int_literal(p)
        if lit is None:
            return None
        out.append(lit)
    return out


def _parse_id_array(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [p.strip() for p in _split_top_level_commas(inner)]


def _resolve_int_array(model: "FlatZincModel", s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    # Literal list
    parsed = _parse_int_array(s)
    if parsed is not None:
        return parsed
    # Named constant array
    if s in model.arrays and not model.arrays[s].get("is_var", False):
        items = model.arrays[s].get("items", [])
        out: List[int] = []
        for it in items:
            lit = _parse_int_literal(it)
            if lit is None:
                return None
            out.append(lit)
        return out
    return None


def _resolve_id_array(model: "FlatZincModel", s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    # Literal list
    parsed = _parse_id_array(s)
    if parsed is not None:
        return parsed
    # Named var array
    if s in model.arrays and model.arrays[s].get("is_var", False):
        items = model.arrays[s].get("items", [])
        return [str(it).strip() for it in items if str(it).strip()]
    return None


def _format_linear_sum(
    model: "FlatZincModel", terms: List[Tuple[int, str]], depth: int, visited: set[str]
) -> str:
    pieces: List[str] = []
    for a, v in terms:
        if a == 0:
            continue
        ve = _expr_for_name(model, v, depth=depth - 1, visited=visited)
        if a == 1:
            pieces.append(f"{ve}")
        elif a == -1:
            pieces.append(f"-({ve})")
        else:
            pieces.append(f"{a}*({ve})")

    if not pieces:
        return "0"
    # Join with + and normalize '+ -' sequences a bit.
    expr = " + ".join(pieces)
    expr = expr.replace("+ -(", "- (")
    return expr

def describe_problem(model):
    if model.problem_type == "satisfy":
        return "This is a satisfaction problem."

    if model.problem_type in {"minimize", "maximize"}:
        direction = "minimization" if model.problem_type == "minimize" else "maximization"
        base = f"This is a {direction} problem."

        objective_name = model.objective if isinstance(model.objective, str) else None
        if not objective_name or objective_name not in model.variables:
            return base + " Objective variable could not be determined."

        v = model.variables[objective_name]
        d = v.get("domain")
        deg = compute_variable_degrees(model).get(objective_name, 0)

        obj_expr = describe_objective_function(model)
        if d is None:
            suffix = (
                f" Objective: {model.problem_type} an objective variable with unknown domain and degree {deg}."
            )
            if obj_expr:
                suffix += f" {obj_expr}."
            return base + suffix

        size = d.max_value - d.min_value + 1
        mean_value = (d.min_value + d.max_value) / 2
        suffix = (
            f" Objective: {model.problem_type} an objective variable with domain [{d.min_value}, {d.max_value}]"
            f" (size {size}, mean {mean_value:.2f}) and degree {deg}."
        )
        if obj_expr:
            suffix += f" {obj_expr}."
        return base + suffix

    return "Problem type could not be determined."

def _constraint_stats_for_name(model: "FlatZincModel", name: str) -> Tuple[int, List[str]]:
    """Return (count, sorted_unique_types) of constraints that mention `name`."""
    if not name:
        return 0, []
    token_re = re.compile(r"\b[A-Za-z]\w*\b")
    types: set[str] = set()
    count = 0
    for c in model.constraints:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        args = c.get("args", "")
        if not ctype or not args:
            continue
        tokens = token_re.findall(args)
        if name in tokens:
            count += 1
            types.add(ctype)
    return count, sorted(types)


def _constraint_texts_for_name(model: "FlatZincModel", name: str, limit: int = 4) -> List[str]:
    """Return up to `limit` constraint call strings (e.g. `fzn_inverse(x,y)`) that mention `name`."""
    if not name or limit <= 0:
        return []
    token_re = re.compile(r"\b[A-Za-z]\w*\b")
    out: List[str] = []
    seen: set[str] = set()

    def _shorten(s: str, max_len: int = 140) -> str:
        s = (s or "").strip().replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    for c in model.constraints:
        if not isinstance(c, dict):
            continue
        args = c.get("args", "")
        if not args:
            continue
        tokens = token_re.findall(args)
        if name not in tokens:
            continue

        # Prefer a clean call string without trailing annotations (e.g., omit `:: domain`).
        ctype = c.get("type") or "constraint"
        txt = f"{ctype}({args})"
        txt = _shorten(txt)
        if txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
        if len(out) >= limit:
            break
    return out


def _domain_text_for_scalar(model: "FlatZincModel", name: str) -> str:
    v = model.variables.get(name)
    if not v:
        return "domain unknown"
    d = v.get("domain")
    if d is None:
        return "domain unknown"
    return f"domain [{d.min_value}, {d.max_value}]"


def _domain_text_for_array(model: "FlatZincModel", name: str) -> str:
    a = model.arrays.get(name)
    if not a:
        return "domain unknown"
    items = a.get("items", [])
    doms: List[Domain] = []
    for it in items:
        v = model.variables.get(it)
        if not v:
            continue
        d = v.get("domain")
        if isinstance(d, Domain):
            doms.append(d)
    if not doms:
        return "domain unknown"
    lo = min(d.min_value for d in doms)
    hi = max(d.max_value for d in doms)
    return f"element domain [{lo}, {hi}]"


def describe_search(search, model: Optional["FlatZincModel"] = None):
    if not search:
        return "No explicit search strategy is specified."

    def _vars_count_text(names: List[str]) -> str:
        count = sum(1 for n in names if n)
        if count == 1:
            return "1 variable"
        return f"{count} variables"

    def _describe_int_search(s: dict) -> str:
        vars_ = _vars_count_text(s.get("vars", []))
        var_sel = SEARCH_VAR_STRATEGY.get(s.get("var_strategy"), s.get("var_strategy"))
        val_sel = SEARCH_VALUE_STRATEGY.get(s.get("val_strategy"), s.get("val_strategy"))
        comp = SEARCH_COMPLETENESS.get(s.get("completeness"), s.get("completeness"))
        strategy_txt = f"using {var_sel}, {val_sel}, and {comp}"

        # If this is a single variable/array name and we have the model, enrich the description.
        if model and s.get("vars") and len(s.get("vars")) == 1:
            name = (s.get("vars") or [""])[0]

            if name in model.variables:
                v = model.variables[name]
                vtype = v.get("type", "int")
                domain_txt = _domain_text_for_scalar(model, name)
                c_count, _c_types = _constraint_stats_for_name(model, name)
                examples = _constraint_texts_for_name(model, name, limit=3) if c_count > 0 else []
                constraints_txt = f"{c_count} constraints"
                if c_count > 0 and examples:
                    constraints_txt += f"( {', '.join(examples)} )"

                text = (
                    f"integer search on 1 {vtype} variable with {domain_txt}, "
                    f"involved in {constraints_txt}, {strategy_txt}"
                )
                return text

            if name in model.arrays:
                a = model.arrays[name]
                elem_type = a.get("elem_type", "int")
                length = a.get("length")
                length_txt = f"length {length}" if isinstance(length, int) else "unknown length"
                domain_txt = _domain_text_for_array(model, name)
                c_count, _c_types = _constraint_stats_for_name(model, name)
                examples = _constraint_texts_for_name(model, name, limit=3) if c_count > 0 else []
                constraints_txt = f"{c_count} constraints"
                if c_count > 0 and examples:
                    constraints_txt += f"( {', '.join(examples)} )"

                text = (
                    f"integer search on 1 {elem_type} array ({length_txt}) with {domain_txt}, "
                    f"involved in {constraints_txt}, {strategy_txt}"
                )
                return text

        return f"integer search on {vars_}, {strategy_txt}"

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
        return f"The model suggests a sequential search strategy with {len(phase_desc)} phases: {joined}."

    if isinstance(search, dict) and search.get("kind") == "int_search":
        return f"The model suggests an {_describe_int_search(search)}."

    # Backwards compatibility if something else assigned a plain dict.
    if isinstance(search, dict) and {"vars", "var_strategy", "val_strategy", "completeness"}.issubset(search.keys()):
        compat = {"kind": "int_search", **search}
        return f"The model suggests an {_describe_int_search(compat)}."

    return "No explicit search strategy is specified."

def describe_variables_detailed(model):
    lines = []
    scalar_domain_sizes_user: List[int] = []
    scalar_domain_sizes_introduced: List[int] = []
    scalar_domain_sizes_all: List[int] = []
    unknown_domain_count_user = 0
    unknown_domain_count_introduced = 0

    def _is_constant_scalar(v: dict) -> bool:
        d = v.get("domain")
        return isinstance(d, Domain) and d.min_value == d.max_value

    def _fmt_domain(d: Optional[Domain]) -> str:
        if d is None:
            return "domain unknown"
        return f"domain [{d.min_value}, {d.max_value}], mean {d.mean_value:.2f}"

    def _domain_size(d: Optional[Domain]) -> Optional[int]:
        if d is None:
            return None
        return d.max_value - d.min_value + 1

    # Gather scalar domain statistics for all variables.
    for v in model.variables.values():
        if _is_constant_scalar(v):
            continue
        size = _domain_size(v.get("domain"))
        origin = v.get("origin", "user")
        if size is None:
            if origin == "introduced":
                unknown_domain_count_introduced += 1
            else:
                unknown_domain_count_user += 1
        else:
            scalar_domain_sizes_all.append(size)
            if origin == "introduced":
                scalar_domain_sizes_introduced.append(size)
            else:
                scalar_domain_sizes_user.append(size)

    def _group_counts(origin: str) -> Tuple[int, int]:
        int_count = sum(
            1
            for v in model.variables.values()
            if v.get("origin", "user") == origin
            and v.get("type") == "int"
            and not _is_constant_scalar(v)
        )
        bool_count = sum(
            1
            for v in model.variables.values()
            if v.get("origin", "user") == origin
            and v.get("type") == "bool"
            and not _is_constant_scalar(v)
        )
        return int_count, bool_count

    user_int, user_bool = _group_counts("user")
    intro_int, intro_bool = _group_counts("introduced")

    # Arrays of var int/bool are often the *user-level* decision variables in FlatZinc.
    user_arr_int = sum(
        1
        for a in model.arrays.values()
        if a.get("is_var", False)
        and a.get("origin", "user") == "user"
        and a.get("elem_type") == "int"
    )
    user_arr_bool = sum(
        1
        for a in model.arrays.values()
        if a.get("is_var", False)
        and a.get("origin", "user") == "user"
        and a.get("elem_type") == "bool"
    )
    intro_arr_int = sum(
        1
        for a in model.arrays.values()
        if a.get("is_var", False)
        and a.get("origin", "user") == "introduced"
        and a.get("elem_type") == "int"
    )
    intro_arr_bool = sum(
        1
        for a in model.arrays.values()
        if a.get("is_var", False)
        and a.get("origin", "user") == "introduced"
        and a.get("elem_type") == "bool"
    )

    header: List[str] = []
    total_scalars = user_int + user_bool + intro_int + intro_bool
    if total_scalars == 0:
        header.append("The model contains no scalar variables.")
    else:
        intro_parts: List[str] = []
        if intro_int:
            intro_parts.append(f"{intro_int} integer")
        if intro_bool:
            intro_parts.append(f"{intro_bool} Boolean")

        user_parts: List[str] = []
        if user_int:
            user_parts.append(f"{user_int} integer")
        if user_bool:
            user_parts.append(f"{user_bool} Boolean")

        header.append(
            "The model contains "
            + f"{intro_int + intro_bool} compiler-introduced ({', '.join(intro_parts) if intro_parts else '0'}) scalar variables "
            + f"and {user_int + user_bool} user-introduced ({', '.join(user_parts) if user_parts else '0'}) scalar variables."
        )

    total_user_arrays = user_arr_int + user_arr_bool
    total_intro_arrays = intro_arr_int + intro_arr_bool
    if total_user_arrays or total_intro_arrays:
        arr_parts: List[str] = []
        if total_user_arrays:
            detail: List[str] = []
            if user_arr_int:
                detail.append(f"{user_arr_int} int[]")
            if user_arr_bool:
                detail.append(f"{user_arr_bool} bool[]")
            arr_parts.append(f"{total_user_arrays} user-defined ({', '.join(detail)})")
        if total_intro_arrays:
            detail = []
            if intro_arr_int:
                detail.append(f"{intro_arr_int} int[]")
            if intro_arr_bool:
                detail.append(f"{intro_arr_bool} bool[]")
            arr_parts.append(
                f"{total_intro_arrays} compiler-introduced ({', '.join(detail)})"
            )
        header.append("Arrays of decision variables: " + " and ".join(arr_parts) + ".")

    def _append_domain_stats(sizes: List[int]):
        if not sizes:
            return
        min_size = min(sizes)
        max_size = max(sizes)
        known_count = len(sizes)

        # Split [min_size, max_size] into 4 equal-ish integer intervals.
        span = max_size - min_size + 1
        bounds = []
        for i in range(4):
            lo = min_size + (span * i) // 4
            hi = min_size + (span * (i + 1)) // 4 - 1
            if i == 3:
                hi = max_size
            if hi >= lo:
                bounds.append((lo, hi))

        bucket_lines: List[str] = []
        for lo, hi in bounds:
            bucket = [s for s in sizes if lo <= s <= hi]
            if not bucket:
                continue
            pct = 100.0 * len(bucket) / known_count
            avg = mean(bucket)
            bucket_lines.append(
                f"{pct:.1f}% have domain size in [{lo}, {hi}] (avg size {avg:.2f})"
            )

        if total_scalars:
            prefix = f"Among {known_count} of the {total_scalars} scalar variables with known finite domains, "
        else:
            prefix = f"Among {known_count} scalar variables with known finite domains, "
        header.append(prefix + "; ".join(bucket_lines) + ".")

    _append_domain_stats(scalar_domain_sizes_all)

    unknown_total = unknown_domain_count_user + unknown_domain_count_introduced
    if unknown_total:
        header.append(f"{unknown_total} scalar variables have unknown domains.")

    if lines:
        return "\n".join(header + [""] + lines)
    return "\n".join(header)

def describe_constraints(model):
    counts = defaultdict(int)
    arity_sums = defaultdict(int)

    def _is_constant_scalar(v: dict) -> bool:
        d = v.get("domain")
        return isinstance(d, Domain) and d.min_value == d.max_value

    scalar_names = {
        name
        for name, v in model.variables.items()
        if not _is_constant_scalar(v)
    }
    array_names = {name for name, a in model.arrays.items() if a.get("is_var", False)}
    token_re = re.compile(r"\b[A-Za-z]\w*\b")

    for c in model.constraints:
        if isinstance(c, dict):
            ctype = c.get("type")
            args = c.get("args", "")
        else:
            ctype = c
            args = ""

        if not ctype:
            continue

        counts[ctype] += 1

        # Best-effort arity: number of distinct variable identifiers mentioned in args.
        # Count both scalar variables and arrays-of-vars (many FlatZinc globals take arrays).
        if args:
            tokens = token_re.findall(args)
            arity = len({t for t in tokens if t in scalar_names or t in array_names})
        else:
            arity = 0
        arity_sums[ctype] += arity

    # Heuristic classification: fixed-arity constraints are treated as "primitive"; others as "global".
    # FlatZinc global constraints typically accept arrays (variable arity), e.g., all_different, element,
    # linear constraints, clauses, and scheduling constraints like cumulative.
    fixed_arity = {
        "int_eq",       # binary
        "int_ne",       # binary
        "int_le",       # binary
        "bool2int",     # unary + int
        "int_times",    # ternary
    }

    def _to_parenthetical(desc: str) -> str:
        desc = (desc or "").strip()
        if desc.endswith("."):
            desc = desc[:-1]
        if desc and desc[0].isupper():
            desc = desc[0].lower() + desc[1:]
        return desc

    def _fmt_line(ctype: str, count: int) -> str:
        desc = CONSTRAINT_TEXT.get(
            ctype,
            f"Constraints of type {ctype} restrict relationships between variables",
        )
        avg_arity = (arity_sums.get(ctype, 0) / count) if count else 0.0
        # Example: "5 array_int_element constraints with average arity 2.00 (element constraints ...)"
        plural = "constraint" if count == 1 else "constraints"
        return (
            f"  {ctype}: {count} {plural} with average arity {avg_arity:.2f} "
            f"({_to_parenthetical(desc)})"
        )

    total = sum(counts.values())
    primitives: List[str] = []
    globals_: List[str] = []
    for ctype, count in sorted(counts.items()):
        if ctype in fixed_arity:
            primitives.append(_fmt_line(ctype, count))
        else:
            globals_.append(_fmt_line(ctype, count))

    lines: List[str] = []
    if globals_:
        lines.append("Global constraints (variable arity):")
        lines.extend(globals_)
    if primitives:
        if lines:
            lines.append("")
        lines.append("Non-global constraints (fixed arity):")
        lines.extend(primitives)

    lines.append(f"\nTotal constraints: {total}")
    return "\n".join(lines)

# ============================================================
# Output
# ============================================================

def summarize(model):
    print("=" * 60)
    print("PROBLEM SUMMARY")
    print("=" * 60)

    print("\nProblem:")
    problem_txt = (describe_problem(model) or "").strip()
    search_txt = (describe_search(model.search, model=model) or "").strip()
    if search_txt:
        # The user asked for: "<problem>. Where the model suggests ..."
        if search_txt.startswith("The model suggests"):
            search_txt = "Where the model suggests" + search_txt[len("The model suggests"):]
        combined = (problem_txt.rstrip(".") + ". " + search_txt) if problem_txt else search_txt
    else:
        combined = problem_txt
    print(" ", combined)

    print("\nVariables:")
    print(describe_variables_detailed(model))

    print("\nConstraints:")
    print(describe_constraints(model))

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
