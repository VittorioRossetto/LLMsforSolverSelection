import re
from typing import Optional, List, Dict, Tuple
from parser import Domain, FlatZincModel, split_top_level_commas, compute_variable_degrees
from nlMappings import *

def describe_objective_function(model: "FlatZincModel", max_depth: int = 2, max_len: int = 100) -> Optional[str]:
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
    parts = split_top_level_commas(args) if args else []

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
    parts = split_top_level_commas(inner)
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
    return [p.strip() for p in split_top_level_commas(inner)]


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
                f" The objective is to {model.problem_type} an objective variable with unknown domain and degree {deg}."
            )
            if obj_expr:
                suffix += f" {obj_expr}."
            return base + suffix

        size = d.max_value - d.min_value + 1
        mean_value = (d.min_value + d.max_value) / 2
        suffix = (
            f" The objective is to {model.problem_type} an objective variable with domain [{d.min_value}, {d.max_value}]"
            f" (size {size}, mean {mean_value:.2f}) and degree {deg}."
        )
        if obj_expr:
            suffix += f" {obj_expr}."
        return base + suffix

    return "Problem type could not be determined."