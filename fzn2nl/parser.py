#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

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


def split_top_level_commas(s: str) -> List[str]:
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

def _parse_vars_expr(expr: str) -> List[str]:
    expr = expr.strip()
    if not expr:
        return []
    if expr.startswith("[") and expr.endswith("]"):
        inner = expr[1:-1].strip()
        if not inner:
            return []
        return [v.strip() for v in split_top_level_commas(inner) if v.strip()]
    # Single variable/identifier
    return [expr]

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
            args = split_top_level_commas(inner)
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
            args = split_top_level_commas(inner)
            if not args:
                return None
            # Usually seq_search([search1, search2, ...])
            first = args[0].strip()
            phases: List[dict] = []
            if first.startswith("[") and first.endswith("]"):
                inner_list = first[1:-1].strip()
                elems = split_top_level_commas(inner_list) if inner_list else []
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