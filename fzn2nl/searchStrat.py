from typing import Optional, List
from parser import FlatZincModel
from nlMappings import SEARCH_COMPLETENESS, SEARCH_VALUE_STRATEGY, SEARCH_VAR_STRATEGY

from parser import *

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
    return f"domain [{lo}, {hi}]"

def _domain_text_for_scalar(model: "FlatZincModel", name: str) -> str:
    v = model.variables.get(name)
    if not v:
        return "domain unknown"
    d = v.get("domain")
    if d is None:
        return "domain unknown"
    return f"domain [{d.min_value}, {d.max_value}]"

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


def describe_search(search, model: Optional["FlatZincModel"] = None):
    if not search:
        return "No explicit search strategy is specified."

    def _array_var_count(a: dict) -> Optional[int]:
        items = a.get("items") or []
        items = [str(it).strip() for it in items if str(it).strip()]
        if items:
            return len(items)
        length = a.get("length")
        return length if isinstance(length, int) and length >= 0 else None

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
                    f"integer search on 1 {vtype} variable with {domain_txt}, {strategy_txt}"
                    # f"involved in {constraints_txt}, {strategy_txt}"
                )
                return text

            if name in model.arrays:
                a = model.arrays[name]
                elem_type = a.get("elem_type", "int")
                length = a.get("length")
                length_txt = f"length {length}" if isinstance(length, int) else "unknown length"
                domain_txt = _domain_text_for_array(model, name)
                n_vars = _array_var_count(a)
                c_count, _c_types = _constraint_stats_for_name(model, name)
                examples = _constraint_texts_for_name(model, name, limit=3) if c_count > 0 else []
                constraints_txt = f"{c_count} constraints"
                if c_count > 0 and examples:
                    constraints_txt += f"( {', '.join(examples)} )"

                origin_clause = f"(from 1 array, {length_txt})"
                if isinstance(n_vars, int):
                    subject = f"{n_vars} {elem_type} variables"
                else:
                    subject = f"{elem_type} variables"

                text = f"integer search on {subject} with {domain_txt}, {origin_clause}, {strategy_txt}"
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
