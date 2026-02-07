import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
from parser import Domain
from nlMappings import CONSTRAINT_TEXT

_FZN_DESCRIPTIONS_CACHE: Optional[Dict[str, str]] = None
_FZN_CATEGORIZED_CACHE: Optional[Tuple[Dict[str, List[str]], Dict[str, str]]] = None

def _load_fzn_constraint_descriptions() -> Dict[str, str]:
    """Load a mapping constraint_name -> description from fzn_descriptions.json.

    Preferred format (simple JSON object):
        {"fzn_table_int": "A table constraint ...", ...}

    Backwards compatibility: we also accept the older list-of-objects format:
        [{"constraint": "fzn_table_int", "description": "...", ...}, ...]

    Best-effort: if the file is missing or malformed we just return {}.
    """
    global _FZN_DESCRIPTIONS_CACHE
    if _FZN_DESCRIPTIONS_CACHE is not None:
        return _FZN_DESCRIPTIONS_CACHE

    path = Path(__file__).with_name("fzn_descriptions.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _FZN_DESCRIPTIONS_CACHE = {}
        return _FZN_DESCRIPTIONS_CACHE

    mapping: Dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key = str(k).strip()
            desc = str(v).strip()
            if not key or not desc:
                continue
            mapping.setdefault(key, desc)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            key = (item.get("constraint") or "").strip()
            desc = (item.get("description") or "").strip()
            if not key or not desc:
                continue
            mapping.setdefault(key, desc)

    _FZN_DESCRIPTIONS_CACHE = mapping
    return _FZN_DESCRIPTIONS_CACHE


def _load_fzn_constraint_categories() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Load categorized constraint descriptions.

    Returns:
      - constraint_to_categories: constraint_name -> list of category names
      - constraint_to_description: constraint_name -> description

    The file format is expected to be:
      {
        "categories": {
          "Some category": {"constraint_name": "desc", ...},
          ...
        }
      }

    Best-effort: if missing/malformed, returns ({}, {}).
    """
    global _FZN_CATEGORIZED_CACHE
    if _FZN_CATEGORIZED_CACHE is not None:
        return _FZN_CATEGORIZED_CACHE

    path = Path(__file__).with_name("fzn_descriptions_categorized.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _FZN_CATEGORIZED_CACHE = ({}, {})
        return _FZN_CATEGORIZED_CACHE

    constraint_to_categories: Dict[str, List[str]] = defaultdict(list)
    constraint_to_description: Dict[str, str] = {}

    if isinstance(data, dict):
        cats = data.get("categories")
        if isinstance(cats, dict):
            for category_name, mapping in cats.items():
                if not isinstance(category_name, str) or not category_name.strip():
                    continue
                if not isinstance(mapping, dict):
                    continue
                cat = category_name.strip()
                for ctype, desc in mapping.items():
                    key = str(ctype).strip()
                    val = str(desc).strip()
                    if not key or not val:
                        continue
                    if cat not in constraint_to_categories[key]:
                        constraint_to_categories[key].append(cat)
                    # Keep the first description we see for a constraint.
                    constraint_to_description.setdefault(key, val)

        # Some entries are explicitly listed as multi-category.
        multi = data.get("multi_category")
        if isinstance(multi, dict):
            for ctype, cat_list in multi.items():
                key = str(ctype).strip()
                if not key:
                    continue
                if isinstance(cat_list, list):
                    for cat in cat_list:
                        cat = str(cat).strip()
                        if not cat:
                            continue
                        if cat not in constraint_to_categories[key]:
                            constraint_to_categories[key].append(cat)

        # Some entries are called out as uncategorized.
        unc = data.get("uncategorized")
        if isinstance(unc, dict):
            for ctype, desc in unc.items():
                key = str(ctype).strip()
                val = str(desc).strip()
                if not key or not val:
                    continue
                if "Uncategorized" not in constraint_to_categories[key]:
                    constraint_to_categories[key].append("Uncategorized")
                constraint_to_description.setdefault(key, val)

    _FZN_CATEGORIZED_CACHE = (dict(constraint_to_categories), constraint_to_description)
    return _FZN_CATEGORIZED_CACHE

def _resolve_constraint_description(
    ctype: str,
    descriptions: Dict[str, str],
) -> Optional[str]:
    """Resolve a human description for a FlatZinc constraint type.

    Lookup strategy (in order):
    1) Exact match (e.g. 'fzn_lex_lesseq_bool_reif')
    2) If prefixed with 'fzn_', strip the prefix AND strip one trailing suffix token
       (e.g. 'fzn_lex_lesseq_bool_reif' -> 'lex_lesseq_bool')
    3) Keep stripping one trailing suffix token at a time
       (e.g. 'lex_lesseq_bool' -> 'lex_lesseq' -> 'lex')
    """
    ctype = (ctype or "").strip()
    if not ctype:
        return None

    def _get(key: str) -> Optional[str]:
        key = (key or "").strip()
        if not key:
            return None
        val = descriptions.get(key)
        return (val or "").strip() or None

    # 1) Exact match
    desc = _get(ctype)
    if desc:
        return desc

    # 2) Strip 'fzn_' prefix + one trailing token
    base = ctype
    if base.startswith("fzn_"):
        base = base[len("fzn_") :]
        parts = [p for p in base.split("_") if p]
        if len(parts) >= 2:
            candidate = "_".join(parts[:-1])
            desc = _get(candidate)
            if desc:
                return desc
            base = candidate

    # 3) Iteratively strip trailing suffix tokens
    parts = [p for p in base.split("_") if p]
    while len(parts) >= 2:
        parts = parts[:-1]
        candidate = "_".join(parts)
        desc = _get(candidate)
        if desc:
            return desc

    return None


def _resolve_key_in_mapping(ctype: str, mapping: Dict[str, object]) -> Optional[str]:
    """Resolve `ctype` to a key that exists in `mapping`.

    This is similar to `_resolve_constraint_description` but returns the matched
    key rather than the description, and it first tries the full name after
    stripping a leading `fzn_`.
    """
    ctype = (ctype or "").strip()
    if not ctype or not mapping:
        return None

    keys = set(mapping.keys())
    if ctype in keys:
        return ctype

    # Try stripping the common FlatZinc prefix.
    candidates: List[str] = []
    if ctype.startswith("fzn_"):
        candidates.append(ctype[len("fzn_") :])
    candidates.append(ctype)

    for base in candidates:
        base = (base or "").strip()
        if not base:
            continue
        if base in keys:
            return base

        parts = [p for p in base.split("_") if p]
        while len(parts) >= 2:
            parts = parts[:-1]
            cand = "_".join(parts)
            if cand in keys:
                return cand

    return None

def describe_constraints(model, categorize: bool = False):
    counts = defaultdict(int)
    arity_sums = defaultdict(int)
    fzn_desc = _load_fzn_constraint_descriptions()
    categorized_map, categorized_desc = _load_fzn_constraint_categories() if categorize else ({}, {})

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

        # Best-effort arity: treat arrays of variables as their element variables.
        # If we know the array items, expand them; otherwise fall back to the array length.
        if args:
            tokens = token_re.findall(args)
            mentioned: set[str] = {t for t in tokens if t in scalar_names}
            anon_count = 0
            for t in tokens:
                if t not in array_names:
                    continue
                a = model.arrays.get(t) or {}
                items = a.get("items") or []
                items = [str(it).strip() for it in items if str(it).strip()]
                if items:
                    for it in items:
                        if it in scalar_names:
                            mentioned.add(it)
                else:
                    length = a.get("length")
                    if isinstance(length, int) and length >= 0:
                        anon_count += length
            arity = len(mentioned) + anon_count
        else:
            arity = 0
        arity_sums[ctype] += arity

    def _to_parenthetical(desc: str) -> str:
        desc = (desc or "").strip()
        if desc.endswith("."):
            desc = desc[:-1]
        if desc and desc[0].isupper():
            desc = desc[0].lower() + desc[1:]
        return desc

    def _fmt_line(ctype: str, count: int) -> str:
        # Prefer categorized descriptions (exact match), then the generic resolver.
        desc = None
        if categorize and categorized_desc:
            key = _resolve_key_in_mapping(ctype, categorized_desc)
            desc = categorized_desc.get(key) if key else None
        desc = (
            (desc or "").strip()
            or _resolve_constraint_description(ctype, fzn_desc)
            or CONSTRAINT_TEXT.get(ctype)
            or f"Constraints of type {ctype} restrict relationships between variables"
        )
        avg_arity = (arity_sums.get(ctype, 0) / count) if count else 0.0
        # Example: "5 array_int_element constraints with average arity 2.00 (element constraints ...)"
        plural = "constraint" if count == 1 else "constraints"
        return (
            f"  {ctype}: {count} {plural} with average arity {avg_arity:.2f} "
            f"({_to_parenthetical(desc)})"
        )

    if not categorize:
        # Return a single unified list (sorted by constraint type).
        lines: List[str] = []
        for ctype, count in sorted(counts.items()):
            lines.append(_fmt_line(ctype, count))
        return "\n".join(lines)

    # Categorized view: group constraint types under the categories from
    # fzn_descriptions_categorized.json. Avoid double-counting types that appear
    # in multiple categories by assigning a single primary category.
    def _primary_category(ctype: str) -> str:
        key = _resolve_key_in_mapping(ctype, categorized_map) if categorized_map else None
        cats = categorized_map.get(key) or []
        if not cats:
            return "Uncategorized"
        # Deterministic choice.
        return sorted(cats)[0]

    category_to_types: Dict[str, List[str]] = defaultdict(list)
    for ctype in counts.keys():
        category_to_types[_primary_category(ctype)].append(ctype)

    def _category_header(cat: str) -> str:
        types = category_to_types.get(cat, [])
        total = sum(counts[t] for t in types)
        total_arity = sum(arity_sums.get(t, 0) for t in types)
        avg = (total_arity / total) if total else 0.0
        t_plural = "type" if len(types) == 1 else "types"
        c_plural = "constraint" if total == 1 else "constraints"
        return f"{cat}: {len(types)} {t_plural}, {total} {c_plural} (avg arity {avg:.2f})"

    lines: List[str] = []
    # Keep Uncategorized last for readability.
    cats_sorted = sorted([c for c in category_to_types.keys() if c != "Uncategorized"])
    if "Uncategorized" in category_to_types:
        cats_sorted.append("Uncategorized")

    for cat in cats_sorted:
        lines.append(_category_header(cat))
        for ctype in sorted(category_to_types.get(cat, [])):
            lines.append(_fmt_line(ctype, counts[ctype]))
        lines.append("")

    # Trim trailing blank line.
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
