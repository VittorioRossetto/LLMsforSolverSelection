import re
from statistics import mean
from typing import Optional, List
from parser import *
from nlMappings import *



def describe_variables_detailed(model):
    lines = []
    domain_sizes_user: List[int] = []
    domain_sizes_introduced: List[int] = []
    domain_sizes_all: List[int] = []
    unknown_domain_count_user = 0
    unknown_domain_count_introduced = 0
    unknown_domain_count_other = 0
    anonymous_array_elems_user = 0
    anonymous_array_elems_introduced = 0
    arrays_with_unknown_length_user = 0
    arrays_with_unknown_length_introduced = 0
    arrays_total = 0
    arrays_elems_estimated = 0

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

    # Track all *variable names* we can identify (avoids double counting when
    # arrays are initialized with named scalar variables).
    counted_names: set[str] = set()

    def _counted_add(name: str) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        if name in counted_names:
            return False
        counted_names.add(name)
        return True

    def _array_var_count(a: dict) -> Optional[int]:
        items = a.get("items") or []
        items = [str(it).strip() for it in items if str(it).strip()]
        if items:
            return len(items)
        length = a.get("length")
        return length if isinstance(length, int) and length >= 0 else None

    # If a variable appears as an element of a user-origin decision array, report it
    # as user-introduced even if the FlatZinc backend declared it as introduced.
    user_decision_var_names: set[str] = set()
    for _aname, a in model.arrays.items():
        if not a.get("is_var", False):
            continue
        if a.get("origin", "user") != "user":
            continue
        items = a.get("items") or []
        for it in items:
            it = str(it).strip()
            if not it:
                continue
            if re.fullmatch(r"-?\d+", it):
                continue
            user_decision_var_names.add(it)

    # Gather (named) variable domain statistics for scalar variables.
    # Domain-size analysis is only meaningful for integer variables; ignore Booleans.
    for name, v in model.variables.items():
        if _is_constant_scalar(v):
            continue
        if v.get("type") == "bool":
            continue
        size = _domain_size(v.get("domain"))
        origin = v.get("origin", "user")
        if name in user_decision_var_names:
            origin = "user"
        if size is None:
            if origin == "introduced":
                unknown_domain_count_introduced += 1
            else:
                unknown_domain_count_user += 1
        else:
            domain_sizes_all.append(size)
            if origin == "introduced":
                domain_sizes_introduced.append(size)
            else:
                domain_sizes_user.append(size)

    # Count variables by iterating names (needed to avoid double-counting array items).
    def _is_int_lit(s: str) -> bool:
        return bool(re.fullmatch(r"-?\d+", (s or "").strip()))

    type_counts = {
        ("user", "int"): 0,
        ("user", "bool"): 0,
        ("introduced", "int"): 0,
        ("introduced", "bool"): 0,
    }

    for name, v in model.variables.items():
        if _is_constant_scalar(v):
            continue
        if not _counted_add(name):
            continue
        origin = v.get("origin", "user")
        if name in user_decision_var_names:
            origin = "user"
        vtype = v.get("type", "int")
        if (origin, vtype) in type_counts:
            type_counts[(origin, vtype)] += 1

    # Treat arrays of decision variables as their element variables.
    for aname, a in model.arrays.items():
        if not a.get("is_var", False):
            continue
        arrays_total += 1
        origin = a.get("origin", "user")
        elem_type = a.get("elem_type", "int")
        items = a.get("items") or []
        items = [str(it).strip() for it in items if str(it).strip()]

        if items:
            arrays_elems_estimated += len(items)
            for it in items:
                if _is_int_lit(it):
                    # Shouldn't happen for var arrays, but be defensive.
                    continue
                if it in model.variables and _is_constant_scalar(model.variables[it]):
                    continue
                if not _counted_add(it):
                    continue
                # If the element isn't declared as a scalar var, we can still count it
                # as a variable with unknown domain (integers only).
                if it not in model.variables and elem_type != "bool":
                    if origin == "introduced":
                        unknown_domain_count_introduced += 1
                    elif origin == "user":
                        unknown_domain_count_user += 1
                    else:
                        unknown_domain_count_other += 1
                    if (origin, elem_type) in type_counts:
                        type_counts[(origin, elem_type)] += 1
            continue

        n = _array_var_count(a)
        if isinstance(n, int):
            arrays_elems_estimated += n
            if origin == "introduced":
                anonymous_array_elems_introduced += n
                if elem_type != "bool":
                    unknown_domain_count_introduced += n
            elif origin == "user":
                anonymous_array_elems_user += n
                if elem_type != "bool":
                    unknown_domain_count_user += n
            else:
                if elem_type != "bool":
                    unknown_domain_count_other += n

            if (origin, elem_type) in type_counts:
                type_counts[(origin, elem_type)] += n
        else:
            if origin == "introduced":
                arrays_with_unknown_length_introduced += 1
            else:
                arrays_with_unknown_length_user += 1

    header: List[str] = []
    total_variables = len(counted_names) + anonymous_array_elems_user + anonymous_array_elems_introduced
    total_user = type_counts[("user", "int")] + type_counts[("user", "bool")]
    total_intro = type_counts[("introduced", "int")] + type_counts[("introduced", "bool")]
    total_int = type_counts[("user", "int")] + type_counts[("introduced", "int")]
    total_bool = type_counts[("user", "bool")] + type_counts[("introduced", "bool")]

    if total_variables == 0:
        header.append("The model contains no variables.")
    else:
        header.append(
            "The model contains "
            + f"{total_variables} variables ({total_int} integer, {total_bool} Boolean): "
            + f"{total_intro} compiler-introduced and {total_user} user-introduced."
        )

    # if arrays_total:
    #     unknown_arrays = arrays_with_unknown_length_user + arrays_with_unknown_length_introduced
    #     note = (
    #         f"Decision-variable arrays are treated as their element variables: "
    #         f"{arrays_total} arrays contribute {arrays_elems_estimated} variables"
    #     )
    #     if unknown_arrays:
    #         note += f" ({unknown_arrays} arrays have unknown length)"
    #     header.append(note + ".")

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

        prefix = f"Among {known_count} integer variables with known finite domains, "
        header.append(prefix + "; ".join(bucket_lines) + ".")

    _append_domain_stats(domain_sizes_all)

    unknown_total = unknown_domain_count_user + unknown_domain_count_introduced + unknown_domain_count_other
    if unknown_total:
        header.append(f"{unknown_total} integer variables have unknown domains.")

    if lines:
        return "\n".join(header + [""] + lines)
    return "\n".join(header)
