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

_FZN_DESCRIPTIONS_CACHE: Optional[Dict[str, str]] = None
_FZN_CATEGORIZED_CACHE: Optional[Tuple[Dict[str, List[str]], Dict[str, str]]] = None
