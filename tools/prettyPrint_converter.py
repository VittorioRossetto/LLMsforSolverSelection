"""
Convert a CSV-style line of feature values into a pretty-printed table.
"""

# --- 1. Ordered list of all feature names ---
FEATURE_NAMES = [
    "c_avg_deg_cons", "c_avg_dom_cons", "c_avg_domdeg_cons",
    "c_bounds_d", "c_bounds_r", "c_bounds_z",
    "c_cv_deg_cons", "c_cv_dom_cons", "c_cv_domdeg_cons",
    "c_domain", "c_ent_deg_cons", "c_ent_dom_cons", "c_ent_domdeg_cons",
    "c_logprod_deg_cons", "c_logprod_dom_cons", "c_max_deg_cons",
    "c_max_dom_cons", "c_max_domdeg_cons", "c_min_deg_cons",
    "c_min_dom_cons", "c_min_domdeg_cons", "c_num_cons", "c_priority",
    "c_ratio_cons", "c_sum_ari_cons", "c_sum_dom_cons",
    "c_sum_domdeg_cons", "d_array_cons", "d_bool_cons",
    "d_bool_vars", "d_float_cons", "d_float_vars", "d_int_cons",
    "d_int_vars", "d_ratio_array_cons", "d_ratio_bool_cons",
    "d_ratio_bool_vars", "d_ratio_float_cons", "d_ratio_float_vars",
    "d_ratio_int_cons", "d_ratio_int_vars", "d_ratio_set_cons",
    "d_ratio_set_vars", "d_set_cons", "d_set_vars", "gc_diff_globs",
    "gc_global_cons", "gc_ratio_diff", "gc_ratio_globs",
    "o_deg", "o_deg_avg", "o_deg_cons", "o_deg_std", "o_dom",
    "o_dom_avg", "o_dom_deg", "o_dom_std", "ns_bool_search",
    "ns_first_fail", "ns_goal", "ns_indomain_max", "ns_indomain_min",
    "ns_input_order", "ns_int_search", "ns_labeled_vars",
    "ns_other_val", "ns_other_var", "ns_set_search",
    "v_avg_deg_vars", "v_avg_dom_vars", "v_avg_domdeg_vars",
    "v_cv_deg_vars", "v_cv_dom_vars", "v_cv_domdeg_vars",
    "v_def_vars", "v_ent_deg_vars", "v_ent_dom_vars",
    "v_ent_domdeg_vars", "v_intro_vars", "v_logprod_deg_vars",
    "v_logprod_dom_vars", "v_max_deg_vars", "v_max_dom_vars",
    "v_max_domdeg_vars", "v_min_deg_vars", "v_min_dom_vars",
    "v_min_domdeg_vars", "v_num_aliases", "v_num_consts",
    "v_num_vars", "v_ratio_bounded", "v_ratio_vars",
    "v_sum_deg_vars", "v_sum_dom_vars", "v_sum_domdeg_vars"
]

# --- Descriptions for each feature (must match order and length of FEATURE_NAMES) ---
FEATURE_DESCRIPTIONS = [
    "Average of the constraints degree",
    "Average of the constraints domain",
    "Average of the ratio constraints domain/degree",
    "No of constraints using 'boundsD' annotation",
    "No of constraints using 'boundsR' annotation",
    "No of constraints using 'boundsZ' or 'bounds' annotation",
    "Coefficient of Variation of constraints degree",
    "Coefficient of Variation of constraints domain",
    "Coefficient of Variation of the ratio constraints domain/degree",
    "No of constraints using 'domain' annotation",
    "Entropy of constraints degree",
    "Entropy of constraints domain",
    "Entropy of the ratio constraints domain/degree",
    "Logarithm of the product of constraints degree",
    "Logarithm of the product of constraints domain",
    "Maximum of the constraints degree",
    "Maximum of the constraints domain",
    "Maximum of the ratio constraints domain/degree",
    "Minimum of the constraints degree",
    "Minimum of the constraints domain",
    "Minimum of the ratio constraints domain/degree",
    "Total no of constraints",
    "No of constraints using 'priority' annotation",
    "Ratio no of constraints / no of variables",
    "Sum of constraints arity",
    "Sum of constraints domain",
    "Sum of the ratio constraints domain/degree",
    "No of array constraints",
    "No of boolean constraints",
    "No of boolean variables",
    "No of float constraints",
    "No of float variables",
    "No of integer constraints",
    "No of integer variables",
    "Ratio array constraints / total no of constraints",
    "Ratio boolean constraints / total no of constraints",
    "Ratio boolean variables / total no of variables",
    "Ratio float constraints / total no of constraints",
    "Ratio float variables / total no of variables",
    "Ratio integer constraints / total no of constraints",
    "Ratio integer variables / total no of variables",
    "Ratio set constraints / total no of constraints",
    "Ratio set variables / total no of variables",
    "No of set constraints",
    "No of set variables",
    "No of different global constraints",
    "Total no of global constraints",
    "Ratio different global constraints / no of global constraints",
    "Ratio no of global constraints / total no of constraints",
    "Degree of the objective variable",
    "Ratio degree of the objective variable / average of var degree",
    "Ratio degree of the objective variable / number of constraints",
    "Standardization of the degree of the objective variable",
    "Domain size of the objective variable",
    "Ratio domain of the objective variable / average of var domain",
    "Ratio domain of the objective variable / degree of the obj var",
    "Standardization of the domain of the objective variable",
    "Number of 'bool_search' annotations",
    "Number of 'int_search' annotations",
    "Solve goal (1 = satisfy, 2 = minimize, 3 = maximize)",
    "Number of 'indomain_max' annotations",
    "Number of 'indomain_min' annotations",
    "Number of 'input_order' annotations",
    "Number of 'int_search' annotations",
    "Number of variables to be assigned",
    "Number of other value search heuristics",
    "Number of other variable search heuristics",
    "Number of 'set_search' annotations",
    "Average of the variables degree",
    "Average of the variables domain",
    "Average of the ratio variables domain/degree",
    "Coefficient of Variation of variables degree",
    "Coefficient of Variation of variables domain",
    "Coefficient of Variation of the ratio variables domain/degree",
    "Number of defined variables",
    "Entropy of variables degree",
    "Entropy of variables domain",
    "Entropy of the ratio variables domain/degree",
    "Number of introduced variables",
    "Logarithm of the product of variables degree",
    "Logarithm of the product of variables domain",
    "Maximum of the variables degree",
    "Maximum of the variables domain",
    "Maximum of the ratio variables domain/degree",
    "Minimum of the variables degree",
    "Minimum of the variables domain",
    "Minimum of the ratio variables domain/degree",
    "Number of alias variables",
    "Number of constant variables",
    "Total no of variables variables",
    "Ratio (aliases + constants) / total no of variables",
    "Ratio no of variables / no of constraints",
    "Sum of variables degree",
    "Sum of variables domain",
    "Sum of the ratio variables domain/degree"
]

# --- CSV input line  ---
csv_line = "2.56793,7.8722,2.85007,0,0,0,1.69744,10.7855,1.1637,0,1.26717,3.02645,2.19232,7965.15,13787.2,318,6878.38,21.6301,1,2,1,6617,0,1.06059,18452,52090.3,18858.9,1180,1056,3680,0,0,2600,2559,0.178329,0.159589,0.589838,0,0,0.392927,0.410162,0,0,0,0,4,583,0.00686106,0.0881064,1,0.367173,0.000151126,-0.861757,2e+07,23.5403,2e+07,0,0,1,2,0,1,0,1,1,0,0,0,2.72351,849605,285377,0.734346,0,0,5436,1.12553,1.27901,1.39443,6611,7630.65,18876.5,10,2e+07,2e+07,0,2,0.2,708,1700,6239,0.385959,0.942874,16992,5.30069e+09,1.78046e+09"

# --- Convert ---
values = csv_line.split(",")

if len(values) != len(FEATURE_NAMES):
    raise ValueError(f"Expected {len(FEATURE_NAMES)} values but got {len(values)}.")

# --- Pretty printing ---
IDENT_WIDTH = 24
VALUE_WIDTH = 20

rows = []
rows.append(f"{'IDENTIFIER':<{IDENT_WIDTH}} {'VALUE':<{VALUE_WIDTH}} DESCRIPTION")
rows.append("=" * (IDENT_WIDTH + VALUE_WIDTH + 30))
if len(FEATURE_DESCRIPTIONS) != len(FEATURE_NAMES):
    raise RuntimeError('FEATURE_DESCRIPTIONS must have the same length as FEATURE_NAMES')

for name, val, desc in zip(FEATURE_NAMES, values, FEATURE_DESCRIPTIONS):
    rows.append(f"{name:<{IDENT_WIDTH}} {val:<{VALUE_WIDTH}} {desc}")

# Print everything as a single line where elements are separated by the literal '\n'
print('\\n'.join(rows))
