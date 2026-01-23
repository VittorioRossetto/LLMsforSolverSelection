import json
import pandas as pd
try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sns = None
from itertools import chain
import numpy as np
import os


def _require_plotting():
    if plt is None or sns is None:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn"
        )


def get_significative_solvers() -> list[str]:
    """Return the project's SIGNIFICATIVE_SOLVERS list.

    Best-effort: import from utils.py if available; otherwise fall back to a
    local copy to keep notebooks usable when paths differ.
    """
    try:
        from utils import SIGNIFICATIVE_SOLVERS  # type: ignore
        return list(SIGNIFICATIVE_SOLVERS)
    except Exception:
        return [
            "cbc-free",
            "choco-solver__cp_-free",
            "choco-solver__cp_-par",
            "choco-solver__cp-sat_-free",
            "cp_optimizer-free",
            "cplex-free",
            "gurobi-free",
            "highs-free",
            "izplus-free",
            "jacop-free",
            "pumpkin-free",
            "scip-free",
            "sicstus_prolog-free",
        ]


def filter_to_solvers(df: pd.DataFrame, solvers: list[str] | set[str] | None, *, solver_col: str) -> pd.DataFrame:
    """Filter a dataframe to rows whose solver column is in `solvers`.

    If `solvers` is None or empty, returns df unchanged.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not solvers:
        return df
    allowed = set(str(s).strip() for s in solvers if str(s).strip())
    if not allowed:
        return df
    if solver_col not in df.columns:
        return df
    out = df.copy()
    out[solver_col] = out[solver_col].astype(str).str.strip()
    return out[out[solver_col].isin(allowed)].copy()

# --- Flatten LLMResults into dataframe ---
def ResultsFlattener(LLMResults):
    llm_rows = []
    for provider, models in (LLMResults or {}).items():
        if not isinstance(models, dict):
            continue
        for model, problems in models.items():
            if not isinstance(problems, dict):
                continue
            for problem, instances in problems.items():
                if not isinstance(instances, dict):
                    continue
                for instance, data in instances.items():
                    if not isinstance(data, dict):
                        top3_val = None
                        time_val = None
                    else:
                        top3_val = data.get('top3') if 'top3' in data else data.get('top_3') if 'top_3' in data else data.get('suggested') if 'suggested' in data else None
                        time_val = data.get('time_seconds') if 'time_seconds' in data else data.get('time') if 'time' in data else None
                    if isinstance(top3_val, list):
                        top3_list = top3_val
                        top3_str = ', '.join(map(str, top3_list)) if top3_list else None
                        top1 = top3_list[0] if top3_list else None
                    elif isinstance(top3_val, str):
                        parts = [p.strip() for p in top3_val.replace(';',',').split(',') if p.strip()]
                        top3_list = parts if parts else None
                        top3_str = ', '.join(parts) if parts else top3_val
                        top1 = parts[0] if parts else (top3_val or None)
                    else:
                        top3_list = None
                        top3_str = None
                        top1 = None
                    llm_rows.append({
                        'provider': provider,
                        'model': model,
                        'problem': problem,
                        'instance': instance,
                        'top3_list': top3_list,
                        'top3': top3_str,
                        'top1': top1,
                        'time_seconds': time_val
                    })

    llm_df = pd.DataFrame(llm_rows)
    if not llm_df.empty:
        llm_df['time_seconds'] = pd.to_numeric(llm_df['time_seconds'], errors='coerce')
    return llm_df

# --- Flatten Minizinc Challenge results into dataframe ---
def mznResultsFlattener(MznResults):
    rows = []

    for problem, problem_data in MznResults.items():
        category = problem_data.get("category")
        
        for instance, solvers in problem_data.items():
            if instance == "category":
                continue  # skip category entry
            for solver_entry in solvers:
                rows.append({
                    "Problem": problem,
                    "Category": category,
                    "Instance": instance,
                    "Solver": solver_entry.get("Solver"),
                    "Status": solver_entry.get("Status"),
                    "Time": solver_entry.get("Time"),
                    "Objective": solver_entry.get("Objective"),
                    "Score": solver_entry.get("Score"),
                    "Score Area": solver_entry.get("Score Area"),
                })

    df = pd.DataFrame(rows)

    # Convert numeric columns
    numeric_cols = ["Time", "Objective", "Score", "Score Area"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


# --- Function to calculate score and add it to the df for singke solver performances on each solver ---
def scoreComputation(scored_df):
    """Compute MiniZinc-style scores per (Problem, Instance, Solver).

    Note: If you want to restrict scoring to a solver subset (e.g.
    SIGNIFICATIVE_SOLVERS), filter `scored_df` first using `filter_to_solvers(...)`.
    """
    # Normalize Status column
    scored_df['Status'] = scored_df['Status'].astype(str).str.upper()

    # Only consider numeric Objectives for best/worst computation
    valid_mask = scored_df['Status'].isin(['S', 'SC']) & scored_df['Objective'].notnull()
    valid_objs = scored_df.loc[valid_mask, ['Problem', 'Instance', 'Objective', 'Category']]

    # Compute best/worst per instance depending on MIN/MAX
    def best_worst(group):
        cat = group['Category'].iloc[0].upper()
        if cat == 'MIN':
            best = group['Objective'].min()
            worst = group['Objective'].max()
        else:
            best = group['Objective'].max()
            worst = group['Objective'].min()
        return pd.Series({'best': best, 'worst': worst})

    # Pandas 2.3+: avoid deprecation warning about grouping columns in apply.
    agg = valid_objs.groupby(['Problem', 'Instance']).apply(best_worst, include_groups=False).reset_index()

    # Merge back to the main dataframe
    scored_df = scored_df.merge(agg, on=['Problem', 'Instance'], how='left')

    # Determine solver-level labels (SAT vs OPT)
    def solver_label(row):
        cat = row['Category'].upper()
        if row['Status'] in ['UNK', 'ERR']:
            return 'UNK'
        if row['Status'] == 'S':
            if cat == 'SAT':
                return 'OPT'
            else:
                return 'SAT'
        if row['Status'] == 'SC':
            if cat == 'SAT':
                return 'OPT'
            if pd.isna(row['Objective']) or pd.isna(row['best']):
                return 'SAT'
            if cat == 'MIN' and row['Objective'] <= row['best']:
                return 'OPT'
            if cat == 'MAX' and row['Objective'] >= row['best']:
                return 'OPT'
            return 'SAT'
        return row['Status']

    scored_df['DerivedStatus'] = scored_df.apply(solver_label, axis=1)

    # Compute final score
    def compute_score(row):
        stat = row['DerivedStatus']
        val = row['Objective']
        best = row['best']
        worst = row['worst']
        cat = str(row['Category']).upper()

        if stat in ['UNK', 'ERR']:
            return 0.0
        if stat in ['OPT']:
            return 1.0
        if stat == 'SAT' and pd.notnull(best) and pd.notnull(worst) and best != worst:
            if cat == 'MIN':
                return 0.25 + 0.5 * (val - worst) / (best - worst)
            elif cat == 'MAX':
                return 0.25 + 0.5 * (best - val) / (best - worst)
        return np.nan

    scored_df['ComputedScore'] = scored_df.apply(compute_score, axis=1)
    return scored_df


def scoreComputation_subset(scored_df: pd.DataFrame, allowed_solvers: list[str] | set[str] | None = None) -> pd.DataFrame:
    """Convenience wrapper: compute scores using only `allowed_solvers`.

    This restricts both:
    - which solver rows are scored
    - and which solvers contribute to per-instance best/worst.
    """
    filtered = filter_to_solvers(scored_df, allowed_solvers, solver_col='Solver')
    if filtered is None or filtered.empty:
        return filtered
    return scoreComputation(filtered)

# --- Compute scores of LLM answers based on single solvers score ---
def compute_llm_scores(llm_df, scored_df, allowed_solvers: list[str] | set[str] | None = None):
    llm_expanded = llm_df.copy()

    llm_expanded = llm_expanded[llm_expanded['top3_list'].notnull()].copy()
    llm_expanded = llm_expanded.explode('top3_list').rename(columns={'top3_list': 'Solver'})
    llm_expanded['Solver'] = llm_expanded['Solver'].astype(str).str.strip()

    # Optional: restrict evaluation to a solver subset (e.g., SIGNIFICATIVE_SOLVERS).
    if allowed_solvers:
        llm_expanded = filter_to_solvers(llm_expanded, allowed_solvers, solver_col='Solver')
        scored_df = filter_to_solvers(scored_df, allowed_solvers, solver_col='Solver')

    # Standardize matching keys
    llm_expanded['problem'] = llm_expanded['problem'].astype(str)
    llm_expanded['instance'] = llm_expanded['instance'].astype(str)
    scored_df['Problem'] = scored_df['Problem'].astype(str)
    scored_df['Instance'] = scored_df['Instance'].astype(str)
    scored_df['Solver'] = scored_df['Solver'].astype(str).str.strip()

    # Merge: (problem, instance, solver)
    llm_scored = llm_expanded.merge(
        scored_df[['Problem', 'Instance', 'Solver', 'ComputedScore']],
        left_on=['problem', 'instance', 'Solver'],
        right_on=['Problem', 'Instance', 'Solver'],
        how='left'
    )

    # Treat missing matches as score 0 and compute best-of-top3 per instance
    llm_scored['ComputedScore'] = llm_scored['ComputedScore'].fillna(0.0)

    per_instance_best = (
        llm_scored
        .groupby(['provider', 'model', 'problem', 'instance'], as_index=False)
        .agg(InstanceBestScore=('ComputedScore', 'max'))
    )

    # Sum the best score per instance for each (provider, model)
    llm_summary = (
        per_instance_best
        .groupby(['provider', 'model'], as_index=False)
        .agg(
            LLM_TotalScore=('InstanceBestScore', 'sum'),
            InstancesCovered=('instance', 'nunique')
        )
    )

    llm_summary['LLM_AvgScore'] = llm_summary['LLM_TotalScore'] / llm_summary['InstancesCovered']
    return llm_summary

# --- Compute scores of LLM answers for best solver based on single solvers score ---
def compute_top1_llm_scores(llm_df, scored_df, allowed_solvers: list[str] | set[str] | None = None):
    # --- Extract Top-1 predictions only ---
    llm_top1 = llm_df.copy()
    llm_top1 = llm_top1[llm_top1['top1'].notnull()].copy()
    llm_top1 = llm_top1.rename(columns={'top1': 'Solver'})
    llm_top1['Solver'] = llm_top1['Solver'].astype(str)

    # Optional: restrict evaluation to a solver subset.
    if allowed_solvers:
        llm_top1 = filter_to_solvers(llm_top1, allowed_solvers, solver_col='Solver')
        scored_df = filter_to_solvers(scored_df, allowed_solvers, solver_col='Solver')

    # --- Standardize key names and merge with solver scores ---
    scored_df['Problem'] = scored_df['Problem'].astype(str)
    scored_df['Instance'] = scored_df['Instance'].astype(str)
    scored_df['Solver'] = scored_df['Solver'].astype(str)

    llm_top1['problem'] = llm_top1['problem'].astype(str)
    llm_top1['instance'] = llm_top1['instance'].astype(str)

    llmTSD_scored = llm_top1.merge(
        scored_df[['Problem', 'Instance', 'Solver', 'ComputedScore']],
        left_on=['problem', 'instance', 'Solver'],
        right_on=['Problem', 'Instance', 'Solver'],
        how='left'
    )

    # --- Aggregate by (provider, model) ---
    llm_top1_summary = (
        llmTSD_scored
        .groupby(['provider', 'model'], as_index=False)
        .agg(
            LLM_Top1_TotalScore=('ComputedScore', 'sum'),
            LLM_Top1_AvgScore=('ComputedScore', 'mean'),
            InstancesCovered=('instance', 'nunique')
        )
    )

    # --- Sort descending by average score ---
    llm_top1_summary = llm_top1_summary.sort_values('LLM_Top1_TotalScore', ascending=False)
    return llm_top1_summary, llmTSD_scored

# --- Function to compute Closed Gap for LLM suggestions ---
def compute_closed_gap(llm_top1_scored, scored_df, allowed_solvers: list[str] | set[str] | None = None, sbs_solver: str | None = 'or-tools_cp-sat-free'):
    """Compute Closed Gap for Top-1 LLM suggestions.

    If `allowed_solvers` is provided, SBS/VBS are computed only over that subset.

    If `sbs_solver` is None, SBS is chosen automatically as the single solver with
    the highest total ComputedScore within the (possibly filtered) scored_df.
    """

    if allowed_solvers:
        scored_df = filter_to_solvers(scored_df, allowed_solvers, solver_col='Solver')

    cg_results = []

    # Compute VBS scores for all instances once (always over all instances)
    vbs_df = (
        scored_df.groupby(['Problem', 'Instance'], as_index=False)['ComputedScore']
        .max()
        .rename(columns={'ComputedScore': 'VBS_Score'})
    )
    vbs_total = vbs_df['VBS_Score'].sum()

    # Compute SBS score over all instances (not limited to model coverage)
    if sbs_solver is None:
        totals = scored_df.groupby('Solver', as_index=False)['ComputedScore'].sum()
        if not totals.empty:
            sbs_solver = str(totals.sort_values('ComputedScore', ascending=False).iloc[0]['Solver'])
        else:
            sbs_solver = ''

    sbs_total = scored_df.loc[scored_df['Solver'] == sbs_solver, 'ComputedScore'].sum()

    for (prov, mod), group in llm_top1_scored.groupby(['provider', 'model']):
        # Instances this model actually made predictions for
        covered = group[['problem', 'instance']].drop_duplicates()

        # Compute total scores for this model (only over its covered predictions)
        score_AS = group['ComputedScore'].sum()
        score_SBS = sbs_total
        score_VBS = vbs_total

        # Compute closed gap
        cg = (score_AS - score_SBS) / (score_VBS - score_SBS) if (score_VBS - score_SBS) != 0 else float('nan')

        cg_results.append({
            'provider': prov,
            'model': mod,
            'InstancesCovered': len(covered),
            'AS': score_AS,
            'SBS': score_SBS,
            'VBS': score_VBS,
            'ClosedGap': cg,
        })
    return cg_results

# --- Function to compute score of single solvers and number of optimal answers
def singleSolverScore(scored_df):
    score_solvers = []

    for solver, group in scored_df.groupby('Solver'):
        num_optimal = (group['ComputedScore'] == 1.0).sum()
        total_score = group['ComputedScore'].sum()
        score_solvers.append({
            'Solver': solver,
            'TotalScore': total_score,
            'NumOptimal': num_optimal
        })
    score_solvers_df = pd.DataFrame(score_solvers).sort_values('NumOptimal', ascending=False)
    return score_solvers_df

# --- Plot comparison of LLM variants ---
def plot_llm_variant_comparison(summaries, variant_names, label):
    _require_plotting()
    combined_df = pd.DataFrame()
    for summary, name in zip(summaries, variant_names):
        temp_df = summary.copy()
        temp_df['Variant'] = name
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=combined_df,
        x=label,
        y='model',
        hue='Variant'
    )
    # plt.title('Comparison of LLM Variants Parallel by ' + label)
    plt.xlabel(label)
    plt.ylabel('LLM Model')
    plt.legend(title='Variant')
    plt.tight_layout()
    plt.show()

# --- Function to merge all LLM variants together for direct comparison and renaming each LLM model accordingly ---
def merge_llm_variants(*summaries, variant_labels):
    merged_df = pd.DataFrame()
    for summary, label in zip(summaries, variant_labels):
        temp_df = summary.copy()
        temp_df['Model_Variant'] = temp_df['model'] + ' (' + label + ')'
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
    return merged_df

# --- Function to aggregate total Top-1 score per (problem, instance) across all LLM variants and list ascending ---
def aggregate_llm_instance_scores(*llm_scored_dfs):
    combined = pd.concat(llm_scored_dfs, ignore_index=True)
    # Treat missing merged ComputedScore as 0 (LLM didn't match a scored solver)
    combined['ComputedScore'] = combined['ComputedScore'].fillna(0.0)
    # Ensure keys are strings
    combined['problem'] = combined['problem'].astype(str)
    combined['instance'] = combined['instance'].astype(str)
    # Aggregate
    inst_scores = combined.groupby(['problem', 'instance'], as_index=False).agg(
        TotalLLMScore=('ComputedScore', 'sum'),
        ModelsSeen=('model', 'nunique'),
        PredictionsCount=('Solver', 'count')
    )
    inst_scores_sorted = inst_scores.sort_values('TotalLLMScore', ascending=True).reset_index(drop=True)
    return inst_scores_sorted


def build_llm_performance_table(
    *,
    top3_summary: pd.DataFrame | None = None,
    top1_summary: pd.DataFrame | None = None,
    closed_gap: pd.DataFrame | None = None,
    merge_on: list[str] | None = None,
    include_provider: bool = False,
    fill_missing_scores: float | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Build a single table: Model, Single Score, Parallel Score, Closed Gap.

    Expects inputs from:
    - compute_llm_scores(...)            -> `top3_summary`
    - compute_top1_llm_scores(...)       -> `top1_summary`
    - compute_closed_gap(...) + DataFrame-> `closed_gap`

    By default it merges on ['provider', 'model'] when present; if `Model_Variant`
    exists in any input and `merge_on` is not provided, it merges on ['Model_Variant'].

    Args:
        top3_summary: DataFrame with at least ['provider','model','LLM_TotalScore'].
        top1_summary: DataFrame with at least ['provider','model','LLM_Top1_TotalScore'].
        closed_gap:   DataFrame with at least ['provider','model','ClosedGap'].
        merge_on:     Explicit merge keys.
        include_provider: If True and provider is available, appends it into `Model`
            (e.g., "gpt-4o (openai)"). Provider is never returned as its own column.
        fill_missing_scores: If set, fills missing Total/Top1 scores with this value.
        sort_by:      Column to sort by (e.g. 'Top1TotalScore', 'TotalScore', 'ClosedGap').
        ascending:    Sort order.

    Returns:
        A merged DataFrame with columns (in order): Model, Single Score,
        Parallel Score, Closed Gap.
    """

    def _copy_or_none(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got: {type(df)}")
        return df.copy()

    t3 = _copy_or_none(top3_summary)
    t1 = _copy_or_none(top1_summary)
    cg = _copy_or_none(closed_gap)

    if t3 is None and t1 is None and cg is None:
        return pd.DataFrame(columns=['Model', 'Single Score', 'Parallel Score', 'Closed Gap'])

    # Infer merge keys if not provided
    if merge_on is None:
        any_has_model_variant = any(
            (df is not None and 'Model_Variant' in df.columns) for df in (t3, t1, cg)
        )
        if any_has_model_variant:
            merge_on = ['Model_Variant']
        else:
            merge_on = ['provider', 'model']

    def _prep_top3(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        cols = set(df.columns)
        if 'LLM_TotalScore' in cols:
            out = df[merge_on + ['LLM_TotalScore']].rename(columns={'LLM_TotalScore': 'ParallelScore'})
        elif 'ParallelScore' in cols:
            out = df[merge_on + ['ParallelScore']].copy()
        elif 'TotalScore' in cols:
            # Backward/alternate naming support
            out = df[merge_on + ['TotalScore']].rename(columns={'TotalScore': 'ParallelScore'})
        else:
            raise KeyError(
                "top3_summary must contain 'LLM_TotalScore' (or already-renamed 'ParallelScore'/'TotalScore')."
            )
        return out

    def _prep_top1(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        cols = set(df.columns)
        if 'LLM_Top1_TotalScore' in cols:
            out = df[merge_on + ['LLM_Top1_TotalScore']].rename(
                columns={'LLM_Top1_TotalScore': 'SingleScore'}
            )
        elif 'SingleScore' in cols:
            out = df[merge_on + ['SingleScore']].copy()
        elif 'Top1TotalScore' in cols:
            # Backward/alternate naming support
            out = df[merge_on + ['Top1TotalScore']].rename(columns={'Top1TotalScore': 'SingleScore'})
        else:
            raise KeyError(
                "top1_summary must contain 'LLM_Top1_TotalScore' (or already-renamed 'SingleScore'/'Top1TotalScore')."
            )
        return out

    def _prep_cg(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        if 'ClosedGap' not in df.columns:
            raise KeyError("closed_gap must contain 'ClosedGap'.")
        return df[merge_on + ['ClosedGap']].copy()

    parts: list[pd.DataFrame] = [p for p in (_prep_top3(t3), _prep_top1(t1), _prep_cg(cg)) if p is not None]

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=merge_on, how='outer')

    # Final presentation: rename `Model_Variant` to `model` if needed.
    if merge_on == ['Model_Variant']:
        merged = merged.rename(columns={'Model_Variant': 'model'})

    # Ensure required columns exist
    for col in ['ParallelScore', 'SingleScore', 'ClosedGap']:
        if col not in merged.columns:
            merged[col] = np.nan

    # Optional filling for missing score columns
    if fill_missing_scores is not None:
        for col in ['ParallelScore', 'SingleScore']:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(fill_missing_scores)

    # Backward-compatible sort key mapping
    if sort_by in ('TotalScore', 'ParallelScore', 'Parallel Score'):
        sort_by = 'ParallelScore'
    elif sort_by in ('Top1TotalScore', 'SingleScore', 'Single Score'):
        sort_by = 'SingleScore'
    elif sort_by in ('ClosedGap', 'Closed Gap'):
        sort_by = 'ClosedGap'

    if sort_by is not None and sort_by in merged.columns:
        merged = merged.sort_values(sort_by, ascending=ascending)

    merged = merged.reset_index(drop=True)

    # Final formatting: exact names + order requested, and no separate provider column.
    if 'model' not in merged.columns:
        merged['model'] = np.nan

    if include_provider and 'provider' in merged.columns:
        merged['Model'] = merged['model'].astype(str) + ' (' + merged['provider'].astype(str) + ')'
    else:
        merged['Model'] = merged['model']

    merged['Single Score'] = merged['SingleScore']
    merged['Parallel Score'] = merged['ParallelScore']
    merged['Closed Gap'] = merged['ClosedGap']

    return merged[['Model', 'Single Score', 'Parallel Score', 'Closed Gap']]

