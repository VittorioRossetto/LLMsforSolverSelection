import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import numpy as np
import os

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

    agg = valid_objs.groupby(['Problem', 'Instance']).apply(best_worst).reset_index()

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

# --- Compute scores of LLM answers based on single solvers score ---
def compute_llm_scores(llm_df, scored_df):
    llm_expanded = llm_df.copy()

    llm_expanded = llm_expanded[llm_expanded['top3_list'].notnull()].copy()
    llm_expanded = llm_expanded.explode('top3_list').rename(columns={'top3_list': 'Solver'})
    llm_expanded['Solver'] = llm_expanded['Solver'].astype(str).str.strip()

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
def compute_top1_llm_scores(llm_df, scored_df):
    # --- Extract Top-1 predictions only ---
    llm_top1 = llm_df.copy()
    llm_top1 = llm_top1[llm_top1['top1'].notnull()].copy()
    llm_top1 = llm_top1.rename(columns={'top1': 'Solver'})
    llm_top1['Solver'] = llm_top1['Solver'].astype(str)

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
def compute_closed_gap(llm_top1_scored, scored_df):
    SBS_SOLVER = 'or-tools_cp-sat-free'

    cg_results = []

    # Compute VBS scores for all instances once (always over all instances)
    vbs_df = (
        scored_df.groupby(['Problem', 'Instance'], as_index=False)['ComputedScore']
        .max()
        .rename(columns={'ComputedScore': 'VBS_Score'})
    )
    vbs_total = vbs_df['VBS_Score'].sum()

    # Compute SBS score over all instances (not limited to model coverage)
    sbs_total = scored_df.loc[scored_df['Solver'] == SBS_SOLVER, 'ComputedScore'].sum()

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