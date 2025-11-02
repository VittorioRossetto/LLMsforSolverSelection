import json
import time
import re
import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # progress bars

# --- Setup imports ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, repo_root)
from utils import *

# --- Constants ---
NON_TESTABLE_MODELS = [
    'playai-tts', 'playai-tts-arabic', 'whisper-large-v3', 'deepseek-r1-distill-llama-70b',
    'gemini-2.0-flash-lite', 'gemma2-9b-it', 'whisper-large-v3-turbo'
]
IGNORED_MODELS = [
    'meta-llama/llama-prompt-guard-2-22m', 'meta-llama/llama-prompt-guard-2-86m'
]

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Benchmark LLM solver recommendations on MiniZinc problems.")
parser.add_argument('--solver-set', choices=['minizinc', 'all', 'free'], default='minizinc',
                    help="Solver set: 'minizinc' (default), 'all', or 'free'.")
parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented',
                    help="MiniZinc script version to use.")
parser.add_argument('--max-workers-models', type=int, default=4,
                    help="Number of models to query in parallel (default=4).")
parser.add_argument('--max-workers-instances', type=int, default=5,
                    help="Number of concurrent instance queries per model (default=5).")
args = parser.parse_args()

# --- Solver set selection ---
if args.solver_set == 'all':
    solver_list = ALL_SOLVERS
    print("Using ALL_SOLVERS set.")
elif args.solver_set == 'free':
    solver_list = FREE_SOLVERS
    print("Using FREE_SOLVERS set.")
else:
    solver_list = MINIZINC_SOLVERS
    print("Using MINIZINC_SOLVERS set.")

print(f"Script version: {args.script_version}")
print(f"Parallel model workers: {args.max_workers_models}")
print(f"Parallel instance workers: {args.max_workers_instances}")

# --- Load problems ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(repo_root, "mznc2025_probs", "problems_with_descriptions.json")
problems = load_problems(dataset_path)
results = {}

# --- Helpers ---
def handle_api_error(e):
    msg = str(e).lower()
    delay, should_retry, is_503_like = None, False, False
    if any(x in msg for x in ["rate_limit", "tpm", "token", "429"]):
        should_retry, delay = True, 60
    elif "resource_exhausted" in msg:
        should_retry, delay = True, 30
    elif any(x in msg for x in ["unavailable", "overloaded", "503"]):
        should_retry, delay, is_503_like = True, 30, True
    elif any(x in msg for x in ["413", "context_length", "too large"]):
        should_retry, delay = True, 60
    return should_retry, delay, is_503_like


def run_single_query(provider, model_id, prob_key, inst_label, prompt, query_func):
    """Executes one LLM query with retry handling."""
    retry_count = 0
    while True:
        try:
            start = time.time()
            response = query_func(prompt, model_name=model_id)
            duration = time.time() - start
            match = re.search(r"\[([^\]]+)\]", response)
            llm_top3 = [s.strip() for s in match.group(1).split(',')] if match else []
            return (prob_key, inst_label, llm_top3, round(duration, 3), None)
        except Exception as e:
            err_text = str(e)
            should_retry, delay, is_503_like = handle_api_error(e)
            retry_count += 1 if should_retry else 0
            if is_503_like and retry_count > 3:
                return (prob_key, inst_label, None, None, f"503 after 3 retries: {err_text}")
            if "413" in err_text and retry_count > 3:
                return (prob_key, inst_label, None, None, f"413 too large after 3 retries: {err_text}")
            if should_retry:
                time.sleep(delay)
                continue
            return (prob_key, inst_label, None, None, f"Non-retryable: {err_text}")


def find_full_script(script_path):
    """Resolves a script path to absolute path."""
    if script_path.startswith('./'):
        rel_path = script_path[2:]
        candidate = os.path.join(repo_root, rel_path)
        if os.path.exists(candidate):
            return candidate
    if os.path.isabs(script_path) and os.path.exists(script_path):
        return script_path
    candidate = os.path.join(repo_root, script_path)
    if os.path.exists(candidate):
        return candidate
    basename = os.path.basename(script_path)
    search_root = os.path.join(repo_root, 'mznc2025_probs')
    for root, _, files in os.walk(search_root):
        if basename in files:
            return os.path.join(root, basename)
    return None


def process_model(provider, model_id, model_label, query_func):
    """Runs all problems for one model concurrently (with a progress bar)."""
    if model_id in NON_TESTABLE_MODELS or model_id in IGNORED_MODELS:
        return provider, model_id, {}

    model_results = {}
    problem_instance_pairs = []

    # Precompute all problem-instance pairs for progress tracking
    for prob_key, prob in problems.items():
        script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
        full_script_path = find_full_script(script_path)
        instance_files = []

        if full_script_path:
            try:
                with open(full_script_path, 'r') as f:
                    script = f.read()
            except Exception as e:
                script = f"[Error reading {full_script_path}: {e}]"
            script_dir = os.path.dirname(full_script_path)
            for fname in os.listdir(script_dir):
                if fname.lower().endswith(('.dzn', '.json')):
                    instance_files.append(os.path.join(script_dir, fname))
        else:
            script = script_path

        if not instance_files:
            instance_files = [None]

        for inst in instance_files:
            problem_instance_pairs.append((prob_key, inst, script))

    total_tasks = len(problem_instance_pairs)
    if total_tasks == 0:
        return provider, model_id, {}

    with ThreadPoolExecutor(max_workers=args.max_workers_instances) as executor, \
         tqdm(total=total_tasks, desc=f"{provider}/{model_id}", leave=False) as pbar:

        futures = {}
        for prob_key, inst, script in problem_instance_pairs:
            inst_label = os.path.basename(inst) if inst else 'base'
            instance_content = ''
            if inst:
                try:
                    with open(inst, 'r') as f:
                        instance_content = f.read()
                except Exception as e:
                    instance_content = f"[Error reading {inst}: {e}]"

            solver_prompt = get_solver_prompt(solver_list, name_only=True)
            prompt = f"\n\nMiniZinc model:\n{script}\n\n"
            if instance_content:
                prompt += f"MiniZinc data:\n{instance_content}\n\n"
            prompt += solver_prompt

            futures[executor.submit(run_single_query, provider, model_id, prob_key, inst_label, prompt, query_func)] = (prob_key, inst_label)

        for future in as_completed(futures):
            prob_key, inst_label = futures[future]
            prob_key, inst_label, llm_top3, duration, err = future.result()
            pbar.update(1)

            if err:
                continue
            model_results.setdefault(prob_key, {})[inst_label] = {
                "top3": llm_top3,
                "time_seconds": duration
            }

    return provider, model_id, model_results


# --- Run all models in parallel (with global progress bar) ---
all_models = []
for provider, models, query_func in [
    ("gemini", GEMINI_MODELS, query_gemini),
    ("groq", GROQ_MODELS, query_groq),
]:
    for model_id, model_label in models:
        all_models.append((provider, model_id, model_label, query_func))

results = {}
with ThreadPoolExecutor(max_workers=args.max_workers_models) as model_executor, \
     tqdm(total=len(all_models), desc="All Models", position=0) as global_bar:

    futures = [model_executor.submit(process_model, provider, model_id, model_label, query_func)
               for provider, model_id, model_label, query_func in all_models]

    for future in as_completed(futures):
        provider, model_id, model_results = future.result()
        results.setdefault(provider, {})[model_id] = model_results
        global_bar.update(1)

# --- Save results ---
output_file = f"LLMsuggestions_{args.solver_set}_{args.script_version}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Results saved to {output_file}")
