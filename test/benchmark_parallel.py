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
    'meta-llama/llama-prompt-guard-2-22m', 'meta-llama/llama-prompt-guard-2-86m',
]

TOP_MODELS = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'moonshotai/kimi-k2-instruct', 'moonshotai/kimi-k2-instruct-0905', 'openai/gpt-oss-120b']

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Benchmark LLM solver recommendations on MiniZinc problems.")
parser.add_argument('--solver-set', choices=['minizinc', 'all', 'free'], default='free',
                    help="Solver set: 'minizinc', 'all', or 'free' (default).")
parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented',
                    help="MiniZinc script version to use.")
parser.add_argument('--max-workers-models', type=int, default=5,
                    help="Number of models to query in parallel (default=5).")
parser.add_argument('--max-workers-instances', type=int, default=1,
                    help="Number of concurrent instance queries per model (default=1).")
parser.add_argument('--top-only', action='store_true', default=False,
                    help="If set, only query models listed in TOP_MODELS.")
parser.add_argument('--dry-run', action='store_true', default=False,
                    help="If set, print planned models and instance counts and exit without querying LLMs.")
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
if args.top_only:
    print("Filtering to TOP_MODELS only.")

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


def load_model_contexts(grok_json_path=os.path.join(repo_root, 'grok_models.json')):
    """Loads model context window info from grok_models.json if available.
    Returns a dict mapping model_id -> context_window (int tokens).
    """
    contexts = {}
    try:
        if os.path.exists(grok_json_path):
            with open(grok_json_path, 'r') as gf:
                data = json.load(gf)
            for entry in data.get('data', []):
                mid = entry.get('id')
                cw = entry.get('context_window')
                if mid and cw:
                    contexts[mid] = int(cw)
    except Exception:
        pass
    return contexts


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: use characters/4 heuristic.

    This is an approximation to avoid importing provider-specific tokenizers.
    """
    if not text:
        return 0
    # count characters (not bytes) and use 4 chars ~= 1 token heuristic
    return max(1, int(len(text) / 4))


def truncate_script_to_budget(script: str, allowed_tokens: int):
    """Truncate the script to approximately allowed_tokens.

    Returns (new_script, was_truncated).
    """
    if allowed_tokens <= 0:
        return ("% [Model script removed due to token limits]\n", True)

    # quick check
    current_tokens = estimate_tokens(script)
    if current_tokens <= allowed_tokens:
        return script, False
    # still too large: truncate by characters using heuristic (4 chars per token)
    allowed_chars = max(20, allowed_tokens * 4)
    truncated = script[:allowed_chars]
    # ensure we end on a newline and add a clear truncation marker as a comment
    truncated = truncated.rstrip() + "\n% [TRUNCATED: script shortened to fit model context]\n"
    return truncated, True


# Load model context windows once
MODEL_CONTEXTS = load_model_contexts()


def get_allowed_total_tokens_for_model(model_id, model_label, fallback_budget=250000, safety_margin=256):
    """Return allowed total tokens for a model after applying safety margin.

    Uses MODEL_CONTEXTS (loaded from grok_models.json) when available. The
    fallback_budget is tuned lower than before to be conservative for general use.
    Returns (allowed_total_tokens, resolved_model_budget).
    """
    model_budget = MODEL_CONTEXTS.get(model_id, None)
    if model_budget is None and model_label:
        # best-effort match by label substring
        for mid, cw in MODEL_CONTEXTS.items():
            if model_label.lower() in mid.lower():
                model_budget = cw
                break
    if model_budget is None:
        model_budget = int(fallback_budget)
    allowed = max(0, int(model_budget) - int(safety_margin))
    return allowed, int(model_budget)


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
    # Determine model context budget (tokens). Use grok_models.json when available,
    # otherwise fall back to a tuned conservative default. Uses helper for consistency.
    allowed_total_tokens, model_budget = get_allowed_total_tokens_for_model(model_id, model_label)

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

            # Estimate tokens for non-script parts (instance data + solver prompt)
            non_script_text = "\n\n"
            if instance_content:
                non_script_text += f"MiniZinc data:\n{instance_content}\n\n"
            non_script_text += solver_prompt
            non_script_tokens = estimate_tokens(non_script_text)

            # Compute allowed tokens for the script portion and truncate if needed
            allowed_script_tokens = allowed_total_tokens - non_script_tokens
            if allowed_script_tokens < 0:
                allowed_script_tokens = 0

            safe_script, was_truncated = truncate_script_to_budget(script, allowed_script_tokens)
            if was_truncated:
                # add a short marker to the prompt explaining truncation
                trunc_note = "% [Model script truncated to fit model context window]\n"
            else:
                trunc_note = ""

            prompt = f"\n\nMiniZinc model:\n{safe_script}\n\n"
            if instance_content:
                prompt += f"MiniZinc data:\n{instance_content}\n\n"
            prompt += trunc_note + solver_prompt

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
        # If top-only requested, skip models not in TOP_MODELS
        if args.top_only and model_id not in TOP_MODELS:
            continue
        all_models.append((provider, model_id, model_label, query_func))

# --- Verbose listing: print which models will be queried ---
print(f"\nPlanned models to query (count={len(all_models)}):")
for prov, mid, mlabel, _ in all_models:
    print(f" - {prov}: {mid} ({mlabel})")
if not all_models:
    print("Warning: no models selected. Check --top-only or the model lists in utils.py.")

if args.dry_run:
    # Compute instance filenames per problem (independent of model)
    problem_instances = {}
    total_instances = 0
    for prob_key, prob in problems.items():
        script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
        full_script_path = find_full_script(script_path)
        inst_names = []
        if full_script_path:
            try:
                script_dir = os.path.dirname(full_script_path)
                for fname in sorted(os.listdir(script_dir)):
                    if fname.lower().endswith(('.dzn', '.json')):
                        inst_names.append(fname)
            except Exception:
                inst_names = []
        # if no instance files found, use a placeholder 'base'
        if not inst_names:
            inst_names = ['base']
        problem_instances[prob_key] = inst_names
        total_instances += len(inst_names)

    print(f"\nDry-run summary:")
    print(f" - problems discovered: {len(problem_instances)}")
    print(f" - total instances (per problem instances summed): {total_instances}")
    per_model_tasks = total_instances
    print(f" - tasks per model: {per_model_tasks}")
    print(f" - models selected: {len(all_models)}")
    print(f" - total tasks if executed: {per_model_tasks * len(all_models)}")

    # show a short sample of problems with their instance names
    print('\nSample problem instance names (first 20 problems):')
    for i, (p, names) in enumerate(sorted(problem_instances.items())):
        if i >= 20:
            break
        sample_names = ', '.join(names[:20])
        more = '...' if len(names) > 20 else ''
        print(f"  {p}: {len(names)} instances -> [{sample_names}]{more}")

    print('\nDry-run complete — no LLM queries were made. Remove --dry-run to execute.')
    # --- Dry-run truncation diagnostic: simulate truncation per selected model ---
    print('\nDry-run truncation diagnostic: simulating which tasks would be truncated')
    trunc_examples_to_show = 5
    for prov, mid, mlabel, _ in all_models:
        allowed_total_tokens, model_budget = get_allowed_total_tokens_for_model(mid, mlabel)
        truncated_tasks = []
        # iterate problems and their instance names
        for prob_key, inst_names in sorted(problem_instances.items()):
            # find script content for this problem
            prob = problems.get(prob_key, {})
            script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
            full_script_path = find_full_script(script_path)
            if full_script_path and os.path.exists(full_script_path):
                try:
                    with open(full_script_path, 'r') as sf:
                        script_text = sf.read()
                except Exception:
                    script_text = ''
            else:
                script_text = script_path

            for inst_name in inst_names:
                # read instance content when available
                instance_content = ''
                if inst_name != 'base':
                    inst_path = None
                    if full_script_path:
                        inst_path = os.path.join(os.path.dirname(full_script_path), inst_name)
                    else:
                        # fallback: try to find the file in mznc2025_probs
                        inst_path = None
                    if inst_path and os.path.exists(inst_path):
                        try:
                            with open(inst_path, 'r') as inf:
                                instance_content = inf.read()
                        except Exception:
                            instance_content = ''

                solver_prompt = get_solver_prompt(solver_list, name_only=True)
                non_script_text = "\n\n"
                if instance_content:
                    non_script_text += f"MiniZinc data:\n{instance_content}\n\n"
                non_script_text += solver_prompt
                non_script_tokens = estimate_tokens(non_script_text)
                allowed_script_tokens = allowed_total_tokens - non_script_tokens
                if allowed_script_tokens < 0:
                    allowed_script_tokens = 0

                orig_script_tokens = estimate_tokens(script_text)
                # determine if truncation would occur
                _, was_truncated = truncate_script_to_budget(script_text, allowed_script_tokens)
                if was_truncated:
                    truncated_tasks.append((prob_key, inst_name, orig_script_tokens, non_script_tokens, allowed_script_tokens))
                # small optimization: if truncated tasks grows large, keep scanning but only store first N
                if len(truncated_tasks) > 200:
                    # avoid unbounded memory use on pathological repos
                    break
        if not truncated_tasks:
            print(f" - {prov}:{mid} -> no truncation predicted (model budget={model_budget} tokens)")
        else:
            print(f" - {prov}:{mid} -> {len(truncated_tasks)} tasks would be truncated (model budget={model_budget} tokens). Examples:")
            for ex in truncated_tasks[:trunc_examples_to_show]:
                pk, iname, orig_toks, non_script_toks, allowed_toks = ex
                print(f"    {pk}/{iname}: script_tokens~{orig_toks}, non_script_tokens~{non_script_toks}, allowed_script_tokens~{allowed_toks}")

    print('\nDry-run complete — no LLM queries were made. Remove --dry-run to execute.')
    sys.exit(0)

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
output_file = f"LLMsuggestions_{args.solver_set}_{args.script_version}_{args.top_only}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Results saved to {output_file}")