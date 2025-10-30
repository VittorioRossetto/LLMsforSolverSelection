import json
import time
import re
import os
import argparse
import sys

# Ensure repo root is on sys.path so imports work when running from test/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, repo_root)

from utils import *


# Models to skip automatically
NON_TESTABLE_MODELS = [
    'playai-tts', 'playai-tts-arabic', 'whisper-large-v3', 'deepseek-r1-distill-llama-70b',
    'gemini-2.0-flash-lite', 'gemma2-9b-it', 'whisper-large-v3-turbo'
]
IGNORED_MODELS = [
    'meta-llama/llama-prompt-guard-2-22m', 'meta-llama/llama-prompt-guard-2-86m'
]
VERY_LIMITED_MODELS = ['allam-2-7b']


# --- Argument parsing for solver set selection ---
# Use '--solver-set minizinc' (default) for MiniZinc Challenge solvers
# Use '--solver-set all' for the full solver list

parser = argparse.ArgumentParser(description="Benchmark LLM solver recommendations on MiniZinc problems.")
parser.add_argument('--solver-set', choices=['minizinc', 'all', 'free'], default='minizinc',
                    help="Which solver set to use in the prompt: 'minizinc' (default) or 'all', 'free'.")
parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented',
                    help="Which script version to use: 'uncommented' (default) or 'commented'.")
args = parser.parse_args()


if args.solver_set == 'all':
    solver_list = ALL_SOLVERS
    print("Using ALL_SOLVERS set for prompt.")
elif args.solver_set == 'free':
    solver_list = FREE_SOLVERS
    print("Using FREE_SOLVERS set for prompt.")
else:
    solver_list = MINIZINC_SOLVERS
    print("Using MINIZINC_SOLVERS set for prompt.")

print(f"Using script version: {args.script_version}")

# Repository root (allow running from inside `test/`)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load all problems (use absolute path so script can be run from any cwd)
dataset_path = os.path.join(repo_root, "mznc2025_probs", "problems_with_descriptions.json")
problems = load_problems(dataset_path)

results = {}

def handle_api_error(e):
    """Return (should_retry, delay_seconds, is_503_like) depending on error type."""
    msg = str(e).lower()

    delay = None
    should_retry = False
    is_503_like = False

    # Token-per-minute or rate limit
    if "rate_limit" in msg or "tpm" in msg or "token" in msg or "429" in msg:
        should_retry = True
        m = re.search(r"retry\s*in\s*([0-9.]+)", msg)
        delay = float(m.group(1)) if m else 60
    # Resource exhausted (quota)
    elif "resource_exhausted" in msg:
        should_retry = True
        m = re.search(r"retry\s*in\s*([0-9.]+)", msg)
        delay = float(m.group(1)) if m else 30
    # Service unavailable or overloaded
    elif "unavailable" in msg or "overloaded" in msg or "503" in msg:
        should_retry = True
        is_503_like = True
        delay = 30
    # Request too large (sometimes transient)
    elif "413" in msg or "context_length" in msg:
        should_retry = True
        delay = 60

    return should_retry, delay, is_503_like


for provider, models, query_func in [
    ("gemini", GEMINI_MODELS, query_gemini),
    ("groq", GROQ_MODELS, query_groq),
]:

    for model_id, model_label in models:
        # Skip non-testable and ignored models
        if model_id in NON_TESTABLE_MODELS or model_id in IGNORED_MODELS:
            print(f"Skipping model {model_id} (non-testable or ignored).")
            continue
        print(f"\nTesting provider={provider}, model={model_id}")
        skip_model = False
        results.setdefault(provider, {}).setdefault(model_id, {})

        prob_keys = list(problems.keys())
        i = 0
        while i < len(prob_keys):
            prob_key = prob_keys[i]
            prob = problems[prob_key]

            # Select script version
            if args.script_version == 'commented':
                script_path = prob.get('script_commented', '')
            else:
                script_path = prob.get('script', '')

            # Find the full .mzn script path and gather .dzn instance files in that directory.
            instance_files = []

            def find_full_script(script_path):
                """Return absolute path to the .mzn script if found, else None."""
                # candidate 1: path relative to repo root (handles './mznc2025_probs/...')
                if script_path.startswith('./'):
                    rel_path = script_path[2:]
                    candidate = os.path.join(repo_root, rel_path)
                    if os.path.exists(candidate):
                        return candidate
                # candidate 2: absolute path
                if os.path.isabs(script_path) and os.path.exists(script_path):
                    return script_path
                # candidate 3: path relative to repo root directly
                candidate = os.path.join(repo_root, script_path)
                if os.path.exists(candidate):
                    return candidate
                # candidate 4: search for the basename under mznc2025_probs
                basename = os.path.basename(script_path)
                search_root = os.path.join(repo_root, 'mznc2025_probs')
                for root, dirs, files in os.walk(search_root):
                    if basename in files:
                        return os.path.join(root, basename)
                return None

            full_script_path = find_full_script(script_path)
            if full_script_path:
                try:
                    with open(full_script_path, 'r') as sf:
                        script = sf.read()
                except Exception as e:
                    script = f"[Error reading {full_script_path}: {e}]"

                script_dir = os.path.dirname(full_script_path)
                if os.path.isdir(script_dir):
                    for fname in os.listdir(script_dir):
                        if fname.lower().endswith('.dzn') or fname.lower().endswith('.json'):
                            instance_files.append(os.path.join(script_dir, fname))
            else:
                # fallback: use script_path as-is (may be inline or unreachable)
                script = script_path

            # If no instance files found, run once with no instance (use None marker)
            if not instance_files:
                instance_files = [None]

            # For each instance (or single None), build prompt and query
            for inst in instance_files:
                inst_label = os.path.basename(inst) if inst else 'base'

                # If there is an instance file, read its content and include in the prompt
                instance_content = ''
                if inst:
                    try:
                        with open(inst, 'r') as inf:
                            instance_content = inf.read()
                    except Exception as e:
                        instance_content = f"[Error reading instance {inst}: {e}]"

                # Use get_solver_prompt to build the prompt for the selected solver set
                solver_prompt = get_solver_prompt(solver_list, name_only=True)

                if instance_content:
                    prompt = f"\n\nMiniZinc model:\n{script}\n\nMiniZinc data:\n{instance_content}\n\n{solver_prompt}"
                else:
                    prompt = f"\n\nMiniZinc model:\n{script}\n\n{solver_prompt}"
                
                retry_count = 0
                skip_problem = False
                while True:
                    try:
                        start_time = time.time()
                        response = query_func(prompt, model_name=model_id)
                        duration = time.time() - start_time
                        break  # success → exit retry loop
                    except Exception as e:
                        err_text = str(e)
                        print(f"  ERROR for model {model_id} on problem {prob_key} (instance={inst_label}): {err_text}")

                        should_retry, delay, is_503_like = handle_api_error(e)
                        is_413 = "413" in err_text or "request too large" in err_text.lower() or "context_length" in err_text.lower()
                        retry_count += 1 if should_retry else 0

                        # If 503/unavailable/overloaded, skip model after 3 tries
                        if is_503_like and retry_count > 3:
                            print(f"503 error persisted after 3 retries — skipping model {model_id}.")
                            skip_model = True
                            break
                        # If 413/request too large, skip this problem/instance after 3 tries
                        if is_413 and retry_count > 3:
                            print(f"413 error (request too large) persisted after 3 retries — skipping problem {prob_key} instance {inst_label} for model {model_id}.")
                            skip_problem = True
                            break
                        if should_retry:
                            print(f"  → Retrying {prob_key} (instance={inst_label}) (attempt {retry_count}) after {delay:.1f}s...")
                            time.sleep(delay)
                            continue  # retry same problem/instance
                        else:
                            print(f"  Skipping model {model_id} for provider {provider} due to non-retryable error.")
                            skip_model = True
                            break

                if skip_model:
                    break  # skip rest of problems for this model
                if skip_problem:
                    print(f"  Skipped problem {prob_key} instance {inst_label} for model {model_id} due to persistent 413 error.")
                    continue

                # Parse solver names
                match = re.search(r"\[([^\]]+)\]", response)
                llm_top3 = [s.strip() for s in match.group(1).split(',')] if match else []

                # Store per-instance results including last-response time (seconds)
                results[provider][model_id].setdefault(prob_key, {})[inst_label] = {
                    'top3': llm_top3,
                    'time_seconds': round(duration, 3) if 'duration' in locals() else None
                }
                print(f"  {prob_key} (instance={inst_label}): {llm_top3} (time={results[provider][model_id][prob_key][inst_label]['time_seconds']}s)")

                # Small delay between requests
                time.sleep(1)

            i += 1  # next problem

        if skip_model:
            print(f"Skipping remaining problems for model {model_id} ({provider}) due to persistent errors.")
            continue

# Save results with filename depending on solver set
result_filename = f"LLMsuggestions_{args.solver_set}_{args.script_version}.json"
with open(result_filename, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. Results saved to {result_filename}")