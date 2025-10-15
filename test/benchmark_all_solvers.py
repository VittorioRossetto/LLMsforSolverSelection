import json
import time
import re
import argparse
import sys
sys.path.insert(1, '../')

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
parser.add_argument('--solver-set', choices=['minizinc', 'all'], default='minizinc',
                    help="Which solver set to use in the prompt: 'minizinc' (default) or 'all'.")
parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented',
                    help="Which script version to use: 'uncommented' (default) or 'commented'.")
args = parser.parse_args()


if args.solver_set == 'all':
    solver_list = ALL_SOLVERS
    print("Using ALL_SOLVERS set for prompt.")
else:
    solver_list = MINIZINC_SOLVERS
    print("Using MINIZINC_SOLVERS set for prompt.")

print(f"Using script version: {args.script_version}")

# Load all problems
dataset_path = "../mznc2025_probs/problems_with_descriptions.json"
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

            # Read file if path is given
            if script_path.startswith('./'):
                try:
                    with open(script_path[2:], 'r') as sf:
                        script = sf.read()
                except Exception as e:
                    script = f"[Error reading {script_path}: {e}]"
            else:
                script = script_path

            # Use get_solver_prompt to build the prompt for the selected solver set
            solver_prompt = get_solver_prompt(solver_list, name_only=True)
            prompt = f"\n\nMiniZinc model:\n{script}\n\n{solver_prompt}"


            retry_count = 0
            skip_problem = False
            while True:
                try:
                    response = query_func(prompt, model_name=model_id)
                    break  # success → exit retry loop
                except Exception as e:
                    err_text = str(e)
                    print(f"  ERROR for model {model_id} on problem {prob_key}: {err_text}")

                    should_retry, delay, is_503_like = handle_api_error(e)
                    is_413 = "413" in err_text or "request too large" in err_text.lower() or "context_length" in err_text.lower()
                    retry_count += 1 if should_retry else 0

                    # If 503/unavailable/overloaded, skip model after 3 tries
                    if is_503_like and retry_count > 3:
                        print(f"503 error persisted after 3 retries — skipping model {model_id}.")
                        skip_model = True
                        break
                    # If 413/request too large, skip this problem after 3 tries
                    if is_413 and retry_count > 3:
                        print(f"413 error (request too large) persisted after 3 retries — skipping problem {prob_key} for model {model_id}.")
                        skip_problem = True
                        break
                    if should_retry:
                        print(f"  → Retrying {prob_key} (attempt {retry_count}) after {delay:.1f}s...")
                        time.sleep(delay)
                        continue  # retry same problem
                    else:
                        print(f"  Skipping model {model_id} for provider {provider} due to non-retryable error.")
                        skip_model = True
                        break

            if skip_model:
                break  # skip rest of problems for this model
            if skip_problem:
                print(f"  Skipped problem {prob_key} for model {model_id} due to persistent 413 error.")
                i += 1
                continue

            # Parse solver names
            match = re.search(r"\[([^\]]+)\]", response)
            llm_top3 = [s.strip() for s in match.group(1).split(',')] if match else []

            results[provider][model_id][prob_key] = llm_top3
            print(f"  {prob_key}: {llm_top3}")

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