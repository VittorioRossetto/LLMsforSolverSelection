import json
import time
import re
from utils import *

# Load all problems
dataset_path = "mznc2025_probs/problems_with_descriptions.json"
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
        print(f"\nTesting provider={provider}, model={model_id}")
        skip_model = False
        results.setdefault(provider, {}).setdefault(model_id, {})

        prob_keys = list(problems.keys())
        i = 0
        while i < len(prob_keys):
            prob_key = prob_keys[i]
            prob = problems[prob_key]
            # description = prob.get('description', '')
            script = prob.get('script', '')

            # Read file if path is given
            if script.startswith('./'):
                try:
                    with open(script[2:], 'r') as sf:
                        script = sf.read()
                except Exception as e:
                    script = f"[Error reading {script}: {e}]"

            prompt = f"\n\nMiniZinc model:\n{script}\n\n{SOLVER_PROMPT_NAME_ONLY}"

            retry_count = 0
            while True:
                try:
                    response = query_func(prompt, model_name=model_id)
                    break  # success → exit retry loop
                except Exception as e:
                    err_text = str(e)
                    print(f"  ERROR for model {model_id} on problem {prob_key}: {err_text}")

                    should_retry, delay, is_503_like = handle_api_error(e)
                    if should_retry:
                        retry_count += 1
                        if is_503_like and retry_count > 3:
                            print(f"503 error persisted after 3 retries — skipping model {model_id}.")
                            skip_model = True
                            break
                        print(f"  → Retrying {prob_key} (attempt {retry_count}) after {delay:.1f}s...")
                        time.sleep(delay)
                        continue  # retry same problem
                    else:
                        print(f"  Skipping model {model_id} for provider {provider} due to non-retryable error.")
                        skip_model = True
                        break

            if skip_model:
                break  # skip rest of problems for this model

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

# Save results
with open("solver_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to solver_benchmark_results.json")
