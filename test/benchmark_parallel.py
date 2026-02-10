import json
import time
import re
import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # progress bars
import logging

# --- Setup imports ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, repo_root)
from utils import *

# --- Logging setup ---
logger = logging.getLogger('benchmark_parallel')
logger.setLevel(logging.INFO)
_log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
_log_file = os.path.join(repo_root, 'benchmark_parallel.log')
fh = logging.FileHandler(_log_file)
fh.setFormatter(_log_formatter)
logger.addHandler(fh)
# avoid printing to console
logger.propagate = False

# --- Constants ---
NON_TESTABLE_MODELS = [
    'playai-tts', 'playai-tts-arabic', 'whisper-large-v3', 'deepseek-r1-distill-llama-70b',
    'gemini-2.0-flash-lite', 'gemma2-9b-it', 'whisper-large-v3-turbo'
]
IGNORED_MODELS = [
    'meta-llama/llama-prompt-guard-2-22m', 'meta-llama/llama-prompt-guard-2-86m',
]

TOP_MODELS = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'moonshotai/kimi-k2-instruct', 'moonshotai/kimi-k2-instruct-0905', 'openai/gpt-oss-120b']

# Model included when using --best-only (kept consistent with benchmark_chat.py)
BEST = 'openai/gpt-oss-120b'

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Benchmark LLM solver recommendations on MiniZinc problems.")
parser.add_argument('--solver-set', choices=['minizinc', 'all', 'free', 'significative'], default='free',
                    help="Solver set: 'minizinc', 'all', 'free' (default), or 'significative'.")
parser.add_argument('--significative-only', action='store_true', default=False,
                    help="Deprecated: use --solver-set significative. If set, restrict solver selection to SIGNIFICATIVE_SOLVERS.")
parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented',
                    help="MiniZinc script version to use.")
parser.add_argument('--max-workers-models', type=int, default=5,
                    help="Number of models to query in parallel (default=5).")
parser.add_argument('--max-workers-instances', type=int, default=1,
                    help="Number of concurrent instance queries per model (default=1).")
parser.add_argument('--top-only', action='store_true', default=False,
                    help="If set, only query models listed in TOP_MODELS.")
parser.add_argument('--best-only', action='store_true', default=False,
                    help="If set, only query the designated BEST model.")
parser.add_argument('--dry-run', action='store_true', default=False,
                    help="If set, print planned models and instance counts and exit without querying LLMs.")
parser.add_argument('--include-problem-desc', action='store_true', default=False,
                    help="If set, include the problem description (from problems_with_descriptions.json) at the start of each prompt.")
parser.add_argument('--use-fzn-parser-outputs', action='store_true', default=False,
                    help="If set, build the prompt from FlatZinc parser summaries instead of sending the MiniZinc model + data.")
parser.add_argument('--fzn-parser-mode', choices=['plain', 'categorized'], default='plain',
                help="When using --use-fzn-parser-outputs, choose which parser summary file to use. "
                    "'plain' uses fzn_parser_outputs.json; 'categorized' uses fzn_parser_outputs_categorized.json.")
parser.add_argument('--fzn-parser-json', type=str, default=None,
                    help="Optional path to the fzn parser outputs JSON. If omitted, the default depends on --fzn-parser-mode under mznc2025_probs/.")
parser.add_argument('--temperature', type=float, default=None,
                    help='Sampling temperature for each question (provider support varies). Default: provider default')
parser.add_argument('--include-solver-desc', action='store_true', default=False,
                    help='If set, include solver descriptions (from a JSON map) in the prompt in addition to solver names.')
parser.add_argument('--solver-desc-file', type=str, default=None,
                    help='Optional path to JSON file with solver descriptions. Defaults to test/data/freeSolversDescription.json')
parser.add_argument('--with-reasoning', action='store_true', default=False,
                    help='If set, ask the model for a bracketed top-3 list followed by a short explanation, and store it in the output JSON.')
parser.add_argument('--dump-prompts', type=str, default=None,
                help="When used with --dry-run, write all prompts that would be sent to a file instead of querying LLMs. "
                    "Use a path, or 'auto' to auto-name under testOutputFree/.")
parser.add_argument('--dump-prompts-format', choices=['jsonl', 'txt'], default='jsonl',
                help="Output format for --dump-prompts (default=jsonl).")
args = parser.parse_args()

# Backwards compatibility: allow --significative-only as an alias for --solver-set significative.
if getattr(args, 'significative_only', False):
    args.solver_set = 'significative'

# --- Solver set selection ---
if args.solver_set == 'significative':
    solver_list = SIGNIFICATIVE_SOLVERS
    print("Using SIGNIFICATIVE_SOLVERS set.")
else:
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
if args.use_fzn_parser_outputs:
    if getattr(args, 'fzn_parser_mode', 'plain') == 'categorized':
        print("Prompt mode: using categorized fzn_parser_outputs summaries (no MiniZinc model/data).")
    else:
        print("Prompt mode: using fzn_parser_outputs summaries (no MiniZinc model/data).")

# --- Load problems ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(repo_root, "mznc2025_probs_sanitized", "problems_with_descriptions.json")
problems = load_problems(dataset_path)
results = {}

# --- Helpers ---
def load_solver_descriptions(path=None):
    """Load a JSON mapping solver_name -> description.

    Prefers an explicit path if provided, otherwise falls back to
    test/data/freeSolversDescription.json.
    """
    alt = os.path.join(os.path.dirname(__file__), 'data', 'freeSolversDescription.json')
    for candidate in [path, alt]:
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, 'r') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except Exception:
                logger.exception(f"Failed to load solver descriptions from {candidate}")
                return None
    return None


def build_solver_description_text(solver_list, solver_desc_map=None) -> str:
    """Build a human-readable list of solvers with descriptions.

    Returns an empty string if solver_desc_map is missing.
    """
    if not solver_desc_map:
        return ""
    lines = [f"- {s}: {solver_desc_map.get(s, '')}" for s in solver_list]
    return "Solvers and descriptions:\n" + "\n".join(lines) + "\n"


SOLVER_DESC_MAP = load_solver_descriptions(args.solver_desc_file) if args.include_solver_desc else None
if args.include_solver_desc and not SOLVER_DESC_MAP:
    print("Warning: --include-solver-desc was set but no solver description map was loaded; continuing without descriptions.")


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
        logger.exception(f"Failed to load model contexts from {grok_json_path}")
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


def truncate_text_to_budget(text: str, allowed_tokens: int, truncation_marker: str = "\n[TRUNCATED]\n"):
    """Truncate arbitrary text to approximately allowed_tokens.

    Returns (new_text, was_truncated).
    """
    if allowed_tokens <= 0:
        return ("[Removed due to token limits]\n", True)
    current_tokens = estimate_tokens(text)
    if current_tokens <= allowed_tokens:
        return text, False
    allowed_chars = max(20, allowed_tokens * 4)
    truncated = text[:allowed_chars]
    truncated = truncated.rstrip() + truncation_marker
    return truncated, True


def load_fzn_parser_outputs(json_path: str) -> dict:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        logger.exception(f"Failed to load fzn parser outputs from {json_path}")
        return {}


def instance_key_from_path(inst_path: str) -> str | None:
    """Return a repo-relative key matching fzn_parser_outputs.json entries.

    The fzn parser outputs are keyed like "black-hole/layout_14.json".
    We normalize absolute instance paths by stripping known dataset roots.
    """
    if not inst_path:
        return None
    candidates = []
    for root in [
        os.path.join(repo_root, 'mznc2025_probs'),
        os.path.join(repo_root, 'mznc2025_probs_sanitized'),
    ]:
        try:
            rel = os.path.relpath(inst_path, root)
            if not rel.startswith('..') and rel != '.':
                candidates.append(rel)
        except Exception:
            pass
    if candidates:
        # Prefer the shortest relative form.
        return sorted(candidates, key=len)[0]
    # fallback: last two segments (problem-dir/filename)
    parts = inst_path.replace('\\', '/').split('/')
    if len(parts) >= 2:
        return '/'.join(parts[-2:])
    return None


FZN_PARSER_JSON_DEFAULT_PLAIN = os.path.join(repo_root, 'mznc2025_probs', 'fzn_parser_outputs.json')
FZN_PARSER_JSON_DEFAULT_CATEGORIZED = os.path.join(repo_root, 'mznc2025_probs', 'fzn_parser_outputs_categorized.json')

if args.use_fzn_parser_outputs:
    if args.fzn_parser_json:
        FZN_PARSER_JSON_PATH = args.fzn_parser_json
    else:
        FZN_PARSER_JSON_PATH = (
            FZN_PARSER_JSON_DEFAULT_CATEGORIZED
            if getattr(args, 'fzn_parser_mode', 'plain') == 'categorized'
            else FZN_PARSER_JSON_DEFAULT_PLAIN
        )
    FZN_PARSER_OUTPUTS = load_fzn_parser_outputs(FZN_PARSER_JSON_PATH)
else:
    FZN_PARSER_JSON_PATH = None
    FZN_PARSER_OUTPUTS = {}


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


def build_suffix_from_args(args) -> str:
    # Keep naming consistent with the main output file suffix.
    _suffix_parts = []
    if args.top_only:
        _suffix_parts.append("top")
    if args.include_problem_desc:
        _suffix_parts.append("desc")
    if args.use_fzn_parser_outputs:
        if getattr(args, 'fzn_parser_mode', 'plain') == 'categorized':
            _suffix_parts.append("fzncat")
        else:
            _suffix_parts.append("fzn")
    if args.include_solver_desc:
        _suffix_parts.append("solverdesc")
    if args.with_reasoning:
        _suffix_parts.append("reason")
    if args.temperature is not None:
        t = f"{float(args.temperature):g}".replace('.', 'p')
        _suffix_parts.append(f"T{t}")
    return ("_" + "_".join(_suffix_parts)) if _suffix_parts else ""


def dump_prompts_for_dry_run(all_models, problems, solver_list, output_path: str, output_format: str = 'jsonl'):
    """Write all prompts that would be sent (per selected model and instance).

    This mirrors the prompt construction logic used in process_model, including:
    - optional problem description
    - optional solver descriptions
    - fzn-parser prompt mode
    - context-window-based truncation
    """
    solver_prompt = get_solver_prompt(solver_list, name_only=True, with_reasoning=args.with_reasoning)
    solver_desc_text = build_solver_description_text(solver_list, SOLVER_DESC_MAP)

    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    except Exception:
        pass

    written = 0
    skipped = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for provider, model_id, model_label, _query_func in all_models:
            if model_id in NON_TESTABLE_MODELS or model_id in IGNORED_MODELS:
                continue

            allowed_total_tokens, model_budget = get_allowed_total_tokens_for_model(model_id, model_label)

            for prob_key, prob in problems.items():
                script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
                full_script_path = find_full_script(script_path)
                instance_files = []

                if full_script_path:
                    try:
                        with open(full_script_path, 'r', encoding='utf-8') as f:
                            script = f.read()
                    except Exception as e:
                        script = f"[Error reading {full_script_path}: {e}]"
                        logger.exception(f"Error reading script file {full_script_path} for problem {prob_key}: {e}")
                    script_dir = os.path.dirname(full_script_path)
                    try:
                        for fname in sorted(os.listdir(script_dir)):
                            if fname.lower().endswith(('.dzn', '.json')):
                                instance_files.append(os.path.join(script_dir, fname))
                    except Exception:
                        instance_files = []
                else:
                    script = script_path

                if not instance_files:
                    instance_files = [None]

                for inst in instance_files:
                    if inst:
                        inst_label = os.path.splitext(os.path.basename(inst))[0]
                    else:
                        inst_label = 'base'

                    instance_content = ''
                    if inst and (not args.use_fzn_parser_outputs):
                        try:
                            with open(inst, 'r', encoding='utf-8') as f:
                                instance_content = f.read()
                        except Exception as e:
                            instance_content = f"[Error reading {inst}: {e}]"
                            logger.exception(f"Error reading instance file {inst} for problem {prob_key}: {e}")

                    if args.use_fzn_parser_outputs:
                        if not inst:
                            skipped += 1
                            continue
                        inst_key = instance_key_from_path(inst)
                        fzn_desc = FZN_PARSER_OUTPUTS.get(inst_key)
                        if not fzn_desc:
                            skipped += 1
                            continue

                        prompt = ""
                        prob_desc = ''
                        if args.include_problem_desc:
                            prob_meta = problems.get(prob_key, {})
                            prob_desc = prob_meta.get('description', '')
                            if prob_desc:
                                prompt += f"Problem description:\n{prob_desc}\n\n"

                        prompt += f"Given this description of a MiniZinc problem:\n{str(fzn_desc).strip()}\n\n"
                        if solver_desc_text:
                            prompt += solver_desc_text + "\n"
                        prompt += solver_prompt

                        # Truncate only the fzn summary chunk if needed.
                        prompt_tokens = estimate_tokens(prompt)
                        was_truncated = False
                        if prompt_tokens > allowed_total_tokens:
                            fixed_tail = (solver_desc_text + "\n" if solver_desc_text else "") + solver_prompt
                            fixed_tokens = estimate_tokens(fixed_tail)
                            prefix = ""
                            if args.include_problem_desc and prob_desc:
                                prefix = f"Problem description:\n{prob_desc}\n\n"
                            prefix += "Given this description of a MiniZinc problem:\n"
                            prefix_tokens = estimate_tokens(prefix)
                            allowed_desc_tokens = allowed_total_tokens - fixed_tokens - prefix_tokens
                            if allowed_desc_tokens < 0:
                                allowed_desc_tokens = 0
                            safe_desc, was_truncated = truncate_text_to_budget(
                                str(fzn_desc).strip(),
                                allowed_desc_tokens,
                                truncation_marker="\n[TRUNCATED: fzn summary shortened to fit model context]\n",
                            )
                            prompt = prefix + safe_desc + "\n\n" + fixed_tail
                    else:
                        non_script_text = "\n\n"
                        prob_desc = ''
                        if args.include_problem_desc:
                            prob_meta = problems.get(prob_key, {})
                            prob_desc = prob_meta.get('description', '')
                            if prob_desc:
                                non_script_text += f"Problem description:\n{prob_desc}\n\n"
                        if instance_content:
                            non_script_text += f"MiniZinc data:\n{instance_content}\n\n"
                        if solver_desc_text:
                            non_script_text += solver_desc_text + "\n"
                        non_script_text += solver_prompt
                        non_script_tokens = estimate_tokens(non_script_text)

                        allowed_script_tokens = allowed_total_tokens - non_script_tokens
                        if allowed_script_tokens < 0:
                            allowed_script_tokens = 0

                        safe_script, was_truncated = truncate_script_to_budget(script, allowed_script_tokens)
                        trunc_note = "% [Model script truncated to fit model context window]\n" if was_truncated else ""

                        prompt = f"\n\nMiniZinc model:\n{safe_script}\n\n"
                        if args.include_problem_desc and prob_desc:
                            prompt = f"Problem description:\n{prob_desc}\n\n" + prompt
                        if instance_content:
                            prompt += f"MiniZinc data:\n{instance_content}\n\n"
                        if solver_desc_text:
                            prompt += solver_desc_text + "\n"
                        prompt += trunc_note + solver_prompt

                    record = {
                        "provider": provider,
                        "model_id": model_id,
                        "model_label": model_label,
                        "model_context_window": model_budget,
                        "allowed_total_tokens": allowed_total_tokens,
                        "problem": prob_key,
                        "instance": inst_label,
                        "use_fzn_parser_outputs": bool(args.use_fzn_parser_outputs),
                        "prompt_tokens_est": estimate_tokens(prompt),
                        "prompt": prompt,
                    }
                    if output_format == 'jsonl':
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        header = (
                            f"===== {provider}/{model_id} | {prob_key}/{inst_label} | "
                            f"tokens~{record['prompt_tokens_est']} allowed~{allowed_total_tokens} =====\n"
                        )
                        out.write(header)
                        out.write(prompt)
                        if not prompt.endswith("\n"):
                            out.write("\n")
                        out.write("\n")
                    written += 1

    print(f"\nPrompt dump complete: wrote {written} prompts to {output_path} (skipped {skipped} tasks).")


def run_single_query(provider, model_id, prob_key, inst_label, prompt, query_func, temperature=None):
    """Executes one LLM query with retry handling.

    Returns (prob_key, inst_label, top3, duration, raw_response, reasoning, error).
    """
    retry_count = 0
    while True:
        try:
            start = time.time()
            if temperature is not None:
                try:
                    response = query_func(prompt, model_name=model_id, temperature=temperature)
                except TypeError:
                    response = query_func(prompt, model_name=model_id)
            else:
                response = query_func(prompt, model_name=model_id)
            duration = time.time() - start
            match = re.search(r"\[([^\]]+)\]", response)
            llm_top3 = [s.strip() for s in match.group(1).split(',')] if match else []
            reasoning = (response[match.end():].strip() if match else response.strip())
            return (prob_key, inst_label, llm_top3, round(duration, 3), response, reasoning, None)
        except Exception as e:
            err_text = str(e)
            # log the exception with context
            try:
                logger.exception(f"Error during query: provider={provider} model={model_id} problem={prob_key} instance={inst_label} error={err_text}")
            except Exception:
                # best-effort logging; do not break on logger issues
                pass
            should_retry, delay, is_503_like = handle_api_error(e)
            retry_count += 1 if should_retry else 0
            if is_503_like and retry_count > 3:
                return (prob_key, inst_label, None, None, None, None, f"503 after 3 retries: {err_text}")
            if "413" in err_text and retry_count > 3:
                return (prob_key, inst_label, None, None, None, None, f"413 too large after 3 retries: {err_text}")
            if should_retry:
                time.sleep(delay)
                continue
            return (prob_key, inst_label, None, None, None, None, f"Non-retryable: {err_text}")


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
    search_root = os.path.join(repo_root, 'mznc2025_probs_sanitized')
    for root, _, files in os.walk(search_root):
        if basename in files:
            return os.path.join(root, basename)
    return None


def process_model(provider, model_id, model_label, query_func):
    """Runs all problems for one model concurrently (with a progress bar)."""
    if model_id in NON_TESTABLE_MODELS or model_id in IGNORED_MODELS:
        return provider, model_id, {}

    solver_prompt = get_solver_prompt(solver_list, name_only=True, with_reasoning=args.with_reasoning)
    solver_desc_text = build_solver_description_text(solver_list, SOLVER_DESC_MAP)

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
                logger.exception(f"Error reading script file {full_script_path} for problem {prob_key}: {e}")
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
            if inst:
                inst_label = os.path.splitext(os.path.basename(inst))[0]
            else:
                inst_label = 'base'
            instance_content = ''
            if inst and (not args.use_fzn_parser_outputs):
                try:
                    with open(inst, 'r') as f:
                        instance_content = f.read()
                except Exception as e:
                    instance_content = f"[Error reading {inst}: {e}]"
                    logger.exception(f"Error reading instance file {inst} for problem {prob_key}: {e}")

            if args.use_fzn_parser_outputs:
                if not inst:
                    # No instance file => no fzn parser entry to look up
                    logger.warning(f"Skipping fzn-parser prompt for {prob_key}/{inst_label}: no instance file")
                    pbar.update(1)
                    continue
                inst_key = instance_key_from_path(inst)
                fzn_desc = FZN_PARSER_OUTPUTS.get(inst_key)
                if not fzn_desc:
                    logger.warning(f"Missing fzn parser output for instance key={inst_key} (path={inst})")
                    pbar.update(1)
                    continue

                # Build prompt: (optional problem desc) + fzn summary + (optional solver descriptions) + solver prompt.
                prompt = ""
                if args.include_problem_desc:
                    prob_meta = problems.get(prob_key, {})
                    prob_desc = prob_meta.get('description', '')
                    if prob_desc:
                        prompt += f"Problem description:\n{prob_desc}\n\n"

                prompt += f"Given this description of a MiniZinc problem:\n{str(fzn_desc).strip()}\n\n"
                if solver_desc_text:
                    prompt += solver_desc_text + "\n"
                prompt += solver_prompt

                # Optional safety: truncate the fzn description if the prompt is too big.
                prompt_tokens = estimate_tokens(prompt)
                if prompt_tokens > allowed_total_tokens:
                    # Keep solver prompt + solver descriptions intact; truncate only the fzn description chunk.
                    fixed_tail = (solver_desc_text + "\n" if solver_desc_text else "") + solver_prompt
                    fixed_tokens = estimate_tokens(fixed_tail)
                    prefix = ""
                    if args.include_problem_desc and prob_desc:
                        prefix = f"Problem description:\n{prob_desc}\n\n"
                    prefix += "Given this description of a MiniZinc problem:\n"
                    prefix_tokens = estimate_tokens(prefix)
                    allowed_desc_tokens = allowed_total_tokens - fixed_tokens - prefix_tokens
                    if allowed_desc_tokens < 0:
                        allowed_desc_tokens = 0
                    safe_desc, _ = truncate_text_to_budget(str(fzn_desc).strip(), allowed_desc_tokens, truncation_marker="\n[TRUNCATED: fzn summary shortened to fit model context]\n")
                    prompt = prefix + safe_desc + "\n\n" + fixed_tail

            else:
                # Estimate tokens for non-script parts (problem description + instance data + solver prompt)
                non_script_text = "\n\n"
                if args.include_problem_desc:
                    prob_meta = problems.get(prob_key, {})
                    prob_desc = prob_meta.get('description', '')
                    if prob_desc:
                        non_script_text += f"Problem description:\n{prob_desc}\n\n"
                if instance_content:
                    non_script_text += f"MiniZinc data:\n{instance_content}\n\n"
                if solver_desc_text:
                    non_script_text += solver_desc_text + "\n"
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
                # Optionally include the problem description (short) at the top of the prompt
                if args.include_problem_desc:
                    prob_meta = problems.get(prob_key, {})
                    prob_desc = prob_meta.get('description', '')
                    if prob_desc:
                        prompt = f"Problem description:\n{prob_desc}\n\n" + prompt
                if instance_content:
                    prompt += f"MiniZinc data:\n{instance_content}\n\n"
                if solver_desc_text:
                    prompt += solver_desc_text + "\n"
                prompt += trunc_note + solver_prompt

            futures[executor.submit(run_single_query, provider, model_id, prob_key, inst_label, prompt, query_func, args.temperature)] = (prob_key, inst_label)

        for future in as_completed(futures):
            prob_key, inst_label = futures[future]
            prob_key, inst_label, llm_top3, duration, raw_response, reasoning, err = future.result()
            pbar.update(1)

            if err:
                continue
            entry = {
                "top3": llm_top3,
                "time_seconds": duration,
            }
            if args.with_reasoning:
                entry["reasoning"] = reasoning
                entry["raw_response"] = raw_response
            model_results.setdefault(prob_key, {})[inst_label] = entry

    return provider, model_id, model_results


# --- Run all models in parallel (with global progress bar) ---
all_models = []
for provider, models, query_func in [
    ("gemini", GEMINI_MODELS, query_gemini),
    ("groq", GROQ_MODELS, query_groq),
]:
    for model_id, model_label in models:
        # If best-only requested, skip models not matching BEST
        if args.best_only and model_id != BEST:
            continue
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
    if args.dump_prompts:
        suffix = build_suffix_from_args(args)
        if args.dump_prompts.strip().lower() == 'auto':
            ext = 'jsonl' if args.dump_prompts_format == 'jsonl' else 'txt'
            dump_path = f"testOutputFree/prompts_{args.solver_set}_{args.script_version}{suffix}.{ext}"
        else:
            dump_path = args.dump_prompts
        dump_prompts_for_dry_run(all_models, problems, solver_list, dump_path, args.dump_prompts_format)
        sys.exit(0)

    # --- Dry-run truncation diagnostic: simulate truncation per selected model ---
    print('\nDry-run truncation diagnostic: simulating which tasks would be truncated')
    solver_desc_text = build_solver_description_text(solver_list, SOLVER_DESC_MAP)
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
                inst_path = None
                if inst_name != 'base':
                    if full_script_path:
                        inst_path = os.path.join(os.path.dirname(full_script_path), inst_name)
                    if inst_path and os.path.exists(inst_path):
                        try:
                            with open(inst_path, 'r') as inf:
                                instance_content = inf.read()
                        except Exception:
                            instance_content = ''

                solver_prompt = get_solver_prompt(solver_list, name_only=True)
                if args.use_fzn_parser_outputs:
                    if inst_name == 'base':
                        continue
                    if not inst_path:
                        continue
                    inst_key = instance_key_from_path(inst_path)
                    fzn_desc = FZN_PARSER_OUTPUTS.get(inst_key)
                    if not fzn_desc:
                        continue
                    prompt_text = f"Given this description of a MiniZinc problem:\n{str(fzn_desc).strip()}\n\n"
                    if solver_desc_text:
                        prompt_text += solver_desc_text + "\n"
                    prompt_text += solver_prompt
                    prompt_tokens = estimate_tokens(prompt_text)
                    if prompt_tokens > allowed_total_tokens:
                        truncated_tasks.append((prob_key, inst_name, prompt_tokens, allowed_total_tokens, inst_key))
                else:
                    non_script_text = "\n\n"
                    if instance_content:
                        non_script_text += f"MiniZinc data:\n{instance_content}\n\n"
                    if solver_desc_text:
                        non_script_text += solver_desc_text + "\n"
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
                if args.use_fzn_parser_outputs:
                    pk, iname, prompt_toks, allowed_toks, inst_key = ex
                    print(f"    {pk}/{iname}: prompt_tokens~{prompt_toks}, allowed_total_tokens~{allowed_toks}, key={inst_key}")
                else:
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
# build output filename with optional flags
_suffix_parts = []
if args.top_only:
    _suffix_parts.append("top")
if args.include_problem_desc:
    _suffix_parts.append("desc")
if args.use_fzn_parser_outputs:
    if getattr(args, 'fzn_parser_mode', 'plain') == 'categorized':
        _suffix_parts.append("fzncat")
    else:
        _suffix_parts.append("fzn")
if args.include_solver_desc:
    _suffix_parts.append("solverdesc")
if args.with_reasoning:
    _suffix_parts.append("reason")
if args.temperature is not None:
    # 0.2 -> 0p2, 0.0 -> 0, 0.80 -> 0p8
    t = f"{float(args.temperature):g}".replace('.', 'p')
    _suffix_parts.append(f"T{t}")
_suffix = ("_" + "_".join(_suffix_parts)) if _suffix_parts else ""

# Name output according to the effective solver set.
output_file = f"testOutputFree/LLMsuggestions_{args.solver_set}_{args.script_version}{_suffix}.json"

# Ensure output directory exists (benchmark_chat does this; keep consistent)
try:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
except Exception:
    pass

with open(output_file, "w") as f:
    try:
        json.dump(results, f, indent=2)
    except Exception as e:
        logger.exception(f"Failed to write results file {output_file}: {e}")
        raise

print(f"\nDone! Results saved to {output_file}")