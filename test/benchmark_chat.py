"""
Benchmark script that packs multiple instance prompts into chat-style conversations.

This script follows the existing benchmarking logic but groups multiple instance
prompts into a single chat conversation that starts with a solver-description
message. When the estimated token budget for a model is reached, a new chat is
opened (i.e. a new conversation is sent) and the solver description message is
sent again.

The script is conservative: it will attempt to call the provider's query function
with a `messages=` parameter (chat-style). If the provider wrapper does not
support chat messages, it will fall back to joining the messages into a single
string prompt and calling the non-chat API.
"""

import os
import sys
import json
import time
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from google.genai import types
from google import genai

# make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, repo_root)
from utils import *

# --- Logging setup ---
logger = logging.getLogger('benchmark_chat')
logger.setLevel(logging.INFO)
_log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
_log_file = os.path.join(repo_root, 'benchmark_chat.log')
try:
    fh = logging.FileHandler(_log_file)
    fh.setFormatter(_log_formatter)
    logger.addHandler(fh)
    logger.propagate = False
except Exception:
    # best-effort: if logging can't be configured, don't crash
    pass


# Models to include when --top-only is used (kept in sync with benchmark_parallel.py)
TOP_MODELS = [
    'gemini-2.5-flash-lite', 'gemini-2.5-flash',
    'moonshotai/kimi-k2-instruct', 'moonshotai/kimi-k2-instruct-0905',
    'openai/gpt-oss-120b'
]

# Model included when using --best-only
BEST = 'openai/gpt-oss-120b'


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


# --- load problems (sanitized dataset) ---
dataset_path = os.path.join(repo_root, "mznc2025_probs_sanitized", "problems_with_descriptions.json")
problems = load_problems(dataset_path)

# optional cache for mzn2feat features (loaded by main when requested)
MZN2FEAT = {}


def load_solver_descriptions(path=None):
    # try a local test JSON first, then fall back to a names-only prompt
    alt = os.path.join(os.path.dirname(__file__), 'data', 'freeSolversDescription.json')
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    if os.path.exists(alt):
        try:
            with open(alt, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def build_solver_description_text(solver_list, solver_desc_map=None):
    if solver_desc_map:
        lines = [f"- {s}: {solver_desc_map.get(s, '')}" for s in solver_list]
        return "Solvers and descriptions:\n" + "\n".join(lines) + "\n"
    return get_solver_prompt(solver_list, name_only=False)


def pack_instances_into_chats(solver_desc_text, common_instruction, script_text, instances, allowed_total_tokens, prob_description=''):
    """Greedy pack instances into chat conversations given token budget.

    instances: list of tuples (inst_label, instance_content)
    returns: list of chats, where each chat is list of messages (dicts with role/content)
    """
    chats = []
    # include the model script once per chat in the system message; account for its tokens
    # also account for the optional problem description when computing budget
    base_non_script = (
        estimate_tokens(solver_desc_text)
        + estimate_tokens(common_instruction)
        + estimate_tokens(script_text)
        + estimate_tokens(prob_description)
    )
    cur = []
    cur_tokens = base_non_script
    for inst_label, content in instances:
        inst_msg = f"Instance: {inst_label}\nMiniZinc data:\n{content}\n\n{common_instruction}"
        inst_tokens = estimate_tokens(inst_msg)
        # if adding would exceed allowed_total_tokens, flush
        if cur and (cur_tokens + inst_tokens > allowed_total_tokens):
            # build messages for cur
            # keep solver descriptions and the model as separate system messages when present
            msgs = []
            if solver_desc_text:
                msgs.append({'role': 'system', 'content': solver_desc_text})
            # include problem description as its own system message when provided
            if prob_description:
                msgs.append({'role': 'system', 'content': "Problem description:\n" + prob_description + "\n"})
            if script_text:
                model_msg = {
                    'role': 'system',
                    'content': "MiniZinc model:\n" + script_text + "\n"
                }
                msgs.append(model_msg)
            for c_label, c_content in cur:
                msgs.append({'role': 'user', 'content': f"Instance: {c_label}\nMiniZinc data:\n{c_content}\n\n{common_instruction}"})
            chats.append(msgs)
            cur = [(inst_label, content)]
            cur_tokens = base_non_script + inst_tokens
        else:
            cur.append((inst_label, content))
            cur_tokens += inst_tokens

    if cur:
        msgs = []
        if solver_desc_text:
            msgs.append({'role': 'system', 'content': solver_desc_text})
        if prob_description:
            msgs.append({'role': 'system', 'content': "Problem description:\n" + prob_description + "\n"})
        if script_text:
            msgs.append({'role': 'system', 'content': "MiniZinc model:\n" + script_text + "\n"})
        for c_label, c_content in cur:
            msgs.append({'role': 'user', 'content': f"Instance: {c_label}\nMiniZinc data:\n{c_content}\n\n{common_instruction}"})
        chats.append(msgs)

    return chats


def send_chat(messages, provider, model_id, query_func):
    """Try to send chat messages with TPM/size-aware retries.

    On TPM/TPH or context/size errors, attempt to truncate the MiniZinc model
    system message progressively and retry a small number of times. If the
    provider wrapper doesn't support chat messages (raises TypeError), fall
    back to joining the messages into a single prompt.
    """
    def handle_api_error(exc):
        msg = str(exc).lower()
        is_tpm = any(x in msg for x in ["tpm", "tph", "tokens per minute", "tokens-per-minute", "tokens/min"])
        is_rate = any(x in msg for x in ["rate_limit", "rate limit", "429", "too many requests"]) or is_tpm
        is_size = any(x in msg for x in ["413", "context_length", "too large", "request too large"]) or "context" in msg and "length" in msg
        is_503_like = any(x in msg for x in ["unavailable", "overloaded", "503"]) or "resource_exhausted" in msg
        return is_rate, is_tpm, is_size, is_503_like

    # Work on a shallow copy of messages so we can edit model text if needed
    curr_messages = [dict(m) for m in messages]
    trunc_attempt = 0
    backoff_base = 2
    while True:
        try:
            # If provider is Gemini, call the GenAI client with system_instruction/config
            if provider and provider.lower().startswith('gemini'):
                # build system instruction from system messages and contents from user messages
                system_parts = [m.get('content','') for m in curr_messages if m.get('role') == 'system']
                system_instruction = '\n'.join(system_parts)
                user_parts = [m.get('content','') for m in curr_messages if m.get('role') == 'user']
                contents = '\n----\n'.join(user_parts)
                client = genai.Client()
                cfg = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
                if cfg:
                    response = client.models.generate_content(model=model_id, config=cfg, contents=contents)
                else:
                    response = client.models.generate_content(model=model_id, contents=contents)
                return getattr(response, 'text', str(response))

            # prefer chat/multi-message API otherwise
            try:
                return query_func(messages=curr_messages, model_name=model_id)
            except TypeError:
                # provider wrapper probably expects plain prompt; call fallback
                joined = "\n----\n".join(m['content'] for m in curr_messages)
                return query_func(joined, model_name=model_id)
        except Exception as e:
            is_rate, is_tpm, is_size, is_503_like = handle_api_error(e)
            logger.exception(f"send_chat error provider={provider} model={model_id}: {e}")
            # Log explicit note for size/context errors to help quick detection
            if is_size:
                try:
                    total_tokens = sum(estimate_tokens(m.get('content', '')) for m in curr_messages)
                except Exception:
                    total_tokens = None
                logger.warning(f"SIZE/CONTEXT error detected for provider={provider} model={model_id}: estimated_request_tokens={total_tokens} error={e}")

            # On size/context (413) errors, only split when the provider explicitly
            # reports 'Request too large for model'. Other size/context errors will
            # use the fixed sleep-and-retry behavior.
            msg = str(e).lower()
            if 'request too large for model' in msg:
                try:
                    # try splitting user messages into two chats
                    msgs1, msgs2 = _split_messages_in_two(curr_messages)
                except Exception as split_exc:
                    logger.exception(f"Failed to split messages for provider={provider} model={model_id}: {split_exc}")
                    logger.info(f"Falling back to fixed sleep after size error for provider={provider} model={model_id}")
                    time.sleep(60)
                    continue

                logger.info(f"Splitting oversized request into 2 chats for provider={provider} model={model_id}")
                # send first half then second half, concatenate results
                resp1 = send_chat(msgs1, provider, model_id, query_func)
                resp2 = send_chat(msgs2, provider, model_id, query_func)
                return (resp1 or '') + "\n" + (resp2 or '')
            elif '413' in msg or is_size:
                logger.info(f"Received size/context error (413) for provider={provider} model={model_id}; sleeping 60s before retry")
                time.sleep(60)
                continue
            if '429' in msg or 'rate limit' in msg or 'too many requests' in msg:
                logger.info(f"Received rate-limit/429 for provider={provider} model={model_id}; sleeping 60s before retry")
                time.sleep(60)
                continue

            # For 503-like errors, keep retrying with exponential backoff (capped at 60s)
            if is_503_like:
                delay = min(60, backoff_base ** min(6, trunc_attempt + 1))
                logger.info(f"503-like error for provider={provider} model={model_id}; backing off for {delay}s")
                time.sleep(delay)
                continue

            # otherwise re-raise the exception to be handled by caller
            raise


def _parse_retry_seconds(msg: str):
    """Try to parse human-readable retry durations like '24m8.928s' into seconds."""
    if not msg:
        return None
    total = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)(h|m|s)", msg, flags=re.I):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'h':
            total += val * 3600.0
        elif unit == 'm':
            total += val * 60.0
        else:
            total += val
    return int(total) if total > 0 else None


def _extract_retry_phrase(msg: str):
    """Return the substring that likely contains a retry duration, e.g. the
    text following 'try again in' or similar phrases. If none found, returns None.
    This prevents picking up unrelated numeric+unit fragments from the whole
    exception text (which previously caused very large waits).
    """
    if not msg:
        return None
    # look for common patterns like 'please try again in 24m8.928s' or 'try again in 9h'
    m = re.search(r"(?:please\s+)?try again in\s+([0-9hms\.\s]+)", msg, flags=re.I)
    if m:
        return m.group(1)
    # also handle patterns like 'please try again after 24m'
    m = re.search(r"(?:please\s+)?try again after\s+([0-9hms\.\s]+)", msg, flags=re.I)
    if m:
        return m.group(1)
    return None


def _split_messages_in_two(messages):
    """Split a chat messages list into two chats.

    - Keeps all `system` messages in both chats.
    - Splits `user` messages into two groups (roughly half each).
    - If there is only one user message, split its content by lines into two.
    """
    system_msgs = [m for m in messages if m.get('role') == 'system']
    user_msgs = [m for m in messages if m.get('role') == 'user']

    if not user_msgs:
        # nothing to split
        return messages, messages

    if len(user_msgs) == 1:
        # split the single user's content approximately in half by lines
        content = user_msgs[0].get('content', '')
        lines = content.splitlines(keepends=True)
        if len(lines) <= 1:
            # fallback: split by characters
            mid = len(content) // 2
            c1 = content[:mid]
            c2 = content[mid:]
        else:
            mid = len(lines) // 2
            c1 = ''.join(lines[:mid])
            c2 = ''.join(lines[mid:])
        um1 = {'role': 'user', 'content': c1}
        um2 = {'role': 'user', 'content': c2}
        msgs1 = system_msgs + [um1]
        msgs2 = system_msgs + [um2]
        return msgs1, msgs2

    mid = len(user_msgs) // 2
    msgs1 = system_msgs + user_msgs[:mid]
    msgs2 = system_msgs + user_msgs[mid:]
    return msgs1, msgs2


def resilient_send_chat(messages, provider, model_id, query_func):
    """Call `send_chat` but never give up: retries indefinitely until a response is returned.

    This wrapper will parse provider error messages for suggested wait times when
    available (e.g. 'Please try again in 24m8.928s') and otherwise use an
    exponential backoff capped to 1 hour between attempts.
    """
    attempt = 0
    while True:
        try:
            return send_chat(messages, provider, model_id, query_func)
        except Exception as e:
            attempt += 1
            msg = str(e)
            logger.exception(f"resilient_send_chat caught exception provider={provider} model={model_id}: {e}")
            # try to respect provider-suggested retry duration, but only if the
            # message explicitly contains a "try again in/after" suggestion.
            retry_phrase = _extract_retry_phrase(msg)
            parsed = _parse_retry_seconds(retry_phrase) if retry_phrase else None
            if parsed:
                delay = parsed
            else:
                # exponential backoff in seconds (2^attempt), min 5s, max 3600s
                delay = min(3600, max(5, 2 ** min(10, attempt)))
            logger.info(f"Retrying chat send for provider={provider} model={model_id} after {delay}s (attempt {attempt})")
            time.sleep(delay)
            continue


def process_model_chat(provider, model_id, model_label, query_func, args):
    if model_id in []:
        return provider, model_id, {}

    allowed_total_tokens = 200000
    try:
        # attempt to read grok_models.json like benchmark_parallel does
        grok_path = os.path.join(repo_root, 'grok_models.json')
        if os.path.exists(grok_path):
            with open(grok_path, 'r') as gf:
                data = json.load(gf)
            for entry in data.get('data', []):
                if entry.get('id') == model_id:
                    allowed_total_tokens = int(entry.get('context_window', allowed_total_tokens)) - 256
                    break
    except Exception as e:
        logger.exception(f"Failed to load grok_models.json for model {model_id}: {e}")

    # build solver description text only if requested
    if getattr(args, 'features_only', False):
        # when sending only features, do not include solver description or script
        solver_desc_text = ''
    else:
        if getattr(args, 'include_solver_desc', False):
            solver_desc_map = load_solver_descriptions(args.solver_desc_file if hasattr(args, 'solver_desc_file') else None)
            solver_desc_text = build_solver_description_text(get_solver_list(args.solver_set if hasattr(args, 'solver_set') else 'free'), solver_desc_map)
        else:
            solver_desc_text = ''
    common_instruction = "For each instance above, output a single line: instance_name: [solver1, solver2, solver3]"

    # collect chats across all problems so we can run them in parallel per model
    all_chats = []  # list of tuples (prob_key, [messages])
    for prob_key, prob in problems.items():
        script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
        full_script_path = os.path.join(repo_root, script_path.lstrip('./')) if script_path else None
        try:
            script_text = open(full_script_path, 'r').read() if full_script_path and os.path.exists(full_script_path) else ''
        except Exception as e:
            script_text = ''
            logger.exception(f"Error reading script file {full_script_path} for problem {prob_key}: {e}")

        insts = []
        script_dir = os.path.dirname(full_script_path) if full_script_path else None
        if script_dir and os.path.exists(script_dir):
            for fname in sorted(os.listdir(script_dir)):
                if fname.lower().endswith(('.dzn', '.json')):
                    inst_path = os.path.join(script_dir, fname)
                    try:
                        with open(inst_path, 'r') as inf:
                            content = inf.read()
                    except Exception as e:
                        content = ''
                        logger.exception(f"Error reading instance file {inst_path} for problem {prob_key}: {e}")
                    inst_label = os.path.splitext(fname)[0]
                    # optionally include precomputed features for this instance
                    feat_text = ''
                    if getattr(args, 'include_features', False):
                        feat_entry = MZN2FEAT.get(prob_key, {}).get(inst_label, {})
                        feat_text = feat_entry.get('features', '') if feat_entry else ''

                    if getattr(args, 'features_only', False):
                        # send only the features block (omit model and instance data)
                        content = "Instance features:\n" + feat_text if feat_text else ''
                    elif getattr(args, 'model_and_features', False):
                        # send only the instance features as instance content; the model
                        # will be included as a system message below
                        content = "Instance features:\n" + feat_text if feat_text else ''
                    else:
                        # default: keep original instance content and optionally append features
                        if feat_text:
                            content = content + "\n\nInstance features:\n" + feat_text

                    insts.append((inst_label, content))
        if not insts:
            insts = [('base', '')]

        # optionally include problem description as a system message at the start
        prob_desc = prob.get('description', '') if getattr(args, 'include_problem_desc', False) else ''
        # when features-only mode is active we intentionally omit the model
        effective_script_text = '' if getattr(args, 'features_only', False) else script_text
        chats = pack_instances_into_chats(solver_desc_text, common_instruction, effective_script_text, insts, allowed_total_tokens, prob_desc)
        for c in chats:
            all_chats.append((prob_key, c))

    total_chats = len(all_chats)
    if args.dry_run:
        print(f"{provider}:{model_id} -> total chats to send: {total_chats}")
        # show a few samples
        for i, (pk, msgs) in enumerate(all_chats[:10]):
            print(f"  sample {i+1}: problem={pk}, messages={len(msgs)}")
        # print an example chat (safe-truncated) to help debugging prompt packing
        if all_chats:
            ex_pk, ex_msgs = all_chats[0]
            print("\nExample chat messages (first chat):")
            for m in ex_msgs:
                content = m.get('content','')
                # show at most 2000 chars per message to keep dry-run readable
                snippet = content if len(content) <= 2000 else content[:2000] + '\n...[truncated]'
                print(f"--- role: {m.get('role')} ---\n{snippet}\n")
            print(f"(Example corresponds to problem: {ex_pk})\n")
        return provider, model_id, {}

    chats_results = {}
    # send chats with a small ThreadPoolExecutor per model, similar to benchmark_parallel
    with ThreadPoolExecutor(max_workers=args.max_workers_instances) as executor, \
         tqdm(total=total_chats, desc=f"{provider}/{model_id}", leave=False) as pbar:

        future_to_chat = {executor.submit(resilient_send_chat, msgs, provider, model_id, query_func): (pk, msgs)
                  for pk, msgs in all_chats}

        for fut in as_completed(future_to_chat):
            pk, msgs = future_to_chat[fut]
            pbar.update(1)
            try:
                resp = fut.result()
            except Exception as e:
                # The resilient sender should avoid errors, but if we still get one,
                # attempt a blocking retry (will retry indefinitely until success).
                logger.exception(f"Future raised unexpectedly for provider={provider} model={model_id} problem={pk}: {e}. Falling back to blocking retry.")
                try:
                    resp = resilient_send_chat(msgs, provider, model_id, query_func)
                except Exception as e2:
                    # This should be unreachable because resilient_send_chat retries forever,
                    # but log and continue to next chat if it happens.
                    logger.exception(f"Blocking retry also failed for provider={provider} model={model_id} problem={pk}: {e2}")
                    continue

            pairs = re.findall(r"([^:\n]+)\s*:\s*\[([^\]]+)\]", resp)
            if pairs:
                for name, inner in pairs:
                    name = name.strip()
                    top3 = [s.strip() for s in inner.split(',')]
                    chats_results.setdefault(pk, {})[name] = {'top3': top3, 'time_seconds': None}
            else:
                brackets = re.findall(r"\[([^\]]+)\]", resp)
                user_msgs = [m for m in msgs if m['role'] == 'user']
                for idx, um in enumerate(user_msgs):
                    first_line = um['content'].splitlines()[0]
                    name = first_line.replace('Instance:','').strip()
                    if idx < len(brackets):
                        chats_results.setdefault(pk, {})[name] = {'top3': [s.strip() for s in brackets[idx].split(',')], 'time_seconds': None}
                    else:
                        chats_results.setdefault(pk, {})[name] = {'top3': None, 'error': 'no parsed response'}

    return 'chat', model_id, chats_results


def get_solver_list(setname):
    if setname == 'all':
        return ALL_SOLVERS
    if setname == 'minizinc':
        return MINIZINC_SOLVERS
    return FREE_SOLVERS


def main(argv=None):
    parser = argparse.ArgumentParser(description='Chat-batching benchmark runner')
    parser.add_argument('--solver-set', choices=['minizinc', 'all', 'free'], default='free')
    parser.add_argument('--script-version', choices=['uncommented', 'commented'], default='uncommented')
    parser.add_argument('--max-workers-models', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-workers-instances', type=int, default=1,
                        help='Number of concurrent chat sends per model (default=1)')
    parser.add_argument('--top-only', action='store_true', help='If set, only run models listed in TOP_MODELS')
    parser.add_argument('--best-only', action='store_true', help='If set, only run the best model')
    parser.add_argument('--include-problem-desc', action='store_true',
                        help='Include the problem description as a system message at the start of each chat (default: False)')
    parser.add_argument('--solver-desc-file', default='/test/data/freeSolversDescription.json',
                        help='Path to JSON file with solver descriptions (default: test/data/freeSolversDescription.json)')
    parser.add_argument('--include-solver-desc', action='store_true',
                        help='Include the solver descriptions system message at the start of each chat (default: False)')
    parser.add_argument('--include-features', action='store_true',
                        help='Include instance features with each instance (default: False)')
    parser.add_argument('--mzn2feat-file', default='test/data/mzn2feat_all_features.json',
                        help='Path to JSON file produced by mzn2feat (relative to repo root)')
    parser.add_argument('--features-only', action='store_true',
                        help='Send only instance features to the model (omit model and instance data)')
    parser.add_argument('--model-and-features', action='store_true',
                        help='Send the problem model as system message and instance features as the instance content (omit raw instance data)')
    args = parser.parse_args(argv)

    # load mzn2feat features if requested (cached globally)
    global MZN2FEAT
    MZN2FEAT = {}
    if getattr(args, 'include_features', False) or getattr(args, 'features_only', False) or getattr(args, 'model_and_features', False):
        mfile = args.mzn2feat_file
        # allow relative path (repo_root relative)
        if not os.path.isabs(mfile):
            mfile = os.path.join(repo_root, mfile)
        try:
            with open(mfile, 'r') as mf:
                MZN2FEAT = json.load(mf)
            logger.info('Loaded mzn2feat features from %s', mfile)
        except Exception as e:
            logger.exception('Failed to load mzn2feat features file %s: %s', mfile, e)
            MZN2FEAT = {}

    models = []
    for provider, mod_list, qf in [( 'gemini', GEMINI_MODELS, query_gemini), ('groq', GROQ_MODELS, query_groq)]:
        for mid, mlabel in mod_list:
            if args.top_only and mid not in TOP_MODELS:
                continue
            models.append((provider, mid, mlabel, qf))

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers_models) as mex, \
         tqdm(total=len(models), desc="All Models", position=0) as global_bar:
        futures = {mex.submit(process_model_chat, prov, mid, mlabel, qf, args): (prov, mid)
                   for prov, mid, mlabel, qf in models}
        for fut in as_completed(futures):
            provider, model_id = futures[fut]
            provider, model_id, r = fut.result()
            results.setdefault(provider, {})[model_id] = r
            global_bar.update(1)

    # choose output filename reflecting selected modes
    base = 'data/testOutputFree/LLMsuggestions'
    if getattr(args, 'features_only', False):
        name = base + '_featOnly'
    elif getattr(args, 'model_and_features', False):
        name = base + '_modelFeat'
    elif getattr(args, 'include_features', False):
        name = base + '_features'
    else:
        name = base + '_chat'

    # append optional suffixes for problem and solver descriptions
    if getattr(args, 'include_problem_desc', False):
        name += '_Pdesc'
    if getattr(args, 'include_solver_desc', False):
        name += '_Sdesc'

    fname = name + '.json'
    try:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as of:
            json.dump(results, of, indent=2)
    except Exception as e:
        logger.exception(f"Failed to write {fname}: {e}")


if __name__ == '__main__':
    main()