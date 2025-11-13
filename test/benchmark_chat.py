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

# make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, repo_root)
from utils import *


# Models to include when --top-only is used (kept in sync with benchmark_parallel.py)
TOP_MODELS = [
    'gemini-2.5-flash-lite', 'gemini-2.0-flash',
    'moonshotai/kimi-k2-instruct', 'moonshotai/kimi-k2-instruct-0905',
    'openai/gpt-oss-120b'
]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


# --- load problems (sanitized dataset) ---
dataset_path = os.path.join(repo_root, "mznc2025_probs_sanitized", "problems_with_descriptions.json")
problems = load_problems(dataset_path)


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


def pack_instances_into_chats(solver_desc_text, common_instruction, script_text, instances, allowed_total_tokens):
    """Greedy pack instances into chat conversations given token budget.

    instances: list of tuples (inst_label, instance_content)
    returns: list of chats, where each chat is list of messages (dicts with role/content)
    """
    chats = []
    # include the model script once per chat in the system message; account for its tokens
    base_non_script = estimate_tokens(solver_desc_text) + estimate_tokens(common_instruction) + estimate_tokens(script_text)
    cur = []
    cur_tokens = base_non_script
    for inst_label, content in instances:
        inst_msg = f"Instance: {inst_label}\nMiniZinc data:\n{content}\n\n{common_instruction}"
        inst_tokens = estimate_tokens(inst_msg)
        # if adding would exceed allowed_total_tokens, flush
        if cur and (cur_tokens + inst_tokens > allowed_total_tokens):
            # build messages for cur
            # keep solver descriptions and the model as two separate system messages
            msgs = [
                {'role': 'system', 'content': solver_desc_text},
            ]
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
        msgs = [
            {'role': 'system', 'content': solver_desc_text},
        ]
        if script_text:
            msgs.append({'role': 'system', 'content': "MiniZinc model:\n" + script_text + "\n"})
        for c_label, c_content in cur:
            msgs.append({'role': 'user', 'content': f"Instance: {c_label}\nMiniZinc data:\n{c_content}\n\n{common_instruction}"})
        chats.append(msgs)

    return chats


def send_chat(messages, provider, model_id, query_func):
    """Try to send chat messages; fall back to joining if provider doesn't accept messages param."""
    try:
        # prefer chat/multi-message API
        return query_func(messages=messages, model_name=model_id)
    except TypeError:
        # provider wrapper probably expects plain prompt
        joined = "\n----\n".join(m['content'] for m in messages)
        return query_func(joined, model_name=model_id)


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
    except Exception:
        pass

    # build solver description text
    solver_desc_map = load_solver_descriptions(args.solver_desc_file if hasattr(args, 'solver_desc_file') else None)
    solver_desc_text = build_solver_description_text(get_solver_list(args.solver_set if hasattr(args, 'solver_set') else 'free'), solver_desc_map)
    common_instruction = "For each instance above, output a single line: instance_name: [solver1, solver2, solver3]"

    # collect chats across all problems so we can run them in parallel per model
    all_chats = []  # list of tuples (prob_key, [messages])
    for prob_key, prob in problems.items():
        script_path = prob.get('script_commented' if args.script_version == 'commented' else 'script', '')
        full_script_path = os.path.join(repo_root, script_path.lstrip('./')) if script_path else None
        try:
            script_text = open(full_script_path, 'r').read() if full_script_path and os.path.exists(full_script_path) else ''
        except Exception:
            script_text = ''

        insts = []
        script_dir = os.path.dirname(full_script_path) if full_script_path else None
        if script_dir and os.path.exists(script_dir):
            for fname in sorted(os.listdir(script_dir)):
                if fname.lower().endswith(('.dzn', '.json')):
                    inst_path = os.path.join(script_dir, fname)
                    try:
                        with open(inst_path, 'r') as inf:
                            content = inf.read()
                    except Exception:
                        content = ''
                    insts.append((os.path.splitext(fname)[0], content))
        if not insts:
            insts = [('base', '')]

        chats = pack_instances_into_chats(solver_desc_text, common_instruction, script_text, insts, allowed_total_tokens)
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
                snippet = content if len(content) <= 10000 else content[:10000] + '\n...[truncated]'
                print(f"--- role: {m.get('role')} ---\n{snippet}\n")
            print(f"(Example corresponds to problem: {ex_pk})\n")
        return provider, model_id, {}

    chats_results = {}
    # send chats with a small ThreadPoolExecutor per model, similar to benchmark_parallel
    with ThreadPoolExecutor(max_workers=args.max_workers_instances) as executor, \
         tqdm(total=total_chats, desc=f"{provider}/{model_id}", leave=False) as pbar:

        future_to_chat = {executor.submit(send_chat, msgs, provider, model_id, query_func): (pk, msgs)
                          for pk, msgs in all_chats}

        for fut in as_completed(future_to_chat):
            pk, msgs = future_to_chat[fut]
            pbar.update(1)
            try:
                resp = fut.result()
            except Exception as e:
                for um in [m for m in msgs if m['role'] == 'user']:
                    first_line = um['content'].splitlines()[0]
                    name = first_line.replace('Instance:','').strip()
                    chats_results.setdefault(pk, {})[name] = {'top3': None, 'error': str(e)}
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
    parser.add_argument('--solver-desc-file', default='/test/data/freeSolversDescription.json',
                        help='Path to JSON file with solver descriptions (default: test/data/freeSolversDescription.json)')
    args = parser.parse_args(argv)

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

    with open('LLMsuggestions_chat.json', 'w') as of:
        json.dump(results, of, indent=2)


if __name__ == '__main__':
    main()