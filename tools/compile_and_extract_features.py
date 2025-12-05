#!/usr/bin/env python3
"""
Compile commented MiniZinc problems with instances and extract features with mzn2feat.

Usage:
  python3 tools/compile_and_extract_features.py \
      --root mznc2025_probs --out mzn2feat_all_features.json

Prerequisites:
  - `minizinc` must be on PATH
  - `mzn2feat` must be on PATH

This script will:
  - find commented model files in each problem folder (files matching '*_commented.mzn' or 'model_commented.mzn')
  - for each instance file (.dzn, .json) in the same folder, run:
      minizinc --compile --solver Gecode --use-gecode <model> <instance> -o <out.fzn>
  - run `mzn2feat -i <out.fzn> -o pp` and capture its stdout
  - write a JSON mapping Problem -> Instance -> {"features": <mzn2feat output>} to the output file

The fzn files are written into the `--fzn-dir` directory (default: `.mzn2fzn_out`).
"""

import argparse
import os
import sys
import subprocess
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('compile_and_extract_features')


def find_commented_models(problem_dir):
    # match common patterns: *_commented.mzn or model_commented.mzn
    res = []
    for fname in os.listdir(problem_dir):
        if fname.lower().endswith('.mzn') and 'commented' in fname.lower():
            res.append(os.path.join(problem_dir, fname))
    return sorted(res)


def find_instances(problem_dir):
    insts = []
    for fname in sorted(os.listdir(problem_dir)):
        low = fname.lower()
        if low.endswith('.dzn') or low.endswith('.json'):
            insts.append(os.path.join(problem_dir, fname))
    return insts


def model_base_name(model_path):
    bn = os.path.splitext(os.path.basename(model_path))[0]
    if bn.endswith('_commented'):
        return bn[:-len('_commented')]
    # also handle patterns like 'model_commented'
    if bn.endswith('commented') and bn.endswith('_commented') is False:
        # best-effort remove suffix 'commented' if preceded by underscore
        parts = bn.rsplit('_', 1)
        if len(parts) == 2 and parts[1].lower() == 'commented':
            return parts[0]
    return bn


def instance_base_name(inst_path):
    return os.path.splitext(os.path.basename(inst_path))[0]


def run_minizinc_compile(model_path, inst_path, out_fzn_path, extra_args=None):
    cmd = [
        'minizinc', '--compile', '--solver', 'Gecode', '--use-gecode',
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([model_path, inst_path, '-o', out_fzn_path])
    logger.debug('Running: %s', ' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode('utf-8', errors='replace'), proc.stderr.decode('utf-8', errors='replace')


def run_mzn2feat(fzn_path):
    # the user requested `mzn2feat -i <fzn> -o pp`
    cmd = ['mzn2feat', '-i', fzn_path, '-o', 'pp']
    logger.debug('Running: %s', ' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode('utf-8', errors='replace'), proc.stderr.decode('utf-8', errors='replace')


def sanitize_fzn_annotations(input_fzn, output_fzn):
    """Remove solver/annotation fragments using '::' from a FlatZinc file.

    This is a best-effort sanitizer: it strips '::' plus either a parenthesized
    annotation or a following identifier token. It preserves the rest of the
    file so `mzn2feat` can parse it when annotations are unsupported.
    """
    with open(input_fzn, 'r', encoding='utf-8', errors='replace') as inf:
        txt = inf.read()

    out_chars = []
    i = 0
    n = len(txt)
    while i < n:
        idx = txt.find('::', i)
        if idx == -1:
            out_chars.append(txt[i:])
            break
        # append up to '::'
        out_chars.append(txt[i:idx])
        j = idx + 2
        # skip whitespace
        while j < n and txt[j].isspace():
            j += 1
        if j < n and txt[j] == '(':
            # skip until matching ')' (simple, non-nested)
            j2 = j + 1
            depth = 1
            while j2 < n and depth > 0:
                if txt[j2] == '(':
                    depth += 1
                elif txt[j2] == ')':
                    depth -= 1
                j2 += 1
            i = j2
        else:
            # skip identifier characters [A-Za-z0-9_:.]
            j2 = j
            while j2 < n and (txt[j2].isalnum() or txt[j2] in '_:.'):
                j2 += 1
            i = j2
    new_txt = ''.join(out_chars)
    # optional: collapse multiple spaces left by removal
    new_txt = re.sub(r'\s{2,}', ' ', new_txt)
    with open(output_fzn, 'w', encoding='utf-8') as outf:
        outf.write(new_txt)


def process_instance(model_path, inst_path, fzn_dir, keep_fzn=False):
    model_base = model_base_name(model_path)
    inst_base = instance_base_name(inst_path)
    fzn_name = f"{model_base}_{inst_base}.fzn"
    fzn_path = os.path.join(fzn_dir, fzn_name)

    os.makedirs(os.path.dirname(fzn_path), exist_ok=True)

    # Try normal compile first
    retcode, out, err = run_minizinc_compile(model_path, inst_path, fzn_path)
    used_allow_unbounded = False
    if retcode != 0:
        logger.debug('Initial minizinc compile stderr: %s', err)
        # If the error mentions unbounded variable, retry with --allow-unbounded-vars
        if 'unbounded variable' in err.lower() or 'allow-unbounded-vars' in err.lower():
            logger.info('Retrying minizinc compile with --allow-unbounded-vars for %s + %s', model_path, inst_path)
            retcode, out, err = run_minizinc_compile(model_path, inst_path, fzn_path, extra_args=['--allow-unbounded-vars'])
            used_allow_unbounded = True

    if retcode != 0:
        logger.warning('minizinc compile failed for %s + %s: rc=%d stderr=%s', model_path, inst_path, retcode, err[:200])
        return inst_-base, {'error': 'minizinc_compile_failed', 'rc': retcode, 'stderr': err, 'used_allow_unbounded': used_allow_unbounded}

    # run mzn2feat
    # Run mzn2feat; if it fails with a syntax error mentioning '::', try sanitizing the .fzn and retry
    rc2, feat_out, feat_err = run_mzn2feat(fzn_path)
    sanitized_used = False
    sanitized_path = None
    if rc2 != 0 and '::' in (feat_err or ''):
        logger.info('mzn2feat reported :: syntax error for %s; attempting to sanitize FZN and retry', fzn_path)
        try:
            sanitized_path = 'sanitized_' + fzn_path
            sanitize_fzn_annotations(fzn_path, sanitized_path)
            sanitized_used = True
            rc2, feat_out, feat_err = run_mzn2feat(sanitized_path)
        except Exception as e:
            logger.exception('Failed to sanitize FZN %s: %s', fzn_path, e)

    if rc2 != 0:
        logger.warning('mzn2feat failed for %s: rc=%d stderr=%s', sanitized_path or fzn_path, rc2, (feat_err or '')[:200])
        result = {'error': 'mzn2feat_failed', 'rc': rc2, 'stderr': feat_err, 'fzn': fzn_path, 'sanitized_fzn': sanitized_path, 'sanitized_used': sanitized_used}
    else:
        # store the features output (stdout)
        result = {'features': feat_out, 'fzn': fzn_path, 'sanitized_fzn': sanitized_path, 'sanitized_used': sanitized_used}

    if not keep_fzn:
        try:
            os.remove(fzn_path)
        except Exception:
            pass
    if sanitized_used and sanitized_path and not keep_fzn:
        try:
            os.remove(sanitized_path)
        except Exception:
            pass

    return inst_base, result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compile commented MiniZinc problems and run mzn2feat')
    parser.add_argument('--root', default='mznc2025_probs', help='Root folder containing problems (default: mznc2025_probs)')
    parser.add_argument('--out', default='mzn2feat_all_features.json', help='Output JSON file')
    parser.add_argument('--fzn-dir', default='.mzn2fzn_out', help='Directory to write temporary .fzn files')
    parser.add_argument('--keep-fzn', action='store_true', help='Keep generated .fzn files')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for instance processing')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    root = args.root
    if not os.path.exists(root):
        logger.error('Root path does not exist: %s', root)
        sys.exit(2)

    all_results = {}

    # iterate problem subfolders
    problems = [p for p in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, p))]
    if not problems:
        logger.error('No problem folders found under %s', root)
        sys.exit(1)

    for prob in problems:
        prob_dir = os.path.join(root, prob)
        models = find_commented_models(prob_dir)
        if not models:
            logger.info('No commented model found in %s, skipping', prob_dir)
            continue
        # We compile each commented model (if multiple, process them all)
        all_results.setdefault(prob, {})
        for model in models:
            instances = find_instances(prob_dir)
            if not instances:
                logger.info('No instance files found in %s; skipping model %s', prob_dir, model)
                continue

            logger.info('Processing problem %s model %s with %d instances', prob, os.path.basename(model), len(instances))

            # process instances in parallel
            futures = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for inst in instances:
                    futures.append(ex.submit(process_instance, model, inst, args.fzn_dir, args.keep_fzn))

                for fut in as_completed(futures):
                    inst_name, res = fut.result()
                    all_results[prob].setdefault(inst_name, {})
                    all_results[prob][inst_name] = res

    # write output JSON
    try:
        with open(args.out, 'w') as of:
            json.dump(all_results, of, indent=2)
        logger.info('Wrote results to %s', args.out)
    except Exception as e:
        logger.exception('Failed to write output JSON: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
