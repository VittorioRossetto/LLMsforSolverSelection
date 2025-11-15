#!/usr/bin/env python3
"""Sanitize large arrays/matrices in .mzn and .dzn files.

This script copies a source directory (default: mznc2025_probs) to a destination
directory and for each .mzn/.dzn file it scans for array/matrix literals. If an
array (counting scalar elements) contains more than --max-elements (default 200)
the script will replace the whole assignment statement with a single-line comment
containing the variable name and the detected dimensions, e.g.:

  // myArray array too long to display, dimensions: (1000,)

Usage:
  python3 tools/sanitize_large_arrays.py --src mznc2025_probs --dst mznc2025_probs_sanitized

Supports --dry-run to preview changes without writing files.
"""
import argparse
import os
import shutil
import sys
import io
import re


def find_matching_bracket(s, start_idx, open_ch='[', close_ch=']'):
    """Return index of matching close_ch for the bracket at start_idx or -1."""
    depth = 0
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level_items(s):
    """Split a bracket-inner string into top-level items (handles nested brackets).
    Returns list of item strings (trimmed).
    """
    items = []
    cur = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '[':
            depth += 1
            cur.append(ch)
        elif ch == ']':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            items.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        items.append(''.join(cur).strip())
    # filter out empty items
    return [it for it in items if it != '']


def count_and_dims_from_list_inner(s):
    """Given the inner text of a bracketed list (without outer [..]),
    return (count, dims) where count is total scalar elements and dims is a tuple-like list.

    dims: If the list contains nested lists of equal length, we provide shape (rows, cols,...).
    Otherwise dims will be a single-element tuple (N,) where N is the total scalar count.
    """
    s = s.strip()
    if not s:
        return 0, (0,)
    # detect if top-level items are themselves lists
    items = split_top_level_items(s)
    # if first item starts with '[' then assume nested lists
    if items and items[0].lstrip().startswith('['):
        inner_counts = []
        inner_dims = None
        for it in items:
            it = it.strip()
            if it.startswith('[') and it.endswith(']'):
                inner = it[1:-1]
                cnt, dims = count_and_dims_from_list_inner(inner)
                inner_counts.append(cnt)
                if inner_dims is None:
                    inner_dims = dims
                elif inner_dims != dims:
                    # irregular inner shapes -> don't report multi-dim dims
                    inner_dims = None
            else:
                # irregular content: treat as scalar
                cnt, dims = 1, (1,)
                inner_counts.append(cnt)
                inner_dims = None

        total = sum(inner_counts)
        if inner_dims is not None:
            # shape is (len(items),) + inner_dims
            shape = (len(items),) + tuple(inner_dims)
            return total, shape
        else:
            return total, (total,)
    else:
        # flat list; count top-level commas +1
        # but ensure we handle trailing commas
        flat_items = items
        return len(flat_items), (len(flat_items),)


def sanitize_file_contents(text, max_elements=200, keep_elements=30, max_chars=80, char_keep=80):
    """Return (new_text, changed) where arrays/matrices above max_elements are replaced.
    Replacement: replace the full assignment statement (var ... = [...] ;) with a single-line
    comment: // <varname> array too long to display, dimensions: <dims>
    If var name can't be determined, use a generic comment.
    """
    out = []
    i = 0
    changed = False
    L = len(text)
    while i < L:
        ch = text[i]
        if ch == '[':
            # find matching ]
            end_br = find_matching_bracket(text, i, '[', ']')
            if end_br == -1:
                # unmatched - copy rest and break
                out.append(text[i:])
                break
            inner = text[i+1:end_br]
            total, dims = count_and_dims_from_list_inner(inner)
            inner_len_chars = len(inner)
            # decide truncation based on element count OR total character length inside brackets
            if total > max_elements or inner_len_chars > max_chars:
                # find a reasonable statement boundary and lhs
                # search for '=' (MiniZinc) or ':' (JSON) before the '['
                eq_pos = text.rfind('=', 0, i)
                colon_pos = text.rfind(':', 0, i)
                # find start of statement (line start)
                stmt_start = 0
                if eq_pos != -1:
                    stmt_start = text.rfind('\n', 0, eq_pos) + 1 if eq_pos != -1 else 0
                elif colon_pos != -1:
                    stmt_start = text.rfind('\n', 0, colon_pos) + 1 if colon_pos != -1 else 0
                else:
                    stmt_start = text.rfind('\n', 0, i) + 1

                # detect trailing punctuation (semicolon or comma) after the closing bracket
                j = end_br + 1
                while j < L and text[j].isspace():
                    j += 1
                trailing = ''
                if j < L and text[j] in ',;':
                    trailing = text[j]
                    stmt_end = j + 1
                else:
                    stmt_end = end_br + 1

                # build lhs string
                lhs_str = None
                if eq_pos != -1:
                    lhs_str = text[stmt_start:eq_pos].rstrip()
                elif colon_pos != -1:
                    # include the colon
                    lhs_str = text[stmt_start:colon_pos+1].rstrip()
                else:
                    # fallback: use content from line start to bracket
                    lhs_str = text[stmt_start:i].rstrip()

                # If truncation is triggered by character-length, truncate by characters
                if inner_len_chars > max_chars:
                    shortened_inner = inner[:char_keep]
                    # indicate truncation inside the bracket
                    shortened_bracket = '[' + shortened_inner + '... ]'
                else:
                    # split top-level items and keep the first keep_elements
                    items = split_top_level_items(inner)
                    kept = items[:keep_elements]
                    shortened_inner = ', '.join(kept)
                    shortened_bracket = '[' + shortened_inner + ']'

                # dims string for comment
                dims_str = '(' + ','.join(str(d) for d in (dims if isinstance(dims, tuple) else (dims,))) + ')'
                comment = f" // array too long to display, dimensions: {dims_str}\n"

                # construct replacement: lhs + separator + shortened bracket + trailing + comment
                replacement = None
                if eq_pos != -1:
                    replacement = f"{lhs_str} = {shortened_bracket}{trailing}{comment}"
                elif colon_pos != -1:
                    # ensure we have a space after colon
                    replacement = f"{lhs_str} {shortened_bracket}{trailing}{comment}"
                else:
                    replacement = comment

                out.append(replacement)
                i = stmt_end
                changed = True
                continue
            else:
                # copy bracketed content as-is
                out.append(text[i:end_br+1])
                i = end_br+1
                continue
        else:
            out.append(ch)
            i += 1

    return ''.join(out), changed


def process_dir(src, dst, max_elements=256, keep_elements=30, max_chars=80, char_keep=80, dry_run=False):
    if os.path.exists(dst):
        print(f"Destination {dst} already exists. Overwriting contents.")
        # keep existing but we'll write files inside
    else:
        shutil.copytree(src, dst)

    files_changed = []
    for root, dirs, files in os.walk(dst):
        for fn in files:
            if fn.lower().endswith(('.dzn', '.mzn', '.json')):
                fp = os.path.join(root, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        txt = f.read()
                except Exception as e:
                    print(f"Failed to read {fp}: {e}")
                    continue

                newtxt, changed = sanitize_file_contents(txt, max_elements=max_elements, keep_elements=keep_elements, max_chars=max_chars, char_keep=char_keep)
                if changed:
                    files_changed.append(fp)
                    if dry_run:
                        print(f"[DRY-RUN] Would sanitize: {fp}")
                    else:
                        try:
                            with open(fp, 'w', encoding='utf-8') as f:
                                f.write(newtxt)
                        except Exception as e:
                            print(f"Failed to write {fp}: {e}")

    return files_changed


def main(argv=None):
    parser = argparse.ArgumentParser(description='Copy and sanitize large arrays in a MiniZinc problems folder')
    parser.add_argument('--src', default='mznc2025_probs', help='Source folder to copy (default: mznc2025_probs)')
    parser.add_argument('--dst', default=None, help='Destination folder (default: <src>_sanitized)')
    parser.add_argument('--max-elements', type=int, default=200, help='Maximum scalar elements allowed before sanitizing (default: 200)')
    parser.add_argument('--keep-elements', type=int, default=30, help='Number of elements to keep when truncating (default: 30)')
    parser.add_argument('--max-chars', type=int, default=80, help='Maximum characters inside brackets before truncating (default: 20000)')
    parser.add_argument('--char-keep', type=int, default=80, help='Number of characters to keep when truncating by character length (default: 50)')
    parser.add_argument('--dry-run', action='store_true', help='Show which files would be changed without modifying files')
    args = parser.parse_args(argv)

    src = args.src
    if args.dst:
        dst = args.dst
    else:
        dst = src.rstrip('/\\') + '_sanitized'

    if not os.path.exists(src):
        print(f"Source folder not found: {src}")
        sys.exit(2)

    changed = process_dir(src, dst, max_elements=args.max_elements, keep_elements=args.keep_elements, max_chars=args.max_chars, char_keep=args.char_keep, dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"Dry-run complete. {len(changed)} files would be sanitized.")
    else:
        print(f"Sanitization complete. {len(changed)} files changed in {dst}.")


if __name__ == '__main__':
    main()
