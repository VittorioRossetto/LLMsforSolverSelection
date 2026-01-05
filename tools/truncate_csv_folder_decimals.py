#!/usr/bin/env python3
"""Recursively copy a folder of CSV files, truncating decimal digits.

Given an input folder, this script walks all subfolders, finds `.csv` files,
creates a sibling output folder named `<folder_name>_short`, and writes copies
of the CSVs with all numeric cells truncated (NOT rounded) to at most `n`
decimal digits.

Only plain decimal representations are truncated (e.g. -12.3456). Scientific
notation and numbers with thousands separators are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional


_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+)(?:\.(\d+))?$")


def truncate_decimal_string(value: str, n: int) -> str:
    """Truncate a plain decimal string to at most n fractional digits.

    Preserves leading/trailing whitespace from the original string.
    If the string is not a plain decimal, returns it unchanged.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    if value is None:
        return value

    leading_ws_len = len(value) - len(value.lstrip())
    trailing_ws_len = len(value) - len(value.rstrip())
    core = value.strip()

    if not core:
        return value

    m = _DECIMAL_RE.match(core)
    if not m:
        return value

    frac = m.group(1)
    if frac is None:
        return value

    if len(frac) <= n:
        return value

    if n == 0:
        truncated_core = core.split(".", 1)[0]
    else:
        int_part, frac_part = core.split(".", 1)
        truncated_core = f"{int_part}.{frac_part[:n]}"

    return (" " * leading_ws_len) + truncated_core + (" " * trailing_ws_len)


@dataclass(frozen=True)
class CopyStats:
    files_processed: int = 0
    rows_written: int = 0


def _sniff_dialect(sample: str) -> csv.Dialect:
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample)
    except Exception:
        return csv.excel


def process_csv_file(src_path: str, dst_path: str, n: int) -> int:
    """Read CSV at src_path, write processed CSV to dst_path. Returns rows written."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(src_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        dialect = _sniff_dialect(sample)
        reader = csv.reader(f, dialect)

        with open(dst_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, dialect)
            rows = 0
            for row in reader:
                writer.writerow([truncate_decimal_string(cell, n) for cell in row])
                rows += 1

    return rows


def build_output_dir(input_dir: str) -> str:
    norm = os.path.normpath(os.path.abspath(input_dir))
    base = os.path.basename(norm)
    parent = os.path.dirname(norm)
    return os.path.join(parent, f"{base}_short")


def copy_folder_csvs(input_dir: str, n: int, overwrite: bool = False) -> CopyStats:
    input_dir = os.path.normpath(os.path.abspath(input_dir))
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    out_dir = build_output_dir(input_dir)

    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. "
                "Use --overwrite to write into it."
            )
    else:
        os.makedirs(out_dir, exist_ok=True)

    stats = CopyStats()

    for root, _dirs, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        dst_root = out_dir if rel == "." else os.path.join(out_dir, rel)
        os.makedirs(dst_root, exist_ok=True)

        for name in files:
            if not name.lower().endswith(".csv"):
                # Folder is expected to contain only CSVs; ignore anything else.
                continue
            src_path = os.path.join(root, name)
            dst_path = os.path.join(dst_root, name)
            rows = process_csv_file(src_path, dst_path, n)
            stats = CopyStats(files_processed=stats.files_processed + 1, rows_written=stats.rows_written + rows)

    return stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy a CSV folder tree into <name>_short, truncating decimals to n digits."
    )
    parser.add_argument("input_dir", help="Path to the folder containing CSV files")
    parser.add_argument("n", type=int, help="Max number of decimal digits to keep (truncate, not round)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing <name>_short folder",
    )
    args = parser.parse_args(argv)

    stats = copy_folder_csvs(args.input_dir, args.n, overwrite=args.overwrite)
    out_dir = build_output_dir(args.input_dir)
    print(f"Wrote {stats.files_processed} CSV files to: {out_dir}")
    print(f"Total rows written: {stats.rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
