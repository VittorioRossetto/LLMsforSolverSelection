#!/usr/bin/env python3

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from bs4 import FeatureNotFound


DEFAULT_START_URL = "https://docs.minizinc.dev/en/stable/lib-flatzinc.html"


def _make_soup(html: str) -> BeautifulSoup:
    """Create a BeautifulSoup instance with a best-effort parser.

    Prefer `lxml` when available; fall back to the stdlib `html.parser` so the
    script runs without extra native dependencies.
    """
    for parser in ("lxml", "html.parser", "html5lib"):
        try:
            return BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
    # Shouldn't happen (html.parser ships with Python), but be defensive.
    return BeautifulSoup(html, "html.parser")


@dataclass(frozen=True)
class LinkTarget:
    page_url: str
    fragment: str


_DECL_RE = re.compile(
    r"\b(predicate|function|annotation|constraint)\b\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


def _normalize_whitespace(s: str) -> str:
    s = (s or "").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tighten_indexing(s: str) -> str:
    """Turn 'as [ b ]' into 'as[b]' (best-effort)."""

    def _fix(m: re.Match) -> str:
        arr = m.group(1)
        idx = _normalize_whitespace(m.group(2))
        return f"{arr}[{idx}]"

    s = re.sub(r"\b(\w+)\s*\[\s*([^\]]+?)\s*\]", _fix, s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s+", ", ", s)
    s = re.sub(r"\s+=\s+", " = ", s)
    s = re.sub(r"\s+<=\s+", " <= ", s)
    s = re.sub(r"\s+>=\s+", " >= ", s)
    s = re.sub(r"\s+<\s+", " < ", s)
    s = re.sub(r"\s+>\s+", " > ", s)
    return s


def _extract_decl_name_from_pre(pre_text: str) -> Optional[str]:
    pre_text = (pre_text or "").strip()
    m = _DECL_RE.search(pre_text)
    if not m:
        return None
    return m.group("name")


def _text_from_cell(cell: Tag) -> str:
    # Preserve inline code-ish elements without backticks.
    txt = cell.get_text(" ", strip=True)
    txt = _normalize_whitespace(txt)
    txt = _tighten_indexing(txt)
    return txt


def _table_for_fragment(soup: BeautifulSoup, fragment: str) -> Optional[Tag]:
    if not fragment:
        return None
    frag = fragment[1:] if fragment.startswith("#") else fragment
    if not frag:
        return None
    anchor = soup.find(id=frag)
    if not isinstance(anchor, Tag):
        return None

    # In the MiniZinc docs, the fragment often identifies a <section> that contains
    # the table (rather than an element nested inside the table).
    table = anchor.find("table", class_="docutils")
    if isinstance(table, Tag):
        return table

    # Fallback: use the next docutils table in document order.
    nxt = anchor.find_next("table", class_="docutils")
    return nxt if isinstance(nxt, Tag) else None


def _extract_from_docutils_table(table: Tag) -> Optional[Tuple[str, str]]:
    # Expected structure:
    #  row 1: <pre> with 'predicate name(...'
    #  row 2: description
    rows = table.find_all("tr")
    if len(rows) < 2:
        return None

    pre = rows[0].find("pre")
    if not isinstance(pre, Tag):
        return None
    name = _extract_decl_name_from_pre(pre.get_text("\n", strip=True))
    if not name:
        return None

    desc_cell = rows[1].find("td")
    if not isinstance(desc_cell, Tag):
        return None
    desc = _text_from_cell(desc_cell)
    if not desc:
        return None
    return name, desc


def _iter_reference_internal_links(soup: BeautifulSoup, base_url: str) -> Iterable[LinkTarget]:
    for a in soup.select("a.reference.internal"):
        if not isinstance(a, Tag):
            continue
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        page_url, frag = urldefrag(abs_url)
        if not frag:
            continue
        # Keep only FlatZinc library reference sections; this avoids navigation links.
        if "lib-flatzinc-" not in page_url:
            continue
        yield LinkTarget(page_url=page_url, fragment=f"#{frag}")


def _load_json_mapping(path: Path) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}")

    if isinstance(data, dict):
        out: Dict[str, str] = {}
        for k, v in data.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                out[kk] = vv
        return out

    raise SystemExit(
        f"{path} must be a JSON object mapping constraint -> description (got {type(data).__name__})"
    )


def _write_json_mapping(path: Path, mapping: Dict[str, str]) -> None:
    text = json.dumps(dict(sorted(mapping.items())), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def scrape(
    start_url: str,
    session: requests.Session,
    delay_s: float,
) -> Dict[str, str]:
    r = session.get(start_url, timeout=30)
    r.raise_for_status()
    start_soup = _make_soup(r.text)
    targets = list(_iter_reference_internal_links(start_soup, start_url))

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[LinkTarget] = []
    for t in targets:
        key = (t.page_url, t.fragment)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    page_cache: Dict[str, BeautifulSoup] = {}
    out: Dict[str, str] = {}

    for i, t in enumerate(uniq, 1):
        soup = page_cache.get(t.page_url)
        if soup is None:
            if delay_s > 0:
                time.sleep(delay_s)
            rr = session.get(t.page_url, timeout=30)
            rr.raise_for_status()
            soup = _make_soup(rr.text)
            page_cache[t.page_url] = soup

        table = _table_for_fragment(soup, t.fragment)
        if not table:
            continue

        extracted = _extract_from_docutils_table(table)
        if not extracted:
            continue
        name, desc = extracted
        out[name] = desc

        if i % 250 == 0:
            print(f"Processed {i}/{len(uniq)} links… extracted {len(out)} entries")

    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape MiniZinc FlatZinc library docs and merge predicate/constraint descriptions into tools/fzn_descriptions.json"
        )
    )
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("fzn_descriptions.json")),
        help="Path to fzn_descriptions.json to update",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing keys in the JSON (default: keep existing values)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay (seconds) between HTTP requests for new pages (default: 0.05)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    existing = _load_json_mapping(out_path) if out_path.exists() else {}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AgenticSolvers/1.0 (scrape_fzn_descriptions.py; contact: local)"
        }
    )

    scraped = scrape(args.start_url, session=session, delay_s=max(0.0, args.delay))

    added = 0
    updated = 0
    merged = dict(existing)
    for k, v in scraped.items():
        if k not in merged:
            merged[k] = v
            added += 1
        elif args.overwrite and merged.get(k) != v:
            merged[k] = v
            updated += 1

    _write_json_mapping(out_path, merged)
    print(
        f"Scraped {len(scraped)} entries. Added {added}, updated {updated}, total now {len(merged)} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())