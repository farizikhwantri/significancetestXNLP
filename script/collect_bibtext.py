#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # optional


TITLE_LINE_RE = re.compile(r'"\s*(.*?)\s*"')

def normalize_title(s: str) -> str:
    # Drop LaTeX commands like \textbf{...}, \LaTeX, \url{...}
    s = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', ' ', s)
    # Unescape TeX special symbols: \& \% \_ \{ \} \# \$ \~ \^ \\
    s = re.sub(r'\\([#\$%&_{}~^\\])', r'\1', s)
    # Remove braces (e.g., {S}em{E}val -> SemEval, {\&} -> & after unescape above)
    s = s.replace("{", "").replace("}", "")
    # Normalize dashes and whitespace; lowercase
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def load_filtered_titles(titles_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    # Return ordered list of normalized titles and a map norm -> list of raw title variants
    ordered_norms: List[str] = []
    norm_to_raws: Dict[str, List[str]] = {}
    for line in titles_path.read_text(encoding="utf-8").splitlines():
        m = TITLE_LINE_RE.search(line)
        if not m:
            continue
        raw = m.group(1).strip()
        norm = normalize_title(raw)
        if norm not in norm_to_raws:
            norm_to_raws[norm] = []
            ordered_norms.append(norm)
        if raw not in norm_to_raws[norm]:
            norm_to_raws[norm].append(raw)
    return ordered_norms, norm_to_raws

def extract_year(entry: dict) -> Optional[int]:
    y = entry.get("year")
    if not y:
        return None
    m = re.search(r'(\d{4})', str(y))
    return int(m.group(1)) if m else None

def index_bib_by_title(bib_db) -> Dict[str, List[Tuple[Optional[int], dict, str]]]:
    idx: Dict[str, List[Tuple[Optional[int], dict, str]]] = {}
    for e in bib_db.entries:
        title = e.get("title")
        if not title:
            continue
        norm = normalize_title(title)
        year = extract_year(e)
        idx.setdefault(norm, []).append((year, e, title))
    return idx

def main():
    ap = argparse.ArgumentParser(description="Match titles to BibTeX first, then filter by year.")
    ap.add_argument("--titles-file", default="./output/filtered_titles.txt")
    ap.add_argument("--bib-file", default="./data/anthology.bib")
    ap.add_argument("--since", type=int, default=2020)
    ap.add_argument("--out", default="./output/filtered_bibtex_since2020.bib")
    ap.add_argument("--fuzzy", action="store_true", help="Enable fuzzy fallback (needs rapidfuzz).")
    ap.add_argument("--fuzzy-thresh", type=int, default=92, help="Fuzzy ratio threshold if --fuzzy.")
    args = ap.parse_args()

    ordered_norms, norm_to_raws = load_filtered_titles(Path(args.titles_file))

    with Path(args.bib_file).open("r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)

    idx = index_bib_by_title(bib_db)

    matched_any: Dict[str, Tuple[Optional[int], dict, str]] = {}
    written_entries: List[dict] = []
    unmatched_norms: List[str] = []

    # Exact normalized title match first (any year)
    for norm in ordered_norms:
        candidates = idx.get(norm, [])
        if not candidates:
            unmatched_norms.append(norm)
            continue
        # Keep the latest-year candidate as canonical match
        best_any = max(candidates, key=lambda t: (t[0] or -1))
        matched_any[norm] = best_any
        # Write all candidates that satisfy year >= since (some titles may have multiple entries)
        for y, e, _ in candidates:
            if y is not None and y >= args.since:
                written_entries.append(e)

    # Optional fuzzy fallback for truly unmatched norms
    if args.fuzzy and fuzz is not None:
        still_unmatched = [n for n in unmatched_norms if n not in matched_any]
        keys = list(idx.keys())
        for norm in still_unmatched:
            best_key, best_score = None, -1
            for k in keys:
                s = fuzz.ratio(norm, k)
                if s > best_score:
                    best_key, best_score = k, s
            if best_key and best_score >= args.fuzzy_thresh:
                cands = idx[best_key]
                best_any = max(cands, key=lambda t: (t[0] or -1))
                matched_any[norm] = best_any
                for y, e, _ in cands:
                    if y is not None and y >= args.since:
                        written_entries.append(e)

        unmatched_norms = [n for n in unmatched_norms if n not in matched_any]

    # Deduplicate written entries by citation key
    seen_keys = set()
    unique_entries = []
    for e in written_entries:
        key = e.get("ID") or e.get("id") or e.get("citation_key")
        if key and key in seen_keys:
            continue
        seen_keys.add(key)
        unique_entries.append(e)

    # Write output
    out_db = BibDatabase()
    out_db.entries = unique_entries

    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"% since>={args.since}; total_titles={len(ordered_norms)}; "
        f"matched_any={len(matched_any)}; written={len(unique_entries)}; "
        f"unmatched={len(unmatched_norms)}\n\n"
    )
    out_path.write_text(header + writer.write(out_db), encoding="utf-8")

    print(f"Wrote {len(unique_entries)} entries (year >= {args.since}) -> {out_path}")
    if unmatched_norms:
        print("Unmatched titles (string mismatch only), showing up to 10:")
        for norm in unmatched_norms[:10]:
            print(f"  - {norm_to_raws.get(norm, [norm])[0]}")

if __name__ == "__main__":
    main()