#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import tqdm

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


def normalize_title(s: str) -> str:
    # strip simple LaTeX, braces, collapse ws, lowercase
    s = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', ' ', s)  # \LaTeX{...}, \url, \textbf, etc.
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def load_titles(path: Path):
    titles_raw, titles_norm = set(), set()
    pat = re.compile(r'"\s*(.*?)\s*"')
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if not m:
            continue
        t = m.group(1).strip()
        titles_raw.add(t)
        titles_norm.add(normalize_title(t))
    return titles_raw, titles_norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles-file", default="./output/filtered_titles.txt")
    ap.add_argument("--bib-file", default="./data/anthology.bib")
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--out", default="./output/filtered_bibtex.bib")
    args = ap.parse_args()

    titles_raw, titles_norm = load_titles(Path(args.titles_file))

    with Path(args.bib_file).open("r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)

    # index by normalized title; keep most recent year on dup titles
    idx = {}
    for entry in tqdm.tqdm(bib_db.entries):
        title = entry.get("title")
        if not title:
            continue
        year = entry.get("year")
        y = None
        if year:
            m = re.search(r'(\d{4})', year)
            if m:
                y = int(m.group(1))
        key = normalize_title(title)
        prev = idx.get(key)
        if not prev or ((y or -1) > (prev[1] or -1)):
            idx[key] = (entry, y)

    matched = []
    for tn in titles_norm:
        hit = idx.get(tn)
        if hit:
            entry, y = hit
            if y is not None and y >= args.since:
                matched.append(entry)

    out_db = BibDatabase()
    out_db.entries = matched

    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"% since>={args.since}; matches={len(matched)}\n\n", encoding="utf-8")
    with out_path.open("a", encoding="utf-8") as f:
        f.write(writer.write(out_db))

    print(f"Collected {len(matched)} entries (year >= {args.since}) -> {out_path}")

if __name__ == "__main__":
    main()