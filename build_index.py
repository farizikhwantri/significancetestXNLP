#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Iterable
from tqdm import tqdm

import numpy as np
import faiss
import bibtexparser
from bibtexparser.bparser import BibTexParser
import torch
# from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


def _split_bibtex_entries(raw: str) -> List[str]:
    # Split by entry starts like "@inproceedings{", "@article{", etc.
    parts = []
    current = []
    depth = 0
    i = 0
    while i < len(raw):
        if raw[i] == '@' and depth == 0:
            # start new entry
            if current:
                parts.append(''.join(current).strip())
                current = []
        current.append(raw[i])
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth = max(0, depth - 1)
        i += 1
    if current:
        parts.append(''.join(current).strip())
    # remove any non-entry noise
    return [p for p in parts if p.startswith('@')]


def parse_bibtex_chunk(chunk_texts: List[str]) -> List[Dict]:
    parser = BibTexParser(common_strings=True)
    # bibtexparser can parse concatenated entries
    db = bibtexparser.loads('\n\n'.join(chunk_texts), parser=parser)
    out = []
    for e in db.entries:
        title = (e.get("title") or "").strip().replace("\n", " ")
        abstract = (e.get("abstract") or "").strip().replace("\n", " ")
        keywords = (e.get("keywords") or "").strip().replace("\n", " ")
        url = (e.get("url") or "").strip()
        doi = (e.get("doi") or "").strip()
        year = (e.get("year") or "").strip()
        citation_key = e.get("ID") or e.get("id") or title[:64]
        if not title and not abstract:
            continue
        out.append({
            "citation_key": citation_key,
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "url": url,
            "doi": doi,
            "year": year,
        })
    return out


def iter_bibtex_entries_in_chunks(bib_path: str, chunk_size: int = 500) -> Iterable[List[Dict]]:
    with open(bib_path, "r", encoding="utf-8") as f:
        raw = f.read()
    entries_text = _split_bibtex_entries(raw)
    for i in range(0, len(entries_text), chunk_size):
        chunk_texts = entries_text[i:i+chunk_size]
        yield parse_bibtex_chunk(chunk_texts)


def select_model_id(choice: str) -> str:
    c = (choice or "specter2").lower()
    if c in ["specter2", "sentence-transformers/allenai-specter"]:
        return "sentence-transformers/allenai-specter"
    if c in ["scibert", "allenai/scibert_scivocab_uncased"]:
        return "allenai/scibert_scivocab_uncased"
    return "sentence-transformers/allenai-specter"
def build_texts(entries: List[Dict]) -> List[str]:
    texts = []
    for e in entries:
        title = e.get("title", "")
        abstract = e.get("abstract", "")
        kw = e.get("keywords", "")
        parts = [title, abstract]
        if kw:
            parts.append(f"Keywords: {kw}")
        text = " \n\n".join([p for p in parts if p])
        texts.append(text if text else title)
    return texts


def build_index(bib_path: str, model_choice: str, out_dir: str, chunk_size: int = 500):
    model_id = select_model_id(model_choice)
    print(f"Loading model: {model_id}")
    # model = AutoModel.from_pretrained(model_id)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SentenceTransformer(model_id)

    all_metas: List[Dict] = []
    all_vecs: List[np.ndarray] = []

    total_entries = 0
    for chunk in tqdm(iter_bibtex_entries_in_chunks(bib_path, chunk_size=chunk_size), desc="Parsing chunks"):
        if not chunk:
            continue
        texts = build_texts(chunk)
        total_entries += len(texts)
        embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = model(**inputs)
        #     embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        vecs = np.asarray(embs, dtype=np.float32)
        all_vecs.append(vecs)
        all_metas.extend(chunk)

    if not all_metas:
        raise SystemExit("No valid entries (need title or abstract).")

    vecs_concat = np.concatenate(all_vecs, axis=0)
    dim = vecs_concat.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via inner product on normalized vectors
    index.add(vecs_concat)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "vectors.faiss"))
    with open(out / "meta.jsonl", "w", encoding="utf-8") as mf:
        for e in all_metas:
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "config.json", "w", encoding="utf-8") as cf:
        json.dump(
            {
                "model_id": model_id,
                "source_bib": bib_path,
                "count": len(all_metas),
                "normalized": True,
                "similarity": "cosine (inner product)",
                "chunk_size": chunk_size,
            },
            cf,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Indexed {len(all_metas)} entries. Index saved to: {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from BibTeX (SciBERT/SPECTER2) with chunked parsing/embedding.")
    ap.add_argument("--bib", required=True, help="Path to BibTeX file")
    ap.add_argument("--model", default="specter2", help="Model: specter2 | scibert")
    ap.add_argument("--out", required=True, help="Output directory for index artifacts")
    ap.add_argument("--chunk-size", type=int, default=500, help="Number of BibTeX entries per parse/encode chunk")
    args = ap.parse_args()
    # add if keyboard interrupt handling needed
    try:
        build_index(args.bib, args.model, args.out, chunk_size=args.chunk_size)
    except KeyboardInterrupt:
        print("\nIndexing interrupted by user. Partial index may have been created.")
        # Optionally, clean up partial output here
        # For now, we just exit
        raise SystemExit(1)


if __name__ == "__main__":
    main()