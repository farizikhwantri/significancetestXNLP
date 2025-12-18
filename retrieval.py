#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict], Dict]:
    p = Path(index_dir)
    index = faiss.read_index(str(p / "vectors.faiss"))
    metas: List[Dict] = []
    with open(p / "meta.jsonl", "r", encoding="utf-8") as mf:
        for line in mf:
            metas.append(json.loads(line))
    with open(p / "config.json", "r", encoding="utf-8") as cf:
        cfg = json.load(cf)
    return index, metas, cfg


def select_model_id(choice: Optional[str], cfg: Dict) -> str:
    if choice:
        c = choice.lower()
        if c in ["specter2", "allenai/specter2_base"]:
            return "allenai/specter2_base"
        if c in ["scibert", "allenai/scibert_scivocab_uncased"]:
            return "allenai/scibert_scivocab_uncased"
    return cfg.get("model_id", "allenai/specter2_base")


def encode_query(query: str, model_id: str) -> np.ndarray:
    model = SentenceTransformer(model_id)
    q = model.encode([query], normalize_embeddings=True)
    return np.asarray(q, dtype=np.float32)


def search(index_dir: str, query: str, topk: int = 5, model_override: Optional[str] = None) -> List[Dict]:
    index, metas, cfg = load_index(index_dir)
    model_id = select_model_id(model_override, cfg)
    qvec = encode_query(query, model_id)
    scores, idxs = index.search(qvec, topk)
    results: List[Dict] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:
            continue
        m = metas[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "title": m.get("title"),
            "abstract": m.get("abstract"),
            "keywords": m.get("keywords"),
            "citation_key": m.get("citation_key"),
            "url": m.get("url"),
            "doi": m.get("doi"),
        })
    return results


def rag(index_dir: str, query: str, topk: int = 3, model_override: Optional[str] = None) -> Dict:
    hits = search(index_dir, query, topk=topk, model_override=model_override)
    contexts = []
    for h in hits:
        title = h.get("title") or ""
        abstract = h.get("abstract") or ""
        snippet = (title + "\n\n" + abstract).strip()
        contexts.append({
            "rank": h["rank"],
            "score": h["score"],
            "context": snippet,
            "metadata": {
                "citation_key": h.get("citation_key"),
                "url": h.get("url"),
                "doi": h.get("doi"),
                "title": title,
                "keywords": h.get("keywords"),
            }
        })
    return {"query": query, "topk": topk, "contexts": contexts}


def main():
    ap = argparse.ArgumentParser(description="Semantic search and RAG over saved BibTeX index.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Semantic search")
    s.add_argument("--index", required=True, help="Index directory")
    s.add_argument("--query", required=True, help="Search query (topics/keywords)")
    s.add_argument("--topk", type=int, default=5)
    s.add_argument("--model", default=None, help="Optional override: specter2 | scibert")

    r = sub.add_parser("rag", help="RAG-style retrieval (contexts only)")
    r.add_argument("--index", required=True, help="Index directory")
    r.add_argument("--query", required=True, help="Query")
    r.add_argument("--topk", type=int, default=3)
    r.add_argument("--model", default=None, help="Optional override: specter2 | scibert")

    args = ap.parse_args()

    if args.cmd == "search":
        results = search(args.index, args.query, topk=args.topk, model_override=args.model)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.cmd == "rag":
        out = rag(args.index, args.query, topk=args.topk, model_override=args.model)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()