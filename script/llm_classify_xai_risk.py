#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import bibtexparser
from llm_classify_xai_eval import normalize_tex, fill_template
from llm_classify_xai_eval import call_chat_once
from llm_classify_xai_eval import SYS_JSON_PROMPT

RISK_JSON_SCHEMA = """
Return JSON only:
{
  "not_annex_iii": true|false,
  "labels": [
    { "category": "string", "subcategory": "string", "confidence": 0.0-1.0 }
  ],
  "overall_confidence": 0.0-1.0,
  "evidence": ["verbatim phrases from abstract/title"]
}
"""

def sanitize_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        s = s.strip()
        if "{" in s and "}" in s:
            start = s.find("{")
            end = s.rfind("}") + 1
            try:
                return json.loads(s[start:end])
            except Exception:
                return None
    return None

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_user_prompt(risk_prompt_text: str, title: str, abstract: str) -> str:
    base = risk_prompt_text.strip()
    title_norm = normalize_tex(title)
    abstract_norm = normalize_tex(abstract or "(no abstract available)")
    return f"{base}\n\nInput:\nTitle: {title_norm}\nAbstract: {abstract_norm}"

def classify_annex_risk(session_id: str, risk_prompt_text: str, title: str, abstract: str,
                        model_id: str, model_name: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(risk_prompt_text, title, abstract)
    resp_text = call_chat_once(
        session_id=session_id,
        sys_prompt=SYS_JSON_PROMPT,
        user_prompt=user_prompt,
        model_id=model_id,
        model_name=model_name,
        temperature=0.0
    )
    parsed = sanitize_json(resp_text) or {}
    return {"raw_response": resp_text, "parsed": parsed}

def main():
    ap = argparse.ArgumentParser(description="Classify Annex III high-risk AI (via Sikt chat) using risk_prompt.")
    ap.add_argument("--bib", required=True, help="Path to input .bib")
    ap.add_argument("--session-id", required=True, help="Sikt session cookie value")
    ap.add_argument("--model-id", default="gpt-4o", help="Model ID")
    ap.add_argument("--model-name", default="gpt-4o", help="Model name")
    ap.add_argument("--risk-prompt", default="./config/risk_prompt.txt", help="Path to risk prompt template")
    ap.add_argument("--out-jsonl", default="./output/risk_results.jsonl", help="Output JSONL")
    args = ap.parse_args()

    risk_prompt_text = load_text(Path(args.risk_prompt))

    with Path(args.bib).open("r", encoding="utf-8") as f:
        db = bibtexparser.load(f)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as outf:
        for e in db.entries:
            pid = e.get("ID", "")
            title = e.get("title", "") or ""
            abstract = e.get("abstract", "") or ""

            res = classify_annex_risk(args.session_id, risk_prompt_text, title, abstract, args.model_id, args.model_name)
            parsed = res.get("parsed", {}) or {}

            labels = parsed.get("labels") or []
            annex_labels: List[Dict[str, Any]] = []
            for l in labels:
                if isinstance(l, dict) and l.get("category"):
                    annex_labels.append({
                        "category": l.get("category"),
                        "subcategory": l.get("subcategory"),
                        "confidence": l.get("confidence"),
                    })

            row = {
                "id": pid,
                "title": normalize_tex(title),
                "year": e.get("year"),
                "annex_not_annex_iii": parsed.get("not_annex_iii"),
                "annex_overall_confidence": parsed.get("overall_confidence"),
                "annex_labels": annex_labels,
                "annex_risk_parsed": parsed,
            }
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL -> {out_path}")

if __name__ == "__main__":
    main()