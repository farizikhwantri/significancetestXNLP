#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

from llm_sikt_chat import gpt_request

XAI_EVAL_CLASSIFY_PROMPT = """You are an NLP meta-reviewer. Decide if a paper is about explainable/interpretable AI and evaluates explanations.
Use only title and abstract. Consider evaluation signals like perturbation/deletion/insertion tests, sensitivity analysis, faithfulness/comprehensiveness/sufficiency, human agreement/evaluation (e.g., Cohen’s/Fleiss’ κ), utility or simulatability.

Input:
Title: {{title}}
Abstract: {{abstract}}

Return JSON only:
{
  "is_xai_interpretable": true|false,
  "has_explanation_evaluation": true|false,
  "evidence": ["verbatim phrases from abstract"],
  "confidence": 0.0-1.0
}
"""

SIG_TEST_PROMPT = """You are an NLP meta-reviewer. For a paper that evaluates explanations, determine if it reports statistical significance testing.
Look for mentions like “statistically significant”, p-values, t-test, Wilcoxon, Mann–Whitney, ANOVA, Kruskal–Wallis, Bonferroni/Holm/FDR, bootstrap, permutation tests, confidence intervals.

Input:
Title: {{title}}
Abstract: {{abstract}}

Return JSON only:
{
  "has_significance_testing": true|false,
  "tests": ["e.g., paired t-test, Wilcoxon", "..."],
  "p_values": ["e.g., p<0.05", "..."],
  "evidence": ["verbatim phrases from abstract"],
  "confidence": 0.0-1.0
}
"""

SYS_JSON_PROMPT = "You are a careful assistant. Return compact, valid JSON only. Do not add prose."

def normalize_tex(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', ' ', s)
    s = re.sub(r'\\([#\$%&_{}~^\\])', r'\1', s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fill_template(tpl: str, title: str, abstract: str) -> str:
    return tpl.replace("{{title}}", title).replace("{{abstract}}", abstract or "(no abstract available)")

def sanitize_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try best-effort extraction
    s = s.strip()
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}") + 1
        try:
            return json.loads(s[start:end])
        except Exception:
            return None
    return None

def call_chat_once(session_id: str, sys_prompt: str, user_prompt: str,
                   model_id: str = "gpt-4o", model_name: str = "gpt-4o",
                   temperature: float = 0.0, max_length: int = 24000, token_limit: int = 8000) -> str:
    data = gpt_request(
        session_id=session_id,
        message=user_prompt,
        prompt=sys_prompt,
        temperature=temperature,
        model_id=model_id,
        model_name=model_name,
        max_length=max_length,
        token_limit=token_limit,
        chatmode=False
    )
    # Extract last assistant message
    msgs = data.get("messages", [])
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    return assistants[-1]["content"] if assistants else ""

def classify_entry(session_id: str, title_raw: str, abstract_raw: str,
                   model_id: str, model_name: str) -> Dict[str, Any]:
    title = normalize_tex(title_raw)
    abstract = normalize_tex(abstract_raw)
    user_prompt = fill_template(XAI_EVAL_CLASSIFY_PROMPT, title, abstract)
    resp_text = call_chat_once(session_id, SYS_JSON_PROMPT, user_prompt, model_id=model_id, model_name=model_name)
    parsed = sanitize_json(resp_text) or {}
    return {
        "raw_response": resp_text,
        "parsed": parsed
    }

def check_significance(session_id: str, title_raw: str, abstract_raw: str,
                       model_id: str, model_name: str) -> Dict[str, Any]:
    title = normalize_tex(title_raw)
    abstract = normalize_tex(abstract_raw)
    user_prompt = fill_template(SIG_TEST_PROMPT, title, abstract)
    resp_text = call_chat_once(session_id, SYS_JSON_PROMPT, user_prompt, model_id=model_id, model_name=model_name)
    parsed = sanitize_json(resp_text) or {}
    return {
        "raw_response": resp_text,
        "parsed": parsed
    }

def write_filtered_bib(entries: list, out_path: Path):
    db = BibDatabase()
    db.entries = entries
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None
    out_path.write_text(writer.write(db), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="LLM-based XAI explanation-eval classification over BibTeX via Sikt chat API.")
    ap.add_argument("--bib", required=True, help="Path to input .bib")
    ap.add_argument("--session-id", required=True, help="Sikt session cookie value")
    ap.add_argument("--model-id", default="gpt-4o", help="Model ID")
    ap.add_argument("--model-name", default="gpt-4o", help="Model name")
    ap.add_argument("--out-jsonl", default="./output/xai_llm_results.jsonl", help="Output JSONL")
    ap.add_argument("--export-bib", default=None, help="Optional path to write filtered .bib (XAI+Eval)")
    args = ap.parse_args()

    with Path(args.bib).open("r", encoding="utf-8") as f:
        db = bibtexparser.load(f)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_entries = []

    with out_path.open("w", encoding="utf-8") as outf:
        for e in db.entries:
            pid = e.get("ID", "")
            title = e.get("title", "") or ""
            abstract = e.get("abstract", "") or ""

            cls = classify_entry(args.session_id, title, abstract, args.model_id, args.model_name)
            parsed_cls = cls.get("parsed", {})
            is_xai = bool(parsed_cls.get("is_xai_interpretable"))
            has_eval = bool(parsed_cls.get("has_explanation_evaluation"))

            sig = {}
            parsed_sig = {}
            if is_xai and has_eval:
                sig = check_significance(args.session_id, title, abstract, args.model_id, args.model_name)
                parsed_sig = sig.get("parsed", {})

            row = {
                "id": pid,
                "title": title,
                "year": e.get("year"),
                "is_xai_interpretable": is_xai,
                "has_explanation_evaluation": has_eval,
                "classification_raw": cls.get("raw_response", ""),
                "classification_parsed": parsed_cls,
                "has_significance_testing": bool(parsed_sig.get("has_significance_testing")),
                "significance_raw": sig.get("raw_response", ""),
                "significance_parsed": parsed_sig,
            }
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

            if is_xai and has_eval and args.export_bib:
                kept_entries.append(e)

    if args.export_bib:
        write_filtered_bib(kept_entries, Path(args.export_bib))
        print(f"Wrote filtered .bib -> {args.export_bib}")

    print(f"Wrote JSONL -> {out_path}")

if __name__ == "__main__":
    main()