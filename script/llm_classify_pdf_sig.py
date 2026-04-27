#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pdfplumber
import bibtexparser

# PDF OCR (optional)
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Import crawler and Sikt chat API
from acl_paper_crawler import ACLPaperCrawler
from llm_sikt_chat import gpt_request

# ------------- Prompts -------------
SYS_JSON_PROMPT = "You are a careful assistant. Return compact, valid JSON only. Do not add prose."

SIG_TEST_PROMPT_PDF = """You are an NLP meta-reviewer. Using the provided PDF snippets (text, tables, OCR), determine if the paper reports statistical significance testing of evaluation results.

Look for mentions such as: “statistically significant”, p-values, t-test, Wilcoxon, Mann–Whitney, ANOVA, Kruskal–Wallis, Bonferroni/Holm/FDR, bootstrap, permutation tests, confidence intervals.

Input:
Title: {{title}}

PDF evidence (snippets with page refs):
{{pdf_snippets}}

Return JSON only:
{
  "has_significance_testing": true|false,
  "tests": ["e.g., paired t-test, Wilcoxon", "..."],
  "p_values": ["e.g., p<0.05", "..."],
  "corrections": ["e.g., Holm, FDR", "..."],
  "confidence_intervals": ["e.g., 95% CI", "..."],
  "evidence": ["verbatim phrases from snippets"],
  "confidence": 0.0-1.0
}
"""

# ------------- Helpers -------------
SECTION_HINTS = re.compile(r"\b(results?|evaluation|experiments?|analysis|metrics?)\b", re.IGNORECASE)
PVAL_RE = re.compile(r"\bp(?:[-\s]?value)?\s*(?:[=:]|<|≤|>|≥)\s*(?:\d*\.?\d+(?:e[-+]?\d+)?|0\.\d+|\.\d+)\b", re.IGNORECASE)
SIG_PHRASES_RE = re.compile(r"\b(statistically\s+significant|significance\s+test(?:ing)?)\b", re.IGNORECASE)
TEST_RE = re.compile(r"\b(t[\-\s]?test|student'?s\s+t[\-\s]?test|welch'?s\s+t[\-\s]?test|wilcoxon|mann[\-\u2013]\s*whitney|anova|kruskal[\-\u2013]\s*wallis|chi[\-\s]?square|χ2|fisher'?s?\s+exact|bootstrap|permutation|randomization)\b", re.IGNORECASE)
CORR_RE = re.compile(r"\b(bonferroni|holm|benjamini[\-\u2013]\s*hochberg|fdr|tukey(?:'s)?\s+hsd)\b", re.IGNORECASE)
CI_RE = re.compile(r"\b\d{1,3}\s*%\s*(?:CI|confidence\s*intervals?)\b", re.IGNORECASE)

def normalize_tex(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', ' ', s)
    s = re.sub(r'\\([#\$%&_{}~^\\])', r'\1', s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fill_template(tpl: str, title: str, pdf_snips: str) -> str:
    return tpl.replace("{{title}}", title).replace("{{pdf_snippets}}", pdf_snips)

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

# ------------- PDF extraction -------------
def extract_text_blocks(pdf_path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            if txt:
                out.append((i + 1, txt))
    return out

def extract_tables_text(pdf_path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }) or []
            for t in tables:
                rows = []
                for row in t:
                    cells = [c for c in row if c is not None]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    out.append((i + 1, "\n".join(rows)))
    return out

def extract_images_ocr(pdf_path: Path, max_images_per_page: int = 6) -> List[Tuple[int, str]]:
    if not OCR_AVAILABLE:
        return []
    out: List[Tuple[int, str]] = []
    doc = fitz.open(str(pdf_path))
    for page_index in range(len(doc)):
        page = doc[page_index]
        imgs = page.get_images(full=True)
        count = 0
        for img in imgs:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_txt = pytesseract.image_to_string(img_pil)
                ocr_txt = re.sub(r"\s+", " ", ocr_txt).strip()
                if ocr_txt:
                    out.append((page_index + 1, ocr_txt))
                count += 1
                if count >= max_images_per_page:
                    break
            except Exception:
                continue
    doc.close()
    return out

def score_block(txt: str) -> int:
    score = 0
    if SECTION_HINTS.search(txt):
        score += 1
    if PVAL_RE.search(txt) or SIG_PHRASES_RE.search(txt):
        score += 1
    if TEST_RE.search(txt):
        score += 1
    return score

def make_pdf_snippets(pdf_path: Path, max_snippets: int = 8, do_ocr: bool = True, max_chars: int = 8000) -> str:
    blocks = []
    for page_no, txt in extract_text_blocks(pdf_path):
        blocks.append(("text", page_no, txt))
    for page_no, txt in extract_tables_text(pdf_path):
        blocks.append(("table", page_no, txt))
    if do_ocr:
        for page_no, txt in extract_images_ocr(pdf_path):
            blocks.append(("ocr", page_no, txt))

    # Rank by heuristic score and length
    ranked = sorted(blocks, key=lambda b: (score_block(b[2]), len(b[2]) > 200), reverse=True)
    snippets: List[str] = []
    total_chars = 0
    for src, pg, txt in ranked[: max_snippets * 3]:  # consider more, then trim by char budget
        snippet = txt.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + " ..."
        s = f"[{src} p.{pg}] {snippet}"
        if total_chars + len(s) <= max_chars:
            snippets.append(s)
            total_chars += len(s)
        if len(snippets) >= max_snippets:
            break

    return "\n".join(snippets) if snippets else "(no PDF snippets extracted)"

# ------------- CSV/XLSX I/O -------------
def load_input_table(path: str, id_col: str = "id", title_col: str = "title") -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".xlsx", ".xls"]:
        # openpyxl via pandas preserves Unicode
        df = pd.read_excel(path, sheet_name=0, dtype={id_col: str, title_col: str}, keep_default_na=False, engine="openpyxl")
    else:
        df = pd.read_csv(path, dtype={id_col: str, title_col: str}, keep_default_na=False, encoding="utf-8-sig")
    for col in [id_col, title_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    df[id_col] = df[id_col].astype(str)
    df[title_col] = df[title_col].astype(str)
    return df

def write_output_table(df: pd.DataFrame, output_path: str) -> None:
    ext = os.path.splitext(output_path.lower())[1]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        with pd.ExcelWriter(output_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="sigtest_pdf")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ------------- BibTeX and PDF resolution -------------
def load_bib_index(bib_paths: List[str]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for p in bib_paths:
        if not os.path.exists(p):
            print(f"Warning: bib file not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as fh:
            db = bibtexparser.load(fh)
            for entry in db.entries:
                eid = entry.get("ID")
                if eid:
                    index[eid] = entry
    return index

def resolve_and_download_pdf(crawler: ACLPaperCrawler, entry: Dict[str, Any], pdf_dir: Path) -> Optional[Path]:
    """
    Use ACLPaperCrawler helpers to resolve anthology ID from URL/DOI and download the PDF.
    Returns local path if available.
    """
    url = entry.get("url", "")
    doi = entry.get("doi", "")
    title = normalize_tex(entry.get("title", "") or "")

    anthology_id = None
    if url:
        anthology_id = crawler.extract_anthology_id_from_url(url)
    if not anthology_id and doi:
        anthology_id = crawler.extract_anthology_id_from_doi(doi)
    if not anthology_id:
        return None

    # Download; ACLPaperCrawler constructs filename internally
    ok = crawler.download_paper(anthology_id, title=title if title else None)
    if not ok:
        # Check if file already exists due to earlier runs
        fname_guess = f"{anthology_id}.pdf"
        files = list(pdf_dir.glob(f"{anthology_id}_*.pdf")) + [pdf_dir / fname_guess]
        for fp in files:
            if fp.exists():
                return fp
        return None

    # Find saved file
    candidates = list(pdf_dir.glob(f"{anthology_id}_*.pdf")) + [pdf_dir / f"{anthology_id}.pdf"]
    for fp in candidates:
        if fp.exists():
            return fp
    return None

# ------------- LLM call -------------
def classify_sig_from_pdf(session_id: str, title: str, pdf_snippets: str,
                          model_id: str, model_name: str) -> Dict[str, Any]:
    user_prompt = fill_template(SIG_TEST_PROMPT_PDF, normalize_tex(title), pdf_snippets)
    resp = gpt_request(
        session_id=session_id,
        message=user_prompt,
        prompt=SYS_JSON_PROMPT,
        temperature=0.0,
        model_id=model_id,
        model_name=model_name,
        max_length=24000,
        token_limit=8000,
        chatmode=False
    )
    msgs = resp.get("messages", [])
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    raw = assistants[-1]["content"] if assistants else ""
    parsed = sanitize_json(raw) or {}
    return {"raw": raw, "parsed": parsed}

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser(description="PDF-based significance testing detection via Sikt chat, using ACLPaperCrawler to fetch PDFs.")
    ap.add_argument("--csv", required=True, help="Input CSV/XLSX with at least columns: id,title,has_explanation_evaluation")
    ap.add_argument("--bibfiles", nargs="+", required=True, help="BibTeX files to map id -> URL/DOI/title")
    ap.add_argument("--pdf-dir", required=True, help="Directory to store/download PDFs (from ACL Anthology)")
    ap.add_argument("--session-id", required=True, help="Sikt session cookie value")
    ap.add_argument("--model-id", default="gpt-4o", help="Model ID for Sikt chat")
    ap.add_argument("--model-name", default="gpt-4o", help="Model name for Sikt chat")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR on page images")
    ap.add_argument("--max-snippets", type=int, default=8, help="Max PDF snippets to include in prompt")
    ap.add_argument("-o", "--output", required=True, help="Output path (.csv or .xlsx)")
    ap.add_argument("--evidence-jsonl", default=None, help="Optional JSONL path for evidence records")
    args = ap.parse_args()

    df = load_input_table(args.csv, id_col="id", title_col="title")
    if "has_explanation_evaluation" not in df.columns:
        raise SystemExit("Input table must contain 'has_explanation_evaluation' column.")

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    bib_index = load_bib_index(args.bibfiles)
    crawler = ACLPaperCrawler(output_dir=str(pdf_dir), delay=1.0)

    rows_out: List[Dict[str, Any]] = []
    evidence_records: List[Dict[str, Any]] = []

    # Process only papers that evaluate explanations
    df_eval = df[df["has_explanation_evaluation"].astype(str).str.lower().isin(["true", "1", "yes"])]

    for _, row in df_eval.iterrows():
        pid = str(row["id"]).strip()
        title = str(row["title"]).strip()
        bib = bib_index.get(pid)

        pdf_path: Optional[Path] = None
        if bib:
            pdf_path = resolve_and_download_pdf(crawler, bib, pdf_dir)

        pdf_snips = "(no PDF found)"
        if pdf_path and pdf_path.exists():
            try:
                pdf_snips = make_pdf_snippets(
                    pdf_path,
                    max_snippets=args.max_snippets,
                    do_ocr=(not args.no_ocr),
                    max_chars=8000
                )
            except Exception as e:
                pdf_snips = f"(error extracting PDF: {e})"

        # Call LLM for classification using PDF evidence
        resp = classify_sig_from_pdf(args.session_id, title, pdf_snips, args.model_id, args.model_name)
        parsed = resp.get("parsed", {})
        has_sig = bool(parsed.get("has_significance_testing"))

        rows_out.append({
            "id": pid,
            "title": title,
            "has_explanation_evaluation": True,
            "has_significance_testing": has_sig,
            "tests": "; ".join(parsed.get("tests", []) or []),
            "p_values": "; ".join(parsed.get("p_values", []) or []),
            "corrections": "; ".join(parsed.get("corrections", []) or []),
            "confidence_intervals": "; ".join(parsed.get("confidence_intervals", []) or []),
            "confidence": parsed.get("confidence", None),
        })

        evidence_records.append({
            "id": pid,
            "title": title,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "llm_raw": resp.get("raw", ""),
            "llm_parsed": parsed,
            "pdf_snippets": pdf_snips,
        })

    # Merge back for non-eval rows (optional, mark False)
    non_eval_ids = set(df["id"]) - set(df_eval["id"])
    for pid in non_eval_ids:
        title = str(df[df["id"] == pid]["title"].iloc[0])
        rows_out.append({
            "id": pid,
            "title": title,
            "has_explanation_evaluation": False,
            "has_significance_testing": False,
            "tests": "",
            "p_values": "",
            "corrections": "",
            "confidence_intervals": "",
            "confidence": None,
        })

    out_df = pd.DataFrame(rows_out)
    # Preserve Unicode in XLSX; UTF-8 BOM for CSV
    write_output_table(out_df, args.output)

    if args.evidence_jsonl:
        ev_path = Path(args.evidence_jsonl)
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        with ev_path.open("w", encoding="utf-8") as f:
            for rec in evidence_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_df)} rows -> {args.output}")
    if args.evidence_jsonl:
        print(f"Wrote evidence -> {args.evidence_jsonl}")

if __name__ == "__main__":
    main()