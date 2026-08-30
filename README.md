<!-- filepath: README.md -->
# significancetestXNLP

This project implements a multi-stage review pipeline for identifying XAI papers that evaluate explanations and checking whether they report statistical significance testing.

## Pipeline overview

Stage 1: Initial regex-based filtering

- Uses filter.sh
- Reads the ACL Anthology BibTeX file
- Extracts title lines from the title field
- Applies regex-based keyword matching to retain likely XAI / explanation-related papers
- This is a broad first-pass filter, not a final decision stage

Stage 2: LLM-based classification

- Uses llm_classify_xai_eval.py
- Uses title + abstract to classify each paper as:
  - `is_xai_interpretable`
  - `has_explanation_evaluation`
- If both are true, it runs a second LLM check for significance testing using the same abstract/title input
- Writes a JSONL result file and can export a filtered BibTeX file

Stage 2b: PDF-based significance screening

- Uses llm_classify_pdf_sig.py
- Resolves selected papers to PDFs
- Extracts snippets from PDF text, tables, and OCR
- Checks whether the paper reports significance testing using PDF evidence
- This complements the abstract-based significance test detection
- This is optional, does not work accurately since the LLMs cannot differentiate significant testing about just performance evaluation or really explanation evaluation using GPT-4o version.
- Note: Using better frontier model version/agentic AI with RAG capability might achieve better results

Stage 3: Venue / conference categorisation and filtering

- Uses filter_mainconf.py
- Reads the booktitle and series fields from BibTeX entries
- Uses Python regex matching to categorise the venue into target conference groups
- Keeps only main venues such as ACL, NAACL, EMNLP, EACL, COLING, LREC, and Findings
- Removes workshop papers, demos, tutorials, shared tasks, student workshops, and other non-main tracks

Stage 4: Additional conference filtering

- Continues the venue-focused filtering in filter_mainconf.py
- Refines the set to the final intended conference pool

Stage 5: Manual review

- Final systematic-review decision is performed manually

## Supporting tools

These are not part of the main filtering pipeline itself, but support the workflow:

- acl_paper_crawler.py
  - downloads papers from ACL Anthology

- sample_by_group.py
  - samples from Stage 2 results for reannotation by the senior author
  - used to build a balanced subset for adjudication and agreement checks

- Agreement-related scripts / analyses
  - compute human-LLM or human-human agreement metrics
  - separate from the main selection pipeline

## Repository layout

- filter.sh
  - Stage 1 regex-based title filtering

- filter_mainconf.py
  - Stage 3/4 venue extraction and workshop filtering

- llm_classify_xai_eval.py
  - LLM classification of XAI / explanation evaluation

- llm_classify_pdf_sig.py
  - PDF-based LLM classification for significance testing

- sample_by_group.py
  - reannotation sampling for senior-author review and agreement work

- acl_paper_crawler.py
  - paper acquisition support

## Main review flow

1. Run Stage 1 filtering with filter.sh
   - regex-based title screening over ACL Anthology entries
2. Run Stage 2 LLM classification
   - detect XAI / interpretable AI papers
   - detect explanation evaluation papers
   - check significance testing from abstract-level evidence
3. Run Stage 2b PDF significance screening
   - use PDF snippets to confirm significance testing
4. Run Stage 3 venue categorisation with filter_mainconf.py
   - extract booktitle
   - classify venue
   - remove workshop/non-main tracks
5. Run Stage 4 additional conference filtering
6. Perform Stage 5 manual review
7. Use sampled reannotation sets for senior-author review and agreement measurement

## Notes

- sample_by_group.py is not a primary pipeline stage.
- It is a supporting tool for reannotation sampling.
- Agreement analysis is separate from the paper selection pipeline and is used to measure annotation quality and consistency.
- Stage 1 is intentionally broad and regex-driven.
- Stage 2 is the main decision stage using LLMs.
- Stage 3 is a structured venue filter based on booktitle / series, with explicit exclusion of workshop tracks.
