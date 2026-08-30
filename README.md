# significancetestXNLP

This project implements a multi-stage review pipeline for identifying XAI papers that evaluate explanations and checking whether they report statistical significance testing.

## Pipeline overview

Stage 1: Initial filtering

- Uses `filter.sh`
- Performs the first pass filtering of candidate papers

Stage 2: LLM-based classification

- Uses `script/llm_classify_xai_eval.py`
- Uses `script/llm_classify_pdf_sig.py`
- Identifies:
  - papers about XAI / interpretable AI
  - papers that evaluate explanations
  - papers that report significance testing

Stage 3 and 4: Conference filtering

- Uses `filter_mainconf.py`
- Applies conference-level filtering logic after the LLM stage

Stage 5: Manual review

- Final output of systematic review is done manually

## Supporting tools

These are not part of the main filtering pipeline itself, but support the review workflow:

- `acl_paper_crawler.py`
  - downloads papers from ACL Anthology

- `script/sample_by_group.py`
  - takes a sample from Stage 2 results for reannotation by the senior author
  - used to create a balanced set for manual agreement checks

- Agreement-related scripts / analyses
  - compute human-LLM or human-human agreement metrics
  - not part of the paper selection pipeline itself

## Repository layout

- `filter.sh`
  - Stage 1 filtering

- `filter_mainconf.py`
  - Stage 3/4 conference filtering

- `script/llm_classify_xai_eval.py`
  - LLM classification of XAI / explanation evaluation

- `script/llm_classify_pdf_sig.py`
  - PDF-based LLM classification for significance testing

- `script/sample_by_group.py`
  - sampling for reannotation and agreement review

- `acl_paper_crawler.py`
  - supporting paper acquisition

## Main review flow

1. Run Stage 1 filtering with `filter.sh`
2. Run Stage 2 LLM classification:
   - XAI evaluation detection
   - significance testing detection
3. Apply Stage 3/4 conference filtering with `filter_mainconf.py`
4. Perform Stage 5 manual review
5. Use sampled reannotation sets for senior-author review and agreement measurement

## Notes

- `sample_by_group.py` is not a primary pipeline stage.
- It is a support tool for reannotation sampling.
- Agreement analysis is separate from the paper selection pipeline and is used to measure annotation quality and consistency.
