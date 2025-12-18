#!/bin/bash

input_file="./data/anthology.bib"
output_file="./output/filtered_titles.txt"

# grep -n -i '^\s*title\s*=\s*"' "$input_file" \
# | rg -n -i -e '\b(xai|explain\w+|interpretable|interpret\w+|transparen\w+|faithful\w*|rationale?s?|rationali[sz]\w*|plausibilit\w*|post-?hoc|salien(?:cy|t)|feature\sattribution|counterfactual\w*|contrastive\s+explanation\w*|example-?based\s+explanation\w*|prototype\w*|critique\w*|attention\s+(?:as|is)\s+explanation|attention\s+weights?|rationale\s+extraction|highlight\w*|evidence\s+(?:extraction|rationale\w*)|explanation\s+generation|integrated\s+gradient\w*|lime|shap\b|lrp\b|grad-?cam|influence\s+function\w*|tracin|shapley\b|eraser|e-?snli|cos-?e|qed\b|ecqa|fever\b|hotpotqa|multirc)\b' \
# > "$output_file"

grep -n -i '^\s*title\s*=\s*"' "$input_file" \
| rg -n -i -e '\b(xai|explain\w+|interpretable|interpret\w+|transparen\w+|faithful\w*|rationale?s?|rationali[sz]\w*|plausibilit\w*|post-?hoc|salien(?:cy|t)|feature\sattribution|attribut\w*|counterfactual\w*|contrastive\s+explanation\w*|example-?based\s+explanation\w*|prototype\w*|critique\w*|attention\s+(?:as|is)\s+explanation|attention\s+weights?|rationale\s+extraction|highlight\w*|evidence\s+(?:extraction|rationale\w*)|explanation\s+generation|integrated\s+gradient\w*|lime|shap\b|lrp\b|grad-?cam|influence\s+function\w*|tracin|shapley\b|eraser|e-?snli|cos-?e|qed\b|ecqa|fever\b|hotpotqa|multirc)\b' \
> "$output_file"