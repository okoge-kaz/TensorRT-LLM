#!/bin/sh

python examples/gemma/code_syntax/lint_code.py \
  --file-path /gs/bs/tga-bayes-crest/Swallow/raw/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-00.jsonl \
  --output-path /gs/bs/tga-NII-LLM/datasets/raw/pretrain/stack-v2/the-stack-v2-train-smol-ids-00-lint.jsonl
