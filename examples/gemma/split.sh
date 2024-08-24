#!/bin/sh

source examples/gemma/.env/bin/activate

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/OpenOrca/

python examples/gemma/split_parquest_jsonl.py \
  --input_files ${DATASET_DIR}/1M-GPT4-Augmented.parquet \
  --output_prefix  ${DATASET_DIR}/split_1M
