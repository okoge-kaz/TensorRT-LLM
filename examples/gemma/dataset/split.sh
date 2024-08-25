#!/bin/sh

source examples/gemma/.env/bin/activate

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data

python examples/gemma/dataset/split_parquest_jsonl.py\
  --input_files ${DATASET_DIR}/train-00000-of-00002.parquet \
  --output_prefix  ${DATASET_DIR}/split

python examples/gemma/dataset/split_parquest_jsonl.py \
  --input_files ${DATASET_DIR}/train-00001-of-00002.parquet \
  --output_prefix  ${DATASET_DIR}/split
