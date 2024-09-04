#!/bin/sh

source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-top1-en-chat-sft/data

python examples/gemma/dataset/split_parquest_jsonl.py\
  --input_files ${DATASET_DIR}/train-00000-of-00001.parquet \
  --output_prefix  ${DATASET_DIR}/split

# python examples/gemma/dataset/split_parquest_jsonl.py \
#   --input_files ${DATASET_DIR}/train-00001-of-00002.parquet \
#   --output_prefix  ${DATASET_DIR}/split
