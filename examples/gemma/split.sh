#!/bin/sh

source examples/gemma/.env/bin/activate

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja/

python examples/gemma/split_dataset.py \
  --input_files ${DATASET_DIR}/filtered.jsonl \
  --output_prefix  ${DATASET_DIR}/filtered_split
