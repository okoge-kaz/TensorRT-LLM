#!/bin/sh

source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/datasets/raw/instruct/OpenOrca

for INDEX in $(seq 3 60); do
  echo "Running job for index: $INDEX"
  python scripts/tsubame/translate_text_filter/filter_translated_text.py \
    --input ${DATASET_DIR}/lm_generated_${INDEX}.jsonl \
    --output ${DATASET_DIR}/open_orca_ja_${INDEX}.jsonl
done
