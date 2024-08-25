#!/bin/sh

START_INDEX=2
END_INDEX=60

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/OpenOrca/

for INDEX in $(seq $START_INDEX $END_INDEX); do
  echo "Running job for index: $INDEX"
  python examples/gemma/preprocess_translated_text.py \
    --input ${DATASET_DIR}/lm_translated_1M-GPT4-Augmented_${INDEX}.jsonl \
    --output ${DATASET_DIR}/lm_translated_${INDEX}.jsonl
done

# 1, 41 は、翻訳ミスにより存在しない
