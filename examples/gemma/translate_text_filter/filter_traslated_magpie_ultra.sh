#!/bin/sh

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00000-of-00002_1.jsonl \
  --output ${DATASET_DIR}/lm_train-00000-of-00002_1.jsonl

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00000-of-00002_2.jsonl \
  --output ${DATASET_DIR}/lm_train-00000-of-00002_2.jsonl

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00000-of-00002_3.jsonl \
  --output ${DATASET_DIR}/lm_train-00000-of-00002_3.jsonl

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00001-of-00002_1.jsonl \
  --output ${DATASET_DIR}/lm_train-00001-of-00002_1.jsonl

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00001-of-00002_2.jsonl \
  --output ${DATASET_DIR}/lm_train-00001-of-00002_2.jsonl

python examples/gemma/translate_text_filter/filter_translated_text.py \
  --input ${DATASET_DIR}/lm_translated_train-00001-of-00002_3.jsonl \
  --output ${DATASET_DIR}/lm_train-00001-of-00002_3.jsonl
