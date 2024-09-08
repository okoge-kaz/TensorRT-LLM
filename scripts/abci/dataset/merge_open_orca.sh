#!/bin/sh

source .env/bin/activate

INPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/OpenOrca
INPUT_FILES=$(find "$INPUT_DIR" -name "open_orca_ja_*.jsonl")

echo "INPUT_FILES: $INPUT_FILES"

OUTPUT_FILE="/bb/llm/gaf51275/datasets/raw/instruct/OpenOrca/processed_openorca_output.jsonl"

python scripts/common/dataset/merge_open_orca.py \
  --input $INPUT_FILES \
  --output $OUTPUT_FILE
