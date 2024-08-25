#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=48:00:00
#$ -o outputs/inference/$JOB_ID.log
#$ -e outputs/inference/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source examples/gemma/.env/bin/activate

variant=27b

CKPT_PATH=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-$variant-it/
UNIFIED_CKPT_PATH=/gs/bs/tga-NII-LLM/checkpoints/tensorRT/unified/gemma-2-$variant-it/bf16/tp1/
ENGINE_PATH=/gs/bs/tga-NII-LLM/checkpoints/tensorRT/engine/gemma2/$variant/bf16/1-gpu/
VOCAB_FILE_PATH=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-$variant-it/tokenizer.model

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja

DATASET_PATH=${DATASET_DIR}/filtered_split_filtered_4.jsonl
OUTPUT_PATH=${DATASET_DIR}/lm_filtered_split_4.jsonl

echo "DATASET_PATH: $DATASET_PATH"

python examples/gemma/language_model_filter.py \
  --jsonl-path $DATASET_PATH \
  --output-path $OUTPUT_PATH \
  --vocab_file ${VOCAB_FILE_PATH} \
  --engine_dir ${ENGINE_PATH} \
  --json-conversation-key "conversations" \
  --verbose
