#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:12:00:00
#$ -j y
#$ -o outputs/gemma/scoring/
#$ -cwd

# Load modules
set -e

source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

source .env/bin/activate

variant=27b

CKPT_PATH=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-$variant-it/
UNIFIED_CKPT_PATH=/gs/bs/tga-NII-LLM/checkpoints/tensorRT/unified/gemma-2-$variant-it/bf16/tp1/
ENGINE_PATH=/gs/bs/tga-NII-LLM/checkpoints/tensorRT/engine/gemma2/$variant/bf16/1-gpu/
VOCAB_FILE_PATH=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-$variant-it/tokenizer.model

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/OpenOrca

INDEX=2

DATASET_PATH=${DATASET_DIR}/lm_translated_${INDEX}.jsonl
OUTPUT_PATH=${DATASET_DIR}/lm_translated_${INDEX}_scored.jsonl

echo "DATASET_PATH: $DATASET_PATH"

python examples/gemma/scoring/translate_scoring.py \
  --jsonl-path $DATASET_PATH \
  --output-path $OUTPUT_PATH \
  --vocab_file ${VOCAB_FILE_PATH} \
  --engine_dir ${ENGINE_PATH} \
  --verbose
