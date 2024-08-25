#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=72:00:00
#$ -o outputs/translate/magpie/$JOB_ID.log
#$ -e outputs/translate/magpie/$JOB_ID.log
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

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data

DATASET_PATH=${DATASET_DIR}/split_train-00001-of-00002_3.jsonl
OUTPUT_PATH=${DATASET_DIR}/lm_translated_train-00001-of-00002_3.jsonl

echo "DATASET_PATH: $DATASET_PATH"

python examples/gemma/translate/translate_magpie.py \
  --jsonl-path $DATASET_PATH \
  --output-path $OUTPUT_PATH \
  --vocab_file ${VOCAB_FILE_PATH} \
  --engine_dir ${ENGINE_PATH} \
  --verbose \
  --batch_size 1
