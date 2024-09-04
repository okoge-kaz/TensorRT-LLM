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

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# gemma-2-27b-it
variant=27b

CKPT_PATH=/groups/gag51395/hf-checkpoints/gemma-2-$variant-it/
UNIFIED_CKPT_PATH=/bb/llm/gaf51275/checkpoints/tensorRT/unified/gemma-2-$variant-it/bf16/tp8/
ENGINE_PATH=/bb/llm/gaf51275/checkpoints/tensorRT/engine/gemma2/$variant/bf16/8-gpu/
VOCAB_FILE_PATH=/groups/gag51395/hf-checkpoints/gemma-2-$variant-it/tokenizer.model

DATASET_DIR=/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-top1-en-chat-sft/data

DATASET_PATH=${DATASET_DIR}/split_train-00000-of-00001_1_conversations.jsonl
OUTPUT_PATH=${DATASET_DIR}/lm_scored.jsonl

echo "DATASET_PATH: $DATASET_PATH"

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x LD_LIBRARY_PATH \
  -x PATH \
  python examples/gemma/scoring/language_model_scoring.py \
    --jsonl-path $DATASET_PATH \
    --output-path $OUTPUT_PATH \
    --vocab_file ${VOCAB_FILE_PATH} \
    --engine_dir ${ENGINE_PATH} \
    --json-conversation-key "conversations" \
    --verbose
