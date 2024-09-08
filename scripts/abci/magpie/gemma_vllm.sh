#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00:00
#$ -j y
#$ -o outputs/gemma/magpie/
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

DATASET_DIR=/bb/llm/gaf51275/datasets/raw/instruct/MAGPIE/gemma2-27b-it
mkdir -p $DATASET_DIR

OUTPUT_PATH=${DATASET_DIR}/financial_1.jsonl

echo "DATASET_PATH: $DATASET_PATH"

python scripts/common/magpie/gemma-2-vllm.py \
  --tensor-parallel 4 \
  --output-path $OUTPUT_PATH \
  --model-path $CKPT_PATH \
  --category "金融、経済"
