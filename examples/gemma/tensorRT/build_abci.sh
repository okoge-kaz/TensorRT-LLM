#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:12:00:00
#$ -j y
#$ -o outputs/gemma/build/
#$ -cwd

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

CKPT_PATH=/groups/gag51395/hf-checkpoints/gemma-2-$variant-it/
UNIFIED_CKPT_PATH=/bb/llm/gaf51275/checkpoints/tensorRT/unified/gemma-2-$variant-it/bf16/tp8/
ENGINE_PATH=/bb/llm/gaf51275/checkpoints/tensorRT/engine/gemma2/$variant/bf16/8-gpu/
VOCAB_FILE_PATH=/groups/gag51395/hf-checkpoints/gemma-2-$variant-it/tokenizer.model

mkdir -p ${UNIFIED_CKPT_PATH}
mkdir -p ${ENGINE_PATH}

python ./examples/gemma/convert_checkpoint.py \
  --ckpt-type hf \
  --model-dir ${CKPT_PATH} \
  --dtype bfloat16 \
  --world-size 8 \
  --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
  --gemm_plugin auto \
  --max_batch_size 16 \
  --max_input_len 3000 \
  --max_seq_len 4096 \
  --lookup_plugin bfloat16 \
  --output_dir ${ENGINE_PATH}
