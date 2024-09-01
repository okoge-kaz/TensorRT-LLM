#!/bin/sh

set -e

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

mkdir -p ${UNIFIED_CKPT_PATH}
mkdir -p ${ENGINE_PATH}

python ./examples/gemma/convert_checkpoint.py \
  --ckpt-type hf \
  --model-dir ${CKPT_PATH} \
  --dtype bfloat16 \
  --world-size 1 \
  --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
  --gemm_plugin auto \
  --max_batch_size 8 \
  --max_input_len 3000 \
  --max_seq_len 4096 \
  --lookup_plugin bfloat16 \
  --output_dir ${ENGINE_PATH}
