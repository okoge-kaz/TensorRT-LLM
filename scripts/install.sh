#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

# python virtualenv
source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

pip install -r examples/gemma/requirements.txt
