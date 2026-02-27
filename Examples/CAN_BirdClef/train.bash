#!/bin/bash

#SBATCH --output=(re)can_51_resnet_2Dssl.out
#SBATCH --error=(re)can_51_resnet_2Dssl.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S,A100,H100
#SBATCH --cpus-per-task=3

set -x
conda init
conda activate pesto-full-py310
srun python -u BaseLooperProto.py