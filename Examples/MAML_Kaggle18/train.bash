#!/bin/bash

#SBATCH --output=MC_51_renset.out
#SBATCH --error=MC_51_resnet.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S,A100,H100
#SBATCH --cpus-per-task=3

set -x
conda init
conda activate pesto-full-py310
srun python -u BaseLooper.py