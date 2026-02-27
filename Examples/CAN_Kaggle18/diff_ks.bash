#!/bin/bash

#SBATCH --output=(re)can_resnet_k_sweep.out
#SBATCH --error=(re)can_resnet_k_sweep.err
#SBATCH --time=24:00:00  # Increased time for 4 sequential runs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S,A100,H100
#SBATCH --cpus-per-task=3

set -x
conda init
conda activate pesto-full-py310

# Values for the ablation study
K_VALUES=(0.2 0.4 0.6 0.8)

for k in "${K_VALUES[@]}"
do
    echo "------------------------------------------------"
    echo "EXECUTING TASK: MASK_K = $k"
    echo "------------------------------------------------"

    # Set the environment variable that Python is now listening for
    export MASK_K=$k
    
    # Run the looper
    srun python -u BaseLooperProto.py
    
    echo "Task for k=$k finished. Cool-down for 30s..."
    sleep 30
done