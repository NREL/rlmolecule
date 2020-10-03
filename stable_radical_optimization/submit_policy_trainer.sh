#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=00:20:00
#SBATCH --partition=debug
#SBATCH --job-name=train_policy_network
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/rlmolecule/gpu.%j.out

source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu

srun python train_policy.py
