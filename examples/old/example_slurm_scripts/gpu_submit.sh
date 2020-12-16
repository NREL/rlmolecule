#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=2-00
#SBATCH --qos=high
#SBATCH --job-name=bde_new
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out

source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu

srun python train_model.py
