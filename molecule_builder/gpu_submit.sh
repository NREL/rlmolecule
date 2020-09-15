#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --qos=high
#SBATCH --job-name=policyNN_training
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/eskordil/git-repos/rlmolecule/gpu.%j.out

source ~/.bashrc
conda activate /scratch/eskordil/conda-envs/rlmolecule

srun python train_model.py