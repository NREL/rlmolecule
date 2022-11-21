#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=radical
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=400000

module load cudnn/8.1.1/cuda-11.2
conda activate rllib

python run_radical.py