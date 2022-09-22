#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=qed
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH -c 36
#SBATCH --partition=debug

module load cudnn/8.1.1/cuda-11.2
conda activate graphenv

for ((i = 0 ; i < 1 ; i++)); do
    srun -l -n 1 --gres=gpu:1 --nodes=1 python run_qed.py -i $i &
done