#!/bin/bash
#SBATCH --account=invpoly
#SBATCH --time=2-00
#SBATCH --job-name=cn_cv
#SBATCH -n 10
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/schnet_uff.%j.out  # %j will be replaced with the job ID

source /home/pstjohn/.bashrc
conda activate tf

srun -l hostname

for ((i = 0 ; i < 10 ; i++)); do
    srun -l -n 1 --gres=gpu:1 --nodes=1 python model_globals.py $i &
done

wait
