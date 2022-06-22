#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --job-name=all_decors_reward
#SBATCH --output=/scratch/jlaw/all_decors/slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=180
##SBATCH --partition=debug
##SBATCH --time=1:00:00
##SBATCH --nodes=2
##SBATCH --ntasks=72

set -e

source $HOME/.bashrc_conda
module use /nopt/nrel/apps/modules/test/modulefiles/
module load cudnn/8.1.1/cuda-11.2
conda activate /home/jlaw/.conda-envs/crystals_nfp0_3

in_file="outputs/20220620_rewards/20220621_all_decors.csv.gz"
num_lines=`gzip -cd $in_file  | wc -l`

num_splits=180
split_size=$((num_lines / num_splits + 1))

curr_line=$split_size
WORKING_DIR="/scratch/jlaw/all_decors"
mkdir -p $WORKING_DIR

for i in `seq 1 1 $num_splits`; do 
    curr_in_file="$WORKING_DIR/$i.csv.gz"
    # write the current lines to a file
echo "gzip -cd $in_file | head -n $curr_line | tail -n $split_size | gzip > $curr_in_file"
    gzip -cd $in_file | head -n $curr_line | tail -n $split_size | gzip > $curr_in_file
    cmd="""python -u scripts/compute_reward_decors.py \
    --config config/20220617_lt15stoich_battclust0_01/r_90.yaml \
    --energy-model inputs/models/2022_06_07_pruned_outliers/icsd_and_battery_scaled/best_model.hdf5 \
    --out-file $WORKING_DIR/${i}-out.csv.gz \
    --decor-ids-file=$WORKING_DIR/${i}.csv.gz"""

    echo "$cmd"

    srun --nodes=1 --ntasks=1 --cpus-per-task=1 $cmd &
    #$cmd
	 
    curr_line=$((curr_line + split_size))

    #break

done

##Wait for all
wait
 
echo
echo "Finished `date`"
