#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=rlmolecule
#SBATCH --time=0:10:00  # start with 10min for debug
#SBATCH --job-name=gridworld_example_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4

export WORKING_DIR=/scratch/${USER}/git-repos/rlmolecule_new/rlmolecule/examples/gym/
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
module purge
conda activate /projects/rlmolecule/eskordil/envs/tf2_gpu_cloned
python -u solve_gridworld.py --train-policy
EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
module purge
conda activate /projects/rlmolecule/eskordil/envs/tf2_cpu_cloned
python -u solve_gridworld.py --rollout
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

# there are 36 cores on eagle nodes.
# run one policy training job
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

# and run 8 cpu rollout jobs
srun --gres=gpu:0 --ntasks=8 --cpus-per-task=4 \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"