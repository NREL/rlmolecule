#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=1:00:00
#SBATCH --job-name=gridworld_example
# --- Policy Trainer ---
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -N 1

export WORKING_DIR=/scratch/${USER}/git-repos/rlmolecule_new/rlmolecule/examples/gym/
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"

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

srun --pack-group=0 \
     --job-name="az-policy" \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

srun --pack-group=1 \
     --ntasks-per-node=6 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"