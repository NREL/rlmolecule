#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=rlmolecule
#SBATCH --time=0:30:00  # start with 10min for debug
#SBATCH --job-name=qed_example_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4

export WORKING_DIR=/scratch/${USER}/rlmolecule/qed/
mkdir -p $WORKING_DIR
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"

export config="config/qed_config_debug.yaml"

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
source ~/.bashrc
conda activate rlmol
module load cudnn/8.1.1/cuda-11.2
python -u optimize_qed.py --train-policy --config="$config"

EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source ~/.bashrc
conda activate rlmol
pwd
python -u optimize_qed.py --rollout --config="$config"

EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

# there are 36 cores on eagle nodes.
# run one policy training job
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

# and run cpu rollout jobs
srun --gres=gpu:0 --ntasks=7 --cpus-per-task=4 \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"

