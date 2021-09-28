#!/bin/bash
#SBATCH --time=0:30:00  # start with 10min for debug
#SBATCH --job-name=qed_example
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --output=qed.out
#SBATCH --error=qed.err

# SWS: removed some SBATCH above
# SWS: changing working directory to container location
export WORKING_DIR=/home/hpcuser/rlmolecule/qed/
mkdir -p $WORKING_DIR
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"

# SWS: maybe need to install other packages in rlmolecule image
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"
export config="config/qed_config_local.yaml"

echo ""
echo "SLURM_SUBMIT_DIR     = $SLURM_SUBMIT_DIR"
echo "START_POLICY_SCRIPT  = $START_POLICY_SCRIPT"
echo "START_ROLLOUT_SCRIPT = $START_ROLLOUT_SCRIPT"
echo "          PYTHONPATH = $PYTHONPATH"
echo ""

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
# source ~/.bashrc
# conda activate rlmol
python -u optimize_qed.py --train-policy --config="$config"

EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
# source ~/.bashrc
# conda activate rlmol
pwd
python -u optimize_qed.py --rollout --config="$config"

EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

# there are 36 cores on eagle nodes.
# run one policy training job
srun --ntasks=1 --cpus-per-task=2 \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

# and run cpu rollout jobs
srun --ntasks=4 --cpus-per-task=2 \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"

