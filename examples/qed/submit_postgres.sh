#!/bin/bash
#SBATCH --account=hpcapps
#SBATCH --time=0:10:00
#SBATCH --job-name=qed_test
#SBATCH -N 2
# --- Policy Trainer ---
#SBATCH -p debug
#SBATCH --gres=gpu:1
# --- MCTS Rollouts ---
##SBATCH hetjob
##SBATCH -p debug
##SBATCH --ntasks-per-node=6

echo "$SLURM_JOB_NODELIST"

export CONDA_ENV=/scratch/hsorense/conda/rlmolecule
export DB_HOST=`scontrol show hostnames | head -1`
./start_postgres.sh
export WORKING_DIR=/scratch/${USER}/rlmolecule/qed/
mkdir -p $WORKING_DIR
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"

export config="config/qed_config_postgres.yaml"

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
echo "Running policy on `hostname`"
#source ~/.bashrc
module load conda
conda activate $CONDA_ENV
module load cudnn/8.1.1/cuda-11.2
python -u optimize_qed.py --train-policy --config="$config"

EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
echo "Running rollout on `hostname`"
#source ~/.bashrc
module load conda
conda activate $CONDA_ENV
pwd
python -u optimize_qed.py --rollout --config="$config"

EOF

srun --gres=gpu:1 \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

srun --ntasks-per-node=6 \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"
