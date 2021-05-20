# Wrapper script to set the run_id and 
# copy the config file to the output directory for a given experiment.
# All the output (policy_checkpoints and log files) will be in that directory

# run_id is the first parameter, config is the second
run_id="$1"
config_file="$2"

if [ "$1" == "" ]; then
    echo "Need to pass <run_id> as first argument"
    exit
fi
if [ "$2" == "" ]; then
    echo "Need to pass <config_file> as second argument"
    exit
fi

WORKING_DIR="/projects/rlmolecule/$USER/stable_radical_opt/${run_id}"
mkdir -p $WORKING_DIR

# copy the config file with the rest of the results
SCRIPT_CONFIG="$WORKING_DIR/run.yaml"
cp $config_file $SCRIPT_CONFIG
# set the run_id in the new config file
sed -i "s/test_stable_rad_opt/$run_id/" $SCRIPT_CONFIG

# create the submission script
echo """#!/bin/bash
#SBATCH --account=bpms
#SBATCH --time=4:00:00  
#SBATCH --job-name=$run_id
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=$USER@nrel.gov
#SBATCH --output=$WORKING_DIR/%j-sbatch.out
# --- Policy Trainer ---
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -N 10

# Track which version of the code generated this output
# by putting the current branch and commit in the log file
echo \"Current git branch & commit:\"
git rev-parse --abbrev-ref HEAD
git log -n 1 --oneline

export WORKING_DIR=$WORKING_DIR
mkdir -p $WORKING_DIR
export START_POLICY_SCRIPT="\$WORKING_DIR/\$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="\$WORKING_DIR/\$JOB/.rollout.sh"
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):\$PYTHONPATH"

model_dir="/projects/rlmolecule/pstjohn/models/"; 
stability_model="\$model_dir/20210214_radical_stability_new_data/"
redox_model="\$model_dir/20210214_redox_new_data/"
bde_model="\$model_dir/20210216_bde_new_nfp/"

cat << EOF > "\$START_POLICY_SCRIPT"
#!/bin/bash
source $HOME/.bashrc
module use /nopt/nrel/apps/modules/test/modulefiles/
module load cudnn/8.1.1/cuda-11.2
conda activate rlmol
python -u stable_radical_opt.py \
    --train-policy \
    --config $SCRIPT_CONFIG \
    --stability-model="\$stability_model" \
    --redox-model="\$redox_model" \
    --bde-model="\$bde_model" 
EOF

cat << EOF > "\$START_ROLLOUT_SCRIPT"
#!/bin/bash
source $HOME/.bashrc
conda activate rlmol
python -u stable_radical_opt.py \
    --rollout \
    --config $SCRIPT_CONFIG \
    --stability-model="\$stability_model" \
    --redox-model="\$redox_model" \
    --bde-model="\$bde_model" 
EOF

chmod +x "\$START_POLICY_SCRIPT" "\$START_ROLLOUT_SCRIPT"

srun --pack-group=0 \
     --job-name="az-policy" \
     --output=$WORKING_DIR/%j-gpu.out \
     "\$START_POLICY_SCRIPT" &

# there are 36 cores on each eagle node.
srun --pack-group=1 \
     --ntasks-per-node=18 \
     --cpus-per-task=2 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/%j-mcts.out \
     "\$START_ROLLOUT_SCRIPT"
""" > $WORKING_DIR/.submit.sh

sbatch $WORKING_DIR/.submit.sh
