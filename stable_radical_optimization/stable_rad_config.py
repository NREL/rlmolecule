import os
import alphazero.config as config

# DB table names modified by the user according to their wish
config.sql_basename = "StablePSJ"

# Experiment id
config.experiment_id = "0009"

config.dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'rlops',
    'password': 'jTeL85L!',
    'options': f'-c search_path=rl',
}

config.checkpoint_filepath = os.path.expandvars(
    f'/scratch/$USER/policy_checkpoints/{config.sql_basename}/{config.experiment_id}')