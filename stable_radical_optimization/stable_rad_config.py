import os
import alphazero.config as config

# DB table names modified by the user according to their wish
config.sql_basename = "StablePSJ"

# Experiment id
config.experiment_id = "03_15_atom_max"

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

config.dirichlet_x = 0.5  # percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise

config.build_kwargs.update({'atom_additions':  ('C', 'N', 'O', 'S'),
                            'sa_score_threshold': 4.})

config.max_atoms = 15
