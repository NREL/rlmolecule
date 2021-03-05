import os

import molecule_game.config as config

# DB table names modified by the user according to their wish
# config.sql_basename = "StablePSJ"
config.sql_basename = "CharlesNodeRefactor"

# Experiment id
config.experiment_id = "01"

# with open('/projects/rlmolecule/rlops_pass', 'r') as f:
#     passwd = f.read().strip()
with open('rlops_pass', 'r') as f:
    passwd = f.read().strip()

config.dbparams = {
    'dbname':   'bde',
    'port':     5432,
    'host':     'yuma.hpc.nrel.gov',
    'user':     'rlops',
    'password': passwd,
    'options':  f'-c search_path=rl',
    }

config.checkpoint_filepath = os.path.expandvars(
    f'/scratch/$USER/policy_checkpoints/{config.sql_basename}/{config.experiment_id}')

config.dirichlet_x = 0.5  # percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise

config.build_kwargs.update({
                               'atom_additions':     ('C', 'N', 'O', 'S'),
                               'sa_score_threshold': 4.
                               })

# config.max_atoms = 15
config.max_atoms = 5

# config.reward_model_path = '/projects/rlmolecule/pstjohn/models/20201020_radical_stability_model'
config.reward_model_path = '/home/ctripp/project/rlmol/data/model/20201020_radical_stability_mod'
