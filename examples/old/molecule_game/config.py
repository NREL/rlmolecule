import logging
import os

loglevel = 'INFO'
logging.basicConfig(level=os.environ.get("LOGLEVEL", loglevel))

# Molecule 
max_atoms = 10  # max atoms in molecule
min_atoms = 4  # max atoms in molecule

# MCTS / rollout
lru_cache_maxsize = 100000
num_rollouts = 1000  # should we limit, if so how much?
# num_simulations = 256  # number of simulations used by MCTS per game step
num_simulations = 20  # number of simulations used by MCTS per game step
root_dirichlet_alpha = 0.0  # 0.3 chess, 0.03 Go, 0.15 shogi
root_exploration_fraction = 0.25
pb_c_base = 1  # 19652 in pseudocode
pb_c_init = 1.25
min_reward = 0.  # Minimum reward to return for invalid actions
dirichlet_noise = True  # whether to add dirichlet noise
dirichlet_alpha = 1.  # dirichlet 'shape' parameter. Larger values spread out probability over more moves.
dirichlet_x = 0.25  # percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise

# Network
features = 64  # used by all network layers
num_heads = 4  # Number of attention heads
num_messages = 3
batch_size = 32  # for gradient updates
steps_per_epoch = 750
policy_lr = 1E-3
checkpoint_filepath = None  # A checkpoint filepath must be provided

# Buffers
ranked_reward_alpha = 0.9
reward_buffer_max_size = 250  # 250 in the R2 paper
reward_buffer_min_size = 50  # Allow R2 calculations without a full buffer

policy_buffer_max_size = 1024  # Only sample from this many most recent games
policy_buffer_min_size = 128  # Don't start training the model until this many games have occurred

# Training
training_steps = 100

# DB tables
sql_basename = "Stable"

# Experiment id
experiment_id = "0001"

# Optional arguments for molecule builder
build_kwargs = {}
