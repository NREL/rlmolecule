# Molecule 
max_atoms = 10  # max atoms in molecule
min_atoms = 4  # max atoms in molecule

# MCTS / rollout
lru_cache_maxsize = 100000 
num_rollouts = 1000   # should we limit, if so how much?
num_simulations = 256  # number of simulations used by MCTS per game step
root_dirichlet_alpha = 0.0  # 0.3 chess, 0.03 Go, 0.15 shogi
root_exploration_fraction = 0.25
pb_c_base = 1   # 19652 in pseudocode
pb_c_init = 1.25
min_reward = -1.  # Minimum reward to return for invalid actions
reward_buffer = 25  # 250 in the R2 paper

# Network
l2_regularization_coef = 1e-4  
features = 16     # used by all network layers
num_messages = 1
num_heads = 4        # Number of attention heads
batch_size = 32           # for gradient updates
checkpoint_frequency = 1      # save new model file every N batches
batch_update_frequency = 10   # get most recent data every N updates
gradient_steps_per_batch = 32  # num step per batch
training_iterations = int(1e06) # training iterations for NN

#assert self.features % self.num_heads == 0, \
#   "dimension mismatch for attention heads"

# Buffers
ranked_reward_alpha = 0.9
buffer_max_size = 512

# Training
training_steps = 100

# DB tables
sql_basename = "Stable"

# Experiment id
experiment_id = "0001"
