    
class AlphaZeroConfig:
    
    def __init__(self):
        
        # Molecule 
        self.max_atoms = 10  # max atoms in molecule
        self.min_atoms = 4  # max atoms in molecule
        
        # MCTS / rollout
        self.lru_cache_maxsize = 100000 
        self.num_rollouts = 1000   # should we limit, if so how much?
        self.num_simulations = 256  # number of simulations used by MCTS per game step
        self.root_dirichlet_alpha = 0.0  # 0.3 chess, 0.03 Go, 0.15 shogi
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 1   # 19652 in pseudocode
        self.pb_c_init = 1.25

        # Network
        self.l2_regularization_coef = 1e-4  
        self.features = 16     # used by all network layers
        self.num_messages = 1
        self.num_heads = 4        # Number of attention heads
        self.batch_size = 32           # for gradient updates
        self.checkpoint_frequency = 1      # save new model file every N batches
        self.batch_update_frequency = 10   # get most recent data every N updates
        self.gradient_steps_per_batch = 32  # num step per batch
        self.training_iterations = int(1e06) # training iterations for NN
        
        assert self.features % self.num_heads == 0, \
            "dimension mismatch for attention heads"

        # Buffers
        self.ranked_reward_alpha = 0.9
        self.buffer_max_size = 512

        # Training
        self.training_steps = 100