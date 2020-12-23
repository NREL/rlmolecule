# Reimplementation of alphazero code

* vertex.py: all mcts logic, ucb score, and gnn network inputs
* mcts.py: run main rollout loop, load model weights, do tree search, etc.
* config.py: global configuration
* policy.py: define tf.keras model, games dataset, and model training loop
* molecule.py: all molecule building logic
