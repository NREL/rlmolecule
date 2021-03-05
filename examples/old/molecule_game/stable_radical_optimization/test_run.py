import stable_rad_config
from molecule_game.stable_radical_optimization.stable_radical_optimization_problem import StableRadicalOptimizationGame

# G = Game(StabilityNode, 'C')
# game = list(G.run_mcts())


game = StableRadicalOptimizationGame(stable_rad_config.config, 'C')
game_history = list(game.run_mcts())
