from alphazero import config
from alphazero.molecule_game import MoleculeGame
from run_mcts import Game, StabilityNode

# G = Game(StabilityNode, 'C')
# game = list(G.run_mcts())

game = MoleculeGame(config, 'C')
game_history = list(game.run_mcts())
