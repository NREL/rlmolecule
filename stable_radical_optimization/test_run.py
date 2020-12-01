from run_mcts import Game, StabilityNode

G = Game(StabilityNode, 'C')
game = list(G.run_mcts())
