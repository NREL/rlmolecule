import random

from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.molecule.molecule_state import MoleculeConfig
# class MCTSHashCanonicalizationTest(unittest.TestCase):
from tests.qed_optimization_problem import QEDOptimizationProblem


def test_get_successors():
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)
    problem = QEDOptimizationProblem(config)
    game = MCTSGame(problem)
    root: MCTSNode = game._make_root()
    game._expand(root)
    root.update(1.0)
    successor0: MCTSNode = root.children[0]
    game._expand(successor0)
    successor0.update(1.0)

    successor1: MCTSNode = root.children[1]
    game._expand(successor1)
    successor1.update(1.0)

    random.seed(42)
    for _ in range(1000):
        game.run()

    child1 = root.children[1].children[0]  # CCN
    child2 = root.children[0].children[1]  # CCN

    assert (child1 == child2)
    assert (child1.value == child2.value)
    assert (child1.visits == child2.visits)

    child1.update(1.)
    assert (child1.value == child2.value)
    assert (child1.visits == child2.visits)
