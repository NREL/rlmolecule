import random

from rlmolecule.mcts.mcts import MCTS
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.molecule.molecule_config import MoleculeConfig
# class MCTSHashCanonicalizationTest(unittest.TestCase):
from tests.qed_optimization_problem import QEDOptimizationProblem


def test_get_successors(engine):
    config = MoleculeConfig(max_atoms=4,
                            min_atoms=1,
                            tryEmbedding=False,
                            sa_score_threshold=None,
                            stereoisomers=False)
    problem = QEDOptimizationProblem(engine, config)
    game = MCTS(problem)
    root: MCTSVertex = game._get_root()
    game._expand(root)
    root.update(1.0)
    successor0: MCTSVertex = root.children[0]
    game._expand(successor0)
    successor0.update(1.0)

    successor1: MCTSVertex = root.children[1]
    game._expand(successor1)
    successor1.update(1.0)

    random.seed(42)
    game.sample(root, 5)

    child1 = root.children[1].children[0]  # CCN
    child2 = root.children[0].children[1]  # CCN

    assert (child1 == child2)
    assert (child1.value == child2.value)
    assert (child1.visit_count == child2.visit_count)

    child1.update(1.)
    assert (child1.value == child2.value)
    assert (child1.visit_count == child2.visit_count)
