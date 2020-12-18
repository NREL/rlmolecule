import pytest
import unittest
import random

from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rdkit.Chem.Descriptors import qed
from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.molecule.molecule_state import MoleculeState, MoleculeConfig
import rdkit.Chem


class QEDOptimizationProblem(MCTSProblem):

    def __init__(self, config: MoleculeConfig) -> None:
        self.__config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self.__config)

    def compute_reward(self, state: MoleculeState) -> float:
        return qed(state.molecule)


class MCTSHashCanonicalizationTest(unittest.TestCase):
    # noinspection PyTypeChecker
    def test_get_successors(self):
        print('ran')

        config = MoleculeConfig(max_atoms=4,
                                min_atoms=1,
                                tryEmbedding=False,
                                sa_score_threshold=None,
                                stereoisomers=False)
        problem = QEDOptimizationProblem(config)
        game = MCTSGame(problem)
        root: MCTSNode = game._get_node_for_state(problem.get_initial_state())
        game._expand(root)
        root.update(1.0)
        successor0: MCTSNode = root.children[0]
        game._expand(successor0)
        successor0.update(1.0)

        successor1: MCTSNode = root.children[1]
        game._expand(successor1)
        successor1.update(1.0)

        random.seed(42)
        for _ in range(2):
            root.mcts_step()

        child1 = root.children[1].children[0]  # CCN
        child2 = root.children[0].children[1]  # CCN

        self.assertEqual(child1, child2)
        self.assertEqual(child1.value, child2.value)
        self.assertEqual(child1.visits, child2.visits)

        child1.update(1.)
        self.assertEqual(child1.value, child2.value)
        self.assertEqual(child1.visits, child2.visits)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
