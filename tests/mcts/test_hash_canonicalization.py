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
    def test_get_successors(self):
        print('ran')

        config = MoleculeConfig(max_atoms=4,
                                min_atoms=1,
                                tryEmbedding=False,
                                sa_score_threshold=None,
                                stereoisomers=False)
        problem = QEDOptimizationProblem(config)
        game = MCTSGame(problem)
        root = MCTSNode(problem.get_initial_state(), game)
        root.expand()
        root.update(1.0)
        successor0 = root.successors[0]
        successor0.expand()
        successor0.update(1.0)

        successor1 = root.successors[1]
        successor1.expand()
        successor1.update(1.0)

        random.seed(42)
        for _ in range(2):
            root.mcts_step()

        child1 = root.successors[1].successors[0]  # CCN
        child2 = root.successors[0].successors[1]  # CCN

        self.assertEqual(child1, child2)
        self.assertEqual(child1.value, child2.value)
        self.assertEqual(child1.visits, child2.visits)

        child1.update(1.)
        self.assertEqual(child1.value, child2.value)
        self.assertEqual(child1.visits, child2.visits)
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()