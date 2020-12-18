import math

from rlmolecule.tree_search.tree_search_game import TreeSearchGame
from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.mcts.mcts_problem import MCTSProblem


class MCTSGame(TreeSearchGame):
    def __init__(
            self,
            problem: MCTSProblem,
            ucb_constant: float = math.sqrt(2),
    ) -> None:
        super().__init__()
        self._problem: MCTSProblem = problem
        self.ucb_constant = ucb_constant

    @property
    def problem(self) -> MCTSProblem:
        return self._problem

    def compute_reward(self, node: MCTSNode) -> float:
        return self._problem.compute_reward(node.state)
