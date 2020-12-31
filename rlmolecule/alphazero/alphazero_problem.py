from abc import abstractmethod

from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts_problem import MCTSProblem


class AlphaZeroProblem(MCTSProblem):

    @abstractmethod
    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        """
        A user-provided function to get value and child prior estimates for the given vertex.

        :return: (value_of_current_vertex, {child_vertex: child_prior for child_vertex in children})
        """
        pass
