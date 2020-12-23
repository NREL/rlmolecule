from abc import abstractmethod

from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts_problem import MCTSProblem


class AlphaZeroProblem(MCTSProblem):

    @abstractmethod
    def get_value_estimate(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        """
        A user-provided function to get value and child prior estimates for the given vertex.

        :return: (value_of_current_vertex, {child_vertex: child_prior for child_vertex in children})
        """
        pass

    # @abstractmethod
    # def policy_predictions(self, policy_inputs_with_children):
    #     """
    #      un the policy network to get value and prior_logit predictions
    #     :param policy_inputs_with_children:
    #     :return: (values, prior_logits) as a tuple
    #     """
    #     pass

    # @abstractmethod
    # def construct_feature_matrices(self, state: GraphSearchState):
    #     """
    #     :param vertex:
    #     :return:  _policy_inputs
    #     """
    #     pass
