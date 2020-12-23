from abc import abstractmethod, ABC

from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class AlphaZeroProblem(ABC):

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def evaluate(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
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
