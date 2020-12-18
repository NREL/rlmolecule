from abc import abstractmethod, ABC

from rlmolecule.alphazero.alphazero_node import AlphaZeroNode
from rlmolecule.tree_search.tree_search_state import TreeSearchState


class AlphaZeroProblem(ABC):

    @abstractmethod
    def get_initial_state(self) -> TreeSearchState:
        pass

    @abstractmethod
    def compute_reward(self, state: TreeSearchState, policy_inputs: any) -> float:
        pass

    @abstractmethod
    def policy_predictions(self, policy_inputs_with_children):
        """
         un the policy network to get value and prior_logit predictions
        :param policy_inputs_with_children:
        :return: (values, prior_logits) as a tuple
        """
        pass

    @abstractmethod
    def construct_feature_matrices(self, state: TreeSearchState):
        """
        :param node:
        :return:  _policy_inputs
        """
        pass

    @abstractmethod
    def policy(self, node: AlphaZeroNode) -> (float, {AlphaZeroNode: float}):
        """
        A user-provided function to get value and prior estimates for the given node. Accepts a list of child
        nodes for the given state, and should return both the predicted value of the current node, as well as prior
        scores for each child node.

        :param children: A list of AlphaZeroNodes corresponding to next potential actions
        :return: (value_of_current_node, {child_node: child_prior for child_node in children})
        """
        pass
