from abc import abstractmethod, ABC

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
