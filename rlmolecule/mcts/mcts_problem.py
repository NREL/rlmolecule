from abc import ABC, abstractmethod

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTSProblem(ABC):

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def get_reward(self, state: GraphSearchState) -> (float, {}):
        """Calculates the reward for the given state. Should return the reward (as a float), and a dictionary of
        optional metadata about the reward.

        :param state: The state for which the current reward is to be calculated
        """
        pass

    def _reward_wrapper(self, state: GraphSearchState) -> float:
        return self.get_reward(state)[0]
