import uuid
from abc import ABC, abstractmethod
from typing import Optional

from rlmolecule.tree_search.reward import RawRewardFactory, RewardFactory, Reward
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTSProblem(ABC):

    def __init__(self, reward_class: Optional[RewardFactory] = None):
        self.__id = None
        if reward_class is None:
            reward_class = RawRewardFactory()

        self.reward_class = reward_class

    @property
    def id(self):
        return self.__id

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

    def initialize_run(self) -> None:
        """ Any per-game initialization, i.e., loading policy weights, should be done here. This is called once
        before conducting the MCTS search every time AlphaZero.run() is called.
        """
        self.__id = uuid.uuid4()

    def reward_wrapper(self, state: GraphSearchState) -> Reward:
        reward, _ = self.get_reward(state)
        return self.reward_class(reward)