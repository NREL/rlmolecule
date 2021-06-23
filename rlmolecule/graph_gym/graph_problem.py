from abc import ABC, abstractmethod

import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class GraphProblem(ABC):

    @abstractmethod
    @property
    def observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def make_null_observation(self) -> any:
        pass

    @abstractmethod
    def make_observation(self, state: GraphSearchState) -> any:
        pass

    @abstractmethod
    @property
    def action_space(self) -> gym.Space:
        pass

    @abstractmethod
    @property
    def max_num_actions(self) -> int:
        pass

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def get_reward(self, state: GraphSearchState) -> float:
        pass

    def is_terminal(self, state: GraphSearchState) -> bool:
        return len(state.get_next_actions()) == 0

    @abstractmethod
    def step(self, state: GraphSearchState) -> (float, bool, dict):
        pass
