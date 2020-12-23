from abc import ABC, abstractmethod

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTSProblem(ABC):

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def get_reward(self, state: GraphSearchState) -> float:
        pass
