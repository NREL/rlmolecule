from abc import ABC, abstractmethod

from rlmolecule.tree_search.tree_search_state import TreeSearchState


class MCTSProblem(ABC):

    @abstractmethod
    def get_initial_state(self) -> TreeSearchState:
        pass

    @abstractmethod
    def compute_reward(self, state: TreeSearchState) -> float:
        pass
