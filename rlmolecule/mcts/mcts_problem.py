import uuid
from abc import ABC, abstractmethod

from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTSProblem(ABC):

    def __init__(self):
        self.__id: uuid.UUID = uuid.uuid4()

    @property
    def id(self) -> uuid.UUID:
        return self.__id

    @abstractmethod
    def get_initial_state(self) -> GraphSearchState:
        pass

    @abstractmethod
    def get_reward(self, state: GraphSearchState) -> float:
        pass
