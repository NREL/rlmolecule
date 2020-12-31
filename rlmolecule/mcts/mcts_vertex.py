import logging
from typing import List, Optional

from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class MCTSVertex:

    def __init__(self, state: GraphSearchState) -> None:
        self.state: GraphSearchState = state
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Optional[List['MCTSVertex']] = None

    def __eq__(self, other: any) -> bool:
        """
        equals method delegates to self._graph_vertex for easy hashing based on graph structure
        """
        return isinstance(other, self.__class__) and self.state == other.state

    def __hash__(self) -> int:
        """
        hash method delegates to self._graph_vertex for easy hashing based on graph structure
        """
        return hash(self.state)

    def __repr__(self) -> str:
        """
        repr method delegates to self._graph_vertex
        """
        return self.state.__repr__()

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count != 0 else 0

    @property
    def expanded(self) -> bool:
        return self.children is not None

    def update(self, reward: float) -> None:
        """
        Updates this vertex with a visit and a reward
        """
        self.visit_count += 1
        self.value_sum += reward
