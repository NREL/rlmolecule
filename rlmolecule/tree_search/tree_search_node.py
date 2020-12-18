from abc import ABC
from typing import Optional, List, TypeVar, Generic

# from rlmolecule.tree_search.tree_search_game import TreeSearchGame
from rlmolecule.tree_search.tree_search_state import TreeSearchState

Node = TypeVar('Node')


class TreeSearchNode(Generic[Node], ABC):

    def __init__(self, state: TreeSearchState) -> None:
        self.state: TreeSearchState = state
        self.children: Optional[List[Node]] = None
        self.visits: int = 0  # visit count
        self.total_value: float = 0.0

    def __eq__(self, other: any) -> bool:
        """
        equals method delegates to self._graph_node for easy hashing based on graph structure
        """
        return isinstance(other, self.__class__) and self.state == other.state

    def __hash__(self) -> int:
        """
        hash method delegates to self._graph_node for easy hashing based on graph structure
        """
        return hash(self.state)

    def __repr__(self) -> str:
        """
        repr method delegates to self._graph_node
        """
        return self.state.__repr__()

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits != 0 else 0

    @property
    def expanded(self) -> bool:
        return self.children is not None

    def update(self, reward: float) -> Node:
        """
        Updates this node with a visit and a reward
        """
        self.visits += 1
        self.total_value += reward
        return self
