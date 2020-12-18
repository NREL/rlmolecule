from abc import ABC, abstractmethod
from typing import Optional, List

# from rlmolecule.tree_search.tree_search_game import TreeSearchGame
from rlmolecule.tree_search.tree_search_state import TreeSearchState


class TreeSearchNode(ABC):

    def __init__(self, state: TreeSearchState, game: 'TreeSearchGame') -> None:
        self._state: TreeSearchState = state
        self._game: 'TreeSearchGame' = game
        self._successors: Optional[List[TreeSearchNode]] = None
        self._visits: int = 0  # visit count
        self._total_value: float = 0.0

    def __eq__(self, other: any) -> bool:
        """
        equals method delegates to self._graph_node for easy hashing based on graph structure
        """
        return isinstance(other, self.__class__) and self._state == other._state

    def __hash__(self) -> int:
        """
        hash method delegates to self._graph_node for easy hashing based on graph structure
        """
        return hash(self._state)

    def __repr__(self) -> str:
        """
        repr method delegates to self._graph_node
        """
        return self._state.__repr__()

    @property
    def value(self) -> float:
        return self._total_value / self._visits if self._visits != 0 else 0

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def expanded(self) -> bool:
        return self.successors is not None

    @property
    def state(self) -> TreeSearchState:
        """
        :return: delegate which defines the graph structure being explored
        """
        return self._state

    @property
    def game(self) -> 'TreeSearchGame':
        """
        :return: delegate which contains game-level configuration
        """
        return self._game

    @property
    def successors(self) -> Optional[List['TreeSearchNode']]:
        return self._successors

    def expand(self) -> ['TreeSearchNode']:
        self._successors = [
            self.game.canonicalize_node(self._make_successor(action))
            for action in self._state.get_next_actions()
        ]
        return self.successors

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    @abstractmethod
    def _make_successor(self, action: TreeSearchState) -> 'TreeSearchNode':
        pass
