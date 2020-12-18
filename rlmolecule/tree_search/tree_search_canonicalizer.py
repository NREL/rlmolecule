from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from rlmolecule.tree_search.tree_search_state import TreeSearchState

Node = TypeVar('Node')


class TreeSearchCanonicalizer(Generic[Node], ABC):

    @abstractmethod
    def get_canonical_node(self, state: TreeSearchState) -> Optional[Node]:
        pass

    @abstractmethod
    def canonicalize_node(self, node: Node) -> Node:
        pass
