import uuid
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from rlmolecule.tree_search.hash_canonicalizer import HashCanonicalizer
from rlmolecule.tree_search.tree_search_canonicalizer import TreeSearchCanonicalizer
from rlmolecule.tree_search.tree_search_state import TreeSearchState

Node = TypeVar('Node')


class TreeSearchGame(Generic[Node], ABC):
    def __init__(self, canonicalizer: Optional[TreeSearchCanonicalizer] = None):
        self.__id: uuid.UUID = uuid.uuid4()
        self.__canonicalizer: HashCanonicalizer[Node] = HashCanonicalizer() if canonicalizer is None else canonicalizer

    @property
    def id(self) -> uuid.UUID:
        return self.__id

    def _get_node_for_state(self, state: TreeSearchState) -> Node:
        node = self.__canonicalizer.get_canonical_node(state)
        if node is None:
            node = self._make_new_node(state)
            node = self.__canonicalizer.canonicalize_node(node)
        return node

    @abstractmethod
    def _make_new_node(self, state: TreeSearchState) -> Node:
        pass
