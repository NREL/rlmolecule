import uuid
from abc import ABC, abstractmethod
from typing import Optional

from rlmolecule.tree_search.hash_canonicalizer import HashCanonicalizer
from rlmolecule.tree_search.tree_search_canonicalizer import TreeSearchCanonicalizer
from rlmolecule.tree_search.tree_search_node import TreeSearchNode


class TreeSearchGame(ABC):
    def __init__(self, canonicalizer: Optional[TreeSearchCanonicalizer] = None):
        self.__id: uuid.UUID = uuid.uuid4()
        self.__canonicalizer = HashCanonicalizer() if canonicalizer is None else canonicalizer

    @property
    def id(self) -> uuid.UUID:
        return self.__id

    @abstractmethod
    def compute_reward(self, node: TreeSearchNode) -> float:
        pass

    def canonicalize_node(self, node: TreeSearchNode) -> TreeSearchNode:
        return self.__canonicalizer.canonicalize_node(node)
