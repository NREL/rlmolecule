from abc import ABC, abstractmethod

from rlmolecule.tree_search.tree_search_node import TreeSearchNode


class TreeSearchCanonicalizer(ABC):

    @abstractmethod
    def canonicalize_node(self, node: TreeSearchNode) -> TreeSearchNode:
        pass
