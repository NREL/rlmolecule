from typing import Optional, TypeVar

from rlmolecule.tree_search.tree_search_canonicalizer import TreeSearchCanonicalizer
from rlmolecule.tree_search.tree_search_state import TreeSearchState

Node = TypeVar('Node')


class HashCanonicalizer(TreeSearchCanonicalizer[Node]):

    def __init__(self):
        self._node_map: {TreeSearchState: Node} = {}

    def get_canonical_node(self, state: TreeSearchState) -> Optional[Node]:
        return self._node_map[state] if state in self._node_map else None

    def canonicalize_node(self, node: Node) -> Node:
        state = node.state
        node_map = self._node_map
        if state not in node_map:
            node_map[state] = node
        return node_map[state]
