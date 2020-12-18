from rlmolecule.tree_search.tree_search_canonicalizer import TreeSearchCanonicalizer
from rlmolecule.tree_search.tree_search_node import TreeSearchNode


class HashCanonicalizer(TreeSearchCanonicalizer):

    def __init__(self):
        self._nodes = {}

    def canonicalize_node(self, node: TreeSearchNode) -> TreeSearchNode:
        nodes = self._nodes
        if node not in nodes:
            nodes[node] = node
        return nodes[node]
