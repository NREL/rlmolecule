from typing import Optional, TypeVar

from rlmolecule.tree_search.canonicalizer.graph_search_canonicalizer import GraphSearchCanonicalizer
from rlmolecule.tree_search.graph_search_state import GraphSearchState

Vertex = TypeVar('Vertex')


class HashCanonicalizer(GraphSearchCanonicalizer[Vertex]):

    def __init__(self):
        self._vertex_map: {GraphSearchState: Vertex} = {}

    def get_canonical_vertex(self, state: GraphSearchState) -> Optional[Vertex]:
        return self._vertex_map[state] if state in self._vertex_map else None

    def canonicalize_vertex(self, vertex: Vertex) -> Vertex:
        state = vertex.state
        vertex_map = self._vertex_map
        if state not in vertex_map:
            vertex_map[state] = vertex
        return vertex_map[state]

    def reset(self) -> None:
        self._vertex_map = {}