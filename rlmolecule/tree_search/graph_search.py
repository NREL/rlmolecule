import uuid
from abc import ABC
from typing import Generic, Optional, Type, TypeVar

from rlmolecule.tree_search.canonicalizer.graph_search_canonicalizer import GraphSearchCanonicalizer
from rlmolecule.tree_search.canonicalizer.hash_canonicalizer import HashCanonicalizer
from rlmolecule.tree_search.graph_search_state import GraphSearchState

Vertex = TypeVar('Vertex')


class GraphSearch(Generic[Vertex], ABC):
    def __init__(self, vertex_class: Type[Vertex], canonicalizer: Optional[GraphSearchCanonicalizer] = None):
        self._canonicalizer: GraphSearchCanonicalizer[Vertex] = \
            HashCanonicalizer() if canonicalizer is None else canonicalizer
        self._vertex_class = vertex_class

    @property
    def canonicalizer(self) -> GraphSearchCanonicalizer:
        return self._canonicalizer

    def get_vertex_for_state(self, state: GraphSearchState) -> Vertex:
        vertex = self._canonicalizer.get_canonical_vertex(state)
        if vertex is None:
            vertex = self._make_new_vertex(state)
            vertex = self._canonicalizer.canonicalize_vertex(vertex)
        return vertex

    # noinspection PyMethodMayBeStatic
    def _make_new_vertex(self, state: GraphSearchState) -> Vertex:
        return self._vertex_class(state)
