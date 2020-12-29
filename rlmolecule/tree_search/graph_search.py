import uuid
from abc import ABC
from typing import Optional, TypeVar, Generic, Type

from rlmolecule.tree_search.canonicalizer.graph_search_canonicalizer import GraphSearchCanonicalizer
from rlmolecule.tree_search.canonicalizer.hash_canonicalizer import HashCanonicalizer
from rlmolecule.tree_search.graph_search_state import GraphSearchState

Vertex = TypeVar('Vertex')


class GraphSearch(Generic[Vertex], ABC):
    def __init__(self, vertex_class: Type[Vertex], canonicalizer: Optional[GraphSearchCanonicalizer] = None):
        self.__id: uuid.UUID = uuid.uuid4()
        self.__canonicalizer: GraphSearchCanonicalizer[Vertex] = \
            HashCanonicalizer() if canonicalizer is None else canonicalizer
        self._vertex_class = vertex_class

    @property
    def id(self) -> uuid.UUID:
        return self.__id

    def get_vertex_for_state(self, state: GraphSearchState) -> Vertex:
        vertex = self.__canonicalizer.get_canonical_vertex(state)
        if vertex is None:
            vertex = self._make_new_vertex(state)
            vertex = self.__canonicalizer.canonicalize_vertex(vertex)
        return vertex

    # noinspection PyMethodMayBeStatic
    def _make_new_vertex(self, state: GraphSearchState) -> Vertex:
        return self._vertex_class(state)
