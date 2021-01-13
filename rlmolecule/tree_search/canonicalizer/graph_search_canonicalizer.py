from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from rlmolecule.tree_search.graph_search_state import GraphSearchState

Vertex = TypeVar('Vertex')


class GraphSearchCanonicalizer(Generic[Vertex], ABC):

    @abstractmethod
    def get_canonical_vertex(self, state: GraphSearchState) -> Optional[Vertex]:
        pass

    @abstractmethod
    def canonicalize_vertex(self, vertex: Vertex) -> Vertex:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
