from typing import Optional, TypeVar

from rlmolecule.tree_search.canonicalizer.graph_search_canonicalizer import GraphSearchCanonicalizer
from rlmolecule.tree_search.graph_search_state import GraphSearchState

Vertex = TypeVar('Vertex')


class PassthroughCanonicalizer(GraphSearchCanonicalizer[Vertex]):

    def get_canonical_vertex(self, state: GraphSearchState) -> Optional[Vertex]:
        return None

    def canonicalize_vertex(self, vertex: Vertex) -> Vertex:
        return vertex
