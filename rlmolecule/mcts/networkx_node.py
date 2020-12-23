from typing import List

from networkx import DiGraph


class NetworkxSuccessorMixin(object):
    """ Class to add new vertices to a networkx DiGraph, such that vertex transpositions (i.e., multiple paths to single
    successor vertices) are handled correctly.

    """

    def __init__(self, *args, **kwargs) -> None:
        super(NetworkxSuccessorMixin, self).__init__(*args, **kwargs)

        if not hasattr(self.game, '_graph'):
            self.game._vertices_dict: dict = {self: self}
            self.game._graph: DiGraph = DiGraph()
            self.game._graph.add_vertex(self)

    def _canconicalize_vertex(self, vertex: 'MCTSVertex') -> 'MCTSVertex':
        """Fixes an odd issue in networkx where the vertices linked in an edge are not always the vertices stored in
        G.vertices """
        if vertex in self.game._vertices_dict:
            return self.game._vertices_dict[vertex]
        else:
            self.game._vertices_dict[vertex] = vertex
            return vertex

    @property
    def successors(self) -> List['MCTSVertex']:
        return list(self.game._graph.children(self))

    def expand(self) -> 'NetworkxSuccessorMixin':
        children = [self.__class__(action, self.game) for action in self._state.get_next_actions()]
        self.game._graph.add_edges_from(((self, self._canconicalize_vertex(child)) for child in children))
        return self
