from typing import List

from networkx import DiGraph


class NetworkxSuccessorMixin(object):
    """ Class to add new nodes to a networkx DiGraph, such that node transpositions (i.e., multiple paths to single
    successor nodes) are handled correctly.

    """

    def __init__(self, *args, **kwargs) -> None:
        super(NetworkxSuccessorMixin, self).__init__(*args, **kwargs)

        if not hasattr(self.game, '_graph'):
            self.game._nodes_dict: dict = {self: self}
            self.game._graph: DiGraph = DiGraph()
            self.game._graph.add_node(self)

    def _canconicalize_node(self, node: 'MCTSNode') -> 'MCTSNode':
        """Fixes an odd issue in networkx where the nodes linked in an edge are not always the nodes stored in
        G.nodes """
        if node in self.game._nodes_dict:
            return self.game._nodes_dict[node]
        else:
            self.game._nodes_dict[node] = node
            return node

    @property
    def successors(self) -> List['MCTSNode']:
        return list(self.game._graph.successors(self))

    def expand(self) -> 'NetworkxSuccessorMixin':
        children = [self.__class__(action, self.game) for action in self._state.get_next_actions()]
        self.game._graph.add_edges_from(((self, self._canconicalize_node(child)) for child in children))
        return self
