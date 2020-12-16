from typing import Iterable

from networkx import DiGraph


class NetworkxSuccessorMixin(object):
    """ Class to add new nodes to a networkx DiGraph, such that node transpositions (i.e., multiple paths to single
    successor nodes) are handled correctly.

    """

    def __init__(self, *args, **kwargs) -> None:
        super(NetworkxSuccessorMixin, self).__init__(*args, **kwargs)

        if not hasattr(self.game, '_graph'):
            self.game._graph: DiGraph = DiGraph()

    def get_successors(self) -> Iterable['NetworkxSuccessorMixin']:

        if self._successors is None:
            children = [self.__class__(action, self.game) for action in self.state.get_next_actions()]
            self.game._graph.add_edges_from([(self, child) for child in children])
            self._successors = list(self.game._graph.successors(self))

        return self._successors