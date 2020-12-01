from abc import abstractmethod
from typing import (
    Iterable,
    Iterator,
    )

from networkx import DiGraph

from alphazero.node.abstract_node import AbstractNode


class NetworkXNode(AbstractNode):
    
    def __init__(self, networkx_graph: DiGraph) -> None:
        self.__networkx_graph: DiGraph = networkx_graph
        self.__open: bool = True  # true if node has not yet been expanded
        # self.__visits: int = 0
        # self.__total_value: float = 0.0
    
    # @property
    # def graph_node(self) -> 'NetworkXNode':
    #     return self.__networkx_graph.nodes[self]
    
    @property
    def successors(self) -> Iterable['NetworkXNode']:
        if self.open:
            self.__open = False
            successors = self._expand()
            self.__networkx_graph.add_edges_from(((self, successor) for successor in successors))
        return self.__networkx_graph.successors(self)
    
    # def update(self, reward:float) -> None:
    #     self.__visits += 1
    #     self.__total_value += reward
    #
    # @property
    # def visits(self) -> int:
    #     return self.__visits
    #
    # @property
    # def value(self) -> float:
    #     return self.__total_value / self.__visits if self.__visits > 0 else 0.0
    
    @property
    def open(self) -> bool:
        """
        :return: True if the node has not been expanded.
        """
        return self.__open
    
    @abstractmethod
    def _expand(self) -> Iterator['MoleculeNode']:
        """
        Generates successor nodes. Should only be called by the successors property when the node is open.
        :return: list of successor nodes. These do not need to be interned nodes.
        """
        pass
