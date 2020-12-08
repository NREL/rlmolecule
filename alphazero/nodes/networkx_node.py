from abc import abstractmethod
from typing import (
    Iterable,
    Iterator,
    TypeVar,
    final,
    )

from networkx import DiGraph

from alphazero.nodes.graph_node import GraphNode

"""
+ Override successors() to use graph memoization
+ successors also have successors() overridden
+ inherits all of delegate's properties


Option 1: generic subclass
    - but how to wrap delegate's result from successors() method?
Option 2: dynamically override successors() with "virus" method
    - hard to detect, a bit ugly
Option 3: delegate with dynamic dispatch on keys
    - but then doesn't pass type tests
    - and ugly hack to mimic delegate's interface


"""


class NetworkXNodeMemoizer:
    
    def __init__(self):
        self.__graph: DiGraph = DiGraph()
    
    def memoize(self, node: GraphNode) -> GraphNode:
        delegate_successors_getter = node.get_successors
        
        def memoized_successors_getter(parent: GraphNode) -> Iterable['NetworkNode']:
            return self.__graph.successors(self)
        
        def memoizing_successors_getter(parent: GraphNode) -> Iterable['NetworkNode']:
            self.__graph.add_edges_from(
                ((parent, self.memoize(successor)) for successor in delegate_successors_getter(parent)))
            parent.get_successors = memoized_successors_getter
            return parent.get_successors()
        
        node.get_successors = memoizing_successors_getter
        return node
    
    @property
    def graph(self) -> DiGraph:
        return self.__graph
