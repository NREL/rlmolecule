from typing import (
    Iterable,
    )

from networkx import DiGraph

from alphazero.graph_node import GraphNode




class NetworkXNodeMemoizer:
    """
    This class injects a networkx-based memoizer into a target GraphNode instance's get_successors() method.
    After injection, this node and all successors will be memoized in a networkx DiGraph.
    
    
    This approach does not feel very 'clean', but consider the options:
    
    Option 1: generic subclass
        - but how to wrap delegate's result from successors() method?
    Option 2: dynamically override successors() with "virus" method (this method)
        - hard to detect, a bit unclean
    Option 3: delegate with dynamic dispatch on keys
        - but then doesn't pass type tests
        - and ugly hack to mimic delegate's interface
    Option 4: include memoization in GraphNode or Game implementations
        - breaks encapsulation
        - hardcodes networkx memoization
            - no easy way to swap out other memoization or caching strategies
        - the ugliest of these approaches due to lack of modularity and intrusive nature
    """
    
    def __init__(self):
        self.__graph: DiGraph = DiGraph()
    
    def memoize(self, node: GraphNode) -> GraphNode:
        """
        Injects the memoizer into the given node.
        After injection, this node and all successors will be memoized in a networkx DiGraph.
        :param node:
        :return: target node
        """
        
        delegate_successors_getter = node.get_successors
        
        def memoized_successors_getter(parent: GraphNode) -> Iterable['NetworkNode']:
            return self.__graph.successors(parent)
        
        def memoizing_successors_getter(parent: GraphNode) -> Iterable['NetworkNode']:
            self.__graph.add_edges_from(
                ((parent, self.memoize(successor)) for successor in delegate_successors_getter()))
            parent.get_successors = memoized_successors_getter.__get__(parent, GraphNode)
            return parent.get_successors()
        
        node.get_successors = memoizing_successors_getter.__get__(node, GraphNode)
        return node
    
    @property
    def graph(self) -> DiGraph:
        """
        :return: The networkx DiGraph used for memoization.
        """
        return self.__graph
