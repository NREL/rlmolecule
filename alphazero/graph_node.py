from abc import (
    ABC,
    abstractmethod,
    )
from typing import Iterable


class GraphNode(ABC):
    """
    Simply defines a directed graph structure which is incrementally navigated via the get_successors() method.
    """
    
    @abstractmethod
    def get_successors(self) -> Iterable['GraphNode']:
        """
        This is a getter instead of a property because it is dynamically overridden in some cases,
        and it is cleaner and simpler to dynamically override a function rather than a property.
        :return: the nodes successors as an iterable.
        """
        pass
    
    def get_successors_list(self) -> ['GraphNode']:
        """
        Syntatic sugar for list(node.get_successors())
        :return: list of successor nodes
        """
        return list(self.get_successors())
    
    @property
    def terminal(self) -> bool:
        """
        Should be overridden if get_successors() is not performant
        :return: True iff this node has no successors
        """
        return not any(True for _ in self.get_successors())
