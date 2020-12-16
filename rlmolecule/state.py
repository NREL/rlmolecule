from abc import (
    ABC,
    abstractmethod,
    )
from typing import (
    Iterable,
    # final
    )


class State(ABC):
    """
    Simply defines a directed graph structure which is incrementally navigated via the get_successors() method.
    """
    
    # @final
    def __eq__(self, other: any) -> bool:
        return isinstance(other, State) and self.equals(other)
    
    # @final
    def __hash__(self) -> int:
        return self.hash()
    
    @abstractmethod
    def equals(self, other: 'State') -> bool:
        """
        Equality method which must be implemented by subclasses.
        Used when memoizing and traversing the graph structure to ensure that only one instance of the same node exists.
        :return: true iff this node should be treated as the same node in the graph as the other node.
        """
        pass
    
    @abstractmethod
    def hash(self) -> int:
        """
        Hash method which must be implemented by subclasses.
        Used when memoizing and traversing the graph structure to ensure that only one instance of the same node exists.
        :return: a valid hash value for this node
        """
        pass
    
    @abstractmethod
    def get_next_actions(self) -> Iterable['State']:
        """
        This is a getter instead of a property because it is dynamically overridden in some cases,
        and it is cleaner and simpler to dynamically override a function rather than a property.
        :return: the nodes successors as an iterable.
        """
        pass

    @property
    def terminal(self) -> bool:
        """
        Should be overridden if get_successors() is not performant
        :return: True iff this node has no successors
        """
        return not any(True for _ in self.get_next_actions())
