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
        Used when memoizing and traversing the graph structure to ensure that only one instance of the same state exists.
        :return: true iff this state should be treated as the same state in the graph as the other state.
        """
        pass

    @abstractmethod
    def hash(self) -> int:
        """
        Hash method which must be implemented by subclasses.
        Used when memoizing and traversing the graph structure to ensure that only one instance of the same state exists.
        :return: a valid hash value for this state
        """
        pass

    @abstractmethod
    def get_next_actions(self) -> Iterable['State']:
        """
        Defines the next possible states that are reachable from the current state. Should return nothing if the
        state is a final or terminal state, where a reward should be calculated.
        :return: the state's successors as an iterable.
        """
        pass

    @property
    def terminal(self) -> bool:
        """
        Should be overridden if get_successors() is not performant
        :return: True iff this state has no successors
        """
        return not any(True for _ in self.get_next_actions())
