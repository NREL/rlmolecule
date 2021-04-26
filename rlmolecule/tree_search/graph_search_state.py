import pickle
import zlib
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Sequence,
    # final
)


class GraphSearchState(ABC):
    """
    Simply defines a directed graph structure which is incrementally navigated via the get_successors() method.
    """
    @abstractmethod
    def equals(self, other: 'GraphSearchState') -> bool:
        """
        Equality method which must be implemented by subclasses.
        Used when memoizing and traversing the graph structure to ensure that only one instance of the same state exists.
        :return: true iff this state should be treated as the same state in the graph as the other state.
        """
        pass

    @abstractmethod
    def get_next_actions(self) -> Sequence['GraphSearchState']:
        """
        Defines the next possible states that are reachable from the current state. Should return nothing if the
        state is a final or terminal state, where a reward should be calculated.
        :return: the state's successors as an iterable.
        """
        pass

    @abstractmethod
    def hash(self) -> int:
        """
        Hash method which must be implemented by subclasses. Used when memoizing and traversing the graph structure
        to ensure that only one instance of the same state exists.

        :return: a valid hash value for this state
        """
        pass

    def serialize(self) -> bytes:
        """
        Convert the state to a unique string representation of the state, sufficient to recreate the state with
        GraphSearchState.deserialize(). Defaults to using python's pickle and base64 encoding. For non-pickleable
        objects, __getstate__ and __setstate__ methods can also be provided.

        :return: A string representation of the state
        """
        return zlib.compress(pickle.dumps(self))

    @staticmethod
    def deserialize(data: bytes) -> 'GraphSearchState':
        """
        Create an instance of the class from a serialized string. Defaults to assuming the data was stored using
        python's pickle and base64 encoding.

        :param data: A string representation of the state
        :return: An initialized GraphSearchState instance
        """
        return pickle.loads(zlib.decompress(data))

    # @final
    def __eq__(self, other: any) -> bool:
        return isinstance(other, GraphSearchState) and self.equals(other)

    # @final
    def __hash__(self) -> int:
        return self.hash()

    # hash_function = hashlib.sha256
    #
    # def digest(self) -> str:
    #     """
    #     A message digest used to compare serialized states.
    #
    #     :return: a valid hash value for this state
    #     """
    #     m = self.hash_function()
    #     m.update(self.serialize())  # todo: should we cache the serialized data?
    #     return m.hexdigest()
