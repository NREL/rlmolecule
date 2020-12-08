from abc import (
    ABC,
    abstractmethod,
    )
from typing import Iterable

import numpy as np


class GraphNode(ABC):
    
    @property
    def terminal(self) -> bool:
        """
        Should be overridden if get_successors() is not performant
        :return: True iff this node has no successors
        """
        return not any(True for _ in self.get_successors())
    
    @abstractmethod
    def get_successors(self) -> Iterable['GraphNode']:
        """
        This is a getter instead of a property because it is dynamically overridden in some cases,
        and it is cleaner and simpler to dynamically override a function rather than a property.
        :return: the nodes successors as an iterable.
        """
        pass
