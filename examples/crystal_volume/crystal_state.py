from typing import Tuple, Optional, Sequence

from rlmolecule.sql import hash_to_integer
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.metrics import collect_metrics


class CrystalState(GraphSearchState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Crystals are generated only at the final state
    """
    def __init__(
        self,
        elements: Tuple,
        action_node: any,
        builder: any,
        composition: Optional[str] = None,
        comp_type: Optional[str] = None,
        terminal: bool = False,
        #structure: any = None,
    ) -> None:
        """
        :param builder: A CrystalBuilder class
        :param action_node: A representation of the current state in one of the action graphs
        :param terminal: Whether this state is a decoration of a specific structure (i.e., final state)
        """
        self._builder: any = builder
        self._elements: str = elements
        self._composition: str = composition
        self._comp_type: str = comp_type
        self._action_node: any = action_node
        self._terminal: bool = terminal

    def __repr__(self) -> str:
        """
        Uses the string representation of the current state
        """
        comp_str = self._composition + '|' if self._composition is not None else ""
        comp_str = "" if self._composition == self._action_node else comp_str
        return comp_str + self._action_node

    # noinspection PyUnresolvedReferences
    def equals(self, other: any) -> bool:
        # the state representations are unique in the first graph,
        # but in the second graph, we need the composition to tell them apart
        # e.g., state_repr: '_1_1_1_1_6|POSCAR_sg11_icsd_084411'
        # with the composition: 'Zn1Hg1Al1F1Cl6'
        return type(other) == type(self) and \
               self._action_node == other._action_node and \
               self._composition == other._composition and \
               self._terminal == other._terminal

    def hash(self) -> int:
        return hash_to_integer(self.__repr__().encode())

    @collect_metrics
    def get_next_actions(self) -> Sequence['CrystalState']:
        result = []
        if not self._terminal:
            result.extend(self.builder(self))

        return result

    @property
    def elements(self) -> str:
        return self._elements

    @property
    def composition(self) -> str:
        return self._composition

    @property
    def comp_type(self) -> str:
        return self._comp_type

    @property
    def action_node(self) -> str:
        return self._action_node

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def builder(self) -> any:
        return self._builder

