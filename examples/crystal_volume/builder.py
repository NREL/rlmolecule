import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np
import networkx as nx

#from examples.crystal_volume.crystal_state import CrystalState
from examples.crystal_volume.crystal_state import CrystalState

logger = logging.getLogger(__name__)


class CrystalBuilder:
    def __init__(self,
                 G: nx.DiGraph,
                 G2: nx.DiGraph,
                 comp_to_comp_type,
                 #structures: dict[str, any],
                 ) -> None:
        """A class to build crystals according to a number of different options

        G: The first networkx graph going from element combinations to compositions
        G2: The second networkx graph going from composition types to decorations
        comp_to_comp_type: Mapping from a composition string to the composition type
        TODO structures: Mapping from a decoration string to a pymatgen structure object
        """
        self.G = G
        self.G2 = G2
        self.comp_to_comp_type = comp_to_comp_type
        #self.structures = structures
        #self.transformation_stack = [
        #    find_next_steps(),
        #]

    def __call__(self, parent_state: any) -> Iterable[any]:
        #inputs = [parent_state]
        #for transformer in self.transformation_stack:
        #    inputs = transformer(inputs)
        states = self.find_next_states(parent_state)
        yield from states


        # TODO fix the typing. should accept a single crystal state and return an Iterable of cyrstal states
    def find_next_states(self, crystal_state: any) -> Iterable[any]:
        """
        For the given state, find the next possible states in the action graphs
        """
        n = crystal_state.action_node
        next_states = []
        node_found = False
        if self.G.has_node(n):
            node_found = True
            # if we're at the end of the first graph, then pull actions from the second
            if self.G.out_degree(n) == 0:
                n = crystal_state.comp_type
            else:
                # for this first graph, the nodes are either a tuple of elements (e.g., (F, Li, Sc)),
                # or a composition of those elements (e.g., Li1Sc1F4)
                for neighbor in self.G.neighbors(n):
                    comp_type = self.comp_to_comp_type.get(neighbor)
                    composition = neighbor if comp_type is not None else None
                    elements = neighbor if comp_type is None else crystal_state.elements
                    next_state = CrystalState(elements=elements,
                                              composition=composition,
                                              comp_type=comp_type,
                                              action_node=neighbor,
                                              terminal=False,
                                              )
                    next_states.append(next_state)
                return next_states

        if self.G2.has_node(n):
            node_found = True
            # if we're at the end of the second graph, then we have reached the end of the action space
            # and we're at a decorated crystal structure. Just return the original state
            if self.G2.out_degree(n) == 0:
                crystal_state.terminal = True
                return [crystal_state]
            else:
                for neighbor in self.G2.neighbors(n):
                    terminal = False
                    structure = None
                    if self.G2.out_degree(neighbor) == 0:
                        terminal = True
                        # TODO add the structure here
                        #structure_key = crystal_state.composition + '|' + neighbor
                        #structure = structures[structure_key]
                    next_state = CrystalState(elements=crystal_state.elements,
                                              composition=crystal_state.composition,
                                              comp_type=crystal_state.comp_type,
                                              action_node=neighbor,
                                              terminal=terminal,
                                              #structure=structure,
                    )
                    next_states.append(next_state)
                return next_states

        if node_found is False:
            raise(f"StateNotFound: Action node '{n}' was not in either action graph.")
