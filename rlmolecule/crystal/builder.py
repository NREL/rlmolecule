import logging
import os
from typing import Iterable, Optional

import networkx as nx

from rlmolecule.crystal.crystal_state import CrystalState

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
action_graph_file = os.path.join(dir_path, 'inputs', 'elements_to_compositions.edgelist.gz')
action_graph2_file = os.path.join(dir_path, 'inputs', 'comp_type_to_decorations.edgelist.gz')


class CrystalBuilder:
    def __init__(self,
                 G: Optional[nx.DiGraph] = None,
                 G2: Optional[nx.DiGraph] = None,
                 ) -> None:
        """A class to build crystals according to a number of different options

        G: The first networkx graph going from element combinations to compositions
        G2: The second networkx graph going from composition types to decorations
        comp_to_comp_type: Mapping from a composition string to the composition type
        structures: Mapping from a decoration string to a pymatgen structure object
        """

        if G is None:
            G = nx.read_edgelist(action_graph_file,
                                 delimiter='\t',
                                 data=False,
                                 create_using=nx.DiGraph())

        if G2 is None:
            G2 = nx.read_edgelist(action_graph2_file,
                                  delimiter='\t',
                                  data=False,
                                  create_using=nx.DiGraph())

        self.G = G
        self.G2 = G2
        # Update: build the comp_to_comp_type dictionary on the fly
        # the first action graph G ends in the compositions, so we can extract those using the out degree
        compositions = [n for n in G.nodes() if G.out_degree(n) == 0]
        self.comp_to_comp_type = {c: CrystalState.split_comp_to_eles_and_type(c)[1] for c in compositions}
        # self.transformation_stack = [
        #    find_next_steps(),
        # ]

    def __call__(self, parent_state: any) -> Iterable[any]:
        # inputs = [parent_state]
        # for transformer in self.transformation_stack:
        #    inputs = transformer(inputs)
        states = self.find_next_states(parent_state)
        yield from states

    def find_next_states(self, crystal_state: CrystalState) -> Iterable[CrystalState]:
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
                n = self.comp_to_comp_type[crystal_state.composition]
            else:
                # for this first graph, the nodes are either a tuple of elements (e.g., (F, Li, Sc)),
                # or a composition of those elements (e.g., Li1Sc1F4)
                for neighbor in self.G.neighbors(n):
                    comp_type = self.comp_to_comp_type.get(neighbor)
                    composition = neighbor if comp_type is not None else None
                    next_state = CrystalState(action_node=neighbor,
                                              builder=self,
                                              composition=composition,
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
                        # TODO cache the decorated structuers?
                        # There could be over 15M which would take a lot of RAM
                        # instead of caching the structes, I can just cache the end reward value
                        # decorated_structure = self.decorated_structures.get()
                        # structure_key = crystal_state.composition + '|' + crystal_state.action_node
                        # icsd_prototype = self.icsd_structures[structure_key]
                    next_state = CrystalState(action_node=neighbor,
                                              builder=self,
                                              composition=crystal_state.composition,
                                              # structure=decorated_structure,
                                              terminal=terminal,
                                              )
                    next_states.append(next_state)
                return next_states

        if node_found is False:
            raise (f"StateNotFound: Action node '{n}' was not in either action graph.")
