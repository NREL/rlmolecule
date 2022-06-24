import logging
import os, sys
from typing import Iterable, Optional, Union
from collections import defaultdict

import networkx as nx

from rlmolecule.crystal.crystal_state import CrystalState

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
# elements to compositions
default_action_graph_file = os.path.join(dir_path,
                                 'inputs',
                                 'eles_to_comps_lt50atoms_lt100dist.edgelist.gz')
# comp_type to decorations
default_action_graph2_file = os.path.join(dir_path,
                                  'inputs',
                                  'comp_type_to_decors_lt50atoms_lt100dist.edgelist.gz')


class CrystalBuilder:
    def __init__(self,
                 G: Optional[Union[nx.DiGraph, str]] = None,
                 G2: Optional[Union[nx.DiGraph, str]] = None,
                 actions_to_ignore: Optional[set] = None,
                 prototypes: Optional[dict] = None,
                 ) -> None:
        """A class to build crystals according to a number of different options

        G: The first networkx graph going from element combinations to compositions
        G2: The second networkx graph going from composition types to decorations
        actions_to_ignore: Action nodes included in this set will not be explored further
            e.g., '_1_2', 'Zn', '1_1_1_1_6|cubic'.
        Can also specify state representations 
            e.g., Mg1Sc1F5|1_1_5|Orthorhombic|POSCAR_sg62_icsd_261430|2
        structures: Mapping from a decoration string to a pymatgen structure object
        """

        if G is None or isinstance(G, str):
            action_graph_file = G if isinstance(G, str) else default_action_graph_file
            G = nx.read_edgelist(action_graph_file,
                                 delimiter='\t',
                                 data=False,
                                 create_using=nx.DiGraph())
            # some of the nodes are meant to be tuples. Fix that here
            nx.relabel_nodes(G, {n: eval(n) for n in G.nodes(data=False) if '(' in n}, copy=False)
            print(f"Read G1: {action_graph_file} "
                  f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

        if G2 is None or isinstance(G2, str):
            action_graph2_file = G2 if isinstance(G2, str) else default_action_graph2_file
            G2 = nx.read_edgelist(action_graph2_file,
                                  delimiter='\t',
                                  data=False,
                                  create_using=nx.DiGraph())
            print(f"Read G2: {action_graph2_file} "
                  f"({G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges)")

        self.G = G
        self.G2 = G2
        self.prototypes = prototypes

        # Update: build the comp_to_comp_type dictionary on the fly
        # the first action graph G ends in the compositions, so we can extract those using the out degree
        compositions = [n for n in G.nodes() if G.out_degree(n) == 0]
        self.comp_to_comp_type = {c: CrystalState.split_comp_to_eles_and_type(c)[1] for c in compositions}
        self.comp_type_to_comp = defaultdict(set)
        for comp, comp_type in self.comp_to_comp_type.items():
            self.comp_type_to_comp[comp_type].add(comp)
        # self.transformation_stack = [
        #    find_next_steps(),
        # ]

        self.actions_to_ignore = set()
        self.actions_to_ignore_G2 = set()
        self.states_to_ignore = defaultdict(set)
        actions_not_recognized = set()
        if actions_to_ignore is not None:
            for a in actions_to_ignore:
                # if a state that has a composition and an action from G2 is given, skip those separately 
                # e.g., Mg1Sc1F5|1_1_5|Orthorhombic|POSCAR_sg62_icsd_261430|2
                if isinstance(a, str):
                    comp = a.split('|')[0]
                    action_node = '|'.join(a.split('|')[1:])
                if G.has_node(a):
                    self.actions_to_ignore.add(a)
                elif G2.has_node(a):
                    self.actions_to_ignore_G2.add(a)
                elif isinstance(a, str) and \
                     G.has_node(comp) and \
                     G2.has_node(action_node):
                    self.states_to_ignore[comp].add(action_node)
                else:
                    actions_not_recognized.add(a)
            print(f"{len(self.actions_to_ignore)} and {len(self.actions_to_ignore_G2)} "
                   "actions to ignore in G and G2, respectively")
            if len(self.states_to_ignore) > 0:
                print(f"{len(self.states_to_ignore)} states to ignore")
            if len(actions_not_recognized) > 0:
                print(f"WARNING: {len(actions_not_recognized)} actions_to_ignore not recognized")
                if len(actions_not_recognized) < 50:
                    print(actions_not_recognized)
            # check to make sure there are no dead-ends e.g., compositions with no prototypes available
            self.remove_dead_ends()

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
        node_found_G = False
        node_found_G2 = False
        if self.G.has_node(n):
            node_found_G = True
            # if we're at the end of the first graph, then pull actions from the second
            if self.G.out_degree(n) == 0:
                n = self.comp_to_comp_type[crystal_state.composition]
            else:
                # for this first graph, the nodes are either a tuple of elements (e.g., (F, Li, Sc)),
                # or a composition of those elements (e.g., Li1Sc1F4)
                for neighbor in self.G.neighbors(n):
                    comp_type = self.comp_to_comp_type.get(neighbor)
                    composition = neighbor if comp_type is not None else None
                    # if this is not a branch of the tree we want to follow, then skip it
                    if neighbor in self.actions_to_ignore:
                        continue
                    next_state = CrystalState(action_node=neighbor,
                                              composition=composition,
                                              )
                    next_states.append(next_state)
                return next_states

        if self.G2.has_node(n):
            node_found_G2 = True
            # if we're at the end of the second graph, then we have reached the end of the action space
            # and we're at a decorated crystal structure. Just return the original state
            if self.G2.out_degree(n) == 0:
                crystal_state.terminal = True
                return [crystal_state]
            else:
                for neighbor in self.G2.neighbors(n):
                    if neighbor in self.actions_to_ignore_G2:
                        continue
                    if crystal_state.composition in self.states_to_ignore \
                       and neighbor in self.states_to_ignore[crystal_state.composition]:
                        continue
                    terminal = False
                    if self.G2.out_degree(neighbor) == 0:
                        terminal = True
                    next_state = CrystalState(action_node=neighbor,
                                              composition=crystal_state.composition,
                                              # structure=decorated_structure,
                                              terminal=terminal,
                                              )
                    if terminal and self.prototypes is not None:
                        # add the element replacements to the state
                        structure_key = '|'.join(next_state.action_node.split('|')[:-1])
                        icsd_prototype = self.prototypes[structure_key]
                        next_state.set_proto_ele_replacements(icsd_prototype)
                    next_states.append(next_state)
                return next_states

        if node_found_G is False and node_found_G2 is False:
            raise (f"StateNotFound: Action node '{n}' was not in either action graph.")
        elif node_found_G2 is False:
            raise (f"StateNotFound: Action node '{crystal_state.action_node}' "
                   f"comp_type '{n}' not found in G2")

    def remove_dead_ends(self,
                         #new_actions_to_ignore=None,
                         #new_actions_to_ignore_G2=None,
                         #new_states_to_ignore=None,
                         ):
        """ The action graph should only have paths that end at a decoration 
        where terminal would be set to True.
        If there are any artificial dead-ends created by actions_to_ignore, 
        handle those here by adding the parent nodes to actions_to_ignore.

        Three main possibilities:
        1. element combination with no composition
        2. prototypes with no decorations
        3. compositions with no prototype structures
        """
        # 2. prototypes with no decorations
        #if new_actions_to_ignore_G2:
        self.actions_to_ignore_G2 |= self.find_dead_ends(self.G2, self.actions_to_ignore_G2)
        # check the actions_to_ignore_G2 for composition_types
        for a in self.actions_to_ignore_G2:
            comps = self.comp_type_to_comp.get(a)
            if comps:
                self.actions_to_ignore.update(comps)

        # 1. element combination with no composition
        self.actions_to_ignore |= self.find_dead_ends(self.G, self.actions_to_ignore)

        # for each comp, then we can check G2 the same as actions_to_ignore_G2
        for comp, action_nodes in self.states_to_ignore.items():
            self.states_to_ignore[comp] |= self.find_dead_ends(self.G2, action_nodes)

        # If the state would already be ignored, then skip it
        states_to_ignore_update = defaultdict(set)
        for comp, action_nodes in self.states_to_ignore.items():
            if comp in self.actions_to_ignore:
                continue
            for a in action_nodes:
                if a in self.actions_to_ignore_G2:
                    continue
                states_to_ignore_update[comp].add(a)
        self.states_to_ignore = states_to_ignore_update

        # 3. Cleanup: Check if any compositions no longer have a prototype structure available
        # 3a. check the states_to_ignore dict
        comps = set(self.states_to_ignore.keys())
        for c in comps:
            # The comp_type is the first action_node of the second graph.
            # If its skipped, then also skip the composition,
            # and remove the comp from states_to_ignore
            comp_type = self.comp_to_comp_type[c]
            if comp_type in self.states_to_ignore[c]:
                self.actions_to_ignore.add(c)
                del self.states_to_ignore[c]

        return

    def find_dead_ends(self, G, actions_to_ignore):
        parents_to_ignore = set()
        parents_checked = set()
        for a in actions_to_ignore:
            # TODO for some reason, some of the reward states have the action_node set to an empty string ''
            # skip for now
            if a == '':
                continue
            parents = set(G.predecessors(a))
            parents -= parents_checked
            for p in parents:
                parents_checked.add(p)
                # if the children of p are all ignored,
                # then p should also be ignored
                if len(set(G.neighbors(p)) - actions_to_ignore) == 0:
                    parents_to_ignore.add(p)
        if len(parents_to_ignore) == 0:
            return parents_to_ignore

        # remove the child nodes since they can't be reached
        for p in parents_to_ignore:
            actions_to_ignore -= set(G.neighbors(p))

        # now recursively check their parents
        parents_to_ignore |= self.find_dead_ends(G, parents_to_ignore)
        return parents_to_ignore

