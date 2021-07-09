import itertools
import re
from copy import deepcopy
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pymatgen.core import Composition, Structure

from rlmolecule.sql import hash_to_integer
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.metrics import collect_metrics


crystal_systems = {'triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic'}


class CrystalState(GraphSearchState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Crystals are generated only at the final state
    """

    def __init__(
            self,
            action_node: any,
            composition: Optional[str] = None,
            # structure: Optional[Structure] = None,
            terminal: bool = False,
    ) -> None:
        """
        :param action_node: A representation of the current state in one of the action graphs
            e.g., 'Zn1Hg1Al1F1Cl6' in the first graph, or '1_1_1_1_6|cubic' of the second graph
        :param terminal: Whether this state is a decoration of a specific structure (i.e., final state)
        """
        self._action_node: any = action_node
        self._composition: str = composition
        # self._structure: Optional[Structure] = structure
        self._terminal: bool = terminal

    def __repr__(self) -> str:
        """
        Uses the string representation of the current state
        """
        comp_str = self._composition + '|' if self._composition is not None else ""
        comp_str = "" if self._composition == self._action_node else comp_str
        return comp_str + str(self._action_node)

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
    # Storing the builder as part of the crystal state was really slowing down the serialization step.
    # so just pass the builder here
    def get_next_actions(self, builder) -> Sequence['CrystalState']:
        """
        :param builder: A CrystalBuilder class
        """
        result = []
        if not self._terminal:
            result.extend(builder(self))

        return result

    @staticmethod
    def split_comp_to_eles_and_type(comp: str) -> Tuple[Iterable[str], str]:
        """
        Extract the elements and composition type from a given composition
        e.g., _1_1_4 from Li1Sc1F4
        """
        # this splits by the digits
        # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
        split = np.asarray(re.split('(\d+)', comp))
        elements = tuple(sorted(split[range(0, len(split) - 1, 2)]))
        stoich = split[range(1, len(split), 2)]
        # sort the stoichiometry to get the correct order of the comp type
        comp_type = '_' + '_'.join(map(str, sorted(map(int, stoich))))
        return elements, comp_type

    @staticmethod
    def get_stoich_from_comp(comp: str) -> Iterable[int]:
        # split by the digits
        # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
        split = np.asarray(re.split('(\d+)', comp))
        stoich = tuple(map(int, split[range(1, len(split), 2)]))
        return stoich

    @staticmethod
    def get_eles_from_comp(comp: str) -> Iterable[str]:
        # split by the digits
        # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
        split = np.asarray(re.split('(\d+)', comp))
        eles = tuple(split[range(0, len(split) - 1, 2)])
        return eles

    @staticmethod
    def decorate_prototype_structure(icsd_prototype: Structure,
                                     composition: str,
                                     decoration_idx: int,
                                     ) -> Structure:
        """
        Replace the atoms in the icsd prototype structure with the elements of this composition.
        The decoration index is used to figure out which combination of
        elements in this composition should replace the elements in the icsd prototype
        """
        comp = Composition(composition)

        prototype_comp = Composition(icsd_prototype.formula).reduced_composition
        prototype_stoic = tuple([int(p) for p in prototype_comp.formula if p.isdigit()])

        # create permutations of order of elements within a composition
        # e.g., for K1Br1: [('K1', 'Br1'), ('Br1', 'K1')]
        comp_permutations = itertools.permutations(comp.formula.split(' '))
        # only keep the permutations that match the stoichiometry of the prototype structure
        valid_comp_permutations = []
        for comp_permu in comp_permutations:
            comp_stoich = CrystalState.get_stoich_from_comp(''.join(comp_permu))
            if comp_stoich == prototype_stoic:
                valid_comp_permutations.append(''.join(comp_permu))

        # TODO find a better way than matching the decoration_idx here
        assert decoration_idx < len(valid_comp_permutations), \
            f"decoration_idx {decoration_idx} must be < num valid comp permutations {len(valid_comp_permutations)} -- " + \
            f"prototype_stoic: {prototype_stoic}, comp_permu: {comp_permu}, " + \
            f"composition: {composition}, prototype_comp: {prototype_comp} "

        # now build the decorated structure for the specific index passed in
        original_ele = ''.join(i for i in prototype_comp.formula if not i.isdigit()).split(' ')
        replacement_ele = CrystalState.get_eles_from_comp(valid_comp_permutations[decoration_idx])

        # dictionary containing original elements as keys and new elements as values
        replacement = {original_ele[i]: replacement_ele[i] for i in range(len(original_ele))}

        # 'replace_species' function from pymatgen to replace original elements with new elements
        strc_subs = deepcopy(icsd_prototype)
        strc_subs.replace_species(replacement)

        return strc_subs, valid_comp_permutations[decoration_idx]

    # @property
    # def elements(self) -> str:
    #     return self._elements

    @property
    def composition(self) -> str:
        return self._composition

    # @property
    # def comp_type(self) -> str:
    #     return self._comp_type

    @property
    def action_node(self) -> Union[str, tuple]:
        return self._action_node

    def get_crystal_sys(self) -> str:
        if '|' not in self.action_node:
            return None
        # extract the crystal system str from the action node
        crystal_sys_str = self.action_node.split('|')[1]
        assert crystal_sys_str in crystal_systems
        return crystal_sys_str

    def get_proto_strc(self) -> str:
        if '|' not in self.action_node or len(self.action_node.split('|')) < 3:
            return None
        # extract the prototype structure str from the action node
        proto_strc_str = self.action_node.split('|')[2]
        # TODO add a check here to ensure this is a valid prototype structure string e.g., POSCAR_sg14_icsd_083588
        return proto_strc_str

    @property
    def terminal(self) -> bool:
        return self._terminal
