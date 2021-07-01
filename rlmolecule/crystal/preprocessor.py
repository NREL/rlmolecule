from typing import List

import numpy as np

from rlmolecule.crystal.crystal_state import CrystalState


class CrystalPreprocessor:

    def __init__(self,
                 elements: List,
                 crystal_systems: List,
                 prototype_structure_names: List,
                 max_stoich: int = 8):
        self.elements = elements
        self.crystal_systems = crystal_systems
        self.prototype_structure_names = prototype_structure_names
        self.max_stoich = max_stoich

    #def build_preprocessor(self):
        elements_and_soich = [(ele + str(i)).replace('0', '') \
                              for ele in self.elements \
                              for i in range(self.max_stoich + 1)]
        self.element_mapping = {ele: i for i, ele in enumerate(elements_and_soich)}

        self.crystal_sys_mapping = {c: i for i, c in enumerate(self.crystal_systems)}
        self.proto_strc_mapping = {p: i for i, p in enumerate(self.prototype_structure_names)}

    def construct_feature_matrices(self, state: CrystalState, train: bool = False) -> {}:
        """ Convert a crystal state to a list of tensors
        'element' : (n_ele_and_stoich,) length list of atom classes
        'crystal_sys' : (n_crystal_sys,) length list of crystal systems
        'proto_strc' : (n_proto_strc, ) length list of prototype structures
        """

        # at the beginning of the action graph, the action node is a tuple of elements
        eles_and_stoich = state.action_node
        if state.composition is not None:
            eles = list(state.get_eles_from_comp(state.composition))
            stoich = list(state.get_stoich_from_comp(state.composition))
            eles_and_stoich = eles + [e + str(stoich[i]) for i, e in enumerate(eles)]

        element_feature_vector = np.zeros(len(self.element_mapping), dtype='int64')
        for ele in eles_and_stoich:
            element_feature_vector[self.element_mapping[ele]] = 1

        crystal_sys_feature_vector = np.zeros(len(self.crystal_sys_mapping), dtype='int64')
        crystal_sys = state.get_crystal_sys()
        if crystal_sys is not None:
            crystal_sys_feature_vector[self.crystal_sys_mapping[crystal_sys]] = 1

        proto_strc_feature_vector = np.zeros(len(self.proto_strc_mapping), dtype='int64')
        proto_strc = state.get_proto_strc()
        if proto_strc is not None:
            proto_strc_feature_vector[self.proto_strc_mapping[proto_strc]] = 1

        return {'element': element_feature_vector,
                'crystal_sys': crystal_sys_feature_vector,
                'proto_strc': proto_strc_feature_vector,
                }