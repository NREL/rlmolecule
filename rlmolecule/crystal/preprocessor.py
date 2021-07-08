import os
from typing import List

import numpy as np

from rlmolecule.crystal.crystal_state import CrystalState


conducting_ions = {'Li', 'Na', 'K', 'Mg', 'Zn'}
anions = {'F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'}
framework_cations = {'Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'}
default_elements = conducting_ions | anions | framework_cations

default_crystal_systems = {'triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic'}

dir_path = os.path.dirname(os.path.realpath(__file__))
default_proto_strc_names_file = os.path.join(dir_path, 'inputs', 'icsd_prototype_filenames.txt')
default_proto_strc_names = set()
with open(default_proto_strc_names_file, 'r') as f:
    for line in f:
        default_proto_strc_names.add(line.rstrip())


class CrystalPreprocessor:

    def __init__(self,
                 elements: List = None,
                 crystal_systems: List = None,
                 proto_strc_names: List = None,
                 max_stoich: int = 8):
        self.elements = elements if elements is not None else default_elements
        self.crystal_systems = crystal_systems if crystal_systems is not None else default_crystal_systems
        self.proto_strc_names = proto_strc_names if proto_strc_names is not None else default_proto_strc_names
        self.max_stoich = max_stoich

    #def build_preprocessor(self):
        elements_and_soich = [(ele + str(i)).replace('0', '')
                              for ele in self.elements
                              for i in range(self.max_stoich + 1)]
        self.element_mapping = {ele: i for i, ele in enumerate(elements_and_soich)}

        self.crystal_sys_mapping = {c: i for i, c in enumerate(self.crystal_systems)}
        self.proto_strc_mapping = {p: i for i, p in enumerate(self.proto_strc_names)}

    def construct_feature_matrices(self, state: CrystalState, train: bool = False) -> {}:
        """ Convert a crystal state to a list of tensors
        'eles_and_stoich' : (n_ele_and_stoich,) vector of the elements and their stoichiometries
        'crystal_sys' : (1,) vector of the crystal system index
        'proto_strc' : (1,) vector of the prototype structures index
        """

        # at the beginning of the action graph, the action node is a tuple of elements
        if state.composition is not None:
            eles = list(state.get_eles_from_comp(state.composition))
            stoich = list(state.get_stoich_from_comp(state.composition))
            eles_and_stoich = eles + [e + str(stoich[i]) for i, e in enumerate(eles)]
        else:
            eles_and_stoich = []
            if state.action_node != 'root':
                eles_and_stoich = state.action_node
                if isinstance(state.action_node, str):
                    eles_and_stoich = [state.action_node]

        element_features = []
        for ele in eles_and_stoich:
            element_features.append(self.element_mapping[ele])

        crystal_sys = state.get_crystal_sys()
        if crystal_sys is not None:
            crystal_sys = self.crystal_sys_mapping[crystal_sys] + 1
        else:
            crystal_sys = 0

        proto_strc = state.get_proto_strc()
        if proto_strc is not None:
            proto_strc = self.proto_strc_mapping[proto_strc] + 1
        else:
            proto_strc = 0

        return {'eles_and_stoich': np.asarray(element_features),
                'crystal_sys': np.asarray([crystal_sys]),
                'proto_strc': np.asarray([proto_strc]),
                }


        # return {'eles_and_stoich': np.asarray([element_features]),
        #         # pad zeros to the end so they're all the same size.
        #         # TODO zero's will be masked
        #         'crystal_sys': np.asarray([[crystal_sys] + [0] * (len(element_features) - 1)]),
        #         'proto_strc': np.asarray([[proto_strc] + [0] * (len(element_features) - 1)]),
        #         }
