import os
from typing import List

import numpy as np

from rlmolecule.crystal.crystal_state import CrystalState

conducting_ions = {'Li', 'Na', 'K'}
anions = {'F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'}
framework_cations = {'Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf',
                     'W', 'Zn', 'Cd', 'Hg', 'B', 'Al',
                     'Si', 'Ge', 'Sn', 'P', 'Sb'}
default_elements = conducting_ions | anions | framework_cations

default_crystal_systems = {'triclinic', 'monoclinic',
                           'orthorhombic', 'tetragonal',
                           'trigonal', 'hexagonal', 'cubic'}

dir_path = os.path.dirname(os.path.realpath(__file__))
default_proto_names_file = os.path.join(dir_path,
                                        'inputs',
                                        'icsd_prototype_filenames_lt50atoms_lt100dist.txt')
default_proto_strc_names = set()
with open(default_proto_names_file, 'r') as f:
    for line in f:
        default_proto_strc_names.add(line.rstrip())


class CrystalPreprocessor:

    def __init__(self,
                 elements: List = None,
                 crystal_systems: List = None,
                 proto_strc_names: List = None,
                 max_stoich: int = 8,
                 max_num_elements: int = 5):
        """ Class for processing the CrystalStates of the action graph
        to input to the policy model. Will convert the elements,
        their stoichiometries, the crystal system and prototype structure 
        to integer/vector representations.
        
        :param elements: Set of all possible elements
        :param crystal_systems: Set of all crystal systems
        :param proto_strc_names: Set of all prototype structure IDs
        :param max_stoich: Maximum stoichiometry among the composition types.
            For example, among the comopsition types _1_1_1_2, _2_3_6, and _1_7,
            the maximum stoichiometry is 7
        :param max_num_elements: Maximum number of elements among the composition types.
            For example, among the comopsition types _1_1_1_2, _2_3_6, and _1_7,
            the max is 4
        """
        self.elements = elements if elements is not None else default_elements
        self.crystal_systems = crystal_systems if crystal_systems is not None else default_crystal_systems
        self.proto_strc_names = proto_strc_names if proto_strc_names is not None else default_proto_strc_names
        self.max_stoich = max_stoich
        self.max_num_elements = max_num_elements

        self.element_mapping = {ele: i for i, ele in enumerate(self.elements)}
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
            assert len(eles) <= self.max_num_elements, \
                (f"Number of elements in composition must be "
                 f"<= max_num_elements {self.max_num_elements}. "
                 f"Given: {len(eles)} ({state = }) ")

            stoich = list(state.get_stoich_from_comp(state.composition))
            assert max(stoich) <= self.max_stoich, \
                (f"Maximum stoichiometry of elements in composition must be "
                 f"<= max_stoich {self.max_stoich}. "
                 f"Given: {max(stoich)} ({state = }) ")

        else:
            eles = []
            stoich = None
            if state.action_node != 'root':
                eles = state.action_node
                if isinstance(state.action_node, str):
                    eles = [state.action_node]

        # the element and stoichiometry feature vector will be
        # the integer mapping of an element followed by its stoichiometry
        ele_and_stoich_features = []
        for i, ele in enumerate(eles):
            ele_and_stoich_features += [self.element_mapping[ele] + 1]
            if stoich is not None:
                ele_and_stoich_features += [stoich[i]]
            else:
                ele_and_stoich_features += [0]

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

        # This vector has each element as well as its stoichiometry.
        # Since there can be anywhere from 1 to 5 elements,
        # maintain the length of the vector, and use 0 as the default
        eles_stoich_vec = np.zeros(self.max_num_elements * 2)
        eles_stoich_vec[:len(ele_and_stoich_features)] = np.asarray(
                ele_and_stoich_features,
                dtype=np.int64)

        return {'eles_and_stoich': eles_stoich_vec,
                'crystal_sys': np.asarray([crystal_sys], dtype=np.int64),
                'proto_strc': np.asarray([proto_strc], dtype=np.int64),
                }

        # return {'eles_and_stoich': np.asarray([element_features]),
        #         # pad zeros to the end so they're all the same size.
        #         # TODO zero's will be masked
        #         'crystal_sys': np.asarray([[crystal_sys] + [0] * (len(element_features) - 1)]),
        #         'proto_strc': np.asarray([[proto_strc] + [0] * (len(element_features) - 1)]),
        #         }
