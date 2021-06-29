import math
import pandas as pd
import random
import gzip

from examples.crystal_volume import optimize_crystal_volume as ocv
#from examples.crystal_volume.crystal_problem import CrystalTFAlphaZeroProblem
from examples.crystal_volume.crystal_state import CrystalState

#from rlmolecule.tree_search.reward import RankedRewardFactory


# these should have different decorations and different crystal volumes
nodes_to_check = ["Na1Br1|_1_1|POSCAR_sg5_icsd_068712|1", "Na1Br1|_1_1|POSCAR_sg5_icsd_068712|2"]

for action_node in nodes_to_check:
    # Now create the decoration of this composition onto this prototype structure
    # the 'action_node' string has the following format at this point:
    # comp_type|prototype_structure|decoration_idx
    # we just need 'comp_type|prototype_structure' to get the icsd structure
    composition = action_node.split('|')[0]
    structure_key = '|'.join(action_node.split('|')[1:-1])
    icsd_prototype = ocv.structures[structure_key]
    decoration_idx = int(action_node.split('|')[-1]) - 1
    print(action_node, composition, structure_key)
    try:
        decorated_structure, comp = CrystalState.decorate_prototype_structure(
            icsd_prototype, composition, decoration_idx=decoration_idx)
        #decorations[descriptor] = decorated_structure.as_dict()
    except AssertionError as e:
        print(f"AssertionError: {e}")
        #volume_stats[descriptor] = (-1, -1, 0, comp_type)
        #return 0.0, {'terminal': True, 'state_repr': repr(state)}
        continue

    # Compute the volume of the conducting ions.
    conducting_ion_vol, total_vol = ocv.compute_structure_vol(decorated_structure)
    frac_conducting_ion_vol = conducting_ion_vol / total_vol if total_vol != 0 else 0
    print(conducting_ion_vol, total_vol, frac_conducting_ion_vol)
    out_file = f"outputs/POSCAR_{action_node.replace('|','-')}"
    print(f"writing {out_file}")
    decorated_structure.to(filename=out_file)
