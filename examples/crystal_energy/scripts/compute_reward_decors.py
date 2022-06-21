
### Generate decorations and compute the reward for each one

import argparse
import logging
import os
import sys
from tqdm import tqdm
import random
import json
import gzip
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nfp import custom_objects
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor

from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PDEntry

from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.crystal.preprocessor import CrystalPreprocessor
from rlmolecule.crystal.ehull import fere_entries
from rlmolecule.sql.run_config import RunConfig
from rlmolecule.crystal.crystal_reward import CrystalStateReward
from rlmolecule.crystal import utils, crystal_reward, reward_utils
from rlmolecule.crystal.ehull import fere_entries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#print(np.__version__)
#print(pd.__version__)


class AtomicNumberPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num


def read_structures_file(structures_file):
    logger.info(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    logger.info(f"\t{len(structures)} structures read")
    return structures


def generate_structures(decor_ids, rewarder):
    for decor_id in tqdm(decor_ids):
        # generate the decorated structure and get the reward
        #Example state:
        # CrystalState("_1_3_6|trigonal|icsd_401335|1", composition="Li3Sc1Br6", terminal=True)
        comp = decor_id.split('|')[0]
        action_node = '|'.join(decor_id.split('|')[1:])
        state = CrystalState(action_node, composition=comp, terminal=True)
        # rather than compute the reward, just generate the structure
        #reward, info = rewarder.get_reward(state)
        #if info_to_keep is not None:
        #    info = [round(info[c], 3) for c in info_to_keep if c in info]
        #else:
        #    info = [round(val, 3) for key, val in info.items()]

        #yield(tuple([decor_id, round(reward, 3)] + info))

        structure = rewarder.generate_structure(state)
        yield(decor_id, structure)


def compute_rewards(decor_ids, rewarder, info_to_keep=None):
    for decor_id in tqdm(decor_ids):
        # generate the decorated structure and get the reward
        #Example state:
        # CrystalState("_1_3_6|trigonal|icsd_401335|1", composition="Li3Sc1Br6", terminal=True)
        comp = decor_id.split('|')[0]
        action_node = '|'.join(decor_id.split('|')[1:])
        state = CrystalState(action_node, composition=comp, terminal=True)
        reward, info = rewarder.get_reward(state)
        if info_to_keep is not None:
            info = [round(info[c], 3) for c in info_to_keep if c in info]
        else:
            info = [round(val, 3) for key, val in info.items()]

        yield(tuple([decor_id, round(reward, 3)] + info))


class GenerateDecorations:
    def __init__(self,
                 builder: CrystalBuilder,
                 rewarder: CrystalStateReward = None,
                 root_state='root',
                 visited=None,
                 ):
        """ enumerate all possible decorations the crystal builder
        """
        self.builder = builder
        self.rewarder = rewarder
        self.root_state = CrystalState(root_state)

        n = 15*10**6  # estimated number of decorations
        self.progress_bar = tqdm(total=n)
        self.visited = visited
        if self.visited is None:
            self.visited = set()
        self.info_to_keep = ['predicted_energy',
                             'decomp_energy',
                             'cond_ion_frac',
                             'reduction',
                             'oxidation',
                             'stability_window',
                             ]

    def generate_all_decorations(self):
        yield from self.generate_decorations(self.root_state)

    def generate_decorations(self, state):
        """ DFS to generate all decorations (state string only) from the ICSD prototype structures
        """
        if str(state) in self.visited:
            return
        children = state.get_next_actions(self.builder)
        for c in children:
            yield from self.generate_decorations(c)
            self.visited.add(str(c))

        if len(children) == 0:
            self.progress_bar.update(1)
            if self.rewarder is None:
                yield(str(state))
            else:
                # generate the decorated structure and get the reward
                reward, info = self.rewarder.get_reward(state)
                info = [round(info[c], 3) for c in self.info_to_keep if c in info]
                yield(tuple([str(state), round(reward, 3)] + info))


def main():
    rewarder = CrystalStateReward(competing_phases,
                                  prototype_structures,
                                  energy_model,
                                  preprocessor,
                                  #vol_pred_site_bias=site_bias,
                                  #sub_rewards=train_config.get('sub_rewards')
                                  )

    if decor_ids is None or len(decor_ids) == 0:
        # generate all the decoration IDs
        prob_config = run_config.problem_config
        builder = CrystalBuilder(G=prob_config.get('action_graph1'),
                                G2=prob_config.get('action_graph2'),
                                actions_to_ignore=prob_config.get('actions_to_ignore'))

        gen_decors = GenerateDecorations(builder, rewarder=rewarder, visited=states_seen)
        decor_ids = gen_decors.generate_all_decorations()

    # now generate the decorations
    decorations = generate_structures(decor_ids, rewarder)

    # TODO Now write the structures

    # code to compute the reward for each decoration
    #info_to_keep = ['predicted_energy',
    #                'decomp_energy',
    #                'cond_ion_frac',
    #                'reduction',
    #                'oxidation',
    #                'stability_window',
    #                ]
    #decoration_rewards = compute_rewards(decor_ids, rewarder, info_to_keep=info_to_keep)
    #with gzip.open(args.out_file, 'w') as out:
    #    #header = ','.join(["id", "reward"] + info_to_keep) + '\n'
    #    #out.write(header.encode())
    #    for row in decoration_rewards:
    #        if len(row) != len(info_to_keep) + 2:
    #            # if there are missing rows, then add nans in their place
    #            row = list(row) + [''] * (len(info_to_keep) + 2 - len(row))
    #        out.write((','.join(str(x) for x in row) + '\n').encode())
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate decorations and compute the reward for each one')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--energy-model',
                        type=pathlib.Path,
                        required=True,
                        help='Model for predicting total energy of a battery system')
    parser.add_argument('--out-file',
                        type=pathlib.Path,
                        help='Output file')
    parser.add_argument('--decor-ids-file',
                        type=pathlib.Path,
                        action='append',
                        help='File with decoration ids for which the reward will be computed')
    #parser.add_argument('--vol-pred-site-bias', type=pathlib.Path,
    #                    help='Apply a volume prediction to the decorated structure '
    #                    'before passing it to the GNN. '
    #                    'Give the path to a file with the average volume per element')

    args = parser.parse_args()

    run_config = RunConfig(args.config)
    run_id = run_config.run_id

    # Dataframe containing competing phases from NRELMatDB
    print("Reading inputs/competing_phases.csv")
    df_competing_phases = pd.read_csv('inputs/competing_phases.csv')
    print(f"\t{len(df_competing_phases)} lines")
    print(df_competing_phases.head(2))

    df_competing_phases['energy'] = (
        df_competing_phases.energyperatom *
        df_competing_phases.sortedformula.apply(lambda x: Composition(x).num_atoms)
    )
    # convert the dataframe to a list of PDEntries used to create the convex hull
    pd_entries = df_competing_phases.apply(
        lambda row: PDEntry(Composition(row.sortedformula),
                            row.energy),
        axis=1
    )
    print(f"\t{len(pd_entries)} entries")
    competing_phases = pd.concat([pd.Series(fere_entries), pd_entries]).reset_index()[0]

    # load the icsd prototype structures
    prob_config = run_config.problem_config
    prototypes_file = "../../rlmolecule/crystal/inputs/icsd_prototypes_lt50atoms_lt100dist.json.gz"
    prototypes_file = prob_config.get('prototypes_file', prototypes_file)
    prototype_structures = read_structures_file(prototypes_file)
    # make sure the prototype structures don't have oxidation states
    from pymatgen.transformations.standard_transformations import OxidationStateRemovalTransformation
    oxidation_remover = OxidationStateRemovalTransformation()
    prototype_structures = {s_id: oxidation_remover.apply_transformation(s)
                            for s_id, s in prototype_structures.items()}

    preprocessor = AtomicNumberPreprocessor()
    print(f"Reading {args.energy_model}")
    energy_model = tf.keras.models.load_model(args.energy_model,
                                              custom_objects={**custom_objects,
                                                              **{'RBFExpansion': RBFExpansion}})

    decor_ids = set()
    if args.decor_ids_file is not None:
        for decor_ids_file in args.decor_ids_file:
            states = set(pd.read_csv(decor_ids_file, header=None)[0])
            decor_ids.update(states)
            print(f"{len(states)} states read from {decor_ids_file}")
        print(f"{len(decor_ids)} total")

    main()
