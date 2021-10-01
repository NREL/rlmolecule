""" Brute-force compute the fractional volume of the conducting ions for all 15M decorations
"""

import gzip
# from dask.distributed import Client
# from dask_jobqueue import SLURMCluster
import logging
import os

import pandas as pd

# Apparently there's an issue with the latest version of pandas.
# Got this fix from here:
# https://github.com/pandas-profiling/pandas-profiling/issues/662#issuecomment-803673639
pd.set_option("display.max_columns", None)
import numpy as np
from tqdm import tqdm

import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')
device = devices[0]
# device = torch.device(cuda:0 if torch.cuda.is_available() else cpu)
tf.config.experimental.set_memory_growth(device, True)

import nfp
from examples.crystal_energy.nfp_extensions import RBFExpansion, CifPreprocessor

from rlmolecule.crystal import utils
from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_state import CrystalState

# from rlmolecule.tree_search.reward import RankedRewardFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

icsd_prototypes_file = "../../rlmolecule/crystal/inputs/icsd_prototypes.json.gz"
icsd_structures = utils.read_structures_file(icsd_prototypes_file)
structures_skipped = {key: len(s.sites) for key, s in icsd_structures.items() if len(s.sites) >= 150}
print(f"\t{len(structures_skipped)}/{len(icsd_structures)} structures with >= 150 sites skipped ")
# print(structures_skipped)
# sys.exit()
icsd_structures = {key: s for key, s in icsd_structures.items() if len(s.sites) < 150}

# Initialize the preprocessor class
preprocessor = CifPreprocessor(num_neighbors=12)
preprocessor.from_json('inputs/preprocessor.json')

# Load the model trained on DFT relaxations
model_file = "inputs/models/icsd_battery_unrelaxed/hypo_randsub0_05_icsd_randsub0_05_seed1/best_model.hdf5"
print(f"loading {model_file}")
model = tf.keras.models.load_model(model_file,
                                   custom_objects={**nfp.custom_objects, **{'RBFExpansion': RBFExpansion}})


def main(out_pref):
    root_state = CrystalState('root')
    # use a composition as the starting state for testing:
    # root_state = CrystalState('K2O1', composition='K2O1')
    builder = CrystalBuilder()

    n = 16 * 10 ** 6
    progress_bar = tqdm(total=n)
    visited = set()

    decorated_structures = list(generate_decorations(builder, root_state, visited, progress_bar=progress_bar))
    decorated_structures = [(state, s) for state, s in decorated_structures if s is not None]
    states, structures = [s[0] for s in decorated_structures], [s[1] for s in decorated_structures]

    os.makedirs(os.path.dirname(out_pref), exist_ok=True)
    out_file = f"{out_pref}all-decorations.txt.gz"
    print(f"writing to {out_file}")
    with gzip.open(out_file, 'w') as out:
        out.write(('\n'.join(states) + '\n').encode())

    # also write the structures to a file
    structures_dict = {state: s.as_dict() for state, s in decorated_structures if s is not None}
    out_file = f"{out_pref}all-decorations.json.gz"
    utils.write_structures_file(out_file, structures_dict, round_float=4)

    # dataset = tf.data.Dataset.from_tensor_slices()
    print(f"Predicting energies of {len(structures)} structures")
    print(structures[:2])
    predicted_energies = predict_structure_energy(structures)
    # predicted_energies = predict_structure_energy(builder, root_state)
    print("Finished")
    print(len(states))
    print(predicted_energies.shape)

    out_file = f"{out_pref}all-decoration-energy-scores.npy"
    print(f"writing to {out_file}")
    np.save(out_file, predicted_energies)
    # out = gzip.open(out_file, 'wb')
    ##print(results)
    # print("Finished")


def predict_structure_energy(structures, max_sites=256, max_bonds=2048):
    """ Predict the total energy of the structures using a GNN model 
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(strc, train=False)
                 # for strc in tqdm(structures) if strc is not None),
                 for strc in tqdm(structures) if strc is not None and len(strc.sites) < 150),
        output_types=preprocessor.output_types,
        output_shapes=preprocessor.output_shapes) \
        .padded_batch(batch_size=128,
                      padded_shapes=preprocessor.padded_shapes(max_sites=max_sites, max_bonds=max_bonds),
                      padding_values=preprocessor.padding_values)

    predicted_energy = model.predict(dataset)

    return predicted_energy


def generate_decorations(builder, state, visited, progress_bar=None):
    """ DFS to generate all of the possible decorations
    """
    if str(state) in visited:
        return
    children = state.get_next_actions(builder)
    for c in children:
        yield from generate_decorations(builder, c, visited, progress_bar)
        visited.add(str(c))

    if len(children) == 0:
        if progress_bar:
            progress_bar.update(1)
            # This is a terminal state, so return the decorated structure.
        # The 'action_node' string has the following format at this point:
        # comp_type|prototype_structure|decoration_idx
        # we just need 'comp_type|prototype_structure' to get the icsd structure
        structure_key = '|'.join(state.action_node.split('|')[:-1])
        ## for now, just return all of the states
        # if structure_key not in icsd_structures:
        #    return 
        # yield str(state)

        if structure_key not in icsd_structures:
            # yield str(state), None
            return
        else:
            icsd_prototype = icsd_structures[structure_key]
            decoration_idx = int(state.action_node.split('|')[-1]) - 1
            try:
                decorated_structure, comp = CrystalState.decorate_prototype_structure(
                    icsd_prototype, state.composition, decoration_idx=decoration_idx)
                ## return the strcture in a cubic lattice
                # decorated_structure.lattice = decorated_structure.lattice.cubic(1.0)
                yield str(state), decorated_structure
            except AssertionError as e:
                logger.warning(f"AssertionError: {e}")
                yield str(state), None


if __name__ == "__main__":
    out_pref = "/projects/rlmolecule/jlaw/crystals/2021-09-22/"
    main(out_pref)
# n_processes = 36  # number of processes to run on each node
# memory = 90000  # to fit on a standard node; ask for 184,000 for a bigmem node
#
# cluster = SLURMCluster(
#    project='bpms',
#    walltime='30',  # 30 minutes to fit in the debug queue; 180 to fit in short
#    job_mem=str(memory),
#    job_cpu=36,
#    interface='ib0',
#    local_directory='/tmp/scratch/dask-worker-space',
#    cores=36,
#    processes=n_processes,
#    memory='{}MB'.format(memory),
#    queue='debug'  # Obviously this is limited to only a single job -- comment this out for larger runs
# )
#
# print(cluster.job_script())
#
## Create the client
# dask_client = Client(cluster)
#
# n_nodes = 1 # set this to the number of nodes you would like to start as workers
# cluster.scale(n_processes * n_nodes)
