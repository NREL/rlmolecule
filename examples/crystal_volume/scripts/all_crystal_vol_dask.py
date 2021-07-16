""" Brute-force compute the fractional volume of the conducting ions for all 15M decorations
"""

import json
import logging
import gzip
import os
import pandas as pd
# Apparently there's an issue with the latest version of pandas. 
# Got this fix from here:
# https://github.com/pandas-profiling/pandas-profiling/issues/662#issuecomment-803673639
pd.set_option("display.max_columns", None)
import numpy as np
import time
from collections import defaultdict
import itertools
import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

from tqdm import tqdm
from pymatgen.core import Composition, Structure
from pymatgen.analysis import local_env

from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.sql.run_config import RunConfig
# from rlmolecule.tree_search.reward import RankedRewardFactory
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory, RankedRewardFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_structure_vol(structure: Structure, comp=None):
    """ compute the total volume and the volume each element type
    """
    # if the voronoi search fails, could try increasing the cutoff here
    for nn in [nn13]:
        try:
            voronoi_stats = nn.get_all_voronoi_polyhedra(structure)
            break
        # this function often fails for large or spaced out structures
        except ValueError as e:
            logger.warning(f"compute_structure_vol:ValueError: {e}  -  {comp}")
            return 0
        except MemoryError as e:
            logger.warning(f"compute_structure_vol:MemoryError: {e}  -  {comp}")
            return 0
        except RuntimeError as e:
            logger.warning(f"compute_structure_vol:RuntimeError: {e}  -  {comp}")
            return 0

    total_vol = 0
    element_vols = defaultdict(int)
    for atom in voronoi_stats:
        for site, site_info in atom.items():
            vol = site_info['volume']
            total_vol += vol

            element = site_info['site'].as_dict()['species'][0]['element']
            element_vols[element] += vol

    vol_sum = sum([vol for ele, vol in element_vols.items()])
    vol_sum = round(vol_sum, 2)
    total_vol = round(total_vol, 2)
    if vol_sum != total_vol:
        logger.warning(f"vol_sum != total_vol ({vol_sum} != {total_vol}) - {comp}")
    #assert vol_sum == total_vol, f"ERROR: vol_sum != total_vol ({vol_sum} != {total_vol}"

    return element_vols


def compute_vols_icsd(structures):
    """ For each of the ICSD structures, compute the volumes of the each of the elements
    This will then be used to extract the conducting ion volume for the predicted decorations

    :param structures: dictionary of structure name: pymatgen structure
    """
    failed_structures = set()
    strc_ele_vols = {}
    for name, strc in tqdm(structures.items()):
        time_start = time.process_time()
        element_vols = compute_structure_vol(strc, name)
        time_taken = time.process_time() - time_start
        if element_vols == 0:
            failed_structures.add(name)
            strc_ele_vols[name + '-time'] = time_taken
            continue

        strc_ele_vols[name] = element_vols
        # add the time to compute the structure
        strc_ele_vols[name + '-time'] = time_taken

    logger.warning(f"  {len(failed_structures)} structures failed to compute the volume")

    return strc_ele_vols


def generate_decorations_from_icsd(builder, state, visited, progress_bar):
    """ DFS to extract the volume stats for all decorations from the precomputed icsd structure volumes
    """
    if str(state) in visited:
        return
    children = state.get_next_actions(builder)
    for c in children:
        generate_decorations_from_icsd(builder, c, visited, progress_bar)
        visited.add(str(c))

    if len(children) == 0:
        progress_bar.update(1) 
        lazy_result = dask.delayed(compute_vol_stats)(state)
        lazy_results.append(lazy_result)
        return
        #yield stats


def compute_vol_stats(state):
    # This is a terminal state, so return the decorated structure.
    # The 'action_node' string has the following format at this point:
    # comp_type|prototype_structure|decoration_idx
    # we just need 'comp_type|prototype_structure' to get the icsd structure
    structure_key = '|'.join(state.action_node.split('|')[:-1])

    icsd_prototype = structures[structure_key]
    decoration_idx = int(state.action_node.split('|')[-1]) - 1
    start_time = time.process_time()
    comp = decorate_prototype_structure(
        icsd_prototype, state.composition, decoration_idx=decoration_idx)
    # if the decoration failed, skip
    if comp is None:
        return [str(state), 0,0,0,0]

    # now line up the elements of this decoration with the icsd structure volumes
    icsd_comp = icsd_prototype.composition.reduced_composition.alphabetical_formula.replace(' ','')
    icsd_elements = state.get_eles_from_comp(icsd_comp)
    decorated_elements = state.get_eles_from_comp(comp)
    ele_mapping = {e: icsd_elements[i] for i, e in enumerate(decorated_elements)}
    # now extract the volume of the conducting ions
    icsd_ele_vols = icsd_strc_vols.get(structure_key)
    # if the voronoi volume calculation failed for this structure, then just return 0s
    if icsd_ele_vols is None:
        conducting_ion_vol = 0
        total_vol = 0
    else:
        conducting_ion_vol = extract_conducting_ion_vol(ele_mapping, icsd_ele_vols)
        total_vol = sum([vol for ele, vol in icsd_ele_vols.items()])

    frac_conducting_ion_vol = conducting_ion_vol / total_vol if total_vol != 0 else 0

    time_taken = time.process_time() - start_time + icsd_strc_vols.get(structure_key + '-time', 0)
    stats = [str(round(x, 4)) for x in (conducting_ion_vol, total_vol, frac_conducting_ion_vol, time_taken)]
    #volume_stats[str(state)] = stats
    #out.write(('\t'.join([str(state)] + stats) + '\n').encode())
    return [str(state)] + stats



def extract_conducting_ion_vol(ele_mapping, icsd_ele_vols):
    conducting_ion_vol = {}
    for decor_ele, icsd_ele in ele_mapping.items():
        icsd_ele_vol = icsd_ele_vols[icsd_ele]
        if decor_ele in conducting_ions:
            conducting_ion_vol[decor_ele] = icsd_ele_vol

    # Zn can be either a conducting ion or a framework cation.
    # Make sure we're counting it correctly here
    if len(conducting_ion_vol) == 1:
        conducting_ion_vol = list(conducting_ion_vol.values())[0]
    elif len(conducting_ion_vol) == 2:
        # remove Zn
        correct_ion = list(set(conducting_ion_vol.keys()) - {'Zn'})[0]
        conducting_ion_vol = conducting_ion_vol[correct_ion]
    else:
        logger.warning(f"Expected 1 conducting ion. Found {len(conducting_ion_vol)}")
        conducting_ion_vol = 0
    return conducting_ion_vol


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

    if decoration_idx < len(valid_comp_permutations):
        return None

    # now build the decorated structure for the specific index passed in
    # since we don't need the structure, just return the composition
    return valid_comp_permutations[decoration_idx]


if __name__ == "__main__":

    # want to maximize the volume around only the conducting ions
    conducting_ions = set(['Li', 'Na', 'K', 'Mg', 'Zn'])
    anions = set(['F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'])
    framework_cations = set(
        ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])

    # Many structures fail with the default cutoff radius in Angstrom to look for near-neighbor atoms (13.0)
    # with the error: "No Voronoi neighbors found for site".
    # see: https://github.com/materialsproject/pymatgen/blob/v2022.0.8/pymatgen/analysis/local_env.py#L639.
    # Increasing the cutoff takes longer. If I bump it up to 1000, it can take over 100 Gb of Memory!
    # 2021-07-14: For now I'm just going to leave it at 13 since that's what I used for the rlmolecule run
    nn13 = local_env.VoronoiNN(cutoff=13, compute_adj_neighbors=False)

    from examples.crystal_volume import optimize_crystal_volume as ocv
    structures = ocv.structures

    icsd_strc_vols_file = "outputs/icsd_strc_vols.json"
    if os.path.isfile(icsd_strc_vols_file):
        print(f"reading {icsd_strc_vols_file}")
        with open(icsd_strc_vols_file, 'r') as f:
            icsd_strc_vols = json.load(f)
        print(f"\t{len([key for key in icsd_strc_vols if 'time' not in key])} structures read")
    else:
        print(f"computing the element volumes for {len(structures)} structures")
        icsd_strc_vols = compute_vols_icsd(structures)
        print(f"writing {icsd_strc_vols_file}")
        os.makedirs(os.path.dirname(icsd_strc_vols_file), exist_ok=True)
        # write this to a file
        with open(icsd_strc_vols_file, 'w') as out:
            out.write(json.dumps(icsd_strc_vols, indent=2, sort_keys=True))

    #root_state = CrystalState('root')
    # use a composition as the starting state for testing:
    root_state = CrystalState('K2O1', composition='K2O1')
    builder = CrystalBuilder()

    n = 16*10**6
    progress_bar = tqdm(total=n)
    # results will be stored in volume_stats
    #volume_stats = {}
    visited = set()
    lazy_results = [] 
    generate_decorations_from_icsd(builder, root_state, visited, progress_bar)
    #print(results)
    print(f"Finished setting up dask command ({len(lazy_results)} tasks). Running")

    n_processes = 36  # number of processes to run on each node
    memory = 90000  # to fit on a standard node; ask for 184,000 for a bigmem node
    
    cluster = SLURMCluster(
        project='bpms',
        walltime='60',  # 30 minutes to fit in the debug queue; 180 to fit in short
        job_mem=str(memory),
        job_cpu=36,
        interface='ib0',
        local_directory='/tmp/scratch/dask-worker-space',
        cores=36,
        processes=n_processes,
        memory='{}MB'.format(memory),
        queue='debug'  # Obviously this is limited to only a single job -- comment this out for larger runs
    )
    
    print(cluster.job_script())

    # Create the client
    dask_client = Client(cluster)

    n_nodes = 1 # set this to the number of nodes you would like to start as workers
    cluster.scale(n_processes * n_nodes)

    ## This creates a local client to run on a debug node
    #cluster = LocalCluster()
    #client = Client(cluster)
    #cluster.scale(35)
    results = dask.compute(*lazy_results)
    print("Finished computing results")
    df = pd.DataFrame(results).T
    df.columns = ['state', 'conducting_ion_vol', 'total_vol', 'fraction', 'time_taken']
    #df = df.sort_values('fraction')
    out_file = "outputs/2021-07-16-all-decoration-vol-stats-dask.tsv"
    print(f"writing {out_file}")
    df.to_csv(out_file, sep='\t')

