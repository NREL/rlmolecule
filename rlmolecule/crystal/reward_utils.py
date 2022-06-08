import logging
from collections import Counter
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from pymatgen.analysis import local_env
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from pymatgen.core import Composition, Element, Structure
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor

from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.crystal.ehull import convex_hull_stability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


default_conducting_ions = set(['Li', 'Na', 'K', 'Mg', 'Zn'])
anions = set(['F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'])
framework_cations = set(
    ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg',
     'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])

# default weights for combining the rewards
reward_weights = {"decomp_energy": .5,
                  "cond_ion_frac": .1,
                  "cond_ion_vol_frac": .1,
                  "reduction": .1,
                  "oxidation": .1,
                  "stability_window": .1,
                  }

# Many structures fail with the default cutoff radius in Angstrom to look for near-neighbor atoms (13.0)
# with the error: "No Voronoi neighbors found for site".
# see: https://github.com/materialsproject/pymatgen/blob/v2022.0.8/pymatgen/analysis/local_env.py#L639.
# Increasing the cutoff takes longer. If I bump it up to 1000, it can take over 100 Gb of Memory!
nn13 = local_env.VoronoiNN(cutoff=13, compute_adj_neighbors=False)


def generate_decoration(state: CrystalState, icsd_prototype) -> Structure:
    # Create the decoration of this composition onto this prototype structure
    # the 'action_node' string has the following format at this point:
    # comp_type|prototype_structure|decoration_idx
    # we just need 'comp_type|prototype_structure' to get the icsd structure
    decoration_idx = int(state.action_node.split('|')[-1]) - 1
    decorated_structure, stoich = CrystalState.decorate_prototype_structure(
        icsd_prototype, state.composition, decoration_idx=decoration_idx)
    return decorated_structure


def get_conducting_ion(comp: Composition, conducting_ions=None):
    """ Find which element is the conducting ion
    """
    if conducting_ions is None:
        conducting_ions = default_conducting_ions

    cond_ions = set(str(e) for e in comp.elements
                    if str(e) in conducting_ions)

    # Zn can be either a conducting ion or a framework cation.
    if len(cond_ions) == 2:
        # remove Zn
        cond_ions -= {'Zn'}
    if len(cond_ions) != 1:
        raise ValueError(f"Expected 1 conducting ion. "
                         f"Found {len(cond_ions)} ({cond_ions}) for {comp}")

    conducting_ion = Element(list(cond_ions)[0])
    return conducting_ion
    

def compute_cond_ion_vol(structure: Structure, state=None):
    """ compute the total volume and the volume of just the conducting ions
    """
    conducting_ion = get_conducting_ion(structure.composition)
    # if the voronoi search fails, could try increasing the cutoff here
    for nn in [nn13]:
        try:
            voronoi_stats = nn.get_all_voronoi_polyhedra(structure)
            break
        # this function often fails for large or spaced out structures
        except ValueError as e:
            if state:
                logger.warning(f"compute_structure_vol:ValueError: {e}  -  {state}")
            return None
        except MemoryError as e:
            if state:
                logger.warning(f"compute_structure_vol:MemoryError: {e}  -  {state}")
            return None
        except RuntimeError as e:
            if state:
                logger.warning(f"compute_structure_vol:RuntimeError: {e}  -  {state}")
            return None

    total_vol = 0
    conducting_ion_vol = 0
    for atom in voronoi_stats:
        for site, site_info in atom.items():
            vol = site_info['volume']
            total_vol += vol

            element = site_info['site'].as_dict()['species'][0]['element']
            if element == conducting_ion:
                conducting_ion_vol += vol

    total_vol = np.round(total_vol, 4)
    vol = np.round(structure.volume, 4)
    if total_vol != vol:
        print(f"WARNING: voronoi volume total_vol = {total_vol} != vol = {vol}")
    # convert the volume to a fraction of total volume
    conducting_ion_vol_frac = conducting_ion_vol / float(total_vol)

    return conducting_ion_vol_frac


def get_conducting_ion_fraction(comp: Composition) -> float:
    """ Get the fraction of atoms that are conducting ions
    """
    conducting_ion = get_conducting_ion(comp)
    frac = comp.get_atomic_fraction(conducting_ion)
    return frac

