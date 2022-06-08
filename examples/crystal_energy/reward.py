
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
from scripts import ehull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO set the elements in the builder
default_conducting_ions = set(['Li', 'Na', 'K', 'Mg', 'Zn'])
anions = set(['F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'])
framework_cations = set(
    ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg',
     'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])

# Many structures fail with the default cutoff radius in Angstrom to look for near-neighbor atoms (13.0)
# with the error: "No Voronoi neighbors found for site".
# see: https://github.com/materialsproject/pymatgen/blob/v2022.0.8/pymatgen/analysis/local_env.py#L639.
# Increasing the cutoff takes longer. If I bump it up to 1000, it can take over 100 Gb of Memory!
nn13 = local_env.VoronoiNN(cutoff=13, compute_adj_neighbors=False)


@tf.function(experimental_relax_shapes=True)
def predict(model: 'tf.keras.Model', inputs):
    return model.predict_step(inputs)


#class CrystalEnergyState(CrystalState):
#    @property
#    def reward(self) -> float:
#        """The reward function for the CrystalState graph.
#        Only states at the end of the graph have a reward,
#        since a decorated structure is required to compute the 
#        total energy and decomposition energy.
#
#        Returns:
#            float: 
#        """
#        return self.rewarder.get_reward(self)


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

    conducting_ion = list(cond_ions)[0]
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
    frac = comp.get_atomic_fraction(Element(conducting_ion))
    return frac


def calc_decomp_energy(structure: Structure, predicted_energy, df_competing_phases, state=None):
    comp = structure.composition.reduced_composition \
                                .alphabetical_formula \
                                .replace(' ', '')
    decomp_energy, stability_borders = \
        ehull.convex_hull_stability(comp,
                                    predicted_energy,
                                    df_competing_phases)

    return decomp_energy, stability_borders


def get_electrochem_stability_window(comp: Composition,
                                     stability_borders: dict) -> Tuple[float, float]:
    """ Get the stability window of the conducting ion for this composition.
    Largest stability window: -5 to 0
    
    :returns: tuple of min to max voltage
    """
    conducting_ion = get_conducting_ion(comp)
    # want a low oxidation potential, capped at -5
    oxidation_potential = max(-5, min(stability_borders[conducting_ion]))
    # want a high reduction potential, capped at 0
    reduction_potential = min(0, max(stability_borders[conducting_ion]))
    electrochem_stability_window = (oxidation_potential, reduction_potential)
    return electrochem_stability_window


class CrystalReward:
    """ Compute the reward 
    """

    def __init__(self,
                 df_competing_phases: 'pd.DataFrame',
                 reward_weights: dict = None,
                 default_reward: float = 0,
        ):
        """ A class to estimate the suitability of a crystal structure as a solid state battery interface.
        If computing rewards for 

        :param prototypes: Dictionary mapping from ID to structure. Used for decorating new structures
        :param energy_model: A tensorflow model to estimate the total energy of a structure
        """
        self.prototypes = prototypes
        self.energy_model = energy_model
        self.preprocessor = preprocessor
        #self.dist_model = dist_model
        self.df_competing_phases = df_competing_phases
        self.vol_pred_site_bias = vol_pred_site_bias
        self.dls_vol_predictor = DLSVolumePredictor()
        self.default_reward = default_reward
        self.default_decomp_energy = -5

        # set the weights of the individual rewards
        self.reward_weights = reward_weights
        if self.reward_weights is None:
            self.reward_weights = {"decomp_energy": .5,
                                   "cond_ion_frac": .1,
                                   "cond_ion_vol_frac": .1,
                                   "reduction": .1,
                                   "oxidation": .1,
                                   "stability_window": .1,
                                   }
        self.reward_ranges = {"decomp_energy": (-5, 2),
                              "cond_ion_frac": (0, 0.8),
                              "cond_ion_vol_frac": (0, 0.8),
                              "reduction": (-5, 0),
                              "oxidation": (0, -5),
                              "stability_window": (0, 5),
                              }
        # store the original reward values
        self.sub_rewards = {k: None for k in self.reward_weights.keys()}

    def generate_structure(self,
                           prototypes: Optional[dict],
                           energy_model: Optional['tf.keras.Model'],
                           preprocessor: Optional[PymatgenPreprocessor],
                           #dist_model: 'tf.keras.Model',
                           vol_pred_site_bias: Optional['pd.Series'] = None,
                           ):

    def get_reward(self, structure: Structure = None, state: CrystalState = None) -> (float, {}):
        """Get the reward for the CrystalState.
        The following sub-rewards are combined:
        1. Decomposition energy: predicts the total energy using a GNN model
            and calculates the corresponding decomposition energy based on the competing phases.
        2. Conducting ion fraction
        3. Conducting ion volume
        4. Reduction potential
        5. Oxidation potential
        6. Electrochemical stability window:
            difference between 4. and 5.
        
        Returns:
            float: reward
            dict: info
        """
        assert state is not None or structure is not None, "Must pass either state or structure"
        info = {}
        if structure is None:
            if not state.terminal:
                return self.default_reward, {'terminal': False,
                                            'state_repr': repr(state)}
            # skip this structure if it is too large for the model.
            # I removed these structures from the action space (see builder.py "action_graph2_file"),
            # so shouldn't be a problem anymore
            structure_key = '|'.join(state.action_node.split('|')[:-1])
            icsd_prototype = self.prototypes[structure_key]

            # generate the decoration for this state
            try:
                structure = generate_decoration(state, icsd_prototype)
            except AssertionError as e:
                print(f"AssertionError: {e}")
                return self.default_reward, {'terminal': True,
                                             'state_repr': repr(state)}

            if self.vol_pred_site_bias is not None:
                # predict the volume of the decorated structure before
                # passing it to the GNN. Use a linear model + pymatgen's DLS predictor
                structure = self.scale_by_pred_vol(structure)

            info.update({'terminal': True,
                         'num_sites': len(structure.sites),
                         'volume': structure.volume,
                         'state_repr': repr(state)})

        # Predict the total energy of this decorated structure
        predicted_energy = self.predict_energy(structure, state) 
        if predicted_energy is None:
            decomp_energy_reward = self.default_decomp_energy - 2
        else:
            decomp_energy, stability_borders = calc_decomp_energy(structure,
                                                                  predicted_energy,
                                                                  self.df_competing_phases,
                                                                  state)
            if decomp_energy is None:
                # subtract 1 to the default energy to distinguish between
                # failed calculation here, and failing to decorate the structure 
                decomp_energy_reward = self.default_decomp_energy - 1
                info.update({'predicted_energy': predicted_energy.astype(float)})
            else:
                # Since more negative is more stable, and higher is better for the reward values,
                # flip the hull energy
                decomp_energy_reward = -1 * decomp_energy.astype(float)

                info.update({'predicted_energy': predicted_energy.astype(float),
                            'decomp_energy': decomp_energy})
                # also compute the stability window rewards
                if decomp_energy < 0 and stability_borders is not None:
                    electrochem_stability_window = get_electrochem_stability_window(
                        structure.composition,
                        stability_borders)
                    # Apply some hard cutoffs on the oxidation and reduction rewards
                    oxidation, reduction = electrochem_stability_window
                    if oxidation < -3:
                        self.sub_rewards['oxidation'] = oxidation
                    if reduction > -3:
                        self.sub_rewards['reduction'] = reduction
                    stability_window_size = reduction - oxidation
                    if stability_window_size > 2:
                        self.sub_rewards['stability_window'] = stability_window_size

                    info.update({'oxidation': oxidation,
                                 'reduction': reduction,
                                 'stability_window': stability_window_size})

        self.sub_rewards['decomp_energy'] = decomp_energy_reward

        cond_ion_frac = get_conducting_ion_fraction(structure.composition)
        self.sub_rewards['cond_ion_frac'] = cond_ion_frac

        cond_ion_vol_frac = compute_cond_ion_vol(structure, state=state)
        self.sub_rewards['cond_ion_vol_frac'] = cond_ion_vol_frac

        combined_reward = self.combine_rewards()

        info.update({'cond_ion_frac': cond_ion_frac,
                     'cond_ion_vol_frac': cond_ion_vol_frac})

        #print(combined_reward, info)

        return combined_reward, info

    def combine_rewards(self) -> float:
        """ Take the weighted average of the normalized sub-rewards
        For example, decomposition energy: 1.2, conducting ion frac: 0.1.
        First, normalize each value between their ranges:
          - decomp_energy: 
        """
        scaled_rewards = {}
        for key, rew in self.sub_rewards.items():
            if rew is None:
                continue
            r_min, r_max = self.reward_ranges[key]
            # first apply the bounds to make sure the values are in the right range
            rew = max(r_min, rew)
            rew = min(r_max, rew)
            # scale between 0 and 1 using the given range of values
            r_max, r_min = (r_min, r_max) if r_min > r_max else (r_max, r_min)
            scaled_reward = (rew - r_min) / (r_max - r_min)
            scaled_rewards[key] = scaled_reward

        # Now apply the weights to each sub-reward
        #weighted_rewards = {k: v * self.reward_weights[k] for k, v in scaled_rewards.items()}
        combined_reward = sum([v * self.reward_weights[k] for k, v in scaled_rewards.items()])

        #print(self.sub_rewards)

        return combined_reward

    def scale_by_pred_vol(self, structure):
        # first predict the volume using the average volume per element (from ICSD)
        site_counts = pd.Series(Counter(
            str(site.specie) for site in structure.sites)).fillna(0)
        curr_site_bias = self.vol_pred_site_bias[
            self.vol_pred_site_bias.index.isin(site_counts.index)]
        linear_pred = site_counts @ curr_site_bias
        structure.scale_lattice(linear_pred)

        # then apply Pymatgen's DLS predictor
        pred_volume = self.dls_vol_predictor.predict(structure)
        structure.scale_lattice(pred_volume)
        return structure

    def get_model_inputs(self, structure) -> {}:
        inputs = self.preprocessor(structure, train=False)
        # scale structures to a minimum of 1A interatomic distance
        min_distance = inputs["distance"].min()
        if np.isclose(min_distance, 0):
            # if some atoms are on top of each other, then this normalization fails
            # TODO remove those problematic prototype structures
            return None
            #raise RuntimeError(f"Error with {structure}")

        # only normalize if the volume is not being predicted
        if self.vol_pred_site_bias is None:
            inputs["distance"] /= inputs["distance"].min()
        return inputs

    # @collect_metrics
    def predict_energy(self, structure: Structure, state=None):
        """ Predict the total energy of the structure using a GNN model (trained on unrelaxed structures)
        """
        model_inputs = self.get_model_inputs(structure)
        if model_inputs is None:
            return None, None
        # predicted_energy = predict(self.energy_model, model_inputs)
        dataset = tf.data.Dataset.from_generator(
            lambda: (s for s in [model_inputs]),
            #lambda: (self.preprocessor.construct_feature_matrices(s, train=False) for s in [structure]),
            output_signature=(self.preprocessor.output_signature),
            )
        dataset = dataset \
            .padded_batch(
                batch_size=1,
                padding_values=(self.preprocessor.padding_values),
            )
        predicted_energy = self.energy_model.predict(dataset)
        #print(f"{predicted_energy = }, {state = }")
        predicted_energy = predicted_energy[0][0]

        return predicted_energy

