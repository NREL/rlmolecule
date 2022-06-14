
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import time
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor

from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.crystal import reward_utils
from rlmolecule.crystal import ehull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tf.function(experimental_relax_shapes=True)
def predict(model: 'tf.keras.Model', inputs):
    return model.predict_step(inputs)


class StructureRewardBattInterface:
    """ Compute the reward for crystal structures
    """

    def __init__(self,
                 competing_phases: List[PDEntry],
                 reward_weights: dict = None,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery interface.

        :param competing_phases: list of competing phases used to 
            construct the convex hull for the elements of the given composition
        :param reward_weights: Weights specifying how the individual rewards will be combined.
        For example: `{"decomp_energy": 0.5, "cond_ion_frac": 0.1, [...]}`
        """
        self.competing_phases = competing_phases
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
        self.reward_ranges = {"decomp_energy": (-5, 2),  # flipped so higher is better
                              "cond_ion_frac": (0, 0.8),
                              "cond_ion_vol_frac": (0, 0.8),
                              "reduction": (-5, 0),
                              "oxidation": (0, 5),  # flipped so higher is better
                              "stability_window": (0, 5),
                              }

    def compute_reward(self,
                       structure: Structure,
                       predicted_energy: float = None,
                       state: CrystalState = None,
                       ):
        """
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
        sub_rewards = {}
        info = {}
        if predicted_energy is None:
            decomp_energy_reward = self.default_decomp_energy
        else:
            info.update({'predicted_energy': predicted_energy})
            start = time.process_time()
            decomp_energy, stability_window = ehull.convex_hull_stability(
                    structure.composition,
                    predicted_energy,
                    self.competing_phases,
            )
            info['decomp_time'] = time.process_time() - start
            if decomp_energy is None:
                # add 1 to the default energy to distinguish between
                # failed calculation here, and failing to decorate the structure 
                decomp_energy_reward = self.default_decomp_energy + 1
            else:
                # Since more negative is more stable, and higher is better for the reward values,
                # flip the hull energy
                decomp_energy_reward = -1 * decomp_energy

                info.update({'decomp_energy': decomp_energy})
                # also compute the stability window rewards
                if decomp_energy < 0 and stability_window is not None:
                    # Apply some hard cutoffs on the oxidation and reduction rewards
                    oxidation, reduction = stability_window
                    if oxidation < -3:
                        # flip the oxidation reward so that higher is better
                        sub_rewards['oxidation'] = -1 * oxidation
                    if reduction > -3:
                        sub_rewards['reduction'] = reduction
                    stability_window_size = reduction - oxidation
                    if stability_window_size > 2:
                        sub_rewards['stability_window'] = stability_window_size

                    info.update({'oxidation': oxidation,
                                 'reduction': reduction,
                                 'stability_window': stability_window_size})

        sub_rewards['decomp_energy'] = decomp_energy_reward

        try:
            cond_ion_frac = reward_utils.get_conducting_ion_fraction(structure.composition)
            sub_rewards['cond_ion_frac'] = cond_ion_frac

            start = time.process_time()
            cond_ion_vol_frac = reward_utils.compute_cond_ion_vol(structure, state=state)
            # if the voronoi volume calculation failed, give a default of
            # the conducting ion's fraction * .5
            if cond_ion_vol_frac is None:
                cond_ion_vol_frac = cond_ion_frac / 2
            sub_rewards['cond_ion_vol_frac'] = cond_ion_vol_frac
            info['cond_ion_vol_time'] = time.process_time() - start

            info.update({'cond_ion_frac': cond_ion_frac,
                        'cond_ion_vol_frac': cond_ion_vol_frac})
        # some structures don't have a conducting ion
        except ValueError as e:
            print(f"ValueError: {e}. State: {state}")

        combined_reward = self.combine_rewards(sub_rewards)
        #print(str(state), combined_reward, info)

        return combined_reward, info

#    def precompute_convex_hulls(self, compositions):
#        self.phase_diagrams = {}
#        for comp in tqdm(compositions):
#            comp = Composition(comp)
#            elements = set(comp.elements)
#            curr_entries = [e for e in self.competing_phases
#                            if len(set(e.composition.elements) - elements) == 0]
#
#            phase_diagram = PhaseDiagram(curr_entries, elements=elements)
#            self.phase_diagrams[comp] = phase_diagram

    def combine_rewards(self, sub_rewards) -> float:
        """ Take the weighted average of the normalized sub-rewards
        For example, decomposition energy: 1.2, conducting ion frac: 0.1.
        """
        scaled_rewards = {}
        for key, rew in sub_rewards.items():
            if rew is None:
                continue
            r_min, r_max = self.reward_ranges[key]
            # first apply the bounds to make sure the values are in the right range
            rew = max(r_min, rew)
            rew = min(r_max, rew)
            # scale between 0 and 1 using the given range of values
            scaled_reward = (rew - r_min) / (r_max - r_min)
            scaled_rewards[key] = scaled_reward

        # Now apply the weights to each sub-reward
        #weighted_rewards = {k: v * self.reward_weights[k] for k, v in scaled_rewards.items()}
        combined_reward = sum([v * self.reward_weights[k] for k, v in scaled_rewards.items()])

        #print(sub_rewards)

        return combined_reward


class CrystalStateReward(StructureRewardBattInterface):
    """ Compute the reward for terminal states in the action space
    """

    def __init__(self,
                 competing_phases: List[PDEntry],
                 prototypes: Dict[str, Structure],
                 energy_model: 'tf.keras.Model',
                 preprocessor: PymatgenPreprocessor,
                 #dist_model: 'tf.keras.Model',
                 vol_pred_site_bias: Optional['pd.Series'] = None,
                 default_reward: float = 0,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery interface.
        Starting from a terminal state, this class will build the structure, predict the total energy, 
        calculate the individual rewards, and combine them together.

        :param competing_phases: list of competing phases used to 
            construct the convex hull for the elements of the given composition
        :param prototypes: Dictionary mapping from prototype ID to structure. Used for decorating new structures
        :param energy_model: A tensorflow model to estimate the total energy of a structure
        :param preprocessor: Used to process the structure into inputs for the energy model

        :param vol_pred_site_bias: Optional argument of average volume per element (e.g., in ICSD).
            Used to predict the volume of the decorated structure 
            before passing it to the GNN. Uses the linear model + pymatgen's DLS predictor

        :param default_reward: Reward given to structures that failed to decorate
        """
        self.prototypes = prototypes
        self.energy_model = energy_model
        self.preprocessor = preprocessor
        #self.dist_model = dist_model
        self.vol_pred_site_bias = vol_pred_site_bias
        self.dls_vol_predictor = DLSVolumePredictor()
        self.default_reward = default_reward
        
        super(CrystalStateReward, self).__init__(competing_phases, **kwargs)

    def get_reward(self, state: CrystalState) -> Tuple[float, dict]:
        """ Get the reward for the given crystal state. 
        This function first generates the structure by decorating the state's selected prototype.
        Then it predicts the total energy using the energy model 
        """
        if not state.terminal:
            return self.default_reward, {'terminal': False,
                                         'state_repr': repr(state)}
        info = {}
        start = time.process_time()
        structure = self.generate_structure(state)
        gen_strc_time = time.process_time() - start
        if structure is None:
            return self.default_reward, {'terminal': True,
                                         'state_repr': repr(state),
                                         'gen_strc_time': gen_strc_time}

        info.update({'terminal': True,
                     'num_sites': len(structure.sites),
                     'volume': structure.volume,
                     'state_repr': repr(state),
                     'gen_strc_time': gen_strc_time,
                     })

        # Predict the total energy of this decorated structure
        start = time.process_time()
        predicted_energy = self.predict_energy(structure, state)
        rew_start = time.process_time()
        reward, reward_info = self.compute_reward(structure,
                                                  predicted_energy,
                                                  state)
        info['pred_time'] = rew_start - start
        info['reward_time'] = time.process_time() - rew_start
        info.update(reward_info)
        return reward, info

    def generate_structure(self, state: CrystalState):
        # skip this structure if it is too large for the model.
        # I removed these structures from the action space (see builder.py "action_graph2_file"),
        # so shouldn't be a problem anymore
        structure_key = '|'.join(state.action_node.split('|')[:-1])
        icsd_prototype = self.prototypes[structure_key]

        # generate the decoration for this state
        try:
            structure = reward_utils.generate_decoration(state, icsd_prototype)
        except AssertionError as e:
            print(f"AssertionError: {e}")
            return

        if self.vol_pred_site_bias is not None:
            # predict the volume of the decorated structure before
            # passing it to the GNN. Use a linear model + pymatgen's DLS predictor
            structure = self.scale_by_pred_vol(structure)
        return structure

    def scale_by_pred_vol(self, structure: Structure) -> Structure:
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

    def get_model_inputs(self, structure) -> dict:
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
    def predict_energy(self, structure: Structure, state=None) -> float:
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
        predicted_energy = predicted_energy[0][0].astype(float)

        return predicted_energy

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
